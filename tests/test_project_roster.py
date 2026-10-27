"""project-assets-roster：归属清单迁移与语义测试（openspec 同名变更）。

覆盖：
- 迁移：影子列 tasks.project_id → project_assets 回填（老库建表+回填、
  entity 缺失回退 tasks.file_md5、幂等可重跑、新库直建）
- 语义（服务层改造后）：多归属 / 重复加入 / 删项目不动他项目 / 归属锚内容哈希

与 test_studio_api.py 同构：临时 SQLite + 打桩，直调路由/迁移函数。
"""

import sys

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "src")

from src.db import models as db_models  # noqa: E402
from src.db.models import Base, Client, FileAnalysis, FileEntity, Project, Task  # noqa: E402


def _mk_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    return engine


def _run_migration(engine):
    """直调 S3.1+roster 合并迁移（与 init_db 内部同一入口）。"""
    conn = engine.connect()
    try:
        db_models._migrate_studio_tables(conn, inspect(conn))
    finally:
        conn.close()


def _roster_count(engine, project_id=None):
    conn = engine.connect()
    try:
        if project_id is None:
            return conn.execute(text("SELECT COUNT(*) FROM project_assets")).scalar()
        return conn.execute(
            text("SELECT COUNT(*) FROM project_assets WHERE project_id = :p"),
            {"p": project_id},
        ).scalar()
    finally:
        conn.close()


# ── 迁移（任务 1.2）────────────────────────────────────────────────

def test_migration_backfills_roster_from_shadow_column(tmp_path):
    """老库影子列有归属数据 → 迁移兜底建表 + 回填清单行。

    覆盖三条回填路径：entity 关联取 entity.md5、无 entity 回退 tasks.file_md5、
    双空行跳过。"""
    engine = _mk_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE project_assets"))  # 模拟老库（表尚不存在）

    Session = sessionmaker(bind=engine)
    db = Session()
    c = Client(name="客户A")
    db.add(c)
    db.commit()
    p1 = Project(client_id=c.id, name="婚礼")
    p2 = Project(client_id=c.id, name="写真")
    db.add_all([p1, p2])
    db.commit()
    e1 = FileEntity(file_md5="md5-1")
    e2 = FileEntity(file_md5="md5-2")
    db.add_all([e1, e2])
    db.commit()
    db.add_all([
        Task(entity_id=e1.id, file_md5="md5-1", project_id=p1.id),
        Task(entity_id=e2.id, file_md5="md5-2", project_id=p1.id),
        Task(file_md5="md5-legacy", project_id=p2.id),   # 无 entity → 回退影子 md5
        Task(project_id=p1.id),                          # 双空 → 跳过
    ])
    db.commit()
    p1_id, p2_id = p1.id, p2.id
    db.close()

    _run_migration(engine)

    assert _roster_count(engine, p1_id) == 2
    assert _roster_count(engine, p2_id) == 1
    conn = engine.connect()
    try:
        md5s = {r[0] for r in conn.execute(
            text("SELECT file_md5 FROM project_assets WHERE project_id = :p"),
            {"p": p2_id}).fetchall()}
    finally:
        conn.close()
    assert md5s == {"md5-legacy"}


def test_migration_idempotent_rerun(tmp_path):
    """迁移重复执行不产生重复清单行（INSERT OR IGNORE + 唯一约束）。"""
    engine = _mk_engine(tmp_path)
    Session = sessionmaker(bind=engine)
    db = Session()
    c = Client(name="客户B")
    db.add(c)
    db.commit()
    p = Project(client_id=c.id, name="婚礼")
    db.add(p)
    db.commit()
    db.add(Task(file_md5="md5-x", project_id=p.id))
    db.commit()
    p_id = p.id
    db.close()

    _run_migration(engine)
    _run_migration(engine)
    _run_migration(engine)

    assert _roster_count(engine, p_id) == 1


def test_new_database_gets_roster_table(tmp_path):
    """新库 create_all 直带全 project_assets（迁移守卫不重复建）。"""
    db_file = tmp_path / "fresh.db"
    db_models.init_db(str(db_file))
    engine = create_engine(f"sqlite:///{db_file}")
    conn = engine.connect()
    try:
        names = {r[0] for r in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}
    finally:
        conn.close()
    assert "project_assets" in names


# ── 语义（服务层，任务 4.1）────────────────────────────────────────

@pytest.fixture()
def studio_env(monkeypatch, tmp_path):
    from src.api.routes import studio
    engine = _mk_engine(tmp_path)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(studio, "get_db_engine", lambda: engine)
    monkeypatch.setattr(studio, "get_session", lambda eng: TestSession())
    db = TestSession()
    yield db, engine
    db.close()


def _mk_prj(db, name) -> int:
    c = Client(name=f"客户-{name}")
    db.add(c)
    db.commit()
    p = Project(client_id=c.id, name=name)
    db.add(p)
    db.commit()
    return p.id


def test_multi_membership_and_scoped_unassign(studio_env):
    """多归属：同一内容 ∈ 多项目；项目级移出只动该项目清单。"""
    from src.api.routes import studio
    db, engine = studio_env
    pa = _mk_prj(db, "婚礼")
    pb = _mk_prj(db, "精选集")
    db.close()

    assert studio.assign_tasks(pa, {"file_md5s": ["md5-1"]})["assigned"] == 1
    assert studio.assign_tasks(pb, {"file_md5s": ["md5-1"]})["assigned"] == 1
    r = studio.assign_tasks(pa, {"file_md5s": ["md5-1"]})
    assert r["assigned"] == 0 and r["duplicates"] == 1  # 重复加入幂等+标注

    studio.unassign_from_project(pa, {"file_md5s": ["md5-1"]})
    assert _roster_count(engine, pa) == 0
    assert _roster_count(engine, pb) == 1  # 其他项目成员资格不动


def test_delete_project_keeps_other_project_membership(studio_env):
    """删除项目只摘本子树清单行；他项目成员资格与内容本体不动。"""
    from src.api.routes import studio
    db, engine = studio_env
    pa = _mk_prj(db, "A")
    pb = _mk_prj(db, "B")
    db.close()

    studio.assign_tasks(pa, {"file_md5s": ["md5-9"]})
    studio.assign_tasks(pb, {"file_md5s": ["md5-9"]})

    assert studio.delete_project(pa)["status"] == "success"
    assert _roster_count(engine, pa) == 0
    assert _roster_count(engine, pb) == 1


def test_task_delete_keeps_roster_tombstone(monkeypatch, tmp_path):
    """删除工单（新语义）不动清单行——同内容重导自动恢复归属（tombstone）。"""
    import asyncio

    from src.api import main as api_main

    engine = _mk_engine(tmp_path)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(api_main, "engine", engine)
    monkeypatch.setattr(api_main, "get_session", lambda eng: TestSession())

    db = TestSession()
    c = Client(name="客户T")
    db.add(c)
    db.commit()
    p = Project(client_id=c.id, name="婚礼")
    db.add(p)
    db.commit()
    e = FileEntity(file_md5="md5-t")
    db.add(e)
    db.commit()
    t = Task(entity_id=e.id, file_md5="md5-t", project_id=p.id)  # 旧数据直带影子列
    db.add(t)
    db.commit()
    p_id, t_id = p.id, t.id
    db.close()

    _run_migration(engine)  # 影子列 → 清单行回填

    result = asyncio.run(api_main.delete_videos_with_options({"task_ids": [t_id]}))
    assert result["success"] is True
    db = TestSession()
    assert db.get(Task, t_id) is None  # 工单已删
    db.close()
    assert _roster_count(engine, p_id) == 1  # 清单行保留（tombstone）


def test_duplicate_task_inherits_completed_result(monkeypatch, tmp_path):
    """重加已有完整分析的内容（design D4）：新工单直接完成态，不重跑。"""
    import src.api.main as api_main

    engine = _mk_engine(tmp_path)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(api_main, "engine", engine)
    monkeypatch.setattr(api_main, "get_session", lambda eng: TestSession())

    db = TestSession()
    c = Client(name="客户D")
    db.add(c)
    db.commit()
    p = Project(client_id=c.id, name="婚礼")
    db.add(p)
    db.commit()
    e = FileEntity(file_md5="md5-d")
    db.add(e)
    db.commit()
    db.add(FileAnalysis(entity_id=e.id, status="READY", progress=100))
    db.commit()
    prior = Task(entity_id=e.id, file_md5="md5-d", status="COMPLETE")
    db.add(prior)
    db.commit()
    dup_file = tmp_path / "dup.mp4"
    dup_file.write_bytes(b"x")
    new_task = Task(entity_id=e.id, file_md5="md5-d", status="QUEUED",
                    file_path=str(dup_file))
    db.add(new_task)
    db.commit()
    e_id, prior_id, new_id = e.id, prior.id, new_task.id
    db.close()

    # 文件存在预检天然过（tmp 文件真实存在）；身份判定打桩直接给 duplicate；
    # VLM 预检放行（否则未就绪会把工单退回 QUEUED）
    import src.core.vlm_readiness as vlm_ready_mod
    monkeypatch.setattr(vlm_ready_mod, "is_vlm_ready", lambda cfg: (True, ""))
    monkeypatch.setattr(
        api_main, "resolve_content_identity",
        lambda db_, t, fp: ("duplicate",
                            db_.get(FileEntity, e_id),
                            db_.get(Task, prior_id)))

    api_main.run_audit_worker_sync(new_id)

    db = TestSession()
    t = db.get(Task, new_id)
    assert t.status == "COMPLETE"
    assert t.progress == 100
    db.close()
