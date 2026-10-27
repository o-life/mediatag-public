"""项目与素材挂接 API 测试：项目/类型词表/分组 CRUD 与筛选/素材挂接。

与 test_actor_split.py 同构：临时 SQLite + 打桩 get_db_engine/get_session，
直接调用路由函数（绕过 HTTP 层；路由函数均为同步实现）。
注意：helper 返回主键 id 而非 ORM 对象——session close 后 ORM 实例过期，
再访问属性会 DetachedInstanceError（本文件首轮实测踩过）。
"""

import sys

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "src")

from src.db.models import Base, Client, EventType, Project, ProjectAsset, Task  # noqa: E402
from src.api.routes import studio  # noqa: E402


@pytest.fixture()
def db_env(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(studio, "get_db_engine", lambda: engine)
    monkeypatch.setattr(studio, "get_session", lambda engine: TestSession())
    db = TestSession()
    yield db
    db.close()


def _mk_client(db, name="张三&李四") -> int:
    c = Client(name=name)
    db.add(c)
    db.commit()
    return c.id


def _mk_type(db, name="婚礼", builtin=1) -> int:
    et = EventType(name=name, is_builtin=builtin, sort_order=1, active=1)
    db.add(et)
    db.commit()
    return et.id


def _mk_project(db, client_id, name="张三婚礼", type_id=None, date="2026-10-01", status="拍摄中") -> int:
    import datetime

    p = Project(
        client_id=client_id,
        event_type_id=type_id,
        name=name,
        event_date=datetime.date.fromisoformat(date) if date else None,
        status=status,
    )
    db.add(p)
    db.commit()
    return p.id


def _mk_task(db, path, project_id=None, md5=None) -> int:
    t = Task(file_path=path, status="COMPLETE", project_id=project_id,
             file_md5=md5 or f"md5-{path}")
    db.add(t)
    db.commit()
    return t.id


# ── 分组 ─────────────────────────────────────────────────────────────


def test_client_crud_roundtrip(db_env):
    cid = _mk_client(db_env, "张三&李四")
    r = studio.create_client({"name": "workshop-a", "contact": "13900"})
    studio.update_client(cid, {"name": "workshop-a-2"})

    lst = studio.list_clients(page=1, size=20)
    assert lst["total"] == 2
    by_name = {c["name"]: c for c in lst["clients"]}
    assert by_name["workshop-a-2"]["contact"] is None  # helper 未写联系方式
    assert by_name["workshop-a"]["contact"] == "13900"

    assert studio.delete_client(r["id"])["status"] == "success"
    assert studio.list_clients(page=1, size=20)["total"] == 1


def test_client_create_requires_name(db_env):
    with pytest.raises(HTTPException) as ei:
        studio.create_client({"name": "  "})
    assert ei.value.status_code == 400


def test_client_delete_refused_with_projects(db_env):
    db = db_env
    cid = _mk_client(db)
    _mk_project(db, cid)
    db.close()
    with pytest.raises(HTTPException) as ei:
        studio.delete_client(cid)
    assert ei.value.status_code == 409


# ── 活动类型词表 ─────────────────────────────────────────────────────


def test_event_type_lifecycle(db_env):
    db = db_env
    custom_id = _mk_type(db, "汉服约拍", builtin=0)
    builtin_id = _mk_type(db, "婚礼", builtin=1)
    db.close()

    lst = studio.list_event_types()
    names = {t["name"] for t in lst["event_types"]}
    assert "汉服约拍" in names and "婚礼" in names

    # 预置类型不可删除（可停用）
    with pytest.raises(HTTPException) as ei:
        studio.delete_event_type(builtin_id)
    assert ei.value.status_code == 400

    # 未被引用的自定义类型可删
    assert studio.delete_event_type(custom_id)["status"] == "success"


# ── 活动（项目）──────────────────────────────────────────────────────


def test_project_create_auto_name_and_get(db_env):
    db = db_env
    cid = _mk_client(db)
    etid = _mk_type(db)
    db.close()

    r = studio.create_project({"client_id": cid, "event_type_id": etid, "event_date": "2026-10-01"})
    pid = r["id"]
    # 自动命名：{名称}·{类型} {MM-DD}
    assert r["name"] == "张三&李四·婚礼 10-01"

    detail = studio.get_project(pid)
    assert detail["client_name"] == "张三&李四"
    assert detail["event_type"] == "婚礼"
    assert detail["asset_count"] == 0


def test_project_assign_unassign(db_env):
    """清单语义：assign 写 project_assets（锚 md5），影子列停写；重复加入幂等。"""
    db = db_env
    cid = _mk_client(db)
    pid = _mk_project(db, cid)
    t1 = _mk_task(db, "/v/a.jpg", md5="md5-a")
    t2 = _mk_task(db, "/v/b.jpg", md5="md5-b")
    unbound = _mk_task(db, "/v/c.jpg", md5="md5-c")
    db.close()

    r = studio.assign_tasks(pid, {"task_ids": [t1, t2]})
    assert r["assigned"] == 2 and r["duplicates"] == 0
    db = studio.get_session(studio.get_db_engine())
    md5s = {m for (m,) in db.query(ProjectAsset.file_md5)
            .filter(ProjectAsset.project_id == pid).all()}
    assert md5s == {"md5-a", "md5-b"}
    assert db.get(Task, unbound).project_id is None  # 影子列停写
    db.close()

    r2 = studio.assign_tasks(pid, {"task_ids": [t1]})
    assert r2["assigned"] == 0 and r2["duplicates"] == 1

    assert studio.unassign_tasks({"task_ids": [t1]})["status"] == "success"
    db = studio.get_session(studio.get_db_engine())
    md5s = {m for (m,) in db.query(ProjectAsset.file_md5)
            .filter(ProjectAsset.project_id == pid).all()}
    assert md5s == {"md5-b"}
    db.close()


def test_project_delete_unassigns_tasks(db_env):
    """删除项目 = 仅摘除清单行：工单/内容本体不动，成员资格消失（回未归集）。"""
    db = db_env
    cid = _mk_client(db)
    pid = _mk_project(db, cid)
    tid = _mk_task(db, "/v/x.jpg", md5="md5-x")
    db.close()

    studio.assign_tasks(pid, {"task_ids": [tid]})
    assert studio.delete_project(pid)["status"] == "success"
    db = studio.get_session(studio.get_db_engine())
    assert db.get(Task, tid) is not None  # 工单本身不删
    assert db.query(ProjectAsset).filter(
        ProjectAsset.project_id == pid).count() == 0  # 清单行摘除
    db.close()


def test_project_filters(db_env):
    db = db_env
    cid = _mk_client(db)
    et1 = _mk_type(db, "婚礼")
    et2 = _mk_type(db, "写真", builtin=0)
    _mk_project(db, cid, name="P1", type_id=et1, date="2026-10-01", status="已交付")
    _mk_project(db, cid, name="P2", type_id=et2, date="2026-09-01", status="拍摄中")
    db.close()

    r = studio.list_projects(event_type_id=et1, page=1, size=20)
    assert [x["name"] for x in r["projects"]] == ["P1"]
    r = studio.list_projects(status="拍摄中", page=1, size=20)
    assert [x["name"] for x in r["projects"]] == ["P2"]
    r = studio.list_projects(date_from="2026-09-15", date_to="2026-10-15", page=1, size=20)
    assert [x["name"] for x in r["projects"]] == ["P1"]


def test_project_custom_fields_roundtrip(db_env):
    """自定义字段：创建时写入 → PATCH 更新 → GET 返回 dict；非法输入容错。"""
    db = db_env
    cid = _mk_client(db)
    db.close()

    r = studio.create_project({
        "client_id": cid,
        "name": "字段项目",
        "custom_fields": {"化妆师": "小王", "交付截止": "10-01"},
    })
    pid = r["id"]
    detail = studio.get_project(pid)
    assert detail["custom_fields"] == {"化妆师": "小王", "交付截止": "10-01"}

    studio.update_project(pid, {"custom_fields": {"化妆师": "小李"}})
    detail = studio.get_project(pid)
    assert detail["custom_fields"] == {"化妆师": "小李"}

    # 非法输入（非 dict）→ 容错为空对象，不抛 500
    studio.update_project(pid, {"custom_fields": "oops"})
    detail = studio.get_project(pid)
    assert detail["custom_fields"] == {}
