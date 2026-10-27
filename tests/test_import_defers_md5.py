"""导入只登记「作业」；内容身份（file_entity）在分析首步才建立。

为什么导入不读文件内容（真实事故）：「添加文件夹」对**每个文件算 MD5 = 读一遍完整内容**。
实测 247 个文件 / 79.3 GB：

| 操作 | 耗时 |
|---|---|
| 只记录路径（os.walk + stat，5365 个文件 / 81.7 GB） | **0.181 秒** |
| 算 MD5（读完整内容） | **约 16.7 分钟** |

相差约 5500 倍。用户看到的就是"添加文件夹"后长时间卡在"导入中"、后端轮询超时。

为什么不写"占位 md5"（早期方案的错误）：``file_entities.file_md5`` 是
**NOT NULL + UNIQUE**，而 entity 表示「内容」。导入时还没算 md5 就写假值，
会让同一个内容变成两个 entity（一个真 md5、一个假 md5），
而不是「1 个 entity + 多条 location」。

数据模型的正确形状（见 openspec/changes/separate-import-from-analysis/design.md §3）：

    file_entities  内容   1 ──< n  file_locations  路径
         ↑ 按 md5 唯一                ↑ 按 file_path 唯一
         │
      tasks  作业  ── entity_id（可空）

- 导入：只写 ``tasks``（entity_id=NULL），不建 entity/location、不读内容
- 分析首步：算/复用 MD5 → 按 MD5 找 entity（命中只补一条 location，未命中才新建）
- 同内容已被/正被处理 ⇒ 判为 ``DUPLICATE``，跳过后续全部业务
"""

import inspect
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
from src.db.models import Base, FileEntity, FileLocation, Task

REAL_MD5 = "14ade155161c05f6b0144d2957edcfc8"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """临时库 + 把模块级 engine/session 指向它；ffprobe 打桩避免依赖真实媒体。"""
    engine = create_engine(f"sqlite:///{tmp_path/'import.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    monkeypatch.setattr(M, "_probe_video_duration", lambda p: None)
    return Session, tmp_path


def _media_dir(tmp, count=3, big_mb=0):
    media = tmp / "media"
    media.mkdir(exist_ok=True)
    for i in range(count):
        (media / f"clip{i}.mp4").write_bytes(b"x" * 1024)
    if big_mb:
        # 稀疏文件：创建瞬时；若被"读完整内容"就会明显耗时
        with open(media / "big.mp4", "wb") as f:
            f.truncate(big_mb * 1024 * 1024)
    return media


def _client():
    return TestClient(M.app)


def _new_task(db, path, status="ANALYZING"):
    """模拟导入产物：只有 tasks 一行，没有 entity / location。"""
    task = Task(entity_id=None, file_path=path, file_md5=None,
                file_size=1024, duration=120.0, status=status)
    db.add(task)
    db.commit()
    return task


class TestImportWritesOnlyTasks:
    def test_never_calls_calculate_file_md5(self, env, monkeypatch):
        Session, tmp = env
        media = _media_dir(tmp, count=3, big_mb=200)

        calls = []

        def _boom(path, *a, **k):
            calls.append(path)
            raise AssertionError(f"导入不得读取文件内容：{path}")

        monkeypatch.setattr(M, "calculate_file_md5", _boom)
        r = _client().post("/task/import", json={"directory": str(media)})

        assert r.status_code == 200, r.text
        assert calls == [], (
            "导入过程中不允许计算 MD5 —— 那是逐文件读完整内容，"
            "会让「添加文件夹」从 0.18 秒变成十几分钟。"
        )
        assert r.json()["imported"] == 4

    def test_creates_no_entity_or_location(self, env):
        """导入只登记作业；内容实体要等内容身份已知（分析首步）才创建。"""
        Session, tmp = env
        media = _media_dir(tmp, count=2)
        _client().post("/task/import", json={"directory": str(media)})

        db = Session()
        assert db.query(FileEntity).count() == 0, "导入阶段不得创建内容实体"
        assert db.query(FileLocation).count() == 0, "导入阶段不得创建路径记录"

        tasks = db.query(Task).all()
        assert len(tasks) == 2
        for t in tasks:
            assert t.status == "PENDING"
            assert t.entity_id is None, "作业尚未关联内容实体"
            assert t.file_md5 is None, "导入阶段不得写任何 md5（真值假值都不写）"
            assert t.file_path, "路径必须记录（这是导入的本职）"

    def test_same_path_imported_twice_is_skipped(self, env):
        Session, tmp = env
        media = _media_dir(tmp, count=2)

        first = _client().post("/task/import", json={"directory": str(media)}).json()
        second = _client().post("/task/import", json={"directory": str(media)}).json()

        assert first["imported"] == 2
        assert second["imported"] == 0
        assert second["duplicate"] == 2, "同一路径已有未结束作业，应跳过"

        db = Session()
        assert db.query(Task).count() == 2, "同一路径不得重复登记出新作业"

    def test_import_source_has_no_md5_computation(self):
        """源码级锁：导入函数体内不得出现 calculate_file_md5。"""
        src = inspect.getsource(M.import_videos)
        assert "calculate_file_md5" not in src, (
            "import_videos 里出现了 calculate_file_md5 —— 会把「添加文件夹」"
            "重新变成逐文件读完整内容。内容身份应在分析首步建立。"
        )


class TestContentIdentity:
    """`resolve_content_identity`：把「作业」变成「内容 + 路径」。"""

    def test_creates_entity_and_location(self, env, monkeypatch):
        Session, _ = env
        db = Session()
        task = _new_task(db, "/x/a.mp4")
        monkeypatch.setattr(M, "calculate_file_md5", lambda p: REAL_MD5)

        state, entity, prior = M.resolve_content_identity(db, task, "/x/a.mp4")

        assert state == "ok" and prior is None
        db.expire_all()
        assert entity.file_md5 == REAL_MD5, "实体的 md5 必须是算出来的真实值"
        assert task.entity_id == entity.id, "作业应关联到内容实体"
        assert task.file_md5 == REAL_MD5
        locs = db.query(FileLocation).all()
        assert len(locs) == 1
        assert locs[0].file_path == "/x/a.mp4" and locs[0].is_primary == 1

    def test_same_content_two_paths_is_one_entity_two_locations(self, env, monkeypatch):
        """核心模型：一个内容 ↔ 多条路径（而不是两个实体）。"""
        Session, _ = env
        db = Session()
        monkeypatch.setattr(M, "calculate_file_md5", lambda p: REAL_MD5)

        first = _new_task(db, "/x/a.mp4", status="COMPLETE")
        assert M.resolve_content_identity(db, first, "/x/a.mp4")[0] == "ok"

        second = _new_task(db, "/y/b.mp4", status="ANALYZING")
        state, entity, prior = M.resolve_content_identity(db, second, "/y/b.mp4")

        assert state == "duplicate", "同内容已被处理 ⇒ 返回 duplicate，调用方跳过业务"
        assert prior is not None and prior.id == first.id
        assert db.query(FileEntity).count() == 1, "同一内容只能有一个实体"
        paths = sorted(l.file_path for l in db.query(FileLocation).all())
        assert paths == ["/x/a.mp4", "/y/b.mp4"], "多路径记录在 file_locations 上"
        assert second.entity_id == entity.id

    def test_reuses_stored_md5_without_recompute(self, env, monkeypatch):
        """已有真实 md5 ⇒ 不重算（历史上同一批文件被读了两遍）。"""
        Session, _ = env
        db = Session()
        entity = FileEntity(file_md5=REAL_MD5, file_size=1, media_type="video")
        db.add(entity)
        db.flush()
        db.add(FileLocation(entity_id=entity.id, file_path="/x/a.mp4",
                            file_name="a.mp4", is_primary=1))
        task = Task(entity_id=entity.id, file_path="/x/a.mp4", file_md5=REAL_MD5,
                    status="ANALYZING")
        db.add(task)
        db.commit()

        monkeypatch.setattr(M, "calculate_file_md5", lambda p: pytest.fail("不应重算 MD5"))
        assert M.resolve_content_identity(db, task, "/x/a.mp4")[0] == "ok"

    def test_error_when_md5_fails(self, env, monkeypatch):
        Session, _ = env
        db = Session()
        task = _new_task(db, "/x/a.mp4")

        def _boom(p):
            raise OSError("读取失败")

        monkeypatch.setattr(M, "calculate_file_md5", _boom)
        state, entity, prior = M.resolve_content_identity(db, task, "/x/a.mp4")

        assert state == "error" and entity is None
        assert db.query(FileEntity).count() == 0, "算不出身份就不该留下实体"

    def test_media_type_by_extension(self):
        assert M._media_type_of("/x/a.mp4") == "video"
        assert M._media_type_of("/x/a.JPG") == "photo"


class TestWorkerWiring:
    def test_worker_never_computes_md5(self):
        """任务执行器本身不得读文件内容算 md5（身份解析已封装在 helper 内）。"""
        src = inspect.getsource(M.run_audit_worker_sync)
        assert "calculate_file_md5" not in src

    def test_worker_resolves_identity_before_analysis_record(self):
        """身份必须先于 FileAnalysis 创建 —— 它是 entity_id 的来源。"""
        src = inspect.getsource(M.run_audit_worker_sync)
        assert "resolve_content_identity" in src
        assert src.index("resolve_content_identity") < src.index("FileAnalysis(")
