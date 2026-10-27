"""导入进度上报：``GET /task/import/status`` 反映最近一次导入的进度快照。

背景：导入是同步 ``def``（FastAPI 丢线程池执行），此前前端只有一个静态的
「导入中...」，无从知道进度。新增进程内进度记录 + 轮询接口，让前端显示 N/M。
"""

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
from src.db.models import Base, Task


@pytest.fixture(autouse=True)
def _reset_progress():
    """模块级单例，测试间必须重置，避免相互污染。"""
    M._update_import_progress(
        active=False, total=0, processed=0, imported=0,
        duplicate=0, failed=0, started_at=None, finished_at=None,
    )
    yield


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'prog.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    monkeypatch.setattr(M, "_probe_video_duration", lambda p: 12.0)
    return Session, tmp_path


def _client():
    return TestClient(M.app)


def _media_dir(tmp, count=4):
    media = tmp / "media"
    media.mkdir(exist_ok=True)
    for i in range(count):
        (media / f"clip{i}.mp4").write_bytes(b"x" * 1024)
    return media


class TestImportProgress:
    def test_status_before_any_import_is_idle(self, env):
        st = _client().get("/task/import/status").json()
        assert st["active"] is False
        assert st["total"] == 0

    def test_status_reflects_last_import(self, env):
        Session, tmp = env
        media = _media_dir(tmp, count=4)
        assert _client().post("/task/import", json={"directory": str(media)}).json()["imported"] == 4

        st = _client().get("/task/import/status").json()
        assert st["active"] is False, "导入结束后不应仍标记 active"
        assert st["total"] == 4
        assert st["processed"] == 4
        assert st["imported"] == 4
        assert st["duplicate"] == 0
        assert st["failed"] == 0
        assert st["started_at"] and st["finished_at"]
        assert st["finished_at"] >= st["started_at"]

    def test_duplicates_reflected_in_last_import(self, env):
        Session, tmp = env
        media = _media_dir(tmp, count=3)
        assert _client().post("/task/import", json={"directory": str(media)}).json()["imported"] == 3

        second = _client().post("/task/import", json={"directory": str(media)}).json()
        assert second["imported"] == 0 and second["duplicate"] == 3

        st = _client().get("/task/import/status").json()
        assert st["total"] == 3 and st["processed"] == 3
        assert st["imported"] == 0 and st["duplicate"] == 3

    def test_mid_flight_snapshot(self):
        """进行中：active=True 且能看到 N/M。"""
        M._update_import_progress(active=True, total=10, processed=4, imported=3)
        st = M._snapshot_import_progress()
        assert st["active"] is True
        assert st["total"] == 10 and st["processed"] == 4 and st["imported"] == 3

    def test_snapshot_is_a_copy(self):
        M._update_import_progress(active=True, total=5)
        snap = M._snapshot_import_progress()
        snap["total"] = 999
        assert M._snapshot_import_progress()["total"] == 5, "快照必须是副本，不得被调用方改写"

    def test_import_backfills_duration_after_response(self, env, monkeypatch):
        """导入本身不探时长；响应后的后台任务并行补探（TestClient 会执行 BackgroundTasks）。"""
        Session, tmp = env
        probed = []
        monkeypatch.setattr(M, "_probe_video_duration",
                            lambda p: (probed.append(p), 33.0)[1])
        media = _media_dir(tmp, count=2)

        assert _client().post("/task/import", json={"directory": str(media)}).json()["imported"] == 2

        db = Session()
        assert sorted(t.duration for t in db.query(Task).all()) == [33.0, 33.0]
        assert sorted(probed) == sorted([str(media / "clip0.mp4"), str(media / "clip1.mp4")])

