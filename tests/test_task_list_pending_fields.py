"""回归：未建立内容实体的 PENDING 任务，任务列表也必须显示文件名/大小/时长。

1.2.31 起「导入只登记作业」（``tasks.entity_id=NULL``，不建 entity/location），
而 ``GET /task`` 的路径 / 大小 / 时长原先**只从 ``task.entity`` 取** ——
于是刚导入的「未启动」任务在列表里**没有文件名**（用户 2026-09-12 反馈）。

本用例锁住：无实体时回退到 ``task.file_path`` / ``task.file_size`` / ``task.duration``；
有实体时仍以实体（内容）为主。
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
from src.db.models import Base, FileEntity, FileLocation, Task


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'tasks.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    return Session


def _client():
    return TestClient(M.app)


def _find(payload, file_path):
    return next((t for t in payload["tasks"] if t["file_path"] == file_path), None)


class TestPendingTaskListFields:
    def test_pending_without_entity_still_has_name_size_duration(self, env):
        """核心回归：没实体也要有文件名/大小/时长。"""
        Session = env
        db = Session()
        db.add(Task(
            entity_id=None, file_path="/Volumes/MediaData/mp4/IMG_0644.mp4",
            file_md5=None, file_size=31578778, duration=226.72, status="PENDING",
        ))
        db.commit()
        db.close()

        r = _client().get("/task")
        assert r.status_code == 200
        t = _find(r.json(), "/Volumes/MediaData/mp4/IMG_0644.mp4")
        assert t is not None
        assert t["file_name"] == "IMG_0644.mp4", "未建立内容实体的 PENDING 任务必须有文件名"
        assert t["folder_path"] == "/Volumes/MediaData/mp4"
        assert t["file_size"] == 31578778
        assert t["duration"] == 226.72

    def test_entity_backed_task_prefers_entity(self, env):
        """有实体时仍以实体主路径 / 实体元数据为准（不回退脏字段）。"""
        Session = env
        db = Session()
        entity = FileEntity(file_md5="a" * 32, file_size=999, duration=12.5, media_type="video")
        db.add(entity)
        db.flush()
        db.add(FileLocation(
            entity_id=entity.id, file_path="/Volumes/MediaData/x/real.mp4",
            file_name="real.mp4", is_primary=1,
        ))
        # task.file_path 故意留了个陈旧的旧路径，验证仍以实体主路径为准
        db.add(Task(
            entity_id=entity.id, file_path="/old/stale.mp4",
            file_size=1, duration=1.0, status="COMPLETE",
        ))
        db.commit()
        db.close()

        r = _client().get("/task")
        t = _find(r.json(), "/Volumes/MediaData/x/real.mp4")
        assert t is not None, "有实体时应使用实体的主路径"
        assert t["file_name"] == "real.mp4"
        assert t["file_size"] == 999
        assert t["duration"] == 12.5

    def test_helpers_fall_back_without_entity(self):
        """直接锁 helper 行为（无 session 的瞬时对象 entity 为 None）。"""
        t = Task(entity_id=None, file_path="/x/a.mp4", file_size=10, duration=5.0, status="PENDING")
        assert M._get_primary_path(t) == "/x/a.mp4"
        info = M._get_task_file_info(t)
        assert info["file_path"] == "/x/a.mp4"
        assert info["file_size"] == 10
        assert info["duration"] == 5.0
