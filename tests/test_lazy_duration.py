"""时长惰性获取：导入不探时长，改在激活/分析首步补探。

背景：导入原本对每个视频跑一次 ffprobe（实测约 48ms/文件，661 个文件 ≈ 32s）。
现在导入只登记作业、不探时长；缺的时长在：
- 激活时并行补探（`_fill_missing_durations`，保留「短任务优先」排序）
- 分析首步兜底（`resolve_content_identity` 建实体前，保证 FileEntity.duration 有值）
"""

import asyncio
import inspect
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
from src.db.models import Base, Task

REAL_MD5 = "14ade155161c05f6b0144d2957edcfc8"


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'lazy.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    return Session


class TestImportDoesNotProbeDuration:
    def test_import_source_has_no_ffprobe(self):
        """源码级锁：导入函数体内不得出现 ffprobe / MD5（都后置了）。"""
        src = inspect.getsource(M.import_videos)
        assert "_probe_video_duration" not in src, "导入不得探时长（应惰性获取）"
        assert "calculate_file_md5" not in src, "导入不得读内容算 MD5"


class TestProbeDurations:
    def test_returns_path_to_duration_mapping(self, monkeypatch):
        values = {"/a.mp4": 1.0, "/b.mp4": 2.0}
        monkeypatch.setattr(M, "_probe_video_duration", lambda p: values.get(p))
        assert M._probe_durations(["/a.mp4", "/b.mp4", "/c.mp4"]) == {
            "/a.mp4": 1.0, "/b.mp4": 2.0, "/c.mp4": None,
        }

    def test_empty_input(self):
        assert M._probe_durations([]) == {}


class TestResolveIdentityFillsDuration:
    def test_missing_duration_is_probed_before_entity_created(self, env, monkeypatch):
        Session = env
        db = Session()
        task = Task(entity_id=None, file_path="/x/a.mp4", file_md5=None,
                    file_size=1, duration=None, status="ANALYZING")
        db.add(task)
        db.commit()
        monkeypatch.setattr(M, "calculate_file_md5", lambda p: REAL_MD5)
        monkeypatch.setattr(M, "_probe_video_duration", lambda p: 123.0)

        state, entity, prior = M.resolve_content_identity(db, task, "/x/a.mp4")

        assert state == "ok"
        assert task.duration == 123.0
        assert entity.duration == 123.0, "实体时长必须有值（否则 /videos 显示 Unknown）"

    def test_existing_duration_is_not_reprobed(self, env, monkeypatch):
        Session = env
        db = Session()
        task = Task(entity_id=None, file_path="/x/a.mp4", file_md5=None,
                    file_size=1, duration=50.0, status="ANALYZING")
        db.add(task)
        db.commit()
        monkeypatch.setattr(M, "calculate_file_md5", lambda p: REAL_MD5)
        monkeypatch.setattr(M, "_probe_video_duration",
                            lambda p: pytest.fail("已有真实时长不得重探"))

        _state, entity, _prior = M.resolve_content_identity(db, task, "/x/a.mp4")
        assert entity.duration == 50.0


class TestFillMissingDurations:
    def test_fills_only_videos_missing_duration(self, env, monkeypatch):
        Session = env
        db = Session()
        v1 = Task(entity_id=None, file_path="/x/a.mp4", duration=None, status="PENDING")
        v2 = Task(entity_id=None, file_path="/x/b.mp4", duration=None, status="PENDING")
        photo = Task(entity_id=None, file_path="/x/c.jpg", duration=None, status="PENDING")
        known = Task(entity_id=None, file_path="/x/d.mp4", duration=9.0, status="PENDING")
        db.add_all([v1, v2, photo, known])
        db.commit()

        probed_paths = []

        def _fake_probe(p):
            probed_paths.append(p)
            return 42.0

        monkeypatch.setattr(M, "_probe_video_duration", _fake_probe)
        asyncio.run(M._fill_missing_durations(db, [v1, v2, photo, known]))

        assert v1.duration == 42.0 and v2.duration == 42.0
        assert photo.duration is None, "照片不探时长"
        assert known.duration == 9.0, "已有时长不重探"
        assert sorted(probed_paths) == ["/x/a.mp4", "/x/b.mp4"]

        db.expire_all()
        assert db.query(Task).filter(Task.id == v1.id).first().duration == 42.0, "应已落库"

    def test_noop_when_nothing_missing(self, env, monkeypatch):
        Session = env
        db = Session()
        t = Task(entity_id=None, file_path="/x/a.mp4", duration=5.0, status="PENDING")
        db.add(t)
        db.commit()
        monkeypatch.setattr(M, "_probe_video_duration", lambda p: pytest.fail("不该探测"))
        asyncio.run(M._fill_missing_durations(db, [t]))


class TestBackfillDurations:
    def test_fills_missing_videos_only(self, env, monkeypatch):
        Session = env
        db = Session()
        video = Task(entity_id=None, file_path="/x/a.mp4", duration=None, status="PENDING")
        photo = Task(entity_id=None, file_path="/x/b.jpg", duration=None, status="PENDING")
        db.add_all([video, photo])
        db.commit()
        monkeypatch.setattr(M, "_probe_video_duration", lambda _p: 7.5)

        M._backfill_durations([video.id, photo.id])

        db.expire_all()
        assert db.query(Task).filter(Task.id == video.id).first().duration == 7.5
        assert db.query(Task).filter(Task.id == photo.id).first().duration is None

    def test_noop_on_empty_ids(self, env):
        M._backfill_durations([])

