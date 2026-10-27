"""导入不得把「抽帧目录」当媒体（方案 B：名字后缀 + 同目录同名视频）。

背景：抽帧目录固定为「视频同目录 / `<视频名>_frames/`」（见 api/main.py 抽帧、
cache_manager.scan_frame_caches）。导入原本不排除它，导致 `unit_*/frame_*.jpg`
被逐帧当成照片登记（2026-09 用户报告，生产库一次积压 5481 条）。

规则 B：目录名以 `_frames` 结尾 **且** 同目录存在同名视频文件才算抽帧目录 ——
避免误伤用户真叫 `xxx_frames` 的相册目录。
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
from src.core.cache_manager import is_frames_dir, is_inside_frames_dir
from src.db.models import Base, Task


class TestIsFramesDir:
    def test_true_when_sibling_video_exists(self, tmp_path):
        (tmp_path / "movie.mp4").write_bytes(b"x")
        d = tmp_path / "movie_frames"
        d.mkdir()
        assert is_frames_dir(d) is True

    def test_false_without_sibling_video(self, tmp_path):
        (tmp_path / "movie_frames").mkdir()
        assert is_frames_dir(tmp_path / "movie_frames") is False

    def test_false_when_sibling_is_a_photo(self, tmp_path):
        """真叫 xxx_frames 的相册目录（旁边是照片）不能被误判。"""
        (tmp_path / "album.jpg").write_bytes(b"x")
        (tmp_path / "album_frames").mkdir()
        assert is_frames_dir(tmp_path / "album_frames") is False

    def test_true_for_uppercase_video_ext(self, tmp_path):
        (tmp_path / "IMG_0275.MP4").write_bytes(b"x")
        (tmp_path / "IMG_0275_frames").mkdir()
        assert is_frames_dir(tmp_path / "IMG_0275_frames") is True

    def test_false_for_plain_dir(self, tmp_path):
        (tmp_path / "photos").mkdir()
        assert is_frames_dir(tmp_path / "photos") is False

    def test_false_for_bare_frames_suffix(self, tmp_path):
        (tmp_path / "_frames").mkdir()
        assert is_frames_dir(tmp_path / "_frames") is False


class TestIsInsideFramesDir:
    def test_nested_frame_file(self, tmp_path):
        (tmp_path / "movie.mp4").write_bytes(b"x")
        frame = tmp_path / "movie_frames" / "unit_0" / "frame_001.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"x")
        assert is_inside_frames_dir(str(frame)) is True

    def test_normal_photo_not_inside(self, tmp_path):
        p = tmp_path / "album" / "a.jpg"
        p.parent.mkdir()
        p.write_bytes(b"x")
        assert is_inside_frames_dir(str(p)) is False


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'frames.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    monkeypatch.setattr(M, "_probe_video_duration", lambda p: None)
    M._update_import_progress(active=False, total=0, processed=0, imported=0,
                              duplicate=0, failed=0, started_at=None, finished_at=None)
    return Session


def _client():
    return TestClient(M.app)


class TestImportExcludesFrames:
    def test_directory_walk_skips_frames(self, env, tmp_path):
        media = tmp_path / "lib"
        media.mkdir()
        (media / "movie.mp4").write_bytes(b"x")
        frames = media / "movie_frames" / "unit_0"
        frames.mkdir(parents=True)
        for i in range(3):
            (frames / f"frame_{i:03d}.jpg").write_bytes(b"x")

        r = _client().post("/task/import", json={"directory": str(media)})
        assert r.json()["imported"] == 1, "只应导入 movie.mp4，抽帧图片必须被剪枝"

        db = env()
        paths = [t.file_path for t in db.query(Task).all()]
        assert paths == [str(media / "movie.mp4")]

    def test_files_list_also_filtered(self, env, tmp_path):
        media = tmp_path / "lib2"
        media.mkdir()
        video = media / "movie.mp4"
        video.write_bytes(b"x")
        frames = media / "movie_frames" / "unit_0"
        frames.mkdir(parents=True)
        frame = frames / "frame_000.jpg"
        frame.write_bytes(b"x")

        r = _client().post("/task/import", json={"files": [str(video), str(frame)]})
        assert r.json()["imported"] == 1, "前端直传 files 时也要剔除抽帧图片"

        db = env()
        assert [t.file_path for t in db.query(Task).all()] == [str(video)]

    def test_legit_frames_named_album_is_kept(self, env, tmp_path):
        """用户真叫 xxx_frames 的相册目录（旁边是同名照片）不应被误伤。"""
        media = tmp_path / "lib3"
        media.mkdir()
        (media / "album.jpg").write_bytes(b"x")
        album = media / "album_frames"
        album.mkdir()
        (album / "p1.jpg").write_bytes(b"x")

        r = _client().post("/task/import", json={"directory": str(media)})
        assert r.json()["imported"] == 2, "album.jpg 与 album_frames/p1.jpg 都应导入"
