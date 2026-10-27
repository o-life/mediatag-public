"""帧缓存：设置页的「统计 / 立即清理」接口 + 自动清帧的保留天数语义。

修复背景：
1. 设置页调用的 ``/frames/cleanup/stats``、``/frames/cleanup`` 在服务端不存在（点不动）；
2. 设置页保存的是 ``cleanup.*``，而后台自动清帧读的是 ``frame_cleanup.*``（设了不生效），
   且阈值 >0 时后台什么都不做。
"""

import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
import src.core.cache_manager as CM


def _make_frames_dir(base: Path, name: str, age_days: float, size_bytes: int = 1000) -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "frame_000.jpg").write_bytes(b"x" * size_bytes)
    ts = time.time() - age_days * 86400
    os.utime(d, (ts, ts))  # 目录 mtime = 年龄（清理按目录 mtime 判定）
    return d


class TestCleanupStats:
    def test_counts_total_and_cleanable(self, tmp_path):
        _make_frames_dir(tmp_path, "a_frames", age_days=40, size_bytes=2_000_000)
        _make_frames_dir(tmp_path, "b_frames", age_days=1)

        stats = CM.get_frame_cache_cleanup_stats(max_age_days=30, scan_dir=str(tmp_path))

        assert stats["total_count"] == 2
        assert stats["cleanable_count"] == 1
        assert stats["cleanable_size_mb"] > 0

    def test_nothing_cleanable_when_all_fresh(self, tmp_path):
        _make_frames_dir(tmp_path, "fresh_frames", age_days=1)
        stats = CM.get_frame_cache_cleanup_stats(max_age_days=30, scan_dir=str(tmp_path))
        assert stats["cleanable_count"] == 0


class TestAutoCleanFrames:
    def test_disabled_keeps_frames(self, tmp_path):
        d = _make_frames_dir(tmp_path, "v_frames", age_days=0)
        M._auto_clean_frames_if_enabled({"enabled": False, "keep_days": 0}, str(d), 1)
        assert d.exists()

    def test_keep_days_zero_removes_immediately(self, tmp_path):
        d = _make_frames_dir(tmp_path, "v_frames", age_days=0)
        M._auto_clean_frames_if_enabled({"enabled": True, "keep_days": 0}, str(d), 1)
        assert not d.exists()

    def test_keep_days_positive_keeps_fresh(self, tmp_path):
        d = _make_frames_dir(tmp_path, "v_frames", age_days=0)
        M._auto_clean_frames_if_enabled({"enabled": True, "keep_days": 7}, str(d), 1)
        assert d.exists(), "刚生成的抽帧目录不应被删（要保留 N 天）"

    def test_keep_days_positive_removes_old(self, tmp_path):
        d = _make_frames_dir(tmp_path, "v_frames", age_days=40)
        M._auto_clean_frames_if_enabled({"enabled": True, "keep_days": 7}, str(d), 1)
        assert not d.exists()

    def test_missing_dir_is_noop(self, tmp_path):
        M._auto_clean_frames_if_enabled({"enabled": True, "keep_days": 0}, str(tmp_path / "nope"), 1)


class TestFramesCleanupRoutes:
    def test_stats_route(self, monkeypatch):
        monkeypatch.setattr(CM, "get_frame_cache_cleanup_stats", lambda max_age_days=30, scan_dir=None: {
            "total_count": 5, "total_size_mb": 12.5,
            "cleanable_count": 2, "cleanable_size_mb": 3.5,
        })
        r = TestClient(M.app).get("/frames/cleanup/stats?days=7")
        assert r.status_code == 200, r.text
        assert r.json()["cleanable_count"] == 2
        assert r.json()["cleanable_size_mb"] == 3.5

    def test_clean_route(self, monkeypatch):
        monkeypatch.setattr(CM, "clean_frame_cache", lambda video_path=None, max_age_days=30: {
            "removed": ["/x/a_frames"], "freed_mb": 1.5,
        })
        r = TestClient(M.app).post("/frames/cleanup", json={"days": 7})
        assert r.status_code == 200, r.text
        assert r.json()["cleaned_count"] == 1
        assert r.json()["cleaned_size_mb"] == 1.5
