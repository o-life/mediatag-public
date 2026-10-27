"""Tests for 封面帧选取：跳过全黑帧。

背景：`_resolve_cover_path` 原先对每个文件直接取「第一个场景的第一帧」
（unit_0/frame_0.jpg）。视频开头那一帧常是黑场/淡入，于是封面全黑 →
前端显示「No Cover (全黑)」。实测 1563 条里 124 条中招，且 124/124 都能靠
「往后找一张非全黑的」救回。

注意：黑/非黑判定阈值（亮度 < 10）与前端 results_page.py 的显示判定一致。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M


def _write_frame(path: Path, mean: int) -> Path:
    """写一张平均灰度约等于 mean 的 JPEG（0=全黑）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((48, 48, 3), mean, dtype=np.uint8)
    assert cv2.imwrite(str(path), img), "cv2 写图失败"
    return path


@pytest.fixture(autouse=True)
def _clear_cache():
    M._COVER_FRAME_CACHE.clear()
    yield
    M._COVER_FRAME_CACHE.clear()


class TestPickCoverFrame:
    def test_skips_black_first_frame(self, tmp_path):
        """场景 0 的第一帧全黑 → 应往后取非黑帧（原先就是这里中招）。"""
        _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 0)
        good = _write_frame(tmp_path / "unit_0" / "frame_1.jpg", 128)

        picked = M._pick_cover_frame(tmp_path)
        assert picked == str(good), "应跳过全黑的首帧"

    def test_falls_through_to_later_unit(self, tmp_path):
        """整个场景 0 都黑 → 应到场景 1 找。"""
        _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 0)
        _write_frame(tmp_path / "unit_0" / "frame_1.jpg", 3)
        good = _write_frame(tmp_path / "unit_1" / "frame_0.jpg", 90)

        assert M._pick_cover_frame(tmp_path) == str(good)

    def test_first_frame_kept_when_already_good(self, tmp_path):
        good = _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 120)
        assert M._pick_cover_frame(tmp_path) == str(good)

    def test_all_black_falls_back_to_first(self, tmp_path):
        """整段全黑 → 退回第一帧（保持原行为，至少给个路径）。"""
        first = _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 0)
        _write_frame(tmp_path / "unit_0" / "frame_1.jpg", 1)

        assert M._pick_cover_frame(tmp_path) == str(first)

    def test_flat_frames_without_units(self, tmp_path):
        """没有 unit_* 子目录时的平铺目录（历史上出现过这种帧目录）。"""
        _write_frame(tmp_path / "frame_000.jpg", 0)
        good = _write_frame(tmp_path / "frame_001.jpg", 100)

        assert M._pick_cover_frame(tmp_path) == str(good)

    def test_empty_dir_returns_none(self, tmp_path):
        assert M._pick_cover_frame(tmp_path) is None

    def test_result_is_cached(self, tmp_path):
        """缓存生效：删掉文件后再取仍走缓存（避免每次 /videos 重扫 1500+ 条）。"""
        _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 0)
        good = _write_frame(tmp_path / "unit_0" / "frame_1.jpg", 128)
        assert M._pick_cover_frame(tmp_path) == str(good)

        for p in tmp_path.rglob("*.jpg"):
            p.unlink()
        assert M._pick_cover_frame(tmp_path) == str(good), "应命中缓存"


class TestResolveCoverPath:
    def test_uses_frames_path_and_skips_black(self, tmp_path):
        _write_frame(tmp_path / "unit_0" / "frame_0.jpg", 0)
        good = _write_frame(tmp_path / "unit_0" / "frame_2.jpg", 111)

        got = M._resolve_cover_path(None, str(tmp_path), None)
        assert got == str(good)

    def test_none_when_no_frames(self, tmp_path):
        assert M._resolve_cover_path(None, str(tmp_path / "nope"), None) is None


class TestFrameDarkness:
    def test_black_and_normal(self, tmp_path):
        black = _write_frame(tmp_path / "b.jpg", 0)
        normal = _write_frame(tmp_path / "n.jpg", 128)
        assert M._frame_is_too_dark(str(black)) is True
        assert M._frame_is_too_dark(str(normal)) is False

    def test_threshold_matches_frontend(self):
        """阈值必须与前端 results_page.py 的 `< 10` 一致，否则两处判定会打架。"""
        assert M._BLACK_FRAME_MEAN == 10

    def test_unreadable_file_treated_as_dark(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        assert M._frame_is_too_dark(str(bad)) is True
