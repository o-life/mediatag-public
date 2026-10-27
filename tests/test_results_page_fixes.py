"""Tests for 两个修复:
1. 结果页"宫格↔列表"切换(切回宫格时不能因悬空 C++ 引用而崩)
2. 取封面时不再探测 macOS 受保护目录(桌面/文稿/下载) → 不再弹系统授权框
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.api.main import _is_tcc_protected_path, _resolve_cover_path


# ---------------------------------------------------------------------------
# 1. TCC 保护目录判定
# ---------------------------------------------------------------------------

class TestTccProtectedPath:
    def test_desktop(self):
        assert _is_tcc_protected_path(Path.home() / "Desktop") is True

    def test_documents_and_downloads(self):
        assert _is_tcc_protected_path(Path.home() / "Documents") is True
        assert _is_tcc_protected_path(Path.home() / "Downloads") is True

    def test_chinese_names(self):
        assert _is_tcc_protected_path(Path.home() / "桌面") is True
        assert _is_tcc_protected_path(Path.home() / "下载") is True

    def test_icloud_drive(self):
        p = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Desktop"
        assert _is_tcc_protected_path(p) is True

    def test_normal_dirs_not_protected(self):
        assert _is_tcc_protected_path("/Volumes/Media/unsorted") is False
        assert _is_tcc_protected_path(Path.home() / "Movies") is False
        assert _is_tcc_protected_path(Path.home() / "Library" / "Application Support") is False

    def test_deep_path_inside_desktop(self):
        assert _is_tcc_protected_path(Path.home() / "Desktop" / "sub" / "deep") is True

    def test_home_itself_not_protected(self):
        assert _is_tcc_protected_path(Path.home()) is False

    def test_bad_input_is_false(self):
        assert _is_tcc_protected_path(None) is False
        assert _is_tcc_protected_path("") is False
        assert _is_tcc_protected_path("relative/path") is False


# ---------------------------------------------------------------------------
# 2. 取封面不得探测受保护目录
# ---------------------------------------------------------------------------

class TestResolveCoverSkipsProtected:
    def test_desktop_video_does_not_probe_desktop(self, monkeypatch):
        """对桌面视频, 不应 stat/glob 其所在目录（否则触发授权弹窗）。"""
        touched = []
        real_exists = Path.exists

        def spy_exists(self):
            s = str(self)
            if "/Desktop/" in s or s.endswith("/Desktop"):
                touched.append(s)
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", spy_exists)
        out = _resolve_cover_path("/Users/demo/Desktop/recording.mov")
        assert touched == [], f"不应探测桌面路径, 实际访问: {touched}"
        assert out is None

    def test_downloads_video_does_not_probe(self, monkeypatch):
        touched = []
        real_exists = Path.exists

        def spy_exists(self):
            if "/Downloads/" in str(self):
                touched.append(str(self))
            return real_exists(self)

        monkeypatch.setattr(Path, "exists", spy_exists)
        _resolve_cover_path(f"{Path.home()}/Downloads/影片.mov")
        assert touched == [], f"不应探测下载目录: {touched}"

    def test_frames_path_still_wins(self, tmp_path):
        """DB 的 frames_path 仍是第一优先(不受该保护影响)。"""
        unit = tmp_path / "unit_0"
        unit.mkdir(parents=True)
        f = unit / "frame_000.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
        out = _resolve_cover_path("/Users/demo/Desktop/whatever.mov", str(tmp_path))
        assert out == str(f), f"应使用 frames_path 取到封面, 实际 {out}"

    def test_normal_dir_still_falls_back(self, tmp_path):
        """非受保护目录的推导回退仍然工作(不误伤普通路径)。"""
        video = tmp_path / "clip.mov"
        video.write_bytes(b"\x00")
        unit = tmp_path / "clip_frames" / "unit_0"
        unit.mkdir(parents=True)
        f = unit / "frame_000.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
        out = _resolve_cover_path(str(video))
        assert out == str(f), f"普通目录应仍能推导出封面, 实际 {out}"


# 说明: 结果页"宫格↔列表"切换的回归不在 pytest 里做 —— 需要真实 QApplication +
# 真实运行库, 且伪造"已销毁的 C++ 对象"在 C++ 层不安全(会段错误而非可捕获异常)。
# 该场景用无头复现脚本验证(见提交说明), 覆盖: 宫格→列表→宫格 能正常回到宫格。
