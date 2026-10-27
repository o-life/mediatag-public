"""分析主题切换器（仿 Obsidian vault switcher）。

用户决策（2026-09-19）：列表**只显示名称**、路径进 tooltip；**不**纳管外部目录；
标签管理页**完全不提**主题（入口抬到顶栏常驻 + 设置页）。

跑法（仓库约定：Qt 测试要先导入 desktop_app，否则 macOS 上 Qt+onnxruntime 会同进程崩）::

    VT_UI_TESTS=1 python -c "import sys;sys.path.insert(0,'src');import ui.desktop_app;
    import pytest;sys.exit(pytest.main(['tests/test_theme_switcher_dialog.py','-q']))"
"""

import os
import sys

import pytest

if not os.environ.get("VT_UI_TESTS"):
    pytest.skip("设置 VT_UI_TESTS=1 才跑 Qt 测试", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.dialogs.theme_switcher_dialog import ThemeSwitcherDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])


class FakeAPI:
    def __init__(self, themes=None, current="男男"):
        self.themes = themes if themes is not None else [
            {"id": 6, "name": "男男", "faces_dir": "/x/themes/男男/faces/", "is_default": 0},
            {"id": 2, "name": "成人内容", "faces_dir": "/x/themes/成人内容/faces/", "is_default": 0},
            {"id": 1, "name": "日常", "faces_dir": "/x/themes/日常/faces/", "is_default": 1},
        ]
        self.current = current
        self.posts = []

    def get(self, path):
        assert path == "/themes"
        return {"current": self.current, "themes": self.themes}

    def post(self, path, json=None):
        self.posts.append((path, json))
        return {"status": "success", "restart_required": True}


def _dlg(api=None):
    return ThemeSwitcherDialog(api or FakeAPI(), None)


class TestThemeDirTooltip:
    def test_workspace_dir_from_faces_child(self):
        """faces_dir 是 <主题>/faces → 展示 <主题>（路径只进 tooltip）。"""
        assert ThemeSwitcherDialog._theme_dir({"faces_dir": "/x/themes/男男/faces/"}) == "/x/themes/男男"

    def test_other_path_shown_as_is(self):
        assert ThemeSwitcherDialog._theme_dir({"faces_dir": "/Volumes/MediaData/f"}) == "/Volumes/MediaData/f"

    def test_missing_path(self):
        assert ThemeSwitcherDialog._theme_dir({}) == "（未指定）"


class TestListRendering:
    def test_name_only(self):
        """列表项**只有名称**（不含路径）；路径在 tooltip。"""
        dlg = _dlg()
        texts = [dlg.list.item(i).text() for i in range(dlg.list.count())]
        assert "日常 ★" in texts
        assert "男男" in texts or "✓ 男男" in texts
        for text in texts:
            assert "/x/themes" not in text and "/Volumes" not in text

    def test_current_marked_with_check(self):
        dlg = _dlg()
        marked = [dlg.list.item(i).text() for i in range(dlg.list.count())
                  if dlg.list.item(i).text().startswith("✓")]
        assert marked == ["✓ 男男"]

    def test_tooltip_carries_path(self):
        dlg = _dlg()
        item = next(dlg.list.item(i) for i in range(dlg.list.count())
                    if "成人内容" in dlg.list.item(i).text())
        assert "/x/themes/成人内容" in item.toolTip()


class TestButtons:
    def test_current_theme_button_disabled(self):
        dlg = _dlg()
        dlg.list.setCurrentRow(0)          # 男男（当前）
        assert dlg.switch_btn.isEnabled() is False
        assert dlg.switch_btn.text() == "当前主题"

    def test_other_theme_button_enabled(self):
        dlg = _dlg()
        dlg.list.setCurrentRow(1)          # 成人内容
        assert dlg.switch_btn.isEnabled() is True
        assert dlg.switch_btn.text() == "切换到该主题"


class TestSearch:
    def test_filter_hides_non_matching(self):
        dlg = _dlg()
        dlg.search.setText("男男")
        visible = [dlg.list.item(i).text() for i in range(dlg.list.count())
                   if not dlg.list.item(i).isHidden()]
        assert visible == ["✓ 男男"]

    def test_empty_search_shows_all(self):
        dlg = _dlg()
        dlg.search.setText("男男")
        dlg.search.setText("")
        assert all(not dlg.list.item(i).isHidden() for i in range(dlg.list.count()))


class TestSwitchFlow:
    def test_switch_posts_activate_and_emits(self, monkeypatch):
        api = FakeAPI()
        dlg = _dlg(api)
        seen = []
        dlg.switched.connect(seen.append)
        # 跳过确认框与重启提示（GUI 交互不在此测）
        monkeypatch.setattr("ui.dialogs.theme_switcher_dialog.QMessageBox.question",
                            lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.Yes)
        monkeypatch.setattr("ui.dialogs.theme_switcher_dialog.prompt_restart",
                            lambda *a, **k: False)

        dlg.list.setCurrentRow(1)          # 成人内容
        dlg._switch()

        assert api.posts == [("/themes/2/activate", None)]
        assert seen == ["成人内容"]

    def test_switch_noop_on_current(self, monkeypatch):
        api = FakeAPI()
        dlg = _dlg(api)
        dlg.list.setCurrentRow(0)          # 当前主题
        dlg._switch()
        assert api.posts == []
