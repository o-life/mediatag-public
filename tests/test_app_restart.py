"""应用重启助手（切换分析主题 = 切库，需要冷切重启）。"""

from src.ui.app_restart import (
    bundle_path,
    is_packaged,
    manual_restart_hint,
    restart_application,
)


class TestDevMode:
    def test_not_packaged_under_pytest(self):
        assert is_packaged() is False

    def test_dev_mode_does_not_auto_restart(self):
        """开发态后端是独立进程，无法自动重开 → 返回 False 由 UI 提示手动重启。"""
        assert restart_application() is False

    def test_manual_hint_mentions_both_processes(self):
        hint = manual_restart_hint()
        assert "start-api.sh" in hint
        assert "desktop_app.py" in hint


class TestBundlePath:
    def test_bundle_is_three_levels_up_from_executable(self):
        """打包态可执行文件在 <bundle>/Contents/MacOS/<exe>，上溯三层即 .app。"""
        import sys
        from pathlib import Path

        assert bundle_path() == Path(sys.executable).parent.parent.parent
