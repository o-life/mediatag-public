"""Tests for 统计条快捷筛选（点「待处理」→ 列表只显示待处理，与筛选下拉同条件）。

覆盖:
- 点各统计卡片 → _current_filter 正确、下拉勾选同步、对应卡片高亮
- 点「总任务」→ 归一为"不过滤"(all)
- 下拉多选 → 不点亮任何卡片
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# 为什么默认跳过：
# 本环境下，在**已加载 onnxruntime / CoreML 的同一进程**里再初始化 Qt
# (Cocoa / NSApplication) 会让进程直接 abort —— 实测把本文件放在
# test_anime_check(会 import face_recognition) 之后跑，进程崩溃。
# 桌面应用本身是 Qt 先初始化，因此不受影响。
# 纯逻辑已在 tests/test_quick_filter_logic.py 覆盖(默认会跑)；
# 需要跑真实 Qt 交互时显式开启：
#   VT_UI_TESTS=1 uv run --active python -m pytest tests/test_dashboard_quick_filter.py
if os.environ.get("VT_UI_TESTS") != "1":
    pytest.skip(
        "Qt UI 交互测试需显式开启: VT_UI_TESTS=1"
        "（与 onnxruntime 同进程初始化 Qt 会崩溃）",
        allow_module_level=True,
    )

pytest.importorskip("PySide6")

# 触发 Qt 插件 rpath 修正（与桌面应用启动一致），否则 QApplication 可能初始化失败
try:  # pragma: no cover
    import ui.desktop_app  # noqa: F401
except Exception:
    pass


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(qapp, monkeypatch):
    """构造 DashboardPage，并拦掉所有网络请求（返回固定数据）。"""
    from ui.components import dashboard as D
    from ui.components.api_client import APIClient

    def fake_get(self, path, params=None):
        if path == "/status":
            return {"total_tasks": 10, "completed": 4, "processing": 2, "pending": 3}
        if path == "/task":
            return {"tasks": [], "total": 0, "page": 1, "page_size": 50, "total_pages": 1}
        return {}

    monkeypatch.setattr(APIClient, "get", fake_get, raising=True)
    p = D.DashboardPage(api_base="http://127.0.0.1:9")   # 端口无用: get 已替换
    yield p
    p.deleteLater()


def _card(page, name):
    return getattr(page, f"stat_{name}")


def _click(page, key):
    """模拟点击统计卡片(直接触发其 clicked 信号)。"""
    card = {"all": "total", "complete": "completed",
            "processing": "processing", "pending": "pending"}[key]
    _card(page, card).clicked.emit(key)


class TestQuickFilter:
    def test_pending_card_filters_pending(self, page):
        _click(page, "pending")
        assert page._current_filter == "pending"
        assert page.filter_combo.checked_values() == ["pending"]
        assert page.stat_pending._active is True
        assert page.stat_completed._active is False
        assert page.stat_total._active is False

    def test_completed_card_filters_complete(self, page):
        _click(page, "complete")
        assert page._current_filter == "complete"
        assert page.filter_combo.checked_values() == ["complete"]
        assert page.stat_completed._active is True

    def test_processing_card_filters_processing(self, page):
        _click(page, "processing")
        assert page._current_filter == "processing"
        assert page.filter_combo.checked_values() == ["processing"]
        assert page.stat_processing._active is True

    def test_total_card_clears_filter(self, page):
        """点「总任务」= 全勾 = 不过滤。"""
        _click(page, "pending")
        assert page._current_filter == "pending"
        _click(page, "all")
        assert page._current_filter == "all"
        # 下拉应为全勾（摘要不再显示单一状态）
        from ui.components.dashboard import STATUS_FILTER_ITEMS
        assert set(page.filter_combo.checked_values()) == {v for v, _ in STATUS_FILTER_ITEMS}
        assert page.stat_total._active is True
        assert page.stat_pending._active is False

    def test_switching_between_cards(self, page):
        """在卡片之间来回点，筛选与高亮始终跟随。"""
        for key, exp in (("pending", "pending"), ("complete", "complete"),
                         ("processing", "processing"), ("pending", "pending")):
            _click(page, key)
            assert page._current_filter == exp
            assert page.filter_combo.checked_values() == [exp]

    def test_initial_state_highlights_total(self, page):
        assert page._current_filter == "all"
        assert page.stat_total._active is True

    def test_multi_select_clears_card_highlight(self, page):
        """下拉多选(两项) → 不点亮任何卡片。"""
        page.filter_combo.set_checked_values(["pending", "processing"])
        page._on_status_filter_changed(["pending", "processing"])
        assert page._current_filter == "pending,processing"
        assert all(not _card(page, n)._active
                   for n in ("total", "completed", "processing", "pending"))

    def test_all_selected_normalizes_to_all(self, page):
        """下拉手动全勾 → 归一为 all（与点「总任务」同一结果）。"""
        from ui.components.dashboard import STATUS_FILTER_ITEMS
        all_keys = [v for v, _ in STATUS_FILTER_ITEMS]
        page.filter_combo.set_checked_values(all_keys)
        page._on_status_filter_changed(all_keys)
        assert page._current_filter == "all"
        assert page.stat_total._active is True

    def test_cards_are_clickable_widgets(self, page):
        """卡片应可点击(有指针光标与 filter_key)。"""
        from PySide6.QtCore import Qt
        for name in ("total", "completed", "processing", "pending"):
            c = _card(page, name)
            assert c._filter_key, f"{name} 缺 filter_key"
            assert c.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_real_mouse_click_filters(self, page):
        """用真实鼠标事件点击卡片（验证 mousePressEvent 接线，而不只是信号）。"""
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        card = page.stat_pending
        QTest.mouseClick(card, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, card.rect().center())
        assert page._current_filter == "pending"
        assert page.filter_combo.checked_values() == ["pending"]
        assert card._active is True
