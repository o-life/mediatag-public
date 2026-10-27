"""zz_band_alignment — 顶带-操作行同线契约（真主窗口）。

⚠️ 本文件必须保持字母序**最后**（z 前缀）：构造真 MainWindow 会一次性
改变应用级渲染状态（调色板/字体/全局样式，还原不彻底），先跑会挤压
后续像素类测试的页面渲染（实测黄金基线 7 连挂）。放最后跑=零影响。
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

if os.environ.get("VT_UI_TESTS") != "1":
    pytest.skip(
        "Qt UI 交互测试需显式开启: VT_UI_TESTS=1",
        allow_module_level=True,
    )

import ui.desktop_app  # noqa: F401  先初始化 Qt 插件路径
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def main_win(qapp):
    """真主窗口夹具（顶带契约需要）。

    MainWindow 构造会改 QApplication 级状态（调色板/字体/全局样式）——
    不还原会泄漏进后续模块的页面渲染（golden 基线漂移，实测 7 连挂）。
    进出快照还原。"""
    import ui.desktop_app as da
    app = QApplication.instance()
    saved_palette = QApplication.instance().palette()
    saved_font = QApplication.instance().font()
    saved_style = app.styleSheet()
    w = da.MainWindow()
    w.resize(1440, 900)
    w.show()
    qapp.processEvents()
    yield w
    for _ in range(3):
        qapp.processEvents()
    w.close()
    app.setPalette(saved_palette)
    app.setFont(saved_font)
    app.setStyleSheet(saved_style)
    for _ in range(3):
        qapp.processEvents()


class TestBandActionAlignment:
    """顶带内容槽首钮与操作行首钮必须同 x（树列收起/展开两态）。"""

    @staticmethod
    def _assert_aligned(win, page_idx, qapp):
        win._switch_page(page_idx)
        win._toggle_sidebar(True)   # 收起态（对齐难点：过渡态 x 未落位）
        win._sync_band()
        QTimer.singleShot(0, win._sync_band)
        qapp.processEvents()
        band_x = win.band_stack.geometry().x() + win.title_band.geometry().x()
        content_x = win.content_stack.geometry().x()
        page = win.content_stack.widget(page_idx)
        ab = page.action_bar if page_idx == 1 else page.action_panel
        inner = ab.widget() if hasattr(ab, "widget") else ab
        # 首钮 x = 容器 x + 布局左边距
        first_btn = min((b for b in inner.findChildren(QPushButton)
                         if b.isVisibleTo(inner)), key=lambda b: b.geometry().x())
        btn_abs = content_x + inner.geometry().x() + first_btn.geometry().x()
        assert band_x == btn_abs, (
            f"[同线契约] 顶带首钮 {band_x} ≠ 操作行首钮 {btn_abs}（页 {page_idx}）")

    def test_collapsed_band_aligns_action_row(self, main_win, qapp):
        self._assert_aligned(main_win, 1, qapp)
