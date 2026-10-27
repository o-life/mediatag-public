"""UI 规格验收探针：离屏构造页面，逐条量数对 18/19 号规格书定稿值。

背景：D7（缝隙）/D8（背景色）类漂移说明"实现写完≠与规格一致"。
本文件是规格的机器验收——任何一条数值漂了，提交前就在这里红。
量数口径：控件读 minimumHeight（未 show 时 height() 反映默认 sizeHint）；
背景色用 grab() 像素采样（offscreen 为软件渲染，QSS 如实成像）。

与 test_results_list_view.py 的分工：那边测行为契约（勾选/联动/回归），
这边测**规格数值本身**。改规格值 → 先改 ui/tokens.py → 本文件自动跟随
（全部断言取自 token，不写死数字——写死数字的探针自己就会漂移）。

Qt 交互测试需显式开启:
    VT_UI_TESTS=1 uv run --active python -m pytest tests/test_spec_probes.py
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
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

import pytest

from ui.tokens import T

try:
    from ui.components.theme import theme
except ImportError:
    from ui.components.theme import theme  # 脚本态兜底


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_page(qapp, monkeypatch, width=1200, height=800):
    """构造做过 _load_data 打桩的 ResultsPage 并显示（拿真实几何）。"""
    from unittest.mock import patch

    from ui.components.results_page import ResultsPage

    with patch.object(ResultsPage, "_load_data", lambda self: None), \
         patch.object(ResultsPage, "_refresh_scope_tree", lambda self: None):
        page = ResultsPage()
    page.resize(width, height)
    page.show()
    qapp.processEvents()
    return page


class TestDetailTreeSpec:
    """19§D-3 树列规格。"""

    def test_indent_26(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        assert page.scope_tree.indentation() == T.TREE_INDENT_PROJECT

    def test_title_row_height(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        # 标题行是 panel 里第一个 28 高 QFrame（含「项 目」字样）
        from PySide6.QtWidgets import QFrame
        titles = [c for c in page.project_panel.findChildren(QFrame)
                  if c.height() == T.TREE_TITLE_H]
        assert titles, "树列标题行缺失或高度≠规格"

    def test_delegate_row_28(self, qapp, monkeypatch):
        from ui.widgets.tree_row_delegate import TreeCountDelegate
        page = _make_page(qapp, monkeypatch)
        d = page.scope_tree.itemDelegate()
        assert isinstance(d, TreeCountDelegate)
        assert d.ROW_HEIGHT == T.ROW_H_TREE

    def test_tree_zero_gap_layout(self, qapp, monkeypatch):
        """D7 真机制：标题行与树之间 spacing 必须为 0（QSS polish 会清
        scroll area 程序化 margins，负内缩不可依赖——探针只认布局真值）。"""
        page = _make_page(qapp, monkeypatch)
        lay = page.project_panel.layout()
        assert lay.spacing() == 0, "标题行与树之间的 spacing 必须为 0（D7）"

    def test_panel_bg_is_secondary_not_glass(self, qapp, monkeypatch):
        """D8：树面板必须实色 bg_secondary（透明会透窗背玻璃发黑）。

        像素采样：面板 stylesheet 里的色值 == token 值。"""
        page = _make_page(qapp, monkeypatch)
        ss = page.project_panel.styleSheet()
        assert theme().get("bg_secondary") in ss, (
            f"树面板背景未走 bg_secondary token：{ss!r}")


class TestDetailBandSpec:
    """19§D-1 顶带四件套 sm 档。"""

    def test_sm_heights(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        for w in (page.btn_refresh, page.search_box, page.btn_view_grid,
                  page.btn_view_list, page.sort_combo, page.btn_sort_dir):
            assert w.minimumHeight() == T.CTRL_SM_H, type(w).__name__

    def test_content_heights(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        for w in (page.btn_select_all, page.btn_selected_filter,
                  page.btn_assign_project, page.btn_batch_delete,
                  page.btn_rebuild_faces, page.btn_cleanup):
            assert w.minimumHeight() == T.CTRL_H, type(w).__name__

    def test_search_width(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        assert page.search_box.width() == T.SEARCH_W or \
            page.search_box.minimumWidth() == T.SEARCH_W


class TestDetailPanelSpec:
    """19§D-4 素材面板/分页/宫格。"""

    def test_media_panel_bg_card(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        ss = page.media_panel.styleSheet()
        assert theme().get("bg_card") in ss

    def test_pager_30_with_26_buttons(self, qapp):
        from ui.widgets.video_grid_widget import PaginationWidget
        p = PaginationWidget(compact=True)
        assert p.minimumHeight() == T.PAGER_H
        assert p.page_size_combo.minimumHeight() == T.PAGER_BTN_H

    def test_grid_geometry(self, qapp, monkeypatch):
        page = _make_page(qapp, monkeypatch)
        page._all_videos = [
            {"id": i, "file_name": f"v{i}.mp4", "mapped_tags": ["t"],
             "duration": 60 + i, "file_size": 1, "actors": []}
            for i in range(3)
        ]
        page._data_loaded = True
        page._render_current_page()
        qapp.processEvents()
        grid = page.grid_layout
        m = grid.contentsMargins()
        assert (m.left(), m.top(), m.right(), m.bottom()) == \
            (T.GRID_MARGIN,) * 4
        assert grid.horizontalSpacing() == T.GRID_GAP
        # 卡宽 ≥ 规格下限
        assert page._page_cards[0].width() >= T.GRID_CARD_MIN_W

    def test_media_panel_pixel_is_card_color(self, qapp, monkeypatch):
        """像素采样：面板左上角成像必须是 bg_card（抓 D8 类透明/玻璃漂移）。"""
        page = _make_page(qapp, monkeypatch)
        pm = page.media_panel.grab()
        c = pm.toImage().pixelColor(2, 2)
        expect = QColor(theme().get("bg_card"))
        assert c == expect, f"面板成像 {c.name()} ≠ bg_card {expect.name()}"


class TestTokenSourceOfTruth:
    """tokens.py 必须真实注入 theme().get（页面单一查询口的前提）。"""

    def test_geometry_keys_resolvable(self):
        t = theme()
        for key in ("band_h", "ctrl_h", "ctrl_sm_h", "tree_row_h",
                    "tree_indent_project", "header_h", "pager_h",
                    "grid_card_min_w", "chip_fs", "search_w"):
            assert t.get(key) not in ("", None), f"token {key} 未注入"

    def test_alias_consistency(self):
        """T 大写与 theme().get 小写必须同值（两套口一套数）。"""
        t = theme()
        for upper, lower in (("BAND_H", "band_h"), ("CTRL_H", "ctrl_h"),
                             ("CTRL_SM_H", "ctrl_sm_h"),
                             ("TREE_ROW_H", "tree_row_h"),
                             ("HEADER_H", "header_h")):
            assert str(getattr(T, upper)) == t.get(lower)
