"""位置契约清单——文件详情页（19 号规格）的机器验收。

契约=设计稿里「A 与 B 的关系」。本文件只写 MediaTag 自己的契约；
断言 API 在 ui/control/geometry.py（通用层，可移植）。
覆盖当年咬过人的位置类缺陷：操作行贴面板（D4.1）、面板满铺（D4.8）、
分页入底栏（D4.17）、树列零缝（D7）、操作行统一间距（18§4.1）。
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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton, QLabel

from ui.control.geometry import Geo

try:
    from ui.tokens import T
except ImportError:
    from ui.tokens import T


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def detail_page(qapp, monkeypatch):
    """构造并显示详情页（数据打桩，几何真实）。"""
    from unittest.mock import patch

    from ui.components.results_page import ResultsPage

    with patch.object(ResultsPage, "_load_data", lambda self: None), \
         patch.object(ResultsPage, "_refresh_scope_tree", lambda self: None):
        page = ResultsPage()
    page._all_videos = [
        {"id": i, "file_name": f"video_{i}.mp4", "mapped_tags": ["婚礼", "室内"],
         "duration": 60 + i, "file_size": 1024 * 1024, "actors": []}
        for i in range(4)
    ]
    page._data_loaded = True
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    yield page
    page.deleteLater()


class TestDetailPositionContracts:
    def test_action_bar_sits_on_media_panel(self, detail_page):
        """19§D4.1：操作行贴素材面板顶，间隙 0（D7 同源缺陷的契约化）。"""
        bar = Geo(detail_page.action_bar, "操作行")
        panel = Geo(detail_page.media_panel, "素材面板")
        bar.bottom_adjacent(panel.top, gap=0, spec="19§D4.1 贴面板")

    def test_media_panel_full_bleed(self, detail_page):
        """19§D4.8：素材面板直角满铺——四缘贴页面（坐标同为页面系）。"""
        page = detail_page
        panel = Geo(page.media_panel, "素材面板")
        # 页面内容列满铺（外层 margins 0）→ 面板四缘=页面四缘
        assert panel.left == 0, f"面板左缘 {panel.left} ≠ 0（应贴树列分隔线）"
        assert panel.right == page.width() - 1, (
            f"面板右缘 {panel.right} ≠ 页面右缘 {page.width() - 1}")
        assert panel.bottom == page.height() - 1, (
            f"面板底缘 {panel.bottom} ≠ 页面底缘 {page.height() - 1}")

    def test_pagination_inside_panel_bottom(self, detail_page):
        """19§D4.17：分页栏在素材面板内部底端（非独立行），高 30。"""
        page = detail_page
        pager = page.pagination
        ancestor = pager.parentWidget()
        while ancestor is not None and ancestor is not page.media_panel:
            ancestor = ancestor.parentWidget()
        assert ancestor is page.media_panel, "分页栏必须在素材面板内部（D4.17）"
        pg = Geo(pager, "分页栏")
        assert pg.height == T.PAGER_H, f"分页栏高 {pg.height} ≠ {T.PAGER_H}"

    def test_tree_title_adjacent_tree(self, detail_page):
        """D7：树列标题行与树零缝（布局 spacing 0 的契约化）。"""
        lay = detail_page.project_panel.layout()
        assert lay.spacing() == 0, "树列面板 spacing 必须为 0（D7 定案）"

    def test_tree_panel_wraps_8(self, detail_page):
        """18§3.8：树列左右呼吸 8。"""
        lay = detail_page.project_panel.layout()
        m = lay.contentsMargins()
        assert (m.left(), m.right()) == (T.TREE_WRAP_X, T.TREE_WRAP_X)

    def test_action_row_uniform_spacing(self, detail_page):
        """18§4.1：操作行流式布局统一间距 8，无分组大间距。"""
        from ui.widgets.flow_layout import FlowLayout
        lay = detail_page._bar_inner.layout()
        assert isinstance(lay, FlowLayout)
        assert lay.spacing() == T.SP_8, f"操作行间距 {lay.spacing()} ≠ {T.SP_8}"

    def test_grid_cards_min_width(self, detail_page):
        """19§D4.8a：宫格卡宽≥190（auto-fill 下限）。"""
        page = detail_page
        page._set_view_mode("grid")
        QApplication.processEvents()
        assert page._page_cards, "宫格卡未渲染"
        for card in page._page_cards:
            assert card.width() >= T.GRID_CARD_MIN_W, (
                f"卡宽 {card.width()} < {T.GRID_CARD_MIN_W}")


# ══ 批③：标签管理页（20 号规格）位置契约 ══════════════════════════


@pytest.fixture()
def label_page(qapp, monkeypatch):
    from unittest.mock import patch

    from ui.components.label_mgmt import LabelManagementPage

    with patch.object(LabelManagementPage, "_load_themes", lambda self: None), \
         patch.object(LabelManagementPage, "_load_dictionary", lambda self: None), \
         patch.object(LabelManagementPage, "_load_audit_queue", lambda self: None):
        page = LabelManagementPage()
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    yield page
    page.deleteLater()


class TestLabelMgmtPositionContracts:
    def test_dict_panel_tree_two_level_indent(self, label_page):
        """LM3.3：标签库两级树——子级缩进 26（与详情/任务树一致）。"""
        assert label_page.dictionary_list.indentation() == T.TREE_INDENT_PROJECT

    def test_dict_tree_uses_shared_delegate(self, label_page):
        """LM3.3：行规格走 TreeCountDelegate 单点（树观感全 app 一致）。"""
        from ui.widgets.tree_row_delegate import TreeCountDelegate
        assert isinstance(label_page.dictionary_list.itemDelegate(),
                          TreeCountDelegate)

    def test_search_box_sm_height(self, label_page):
        """LM3.2：侧栏搜索框 26 高。"""
        assert label_page.search_box.minimumHeight() == T.CTRL_SM_H

    def test_audit_bar_sm_buttons(self, label_page):
        """LM1.6：顶带审核条操作钮 26 sm 档。"""
        for w in (label_page.batch_confirm_btn, label_page.batch_reject_btn,
                  label_page.batch_delete_btn, label_page.audit_refresh_btn):
            assert w.minimumHeight() == T.CTRL_SM_H, type(w).__name__

    def test_queue_header_32(self, label_page):
        """LM4.1：队列面板头行 32（min=max=32 的 QLabel）。"""
        from PySide6.QtWidgets import QLabel
        heads = [c for c in label_page.findChildren(QLabel)
                 if c.minimumHeight() == T.HEADER_H
                 and c.maximumHeight() == T.HEADER_H]
        assert heads, "队列头行缺失或高度≠32"

    def test_audit_card_gap_8(self, label_page):
        """LM4.2：审核卡间距 8。"""
        assert label_page.audit_flow.spacing() == T.SP_8


# ══ 批④：标签浏览页（21 号规格）位置契约 ══════════════════════════


@pytest.fixture()
def browse_page(qapp, monkeypatch):
    from unittest.mock import patch

    from ui.pages.tag_browse_page import TagBrowsePage

    with patch.object(TagBrowsePage, "initialize", lambda self: None):
        page = TagBrowsePage()
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    yield page
    page.deleteLater()


class TestTagBrowsePositionContracts:
    def test_tree_indent_26_and_delegate(self, browse_page):
        """TB3.3：标签树缩进 26、共享委托（树观感全 app 一致）。"""
        from ui.widgets.tree_row_delegate import TreeCountDelegate
        tree = browse_page.tag_tree.tree
        assert tree.indentation() == T.TREE_INDENT_PROJECT
        assert isinstance(tree.itemDelegate(), TreeCountDelegate)

    def test_tag_row_count_role(self, browse_page):
        """TB3.3：标签行计数=文件数（UserRole+1 委托角色）。"""
        browse_page.tag_tree.set_tags({"场景": [("婚礼", 22)]})
        tree = browse_page.tag_tree.tree
        cat = tree.topLevelItem(1)
        assert cat.text(0) == "场景"
        assert cat.child(0).data(0, Qt.ItemDataRole.UserRole + 1) == 22

    def test_toggle_deselect_returns_guide(self, browse_page):
        """TB3.4：再点已选标签=取消过滤回空态引导。"""
        page = browse_page
        page._on_tag_selected("婚礼")
        assert not page.tag_chip.isHidden()
        page._on_tag_selected("婚礼")  # 再点
        assert page.tag_chip.isHidden()
        assert page._empty_overlay.isVisible()
        assert not page.btn_modify_tag.isEnabled()

    def test_band_controls_sm(self, browse_page):
        """TB1.2/1.3：顶带三钮 26 sm 档。"""
        for w in (browse_page.btn_modify_tag, browse_page.btn_delete_tag,
                  browse_page.btn_refresh):
            assert w.minimumHeight() == T.CTRL_SM_H



# ══ 批⑤：人脸管理页（22 号规格）位置契约 ══════════════════════════


@pytest.fixture()
def face_page(qapp, monkeypatch):
    from unittest.mock import patch

    from ui.components.actor_mgmt import ActorManagementPage
    with patch.object(ActorManagementPage, "_load_actors",
                      lambda self: None, create=True):
        page = ActorManagementPage(api_base="http://localhost:8000")
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    yield page
    page.deleteLater()


class TestFacePageContracts:
    def test_face_bar_buttons_28_and_gap_12(self, face_page):
        """FM1.1：七钮一档 28、每钮间距 12。"""
        bar = face_page._face_bar
        lay = bar.layout()
        assert lay.spacing() == T.SP_12
        for btn in face_page._face_bar.findChildren(QPushButton):
            assert btn.minimumHeight() == T.CTRL_H, btn.text()

    def test_card_checkbox_floats_top_right(self, face_page):
        """FM4.2：勾选框悬浮封面右上（不占行）。"""
        from ui.components.actor_mgmt import ActorCardWidget
        card = ActorCardWidget({
            "id": 1, "code": "ACTOR_001", "name": "甲",
            "cover_face": "", "source_count": 3, "face_count": 5,
        })
        cb = card.checkbox
        assert cb.parentWidget() is card.cover_container, "勾选框应悬浮于封面容器"
        assert cb.geometry().y() == 8, "勾选框应贴封面顶部"
        assert cb.geometry().x() + cb.width() > card.cover_container.width() - 30, (
            "勾选框应贴封面右侧")

    def test_card_meta_copy(self, face_page):
        """FM4.3/4.4：编号「人物 N」+「出现于 N 个文件」+「N 张脸」角标。"""
        from ui.components.actor_mgmt import ActorCardWidget
        card = ActorCardWidget({
            "id": 1, "code": "ACTOR_001", "name": "",
            "cover_face": "", "source_count": 3, "face_count": 5,
        })
        labels = [l.text() for l in card.findChildren(QLabel)]
        assert "人物 001" in labels, "编号应显示「人物 N」（保留原编号格式）"
        assert "出现于 3 个文件" in labels
        assert "未命名" in labels, "无名称应显示「未命名」"
        assert any("5 张脸" in t for t in labels), "封面角标「N 张脸」"

    def test_page_full_bleed(self, face_page):
        """FM-4：面板满铺贴边（根边距 0）。"""
        lay = face_page.layout()
        m = lay.contentsMargins()
        assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)

    def test_hero_empty_with_go_tasks(self, face_page, qapp):
        """FM4.6：空态 hero+「去导入任务」单钮。"""
        page = face_page
        page.actor_grid.actors = []
        page.actor_grid.show_empty("还没有人物数据", hero=True,
                                   button_text="去导入任务")
        qapp.processEvents()
        assert page.actor_grid.empty_label.isVisible()
        assert page.actor_grid.go_tasks_btn is not None
        assert page.actor_grid.go_tasks_btn.text() == "去导入任务"
