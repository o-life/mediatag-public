"""黄金截图回归——详情页三视图（宫格/列表/空态）。

基线在 tests/golden_baselines/*.png（入库=审批）。渲染漂移超阈值即红，
热图落 test-artifacts/golden_diff/。有意改版后：
    UI_GOLDEN_UPDATE=1 VT_UI_TESTS=1 pytest tests/test_golden_screenshots.py
新基线肉眼过目后随改动一起提交。
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
from PySide6.QtWidgets import QApplication

from ui.control.golden import assert_golden

WIDTH, HEIGHT = 1200, 800

VIDEOS = [
    {"id": i, "file_name": f"婚礼-正片-{i:02d}.mp4",
     "mapped_tags": ["婚礼", "室内", "双人", "花絮", "外景"][: 1 + i % 5],
     "duration": 600 + i * 137, "file_size": 1024 * 1024 * (17 + i),
     "actors": []}
    for i in range(6)
]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _pin_dark_theme():
    """黄金基线=深色主题生成。主题态存 QSettings（机器级共享）——
    开发者手动切浅色后运行测试，全部基线都会「漂移」（实测 7 连挂）。
    本模块固定深色，退出还原。"""
    from ui.components.theme import theme
    mgr = theme()
    saved = mgr.mode
    mgr.mode = "dark"
    yield
    mgr.mode = saved


def _page(qapp):
    from unittest.mock import patch

    from ui.components.results_page import ResultsPage

    with patch.object(ResultsPage, "_load_data", lambda self: None), \
         patch.object(ResultsPage, "_refresh_scope_tree", lambda self: None):
        page = ResultsPage()
    page._all_videos = [dict(v) for v in VIDEOS]
    page._data_loaded = True
    page.resize(WIDTH, HEIGHT)
    page.show()
    # 布局稳定循环：合并跑时（前面模块有真主窗口夹具）首轮 processEvents
    # 不足以让顶层 resize+子布局落位——迭代到几何稳定为止，防挤压渲染
    for _ in range(10):
        QApplication.processEvents()
        if page.width() == WIDTH and page.height() == HEIGHT:
            break
        page.resize(WIDTH, HEIGHT)
    QApplication.processEvents()
    assert page.width() == WIDTH and page.height() == HEIGHT, (
        f"页面几何未落位: {page.width()}x{page.height()}")
    # 视口落位后按最终宽度重渲染一次——首渲染常按中间视口（~1000px）
    # 算列数，挤压入基线（7 连挂根因）
    page._render_current_page()
    QApplication.processEvents()
    return page


def test_golden_detail_grid(qapp):
    page = _page(qapp)
    page._set_view_mode("grid")
    QApplication.processEvents()
    assert_golden(page, "detail_grid")
    page.deleteLater()


def test_golden_detail_list(qapp):
    page = _page(qapp)
    page._set_view_mode("list")
    QApplication.processEvents()
    assert_golden(page, "detail_list")
    page.deleteLater()


def test_golden_detail_empty(qapp):
    page = _page(qapp)
    page._all_videos = []
    page._data_loaded = True
    page._render_current_page()
    QApplication.processEvents()
    assert_golden(page, "detail_empty")
    page.deleteLater()


# ══ 批③：标签管理页 ═════════════════════════════════════════════

AUDIT_ITEMS = [
    {"id": "1", "raw_label": "4k超清", "count": 12, "ai_recommendation": "画质"},
    {"id": "2", "raw_label": "室内婚礼", "count": 8, "ai_recommendation": "婚礼"},
    {"id": "3", "raw_label": "iPhone 15 Pro", "count": 3, "ai_recommendation": "无匹配"},
]


def _label_page(qapp):
    from unittest.mock import patch

    from ui.components.label_mgmt import LabelManagementPage

    with patch.object(LabelManagementPage, "_load_themes", lambda self: None), \
         patch.object(LabelManagementPage, "_load_dictionary", lambda self: None), \
         patch.object(LabelManagementPage, "_load_audit_queue", lambda self: None):
        page = LabelManagementPage()
    page.resize(WIDTH, HEIGHT)
    page.show()
    QApplication.processEvents()
    return page


def test_golden_label_audit_queue(qapp):
    page = _label_page(qapp)
    page._on_dictionary_loaded({
        "categories": ["场景", "画质"],
        "tags": [
            {"name": "室内", "category": "场景", "aliases": ["内景"], "file_count": 18},
            {"name": "婚礼", "category": "场景", "aliases": [], "file_count": 22},
            {"name": "4k超清", "category": "画质", "aliases": [], "file_count": 9},
        ],
    })
    page._on_audit_queue_loaded({"items": [dict(i) for i in AUDIT_ITEMS]})
    QApplication.processEvents()
    assert_golden(page, "label_audit_queue")
    page.deleteLater()


def test_golden_label_empty_queue(qapp):
    page = _label_page(qapp)
    page._on_audit_queue_loaded({"items": []})
    QApplication.processEvents()
    assert_golden(page, "label_empty_queue")
    page.deleteLater()


# ══ 批④：标签浏览页 ═════════════════════════════════════════════

TAG_VIDEOS = [
    {"id": i, "title": f"婚礼-片段-{i:02d}.mp4",
     "file_path": f"/data/婚礼-片段-{i:02d}.mp4",
     "duration": 480 + i * 91, "frame_number": 100 + i,
     "video_frame_number": 100 + i, "cover_path": ""}
    for i in range(4)
]


def _browse_page(qapp):
    from unittest.mock import patch

    from ui.pages.tag_browse_page import TagBrowsePage

    with patch.object(TagBrowsePage, "initialize", lambda self: None):
        page = TagBrowsePage()
    page.resize(WIDTH, HEIGHT)
    page.show()
    QApplication.processEvents()
    return page


def test_golden_browse_guide(qapp):
    """未选标签：空态引导（TB4.4）。"""
    page = _browse_page(qapp)
    page._on_all_selected()
    QApplication.processEvents()
    assert_golden(page, "browse_guide")
    page.deleteLater()


def test_golden_browse_tag_selected(qapp):
    """选中标签：chip+网格卡片+起播行（TB1.1/TB4.2）。"""
    page = _browse_page(qapp)
    page.tag_tree.set_tags({"场景": [("婚礼", 4), ("室内", 2)]})
    page._on_tag_selected("婚礼")
    # 桩数据直塞组件（不走 API）
    page.video_grid.set_videos([dict(v) for v in TAG_VIDEOS], total_count=4)
    QApplication.processEvents()
    assert_golden(page, "browse_tag_selected")
    page.deleteLater()
