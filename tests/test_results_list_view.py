"""Tests for 文件详情页（列表视图）的三个修复:

1. 列表视图没有勾选列 → 「编辑标签 / 重新识别人脸 / 删除」永远禁用
2. 「标签」列读的是 f["tags"]（该键不存在）→ 整列永远空白
3. stretchLastSection 让「状态」列吃掉剩余宽度 → 列宽不协调

外加: 演员列悬浮显示人脸封面。

Qt 交互测试需显式开启（与 onnxruntime 同进程会崩，见 test_dashboard_quick_filter）:
    VT_UI_TESTS=1 uv run --active python -m pytest tests/test_results_list_view.py
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
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget, QLabel
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QEnterEvent

from ui.components.results_page import (
    ResultsPage, _ActorHoverLabel, _load_face_pixmap, _make_actor_face_preview,
)


def _write_jpg(path: Path, size=24):
    """写一张真实 JPEG（Qt 无 JPEG 插件，必须用 OpenCV 编码）。"""
    import cv2
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # BGR → 红
    assert cv2.imwrite(str(path), img), "cv2 写图失败"
    return path


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _video(i, tags=None, actors=None, face_scan=None):
    return {
        "id": 1000 + i,
        "task_id": 2000 + i,
        "file_name": f"video_{i}.mp4",
        "file_path": f"/Volumes/MediaData/video_{i}.mp4",
        "file_size": 1024 * 1024 * (i + 1),
        "status": "READY",
        "mapped_tags": tags if tags is not None else ["亚洲人", "写实", f"标签{i}"],
        "actors": actors if actors is not None else [],
        "face_scan_status": face_scan,
    }


@pytest.fixture
def page(qapp, monkeypatch):
    # __init__ 会 _load_data() 打真实 API —— 测试里全部离线，只验证渲染/联动
    monkeypatch.setattr(ResultsPage, "_load_data", lambda self: None)
    p = ResultsPage()
    p._view_mode = "list"
    p._all_videos = [_video(i) for i in range(5)]
    yield p
    p.deleteLater()


class TestListViewHasCheckboxes:
    def test_every_row_has_checkbox(self, page):
        table = page._build_list_table(page._all_videos, track_selection=True)
        assert table.columnCount() == 6, "应是 勾选+文件名+大小+时长+标签+ID（19§D4.8b）"
        for row in range(table.rowCount()):
            holder = table.cellWidget(row, 0)
            assert holder is not None, f"第 {row} 行没有勾选单元格"
            assert holder.findChild(QCheckBox) is not None

    def test_checking_enables_action_buttons(self, page):
        table = page._build_list_table(page._all_videos, track_selection=True)
        assert page.btn_rebuild_faces.isEnabled() is False
        assert page.btn_batch_delete.isEnabled() is False

        cb = table.cellWidget(0, 0).findChild(QCheckBox)
        cb.setChecked(True)

        assert page.selected_videos == {1000}
        assert page.btn_rebuild_faces.isEnabled() is True
        assert page.btn_batch_delete.isEnabled() is True

    def test_unchecking_disables_again(self, page):
        table = page._build_list_table(page._all_videos, track_selection=True)
        cb = table.cellWidget(0, 0).findChild(QCheckBox)
        cb.setChecked(True)
        cb.setChecked(False)
        assert page.selected_videos == set()
        assert page.btn_rebuild_faces.isEnabled() is False

    def test_no_toolbar_edit_tags_button(self, page):
        """标签编辑已改为双击标签列（单文件）；工具栏按钮应已移除。"""
        assert not hasattr(page, "btn_edit_tags")

    def test_selected_ids_are_analysis_ids(self, page):
        """勾选存 FileAnalysis.id —— _rebuild_faces/_batch_delete 靠它映射 task_id。"""
        table = page._build_list_table(page._all_videos, track_selection=True)
        for row in range(3):
            table.cellWidget(row, 0).findChild(QCheckBox).setChecked(True)
        assert page.selected_videos == {1000, 1001, 1002}

    def test_toggle_select_all_button(self, page):
        table = page._build_list_table(page._all_videos, track_selection=True)
        page._toggle_select_all()
        assert page.selected_videos == {1000 + i for i in range(5)}
        assert page.btn_select_all.text() == "取消全选"
        for row in range(table.rowCount()):
            assert table.cellWidget(row, 0).findChild(QCheckBox).isChecked()

        page._toggle_select_all()
        assert page.selected_videos == set()
        assert page.btn_select_all.text() == "全选"

    def test_reset_selection_unchecks_rows(self, page):
        table = page._build_list_table(page._all_videos, track_selection=True)
        page._toggle_select_all()
        page._reset_selection()
        assert page.selected_videos == set()
        for row in range(table.rowCount()):
            assert not table.cellWidget(row, 0).findChild(QCheckBox).isChecked()

    def test_rebuild_preserves_checked_state_of_selected_ids(self, page):
        """重绘时按 selected_videos 回填勾选（分页/刷新后不丢失）。"""
        page.selected_videos = {1002}
        table = page._build_list_table(page._all_videos, track_selection=True)
        assert table.cellWidget(2, 0).findChild(QCheckBox).isChecked()
        assert not table.cellWidget(0, 0).findChild(QCheckBox).isChecked()


def _cell_text(table, row, col):
    """读取单元格文本：标签列已是 chips 控件，退回 item 文本兜底。"""
    w = table.cellWidget(row, col)
    if w is not None:
        if hasattr(w, "text"):
            return w.text()
        lbl = w.findChild(QLabel)
        if lbl is not None:
            return lbl.text()
    item = table.item(row, col)
    return item.text() if item is not None else ""


def _cell_tags(table, row, col):
    """读取标签 chips 单元格的全部 chip 文本（19§D4.8b chips）。"""
    w = table.cellWidget(row, col)
    if w is None:
        return []
    return [lbl.text() for lbl in w.findChildren(QLabel)]


class TestTagsColumn:
    TAGS_COL = 4  # 19§D4.8b：复选0/文件名1/大小2/时长3/标签4/ID5

    def test_tags_column_reads_mapped_tags(self, page):
        table = page._build_list_table(page._all_videos)
        chips = _cell_tags(table, 0, self.TAGS_COL)
        assert chips, "标签列不能为空（原先读不存在的 f['tags']）"
        assert "亚洲人" in chips and "写实" in chips

    def test_empty_tags_renders_empty_not_crash(self, page):
        table = page._build_list_table([_video(0, tags=[])])
        assert _cell_tags(table, 0, self.TAGS_COL) == []

    def test_all_tags_shown_not_truncated(self, page):
        """展示全部标签（chips 不截断；宫格卡才收 +N，列表全量）。"""
        vid = _video(0, tags=[f"t{i}" for i in range(20)])
        table = page._build_list_table([vid])
        chips = _cell_tags(table, 0, self.TAGS_COL)
        assert len(chips) == 20, "应展示全部 20 个标签"
        assert "t19" in chips

    def test_tags_cell_is_chips_not_text(self, page):
        """标签列用 chips 呈现（底 glass/字 txt2/圆角 4，11px）。"""
        import ui.components.results_page as RP
        table = page._build_list_table(page._all_videos)
        cell = table.cellWidget(0, self.TAGS_COL)
        assert isinstance(cell, RP._TagsChipsCell)
        chip = cell.findChild(QLabel)
        assert chip is not None
        assert "border-radius: 4px" in chip.styleSheet()
        assert "font-size: 11px" in chip.styleSheet()


class TestWrappingColumns:
    """标签列 chips 折行，且行高随内容增长（否则折行也被裁掉）。"""

    def test_tags_cell_uses_flow_layout(self, page):
        from ui.widgets.flow_layout import FlowLayout
        table = page._build_list_table(page._all_videos)
        cell = table.cellWidget(0, 4)
        assert isinstance(cell.layout(), FlowLayout)
        assert cell.layout().hasHeightForWidth(), "布局须支持按宽度算高度才能折行"

    def test_flow_layout_positions_children(self, page):
        """回归: FlowLayout 必须覆写 setGeometry。

        QLayout 是 C++ 类，不覆写 setGeometry 就不会调用 _do_layout，
        子控件将永远停在默认 640x480 → 表现为"演员列看不到文字"。
        """
        from ui.widgets.flow_layout import FlowLayout
        holder = QWidget()
        lay = FlowLayout(holder, margin=0, spacing=8)
        labels = [QLabel(f"演员{i}") for i in range(6)]
        for lbl in labels:
            lay.addWidget(lbl)

        holder.resize(120, 200)
        holder.show()
        QApplication.processEvents()

        assert labels[0].size().width() < 640, f"子控件未被布局: {labels[0].size()}"
        assert len({lbl.pos().y() for lbl in labels}) > 1, "窄容器下 6 个标签应折行"

    def test_row_height_grows_with_wrapped_content(self, page):
        many = _video(0, tags=[f"很长的标签{i}" for i in range(40)])
        few = _video(1, tags=["短"])
        table = page._build_list_table([many, few])
        table.resize(700, 400)
        table.fit_wrapped_rows()
        base = table.verticalHeader().defaultSectionSize()
        assert table.rowHeight(0) > table.rowHeight(1), "标签多的行应更高"
        assert table.rowHeight(0) > base

    def test_tags_cell_double_click_opens_single_file_editor(self, page, monkeypatch):
        """双击标签单元格直接编辑该文件 —— 无需先勾选。"""
        import ui.components.results_page as RP
        import ui.components.tag_edit_dialog as TED

        captured = {}

        class _FakeDialog:
            def __init__(self, analysis_id, tags=None, parent=None):
                captured["analysis_id"] = analysis_id
                captured["tags"] = tags
            def exec(self):
                return False

        monkeypatch.setattr(TED, "TagEditDialog", _FakeDialog)

        table = page._build_list_table(page._all_videos)
        cell = table.cellWidget(0, 4)
        assert isinstance(cell, RP._TagsChipsCell)
        # 未勾选任何行，仍能编辑
        assert page.selected_videos == set()

        cell.double_clicked.emit(1002)

        assert captured["analysis_id"] == 1002, "应只针对被双击的那一个文件"
        assert "标签2" in captured["tags"]

    def test_load_data_does_not_block_main_thread(self, qapp, monkeypatch):
        """回归: `_load_data` 必须异步，不得在主线程同步请求。

        原实现是主线程同步 GET /videos + 同步渲染，后端一忙（4 个并发分析
        任务）单次就 10 秒量级，期间整个 UI 冻结。
        """
        import time
        from ui.components.results_page import ResultsPage as RP

        # 只在本 context 内打桩，避免构造时真去请求；出 context 自动恢复真实现
        with monkeypatch.context() as m:
            m.setattr(RP, "_load_data", lambda self: None)
            page = RP()

        def _slow_get(path, **kwargs):
            time.sleep(1.0)
            return []

        class _SlowAPI:
            get = staticmethod(_slow_get)

        page.api = _SlowAPI()
        t0 = time.time()
        page._load_data()
        elapsed = time.time() - t0
        assert elapsed < 0.3, f"_load_data 阻塞了主线程 {elapsed:.2f}s（应异步加载）"

        # 加载中再调用应被重入保护挡住(不重复发起)
        page._load_data()

        # 等后台线程收尾,避免残留 QThread
        for _ in range(25):
            QApplication.processEvents()
            time.sleep(0.1)
        assert page._loading_data is False

    def test_toggle_view_mode_does_not_reload(self, page, monkeypatch):
        """回归: 切换宫格/列表只应重渲染，不应重新联网加载。

        原先 `_toggle_view_mode` 调 `_load_data()` —— 数据明明已缓存在
        `_all_videos` 里，却要重新拉全量、丢勾选、跳回第 1 页。
        """
        events = []
        monkeypatch.setattr(page, "_load_data", lambda *a, **k: events.append("load"))
        monkeypatch.setattr(page, "_render_current_page", lambda: events.append("render"))

        page._toggle_view_mode()

        assert events == ["render"], f"切换视图不应重新加载: {events}"


class TestSilentAutoRefresh:
    """10 秒自动刷新必须是「静默」的，否则表现为"频繁刷新、多次重新加载"。"""

    @staticmethod
    def _spy(page, monkeypatch):
        calls = []
        monkeypatch.setattr(page, "_render_current_page", lambda: calls.append(1))
        return calls

    def test_no_change_does_not_render(self, page, monkeypatch):
        calls = self._spy(page, monkeypatch)
        vids = [_video(i) for i in range(3)]
        page._all_videos = vids
        page._videos_sig = page._videos_signature(vids)
        page._silent_load = True

        page._on_videos_loaded(list(vids))      # 同一份数据

        assert calls == [], "数据没变时后台刷新不应重渲染（否则每 10 秒闪一次）"

    def test_no_change_keeps_selection_and_page(self, page, monkeypatch):
        self._spy(page, monkeypatch)
        vids = [_video(i) for i in range(3)]
        page._videos_sig = page._videos_signature(vids)
        page.selected_videos = {1001}
        page._current_page = 4
        page._silent_load = True

        page._on_videos_loaded(list(vids))

        assert page.selected_videos == {1001}, "后台刷新不得丢勾选"
        assert page._current_page == 4, "后台刷新不得跳页"

    def test_change_renders_but_keeps_selection_and_page(self, page, monkeypatch):
        calls = self._spy(page, monkeypatch)
        vids = [_video(i) for i in range(3)]
        page._videos_sig = page._videos_signature(vids)
        page.selected_videos = {1002}
        page._current_page = 3
        page._silent_load = True

        changed = [_video(0), _video(1), _video(2, tags=["换了标签"])]
        page._on_videos_loaded(changed)

        assert calls == [1], "数据变了应当重渲染"
        assert page.selected_videos == {1002}, "重渲染也不该丢勾选"
        assert page._current_page == 3, "重渲染不该跳回第 1 页"

    def test_manual_reload_resets_page(self, page, monkeypatch):
        self._spy(page, monkeypatch)
        page._current_page = 5
        page._silent_load = False

        page._on_videos_loaded([_video(0)])

        assert page._current_page == 1, "用户主动刷新才回到第 1 页"

    def test_silent_failure_keeps_ui(self, page, monkeypatch):
        """后台刷新失败不该把界面替换成错误页（用户没主动操作）。"""
        shown = []
        monkeypatch.setattr(page, "_show_load_error", lambda m: shown.append(m))

        page._silent_load = True
        page._on_load_failed("boom")

        assert shown == [], "后台静默刷新失败不应弹错误页"
        assert page._loading_data is False

    def test_signature_detects_tag_and_status_change(self, page):
        base = [_video(0)]
        sig = page._videos_signature(base)
        assert page._videos_signature([_video(0, tags=["别的"])]) != sig
        changed = [_video(0)]
        changed[0]["face_scan_status"] = "complete"
        assert page._videos_signature(changed) != sig

    def test_no_clipping_after_fit(self, page):
        """回归: 折行后行高必须真的容得下内容。

        两个坑（都会导致"换行了但最后一行被裁"）：
        1. 单元格控件比列窄 → 必须用控件实际宽度算高；
        2. 单元格控件比行高矮（差 9px）→ 必须在行高上补回这个内缩。
        """
        heavy = _video(0, tags=[f"很长的标签名称{i}" for i in range(25)])
        light = _video(1, tags=["短"])
        table = page._build_list_table([heavy, light])
        table.resize(1100, 500)
        table.show()
        QApplication.processEvents()
        table.fit_wrapped_rows()
        QApplication.processEvents()

        for row in range(2):
            cell = table.cellWidget(row, 4)
            need = cell.heightForWidth(cell.width())
            assert cell.height() >= need, (
                f"行{row} 标签列被裁: 控件高 {cell.height()} < 需要 {need}"
            )


class TestColumnWidths:
    def test_last_section_not_stretched(self, page):
        """ID 列不得吃掉剩余宽度（归标签列）。"""
        table = page._build_list_table(page._all_videos, stretch=True)
        assert table.horizontalHeader().stretchLastSection() is False

    def test_tags_column_stretches(self, page):
        from PySide6.QtWidgets import QHeaderView
        table = page._build_list_table(page._all_videos, stretch=True)
        mode = table.horizontalHeader().sectionResizeMode(4)
        assert mode == QHeaderView.ResizeMode.Stretch

    def test_name_column_stretches(self, page):
        from PySide6.QtWidgets import QHeaderView
        table = page._build_list_table(page._all_videos, stretch=True)
        mode = table.horizontalHeader().sectionResizeMode(1)
        assert mode == QHeaderView.ResizeMode.Stretch

    def test_numeric_columns_fit_contents(self, page):
        """大小/时长/ID 右列按内容取宽（19§D4.8b）。"""
        from PySide6.QtWidgets import QHeaderView
        table = page._build_list_table(page._all_videos, stretch=True)
        h = table.horizontalHeader()
        assert h.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents  # 大小
        assert h.sectionResizeMode(3) == QHeaderView.ResizeMode.ResizeToContents  # 时长
        assert h.sectionResizeMode(5) == QHeaderView.ResizeMode.ResizeToContents  # ID

    def test_checkbox_column_36(self, page):
        """复选列 36（19§D4.8b）。"""
        table = page._build_list_table(page._all_videos, stretch=True)
        assert table.columnWidth(0) == 36

    def test_header_height_32_row_28(self, page):
        """表头 32 / 行 28（19§D4.8b）。"""
        table = page._build_list_table(page._all_videos, stretch=True)
        assert table.horizontalHeader().height() == 32
        assert table.verticalHeader().defaultSectionSize() == 28


class TestListViewColumns:
    """列契约（19§D4.8b）：复选/文件名/大小/时长/标签/ID。"""

    def test_header_present(self, page):
        table = page._build_list_table(page._all_videos)
        labels = [table.horizontalHeaderItem(i).text() for i in range(6)]
        assert labels == ["", "文件名", "大小", "时长", "标签", "ID"]

    def test_size_and_duration_right_aligned(self, page):
        from PySide6.QtCore import Qt
        table = page._build_list_table([_video(0)])
        for col in (2, 3, 5):
            item = table.item(0, col)
            assert item is not None
            assert item.textAlignment() & Qt.AlignmentFlag.AlignRight, f"列{col} 应右对齐"

    def test_duration_formatted(self, page):
        vid = _video(0)
        vid["duration"] = 2538
        table = page._build_list_table([vid])
        assert table.item(0, 3).text() == "42:18"

    def test_duration_missing_shows_dash(self, page):
        vid = _video(0)
        vid["duration"] = 0
        table = page._build_list_table([vid])
        assert table.item(0, 3).text() == "—"

    def test_id_column_text(self, page):
        table = page._build_list_table([_video(0)])
        assert table.item(0, 5).text() == "1000"

    def test_face_scan_display_pure_function(self):
        """纯函数契约保留（宫格/人脸管理页共用口径）。"""
        from ui.components.results_page import face_scan_display
        assert face_scan_display("complete") == ("✓ 已识别", "success")
        assert face_scan_display("none") == ("🙅 无人脸", "text_secondary")
        assert face_scan_display("skipped") == ("⏭ 跳过", "warning")
        assert face_scan_display("low_confidence") == ("⚠ 低置信", "warning")
        assert face_scan_display(None) == ("—", "text_secondary")


class TestBandAndViews:
    """顶带四件套 sm 档（19§D1.2）与视图切换互斥组（19§D1.4）。"""

    def test_band_controls_are_26_high(self, page):
        # 未 show 的控件 height() 反映默认 sizeHint，钳制值读 min/max
        assert page.btn_refresh.minimumHeight() == 26
        assert page.search_box.minimumHeight() == 26
        assert page.btn_view_grid.minimumHeight() == 26
        assert page.btn_view_list.minimumHeight() == 26
        assert page.sort_combo.minimumHeight() == 26
        assert page.btn_sort_dir.minimumHeight() == 26

    def test_content_buttons_are_28_high(self, page):
        assert page.btn_select_all.minimumHeight() == 28
        assert page.btn_assign_project.minimumHeight() == 28

    def test_view_group_exclusive_default_grid(self, page):
        assert page.btn_view_grid.isChecked()
        assert not page.btn_view_list.isChecked()
        page._set_view_mode("list")
        assert page.btn_view_list.isChecked()
        assert not page.btn_view_grid.isChecked()
        page._set_view_mode("grid")
        assert page.btn_view_grid.isChecked()

    def test_no_view_toggle_single_button(self, page):
        """单钮「列表视图」已由双钮互斥组取代。"""
        assert not hasattr(page, "btn_view_toggle")


class TestPaginationCompact:
    """分页栏（18§4.17/19§D4.17）：无跳转控件、高 30。"""

    def test_compact_hides_jump_controls(self, qapp):
        from ui.widgets.video_grid_widget import PaginationWidget
        p = PaginationWidget(compact=True)
        assert p.first_btn.isHidden()
        assert p.last_btn.isHidden()
        assert p.page_jump_input.isHidden()
        assert p.minimumHeight() == 30

    def test_single_form_no_jump_controls_any_mode(self, qapp):
        """2026-09-29 用户裁定：全 app 唯一分页形态——任何模式下
        首页/末页/跳转都不出现（分页只管分页）。"""
        from ui.widgets.video_grid_widget import PaginationWidget
        p = PaginationWidget(compact=False)
        assert p.first_btn.isHidden()
        assert p.last_btn.isHidden()
        assert p.page_jump_input.isHidden()


class TestFileCardGrid:
    """宫格卡（19§D4.8a）：16:9 封面/时长角标/chips≤3 收 +N。"""

    def _card(self, qapp, video):
        from ui.components.results_page import FileCardWidget
        return FileCardWidget(video, card_width=232)

    def test_card_fixed_width(self, qapp):
        card = self._card(qapp, _video(0))
        assert card.width() == 232

    def test_cover_is_16_9(self, qapp):
        card = self._card(qapp, _video(0))
        # 封面高 = 卡宽 * 9 // 16（16:9 以卡宽为基准；容器宽=卡宽-2 留边框）
        assert card.cover_container.height() == (232 * 9) // 16

    def test_duration_badge(self, qapp):
        vid = _video(0)
        vid["duration"] = 2538
        card = self._card(qapp, vid)
        badges = [l for l in card.cover_container.findChildren(QLabel) if l.text() == "42:18"]
        assert badges, "时长角标缺失"

    def test_chips_max_three_with_overflow(self, qapp):
        vid = _video(0, tags=["a", "b", "c", "d", "e"])
        card = self._card(qapp, vid)
        texts = [l.text() for l in card.basic_info.findChildren(QLabel)]
        assert "a" in texts and "c" in texts
        assert "d" not in texts and "+2" in texts

    def test_chips_all_under_three(self, qapp):
        vid = _video(0, tags=["a", "b"])
        card = self._card(qapp, vid)
        texts = [l.text() for l in card.basic_info.findChildren(QLabel)]
        assert "a" in texts and "b" in texts and not any(t.startswith("+") for t in texts)

    def test_name_single_line_elided(self, qapp):
        vid = _video(0)
        vid["file_name"] = "特别长的文件名" * 20 + ".mp4"
        card = self._card(qapp, vid)
        name_lbl = card.basic_info.findChildren(QLabel)[0]
        assert "…" in name_lbl.text(), "超长文件名应省略号收尾"
        assert name_lbl.wordWrap() is False

    def test_no_inline_expansion(self, qapp):
        """内层详情已改为 D4.20 抽屉（P4 批），卡片不再原地展开。"""
        card = self._card(qapp, _video(0))
        assert not hasattr(card, "details_widget")
        assert not hasattr(card, "is_expanded")


class TestTreeDelegate:
    """树行委托（18§3.3-3.7）：行高 28 / 计数右对齐 / 详情页与任务页共用。"""

    def test_row_height_28(self, qapp):
        from ui.widgets.tree_row_delegate import TreeCountDelegate
        d = TreeCountDelegate()
        assert d.sizeHint(None, None).height() == 28

    def test_scope_tree_uses_delegate(self, page):
        from ui.widgets.tree_row_delegate import TreeCountDelegate
        assert isinstance(page.scope_tree.itemDelegate(), TreeCountDelegate)

    def test_scope_nodes_carry_count_data(self, page):
        page._projects_flat = [
            {"id": 1, "parent_id": None, "name": "婚礼", "asset_count": 3},
        ]
        page._on_scope_loaded(page._projects_flat)
        item = page.scope_tree.topLevelItem(0)
        assert item.text(0) == "全部素材"
        node = page.scope_tree.topLevelItem(2)
        assert node.text(0) == "婚礼"
        assert node.data(0, 256 + 1) == 3  # UserRole+1


class TestActorHoverPreview:
    @staticmethod
    def _enter():
        return QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1))

    def test_actor_cell_uses_hover_labels(self, page, tmp_path):
        """人物列已随 19§D4.8b 列契约移出列表——悬浮标签能力经
        _build_actor_cell 保留（宫格卡/后续视图复用）。"""
        face = _write_jpg(tmp_path / "ACTOR_001" / "face_0.jpg")
        actors = [{"id": 1, "code": "ACTOR_001", "name": "甲", "cover_face": str(face)},
                  {"id": 2, "code": "ACTOR_002", "name": "乙", "cover_face": None}]
        cell = page._build_actor_cell(actors)
        labels = cell.findChildren(_ActorHoverLabel)
        assert [l.text() for l in labels] == ["甲", "乙"], "每个演员单独一个标签"

    def test_hover_shows_and_leave_hides(self, page, tmp_path):
        face = _write_jpg(tmp_path / "f.jpg")
        label = _ActorHoverLabel("甲", str(face))
        label.show()
        assert label._preview is None

        label.enterEvent(self._enter())
        assert label._preview is not None, "悬浮应创建预览窗"
        assert label._preview.isVisible()
        shown = label._preview.pixmap()
        assert shown is not None and not shown.isNull(), "预览窗必须真的显示图"

        label.leaveEvent(QEvent(QEvent.Type.Leave))
        assert label._preview is None, "移开应关闭预览"

    def test_jpeg_decodes_via_opencv(self, page, tmp_path):
        """_load_face_pixmap 必须返回有效位图（Qt 无 JPEG 插件的环境靠
        OpenCV 解码；有插件的环境直接 QPixmap 也可——前提不再强断言）。"""
        face = _write_jpg(tmp_path / "f.jpg")
        pm = _load_face_pixmap(str(face))
        assert pm is not None and not pm.isNull()
        assert pm.width() > 0

    def test_no_face_no_preview(self, page, tmp_path):
        label = _ActorHoverLabel("乙", None)
        label.show()
        label.enterEvent(self._enter())
        assert label._preview is None

    def test_missing_face_file_no_preview(self, page, tmp_path):
        label = _ActorHoverLabel("丙", str(tmp_path / "nope.jpg"))
        label.show()
        label.enterEvent(self._enter())
        assert label._preview is None

    def test_helper_returns_none_for_bad_input(self, page):
        assert _make_actor_face_preview(None) is None
        assert _make_actor_face_preview("") is None
        assert _make_actor_face_preview("/no/such/file.jpg") is None


class TestRebuildFacesFromListView:
    """列表视图勾选后「重新识别人脸」必须真的发起重建。

    回归: 原先 _rebuild_faces 从宫格卡片(_get_all_cards)解析 task_id，
    列表视图下取不到任何卡片 → 静默不执行。
    """

    class _FakeAPI:
        def __init__(self):
            self.posts = []
        def post(self, path, json=None):
            self.posts.append(path)
            return {"status": "completed", "message": ""}
        def get(self, path):
            return {"status": "completed"}

    class _FakeMessageBox:
        # 只用到 Yes 常量与两个弹窗方法（确认框走 ConfirmDialog，另行 patch）
        Yes = "YES"
        @staticmethod
        def information(*a, **k):
            return None
        @staticmethod
        def warning(*a, **k):
            return None

    def test_selected_rows_resolve_task_ids_and_call_api(self, page, monkeypatch):
        import ui.components.results_page as RP

        monkeypatch.setattr(RP, "QMessageBox", self._FakeMessageBox)
        monkeypatch.setattr(
            RP.ConfirmDialog, "question",
            staticmethod(lambda *a, **k: RP.QMessageBox.Yes),
        )

        fake = self._FakeAPI()
        page.api = fake

        table = page._build_list_table(page._all_videos, track_selection=True)
        table.cellWidget(0, 0).findChild(QCheckBox).setChecked(True)
        table.cellWidget(2, 0).findChild(QCheckBox).setChecked(True)

        page._rebuild_faces()

        # _video(i) → id=1000+i, task_id=2000+i
        assert "/tasks/2000/rebuild-faces" in fake.posts
        assert "/tasks/2002/rebuild-faces" in fake.posts
