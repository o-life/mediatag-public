"""PaginationWidget 单测——离屏,覆盖首页/末页恢复(2026-10-07)与不折行。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from src.ui.widgets.video_grid_widget import PaginationWidget  # noqa: E402


def make_widget(total_pages=256, page=1):
    w = PaginationWidget(default_page_size=24)
    w.update_pagination(current_page=page, total_count=total_pages * 24,
                        page_size=24)
    return w


def test_first_last_buttons_visible():
    w = make_widget()
    assert w.first_btn.isVisible() is False  # 未 show 前用 not hidden 判定
    assert not w.first_btn.isHidden() and not w.last_btn.isHidden()
    assert not w.jump_label.isHidden() and not w.page_jump_input.isHidden()  # 跳转已恢复


def test_first_last_enabled_by_page_state():
    w = make_widget(total_pages=256, page=1)
    assert not w.first_btn.isEnabled()   # 第 1 页:首页禁用
    assert w.last_btn.isEnabled()
    w.update_pagination(current_page=256, total_count=256 * 24, page_size=24)
    assert w.first_btn.isEnabled()
    assert not w.last_btn.isEnabled()    # 末页:末页禁用


def test_first_last_click_navigates():
    w = make_widget(total_pages=256, page=5)
    seen = []
    w.page_changed.connect(lambda p: seen.append(p))
    w.last_btn.click()
    assert seen and seen[-1] == 256
    w.first_btn.click()
    assert seen[-1] == 1


def test_single_row_no_wrap_at_min_width():
    """最小内容宽 720 下分页栏不折行:bar 高恒为固定值(30+边框),且
    首页/末页按钮几何在栏内。"""
    from PySide6.QtWidgets import QWidget, QVBoxLayout
    host = QWidget()
    host.resize(720, 200)
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    w = make_widget(total_pages=256, page=3)
    lay.addWidget(w)
    host.show()
    app.processEvents()
    bar_h = w.height()
    assert bar_h <= 40, f"分页栏高度 {bar_h} 异常(疑似折行)"
    assert w.first_btn.isVisible() and w.last_btn.isVisible()
    assert (w.first_btn.y() == w.prev_btn.y()
            and w.last_btn.y() == w.next_btn.y()
            and w.page_jump_input.y() == w.next_btn.y()), "控件不在同一行=折行"
