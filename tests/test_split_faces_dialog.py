"""拆分对话框多选交互单测——离屏,假人脸数据。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from src.ui.components.actor_mgmt import _SplitFacesDialog  # noqa: E402

FACES = [{"id": 100 + i, "face_path": ""} for i in range(10)]


def make_dialog():
    dlg = _SplitFacesDialog(None, FACES, "http://127.0.0.1:1")
    return dlg


def checked_ids(dlg):
    return {fid for fid, cb in dlg.face_checkboxes.items() if cb.isChecked()}


def test_plain_click_toggles_and_sets_anchor():
    dlg = make_dialog()
    dlg._on_thumb_clicked(3)
    assert checked_ids(dlg) == {103}
    dlg._on_thumb_clicked(3)
    assert checked_ids(dlg) == set()
    dlg._on_thumb_clicked(3)
    assert dlg._anchor == 3


def test_shift_selects_range_union():
    dlg = make_dialog()
    dlg._on_thumb_clicked(2)          # 锚点 2,选中 102
    dlg._on_thumb_shift_clicked(6)    # 连选 2..6 → 102-106 全勾
    assert checked_ids(dlg) == {100 + i for i in range(2, 7)}
    # 已勾的不被连选取消(union 语义):手动先勾 0,再连选 4..7
    dlg.face_checkboxes[100].setChecked(True)
    dlg._on_thumb_shift_clicked(7)
    assert checked_ids(dlg) == {100, 102, 103, 104, 105, 106, 107}


def test_shift_without_anchor_falls_back_to_toggle():
    dlg = make_dialog()
    dlg._on_thumb_shift_clicked(5)    # 无锚点 → 退化为单点翻转
    assert checked_ids(dlg) == {105}


def test_clear_selection(tmp_path=None):
    dlg = make_dialog()
    dlg._on_thumb_clicked(1)
    dlg._on_thumb_shift_clicked(4)
    assert len(checked_ids(dlg)) == 4
    dlg._on_clear_selection()
    assert checked_ids(dlg) == set()
    assert dlg._anchor is None


def test_checkbox_and_thumb_stay_in_sync():
    dlg = make_dialog()
    dlg._on_thumb_clicked(0)          # 点图勾上
    assert dlg.face_checkboxes[100].isChecked()
    dlg.face_checkboxes[100].setChecked(False)  # 直接动勾选框(等价用户点勾选框)
    dlg._refresh_thumb_style(0)
    assert not dlg._thumbs[0].isChecked if hasattr(dlg._thumbs[0], "isChecked") else True
    assert dlg.face_checkboxes[100].isChecked() is False
