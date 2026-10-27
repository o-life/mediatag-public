"""Tests for 单文件标签编辑器（TagEditDialog）。

设计取舍（已在 UI 中确认）：只保留"整表编辑"一种操作。
- 不做"添加标签"：标签需定位到对应帧（tag_frame_mappings），
  单独追加一段文字没有意义。
- 不做"替换标签"：同理。
- 删除 = 在编辑框里删掉那一行。

历史回归: 该对话框曾调用三个不存在的后端接口（GET /videos/{id} → 405、
GET /tags → 404、POST /videos/batch-tags → 405），导致"勾选后点编辑标签
→ 提示（无标签）→ 无法编辑"。现在标签由调用方传入，保存走
POST /api/videos/batch-tags（整表覆盖）+ analysis_ids。

Qt 交互测试需显式开启:
    VT_UI_TESTS=1 uv run --active python -m pytest tests/test_tag_edit_dialog.py
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

import ui.components.tag_edit_dialog as TED
from ui.components.tag_edit_dialog import TagEditDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeAPI:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        _FakeAPI.instances.append(self)

    def get(self, path, **kwargs):
        self.calls.append(("get", path))
        raise AssertionError(f"编辑器不应再拉取接口: {path}")

    def post(self, path, json=None, **kwargs):
        self.calls.append(("post", path, json))
        return {"updated_count": 1, "failed_ids": []}


@pytest.fixture
def fake_api(qapp, monkeypatch):
    monkeypatch.setattr(TED, "APIClient", _FakeAPI)
    _FakeAPI.instances = []
    yield _FakeAPI


@pytest.fixture
def no_popup(monkeypatch):
    calls = {}
    monkeypatch.setattr(TED.QMessageBox, "information",
                        lambda *a, **k: calls.setdefault("info", a))
    monkeypatch.setattr(TED.QMessageBox, "warning",
                        lambda *a, **k: calls.setdefault("warning", a))
    return calls


def _dialog(analysis_id=1000, tags=None):
    return TagEditDialog(analysis_id, tags)


class TestEditorShape:
    def test_no_tab_widget(self, fake_api):
        """已合并为单一编辑界面（去掉 添加/删除/替换 三个页）。"""
        d = _dialog()
        assert not hasattr(d, "tab_widget")

    def test_removed_features_absent(self, fake_api):
        """添加/替换/建议勾选等已移除，不应留下残骸属性。"""
        d = _dialog()
        for attr in ("_add_tags", "_remove_tags", "_replace_tags",
                     "_create_add_tab", "_create_replace_tab",
                     "add_checkboxes", "replace_old_input",
                     "_load_tags", "_populate_tag_suggestions"):
            assert not hasattr(d, attr), f"应已移除: {attr}"

    def test_no_api_get_on_open(self, fake_api):
        """打开对话框不应请求任何 GET（历史 405/404 的根因）。"""
        _dialog()
        assert _FakeAPI.instances[0].calls == []


class TestPrelfill:
    def test_prefilled_one_per_line(self, fake_api):
        d = _dialog(tags=["甲", "乙", "丙"])
        assert d.tags_input.toPlainText() == "甲\n乙\n丙"

    def test_empty_tags(self, fake_api):
        d = _dialog(tags=[])
        assert d.tags_input.toPlainText() == ""

    def test_none_tags_tolerated(self, fake_api):
        d = _dialog(tags=None)
        assert d.tags_input.toPlainText() == ""


class TestSave:
    def test_posts_single_id_and_tags(self, fake_api, no_popup):
        d = _dialog(analysis_id=1010, tags=["旧"])
        d.tags_input.setPlainText("新甲\n新乙")
        d._save()

        _, path, payload = _FakeAPI.instances[0].calls[-1]
        assert path == "/api/videos/batch-tags"
        assert payload["analysis_ids"] == [1010]
        assert payload["tags"] == ["新甲", "新乙"]
        # 接口只有整表覆盖一种语义，不再有 action 判别字段
        assert "action" not in payload
        assert "info" in no_popup

    def test_strips_blank_and_dedupes(self, fake_api, no_popup):
        d = _dialog(tags=[])
        d.tags_input.setPlainText("甲\n\n  甲  \n乙\n   ")
        d._save()

        _, _, payload = _FakeAPI.instances[0].calls[-1]
        assert payload["tags"] == ["甲", "乙"]

    def test_can_clear_all(self, fake_api, no_popup):
        """删除全部标签 = 清空列表（删除能力内建于编辑）。"""
        d = _dialog(tags=["甲", "乙"])
        d.tags_input.setPlainText("")
        d._save()

        _, _, payload = _FakeAPI.instances[0].calls[-1]
        assert payload["tags"] == []

    def test_no_api_warns_and_does_not_post(self, fake_api, no_popup):
        d = _dialog(tags=["甲"])
        d.api = None
        d._save()
        assert "warning" in no_popup
        assert not [c for c in _FakeAPI.instances[0].calls if c[0] == "post"]
