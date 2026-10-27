"""人脸管理勾选集合的自动清理：演员被删除后其 id 应从跨页持久勾选中移除。

背景：`_ACTOR_SELECTION` 是进程级持久集合（跨页保留），原先只增不减——
演员被删除（本页删除 / 后端识别自动合并 / 重建）后 id 残留成"幽灵勾选"，
之后手动合并会因该 id 不存在而被后端整体 404。
修法：每次列表加载成功后按 `GET /actors/ids` 返回的全量有效 id 收敛。

Qt 交互测试需显式开启（与 onnxruntime 同进程会崩）:
    VT_UI_TESTS=1 uv run --active python -m pytest tests/test_actor_selection_prune.py
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

from ui.components.actor_grid import ActorGridWidget, _ACTOR_SELECTION


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clean_selection():
    _ACTOR_SELECTION.clear()
    yield
    _ACTOR_SELECTION.clear()


class TestPruneSelection:
    def test_drops_deleted_ids_keeps_valid(self, qapp):
        grid = ActorGridWidget()
        _ACTOR_SELECTION.update({1, 2, 3, 999})
        grid.prune_selection([1, 2, 3, 4, 5])
        assert _ACTOR_SELECTION == {1, 2, 3}, "已不存在的 id 应被剔除，有效 id 保留"

    def test_emits_selection_changed_when_pruned(self, qapp):
        grid = ActorGridWidget()
        seen = []
        grid.selection_changed.connect(lambda ids: seen.append(set(ids)))
        _ACTOR_SELECTION.update({1, 999})
        grid.prune_selection([1])
        assert seen and seen[-1] == {1}, "剔除后应通知计数/按钮刷新"

    def test_empty_valid_list_clears_all(self, qapp):
        grid = ActorGridWidget()
        _ACTOR_SELECTION.update({1, 2})
        grid.prune_selection([])
        assert _ACTOR_SELECTION == set(), "服务端无任何有效 id 时勾选应全部失效"

    def test_no_signal_when_nothing_pruned(self, qapp):
        grid = ActorGridWidget()
        seen = []
        grid.selection_changed.connect(lambda ids: seen.append(set(ids)))
        _ACTOR_SELECTION.update({1, 2})
        grid.prune_selection([1, 2, 3])
        assert seen == [], "没有失效 id 时不应触发刷新（避免多余重绘）"
