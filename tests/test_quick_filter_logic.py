"""Tests for 统计条快捷筛选的**决策逻辑**（纯函数，不依赖 Qt）。

背景: 点统计卡片「待处理」应等价于"筛选下拉只勾待处理"，两者的条件必须一致。
这里覆盖三个纯函数，避免依赖 Qt(与 onnxruntime 同进程初始化 Qt 会崩)。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.components.dashboard import (  # noqa: E402
    STATUS_FILTER_ITEMS,
    active_quick_filter_key,
    normalize_status_filter,
    quick_filter_values,
)

ALL_KEYS = [v for v, _ in STATUS_FILTER_ITEMS]


class TestQuickFilterValues:
    def test_pending_card_maps_to_pending_only(self):
        assert quick_filter_values("pending") == ["pending"]

    def test_completed_card_maps_to_complete_only(self):
        assert quick_filter_values("complete") == ["complete"]

    def test_processing_card_maps_to_processing_only(self):
        assert quick_filter_values("processing") == ["processing"]

    def test_total_card_maps_to_all_keys(self):
        """「总任务」= 全勾 = 不过滤。"""
        assert set(quick_filter_values("all")) == set(ALL_KEYS)
        assert quick_filter_values("all") == ALL_KEYS

    def test_empty_or_unknown_falls_back_to_all(self):
        for key in ("", "bogus", "not_a_status"):
            assert set(quick_filter_values(key)) == set(ALL_KEYS)

    def test_other_real_statuses_still_work(self):
        """下拉里存在但统计条没有的状态(错误/重复/未启动)也可用。"""
        assert quick_filter_values("error") == ["error"]
        assert quick_filter_values("duplicate") == ["duplicate"]
        assert quick_filter_values("not_started") == ["not_started"]


class TestNormalizeStatusFilter:
    def test_empty_means_no_filter(self):
        assert normalize_status_filter([]) == []
        assert normalize_status_filter(None) == []

    def test_all_selected_means_no_filter(self):
        """全勾归一到"不过滤"，与点「总任务」同一结果。"""
        assert normalize_status_filter(ALL_KEYS) == []

    def test_single_and_multi_pass_through(self):
        assert normalize_status_filter(["pending"]) == ["pending"]
        assert normalize_status_filter(["pending", "processing"]) == ["pending", "processing"]

    def test_unknown_values_dropped(self):
        assert normalize_status_filter(["pending", "nope"]) == ["pending"]
        assert normalize_status_filter(["nope"]) == []

    def test_order_preserved(self):
        assert normalize_status_filter(["complete", "pending"]) == ["complete", "pending"]


class TestActiveQuickFilterKey:
    def test_none_filter_highlights_total(self):
        assert active_quick_filter_key("all") == "all"
        assert active_quick_filter_key("") == "all"

    def test_single_filter_highlights_that_card(self):
        assert active_quick_filter_key("pending") == "pending"
        assert active_quick_filter_key("complete") == "complete"

    def test_multi_filter_highlights_nothing(self):
        assert active_quick_filter_key("pending,processing") == ""


class TestConsistencyWithCards:
    """统计条的 4 张卡片都能映射到下拉里真实存在的取值。"""

    @pytest.mark.parametrize("key", ["all", "pending", "processing", "complete"])
    def test_card_keys_are_valid(self, key):
        vals = quick_filter_values(key)
        assert vals, f"{key} 应映射出至少一个取值"
        for v in vals:
            assert v in ALL_KEYS, f"{v} 不在下拉取值里"

    def test_each_card_round_trips(self):
        """点卡片 → 归一 → 该卡片仍应高亮（回读一致）。"""
        for key in ("pending", "processing", "complete", "all"):
            vals = quick_filter_values(key)
            norm = normalize_status_filter(vals)
            current = ",".join(norm) if norm else "all"
            assert active_quick_filter_key(current) == key, (
                f"{key} 点完后应保持高亮, 实际 current={current!r}"
            )
