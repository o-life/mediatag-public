"""Tests for 调度器短任务优先排序 (src/core/scheduler.py)。

设计依据: docs/架构/调度与进度呈现-模拟评估.md §5.3/§6 ——
"短任务优先必须作用在全链路派发"; 采用 §5.4 定稿的"纯排序"方案:
派发键 = (优先级, 时长升序, task_id)。

覆盖:
1. _sort_key: 已知时长 / 未知时长 / 关闭开关 / 缓存命中
2. 队列出队顺序: 短任务优先 / 关闭后退回 FIFO / 优先级仍高于时长
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.scheduler import SchedulerStatus, TaskPriority, TaskScheduler


def _make_scheduler(short_task_first: bool) -> TaskScheduler:
    """构造调度器(显式传并发参数 → 不触发硬件自动检测)。"""
    sch = TaskScheduler(
        max_concurrent=4,
        max_md5_concurrent=2,
        max_sifting_concurrent=4,
        max_vlm_concurrent=3,
    )
    sch.short_task_first = short_task_first
    return sch


def _drain(sch: TaskScheduler):
    order = []
    while not sch.task_queue.empty():
        _prio, _dur, tid = sch.task_queue.get_nowait()
        order.append(tid)
    return order


class TestSortKey:
    def test_disabled_returns_zero(self):
        """关闭开关 → 中位排序键恒为 0(退化为 FIFO)。"""
        sch = _make_scheduler(False)
        assert sch._sort_key(1, 3600.0) == 0.0

    def test_known_duration_returned_as_is(self):
        sch = _make_scheduler(True)
        assert sch._sort_key(1, 600.0) == 600.0

    def test_unknown_duration_goes_last(self, monkeypatch):
        """未知时长用 inf(排末位), 不用 0 —— 避免"未知"被误当最短。"""
        sch = _make_scheduler(True)
        monkeypatch.setattr(sch, "_lookup_duration", lambda task_id: None)
        assert sch._sort_key(1) == float("inf")

    def test_zero_duration_goes_last(self, monkeypatch):
        sch = _make_scheduler(True)
        monkeypatch.setattr(sch, "_lookup_duration", lambda task_id: None)
        assert sch._sort_key(1, 0) == float("inf")

    def test_cache_is_used(self, monkeypatch):
        sch = _make_scheduler(True)
        sch._duration_cache[7] = 42.0

        def _boom(tid):
            pytest.fail("应命中缓存, 不应查库")

        monkeypatch.setattr(sch, "_lookup_duration", _boom)
        assert sch._sort_key(7) == 42.0


class TestQueueOrdering:
    def test_shortest_dispatched_first(self, monkeypatch):
        """id 递增但时长递减 → 出队应为短到长(而非 FIFO 的 1,2,3)。"""

        async def run():
            sch = _make_scheduler(True)
            sch.status = SchedulerStatus.RUNNING

            async def _noop():
                return None

            monkeypatch.setattr(sch, "_schedule_next", _noop)
            await sch.submit_task(1, duration=3600.0)
            await sch.submit_task(2, duration=600.0)
            await sch.submit_task(3, duration=60.0)
            return _drain(sch)

        assert asyncio.run(run()) == [3, 2, 1]

    def test_fifo_when_disabled(self, monkeypatch):
        """关闭短任务优先 → 退回按 task_id 的 FIFO。"""

        async def run():
            sch = _make_scheduler(False)
            sch.status = SchedulerStatus.RUNNING

            async def _noop():
                return None

            monkeypatch.setattr(sch, "_schedule_next", _noop)
            await sch.submit_task(3, duration=60.0)
            await sch.submit_task(1, duration=3600.0)
            await sch.submit_task(2, duration=600.0)
            return _drain(sch)

        assert asyncio.run(run()) == [1, 2, 3]

    def test_priority_outranks_duration(self, monkeypatch):
        """优先级仍高于时长: HIGH 的长任务先于 NORMAL 的短任务。"""

        async def run():
            sch = _make_scheduler(True)
            sch.status = SchedulerStatus.RUNNING

            async def _noop():
                return None

            monkeypatch.setattr(sch, "_schedule_next", _noop)
            await sch.submit_task(1, priority=TaskPriority.NORMAL, duration=60.0)
            await sch.submit_task(2, priority=TaskPriority.HIGH, duration=3600.0)
            return _drain(sch)

        assert asyncio.run(run()) == [2, 1]

    def test_unknown_duration_last_within_priority(self, monkeypatch):
        """未知时长任务排在同一优先级的已知短/长任务之后。"""

        async def run():
            sch = _make_scheduler(True)
            sch.status = SchedulerStatus.RUNNING

            async def _noop():
                return None

            monkeypatch.setattr(sch, "_schedule_next", _noop)
            monkeypatch.setattr(sch, "_lookup_duration", lambda task_id: None)
            await sch.submit_task(1)                  # 未知
            await sch.submit_task(2, duration=9999.0)
            await sch.submit_task(3, duration=10.0)
            return _drain(sch)

        assert asyncio.run(run()) == [3, 2, 1]
