"""任务页「取消排队」：调度器内存拦截 + /task/batch-unqueue 端点。

背景：排队任务在 API 启动时被调度器自动加载执行；「取消排队」= QUEUED →
PENDING（退回未激活），且必须拦住已在内存队列里的任务——worker 派发前
不复查库状态，光改库挡不住（会照跑并写回 COMPLETE）。

覆盖:
1. cancel_task 记入取消集 → _schedule_next 派发时丢弃（不派发、记账平）
2. submit_task 显式重入队 → 解除取消（用户重新点「分析」可照常跑）
3. batch-unqueue 端点：只翻 QUEUED；运行中（running_tasks）不动；
   task_ids 与 all_queued 两种作用域
"""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.db.models as models
from src.core.scheduler import SchedulerStatus, TaskScheduler
from src.db.models import Base, Task


def _make_scheduler() -> TaskScheduler:
    """显式传并发参数 → 不触发硬件自动检测。"""
    return TaskScheduler(
        max_concurrent=4,
        max_md5_concurrent=2,
        max_sifting_concurrent=4,
        max_vlm_concurrent=3,
    )


class TestSchedulerCancelIntercept:
    def test_cancelled_task_not_dispatched(self, monkeypatch):
        """取消集内的任务在派发点被丢弃，未取消的照常派发。"""

        async def run():
            sch = _make_scheduler()
            monkeypatch.setattr(sch, "_is_pinned", lambda tid: False)
            dispatched = []

            async def _fake_exec(tid):
                dispatched.append(tid)

            monkeypatch.setattr(sch, "_execute_task", _fake_exec)
            sch.status = SchedulerStatus.RUNNING

            # 提交阶段屏蔽自动派发，把"入队 → 取消 → 派发"的时序摆清楚
            orig_next = sch._schedule_next

            async def _noop():
                return None

            sch._schedule_next = _noop  # type: ignore[method-assign]
            await sch.submit_task(1)
            await sch.submit_task(2)
            await sch.cancel_task(1)
            sch._schedule_next = orig_next  # type: ignore[method-assign]

            await sch._schedule_next()  # 弹出 1（取消 → 丢弃）
            await sch._schedule_next()  # 弹出 2（派发）
            await asyncio.sleep(0)  # 让 create_task 的桩跑完
            return sorted(dispatched)

        assert asyncio.run(run()) == [2]

    def test_cancel_is_idempotent_and_survives_restarts(self):
        """重复取消无害；取消集不随实例内 stop/start 清空（防陈旧队列项复活）。"""

        async def run():
            sch = _make_scheduler()
            await sch.cancel_task(1)
            await sch.cancel_task(1)
            assert sch._cancelled_tasks == {1}

        asyncio.run(run())

    def test_resubmit_clears_cancel(self, monkeypatch):
        """用户显式重新入队（submit_task）= 撤销取消，任务恢复可派发。"""

        async def run():
            sch = _make_scheduler()
            monkeypatch.setattr(sch, "_is_pinned", lambda tid: False)
            sch.status = SchedulerStatus.RUNNING

            async def _noop():
                return None

            monkeypatch.setattr(sch, "_schedule_next", _noop)
            await sch.submit_task(1)
            await sch.cancel_task(1)
            assert 1 in sch._cancelled_tasks
            await sch.submit_task(1)
            assert 1 not in sch._cancelled_tasks

        asyncio.run(run())


# ── 端点 ────────────────────────────────────────────────────────────


@pytest.fixture
def env(tmp_path, monkeypatch):
    """临时库 + 可控调度器（main.py 端点经 get_scheduler()/models 取依赖）。"""
    import src.api.main as M

    engine = create_engine(f"sqlite:///{tmp_path/'unqueue.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(models, "get_db_engine", lambda db_path=None: engine)

    sched = TaskScheduler(
        max_concurrent=1,
        max_md5_concurrent=1,
        max_sifting_concurrent=1,
        max_vlm_concurrent=1,
    )
    monkeypatch.setattr(M, "get_scheduler", lambda: sched)
    return {"engine": engine, "scheduler": sched}


def _add_task(engine, path, status, duration=100.0):
    Session = sessionmaker(bind=engine)
    db = Session()
    task = Task(file_path=path, status=status, duration=duration)
    db.add(task)
    db.commit()
    tid = task.id
    db.close()
    return tid


def _status_of(engine, task_id):
    Session = sessionmaker(bind=engine)
    db = Session()
    status = db.query(Task).filter(Task.id == task_id).first().status
    db.close()
    return status


def _client():
    import src.api.main as M

    return TestClient(M.app)


class TestBatchUnqueueEndpoint:
    def test_flips_only_queued_and_skips_running(self, env):
        """只翻真正的 QUEUED；运行中（running_tasks）不动；其他状态原样。"""
        engine, sched = env["engine"], env["scheduler"]
        q1 = _add_task(engine, "/x/q1.mp4", "QUEUED")
        q2 = _add_task(engine, "/x/q2.mp4", "QUEUED")
        running = _add_task(engine, "/x/r.mp4", "QUEUED")
        pending = _add_task(engine, "/x/p.mp4", "PENDING")
        done = _add_task(engine, "/x/c.mp4", "COMPLETE")
        sched.running_tasks.add(running)  # 已派发、线程池在跑

        resp = _client().post(
            "/task/batch-unqueue",
            json={"task_ids": [q1, q2, running, pending, done]},
        )
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "unqueued": 2}

        assert _status_of(engine, q1) == "PENDING"
        assert _status_of(engine, q2) == "PENDING"
        assert _status_of(engine, running) == "QUEUED", "运行中任务不得被取消排队"
        assert _status_of(engine, pending) == "PENDING"
        assert _status_of(engine, done) == "COMPLETE"
        assert q1 in sched._cancelled_tasks and q2 in sched._cancelled_tasks
        assert running not in sched._cancelled_tasks

    def test_all_queued_scope(self, env):
        """all_queued=true：全库 QUEUED 退回，其他状态不碰。"""
        engine, sched = env["engine"], env["scheduler"]
        q1 = _add_task(engine, "/x/q1.mp4", "QUEUED")
        q2 = _add_task(engine, "/x/q2.mp4", "QUEUED")
        paused = _add_task(engine, "/x/pa.mp4", "PAUSED")

        resp = _client().post("/task/batch-unqueue", json={"all_queued": True})
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "unqueued": 2}

        assert _status_of(engine, q1) == "PENDING"
        assert _status_of(engine, q2) == "PENDING"
        assert _status_of(engine, paused) == "PAUSED"
        assert sched._cancelled_tasks == {q1, q2}

    def test_no_queued_is_noop(self, env):
        """所选里没有 QUEUED → 计数 0、无副作用。"""
        engine, sched = env["engine"], env["scheduler"]
        done = _add_task(engine, "/x/c.mp4", "COMPLETE")

        resp = _client().post(
            "/task/batch-unqueue", json={"task_ids": [done]})
        assert resp.json() == {"success": True, "unqueued": 0}
        assert _status_of(engine, done) == "COMPLETE"
        assert sched._cancelled_tasks == set()
