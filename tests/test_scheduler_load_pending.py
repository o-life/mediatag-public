"""调度器重启加载：PENDING 不得被自动扫走。

背景（真实事故）：导入只把文件登记为 ``PENDING``（已导入、未激活），
但 ``_load_pending_tasks()`` 曾把 ``PENDING`` / ``PAUSED`` / ``FAILED`` 一并
加载进队列，导致**每次重启 App，未激活的 PENDING 都被自动分析**，
用户没点过「分析」却看到文件已处理完。

设计语义（见 openspec/changes/separate-import-from-analysis/proposal.md）：

| 状态 | 含义 | 重启行为 |
|---|---|---|
| ``QUEUED`` | 用户已激活 | 继续执行 |
| 中间态（SIFTING/ANALYZING/...） | 上次运行被中断的孤儿 | 重置为 QUEUED 恢复 |
| ``PENDING`` | 已导入、未激活 | **不得自动分析** |
| ``PAUSED`` | 用户暂停 | 保持暂停 |
| ``FAILED`` | 失败 | 须用户主动重试 |
"""

import asyncio
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.db.models as models
from src.core.scheduler import (
    INTERRUPTED_TASK_STATUSES,
    QUEUED_TASK_STATUSES,
    TaskScheduler,
)
from src.db.models import Base, Task


@pytest.fixture
def db_engine(tmp_path, monkeypatch):
    """临时库，并把 scheduler 内部 `get_db_engine` 指向它。"""
    engine = create_engine(f"sqlite:///{tmp_path/'sched.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(models, "get_db_engine", lambda db_path=None: engine)
    return engine


def _make_scheduler() -> TaskScheduler:
    """显式传并发参数 → 不触发硬件自动检测。"""
    return TaskScheduler(
        max_concurrent=4,
        max_md5_concurrent=2,
        max_sifting_concurrent=4,
        max_vlm_concurrent=3,
    )


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


def _drain(sch):
    ids = []
    while not sch.task_queue.empty():
        _prio, _dur, tid = sch.task_queue.get_nowait()
        ids.append(tid)
    return ids


def _load(sch):
    asyncio.run(sch._load_pending_tasks())


class TestRestartLoadsOnlyActivated:
    def test_pending_not_auto_loaded(self, db_engine):
        """核心回归：未激活的 PENDING 重启后仍为 PENDING、不进队列。"""
        tid = _add_task(db_engine, "/x/a.mp4", "PENDING")
        sch = _make_scheduler()

        _load(sch)

        assert _drain(sch) == [], "PENDING 不得被自动加载执行"
        assert _status_of(db_engine, tid) == "PENDING", "未激活状态必须原样保留"

    def test_queued_is_loaded_and_kept(self, db_engine):
        """用户已激活的 QUEUED → 继续执行（状态保持 QUEUED）。"""
        tid = _add_task(db_engine, "/x/a.mp4", "QUEUED")
        sch = _make_scheduler()

        _load(sch)

        assert _drain(sch) == [tid]
        assert _status_of(db_engine, tid) == "QUEUED"

    def test_paused_keeps_paused(self, db_engine):
        """暂停的任务跨重启保持暂停，不自动恢复。"""
        tid = _add_task(db_engine, "/x/a.mp4", "PAUSED")
        sch = _make_scheduler()

        _load(sch)

        assert _drain(sch) == []
        assert _status_of(db_engine, tid) == "PAUSED"

    def test_failed_and_error_need_manual_retry(self, db_engine):
        """失败任务不自动重试；ERROR 同样不自动加载。"""
        failed = _add_task(db_engine, "/x/f.mp4", "FAILED")
        error = _add_task(db_engine, "/x/e.mp4", "ERROR")
        sch = _make_scheduler()

        _load(sch)

        assert _drain(sch) == []
        assert _status_of(db_engine, failed) == "FAILED"
        assert _status_of(db_engine, error) == "ERROR"


class TestOrphanRecoveryStillWorks:
    @pytest.mark.parametrize("orphan", INTERRUPTED_TASK_STATUSES)
    def test_orphan_reset_to_queued_and_loaded(self, db_engine, orphan):
        """上次运行被中断的中间态 → 重置为 QUEUED 并继续（不得因此丢失恢复能力）。"""
        tid = _add_task(db_engine, "/x/a.mp4", orphan)
        sch = _make_scheduler()

        _load(sch)

        assert _drain(sch) == [tid], "孤儿任务必须被恢复"
        assert _status_of(db_engine, tid) == "QUEUED", "孤儿任务应被重置为 QUEUED"

    def test_mixed_batch(self, db_engine):
        """混合场景：只加载 QUEUED + 孤儿，其余原样不动。"""
        queued = _add_task(db_engine, "/q.mp4", "QUEUED")
        orphan = _add_task(db_engine, "/o.mp4", "ANALYZING")
        pending = _add_task(db_engine, "/p.mp4", "PENDING")
        paused = _add_task(db_engine, "/pa.mp4", "PAUSED")
        failed = _add_task(db_engine, "/f.mp4", "FAILED")
        complete = _add_task(db_engine, "/c.mp4", "COMPLETE")

        sch = _make_scheduler()
        _load(sch)

        assert sorted(_drain(sch)) == sorted([queued, orphan])
        assert _status_of(db_engine, pending) == "PENDING"
        assert _status_of(db_engine, paused) == "PAUSED"
        assert _status_of(db_engine, failed) == "FAILED"
        assert _status_of(db_engine, complete) == "COMPLETE"


class TestStatusConstants:
    def test_constants_exclude_unactivated_states(self):
        """锁住语义：可加载状态里不得出现 PENDING/PAUSED/FAILED。"""
        loadable = QUEUED_TASK_STATUSES + INTERRUPTED_TASK_STATUSES
        for forbidden in ("PENDING", "PAUSED", "FAILED", "ERROR", "COMPLETE"):
            assert forbidden not in loadable
