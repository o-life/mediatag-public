"""导入分批提交的「database is locked」退避重试。

SQLite 写-写串行：并发的分析 worker / 人脸重建会与导入短暂互斥。一次瞬时锁不该
把整批（最多 IMPORT_COMMIT_BATCH 条）已登记的作业丢掉 —— 失败即回滚、重新 add 本批、
退避后重试。
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M


class _FakeDB:
    """记录 commit/rollback/add 调用的最小替身。"""

    def __init__(self, fail_times: int = 0, lock: bool = True):
        self.fail_times = fail_times
        self.lock = lock
        self.commits = 0
        self.rollbacks = 0
        self.added = []

    def commit(self):
        self.commits += 1
        if self.commits <= self.fail_times:
            msg = "database is locked" if self.lock else "some other failure"
            raise OperationalError("COMMIT", {}, Exception(msg))

    def rollback(self):
        self.rollbacks += 1

    def add(self, obj):
        self.added.append(obj)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """退避 sleep 置空，测试不真等。"""
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)


class TestCommitBatchWithLockRetry:
    def test_commits_once_when_ok(self):
        db = _FakeDB()
        M._commit_batch_with_lock_retry(db, ["a", "b"])
        assert db.commits == 1
        assert db.rollbacks == 0
        assert db.added == []

    def test_retries_on_lock_and_readds_the_batch(self):
        db = _FakeDB(fail_times=2)
        batch = ["a", "b"]
        M._commit_batch_with_lock_retry(db, batch, attempts=5, base_delay=0.0)
        assert db.commits == 3, "2 次锁失败 + 1 次成功"
        assert db.rollbacks == 2
        assert db.added == batch + batch, "每次重试前都要重新 add 本批"

    def test_non_lock_error_raises_without_retry(self):
        db = _FakeDB(fail_times=1, lock=False)
        with pytest.raises(OperationalError):
            M._commit_batch_with_lock_retry(db, ["a"], attempts=5, base_delay=0.0)
        assert db.commits == 1, "非锁错误不重试"
        assert db.rollbacks == 1

    def test_gives_up_after_attempts(self):
        db = _FakeDB(fail_times=99)
        with pytest.raises(OperationalError):
            M._commit_batch_with_lock_retry(db, ["a"], attempts=3, base_delay=0.0)
        assert db.commits == 3
