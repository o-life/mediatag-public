"""SQLite 写闸（write_gate）单测。

见 openspec/changes/fix-sqlite-write-lock：
- A：进程级单写闸串行化所有写（flush/commit）
- B：写引擎以 BEGIN IMMEDIATE 预取写锁
"""

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db import write_gate as wg


@pytest.fixture(autouse=True)
def _reset_gate():
    wg.reset_enabled_cache()
    wg.reset_lock_stats()
    yield
    wg.reset_enabled_cache()
    wg.reset_lock_stats()


class TestBasics:
    def test_is_locked_error(self):
        assert wg.is_locked_error(Exception("database is locked"))
        assert wg.is_locked_error(Exception("(sqlite3.OperationalError) database is locked"))
        assert not wg.is_locked_error(Exception("no such table: x"))

    def test_depth_and_in_guard(self):
        assert not wg.in_write_guard()
        with wg.write_guard():
            assert wg.in_write_guard() and wg.guard_depth() == 1
            with wg.write_guard():          # 可重入
                assert wg.guard_depth() == 2
            assert wg.guard_depth() == 1
        assert not wg.in_write_guard()

    def test_context_manager_releases_on_exception(self):
        with pytest.raises(ValueError):
            with wg.write_guard():
                raise ValueError("boom")
        assert not wg.in_write_guard()
        # 锁已释放：其它线程能立刻拿到
        got = []

        def worker():
            with wg.write_guard():
                got.append(1)

        t = threading.Thread(target=worker)
        t.start(); t.join(timeout=2)
        assert got == [1]


class TestMutex:
    def test_serializes_two_threads(self):
        """两个线程各自在闸内停留，不得重叠。"""
        order = []
        barrier = threading.Barrier(2)

        def worker(name):
            barrier.wait()
            with wg.write_guard():
                order.append(f"{name}:enter")
                time.sleep(0.05)
                order.append(f"{name}:exit")

        ts = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
        for t in ts: t.start()
        for t in ts: t.join(timeout=5)
        # 必须成对出现且不交错：enter/exit 相邻
        assert order[0].endswith("enter") and order[1].endswith("exit")
        assert order[2].endswith("enter") and order[3].endswith("exit")
        assert order[0].split(":")[0] == order[1].split(":")[0]

    def test_timeout_raises_and_counts(self):
        """另一线程持有闸时，带超时的获取应抛 TimeoutError 并计数。"""
        held = threading.Event()
        release = threading.Event()

        def holder():
            with wg.write_guard():
                held.set()
                release.wait(5)

        t = threading.Thread(target=holder); t.start()
        assert held.wait(2)
        try:
            with pytest.raises(TimeoutError):
                with wg.write_guard(timeout=0.2):
                    pass
        finally:
            release.set(); t.join(timeout=5)
        assert wg.lock_stats()["lock_failure_total"] >= 1


class TestDisabled:
    def test_disabled_is_noop(self, monkeypatch):
        monkeypatch.setattr(wg, "enabled", lambda: False)
        with wg.write_guard():
            assert not wg.in_write_guard()   # 不标记、不持锁


class TestGuardedSession:
    """A：GuardedSession 的 commit 经写闸串行化。

    说明：B（写事务 BEGIN IMMEDIATE）经压测证明有害，已放弃，见
    openspec/changes/fix-sqlite-write-lock；此处只验证写闸本身生效。
    """

    def test_commit_waits_for_gate(self, tmp_path):
        from src.db import models as M

        eng = M.get_db_engine(str(tmp_path / "s.db"))
        M.Base.metadata.create_all(eng)

        elapsed = []

        def worker():
            s = M.get_session(eng)          # GuardedSession
            s.add(M.Actor(code="B"))
            t0 = time.time()
            s.commit()                      # 应阻塞，直到主线程释放写闸
            elapsed.append(time.time() - t0)
            s.close()

        with wg.write_guard():              # 主线程持有写闸
            t = threading.Thread(target=worker)
            t.start()
            time.sleep(0.3)
            assert not elapsed, "commit 不应在写闸被占用时完成"
        t.join(timeout=5)
        assert elapsed and elapsed[0] >= 0.2, elapsed


class TestGuardedExecute:
    """不经 flush 的写（Core/ORM execute、text DML）也必须持有写闸。"""

    def _engine(self, tmp_path):
        from src.db import models as M
        eng = M.get_db_engine(str(tmp_path / "e.db"))
        M.Base.metadata.create_all(eng)
        return eng

    def test_core_insert_acquires_gate(self, tmp_path):
        from sqlalchemy import insert
        from src.db import models as M
        s = M.get_session(self._engine(tmp_path))
        try:
            s.execute(insert(M.Actor).values(code="X"))   # Core execute（不过 flush）
            assert s._gate_held is True, "Core INSERT 应已持有写闸"
            s.commit()
            assert s._gate_held is False, "commit 后应释放写闸"
        finally:
            s.close()

    def test_text_select_does_not_acquire_gate(self, tmp_path):
        from sqlalchemy import text
        from src.db import models as M
        s = M.get_session(self._engine(tmp_path))
        try:
            s.execute(text("SELECT 1"))
            assert s._gate_held is False, "只读语句不应占用写闸"
        finally:
            s.close()

    def test_statement_is_write_classifier(self):
        from sqlalchemy import insert, select, text
        from src.db import models as M
        assert M._statement_is_write(insert(M.Actor).values(code="x")) is True
        assert M._statement_is_write(text("UPDATE tasks SET progress=1")) is True
        assert M._statement_is_write(text("DELETE FROM tasks")) is True
        assert M._statement_is_write(select(M.Actor)) is False
        assert M._statement_is_write(text("SELECT 1")) is False
