"""SQLite 写串行化（write gate）。

见 `openspec/changes/fix-sqlite-write-lock/`。

- **A（本模块）**：进程级单写闸。所有 DB 写（flush / commit）经 `write_guard()`
  串行化，保证进程内同一时刻只有一个写者 ⇒ 减少"写-写争抢"导致的
  `database is locked`。
- **B（BEGIN IMMEDIATE）已放弃**：经压测证伪——`begin` 事件对只读语句也会触发，
  会让 SELECT 也进入事务并持有快照，随后写回时 WAL 快照过期升级，反而**制造**
  锁错误（见 `models.get_db_engine` 注释与 design.md）。故只保留 A。

约定：`write_guard()` **只包 DB 写段**，不要把推理 / 文件 I/O 放进闸内，
否则会把长流程串行化，反而更慢。
"""

import threading
from contextlib import contextmanager

_WRITE_LOCK = threading.RLock()          # 可重入：同线程嵌套安全
_state = threading.local()               # 每线程的进入深度
_stats_lock = threading.Lock()
_stats = {"lock_retry_total": 0, "lock_failure_total": 0}
_enabled_cache = None
_timeout_cache = None


def is_locked_error(exc: BaseException) -> bool:
    """是否为 SQLite 写锁冲突（而非其它错误）。"""
    s = str(exc).lower()
    return "database is locked" in s or "database table is locked" in s


def in_write_guard() -> bool:
    """当前线程是否处于写闸内（供 begin 事件决定是否发 BEGIN IMMEDIATE）。"""
    return getattr(_state, "depth", 0) > 0


def guard_depth() -> int:
    return getattr(_state, "depth", 0)


def note_lock_retry() -> None:
    with _stats_lock:
        _stats["lock_retry_total"] += 1


def note_lock_failure() -> None:
    with _stats_lock:
        _stats["lock_failure_total"] += 1


def lock_stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def reset_lock_stats() -> None:
    with _stats_lock:
        _stats.update({"lock_retry_total": 0, "lock_failure_total": 0})


def _config_value(key, default):
    try:
        from src.core.config import ConfigManager
        return ConfigManager().get(key, default)
    except Exception:
        return default


def enabled() -> bool:
    """是否启用写闸（缺配置时默认启用）。结果缓存，避免热路径反复读配置。"""
    global _enabled_cache
    if _enabled_cache is None:
        _enabled_cache = bool(_config_value("db.write_serialization.enabled", True))
    return _enabled_cache


def reset_enabled_cache() -> None:
    """测试或配置变更后重置开关/超时缓存。"""
    global _enabled_cache, _timeout_cache
    _enabled_cache = None
    _timeout_cache = None


def guard_timeout() -> float | None:
    """写闸获取超时（秒）。**缓存**：写闸在每次 flush/commit 都会进入，
    热路径上绝不能每次去读配置文件。"""
    global _timeout_cache
    if _timeout_cache is None:
        try:
            t = _config_value("db.write_serialization.guard_timeout_seconds", 60)
            t = float(t) if t is not None else None
            _timeout_cache = t if (t and t > 0) else 0.0   # 0 → 无限等待
        except Exception:
            _timeout_cache = 60.0
    return _timeout_cache or None


@contextmanager
def write_guard(timeout: float | None = None):
    """进入写闸（可重入）。超时未取得则抛 TimeoutError 并计入失败指标。"""
    if not enabled():
        yield
        return

    if timeout is None:
        timeout = guard_timeout()

    if timeout and timeout > 0:
        acquired = _WRITE_LOCK.acquire(timeout=timeout)
    else:
        _WRITE_LOCK.acquire()
        acquired = True

    if not acquired:
        note_lock_failure()
        raise TimeoutError("write_guard: 获取写闸超时（可能有长写事务未释放）")

    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1
        _WRITE_LOCK.release()


def acquire_write_gate(timeout: float | None = None) -> bool:
    """手动获取写闸（供 `GuardedSession` 做**事务级**串行化：从首次写持有到
    commit/rollback 释放）。超时返回 False。"""
    if not enabled():
        return True
    if timeout is None:
        timeout = guard_timeout()
    if timeout and timeout > 0:
        return _WRITE_LOCK.acquire(timeout=timeout)
    _WRITE_LOCK.acquire()
    return True


def release_write_gate() -> None:
    """释放 `acquire_write_gate()` 取得的写闸。"""
    if not enabled():
        return
    try:
        _WRITE_LOCK.release()
    except RuntimeError:
        pass
