"""授权模式运行时状态（进程内共享：Qt 主线程与 FastAPI 线程同读）。

模式只有两种：FREE（免费版，额度限制）/ PRO（完整版）。
判定规则见 docs/界面设计/07-授权管理.md §1.2。
"""

import threading
from enum import Enum


class Mode(Enum):
    FREE = "free"
    PRO = "pro"


_current = Mode.FREE
_reason = "init"
_lock = threading.Lock()
_listeners: list = []


def set_mode(mode: Mode, reason: str = "") -> None:
    global _current, _reason
    with _lock:
        _current = mode
        _reason = reason
    for cb in list(_listeners):
        try:
            cb(mode, reason)
        except Exception:
            pass


def current() -> Mode:
    with _lock:
        return _current


def reason() -> str:
    with _lock:
        return _reason


def is_pro() -> bool:
    return current() is Mode.PRO


def add_listener(cb) -> None:
    _listeners.append(cb)
