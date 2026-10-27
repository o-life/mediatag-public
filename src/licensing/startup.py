"""启动时的授权模式探测：后台线程解析，回调在主线程执行（QObject 方法时由 Qt 队列保证）。"""

from .qt_worker import AsyncWorker

_resolvers: list = []  # 持引用防止 worker 被 GC


def start_mode_detection(callback) -> None:
    """解析授权模式（完整版/免费版）。callback(ModeResolution) 建议传 QObject 绑定方法，
    以获得主线程队列执行（如 MainWindow.license_mode_ready）。"""
    from .gate import resolve_mode

    worker = AsyncWorker(resolve_mode)
    worker.done.connect(callback)
    worker.failed.connect(
        lambda exc: callback(
            _fallback_resolution(f"授权检测异常：{exc}")
        )
    )
    _resolvers.append(worker)
    worker.start()


def _fallback_resolution(note: str):
    from . import state
    from .gate import ModeResolution

    state.set_mode(state.Mode.FREE, "error")
    return ModeResolution(mode=state.Mode.FREE, reason="error", notice=note)
