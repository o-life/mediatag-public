"""轻量 Qt 后台任务封装：网络调用一律不阻塞主线程（防启动假死/未响应）。"""

from PySide6.QtCore import QThread, Signal


class AsyncWorker(QThread):
    """在 QThread 中执行 fn；结果经 done/failed 信号回主线程。"""

    done = Signal(object)
    failed = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # 业务/网络异常统一回传，由 UI 分类展示
            self.failed.emit(exc)
