"""AsyncLoader — 通用异步加载器（把耗时调用挪到工作线程，避免阻塞 UI）。

用法：
    loader = AsyncLoader(self.api.get, "/videos")
    loader.finished.connect(self._on_loaded)   # 回到主线程执行
    loader.error.connect(self._on_failed)
    loader.run()

注意：
- 被加载的函数**不得触碰 Qt 控件**（Qt 控件只能在主线程创建/修改）；
  典型用途是纯网络/IO 调用，返回纯数据，再由主线程渲染。
- 调用方需自己持有 loader 引用并在发起期间防重入，否则会被 GC。
"""

from PySide6.QtCore import QObject, QThread, Signal


class AsyncLoader(QObject):
    """通用异步加载器，避免阻塞 UI。"""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._thread: QThread | None = None

    def run(self):
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        self._thread.finished.connect(self.deleteLater)
        self._thread.start()

    def _execute(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if self._thread:
                self._thread.quit()
