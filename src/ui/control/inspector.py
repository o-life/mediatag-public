"""inspector.py — 运行时元素检查器（Qt Widgets 通用，零业务依赖）。

对标：Web DevTools 元素面板 / Xcode View Debugger / GammaRay（KDAB）
的"悬停即看几何"。自研轻量版：⌥I 切换后，鼠标悬停任意控件，屏幕上
直接叠出该控件的 x/y/w/h、margins、objectName 与类名——开发态精准
定位元素，不再靠截图像素猜测（D8 排查半天的教训）。

实现要点（坑位已按项目 pitfalls 处理）：
- 覆盖层 WA_TransparentForMouseEvents——绝不挡交互；
- 50ms 轮询 QApplication.widgetAt(光标全局位)——不用事件重放，
  不与 _interactive_under 的 widgetAt 转发互相干扰；
- 只在开发态手动开启，打包态默认关（不进截图、不进黄金基线）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


class InspectorOverlay(QWidget):
    """窗口级透明覆盖层：高亮光标下的控件并标注几何信息。"""

    REFRESH_MS = 60
    ACCENT = QColor(10, 132, 255)

    def __init__(self, window: QWidget):
        super().__init__(window)
        self._window = window
        self._target: QWidget | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(window.rect())
        self.hide()

        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._poll)
        window.destroyed.connect(lambda *_: self._stop())

    # ── 生命周期 ─────────────────────────────────────────────

    def toggle(self) -> bool:
        if self.isVisible():
            self._stop()
            return False
        self._timer.start()
        self.setGeometry(self._window.rect())
        self.show()
        self.raise_()
        return True

    def _stop(self) -> None:
        self._timer.stop()
        self._target = None
        self.hide()

    # ── 采样 ─────────────────────────────────────────────────

    def _poll(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        global_pos = QCursor.pos()
        pos = self._window.mapFromGlobal(global_pos)
        if not self._window.rect().contains(pos):
            if self._target is not None:
                self._target = None
                self.update()
            return
        target = QApplication.widgetAt(global_pos)
        # 覆盖层自己与窗口 chrome 不作为检查目标
        if target is self or target is self._window:
            target = None
        if target is not self._target:
            self._target = target
            self.update()
        elif target is not None:
            self.update()  # 跟手重画标签

    def _window_rect(self, w: QWidget):
        """控件在窗口坐标系中的矩形——用几何累加（自洽）。

        ⚠️ 不用 mapTo：content_stack 有一次 reparent 遗留，mapTo 在该跳
        坐标翻倍（实测 x 61→122），检查器读数会系统性虚大。"""
        rect = w.geometry()
        cur = w.parentWidget()
        while cur is not None and cur is not self._window:
            rect.translate(cur.geometry().topLeft())
            cur = cur.parentWidget()
        return rect

    # ── 绘制 ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self._target is None:
            return
        w = self._target
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 目标在窗口坐标系中的矩形（几何累加，防 mapTo 翻倍跳）
        rect = self._window_rect(w)

        pen = QPen(self.ACCENT, 2)
        painter.setPen(pen)
        painter.setBrush(QColor(10, 132, 255, 28))
        painter.drawRect(rect)

        # 标注框
        margins = w.contentsMargins()
        label = (
            f"{type(w).__name__}"
            + (f"#{w.objectName()}" if w.objectName() else "")
            + f"  x={rect.x()} y={rect.y()}  {rect.width()}×{rect.height()}"
            + f"  margins T{margins.top()} B{margins.bottom()}"
              f" L{margins.left()} R{margins.right()}"
        )
        font = QFont()
        font.setPixelSize(12)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(label) + 16
        th = metrics.height() + 8
        lx = min(rect.x(), self.width() - tw - 4)
        ly = rect.y() - th - 4
        if ly < 2:
            ly = rect.bottom() + 4
        painter.setBrush(QColor(28, 28, 30, 235))
        painter.setPen(QPen(self.ACCENT, 1))
        painter.drawRect(lx, ly, tw, th)
        painter.setPen(QColor(245, 245, 247))
        painter.drawText(lx + 8, ly + th - 7, label)
        painter.end()
