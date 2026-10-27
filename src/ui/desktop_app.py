#!/usr/bin/env python3
"""Desktop app for ai-video-factory - Sidebar navigation with macOS native design."""

import os
import re
import stat
import sys
import json
import datetime
import threading
import requests
from pathlib import Path

# ── Emoji sanitizer ──────────────────────────────────────────────────
# macOS allows emoji in file names, but PySide6 triggers EXC_BAD_ACCESS
# in CoreText when rendering them. Strip all emoji/unicode symbols.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # supplemental symbols
    "\u2600-\u27BF"           # misc symbols
    "\u2300-\u23FF"           # misc technical
    "\uFE00-\uFE0F"           # variation selectors
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F000-\U0001F02F"  # mahjong/domino
    "\U0001F0A0-\U0001F0FF"  # playing cards
    "\U0001F100-\U0001F64F"  # enclosed alphanumerics + emoji
    "\U0001F680-\U0001F6FF"  # transport + map symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # extended symbols
    "\U0000200D"             # zero-width joiner
    "\U0000FE0F"             # variation selector-16
    "\u200C-\u200D"          # ZWJ/ZWNJ
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U0001F201-\U0001F2FF"  #CJK symbols
    "]",
)

def sanitize_for_qt(text: str) -> str:
    """Remove all emoji/special unicode from text safe for Qt rendering."""
    return _EMOJI_RE.sub("", text)

# ── Fix Qt platform plugin path on macOS ──────────────────────────────
# PySide6 installed via uv uses @rpath which doesn't resolve in venv.
# Fix by adding LC_RPATH to the cocoa platform plugin so it can find Qt frameworks.

def _site_packages() -> Path:
    """Locate site-packages for both the dev venv and the frozen bundle."""
    site_packages = Path(__file__).resolve().parent.parent.parent / ".venv" / "lib" / "python3.11" / "site-packages"
    if not site_packages.exists():
        site_packages = Path(__file__).resolve().parent / "site-packages"
    return site_packages


def _fix_qt_plugin():
    """Add LC_RPATH to Qt plugins so @rpath resolves correctly."""
    site_packages = _site_packages()
    plugins = site_packages / "PySide6" / "Qt" / "plugins"
    lib = site_packages / "PySide6" / "Qt" / "lib"
    if not plugins.exists() or not lib.exists():
        return
    try:
        import subprocess
        plugin = plugins / "platforms" / "libqcocoa.dylib"
        if plugin.exists():
            result = subprocess.run(
                ["install_name_tool", "-add_rpath", str(lib), str(plugin)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # Already has rpath, try to remove and re-add
                result = subprocess.run(
                    ["install_name_tool", "-delete_rpath", str(lib), str(plugin)],
                    capture_output=True, text=True
                )
                result = subprocess.run(
                    ["install_name_tool", "-add_rpath", str(lib), str(plugin)],
                    capture_output=True, text=True
                )
    except Exception:
        pass


def _fix_qt_plugin_hidden_flag():
    """Clear the macOS UF_HIDDEN flag on the Qt plugin tree.

    Qt's QDir/QDirIterator skip hidden entries by default, so a plugin tree
    carrying UF_HIDDEN makes the platform plugin loader enumerate nothing and
    QApplication aborts with "Could not find the Qt platform plugin cocoa".
    `ls` does not hide UF_HIDDEN files, so the tree looks perfectly normal in
    a shell — hence an explicit check here.
    """
    if not (hasattr(os, "chflags") and hasattr(stat, "UF_HIDDEN")):
        return
    try:
        plugins = _site_packages() / "PySide6" / "Qt" / "plugins"
        if not plugins.is_dir():
            return
        cleared = 0
        for path in [plugins, *plugins.rglob("*")]:
            try:
                flags = path.lstat().st_flags
            except OSError:
                continue
            if flags & stat.UF_HIDDEN:
                try:
                    os.chflags(path, flags & ~stat.UF_HIDDEN)
                    cleared += 1
                except OSError:
                    pass
        if cleared:
            import logging
            logging.getLogger(__name__).warning(
                "Qt 插件树带 macOS 隐藏标志（Qt 会枚举不到平台插件），已清除 %d 项", cleared
            )
    except Exception:
        pass


_fix_qt_plugin_hidden_flag()
_fix_qt_plugin()

# macOS 融合标题栏配套：空 NSToolbar(unified) 让系统把红绿灯垂直居中到
# 更高的标题栏带（官方路径，非私有 API），顶部（含红绿灯上方）留白加大。
# 委托必须常驻引用，否则 objc 对象被 GC 后 AppKit 回调即崩。
try:
    import AppKit as _AppKit

    class _EmptyToolbarDelegate(_AppKit.NSObject):
        def toolbarAllowedItemIdentifiers_(self, toolbar):
            return []

        def toolbarDefaultItemIdentifiers_(self, toolbar):
            return []

    class _TitlebarDragView(_AppKit.NSView):
        """标题栏 accessory 拖动条（10 号方案：红绿灯右侧可拖动的日常习惯）。

        按 Apple 官方方式实现自定义拖动区（AppKit 文档：自定义拖动区域
        在 NSView.mouseDown 里调用 NSWindow.performDrag(with:)，由系统
        进入原生拖动循环；视图同时以 mouseDownCanMoveWindow=NO 声明
        自管拖动，避免与系统背景拖动会话叠加造成双重移动）。
        Qt 层事件在标题栏区会被 NSTitlebarContainerView 吞掉（实测），
        因此拖动区必须在 Cocoa 层实现。视图类须常驻模块级引用。
        """

        def mouseDownCanMoveWindow(self):
            return False

        def acceptsFirstMouse_(self, ev):
            return True

        def mouseDown_(self, ev):
            self.window().performDragWithEvent_(ev)

except Exception:
    _AppKit = None
    _EmptyToolbarDelegate = None
    _TitlebarDragView = None


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QTextEdit, QProgressBar, QComboBox, QFrame, QStackedWidget,
    QDialog,
)
from PySide6.QtCore import QPoint, Qt, QTimer, QThread, Signal, QUrl, QSize
from PySide6.QtGui import QCursor, QIcon, QAction, QKeySequence, QDesktopServices, QColor


# ── New component imports ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ui.components.dashboard import DashboardPage
    from ui.components.label_mgmt import LabelManagementPage
    from ui.components.actor_mgmt import ActorManagementPage
    from ui.components.settings import SettingsPage
    from ui.components.theme import theme, global_stylesheet
    from ui.tokens import T as _T
    from ui.components.api_client import APIClient
    from ui.components.results_page import ResultsPage
    from ui.pages.tag_browse_page import TagBrowsePage
else:
    from .components.dashboard import DashboardPage
    from .components.label_mgmt import LabelManagementPage
    from .components.actor_mgmt import ActorManagementPage
    from .components.settings import SettingsPage
    from .components.theme import theme, global_stylesheet
    from .tokens import T as _T
    from .components.api_client import APIClient
    from .components.results_page import ResultsPage
    from .pages.tag_browse_page import TagBrowsePage

#: 外观两态的中文标签（顶栏按钮文案用）。
# 「玻璃」第三主题已退役（2026-09-22）：SDK 27 链接后系统全时提供官方新设计
# （玻璃窗口控件等），玻璃档与深色再无差异；历史持久化值 "glass" 归一化为深色。
_MODE_LABELS = {"dark": "深色", "light": "浅色"}

API_BASE = "http://127.0.0.1:8000"


class StatusCheckWorker:
    """Check backend status in a daemon thread."""
    
    def __init__(self, callback):
        """callback: callable(bool) invoked on main thread."""
        self._callback = callback
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        try:
            resp = requests.get(f"{API_BASE}/status", timeout=3)
            connected = resp.status_code == 200
        except Exception:
            connected = False
        # Use QTimer.singleShot to call back on main thread
        QTimer.singleShot(0, lambda: self._callback(connected))


class PageDataLoader:
    """Worker for loading page data asynchronously (daemon thread, no QThread)."""
    
    def __init__(self, index: int, api_base: str, callback):
        """callback: callable(int, data) invoked on main thread via QTimer.singleShot."""
        self.index = index
        self.api_base = api_base
        self._callback = callback
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def _run(self):
        try:
            if self.index == 1:  # Results page
                resp = requests.get(f"{self.api_base}/task", timeout=10)
                print(f"[DEBUG] PageDataLoader: API响应状态码={resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"[DEBUG] PageDataLoader: 数据类型={type(data).__name__}, 数量={len(data) if isinstance(data, list) else 'N/A'}")
                    QTimer.singleShot(0, lambda: self._callback(self.index, data))
                    print("[DEBUG] PageDataLoader: Callback已调度")
            elif self.index == 2:  # Labels page
                resp = requests.get(f"{self.api_base}/dictionary", timeout=10)
                if resp.status_code == 200:
                    QTimer.singleShot(0, lambda: self._callback(self.index, resp.json()))
            elif self.index == 3:  # Actors page
                resp = requests.get(f"{self.api_base}/actors", timeout=10)
                if resp.status_code == 200:
                    QTimer.singleShot(0, lambda: self._callback(self.index, resp.json()))
            elif self.index == 4:  # Settings page
                resp = requests.get(f"{self.api_base}/config", timeout=10)
                if resp.status_code == 200:
                    QTimer.singleShot(0, lambda: self._callback(self.index, resp.json()))
        except Exception as e:
            print(f"Error loading data for page {self.index}: {e}")


class _SidebarHandle(QFrame):
    """树列拖宽手柄：按住横向拖动，把宽度增量交给主窗口处理。"""

    dragDelta = Signal(float)
    dragFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(_T.SIDEBAR_HANDLE_W)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        # 即树列与内容区的完整边界线（用户定案：衔接处需要一条完整的线）
        self.setStyleSheet(f"background: {theme().get('border')};")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_x = event.globalPosition().x()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press_x is not None:
            delta = event.globalPosition().x() - self._press_x
            self._press_x = event.globalPosition().x()
            self.dragDelta.emit(delta)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._press_x = None
        self.dragFinished.emit()
        event.accept()


class _DragFrame(QFrame):
    """空白区域（childAt 为 None 处）可拖动窗口的 QFrame。

    startSystemMove() 官方 API 优先，失败回退手动位移；双击切换最大化。
    drag_band=True 时作为顶部拖动条使用：交互控件（按钮/下拉/输入框等）
    让位并转发鼠标事件，非交互部件（标签/进度条）区域直接拖窗。
    """

    _drag_from = None
    drag_band = False
    _INTERACTIVE = None
    _fwd = None
    _last_press = None  # (perf_counter ms, 全局 QPoint)：双击自判用

    def __init__(self, parent=None):
        super().__init__(parent)
        # 自定义 QFrame 子类：QSS background 生效前提。缺它=透明带，
        # 顶带显示的是背后 _central 的底色——换肤「切不动」的根因
        # （上一轮误加到 DragRootWidget，本类漏配，2026-09-29 实测）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _is_double_press(self, event) -> bool:
        """startSystemMove 的原生拖窗会话会打断系统双击事件序列（第二次
        点击常以普通 Press 送达），双击改由按下序列自判：间隔 ≤ 系统双击
        间隔且位移 ≤ 20px 视为第二次按下。"""
        from PySide6.QtWidgets import QApplication
        import time as _t
        now = _t.perf_counter() * 1000.0
        gp = event.globalPosition().toPoint()
        last = _DragFrame._last_press
        _DragFrame._last_press = (now, gp)
        if last is None:
            return False
        interval = QApplication.styleHints().mouseDoubleClickInterval()
        return (now - last[0]) <= interval and (gp - last[1]).manhattanLength() <= 20

    def _toggle_maximized(self):
        w = self.window()
        w.showNormal() if w.isMaximized() else w.showMaximized()

    def _interactive_under(self, gpos):
        if _DragFrame._INTERACTIVE is None:
            from PySide6.QtWidgets import (
                QAbstractButton, QComboBox, QAbstractItemView,
                QLineEdit, QTextEdit, QSpinBox, QSlider,
            )
            _DragFrame._INTERACTIVE = (
                QAbstractButton, QComboBox, QAbstractItemView,
                QLineEdit, QTextEdit, QSpinBox, QSlider,
            )
        from PySide6.QtWidgets import QApplication
        self.hide()
        try:
            under = QApplication.widgetAt(gpos)
        finally:
            self.show()
        if under is None or isinstance(under, _DragFrame._INTERACTIVE):
            return under
        # 自绘点击部件（如统计卡 StatCard：QFrame+clicked 信号）不在
        # _INTERACTIVE 名单里，但同样要让位转发——否则其顶部会被当拖窗区。
        if getattr(under, "clicked", None) is not None and not isinstance(under, _DragFrame):
            return under
        return None

    def _clone_event(self, event, target):
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtCore import QPointF
        gp = event.globalPosition()
        local = QPointF(target.mapFromGlobal(gp.toPoint()))
        return QMouseEvent(
            event.type(), local, gp, event.button(), event.buttons(),
            event.modifiers(),
        )

    def mousePressEvent(self, event):
        if self.drag_band and event.button() == Qt.MouseButton.LeftButton:
            from PySide6.QtWidgets import QApplication
            under = self._interactive_under(event.globalPosition().toPoint())
            if under is not None:
                _DragFrame._fwd = under
                QApplication.sendEvent(under, self._clone_event(event, under))
            else:
                # 无交互控件（空白区/已忽略事件的容器）→ 拖窗。
                # 注意不能用 childAt 判空：带内子容器（face_bar 等）会盖住
                # 空白区使 childAt 恒非空——传播到本层即说明没有控件认领。
                if self._is_double_press(event):
                    _DragFrame._last_press = None
                    self._toggle_maximized()
                    event.accept()
                    return
                self._begin_move(event)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.childAt(
            event.position().toPoint()
        ) is None:
            if self._is_double_press(event):
                _DragFrame._last_press = None
                self._toggle_maximized()
                event.accept()
                return
            self._begin_move(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def _begin_move(self, event):
        w = self.window()
        if not w.isMaximized():
            import time as _t
            _t0 = _t.perf_counter()
            wh = w.windowHandle()
            moved = wh is not None and wh.startSystemMove()
            print(f"[PERF] band startSystemMove: {(_t.perf_counter()-_t0)*1000:.0f}ms moved={moved}")
            if not moved:
                self._drag_from = (
                    event.globalPosition().toPoint()
                    - w.frameGeometry().topLeft()
                )

    def mouseMoveEvent(self, event):
        if self.drag_band and getattr(self, "_fwd", None) is not None:
            from PySide6.QtWidgets import QApplication
            QApplication.sendEvent(self._fwd, self._clone_event(event, self._fwd))
            event.accept()
            return
        offset = getattr(self, "_drag_from", None)
        if offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            w = self.window()
            if not w.isMaximized():
                w.move(event.globalPosition().toPoint() - offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_band and getattr(self, "_fwd", None) is not None:
            from PySide6.QtWidgets import QApplication
            QApplication.sendEvent(self._fwd, self._clone_event(event, self._fwd))
            if not event.buttons():
                _DragFrame._fwd = None
            event.accept()
            return
        if getattr(self, "_drag_from", None) is not None:
            self._drag_from = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drag_band:
            under = self._interactive_under(event.globalPosition().toPoint())
            if under is not None:
                return  # 交互控件上的双击留给控件，不触发最大化
        if self.childAt(event.position().toPoint()) is None:
            w = self.window()
            w.showNormal() if w.isMaximized() else w.showMaximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class DragRootWidget(QWidget):
    """10 号结构的中央根部件。

    拖动采用官方 API 组合（2026-09-23）：
    - Qt 层空白区域：QWindow.startSystemMove()（Qt 官方接口，macOS 内部
      走 AppKit 窗口拖动），失败时回退手动位移（1.2.71 真机验证过的老逻辑）；
    - 标题栏带顶部（y<28）：Cocoa accessory 拖动条（_TitlebarDragView，
      NSWindow.performDrag(with:)，Apple 官方自定义拖动区接口）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_from = None
        # 自定义 QFrame 子类：QSS background 生效前提（缺它换肤后仍画旧色
        # ——浅色主题顶带残留深色的根因，2026-09-29 实测）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.childAt(
            event.position().toPoint()
        ) is None:
            w = self.window()
            if not w.isMaximized():
                import time as _t
                _t0 = _t.perf_counter()
                wh = w.windowHandle()
                moved = wh is not None and wh.startSystemMove()
                print(f"[PERF] startSystemMove: {(_t.perf_counter()-_t0)*1000:.0f}ms moved={moved}")
                if not moved:
                    self._drag_from = (
                        event.globalPosition().toPoint()
                        - w.frameGeometry().topLeft()
                    )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        offset = getattr(self, "_drag_from", None)
        if offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            w = self.window()
            if not w.isMaximized():
                w.move(event.globalPosition().toPoint() - offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, "_drag_from", None) is not None:
            self._drag_from = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.childAt(event.position().toPoint()) is None:
            w = self.window()
            w.showNormal() if w.isMaximized() else w.showMaximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def _text_glyph_icon(glyph: str, color: str):
    """把字符（如 ‹ ›）渲染成单色 QIcon——供 QPushButton 文字绘制不可靠
    场景使用（见 cap_fold_btn 注释）。超采样 4 倍抗锯齿。"""
    from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
    pm = QPixmap(56, 56)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(color))
    f = QFont()
    f.setPixelSize(42)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    return QIcon(pm)


def _rail_icon(kind: str, color: str):
    """QPainter 单色矢量图标（10 号方案 §4，对照 SF Symbols 造型）。

    返回 QIcon。88px 超采样画布、按 44 坐标系绘制（p.scale(2)），由
    QIcon 缩到 setIconSize(22,22)——不使用 QPixmap devicePixelRatio
    元数据（实测在本机图标管线会被当作 22px 画布裁掉右下四分之三）。
    """
    from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QPainterPath
    from PySide6.QtCore import QPointF, QRectF

    pm = QPixmap(88, 88)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2.0, 2.0)
    # 图标填满高亮框（用户定案）：字形原留白约一圈，围绕画布中心放大 1.3 倍
    p.translate(22, 22)
    p.scale(1.3, 1.3)
    p.translate(-22, -22)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    c = QPointF(22, 22)
    if kind == "list":  # 任务 list.bullet
        for y in (15, 22, 29):
            p.drawLine(QPointF(12.6, y), QPointF(14.4, y))
            p.drawLine(QPointF(18, y), QPointF(31, y))
    elif kind == "grid":  # 文件详情 square.grid.2x2
        for x in (12.5, 23.5):
            for y in (12.5, 23.5):
                p.drawRoundedRect(QRectF(x, y, 8, 8), 2, 2)
    elif kind == "tag":  # 标签管理 tag
        path = QPainterPath()
        path.moveTo(20.5, 11.5)
        path.lineTo(30.5, 11.5)
        path.quadTo(32.5, 11.5, 32.5, 13.5)
        path.lineTo(32.5, 23.5)
        path.quadTo(32.5, 25.5, 30.5, 25.5)
        path.lineTo(20.5, 25.5)
        path.lineTo(11.5, 18.5)
        path.closeSubpath()
        p.drawPath(path)
        p.drawPoint(QPointF(28, 18.5))
    elif kind == "eye":  # 标签浏览 eye
        path = QPainterPath()
        path.moveTo(11, 22)
        path.quadTo(22, 12.5, 33, 22)
        path.quadTo(22, 31.5, 11, 22)
        p.drawPath(path)
        p.drawEllipse(c, 4, 4)
    elif kind == "face":  # 人脸管理 face.smiling
        p.drawEllipse(c, 10, 10)
        p.drawPoint(QPointF(18, 19))
        p.drawPoint(QPointF(26, 19))
        path = QPainterPath()
        path.moveTo(17, 25.5)
        path.quadTo(22, 29.5, 27, 25.5)
        p.drawPath(path)
    elif kind == "gear":  # 设置：MediaTag 自绘齿轮——齿形轮廓连体填充+挖中孔。
        # 旧版（齿块+内盘圆 OddEven）齿与内盘只在 r_root 重合边相接，
        # 视觉上是"圆盘+漂浮方块"（用户 2026-09-28 判定不像齿轮）。
        # 本版参数与齿形均为自有设计，无第三方出处、无署名义务。
        import math
        from PySide6.QtGui import QPolygonF
        teeth = 8
        r_out, r_root, r_hole = 10.2, 7.0, 3.6
        step = 2 * math.pi / teeth
        half_tooth = step * 0.22
        half_root = step * 0.34
        pts = []
        for i in range(teeth):
            a = i * step
            for ang, r in ((a - half_root, r_root), (a - half_tooth, r_out),
                           (a + half_tooth, r_out), (a + half_root, r_root)):
                pts.append(QPointF(c.x() + r * math.cos(ang), c.y() + r * math.sin(ang)))
        gear = QPainterPath()
        gear.addPolygon(QPolygonF(pts))
        gear.closeSubpath()
        hole = QPainterPath()
        hole.addEllipse(c, r_hole, r_hole)
        body = gear.subtracted(hole)   # 连体齿形，只挖中孔——不再用内盘圆掏空身体
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        p.drawPath(body)

    elif kind == "moon":  # 外观 moon（饱满新月：大圆减偏移圆，填充）
        moon = QPainterPath()
        moon.addEllipse(QPointF(22, 22), 9.5, 9.5)
        cut = QPainterPath()
        cut.addEllipse(QPointF(28.5, 17.5), 8.2, 8.2)
        body = moon.subtracted(cut)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        p.drawPath(body)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pen_r = QPen(QColor(color), 1.6)
        pen_r.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen_r.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen_r)

    elif kind == "sun":  # 外观 sun（中心圆 + 8 射线，对照参考样式）
        import math
        p.drawEllipse(c, 4.2, 4.2)
        for i in range(8):
            a = math.pi / 4 * i
            p.drawLine(QPointF(c.x() + 6.6 * math.cos(a), c.y() + 6.6 * math.sin(a)),
                       QPointF(c.x() + 9.6 * math.cos(a), c.y() + 9.6 * math.sin(a)))

    p.end()
    return QIcon(pm)


RAIL_W = 60  # 导航条宽度（12 号定稿 48 → 用户要求逐轮放大至 60）




class ModuleRail(_DragFrame):
    """模块图标侧栏（10 号方案 §4）：48px 图标化 + 红绿灯顶部一体 + 齿轮落底部。"""

    module_clicked = Signal(int)
    appearance_toggle = Signal()

    # (icon kind, 名称, 页索引)——顶部五项；设置(4)固定在底部（VS Code 惯例）
    MODULES = [
        ("list", "任务", 0),
        ("grid", "文件详情", 1),
        ("tag", "标签管理", 2),
        ("eye", "标签浏览", 5),
        ("face", "人脸管理", 3),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(RAIL_W)
        t = theme()
        self.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_sidebar')}; "
            f"border-right: 1px solid {t.get('border')}; }}"
        )

        v = QVBoxLayout(self)
        # 顶部 30pt = 红绿灯下缘（原生灯占 y8-26），首钮直接贴上不留白
        # （用户定案：第一枚图标上方不保留空白）
        v.setContentsMargins(0, 8, 0, 10)  # 首钮与树列顶部对齐（上对齐，2026-09-28）
        v.setSpacing(6)

        self._buttons = []
        for kind, name, idx in self.MODULES:
            b = self._make_button(kind, name, idx)
            v.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
            self._buttons.append((idx, kind, b))

        v.addStretch()

        self.btn_appearance = self._make_button("moon", "切换外观（深色/浅色）", None)
        self.btn_appearance.clicked.connect(self.appearance_toggle.emit)
        v.addWidget(self.btn_appearance, 0, Qt.AlignmentFlag.AlignHCenter)

        self.btn_settings = self._make_button("gear", "设置", 4)
        v.addWidget(self.btn_settings, 0, Qt.AlignmentFlag.AlignHCenter)
        self._buttons.append((4, "gear", self.btn_settings))

        self._refresh_button_styles()

    def _make_button(self, kind: str, tip: str, idx):
        b = QPushButton()
        b.setCheckable(True)
        b.setFixedSize(54, 50)
        b.setIconSize(QSize(46, 46))
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        if idx is not None:
            b.clicked.connect(lambda _=False, i=idx: self.module_clicked.emit(i))
        return b

    def _refresh_button_styles(self):
        t = theme()
        normal = t.get("text_secondary")
        active = t.get("accent")
        for idx, kind, b in self._buttons:
            b.setIcon(_rail_icon(kind, active if b.isChecked() else normal))
        # 主题 mode 标记可能缺失——按背景亮度稳健判断深浅
        from PySide6.QtGui import QColor as _QC
        _is_dark = _QC(t.get("bg_primary", "#FFFFFF")).lightness() < 128
        self.btn_appearance.setIcon(_rail_icon("moon" if _is_dark else "sun", normal))
        hover_bg = t.get("bg_hover")
        qss = f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 8px;
            }}
            QPushButton:hover {{ background: {hover_bg}; }}
            QPushButton:checked {{ background: {t.get('table_selected')}; }}
        """
        for _, _, b in self._buttons:
            b.setStyleSheet(qss)
        self.btn_appearance.setStyleSheet(qss)

    # 空白区拖动/双击最大化逻辑继承自 _DragFrame

    def set_current(self, index: int):
        for idx, kind, b in self._buttons:
            b.setChecked(idx == index)
        self._refresh_button_styles()

    def update_theme(self):
        self._refresh_button_styles()


class ModelDownloadDialog(QDialog):
    """Dialog for displaying model download progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型下载")
        # 非模态: 下载期间用户仍可操作主窗口(切换页面/查看其他内容),
        # 而不是被对话框锁死。下载进度通过状态栏全局可见。
        self.setModal(False)
        self.setMinimumWidth(500)

        t = theme()
        self.setStyleSheet(f"""
            QDialog {{
                background: {t.get('bg_primary')};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        self.title_label = QLabel("正在下载 MiniCPM-V 模型")
        self.title_label.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 600;
            color: {t.get('text_primary')};
        """)
        layout.addWidget(self.title_label)

        # Status label
        self.status_label = QLabel("准备中...")
        self.status_label.setStyleSheet(f"""
            font-size: 13px;
            color: {t.get('text_secondary')};
        """)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {t.get('bg_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                text-align: center;
                font-size: 12px;
                color: {t.get('text_primary')};
                height: 24px;
            }}
            QProgressBar::chunk {{
                background: {t.get('bg_hover')};
                border-radius: {t.get('radius_md')}px;
            }}
        """)
        self.progress_bar.setFixedHeight(24)
        layout.addWidget(self.progress_bar)

        # Details section
        details_widget = QWidget()
        details_layout = QHBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(24)

        # Downloaded size
        self.downloaded_label = QLabel("已下载: --")
        self.downloaded_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.downloaded_label)

        # Total size
        self.total_label = QLabel("总计: --")
        self.total_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.total_label)

        # Speed
        self.speed_label = QLabel("速度: --")
        self.speed_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.speed_label)

        details_layout.addStretch()
        layout.addWidget(details_widget)

        # Retry info
        self.retry_label = QLabel("")
        self.retry_label.setStyleSheet(f"""
            font-size: 11px;
            color: {t.get('warning')};
            font-style: italic;
        """)
        self.retry_label.setWordWrap(True)
        layout.addWidget(self.retry_label)

        layout.addStretch()

        # Button box
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {t.get('text_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        button_layout.addWidget(self.cancel_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                padding: 6px 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        self.close_btn.setVisible(False)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Connect signals
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.close_btn.clicked.connect(self.accept)

        self._cancelled = False

    def _on_cancel(self):
        """Handle cancel button click."""
        self._cancelled = True
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("正在取消...")
        self.status_label.setText("正在取消下载...")

    def closeEvent(self, event):
        """关闭窗口 = 隐藏对话框, 不取消下载。

        下载继续在后台进行, 进度始终显示在主窗口状态栏;
        完成后会弹「下载完成」提示, 用户可随时重新打开/查看。
        """
        event.ignore()
        self.hide()

    def is_cancelled(self) -> bool:
        """Check if download was cancelled by user."""
        return self._cancelled

    def update_progress(self, progress):
        """Update progress display from DownloadProgress object."""
        # percent=0 的状态行(manifest/verify 等)不动进度条, 避免打回 0
        if progress.percent > 0:
            self.progress_bar.setValue(int(progress.percent))

        # Update status
        if progress.status and progress.status not in ("pulling manifest", "verifying", "writing"):
            self.status_label.setText(progress.status)

        # Update details
        if progress.downloaded and progress.total:
            self.downloaded_label.setText(f"已下载: {progress.downloaded}")
            self.total_label.setText(f"总计: {progress.total}")

        if progress.speed:
            self.speed_label.setText(f"速度: {progress.speed}")

    def set_completed(self, success: bool, message: str):
        """Set dialog to completed state."""
        if success:
            self.title_label.setText("下载完成")
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("完成")

            # Change accent color to success
            t = theme()
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {t.get('bg_secondary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_md')}px;
                    text-align: center;
                    font-size: 12px;
                    color: {t.get('text_primary')};
                    height: 24px;
                }}
                QProgressBar::chunk {{
                    background: {t.get('bg_hover')};
                    border-radius: {t.get('radius_md')}px;
                }}
            """)
        else:
            self.title_label.setText("下载失败")
            self.status_label.setText(message)

            # Change accent color to danger
            t = theme()
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {t.get('bg_secondary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_md')}px;
                    text-align: center;
                    font-size: 12px;
                    color: {t.get('text_primary')};
                    height: 24px;
                }}
                QProgressBar::chunk {{
                    background: {t.get('bg_hover')};
                    border-radius: {t.get('radius_md')}px;
                }}
            """)

        # Show close button, hide cancel button
        self.cancel_btn.setVisible(False)
        self.close_btn.setVisible(True)

    def set_retry_info(self, attempt: int, max_retries: int):
        """Show retry information."""
        if attempt > 1:
            t = theme()
            self.retry_label.setText(f"正在重试（第 {attempt} 次，共 {max_retries} 次）...")
            self.retry_label.setStyleSheet(f"""
                font-size: 11px;
                color: {t.get('warning')};
                font-style: italic;
            """)
        else:
            self.retry_label.setText("")


class _PixelRulerOverlay(QWidget):
    """调试像素标尺（⌥P 切换）：真机渲染上叠加边界线与坐标标注。

    三条虚线：镜片行底(橙)/树视口顶(青)/树首项顶(绿)，线上标真实 y 值，
    镜片行底与首项顶之间直接标注缝隙像素数。用户肉眼指认空隙归属，
    不经探针转述。仅调试用。
    """

    def __init__(self, main_win):
        super().__init__(main_win)
        self._mw = main_win
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(main_win.rect())
        self.hide()

    def refresh(self):
        self.setGeometry(self._mw.rect())
        self.update()

    def paintEvent(self, ev):
        from PySide6.QtGui import QPainter, QPen, QFont, QColor
        d = getattr(self._mw, "_dashboard", None)
        if d is None or not hasattr(d, "_lens_row"):
            return
        p = QPainter(self)
        p.setFont(QFont("Menlo", 12))

        lens = d._lens_row
        lens_b = lens.mapTo(self._mw, lens.rect().topLeft()).y() + lens.height()
        st = d.source_tree
        vp_y = st.viewport().mapTo(self._mw, st.viewport().rect().topLeft()).y()
        it0 = st.topLevelItem(0)
        first_y = vp_y + st.visualItemRect(it0).y() if it0 is not None else -1

        def hline(y, color, label):
            if y < 0:
                return
            p.setPen(QPen(QColor(color), 1, Qt.PenStyle.DashLine))
            p.drawLine(0, y, self.width(), y)
            p.setPen(QColor(color))
            p.drawText(6, y - 4, f"{label}  y={y}")

        hline(lens_b, "#FF9F0A", "镜片行底")
        hline(vp_y, "#64D2FF", "树视口顶")
        hline(first_y, "#30D158", "首项顶")
        if first_y >= 0:
            p.setPen(QColor("#FFFFFF"))
            p.drawText(6, lens_b + 18, f"缝隙 = {first_y - lens_b}px")
        p.end()


class MainWindow(QMainWindow):
    # 连接状态探测在后台线程执行，经信号回主线程刷新（不阻塞 UI）
    status_checked = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediaTag")
        # 根参数钉板（2026-09-28）：默认 1440×900、最小 1152×720、记忆上次大小位置
        self.setMinimumSize(1152, 720)
        from PySide6.QtCore import QByteArray as _QBA
        from PySide6.QtCore import QSettings as _QWin
        _geo = _QWin().value("ui/window_geometry", "", type=str)
        if _geo and not self.restoreGeometry(_QBA.fromBase64(_geo.encode("ascii"))):
            self.setGeometry(100, 100, 1440, 900)
        if not _geo:
            self.setGeometry(100, 100, 1440, 900)
        
        # 设置应用图标
        icon_path = "assets/app_icon.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Apply global stylesheet
        self.setStyleSheet(global_stylesheet())

        # ── macOS 菜单栏（主导航已在菜单栏，08 方案）─────────────────────
        menubar = self.menuBar()

        # 应用菜单合并项（AboutRole/ApplicationSpecificRole 自动并入
        # 屏幕左上角 MediaTag 菜单；容器菜单合并后自动隐藏）
        about_menu = menubar.addMenu("关于")
        about_action = about_menu.addAction("关于")
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self._show_about)
        # 授权管理（应用菜单入口，docs/界面设计/07-授权管理.md §3）
        license_action = about_menu.addAction("授权管理…")
        license_action.setMenuRole(QAction.MenuRole.ApplicationSpecificRole)
        license_action.triggered.connect(self._open_license_manager)

        # 文件
        m_file = menubar.addMenu("文件")
        for text, slot in (
            ("添加文件…", "add_files"),
            ("添加文件夹…", "add_folder"),
            ("从照片导入…", "import_from_photos"),
        ):
            act = QAction(text, self)
            act.triggered.connect(
                lambda _=False, s=slot: getattr(self._dashboard, s)()
            )
            m_file.addAction(act)
        m_file.addSeparator()
        act_close = QAction("关闭窗口", self)
        act_close.setShortcut(QKeySequence.Close)
        act_close.triggered.connect(self.close)
        m_file.addAction(act_close)

        # 编辑（标准角色——文本框撤销/剪贴板正常工作的系统前提）
        m_edit = menubar.addMenu("编辑")
        edit_items = (
            ("撤销", "undo", QKeySequence.Undo),
            ("重做", "redo", QKeySequence.Redo),
            ("剪切", "cut", QKeySequence.Cut),
            ("拷贝", "copy", QKeySequence.Copy),
            ("粘贴", "paste", QKeySequence.Paste),
            ("全选", "selectAll", QKeySequence.SelectAll),
        )
        for text, slot, seq in edit_items:
            act = QAction(text, self)
            act.setShortcut(seq)
            act.triggered.connect(lambda _=False, s=slot: self._route_edit(s))
            m_edit.addAction(act)

        # 前往菜单（09 方案：页面顶级菜单收敛；可视主导航在图标侧栏，
        # ⌘1-6 快捷键的可见入口保留在此）
        m_go = menubar.addMenu("前往")
        go_items = (
            ("前往任务", 0, 1),
            ("前往文件详情", 1, 2),
            ("前往标签管理", 2, 3),
            ("前往标签浏览", 5, 4),
            ("前往人脸管理", 3, 5),
            ("前往设置", 4, 6),
        )
        for text, page_index, seq_n in go_items:
            act = QAction(text, self)
            act.setShortcut(QKeySequence(f"Ctrl+{seq_n}"))  # Qt 的 Ctrl 在 macOS 即 ⌘
            act.triggered.connect(lambda _=False, i=page_index: self._switch_page(i))
            m_go.addAction(act)

        # 窗口
        m_window = menubar.addMenu("窗口")
        act_min = QAction("最小化", self)
        act_min.setShortcut(QKeySequence("Ctrl+M"))
        act_min.triggered.connect(self.showMinimized)
        m_window.addAction(act_min)
        act_zoom = QAction("缩放", self)
        act_zoom.triggered.connect(
            lambda: self.showNormal() if self.isMaximized() else self.showMaximized()
        )
        m_window.addAction(act_zoom)

        # 帮助
        m_help = menubar.addMenu("帮助")
        act_notice = QAction("第三方组件声明", self)
        act_notice.triggered.connect(self._show_third_party)
        m_help.addAction(act_notice)
        act_feedback = QAction("常见问题与反馈…", self)
        act_feedback.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("mailto:support@cloudcraft.xin"))
        )
        m_help.addAction(act_feedback)
        act_home = QAction("访问官网…", self)
        act_home.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://cloudcraft.xin"))
        )
        m_help.addAction(act_home)

        # Central widget with vertical layout (content + status bar)
        # 10 号方案 VS Code 式三栏：模块图标侧栏 + 统一侧栏树容器 + 内容栈；
        # 底部状态栏改为全宽（root_v 底行），不再只在内容列下方。
        central = DragRootWidget(self)
        central.setStyleSheet(f"background: {theme().get('bg_primary')}; color: {theme().get('text_primary')};")
        # 不走 setCentralWidget：QMainWindowLayout 会按 safeAreaMargins（顶部
        # 28px）把 central 整体下推，形成全宽死带（红绿灯悬空、利用率低）。
        # 改为普通子部件，由 resizeEvent/showEvent 手动铺满窗口（见下）。
        self._central = central
        self._root_v = root_v = QVBoxLayout(central)
        # 底部：只保留 28pt 垫层（方案 B）——此前 32pt 边距与垫层叠加
        # 共 60pt，导航条/树列提前 60pt 截断，观感像"没展示完整"（用户报告）
        # 三列从标题栏（48pt）下方开始——此前三列顶进标题栏底下，
        # 任务页操作栏等顶部内容被顶带盖住“消失”（用户报告）
        self._band_h = 40  # 顶带高度（窄条，类似标题栏）
        root_v.setContentsMargins(0, self._band_h, 0, 0)
        root_v.setSpacing(0)
        central_h = QHBoxLayout()
        central_h.setContentsMargins(0, 0, 0, 0)
        central_h.setSpacing(0)
        root_v.addLayout(central_h, 1)

        self.rail = ModuleRail(central)
        # 像素标尺（⌥P 切换，开发调试）：真实窗口上叠加边界线
        self._pixel_ruler = _PixelRulerOverlay(self)
        from PySide6.QtGui import QShortcut as _QSC
        _sc = _QSC(__import__("PySide6.QtGui", fromlist=["QKeySequence"])
                   .QKeySequence("Alt+P"), self)
        _sc.activated.connect(self._toggle_pixel_ruler)
        # ⌘R 热重载像素 token（ui/tokens.py=唯一数值源，2026-09-28 用户令：
        # 界面数值必须单一来源）。改 tokens.py → ⌘R → 全局 QSS+各页重放。
        _sc_r = _QSC(__import__("PySide6.QtGui", fromlist=["QKeySequence"])
                     .QKeySequence("Ctrl+R"), self)
        _sc_r.activated.connect(self._reload_pixel_tokens)
        # ⌥I 元素检查器（ui/control/inspector.py）：悬停显示 x/y/w/h/margins
        try:
            from .control.inspector import InspectorOverlay as _InspectorOverlay
        except ImportError:
            from ui.control.inspector import InspectorOverlay as _InspectorOverlay
        self._inspector = _InspectorOverlay(self)
        _sc_i = _QSC(__import__("PySide6.QtGui", fromlist=["QKeySequence"])
                     .QKeySequence("Alt+I"), self)
        _sc_i.activated.connect(self._toggle_inspector)
        self.rail.module_clicked.connect(self._switch_page)
        self.rail.appearance_toggle.connect(self._toggle_theme)
        central_h.addWidget(self.rail)

        # 统一侧栏树容器：树本体=QStackedWidget；无树的模块侧栏+热条都隐藏
        # （区域宽度规范：60px 导航条 / 212px 树列 / 内容自适应，可拖 180-320）
        from PySide6.QtCore import QSettings as _QSw
        self._sidebar_w = max(180, min(320, int(_QSw().value("ui/sidebar_width", 212, type=int))))  # 可拖 180-320（钉板）
        self.sidebar = _DragFrame()
        self.sidebar.setFixedWidth(self._sidebar_w)
        self.sidebar.setStyleSheet(
            f"QFrame {{ background: {theme().get('bg_secondary')}; "
            f"border-right: 1px solid {theme().get('border')}; }}"
        )
        sb_v = QVBoxLayout(self.sidebar)
        sb_v.setContentsMargins(0, 0, 0, 0)
        sb_v.setSpacing(0)
        # 树内容顶格贴顶带分隔线（P12①：原 28px 头部占位 sb_cap_gap 已随
        # 头部盖板方案废弃，2026-09-28 移除）；QStackedLayout 默认 9px
        # margin 显式归零，左右呼吸交给树包裹层统一给 8px
        self.sidebar_stack = QStackedWidget()
        self.sidebar_stack.layout().setContentsMargins(0, 0, 0, 0)
        sb_v.addWidget(self.sidebar_stack, 1)
        central_h.addWidget(self.sidebar)

        # 树列拖宽手柄：按住横向拖动 180-360pt，QSettings 持久化
        self.sidebar_handle = _SidebarHandle()
        self.sidebar_handle.dragDelta.connect(self._on_sidebar_drag)
        self.sidebar_handle.dragFinished.connect(self._on_sidebar_drag_done)
        central_h.addWidget(self.sidebar_handle)

        # 收起后的展开热条：**一条线**（用户 2026-09-28：收窄到只有一条线，
        # 不得留宽列）。列宽=3px（与拖宽手柄同宽并成一条分隔线）；
        # ‹ 钮浮出于该线之上（14×24，骑缝居中），位置由 _sync_band 维护。
        self.sidebar_unfold = QFrame()
        self.sidebar_unfold.setFixedWidth(1)
        self.sidebar_unfold.setStyleSheet(
            f"QFrame {{ background: {theme().get('bg_secondary')}; "
            f"border-right: 1px solid {theme().get('border')}; }}"
        )
        self.sidebar_unfold_btn = QPushButton(central)
        self.sidebar_unfold_btn.setFixedSize(14, 24)
        self.sidebar_unfold_btn.setToolTip("展开侧栏")
        self.sidebar_unfold_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar_unfold_btn.setIcon(
            _text_glyph_icon("›", theme().get('text_secondary')))
        self.sidebar_unfold_btn.setIconSize(QSize(14, 14))
        self.sidebar_unfold_btn.setStyleSheet(
            f"QPushButton {{ background: {theme().get('bg_secondary')}; "
            f"border: 1px solid {theme().get('border')}; border-radius: 4px; padding: 0; }}"
        )
        self.sidebar_unfold_btn.clicked.connect(lambda: self._toggle_sidebar(False))
        self.sidebar_unfold.setVisible(False)
        self.sidebar_unfold_btn.setVisible(False)
        # 自动隐藏：鼠标距分隔线 ≤24px 时现身，停用 2.5s 后隐去
        self._unfold_hide_timer = QTimer(self)
        self._unfold_hide_timer.setSingleShot(True)
        self._unfold_hide_timer.setInterval(2500)
        self._unfold_hide_timer.timeout.connect(self._unfold_hide_if_idle)
        self._seam_poll = QTimer(self)
        self._seam_poll.setInterval(400)
        self._seam_poll.timeout.connect(self._poll_seam_hover)
        self._seam_poll.start()
        central_h.addWidget(self.sidebar_unfold)

        # 侧栏登记表：内容页索引 → 侧栏栈索引（切片 3/5 继续登记 1/5 号页）
        self._sidebar_pages = {}
        from PySide6.QtCore import QSettings as _QS
        self._sidebar_collapsed = _QS().value("ui/sidebar_collapsed", False, type=bool)

        # Content area with stacked widget
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet(f"""
            QStackedWidget {{
                background: {theme().get('bg_primary')};
            }}
        """)
        central_h.addWidget(self.content_stack, 1)

        # Status bar at the bottom（10 号方案：全宽）
        # 底部状态栏已删除（12 号定稿·方案 B）：macOS 27 对窗口底部 28pt
        # 底部 28pt 补偿垫层：macOS 27 系统会在窗口底缘强制叠加玻璃
        # （把该区域提亮约一倍），垫层预涂"半亮度"底色，玻璃叠加后
        # 恰与页面背景融合——分页器等控件保持在玻璃区上方不受雾化。
        # （无此垫层时贴底控件会被整体玻璃化，即用户报告的胶囊问题）
# 底部垫层已移除（用户 2026-09-28：无内容却占 28px）；方案 B 的
        # setOpaque_(True) 已消除底缘玻璃胶囊，内容可直铺到底缘



        # 统一标题栏（13 号定稿）：全宽顶带（高 48pt），红绿灯原生浮于左端，
        # 底边 1px 分隔线；中段为「页内容槽」（各页可注入首行控件，当前
        # 仅任务页注册统计行）；右端 = 连接圆点 + 侧栏收起钮。
        # 几何随窗口 _pin_central 维护；显隐/页切换由 _sync_cap 维护。
        self.title_band = _DragFrame(central)
        self.title_band.setObjectName("TitleBand")
        # 顶带拖动：title_band 自身 drag_band=True（子部件下层），带内空白区
        # 直接拖窗；按钮/搜索等子部件拿原生鼠标事件（悬停/点击不再被透明
        # 拖动帧罩住——旧 _drag_top 覆盖层架构致"悬停死区"，2026-10-01 用户
        # 实测裁定拆除）。face_bar/_band_row1/band_gap 设鼠标透明让空隙穿透。
        self.title_band.drag_band = True
        self.title_band.setFixedHeight(self._band_h)
        self.title_band.setStyleSheet(f"""
            QFrame#TitleBand {{
                background: {theme().get('bg_primary')};
                border-bottom: 1px solid {theme().get('border')};
            }}
        """)
        _band_v = QVBoxLayout(self.title_band)
        _band_v.setContentsMargins(0, 0, 12, 0)
        _band_v.setSpacing(0)
        _band_row1 = QWidget()
        tb_h = QHBoxLayout(_band_row1)
        tb_h.setContentsMargins(0, 3, 0, 3)
        tb_h.setSpacing(8)
        _band_v.addWidget(_band_row1)
        # 顶带=全 app 单行 48pt，只允许一行内容（用户裁定 2026-09-27）——
        # 曾试过"行2=每页操作面板堆栈"，已推翻拆除：工具栏一律内容区。
        # 左占位：跳过「导航条+来源树」宽度，统计行仅在内容区内显示
        # （用户定案：统计行不可左移压到来源树上方）
        self.band_gap = QWidget(self.title_band)
        self.band_gap.setFixedHeight(1)
        tb_h.addWidget(self.band_gap)
        self.cap_fold_btn = QPushButton(self.title_band)
        self.cap_fold_btn.setFixedSize(24, 24)  # 17 号规范：带内字形钮 24
        self.cap_fold_btn.setToolTip("收起/展开侧栏")
        self.cap_fold_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # 字形走图标而非 text：macOS 下 QPushButton 的 QSS 文字绘制链路
        # 脆弱（全局表的 border/padding 叠加后文字可整体不画，探针实测），
        # 而图标绘制与 rail 钮同一链路、始终可靠。
        self.cap_fold_btn.setIcon(_text_glyph_icon("‹", theme().get('text_secondary')))
        self.cap_fold_btn.setIconSize(QSize(14, 14))
        self.cap_fold_btn.setStyleSheet(
            f"QPushButton {{ background: {theme().get('bg_primary')}; border: none; "
            f"border-radius: 4px; padding: 0; }}"
            f"QPushButton:hover {{ background: {theme().get('btn_glass_hover')}; }}"
        )
        self.cap_fold_btn.clicked.connect(
            lambda: self._toggle_sidebar(not self._sidebar_collapsed))
        tb_h.addWidget(self.cap_fold_btn)
        tb_h.addStretch()
        # 连接圆点（title_band 右端）
        self.conn_dot = QLabel("● 断开")
        self.conn_dot.setToolTip("API 连接状态")
        self.conn_dot.setStyleSheet(f"""
            font-size: 11px;
            color: {theme().get('danger')};
        """)
        self.conn_dot.hide()
        tb_h.addWidget(self.conn_dot)
        # 页内容槽（各页首行控件注入处，current 由 _sync_band 切换）
        self.band_stack = QStackedWidget(self.title_band)
        self.band_stack.setFixedHeight(36)  # 容 28 控件/32 统计卡（钉板）
        tb_h.insertWidget(1, self.band_stack, 1)

        # Initialize pages (lazy loaded) —— 客单页已随业务范围裁剪移除
        self._pages_built = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False}

        # Page 0: Dashboard (always loaded)
        self._dashboard = DashboardPage(api_base=API_BASE)
        self.content_stack.addWidget(self._dashboard)

        # 10 号方案切片 2：任务来源树迁往主窗口统一侧栏
        _src_panel = self._dashboard.take_source_panel()
        if _src_panel is not None:
            self._register_sidebar_page(0, _src_panel)
        # 13 号：统计行迁往统一标题栏（红绿灯右侧）
        _title_cards = self._dashboard.take_title_cards()
        if _title_cards is not None:
            self.band_stack.addWidget(_title_cards)

        # 其余分区顶带标题（14 号：顶带内容随页切换）
        def _band_label(text):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            lb = QLabel(text)
            lb.setStyleSheet("font-size: 13px; font-weight: 700;")
            v.addWidget(lb)
            return w
        _detail_band = QWidget()
        _db_h = QHBoxLayout(_detail_band)
        _db_h.setContentsMargins(0, 0, 0, 0)
        # 占位即空（保持 band_stack 索引对齐）；构建后由 D6 操作面板换入。
        # 「已选中 N」胶囊经用户裁定撤除（2026-09-27）——内容区工具栏已有
        # 同款筛选钮，顶带不再重复。
        self.band_stack.addWidget(_detail_band)
        # 其余分区顶带占位（14 号：顶带内容随页切换）。
        # ⚠ 顺序必须与页面索引一致：2=标签管理 3=人脸 4=设置 5=标签浏览
        # （曾按错误顺序入栈，标签浏览页顶带错显「设置」——台账 B4 根因）。
        # 各页真正富内容在其 _build_page 分支经 _set_band_page 换入。
        self.band_stack.addWidget(_band_label("标签管理"))   # idx 2
        self.band_stack.addWidget(_band_label("人脸管理"))   # idx 3
        self.band_stack.addWidget(_band_label("设置"))       # idx 4
        self.band_stack.addWidget(_band_label("标签浏览"))   # idx 5


        # 自动化测试钩子（仅显式设环境变量时触发）：启动后自动打开照片导入流程
        if os.environ.get("MEDIATAG_AUTOTEST_PHOTOS") == "1":
            QTimer.singleShot(6000, self._dashboard.import_from_photos)

        # Placeholder pages for lazy loading
        for i in range(1, 6):
            stub = QWidget()
            stub_layout = QVBoxLayout(stub)
            stub_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stub_label = QLabel("加载中...")
            stub_label.setStyleSheet(f"color: {theme().get('text_secondary')}; font-size: 15px;")
            stub_layout.addWidget(stub_label)
            self.content_stack.addWidget(stub)

        # 分析主题入口已随 10 号方案定稿迁往设置页（切片 5 落地），rail 引用移除
        self._refresh_analysis_theme_label()
        self.rail.set_current(0)
        self._update_sidebar_for_page(0)

        self._apply_chrome_theme()

        # Auto-refresh timer for results page
        self._results_timer = QTimer(self)
        self._results_timer.setInterval(10000)  # 10 seconds
        self._results_timer.timeout.connect(self._auto_refresh_results)

        # Status indicator timer
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(5000)  # 5 seconds
        self._status_timer.timeout.connect(self._update_status_indicator)
        self.status_checked.connect(self._on_status_checked)
        self._status_timer.start()

        # TEMP-PERF：主线程卡顿探测器（临时插桩，定位首点转圈后移除）
        import time as _time_mod
        self._perf_last = _time_mod.perf_counter()
        self._perf_timer = QTimer(self)
        self._perf_timer.setInterval(500)

        def _perf_tick():
            now = _time_mod.perf_counter()
            lag = (now - self._perf_last - 0.5) * 1000
            if lag > 800:
                print(f"[PERF] 主线程卡顿 {lag:.0f}ms")
            self._perf_last = now

        self._perf_timer.timeout.connect(_perf_tick)
        self._perf_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self._pin_central()
        self._reapply_titlebar_appearance()
        if not getattr(self, "_fused_chrome_applied", False):
            self._fused_chrome_applied = True
            QTimer.singleShot(0, self._apply_chrome)
        # 启动期布局漂移会在侧栏顶部留下未重绘残影（树列顶端蓝线），
        # 稳定后补一次全窗重绘清掉（TEMP-DBG 探针实验证实）
        for _ms in (1200, 3000, 6000):
            QTimer.singleShot(_ms, self._settle_repaint)
        if os.environ.get("MEDIATAG_PROBE_BAND") == "1":
            QTimer.singleShot(4000, self._probe_band_tree)
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Alt+P"), self, self._toggle_pixel_ruler)
        if os.environ.get("MEDIATAG_GRAB_SIDEBAR") == "1":
            QTimer.singleShot(5000, lambda: self._grab_sidebar_dbg())

    def _grab_sidebar_dbg(self):
        """TEMP-DBG：真机侧栏几何打印（镜片行/树视口/首项精确坐标）。"""
        try:
            d = self._dashboard
            if d is None or d._lens_row is None:
                print("[SIDEBAR-GEO] dashboard 未构建", flush=True)
                return
            lens = d._lens_row
            lb = lens.mapTo(self, lens.rect().topLeft()).y() + lens.height()
            st = d.source_tree
            vp_y = st.viewport().mapTo(self, st.viewport().rect().topLeft()).y()
            it0 = st.topLevelItem(0)
            first = (vp_y + st.visualItemRect(it0).y()) if it0 is not None else -1
            n_items = st.topLevelItemCount()
            print(f"[SIDEBAR-GEO] 镜片行底={lb} 树视口顶={vp_y} "
                  f"首项y={first} 项数={n_items} "
                  f"几何缝隙={first - lb if first >= 0 else '树空'}", flush=True)
            pm = self.grab()
            pm.copy(0, 0, 480, 260).save("/tmp/mediatag_probe/live_sidebar_1x.png")
            pm.save("/tmp/mediatag_probe/live_full.png")
            print("[GRAB-SIDEBAR] saved", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("[GRAB-SIDEBAR] fail:", e, flush=True)

    def _probe_band_tree(self):
        """TEMP-DBG：顶带控件树全量打印（定位统计卡残留）。"""
        from PySide6.QtWidgets import QWidget

        def walk(w, depth=0):
            g = w.geometry()
            txt = ""
            try:
                txt = (w.text() or "")[:12]
            except Exception:
                pass
            print(f"[BAND-TREE]{'  ' * depth}{type(w).__name__}"
                  f" obj={w.objectName() or '-'}"
                  f" geo=({g.x()},{g.y()},{g.width()},{g.height()})"
                  f" vis={w.isVisible()} txt={txt!r}", flush=True)
            for c in w.children():
                if isinstance(c, QWidget):
                    walk(c, depth + 1)
        walk(self.title_band)

    def _settle_repaint(self):
        self.update()
        for c in self.children():
            if isinstance(c, QWidget):
                c.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._pin_central()

    def _pin_central(self):
        """把 _central 手动铺满窗口内容视图，绕过 safeAreaMargins 下推。"""
        c = getattr(self, "_central", None)
        if c is not None:
            c.setGeometry(self.rect())
        band = getattr(self, "title_band", None)
        if band is not None:
            band.setGeometry(0, 0, self.width(), 48)
        # 13 号：header_cap 已废，统一标题栏每次几何变化都要同步
        # （否则 band_gap 停留在 init 时的 RAIL_W——窗口显示前侧栏
        # 尚不可见，统计行会压到来源树上方）
        self._sync_band()

    def _apply_chrome(self):
        """09 三段结构：全尺寸内容视图 + 透明标题栏 + 隐藏标题。

        红绿灯悬浮在模块图标侧栏顶部（不再使用 NSToolbar——紧凑位置即
        侧栏顶部区）。窗口拖动：侧栏/标题条空白区已接手动位移拖动。
        """
        try:
            from PySide6.QtWidgets import QApplication as _QA2
            if _QA2.instance().platformName() == "offscreen":
                return  # offscreen 探针无真实 NSView，objc 消息会段错误
            import objc

            nsview = objc.objc_object(c_void_p=int(self.winId()))
            win = nsview.window()
            if win is None or _AppKit is None:
                return
            mask = win.styleMask()
            mask |= _AppKit.NSWindowStyleMaskFullSizeContentView
            win.setStyleMask_(mask)

            win.setTitlebarAppearsTransparent_(True)
            win.setTitleVisibility_(_AppKit.NSWindowTitleHidden)
            # 方案 B（2026-09-24 定稿）：窗口标记不透明——系统不再对底部
            # 区域做玻璃合成（此前状态栏文字被雾化）；Qt 各区域均实色绘制，
            # 无视觉损失。窗口背景同步为主题底色，避免 resize 闪白。
            win.setOpaque_(True)
            # 透明标题栏会让系统在窗口底部自动画"分隔/玻璃"胶囊带（探针实测
            # 存在 setTitlebarDrawsBottomDividerWhenTransparent_）——显式关闭，
            # 去掉底部特殊效果（用户定案）。
            try:
                if hasattr(win, "setTitlebarDrawsBottomDividerWhenTransparent_"):
                    win.setTitlebarDrawsBottomDividerWhenTransparent_(False)
            except Exception:
                pass
            try:
                if hasattr(win, "setShowsContentSeparator_forEdge_"):
                    win.setShowsContentSeparator_forEdge_(False, 3)
            except Exception:
                pass

            try:
                _c = QColor(theme().get('bg_primary'))
                win.setBackgroundColor_(
                    _AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                        _c.redF(), _c.greenF(), _c.blueF(), 1.0))
            except Exception:
                pass
            # 10 号方案：原生背景拖动兜底（setMovableByWindowBackground）+
            # Qt 侧 title_band 自身 drag_band 拖窗（旧 _drag_top 覆盖层已于
            # 2026-10-01 拆除——罩住带内控件造成悬停/点击死区）。
            # 曾尝试 NSTitlebarAccessoryViewController 承接标题栏
            # 拖动，实测 AppKit 命中测试把顶部带点击判给 Qt 内容视图
            # （QNSView），accessory 收不到事件——已移除，勿再加回。
            win.setMovableByWindowBackground_(True)
            # 说明：macOS 27 窗口背面是系统玻璃（_glassWindowBackingGlassView），
            # Qt 侧不透明绘制的区域不会透出玻璃。底部状态栏须保持实色（见
            # status_bar 的 palette+QSS 双保险），勿给其加透明/圆角 QSS。

            # 玻璃顶栏试点（NSGlassEffectView + 原生 glass 折叠钮）已随
            # 13 号统一标题栏作废：48pt Qt 顶带占满顶部，原生玻璃条叠在
            # 其上半部会挡住带内控件命中。收起钮 = title_band 内 Qt 钮。
        except Exception as exc:
            import logging, traceback
            logging.getLogger(__name__).warning(
                "窗口 chrome 应用失败: %s\n%s", exc, traceback.format_exc()
            )

    def changeEvent(self, event):
        super().changeEvent(event)
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            # macOS: 窗口激活时系统会按系统外观重置标题栏(系统浅色→标题栏白),
            # 启动时的 apply_palette 早于此。下一拍重放外观，与手动切换主题同路径。
            QTimer.singleShot(0, self._reapply_titlebar_appearance)

    def _reapply_titlebar_appearance(self):
        """重放主题外观，让 macOS 标题栏跟随主题（而非系统外观）。"""
        import time as _t
        _t0 = _t.perf_counter()
        try:
            theme().apply_palette()
        except Exception:
            pass
        print(f"[PERF] apply_palette: {(_t.perf_counter()-_t0)*1000:.0f}ms")

    def _show_about(self):
        """显示关于对话框"""
        try:
            from .components.about_dialog import AboutDialog
        except ImportError:
            # 脚本模式直接运行时(__main__)相对导入不可用,
            # 此时 src 已在 sys.path, 用 ui.components 包路径导入。
            from ui.components.about_dialog import AboutDialog
        dialog = AboutDialog(self)
        dialog.exec()

    def license_mode_ready(self, resolution):
        """授权模式探测完成（后台线程 → 主线程队列回调）。

        resolution: licensing.gate.ModeResolution；被动取消激活等场景
        弹一次性非阻断提示，其余静默。
        """
        notice = getattr(resolution, "notice", None)
        if notice:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "授权", notice)

    def _open_license_manager(self):
        """应用菜单「授权管理…」→ 授权管理窗口。"""
        try:
            from licensing.manage_window import LicenseManageDialog
        except ImportError:
            from src.licensing.manage_window import LicenseManageDialog
        LicenseManageDialog(self).exec()

    def _route_edit(self, slot: str):
        """编辑菜单动作 → 当前焦点控件的对应槽（撤销/剪切/拷贝/粘贴/全选）。"""
        from PySide6.QtWidgets import QApplication
        w = QApplication.focusWidget()
        if w is not None and hasattr(w, slot):
            try:
                getattr(w, slot)()
            except Exception:
                pass

    def _show_third_party(self):
        """帮助菜单 → 第三方组件声明弹窗。"""
        try:
            from .components.about_dialog import ThirdPartyDialog
        except ImportError:
            from ui.components.about_dialog import ThirdPartyDialog
        ThirdPartyDialog(self).exec()

    def update_task_stats(self, text: str):
        """兼容接口：底部状态栏已删除（方案 B），统计文字不再展示。"""
        # 任务页自身的选中/统计由页面内控件表达；此处保留空实现避免
        # dashboard 调用点崩潰。

    # ── 统一侧栏（10 号方案 §3/§4）──────────────────────────────
    _SIDEBAR_TITLES = {0: "", 1: "", 2: "", 5: ""}  # 说明性文字一律不显示（用户定案：只删文字不删组件）

    def _register_sidebar_page(self, page_index: int, widget):
        """登记某内容页的侧栏树；未登记的模块切过去时侧栏自动收起。"""
        self._sidebar_pages[page_index] = self.sidebar_stack.addWidget(widget)

    def _apply_chrome_theme(self):
        """窗口级部件主题重涂（单一维护点）：切外观与构建完成时各调一次。

        此前导航条/侧栏/热条/头部盖板的背景在构造时写死，切浅色不跟随
        （用户报告）。页面级 apply_theme 链路不变。"""
        t = theme()
        self.rail.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_sidebar')}; "
            f"border-right: 1px solid {t.get('border')}; }}")
        self.sidebar.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_secondary')}; "
            f"border-right: 1px solid {t.get('border')}; }}")
        self.sidebar_unfold.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_secondary')}; "
            f"border-right: 1px solid {t.get('border')}; }}")
        # 侧栏树面板重放：三页面板交接后页面级 apply_theme 够不到
        # （构造时烘焙 bg_secondary，切浅色残留深色列——2026-09-29 实测）
        try:
            from .components.theme import sidebar_panel_style as _sps
        except ImportError:  # 脚本态（无包父级）
            from ui.components.theme import sidebar_panel_style as _sps
        for _i in range(self.sidebar_stack.count()):
            _panel = self.sidebar_stack.widget(_i)
            if _panel is not None:
                _panel.setStyleSheet(_sps())
        self.title_band.setStyleSheet(f"""
            QFrame#TitleBand {{
                background: {t.get('bg_primary')};
                border-bottom: 1px solid {t.get('border')};
            }}
        """)
        # 换肤后强制重绘——否则顶带渲染滞后一拍（显示上一次主题，
        # 用户感知「点多次颜色不变」，2026-09-29 实测）
        self.title_band.update()
        self.sidebar_handle.setStyleSheet(f"background: {t.get('border')};")
        # 底部垫层：深色模式预涂半亮度底色补偿系统底缘玻璃提亮
        from PySide6.QtGui import QPalette
        _pad_c = "#0E0E0E" if theme().mode == "dark" else t.get('bg_primary')
        self.bottom_pad = getattr(self, "bottom_pad", None)
        if self.bottom_pad is not None:
            self.bottom_pad.setStyleSheet(f"background: {_pad_c};")
            _bp = self.bottom_pad.palette()
            _bp.setColor(QPalette.ColorRole.Window, QColor(_pad_c))
            self.bottom_pad.setPalette(_bp)
        self.cap_fold_btn.setStyleSheet(
            f"QPushButton {{ background: {t.get('bg_primary')}; border: none; "
            f"border-radius: 4px; padding: 0; }}"
            f"QPushButton:hover {{ background: {t.get('btn_glass_hover')}; }}"
        )
        # 窗口背景同步（不透明窗口的底色），避免切换后边缘残留旧色
        # （offscreen 探针平台 winId 非真实 NSView，objc 消息会段错误——跳过）
        try:
            from PySide6.QtWidgets import QApplication as _QA
            if _QA.instance().platformName() == "offscreen":
                raise RuntimeError("offscreen")
            import objc as _objc
            _nsview = _objc.objc_object(c_void_p=int(self.winId()))
            _win = _nsview.window()
            if _win is not None and _AppKit is not None:
                _c = QColor(t.get('bg_primary'))
                _win.setBackgroundColor_(
                    _AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                        _c.redF(), _c.greenF(), _c.blueF(), 1.0))
                # NSWindow 外观随应用主题（原生标题栏玻璃用它上色——
                # 不同步则浅色主题下顶带原生条残留深色，2026-09-29 实测）
                _name = (_AppKit.NSAppearanceNameAqua
                         if theme().mode == "light"
                         else _AppKit.NSAppearanceNameDarkAqua)
                _appe = _AppKit.NSAppearance.appearanceNamed_(_name)
                if _appe is not None:
                    _win.setAppearance_(_appe)
        except Exception:
            pass
        self.rail.update_theme()
        # 顶带富内容面板已 reparent 出页面（页面级样式表够不到）→ 逐页挂钩
        for _i in range(self.content_stack.count()):
            _w = self.content_stack.widget(_i)
            if _w is not None and hasattr(_w, "restyle_band_bar"):
                try:
                    _w.restyle_band_bar()
                except RuntimeError:
                    pass  # 页面 C++ 对象已被销毁（换页重建竞态）

    def _update_sidebar_for_page(self, index: int):
        """切页联动：有树的模块切到对应树并按折叠偏好显隐。

        无树的模块侧栏与展开热条都隐藏——内容区直接填满（用户定案：
        三区域宽度全局一致，无树页面不得残留窄间隔段）。"""
        stack_idx = self._sidebar_pages.get(index)
        if stack_idx is not None:
            self.sidebar_stack.setCurrentIndex(stack_idx)
            self.sidebar.setVisible(not self._sidebar_collapsed)
            self.sidebar_unfold.setVisible(self._sidebar_collapsed)
        else:
            self.sidebar.setVisible(False)
            self.sidebar_unfold.setVisible(False)
        self._sync_cap()

    def _sync_band(self):
        """统一标题栏：随窗口宽布局；页内容槽随当前页切换。

        顶带全 app 单行 48pt（用户裁定 2026-09-27，禁止两行）。
        三列起点随之调整（root_v 顶边距 = 顶带高度）。
        """
        band = getattr(self, "title_band", None)
        if band is None:
            return
        idx = self.content_stack.currentIndex()
        band.setFixedHeight(40)
        band.setGeometry(0, 0, self.width(), 40)
        if hasattr(self, "_root_v"):
            self._root_v.setContentsMargins(0, 40, 0, 0)
        if hasattr(self, "band_stack"):
            self.band_stack.setCurrentIndex(idx if idx < self.band_stack.count() else 0)
        btn = getattr(self, "cap_fold_btn", None)
        dot = getattr(self, "conn_dot", None)
        if hasattr(self, "sidebar_handle"):
            self.sidebar_handle.setVisible(self.sidebar.isVisible())
        # ‹ 展开浮钮（收起态）：骑缝在树列分隔线上（x=RAIL_W-7）垂直居中；
        # 显隐由悬停轮询/收起事件管理（自动隐藏，用户 2026-09-28）
        ufb = getattr(self, "sidebar_unfold_btn", None)
        if ufb is not None and ufb.isVisible():
            ufb.move(RAIL_W - 7, max(12, (self.height() - ufb.height()) // 2))
            ufb.raise_()
        if idx in self._sidebar_pages:
            if btn is not None:
                btn.show()
        else:
            if btn is not None:
                btn.hide()

        # 左占位 = 顶带内容起点对齐**操作行首钮**（P13① + 2026-09-29 用户令：
        # 收起/展开两态下顶带与操作行必须同线）。操作行左呼吸=TOOLBAR_MARGIN(10)
        # ——同源一个 token；band_gap 与 band_stack 间有 tb_h 8px 间距 →
        # gap = 目标 x - 8。下限=RAIL_W+11：收起态让开 ‹ 浮钮（右缘 67，
        # 71-67=4px 视觉间隙）。
        gap = getattr(self, "band_gap", None)
        if gap is not None:
            # 顶带内容与操作行首钮同线：内容左缘 + 操作行左呼吸（同源 token）。
            # x 读取时机问题（切换过渡态未落位）由 _toggle_sidebar 的
            # singleShot 延迟重算兜底——布局落位后再补一次本方法。
            # FM1.0：无树页（人脸/设置）顶带左占位 76——红绿灯区 16-68+呼吸 8，
            # 71 会与红绿灯末端重叠（22 号样机发现）
            floor = 76 if self.content_stack.currentIndex() not in self._sidebar_pages \
                else RAIL_W + 11
            target_x = max(self.content_stack.x() + _T.TOOLBAR_MARGIN, floor)
            gap.setFixedWidth(max(0, target_x - 8))
            # 强制侧栏重绘：清掉启动期布局漂移留下的残影（未经重绘的旧像素）
        if dot is not None:
            dot.show()
            dot.raise_()
        if btn is not None:
            btn.raise_()
        self.sidebar.update()
        self.sidebar_stack.update()

    def _poll_seam_hover(self):
        """收起态：鼠标靠近树列分隔线（≤24px）时显示 ‹ 浮钮，否则隐去。"""
        ufb = getattr(self, "sidebar_unfold_btn", None)
        if ufb is None or self.sidebar.isVisible() or not self.sidebar_unfold.isVisible():
            if ufb is not None and ufb.isVisible():
                ufb.setVisible(False)
            return
        g = QCursor.pos()
        origin = self.mapToGlobal(QPoint(0, 0))
        dx = abs(g.x() - (origin.x() + RAIL_W))
        in_v = origin.y() <= g.y() <= origin.y() + self.height()
        if dx <= 24 and in_v:
            if not ufb.isVisible():
                ufb.move(RAIL_W - 7, max(12, (self.height() - ufb.height()) // 2))
                ufb.setVisible(True)
                ufb.raise_()
            self._unfold_hide_timer.start()
        elif not ufb.underMouse():
            ufb.setVisible(False)

    def _unfold_hide_if_idle(self):
        ufb = self.sidebar_unfold_btn
        if ufb.isVisible() and not ufb.underMouse():
            ufb.setVisible(False)

    def _toggle_inspector(self):
        """⌥I 元素检查器：悬停任意控件显示 x/y/w/h/margins（开发态定位）。"""
        on = self._inspector.toggle()
        print(f"[INSPECTOR] {'ON' if on else 'OFF'}", flush=True)

    def _reload_pixel_tokens(self):
        """⌘R 热重载像素 token：重灌 SHARED → 全局 QSS 重放 → 各已构建页
        apply_theme（仅换肤不重建，任务进度/勾选保留）。
        注意：QSS 换不了的构建期几何（如统计卡重建、树列宽）不在重放范围，
        改这类数值后仍需重启验收。"""
        from ui.components.theme import reload_pixel_tokens
        n = reload_pixel_tokens()
        from PySide6.QtWidgets import QApplication as _QA
        qapp = _QA.instance()
        if qapp is not None:
            qapp.setStyleSheet(global_stylesheet())
        self.setStyleSheet(global_stylesheet())
        self._apply_chrome_theme()
        current = self.content_stack.currentIndex()
        for idx in list(self._pages_built.keys()):
            page_widget = self.content_stack.widget(idx)
            if page_widget is not None and hasattr(page_widget, "apply_theme"):
                try:
                    page_widget.apply_theme()
                except Exception:
                    import traceback
                    traceback.print_exc()
        self.content_stack.setCurrentIndex(current)
        self._sync_band()
        print(f"[TOKENS] ⌘R 重载 {n} 项", flush=True)

    def _toggle_pixel_ruler(self):
        self._pixel_ruler.refresh()
        self._pixel_ruler.setVisible(not self._pixel_ruler.isVisible())
        print(f"[RULER] {'ON' if self._pixel_ruler.isVisible() else 'OFF'}", flush=True)

    def _sync_cap(self):
        """兼容别名：旧调用点统一走 _sync_band。"""
        self._sync_band()

    def _on_sidebar_drag(self, delta: float):
        """树列拖宽：增量夹取 180-320pt（钉板）。"""
        self._sidebar_w = int(max(180, min(320, self._sidebar_w + delta)))
        self.sidebar.setFixedWidth(self._sidebar_w)
        self._sync_cap()

    def _on_sidebar_drag_done(self):
        from PySide6.QtCore import QSettings
        QSettings().setValue("ui/sidebar_width", int(self._sidebar_w))

    def _apply_sidebar(self, show: bool):
        self.sidebar.setVisible(show)
        self.sidebar_unfold.setVisible(not show)

    def _toggle_sidebar(self, collapse: bool):
        from PySide6.QtCore import QSettings
        self._sidebar_collapsed = collapse
        if collapse and hasattr(self, "sidebar_unfold_btn"):
            ufb = self.sidebar_unfold_btn
            ufb.move(RAIL_W - 7, max(12, (self.height() - ufb.height()) // 2))
            ufb.setVisible(True)
            ufb.raise_()
            self._unfold_hide_timer.start()
        QSettings().setValue("ui/sidebar_collapsed", collapse)
        self._update_sidebar_for_page(self.content_stack.currentIndex())
        # 布局落位后补一次顶带对齐（过渡态 content_stack.x 未定，首算会偏）
        QTimer.singleShot(0, self._sync_band)

    def closeEvent(self, event):
        """优雅关闭：停止所有后台线程和定时器，防止 QThread 崩溃。"""
        # 停止定时器
        if hasattr(self, '_status_timer'):
            self._status_timer.stop()
        if hasattr(self, '_results_timer'):
            self._results_timer.stop()

        # 停止 data workers
        if hasattr(self, '_data_workers'):
            for w in self._data_workers:
                if w.isRunning():
                    w.quit()
                    w.wait(3000)

        # 停止 page workers
        if hasattr(self, '_page_workers'):
            for w in self._page_workers:
                if w.isRunning():
                    w.quit()
                    w.wait(3000)

        # 清理 DashboardPage 的后台线程（关键修复）
        if hasattr(self, '_dashboard') and self._dashboard is not None:
            if hasattr(self._dashboard, 'cleanup'):
                self._dashboard.cleanup()

        # 记忆窗口大小/位置（根参数钉板 2026-09-28）
        from PySide6.QtCore import QSettings as _QClose
        _QClose().setValue(
            "ui/window_geometry",
            bytes(self.saveGeometry().toBase64()).decode("ascii"))

        super().closeEvent(event)

    def _switch_page(self, index: int):
        """Switch to a page, lazy-loading if needed."""
        print(f"[PAGE] _switch_page({index}) 当前页={self.content_stack.currentIndex() if hasattr(self, 'content_stack') else '?'}")
        # 再点当前页的导航钮/菜单项（用户功能 2026-09-27）：有管理树的页面
        # 收起/展开树（toggle）——VS Code 活动栏同款交互；无树页无动作
        if (hasattr(self, "content_stack")
                and index == self.content_stack.currentIndex()
                and index in self._sidebar_pages):
            self._toggle_sidebar(not self._sidebar_collapsed)
            return
        # 导航在菜单栏 + 图标侧栏（⌘1-⌘6 与 _build_page 的索引映射一致）；
        # 侧栏选中态随切页刷新
        self.rail.set_current(index)

        # Lazy load page content
        just_built = not self._pages_built.get(index, False)
        if just_built:
            self._pages_built[index] = True
            try:
                # Build page synchronously (fast operation)
                self._build_page(index)
            except Exception as exc:
                # 构建失败：复位标记允许下次点击重试；弹窗报告而不是
                # 让未处理异常杀死整个前端（PySide6 槽内未捕获异常会 abort）
                self._pages_built[index] = False
                import logging, traceback
                logging.getLogger(__name__).error(
                    "页面 %s 构建失败: %s\n%s", index, exc, traceback.format_exc())
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "页面加载失败",
                    f"页面构建失败，请重试或查看日志。\n\n{exc}")
                return
        
        # Switch to page immediately
        self.content_stack.setCurrentIndex(index)
        # 统一侧栏随页联动（10 号方案：有树显示/无树自动收起）
        self._update_sidebar_for_page(index)
        # 切页即刷新被切换页面的数据（保留分页/筛选/勾选状态,非重建）。
        # 背景：后端预热窗口内首屏加载失败后,页面停留空白且不自愈
        # （2026-10-07 用户实测）。未构建页跳过（构建路径自带首载）。
        self._refresh_page_content_on_switch(index)



        # Start/stop auto-refresh timer based on current page
        if index == 1:
            # Results page — start auto refresh
            self._results_timer.start()
            # 只有"重新进入"才需要主动刷新：首次进入时 ResultsPage.__init__
            # 已经加载过一次，再调一次等于把 1500+ 条记录与封面全部解两遍。
            if not just_built:
                widget = self.content_stack.widget(1)
                if widget is not None and hasattr(widget, '_load_data'):
                    # 静默刷新：保留界面与勾选，后台取数（消除切页"缓存卡顿"）
                    widget._load_data(silent=True)
        else:
            self._results_timer.stop()
    
    def _set_band_page(self, index: int, widget):
        """顶带内容槽换富内容：占位标签（若在）回收，同索引换入页面面板。"""
        old = self.band_stack.widget(index)
        if old is not None:
            self.band_stack.removeWidget(old)
            old.deleteLater()
        self.band_stack.insertWidget(index, widget)

    def _build_page(self, index: int):
        """Build a page widget and replace the stub."""
        print(f"[PAGE] _build_page({index}) 开始")
        old_widget = self.content_stack.widget(index)
        
        if index == 1:
            # Results page (New static UI)
            real_widget = ResultsPage()
            # 空态「去导入任务」→ 切回任务页（19§D4.14）
            real_widget.go_tasks_requested.connect(lambda: self._switch_page(0))
            # 12 号：项目管理树迁入统一侧栏（红绿灯头部区横跨其上）
            _proj_panel = real_widget.take_project_panel()
            if _proj_panel is not None and index not in self._sidebar_pages:
                self._register_sidebar_page(index, _proj_panel)
            # D6：刷新/搜索/视图/排序上移顶带（已选中胶囊已裁定撤除）
            _dbar = real_widget.take_detail_bar()
            _wrap = QWidget()
            _h = QHBoxLayout(_wrap)
            _h.setContentsMargins(0, 0, 0, 0)
            _h.setSpacing(8)
            if _dbar is not None:
                _h.addWidget(_dbar)
            _h.addStretch()  # 面板compact居左，避免 band_stack 均摊宽度拉宽按钮
            self._set_band_page(1, _wrap)
        elif index == 2:
            # Labels page
            real_widget = LabelManagementPage(api_base=API_BASE)
            # 切换分析主题 → 广播给主窗口，刷新所有已构建页面（提示词/人物/看板等）
            if hasattr(real_widget, "analysis_theme_changed"):
                real_widget.analysis_theme_changed.connect(self._on_analysis_theme_changed)
            # 11 号三栏：标准标签库面板迁往主窗口统一侧栏
            _dict_panel = real_widget.take_dictionary_panel()
            if _dict_panel is not None and index not in self._sidebar_pages:
                self._register_sidebar_page(index, _dict_panel)
            # L1：审核操作条（全选/已选/确认/拒绝/删除/刷新）上移顶带
            _audit_bar = real_widget.take_audit_bar()
            if _audit_bar is not None:
                self._set_band_page(2, _audit_bar)
        elif index == 3:
            # Actors page
            real_widget = ActorManagementPage(api_base=API_BASE)
            # FM4.6 空态「去导入任务」→ 切回任务页
            real_widget.actor_grid.go_tasks_requested.connect(
                lambda: self._switch_page(0))
            # F1：批量操作七钮上移顶带
            _face_bar = real_widget.take_face_bar()
            if _face_bar is not None:
                self._set_band_page(3, _face_bar)
        elif index == 4:
            # Settings page
            try:
                real_widget = SettingsPage(api_base=API_BASE)
                # 分析主题切换 → 广播刷新所有已构建页面（与标签管理同机制）
                if hasattr(real_widget, "analysis_theme_changed"):
                    real_widget.analysis_theme_changed.connect(
                        self._on_analysis_theme_changed)
                if hasattr(real_widget, "manage_tags_requested"):
                    real_widget.manage_tags_requested.connect(
                        lambda: self._switch_page(2))
                # 设置页「环境初始化」→ 主窗口弹同一勾选式下载框
                if hasattr(real_widget, "env_init_requested"):
                    real_widget.env_init_requested.connect(self.show_download_wizard)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] 设置页创建失败: {e}")
                raise
        elif index == 5:
            # Tag browse page
            real_widget = TagBrowsePage(api_client=APIClient(base_url=API_BASE))
            real_widget.initialize()  # 初始化加载标签树
            # 10 号三栏：标签筛选树迁往主窗口统一侧栏（与任务页来源树同机制）
            _tag_panel = real_widget.take_tree_panel()
            if _tag_panel is not None and index not in self._sidebar_pages:
                self._register_sidebar_page(index, _tag_panel)
            # B2：修改标签/删除标签/刷新上移顶带
            _tag_bar = real_widget.take_tag_bar()
            if _tag_bar is not None:
                self._set_band_page(5, _tag_bar)
        else:
            return

        # Replace stub with real widget
        if old_widget is not None:
            self.content_stack.removeWidget(old_widget)
            old_widget.deleteLater()
        self.content_stack.insertWidget(index, real_widget)
    
    def _refresh_page_content_on_switch(self, index: int) -> None:
        """切页刷新：重取当前页数据,保留页面既有状态（分页/筛选/勾选）。

        仅对已构建页面生效（未构建页由构建路径自带首载,避免双载）。
        刷新方法映射与 _on_analysis_theme_changed 的广播保持同款;
        失败静默（后端未就绪时由状态灯与下次切页兜底）。
        """
        widget = self.content_stack.widget(index)
        if widget is None:
            return
        try:
            if index == 0 and hasattr(widget, "refresh_data"):
                widget.refresh_data()
            elif index == 1 and hasattr(widget, "_load_data"):
                widget._load_data()
            elif index == 3 and hasattr(widget, "load_actors"):
                widget.load_actors()
            # index 2 标签管理自带刷新钮、4 设置无数据刷新、5 标签树有顶带
            # 刷新钮——均不在此列,避免重建损失 UI 状态。
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"切页刷新 index={index} 失败(忽略): {e}")

    def _refresh_results_async(self):
        """Refresh results page — delegate to ResultsPage._load_data()."""
        widget = self.content_stack.widget(1)
        if widget is not None and hasattr(widget, '_load_data'):
            widget._load_data()
    
    def _on_results_loaded(self, index: int, data):
        """Handle results data loaded from background."""
        print(f"[DEBUG] _on_results_loaded: index={index}, data类型={type(data).__name__}")
        if index != 1:
            return
        
        # API返回的是列表，不是字典
        if isinstance(data, list):
            tasks = data
        else:
            tasks = data.get("tasks", [])
        
        print(f"[DEBUG] _on_results_loaded: tasks数量={len(tasks)}")
        
        # Filter: exclude PENDING and SIFTED (only show tasks that have completed analysis or are in progress)
        excluded_statuses = {"PENDING", "SIFTED"}
        tasks = [t for t in tasks if t.get("status") not in excluded_statuses]

        # Filter by status dropdown
        processing_statuses = {"CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "PROCESSING"}
        filter_map = {
            "全部": None, 
            "已完成": "COMPLETE", 
            "待审核": "CONFIRMED", 
            "处理中": processing_statuses,
            "错误": "ERROR"
        }
        filter_text = self.result_filter.currentText() if hasattr(self, 'result_filter') else "全部"
        if filter_text not in filter_map:
            filter_text = "全部"
        status_filter = filter_map.get(filter_text)

        if status_filter:
            if isinstance(status_filter, set):
                tasks = [t for t in tasks if t.get("status") in status_filter]
            else:
                tasks = [t for t in tasks if t.get("status") == status_filter]

        # Sort by updated_at descending (newest first)
        tasks.sort(key=lambda t: t.get("updated_at") or t.get("created_at", ""), reverse=True)

        # Clear existing cards
        while self.result_layout.count() > 1:  # Keep the stretch at the end
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create cards for each task
        for task in tasks:
            card = self._create_result_card(task)
            self.result_layout.insertWidget(self.result_layout.count() - 1, card)
        
        # Clean up worker reference
        if hasattr(self, '_data_workers'):
            self._data_workers = [w for w in self._data_workers if w.isRunning()]
    
    def _on_page_ready(self, index: int, widget: QWidget):
        """Handle page initialization completion."""
        old_widget = self.content_stack.widget(index)
        
        # Replace stub with real widget
        if old_widget is not None:
            self.content_stack.removeWidget(old_widget)
            old_widget.deleteLater()
        self.content_stack.insertWidget(index, widget)
        
        # If this is the current page, switch to it
        if self.content_stack.currentIndex() == index:
            self.content_stack.setCurrentIndex(index)
        
        # Connect signals for results page
        # Note: _build_result_page already sets self.result_filter, self.refresh_results_btn,
        # self.result_container, and self.result_layout, and connects their signals.
        # This method is kept for potential future use but the attribute check is corrected.
        if index == 1 and hasattr(self, 'result_filter'):
            # Signals already connected in _build_result_page
            # Trigger initial data load if not already loaded
            if not hasattr(self, '_results_loaded'):
                self._results_loaded = True
                self.refresh_results()
        
        # Clean up worker reference
        if hasattr(self, '_page_workers'):
            self._page_workers = [w for w in self._page_workers if w.isRunning()]

    def _build_result_page(self, parent):
        """Build the results/analysis page — card view with detailed analysis, newest first."""
        t = theme()
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Top bar: filter + refresh
        top_bar = QHBoxLayout()
        self.result_filter = QComboBox()
        self.result_filter.addItems(["全部", "已完成", "待审核", "处理中", "错误"])
        self.result_filter.setFixedWidth(100)
        self.result_filter.currentIndexChanged.connect(self.refresh_results)
        top_bar.addWidget(self.result_filter)
        top_bar.addStretch()
        self.refresh_results_btn = QPushButton("刷新")
        self.refresh_results_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                padding: 3px 12px;
                font-size: {t.get('font_size_sm')}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        self.refresh_results_btn.clicked.connect(self.refresh_results)
        top_bar.addWidget(self.refresh_results_btn)
        layout.addLayout(top_bar)

        # Scroll area for result cards
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.setSpacing(12)
        self.result_layout.addStretch()
        
        scroll.setWidget(self.result_container)
        layout.addWidget(scroll)
        
        # 页面构建完成后立即加载数据
        self.refresh_results()

    def _toggle_theme(self):
        """两态循环切换外观（深色 → 浅色），重建/重涂所有已构建页面。"""
        t = theme()
        t.toggle()
        # 更新应用级样式表（关键：全局 QLabel 颜色等规则随主题刷新）
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance()
        if qapp is not None:
            qapp.setStyleSheet(global_stylesheet())
        # Reapply global stylesheet
        self.setStyleSheet(global_stylesheet())
        self._apply_chrome_theme()
        # Update content stack background
        self.content_stack.setStyleSheet(f"""
            QStackedWidget {{
                background: {theme().get('bg_primary')};
            }}
        """)
        # 底部状态栏已删除（方案 B），无需重涂  # 备注：macOS 27 会对窗口底部 28pt 强制玻璃化（实验证实），状态文字需高对比
        # 刷新所有已构建页面：优先走 apply_theme()（仅重上色、保留状态），
        # 未实现 apply_theme 的页面回退为重建（原逻辑）。
        # 关键：首页 Dashboard 持有批量人脸识别队列/轮询等状态，切主题必须
        # 保活——apply_theme 只换肤不销毁，进度不会丢。
        current = self.content_stack.currentIndex()
        for idx in list(self._pages_built.keys()):
            page_widget = self.content_stack.widget(idx)
            if page_widget is not None and hasattr(page_widget, "apply_theme"):
                try:
                    page_widget.apply_theme()
                    continue
                except Exception as _te:
                    import traceback as _tb
                    _tb.print_exc()
            # 无 apply_theme（或调用失败）→ 回退重建
            if idx == 0:
                # 首页 Dashboard 特殊处理：直接重建
                old_dash = self.content_stack.widget(0)
                if old_dash is not None:
                    self.content_stack.removeWidget(old_dash)
                    old_dash.deleteLater()
                self._dashboard = DashboardPage(api_base=API_BASE)
                self.content_stack.insertWidget(0, self._dashboard)
                continue
            if self._pages_built.get(idx, False):
                self._pages_built[idx] = False
                self._build_page(idx)
        # 切回当前页
        self.content_stack.setCurrentIndex(current)

    def _api_client(self):
        try:
            from .components.api_client import APIClient
        except ImportError:
            # 脚本模式直接运行时(__main__)相对导入不可用,
            # 此时 src 已在 sys.path, 用 ui.components 包路径导入。
            from ui.components.api_client import APIClient

        return APIClient(base_url=API_BASE)

    def _refresh_analysis_theme_label(self):
        """拉取当前生效主题名缓存到 self._current_theme_name（失败静默，不阻断启动）。

        原刷新 rail 按钮文案的职责随 10 号方案移除；名字留给设置页分区（切片 5）。
        """
        try:
            data = self._api_client().get("/themes", timeout=2) or {}
            name = data.get("current") or ""
        except Exception:
            name = ""
        self._current_theme_name = name

    def _open_theme_switcher(self):
        """顶栏常驻入口 → 主题切换器（仿 Obsidian vault switcher）。"""
        # 免费版 vault 守卫：只允许使用当前 1 个主题工作区（07-授权管理.md §1.4）
        try:
            from licensing import state as _license_state
        except ImportError:
            from src.licensing import state as _license_state
        if not _license_state.is_pro():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "授权",
                "免费版仅支持 1 个 vault（主题工作区）。\n激活后可创建与切换多个工作区。",
            )
            return
        try:
            from .dialogs.theme_switcher_dialog import ThemeSwitcherDialog
        except ImportError:
            # 脚本模式直接运行时(__main__)相对导入不可用,
            # 此时 src 已在 sys.path, 用 ui.dialogs 包路径导入。
            from ui.dialogs.theme_switcher_dialog import ThemeSwitcherDialog
        dlg = ThemeSwitcherDialog(self._api_client(), self)
        dlg.switched.connect(self._on_analysis_theme_changed)
        dlg.exec()

    def _on_analysis_theme_changed(self, name: str):
        """切换分析主题后，刷新所有已构建页面（含设置页提示词跟随）。

        照抄 _toggle_theme 的遍历骨架：页面会被 _build_page/_toggle_theme 重建
        （deleteLater），因此必须每次 content_stack.widget(idx) 现取 + hasattr 守卫，
        不可缓存页面引用。
        """
        print(f"[THEME] 切换分析主题: {name} → 刷新已构建页面")
        for idx in list(self._pages_built.keys()):
            widget = self.content_stack.widget(idx)
            if widget is None:
                continue
            try:
                if idx == 0 and hasattr(widget, "refresh_data"):
                    widget.refresh_data()
                elif idx == 1 and hasattr(widget, "_load_data"):
                    widget._load_data()
                elif idx == 2:
                    pass  # 标签管理页已在 _on_theme_changed 内自行刷新
                elif idx == 3 and hasattr(widget, "load_actors"):
                    widget.load_actors()
                elif idx == 4 and hasattr(widget, "_load_prompt_themes"):
                    # 关键：提示词随之跟随（重读 active_theme → 重建下拉 → 重载编辑器）
                    widget._load_prompt_themes()
                elif idx == 5 and hasattr(widget, "initialize"):
                    widget.initialize()
            except Exception as exc:
                print(f"[THEME] 刷新页面 {idx} 失败: {exc}")

    # ── Dashboard refresh ───────────────────────────────────────────

    def refresh_dashboard(self):
        """Refresh the Dashboard tab immediately after import."""
        if hasattr(self, '_dashboard') and self._dashboard:
            self._dashboard.refresh_data()

    # ── Status Indicator ────────────────────────────────────────────

    def _update_status_indicator(self):
        """后台线程探测 /status，信号回主线程刷新（不阻塞事件循环）。"""
        import threading as _threading

        def _probe():
            try:
                resp = requests.get(f"{API_BASE}/status", timeout=3)
                self.status_checked.emit(resp.status_code == 200)
            except Exception:
                self.status_checked.emit(False)

        _threading.Thread(target=_probe, daemon=True).start()
    
    def _on_status_checked(self, connected: bool):
        """Handle status check result."""
        if connected:
            self.conn_dot.setText("● 已连接")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('success')};
            """)
        else:
            self.conn_dot.setText("● 已断开")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('danger')};
            """)

    # ── Model Download ──────────────────────────────────────────────

    def start_model_download(self):
        """Start model download with progress dialog."""
        from src.core.dependency_checker import DependencyChecker
        from src.core.model_downloader import ModelDownloader

        checker = DependencyChecker()

        # Check if model already exists
        deps = checker.check_all()
        minicpm_status = deps.get('minicpm_model')
        if minicpm_status and minicpm_status.installed:
            QMessageBox.information(self, "模型下载", "MiniCPM-V 模型已存在，无需下载。")
            return

        bundled_ollama = checker._get_bundled_ollama_path()
        if not bundled_ollama:
            QMessageBox.warning(self, "模型下载", "未找到内置 Ollama，无法下载模型。")
            return

        # Create downloader
        self._model_downloader = ModelDownloader(
            ollama_path=str(bundled_ollama),
            max_retries=3
        )

        # Create progress dialog
        self._download_dialog = ModelDownloadDialog(self)

        # Connect cancel button
        self._download_dialog.cancel_btn.clicked.connect(self._cancel_download)

        # Start download with callbacks
        self._model_downloader.download_async(
            model_name='minicpm-v4.5',
            on_progress=self._on_download_progress,
            on_completed=self._on_download_completed,
            on_started=self._on_download_started
        )

        # Show dialog
        self._download_dialog.show()

    def start_model_download_with_wizard(self):
        """Start model download with wizard-style progress dialog."""
        from src.core.dependency_checker import DependencyChecker
        from src.core.model_downloader import ModelDownloader
        from src.ui.components.first_run_wizard import DownloadProgressDialog

        checker = DependencyChecker()

        # Check if model already exists
        deps = checker.check_all()
        minicpm_status = deps.get('minicpm_model')
        if minicpm_status and minicpm_status.installed:
            return

        bundled_ollama = checker._get_bundled_ollama_path()
        if not bundled_ollama:
            QMessageBox.warning(self, "模型下载", "未找到内置 Ollama，无法下载模型。")
            return

        # Create downloader
        self._model_downloader = ModelDownloader(
            ollama_path=str(bundled_ollama),
            max_retries=3
        )

        # Create wizard-style progress dialog
        self._wizard_download_dialog = DownloadProgressDialog(self)

        # Connect cancel button
        self._wizard_download_dialog.cancel_requested.connect(self._cancel_wizard_download)

        # Start download with callbacks
        self._model_downloader.download_async(
            model_name='minicpm-v4.5',
            on_progress=self._on_wizard_download_progress,
            on_completed=self._on_wizard_download_completed,
            on_started=self._on_download_started
        )

        # Show dialog
        self._wizard_download_dialog.show()

    def _on_wizard_download_progress(self, progress):
        """Handle wizard download progress update.

        状态栏不依赖对话框可见——用户关闭下载对话框后切到任意页面,
        仍能从状态栏看到下载进度。
        """
        if hasattr(self, '_wizard_download_dialog') and self._wizard_download_dialog.isVisible():
            self._wizard_download_dialog.update_progress(
                percent=progress.percent,
                downloaded=progress.downloaded,
                total=progress.total,
                speed=progress.speed,
                status=progress.status
            )

        # 状态栏始终更新(全局可见)
        if progress.percent > 0:
            self.conn_dot.setText(f"下载中: {progress.percent:.1f}%")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('accent')};
            """)
        else:
            self.conn_dot.setText(progress.status)
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('text_secondary')};
            """)

    def _on_wizard_download_completed(self, success: bool, message: str):
        """Handle wizard download completion."""
        if hasattr(self, '_wizard_download_dialog') and self._wizard_download_dialog.isVisible():
            self._wizard_download_dialog.set_completed(success, message)

        # Update status bar
        if success:
            self.conn_dot.setText("● 已连接")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('success')};
            """)
        else:
            self.conn_dot.setText("状态: 断开")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('danger')};
            """)

    def _cancel_wizard_download(self):
        """Cancel the wizard download."""
        if hasattr(self, '_model_downloader') and self._model_downloader:
            self._model_downloader.cancel()
            if hasattr(self, '_wizard_download_dialog') and self._wizard_download_dialog.isVisible():
                self._wizard_download_dialog.set_completed(False, "下载已取消")


    def _on_download_started(self, model_name: str):
        """Handle download start."""
        print(f"开始下载模型: {model_name}")

    def _on_download_progress(self, progress):
        """Handle download progress update.

        状态栏更新不依赖对话框可见——用户关闭/最小化下载对话框后,
        切到任何页面仍能从状态栏看到下载进度(对话框只是详细视图)。
        """
        # 对话框可见时更新详细进度
        if hasattr(self, '_download_dialog') and self._download_dialog.isVisible():
            self._download_dialog.update_progress(progress)

        # 状态栏始终更新(全局可见, 任意页面都能看到下载状态)
        if progress.percent > 0:
            self.conn_dot.setText(f"下载中: {progress.percent:.1f}%")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('accent')};
            """)
        else:
            self.conn_dot.setText(progress.status)
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('text_secondary')};
            """)

    def _on_download_completed(self, success: bool, message: str):
        """Handle download completion."""
        if hasattr(self, '_download_dialog') and self._download_dialog.isVisible():
            self._download_dialog.set_completed(success, message)

        # Update status bar
        if success:
            self.conn_dot.setText("状态: 已连接")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('success')};
            """)
            # Show success notification
            QMessageBox.information(self, "下载完成", message)
        else:
            self.conn_dot.setText("状态: 断开")
            self.conn_dot.setStyleSheet(f"""
                font-size: 12px;
                color: {theme().get('danger')};
            """)
            # Show error notification
            QMessageBox.critical(self, "下载失败", message)

    def _cancel_download(self):
        """Cancel the current download."""
        if hasattr(self, '_model_downloader') and self._model_downloader:
            self._model_downloader.cancel()
            if hasattr(self, '_download_dialog') and self._download_dialog.isVisible():
                self._download_dialog.set_completed(False, "下载已取消")

    def show_download_wizard(self):
        """弹出下载勾选对话框（欢迎页）——人脸模型必选/ VLM 可选。

        首次运行与设置页「环境初始化」共用同一入口: 按勾选分别下载
        人脸模型(env/init)与 MiniCPM-V(ollama)。
        """
        from src.ui.components.first_run_wizard import FirstRunWizard

        wizard = FirstRunWizard(self)
        wizard.download_required_requested.connect(self.start_env_init_with_progress)
        wizard.download_vlm_requested.connect(self.start_model_download_with_wizard)
        wizard.exec()

    def start_env_init_with_progress(self):
        """后台执行「环境初始化」下载必选组件(人脸模型等)。

        欢迎页点击「下载人脸模型」触发。env/init 是同步长请求(可能
        数分钟~数小时, 视网络), 放后台线程避免卡 UI; 状态栏显示进行中,
        完成/失败弹窗提示。
        """
        import requests as _req
        from PySide6.QtCore import Signal as _Sig

        if getattr(self, "_env_init_running", False):
            QMessageBox.information(self, "环境初始化", "环境初始化正在进行中，请稍候…")
            return
        self._env_init_running = True

        class _EnvInitWorker(QThread):
            done = _Sig(dict)
            failed = _Sig(str)

            def run(self):
                try:
                    resp = _req.post(f"{API_BASE}/env/init", json={}, timeout=7200)
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"status": "error", "steps": [],
                                "message": f"HTTP {resp.status_code}"}
                    self.done.emit(data)
                except Exception as e:
                    self.failed.emit(str(e))

        self._env_init_worker = _EnvInitWorker(self)
        self._env_init_worker.done.connect(self._on_env_init_done)
        self._env_init_worker.failed.connect(self._on_env_init_failed)

        self.conn_dot.setText("环境初始化: 下载必要组件中…")
        self.conn_dot.setStyleSheet(f"""
            font-size: 12px;
            color: {theme().get('warning')};
        """)
        self._env_init_worker.finished.connect(lambda: setattr(self, "_env_init_running", False))
        self._env_init_worker.start()

    def _on_env_init_done(self, data: dict):
        """环境初始化完成"""
        steps = data.get("steps", [])
        lines = [
            f"{'✅' if s.get('status') == 'ok' else '❌'} {s.get('step')}: {s.get('message')}"
            for s in steps
        ]
        ok = data.get("status") == "success"
        self.conn_dot.setText("状态: 已连接")
        self.conn_dot.setStyleSheet(f"""
            font-size: 12px;
            color: {theme().get('success')};
        """)
        if ok:
            QMessageBox.information(self, "环境初始化完成", "\n".join(lines) or "完成")
        else:
            QMessageBox.warning(self, "环境初始化未完成", "\n".join(lines) or data.get("message", "部分组件失败"))

        # 下载完成(或部分完成)后刷新设置页环境状态
        try:
            w = self.content_stack.widget(4)
            if w is not None and hasattr(w, "refresh_environment"):
                w.refresh_environment()
        except Exception:
            pass

    def _on_env_init_failed(self, err: str):
        """环境初始化失败"""
        self.conn_dot.setText("状态: 断开")
        self.conn_dot.setStyleSheet(f"""
            font-size: 12px;
            color: {theme().get('danger')};
        """)
        QMessageBox.critical(self, "环境初始化失败", str(err))

    # ── Results Tab ─────────────────────────────────────────────────

    def _auto_refresh_results(self):
        """Auto-refresh results page every 10 seconds when on results page.

        用 **silent** 刷新：不清屏、不丢勾选、不跳页，且数据没变就不重渲染 ——
        否则每 10 秒清空重建一次，用户看到的就是"频繁刷新、多次重新加载"。
        """
        if self.content_stack.currentIndex() == 1:
            widget = self.content_stack.widget(1)
            if widget is not None and hasattr(widget, '_load_data'):
                widget._load_data(silent=True)

    def refresh_results(self):
        """Refresh results — card view with detailed analysis, newest first."""
        if not hasattr(self, 'result_layout'):
            return
            
        try:
            resp = requests.get(f"{API_BASE}/task", timeout=10)
            if resp.status_code != 200:
                return
            tasks = resp.json()
        except Exception:
            return

        # Filter: exclude PENDING and SIFTED (only show tasks that have completed analysis or are in progress)
        excluded_statuses = {"PENDING", "SIFTED"}
        tasks = [t for t in tasks if t.get("status") not in excluded_statuses]

        # Filter by status dropdown
        processing_statuses = {"CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "PROCESSING"}
        filter_map = {
            "全部": None, 
            "已完成": "COMPLETE", 
            "待审核": "CONFIRMED", 
            "处理中": processing_statuses,
            "错误": "ERROR"
        }
        filter_text = self.result_filter.currentText() if hasattr(self, 'result_filter') else "全部"
        # 如果 filter_text 不在 filter_map 中，默认为"全部"
        if filter_text not in filter_map:
            filter_text = "全部"
        status_filter = filter_map.get(filter_text)

        if status_filter:
            if isinstance(status_filter, set):
                tasks = [t for t in tasks if t.get("status") in status_filter]
            else:
                tasks = [t for t in tasks if t.get("status") == status_filter]

        # Sort by updated_at descending (newest first)
        tasks.sort(key=lambda t: t.get("updated_at") or t.get("created_at", ""), reverse=True)

        # Clear existing cards
        while self.result_layout.count() > 1:  # Keep the stretch at the end
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create cards for each task
        for task in tasks:
            card = self._create_result_card(task)
            self.result_layout.insertWidget(self.result_layout.count() - 1, card)

    def _create_result_card(self, task):
        """Create a card widget showing detailed analysis for a task."""
        from src.ui.components.theme import theme
        t = theme()
        card = QWidget()
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(4)
        
        # Header: filename + status + time
        header = QHBoxLayout()
        name = sanitize_for_qt(os.path.basename(task.get("file_path", "")))
        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-size: 13px; font-weight: 500; color: {t.get('text_primary')};")
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Status (simple text)
        status = task.get("status", "")
        from src.ui.components.theme import theme
        t = theme()
        STATUS_MAP = {
            "PENDING": ("待处理", t.get("text_secondary")),
            "CHECKING": ("检查中", t.get("text_secondary")),
            "SIFTED": ("已抽帧", t.get("accent")),
            "PROCESSING": ("处理中", t.get("warning")),
            "SIFTING": ("筛选中", t.get("accent")),
            "ANALYZING": ("分析中", t.get("info")),
            "CLUSTERING": ("聚类中", t.get("info")),
            "CONFIRMED": ("待审核", t.get("warning")),
            "COMPLETE": ("已完成", t.get("success")),
            "ERROR": ("错误", t.get("danger")),
            "DUPLICATE": ("重复", t.get("text_secondary")),
            "RESUMABLE": ("可恢复", t.get("warning")),
        }
        status_info = STATUS_MAP.get(status, ("未知", t.get("text_secondary")))
        status_label = QLabel(status_info[0])
        status_label.setStyleSheet(f"font-size: 11px; color: {status_info[1]};")
        header.addWidget(status_label)
        
        # Time
        created = task.get("updated_at") or task.get("created_at", "")
        if created and "T" in str(created):
            try:
                dt = datetime.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                created = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        time_label = QLabel(str(created))
        time_label.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')}; margin-left: 8px;")
        header.addWidget(time_label)
        
        # File size
        file_size = task.get("file_size")
        if file_size:
            # 格式化文件大小
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            elif file_size < 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"
            size_label = QLabel(size_str)
            size_label.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')}; margin-left: 8px;")
            header.addWidget(size_label)
        
        layout.addLayout(header)
        
        # Parse analysis_details
        analysis_details_str = task.get("analysis_details") or ""
        if analysis_details_str:
            try:
                details = json.loads(analysis_details_str)
                
                # MD5 section
                md5_info = details.get("md5")
                if md5_info:
                    md5_section = self._create_section("MD5提取", t)
                    md5_content = QVBoxLayout()
                    md5_content.addWidget(self._create_info_line("哈希值", md5_info.get('hash', 'N/A'), t))
                    md5_content.addWidget(self._create_info_line("耗时", f"{md5_info.get('time', 0)}秒", t))
                    md5_section.addLayout(md5_content)
                    layout.addLayout(md5_section)
                
                # Scenes section
                scenes = details.get("scenes", [])
                if scenes:
                    scenes_section = self._create_section("场景划分", t)
                    scenes_content = QVBoxLayout()
                    scenes_content.addWidget(self._create_info_line("场景数量", str(len(scenes)), t))
                    for i, scene in enumerate(scenes):
                        if scene.get("type") == "fallback":
                            scenes_content.addWidget(self._create_info_line(f"场景 {i+1}", "回退模式 (固定抽帧)", t))
                        else:
                            start = scene.get("start", 0)
                            end = scene.get("end", 0)
                            scenes_content.addWidget(self._create_info_line(f"场景 {i+1}", f"帧 {start} - {end}", t))
                    scene_time = details.get("timing", {}).get("scene_detection", 0)
                    scenes_content.addWidget(self._create_info_line("耗时", f"{scene_time}秒", t))
                    scenes_section.addLayout(scenes_content)
                    layout.addLayout(scenes_section)
                
                # Frames section
                frames = details.get("frames", [])
                if frames:
                    frames_section = self._create_section("关键帧抽帧", t)
                    frames_content = QVBoxLayout()
                    total_frames = sum(f.get("count", 0) for f in frames)
                    frames_content.addWidget(self._create_info_line("总帧数", str(total_frames), t))
                    for frame_info in frames:
                        scene_idx = frame_info.get("scene_index", 0)
                        count = frame_info.get("count", 0)
                        frames_content.addWidget(self._create_info_line(f"场景 {scene_idx + 1}", f"{count} 帧", t))
                    frame_time = details.get("timing", {}).get("frame_extraction", 0)
                    frames_content.addWidget(self._create_info_line("耗时", f"{frame_time}秒", t))
                    frames_section.addLayout(frames_content)
                    layout.addLayout(frames_section)
                
                # VLM Analysis section
                vlm_raw = details.get("vlm_raw")
                if vlm_raw:
                    vlm_section = self._create_section("VLM分析识别", t)
                    vlm_content = QVBoxLayout()
                    vlm_text = QTextEdit()
                    vlm_text.setReadOnly(True)
                    vlm_text.setPlainText(vlm_raw)
                    vlm_text.setMaximumHeight(100)
                    vlm_text.setStyleSheet(f"""
                        QTextEdit {{
                            font-family: 'Menlo', monospace;
                            font-size: 12px;
                            background: {t.get('bg_primary')};
                            color: {t.get('text_primary')};
                            border: 1px solid {t.get('border_light')};
                            border-radius: {t.get('radius_sm')}px;
                            padding: 6px;
                        }}
                    """)
                    vlm_content.addWidget(vlm_text)
                    vlm_time = details.get("timing", {}).get("vlm_analysis", 0)
                    vlm_content.addWidget(self._create_info_line("耗时", f"{vlm_time}秒", t))
                    vlm_section.addLayout(vlm_content)
                    layout.addLayout(vlm_section)
                
                # Tag Mapping section
                mapped_tags = details.get("mapped_tags", [])
                unmatched = details.get("unmatched", [])
                if mapped_tags or unmatched:
                    tags_section = self._create_section("标签映射", t)
                    tags_content = QVBoxLayout()
                    if mapped_tags:
                        tags_content.addWidget(self._create_info_line("已映射标签", f"{len(mapped_tags)}个", t))
                        tags_text = QLabel(", ".join(mapped_tags))
                        tags_text.setWordWrap(True)
                        tags_text.setStyleSheet(f"font-size: 12px; color: {t.get('text_primary')}; padding-left: 12px;")
                        tags_content.addWidget(tags_text)
                    if unmatched:
                        tags_content.addWidget(self._create_info_line("未匹配标签", f"{len(unmatched)}个", t))
                        unmatched_display = unmatched[:10]
                        if len(unmatched) > 10:
                            unmatched_display.append(f"... 还有 {len(unmatched) - 10} 个")
                        unmatched_text = QLabel(", ".join(unmatched_display))
                        unmatched_text.setWordWrap(True)
                        unmatched_text.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')}; padding-left: 12px;")
                        tags_content.addWidget(unmatched_text)
                    mapping_time = details.get("timing", {}).get("tag_mapping", 0)
                    tags_content.addWidget(self._create_info_line("耗时", f"{mapping_time}秒", t))
                    tags_section.addLayout(tags_content)
                    layout.addLayout(tags_section)
                
                # Total time
                timing = details.get("timing", {})
                total_time = sum(timing.values())
                total_time_label = self._create_info_line("总耗时", f"{total_time:.2f}秒", t)
                total_time_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {t.get('accent')};")
                layout.addWidget(total_time_label)
                
            except (json.JSONDecodeError, TypeError) as e:
                error_label = QLabel(f"分析详情解析失败: {e}")
                error_label.setStyleSheet(f"color: {t.get('danger')}; font-size: 12px;")
                layout.addWidget(error_label)
        
        # Frames path
        frames_path = task.get("frames_path") or ""
        if frames_path:
            path_label = QLabel(f"帧文件目录: {frames_path}")
            path_label.setStyleSheet(f"font-size: 11px; color: {t.get('text_secondary')};")
            path_label.setWordWrap(True)
            layout.addWidget(path_label)
        
        # View log button
        task_id = task.get("id")
        if task_id:
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            view_log_btn = QPushButton("查看日志")
            view_log_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {t.get('bg_secondary')};
                    color: {t.get('text_primary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_sm')}px;
                    padding: 4px 12px;
                    font-size: 12px;
                }}
                QPushButton:hover {{
                    background: {t.get('bg_tertiary')};
                }}
            """)
            view_log_btn.clicked.connect(lambda checked, tid=task_id: self._view_analysis_log(tid))
            btn_layout.addWidget(view_log_btn)
            layout.addLayout(btn_layout)
        
        return card
    
    def _create_section(self, title, theme):
        """Create a collapsible section header."""
        section_layout = QVBoxLayout()
        section_layout.setSpacing(4)
        header = QLabel(f"【{title}】")
        header.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {theme.get('text_primary')}; margin-top: 4px;")
        section_layout.addWidget(header)
        return section_layout
    
    def _create_info_line(self, label, value, theme):
        """Create a label-value info line."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(8)
        
        label_widget = QLabel(f"{label}:")
        label_widget.setStyleSheet(f"font-size: 12px; color: {theme.get('text_secondary')}; min-width: 80px;")
        layout.addWidget(label_widget)
        
        value_widget = QLabel(value)
        value_widget.setStyleSheet(f"font-size: 12px; color: {theme.get('text_primary')};")
        value_widget.setWordWrap(True)
        layout.addWidget(value_widget, 1)
        
        return widget

    def _show_result_detail(self, task):
        """Show detailed result for a single task."""
        task_id = task.get("id")
        name = sanitize_for_qt(os.path.basename(task.get("file_path", "")))
        
        # Get full task details
        try:
            resp = requests.get(f"{API_BASE}/task/{task_id}", timeout=10)
            if resp.status_code != 200:
                QMessageBox.warning(self, "错误", f"无法获取任务详情 (HTTP {resp.status_code})")
                return
            detail = resp.json()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"获取详情失败: {e}")
            return
        
        # Build detail text
        lines = []
        lines.append(f"文件名: {name}")
        lines.append(f"状态: {detail.get('status', '未知')}")
        lines.append(f"进度: {detail.get('progress', 0)}%")
        
        # Parse analysis_details
        analysis_details_str = detail.get("analysis_details") or ""
        if analysis_details_str:
            try:
                details = json.loads(analysis_details_str)
                
                # MD5
                md5_info = details.get("md5")
                if md5_info:
                    lines.append("\n【MD5提取】")
                    lines.append(f"  哈希值: {md5_info.get('hash', 'N/A')}")
                    lines.append(f"  耗时: {md5_info.get('time', 0)}秒")
                
                # Scenes
                scenes = details.get("scenes", [])
                if scenes:
                    lines.append("\n【场景划分】")
                    lines.append(f"  场景数量: {len(scenes)}")
                    for i, scene in enumerate(scenes):
                        if scene.get("type") == "fallback":
                            lines.append(f"  场景 {i+1}: 回退模式 (固定抽帧)")
                        else:
                            start = scene.get("start", 0)
                            end = scene.get("end", 0)
                            lines.append(f"  场景 {i+1}: 帧 {start} - {end}")
                    scene_time = details.get("timing", {}).get("scene_detection", 0)
                    lines.append(f"  耗时: {scene_time}秒")
                
                # Frames
                frames = details.get("frames", [])
                if frames:
                    lines.append("\n【关键帧抽帧】")
                    total_frames = sum(f.get("count", 0) for f in frames)
                    lines.append(f"  总帧数: {total_frames}")
                    for frame_info in frames:
                        scene_idx = frame_info.get("scene_index", 0)
                        count = frame_info.get("count", 0)
                        lines.append(f"  场景 {scene_idx + 1}: {count} 帧")
                    frame_time = details.get("timing", {}).get("frame_extraction", 0)
                    lines.append(f"  耗时: {frame_time}秒")
                
                # VLM Analysis
                vlm_raw = details.get("vlm_raw")
                if vlm_raw:
                    lines.append("\n【VLM分析识别】")
                    lines.append("  原始输出:")
                    lines.append(f"  {vlm_raw}")
                    vlm_time = details.get("timing", {}).get("vlm_analysis", 0)
                    lines.append(f"  耗时: {vlm_time}秒")
                
                # Tag Mapping
                mapped_tags = details.get("mapped_tags", [])
                unmatched = details.get("unmatched", [])
                if mapped_tags or unmatched:
                    lines.append("\n【标签映射】")
                    if mapped_tags:
                        lines.append(f"  已映射标签 ({len(mapped_tags)}个):")
                        for tag in mapped_tags:
                            lines.append(f"    • {tag}")
                    if unmatched:
                        lines.append(f"  未匹配标签 ({len(unmatched)}个):")
                        for tag in unmatched[:10]:  # 只显示前10个
                            lines.append(f"    • {tag}")
                        if len(unmatched) > 10:
                            lines.append(f"    ... 还有 {len(unmatched) - 10} 个")
                    mapping_time = details.get("timing", {}).get("tag_mapping", 0)
                    lines.append(f"  耗时: {mapping_time}秒")
                
                # Total time
                timing = details.get("timing", {})
                total_time = sum(timing.values())
                lines.append("\n【总耗时】")
                lines.append(f"  {total_time:.2f}秒")
                
            except (json.JSONDecodeError, TypeError) as e:
                lines.append(f"\n分析详情解析失败: {e}")
        
        # Frames path
        frames_path = detail.get("frames_path") or ""
        if frames_path:
            lines.append(f"\n帧文件目录: {frames_path}")
        
        # Show in dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"分析详情 - {name}")
        dialog.setMinimumSize(700, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText("\n".join(lines))
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Menlo', monospace;
                font-size: 13px;
                background: {theme().get('bg_secondary')};
                color: {theme().get('text_primary')};
                border: 1px solid {theme().get('border')};
                border-radius: {theme().get('radius_md')}px;
                padding: 12px;
            }}
        """)
        layout.addWidget(text_edit)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {theme().get('bg_secondary')};
                color: {theme().get('text_primary')};
                border: 1px solid {theme().get('border')};
                border-radius: {theme().get('radius_md')}px;
                padding: 6px 20px;
                font-size: {theme().get('font_size_md')}px;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec()

    def _view_analysis_log(self, task_id):
        """View analysis log for a task."""
        try:
            resp = requests.get(f"{API_BASE}/task/{task_id}/analysis-log", timeout=10)
            if resp.status_code == 404:
                QMessageBox.information(self, "提示", "该任务没有分析日志")
                return
            if resp.status_code != 200:
                QMessageBox.warning(self, "错误", f"获取日志失败 (HTTP {resp.status_code})")
                return
            
            data = resp.json()
            logs = data.get("logs", [])
            
            if not logs:
                QMessageBox.information(self, "提示", "该任务没有分析日志")
                return
            
            # Build log content
            lines = []
            for log in logs:
                unit_index = log.get("unit_index", 0)
                content = log.get("content", "")
                created_at = log.get("created_at", "")
                
                lines.append(f"=== Unit {unit_index} ===")
                lines.append(f"时间: {created_at}")
                lines.append("")
                
                # Try to parse as JSON and pretty print
                try:
                    parsed = json.loads(content)
                    lines.append(json.dumps(parsed, indent=2, ensure_ascii=False))
                except:
                    lines.append(content)
                
                lines.append("")
                lines.append("")
            
            # Show in dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"分析日志 - 任务 {task_id}")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText("\n".join(lines))
            text_edit.setStyleSheet(f"""
                QTextEdit {{
                    font-family: 'Menlo', monospace;
                    font-size: 12px;
                    background: {theme().get('bg_secondary')};
                    color: {theme().get('text_primary')};
                    border: 1px solid {theme().get('border')};
                    border-radius: {theme().get('radius_md')}px;
                    padding: 12px;
                }}
            """)
            layout.addWidget(text_edit)
            
            # Close button
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            close_btn = QPushButton("关闭")
            close_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {theme().get('bg_secondary')};
                    color: {theme().get('text_primary')};
                    border: 1px solid {theme().get('border')};
                    border-radius: {theme().get('radius_md')}px;
                    padding: 6px 20px;
                    font-size: {theme().get('font_size_md')}px;
                }}
            """)
            close_btn.clicked.connect(dialog.accept)
            btn_layout.addWidget(close_btn)
            layout.addLayout(btn_layout)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"获取日志失败: {e}")


# ── Main ─────────────────────────────────────────────────────────────

def _install_qt_translator(app: "QApplication") -> None:
    """加载 Qt 自带简体中文翻译。

    让 macOS 应用菜单里由 Qt 生成的系统项显示中文:
    AboutRole 合并项文案由 Qt 生成("About %1"), 安装 qtbase 翻译后显示为「关于 %1」。
    """
    from PySide6.QtCore import QTranslator, QLibraryInfo
    from pathlib import Path

    # 优先定位 PySide6 自带的翻译目录; QLibraryInfo 的路径在框架布局下可能不准, 作为兜底。
    import PySide6
    candidates = [
        Path(PySide6.__file__).resolve().parent / "Qt" / "translations",
        Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)),
    ]
    for directory in candidates:
        for name in ("qtbase_zh_CN", "qt_zh_CN"):
            qm_path = directory / f"{name}.qm"
            if not qm_path.exists():
                continue
            translator = QTranslator(app)  # app 持有, 避免被回收
            if translator.load(str(qm_path)):
                app.installTranslator(translator)


def _fix_dev_app_menu_names() -> None:
    """修正无主 bundle 场景(dev 壳 exec 外部 python)下 Qt 生成的菜单文案。

    此时 Qt 读不到 Info.plist 的 CFBundleName, 回退成 argv[0] 的文件名
    (desktop_app.py)。打包版主程序在 .app 内, 文案天然正确, 此修正无副作用。
    """
    try:
        from AppKit import NSApplication
        ns_app = NSApplication.sharedApplication()
        main_menu = ns_app.mainMenu()
        if not main_menu or main_menu.numberOfItems() == 0:
            return
        app_item = main_menu.itemAtIndex_(0)
        # 顶部应用菜单名(进程无 bundle 时可能显示 python/desktop_app.py)
        if app_item.title() and app_item.title() != "MediaTag":
            app_item.setTitle_("MediaTag")
        submenu = app_item.submenu()
        if not submenu:
            return
        for i in range(submenu.numberOfItems()):
            item = submenu.itemAtIndex_(i)
            title = item.title() or ""
            if "desktop_app" in title or title.startswith(("关于", "About")):
                item.setTitle_("关于 MediaTag")
                break
    except Exception as exc:  # 修正失败不影响运行
        print(f"⚠ 修正应用菜单名称失败(可忽略): {exc}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MediaTag")
    app.setApplicationDisplayName("MediaTag")
    app.setStyle("Fusion")

    # macOS 应用菜单(About/Quit 等)由 Qt 按系统语言生成, 加载简体中文翻译
    _install_qt_translator(app)
    # 应用调色板（让默认控件文字/背景跟随主题，初始为深色）
    theme().apply_palette()
    
    # Set application icon
    icon_path = "assets/app_icon.png"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
        # macOS: Set Dock icon using NSApplication
        if sys.platform == 'darwin':
            try:
                from AppKit import NSApplication, NSImage
                ns_app = NSApplication.sharedApplication()
                ns_image = NSImage.alloc().initByReferencingFile_(icon_path)
                ns_app.setApplicationIconImage_(ns_image)
            except ImportError:
                pass  # AppKit not available
    
    # Apply theme-aware stylesheet
    app.setStyleSheet(global_stylesheet())

    window = MainWindow()
    window.show()

    # ── 首装引导（P2）：无分析主题时模态引导——完成前不创建主窗体 ──
    # （定案：引导未完成不出主界面，避免用户带病使用报错；删光主题会重新触发）
    try:
        from ui.components.onboarding import OnboardingDialog, _needs_onboarding
    except ImportError:
        from components.onboarding import OnboardingDialog, _needs_onboarding
    if _needs_onboarding():
        ob = OnboardingDialog()
        ob.exec()
        if not ob.created_theme:
            # 未完成引导（无主题系统无法运行）——退出
            return

    # ── 授权模式探测（后台）：完整版/免费版只影响额度，不阻塞呈现 ──
    # 免费额度模式见 docs/界面设计/07-授权管理.md；无任何启动门/锁定窗。
    try:
        from licensing.startup import start_mode_detection
    except ImportError:
        from src.licensing.startup import start_mode_detection
    start_mode_detection(window.license_mode_ready)

    # 菜单栏原生同步完成后, 修正无 bundle 场景的应用菜单文案。
    # Qt 会在窗口激活等时机重同步菜单并覆盖标题, 故周期性复查(文案正确时为 no-op)。
    if sys.platform == "darwin":
        from PySide6.QtCore import QTimer
        _menu_fix_timer = QTimer()
        _menu_fix_timer.timeout.connect(_fix_dev_app_menu_names)
        _menu_fix_timer.start(1500)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
