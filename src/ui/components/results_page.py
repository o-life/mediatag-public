"""ResultsPage — New analysis results view with series aggregation and Grid View."""

import os
import json
import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

import subprocess
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QPushButton, QSizePolicy, QGridLayout, QDialog, QMessageBox, QMenu, QCheckBox,
    QComboBox, QDateEdit, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QObject
from PySide6.QtGui import QPixmap, QCursor, QImage, QColor, QFontMetrics

# 人物编号展示层转换（ACTOR_122 → 人物 122）：双路径导入，
# 与 desktop_app 的范式一致（包内相对导入优先，脚本态回退绝对导入）
try:
    from .display_names import actor_code_display
except ImportError:
    from ui.components.display_names import actor_code_display

# Import theme and API client if available
try:
    from .theme import (
        theme, table_style,
        button_primary_style, button_secondary_style, button_danger_style,
        sm_button_style, sm_combo_style, search_box_style,
        tag_chip_style, sidebar_panel_style, rgba_str,
    )
    from ..tokens import T
    from .api_client import APIClient
    from ..widgets.flow_layout import FlowLayout
    from ..widgets.async_loader import AsyncLoader
    from ..widgets.video_grid_widget import PaginationWidget
    from ..widgets.tree_row_delegate import TreeCountDelegate
except ImportError:
    from ui.tokens import T

    def theme():
        return {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#252525',
            'bg_card': '#2a2a2a',
            'text_primary': '#ffffff',
            'text_secondary': '#aaaaaa',
            'accent': '#007aff',
            'tag_bg': '#333333',
        }
    def table_style():
        return ""
    def button_primary_style():
        return ""
    def button_secondary_style():
        return ""
    def button_danger_style():
        return ""
    # 退化路径（相对导入不可用时）：不折行，仅保证模块可导入
    FlowLayout = QHBoxLayout

    class AsyncLoader(QObject):  # 占位
        finished = Signal(object)
        error = Signal(str)

        def __init__(self, *a, **k):
            super().__init__()
        def run(self):
            self.error.emit("异步加载不可用（导入失败）")
    class PaginationWidget:
        class _MockSignal:
            def connect(self, *args, **kwargs): pass
        def __init__(self, *args, **kwargs):
            self.page_changed = self._MockSignal()
        def update_pagination(self, *args, **kwargs): pass


class _FlowToolbar(QScrollArea):
    """承载 FlowLayout 的工具栏滚动区。

    QScrollArea 自身不回报 heightForWidth，父布局会按 sizeHint 分配高度
    （过高）；这里把内容件的折行高度显式回报给父布局，工具栏高度随
    折行数自适应。
    """

    def hasHeightForWidth(self):  # noqa: N802 - Qt 命名
        w = self.widget()
        return bool(w is not None and w.hasHeightForWidth())

    def heightForWidth(self, width):  # noqa: N802 - Qt 命名
        w = self.widget()
        if w is None:
            return 0
        inner = max(50, width - 2 * self.frameWidth())
        return w.heightForWidth(inner) + 2 * self.frameWidth()


def _load_face_pixmap(path) -> "QPixmap | None":
    """读取人脸图片为 QPixmap。

    必须走 OpenCV：当前 Qt 构建不含 JPEG 插件（QImageReader 支持的格式只有
    png/bmp/pbm/...），直接 QPixmap(jpg) 会得到空图。代码库其它地方
    （FileCardWidget / ActorDetailDialog）同样用 OpenCV 解码。
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import cv2
        img = cv2.imread(path)
    except Exception:
        img = None
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if not pixmap.isNull():
            return pixmap
    # 兜底：极少数环境 Qt 自带该格式插件
    pixmap = QPixmap(path)
    return None if pixmap.isNull() else pixmap


def _make_actor_face_preview(face_path) -> "QLabel":
    """为一张人脸封面创建置顶预览小窗（无图则返回 None）。

    用独立置顶窗而非 Qt tooltip：tooltip 富文本里的 <img> 对本地文件路径
    不保证渲染，自绘窗口更可控、也能统一描边样式。
    """
    pixmap = _load_face_pixmap(face_path)
    if pixmap is None:
        return None

    preview = QLabel()
    preview.setWindowFlags(
        Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
    )
    # 不接收鼠标事件：否则预览窗盖住光标会触发标签 leaveEvent → 预览闪烁
    preview.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    preview.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
    preview.setPixmap(pixmap.scaled(
        180, 180,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    ))
    preview.setStyleSheet(
        f"border: 2px solid {theme().get('accent')}; border-radius: 6px;"  # ui-lint:allow 遗留债（迁移批清零）
        "background: #000000; padding: 2px;"
    )
    preview.move(QCursor.pos() + QPoint(16, 16))
    preview.show()
    return preview


class _ActorHoverLabel(QLabel):
    """人物名标签：浮标悬浮时浮现其人脸封面，移开即消失。

    预览窗由标签自己持有（而非全局单例），这样相邻两个标签的
    enter/leave 交错时不会互相把对方的预览关掉。
    """

    def __init__(self, text: str, face_path=None, parent=None):
        super().__init__(text, parent)
        self._face_path = face_path
        self._preview = None
        # 显式设主题文字色：作为表格单元格控件时，不保证继承到应用级
        # QLabel 样式，默认调色板前景在深色底上可能不可读。
        self.setStyleSheet(f"color: {theme().get('text_primary')};")
        if face_path:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _close_preview(self):
        if self._preview is not None:
            self._preview.hide()
            self._preview.deleteLater()
            self._preview = None

    def enterEvent(self, event):  # noqa: N802 - Qt 命名
        self._close_preview()
        self._preview = _make_actor_face_preview(self._face_path)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt 命名
        self._close_preview()
        super().leaveEvent(event)

    def hideEvent(self, event):  # noqa: N802 - Qt 命名
        # 表格重绘/切页会隐藏标签，此时必须收掉预览，避免残留
        self._close_preview()
        super().hideEvent(event)


def face_scan_display(scan_status):
    """人脸识别状态 → (展示文本, 主题色键)。

    与首页任务表「人脸」列同一套口径（文件维度 face_scan_status，任务清空后仍保留）：
    complete / none / skipped / low_confidence / 未扫描。
    """
    if scan_status == "complete":
        return "✓ 已识别", "success"
    if scan_status == "none":
        return "🙅 无人脸", "text_secondary"
    if scan_status == "skipped":
        return "⏭ 跳过", "warning"
    if scan_status == "low_confidence":
        return "⚠ 低置信", "warning"
    return "—", "text_secondary"


class _TagsCellLabel(QLabel):
    """标签单元格：双击直接进入该文件的标签编辑。

    单元格是控件而非 item，双击不会走 QTableWidget.doubleClicked，
    所以在这里自己接鼠标事件并抛信号。
    """

    double_clicked = Signal(int)  # analysis_id

    def __init__(self, text: str, analysis_id, parent=None):
        super().__init__(text, parent)
        self._analysis_id = analysis_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip((text + "\n\n" if text else "") + "双击可编辑该文件的标签")

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton and self._analysis_id is not None:
            self.double_clicked.emit(self._analysis_id)
        super().mouseDoubleClickEvent(event)


class _TagsChipsCell(QWidget):
    """标签 chips 单元格容器（19§D4.8b）：chips 折行 + 双击整格编辑标签。

    chip QLabel 全部 WA_TransparentForMouseEvents——双击落到容器上，
    才能拿到整格的双击语义（单元格是控件而非 item，
    双击不会走 QTableWidget.doubleClicked）。
    """

    double_clicked = Signal(int)  # analysis_id

    def __init__(self, analysis_id, parent=None):
        super().__init__(parent)
        self._analysis_id = analysis_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("双击可编辑该文件的标签")

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton and self._analysis_id is not None:
            self.double_clicked.emit(self._analysis_id)
        super().mouseDoubleClickEvent(event)


class _WrappingListTable(QTableWidget):
    """列表视图表格：折行列（人物/标签）随列宽变化重算行高。

    QTableWidget 行高默认固定，折行内容会被裁掉；这里在列宽/窗口变化时，
    按当前列宽用 heightForWidth 算出每行所需高度。重入保护避免与 Qt 自身
    的布局-重绘互相触发。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wrap_cells = []  # [(row, col, widget)]
        self._fitting = False
        self._refit_pending = False
        # 纵向滚动条出现/消失会改变视口宽度 → 列宽变化 → 折行高度需重算
        self.verticalScrollBar().rangeChanged.connect(self._schedule_refit)

    def register_wrap_cell(self, row: int, col: int, widget: QWidget) -> None:
        self._wrap_cells.append((row, col, widget))

    def _schedule_refit(self, *args) -> None:
        """滚动条变化后补算一次行高（合并多次触发，避免与布局互相打架）。"""
        if self._refit_pending:
            return
        self._refit_pending = True
        QTimer.singleShot(0, self._do_refit)

    def _do_refit(self) -> None:
        self._refit_pending = False
        self.fit_wrapped_rows()

    def resizeEvent(self, event):  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        self.fit_wrapped_rows()

    def fit_wrapped_rows(self) -> None:
        """按单元格**实际宽度**重算每行所需高度。

        两个必须用实测值、不能用列宽推算的地方：
        1. 单元格控件比列窄（实测 140 列 → 123 控件），用列宽算会少算一整行；
        2. 单元格控件比行高矮（实测行高 148 → 控件 139），不补这 9px
           最后一行仍会被裁 —— 即使高度算对了。
        """
        if self._fitting or not self._wrap_cells:
            return
        self._fitting = True
        try:
            base = self.verticalHeader().defaultSectionSize()
            needed = {}
            inset = {}  # row -> 垂直内缩（行高 - 控件高）
            for row, col, widget in self._wrap_cells:
                try:
                    width = widget.width()
                    if width <= 0:  # 尚未布局 → 退回列宽估算
                        width = self.columnWidth(col)
                    if width <= 0:
                        continue
                    need = widget.heightForWidth(width)
                    if widget.height() > 0 and self.rowHeight(row) >= widget.height():
                        inset[row] = max(inset.get(row, 0),
                                         self.rowHeight(row) - widget.height())
                except RuntimeError:
                    # 单元格控件已被销毁（切页/重绘）→ 跳过
                    continue
                if need > 0:
                    needed[row] = max(needed.get(row, 0), need)

            for row, need in needed.items():
                height = max(base, need + inset.get(row, 0))
                if self.rowHeight(row) != height:
                    self.setRowHeight(row, height)
        finally:
            self._fitting = False


def format_file_size(size) -> str:
    """Format file size in human-readable form."""
    try:
        size = int(size or 0)
    except (TypeError, ValueError):
        return ""
    if size <= 0:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    return f"{size / (1024 * 1024 * 1024):.1f}GB"

class ConfirmDialog(QDialog):
    """Custom confirmation dialog replacing QMessageBox.question/warning.

    On macOS, global QPushButton stylesheets interfere with QMessageBox
    internal buttons, making them invisible. This dialog uses manually
    created QPushButton instances with explicit styles.
    """

    def __init__(self, parent, title, message, icon_type="warning",
                 confirm_text="确认", cancel_text="取消", default_confirm=False,
                 extra_confirm_text=None, extra_confirm_role="extra"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(440)  # ui-lint:allow 遗留债（迁移批清零）
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setObjectName("ConfirmDialog")

        t = theme()
        self._result = None

        self.setStyleSheet(f"""
            #ConfirmDialog {{
                background: {t.get('bg_card')};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)  # ui-lint:allow 遗留债（迁移批清零）
        layout.setSpacing(16)  # ui-lint:allow 遗留债（迁移批清零）

        # ── Top row: icon + message ──────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(12)  # ui-lint:allow 遗留债（迁移批清零）

        icon_label = QLabel()
        icon_map = {
            "warning": "⚠️",
            "question": "❓",
            "info": "ℹ️",
        }
        icon_label.setText(icon_map.get(icon_type, "⚠️"))
        icon_label.setStyleSheet("font-size: 32px; background: transparent;")  # ui-lint:allow 遗留债（迁移批清零）
        icon_label.setFixedWidth(40)  # ui-lint:allow 遗留债（迁移批清零）
        icon_label.setAlignment(Qt.AlignTop)
        top_row.addWidget(icon_label)

        msg_label = QLabel(message)
        msg_label.setStyleSheet(
            f"color: {t.get('text_primary')}; "
            f"font-size: {t.get('font_size_md')}px; "
            f"background: transparent;"
        )
        msg_label.setWordWrap(True)
        top_row.addWidget(msg_label, 1)
        layout.addLayout(top_row)

        # ── Bottom: buttons ─────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)  # ui-lint:allow 遗留债（迁移批清零）
        btn_layout.addStretch()

        # Cancel button (secondary)
        cancel_btn = QPushButton(cancel_text)
        cancel_btn.setMinimumWidth(80)
        cancel_btn.setFixedHeight(28)  # ui-lint:allow 遗留债（迁移批清零）
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_hover')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 4px 16px;  # ui-lint:allow 遗留债（迁移批清零）
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('border')};
            }}
            QPushButton:pressed {{
                background: {t.get('bg_secondary')};
            }}
        """)
        cancel_btn.clicked.connect(lambda: self._on_button("cancel"))
        btn_layout.addWidget(cancel_btn)

        # Confirm button (accent)
        confirm_btn = QPushButton(confirm_text)
        confirm_btn.setMinimumWidth(80)
        confirm_btn.setFixedHeight(28)  # ui-lint:allow 遗留债（迁移批清零）
        confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 4px 16px;  # ui-lint:allow 遗留债（迁移批清零）
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
                border-color: {t.get('accent')};
            }}
            QPushButton:pressed {{
                background: {t.get('bg_hover')};
            }}
        """)
        confirm_btn.clicked.connect(lambda: self._on_button("confirm"))
        btn_layout.addWidget(confirm_btn)

        # Optional extra confirm button (danger style, e.g. "删除记录和文件")
        if extra_confirm_text:
            extra_btn = QPushButton(extra_confirm_text)
            extra_btn.setMinimumWidth(100)
            extra_btn.setFixedHeight(28)  # ui-lint:allow 遗留债（迁移批清零）
            extra_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {t.get('bg_secondary')};
                    color: {t.get('danger', '#E5534B')};
                    border: 1px solid {t.get('danger', '#E5534B')};
                    border-radius: {t.get('radius_md')}px;
                    padding: 4px 16px;  # ui-lint:allow 遗留债（迁移批清零）
                    font-size: {t.get('font_size_md')}px;
                }}
                QPushButton:hover {{
                    background: {t.get('bg_hover')};
                }}
            """)
            extra_btn.clicked.connect(lambda: self._on_button(extra_confirm_role))
            btn_layout.addWidget(extra_btn)

        if default_confirm:
            confirm_btn.setDefault(True)
            confirm_btn.setFocus()
        else:
            cancel_btn.setDefault(True)
            cancel_btn.setFocus()

        layout.addLayout(btn_layout)

    def _on_button(self, role):
        self._result = role
        self.accept()

    @staticmethod
    def question(parent, title, message, default_confirm=False):
        """Show a question-style confirmation dialog.

        Returns QMessageBox.Yes if confirmed, QMessageBox.No otherwise.
        """
        dlg = ConfirmDialog(
            parent, title, message,
            icon_type="question",
            default_confirm=default_confirm,
        )
        dlg.exec()
        return QMessageBox.Yes if dlg._result == "confirm" else QMessageBox.No

    @staticmethod
    def warning(parent, title, message, default_confirm=False):
        """Show a warning-style confirmation dialog.

        Returns QMessageBox.Yes if confirmed, QMessageBox.No otherwise.
        """
        dlg = ConfirmDialog(
            parent, title, message,
            icon_type="warning",
            default_confirm=default_confirm,
        )
        dlg.exec()
        return QMessageBox.Yes if dlg._result == "confirm" else QMessageBox.No

    @staticmethod
    def delete_choice(parent, title, message):
        """删除操作的三选对话框(取消 / 仅删记录 / 记录+磁盘文件)。

        Returns:
            None      -> 取消
            "record"  -> 仅删除登记记录
            "file"    -> 删除登记记录 + 磁盘原文件
        """
        dlg = ConfirmDialog(
            parent, title, message, icon_type="warning",
            confirm_text="仅删除记录", cancel_text="取消",
            extra_confirm_text="删除记录和文件",
        )
        dlg.exec()
        result = dlg._result
        if result in ("record", "file"):
            return result
        return None

class FileCardWidget(QFrame):
    """素材封面卡（19§D4.8a）：16:9 封面 + 时长角标 + 文件名 + 标签 chips。

    纯展示卡：点击行为留给 D4.20 内层抽屉（P4 功能批接线），现仅勾选框
    与右键菜单可交互。宽度由网格按视口均分后传入——封面 16:9 需定宽定高。
    """

    delete_requested = Signal(int)  # emits analysis_id
    selection_changed = Signal(int, bool)  # emits (analysis_id, is_selected)

    def __init__(self, file_data: dict, card_width: int = 220, parent=None):
        super().__init__(parent)
        self.file_data = file_data
        self.is_selected = False
        self._card_w = max(190, int(card_width))
        self.setObjectName("FileCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(self._card_w)
        self._setup_ui()

    def _setup_ui(self):
        t = theme()
        cover_h = self._card_w * 9 // 16  # ui-lint:allow 16:9 宽高比
        self.setStyleSheet(f"""
            #FileCard {{
                background: {t.get('bg_primary')};
                border-radius: {t.get('radius_card')}px;
                border: 1px solid transparent;
            }}
            #FileCard:hover {{
                border: 1px solid {rgba_str(t.get('accent'), 0.5)};
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 封面 16:9（顶部圆角与卡片同弧 8）──────────────────
        self.cover_container = QFrame()
        self.cover_container.setObjectName("CoverContainer")
        self.cover_container.setFixedSize(self._card_w - 2, cover_h)  # ui-lint:allow -2 边框补偿
        self.cover_container.setStyleSheet(f"""
            #CoverContainer {{
                background: {t.get('bg_secondary')};
                border-top-left-radius: {t.get('radius_card')}px;
                border-top-right-radius: {t.get('radius_card')}px;
            }}
        """)
        cover_layout = QVBoxLayout(self.cover_container)
        cover_layout.setContentsMargins(0, 0, 0, 0)

        self.cover_label = QLabel()
        self.cover_label.setAlignment(Qt.AlignCenter)
        self.cover_label.setFixedSize(self._card_w - 2, cover_h)  # ui-lint:allow -2 边框补偿
        self.cover_label.setStyleSheet(
            f"color: {t.get('text_tertiary')}; font-size: 10px; background: transparent;")  # ui-lint:allow 遗留债（迁移批清零）
        self._load_cover()
        cover_layout.addWidget(self.cover_label)

        # 时长角标：右下 6,6（黑 60% 底白字 10、圆角 4，19§D4.8a）
        duration = self.file_data.get('duration') or 0
        try:
            dur = int(duration)
        except (TypeError, ValueError):
            dur = 0
        if dur > 0:
            badge = QLabel(f"{dur // 60}:{dur % 60:02d}", self.cover_container)
            badge.setStyleSheet(
                f"QLabel {{ font-size: {t.get('dur_badge_fs')}px; color: #FFFFFF;"
                f" background: rgba(0,0,0,0.6);"
                f" border-radius: {t.get('dur_badge_r')}px;"
                f" padding: 1px {t.get('dur_badge_pad_x')}px; }}")
            badge.adjustSize()
            badge.move(self.cover_container.width() - badge.width() - T.DUR_BADGE_CORNER,
                       self.cover_container.height() - badge.height() - T.DUR_BADGE_CORNER)
            badge.show()
            badge.raise_()

        main_layout.addWidget(self.cover_container)

        # ── 文字区 padding 7 9 9：文件名 12.5 单行省略 + 标签 chips ──
        self.basic_info = QWidget()
        self.basic_info.setStyleSheet("background: transparent;")
        basic_layout = QVBoxLayout(self.basic_info)
        basic_layout.setContentsMargins(T.CARD_META_PAD, T.CARD_META_PAD_TOP,
                                        T.CARD_META_PAD, T.CARD_META_PAD)
        basic_layout.setSpacing(0)

        file_name = self.file_data.get('file_name') or 'Unknown'
        lbl_name = QLabel(file_name)
        lbl_name.setStyleSheet(
            f"color: {t.get('text_primary')}; font-size: {t.get('font_small')}px;"
            f" background: transparent;")
        _text_w = self._card_w - 2 * T.CARD_META_PAD
        lbl_name.setFixedWidth(_text_w)
        _fm = QFontMetrics(lbl_name.font())
        lbl_name.setText(_fm.elidedText(file_name, Qt.ElideRight, _text_w))
        basic_layout.addWidget(lbl_name)

        tags = self.file_data.get('mapped_tags', [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        tags = [str(x) for x in (tags or []) if str(x).strip()]
        if tags:
            basic_layout.addSpacing(T.CHIP_MARGIN_TOP)
            chips_row = QHBoxLayout()
            chips_row.setContentsMargins(0, 0, 0, 0)
            chips_row.setSpacing(T.CHIP_GAP)
            for tag in tags[:3]:
                chip = QLabel(tag)
                chip.setStyleSheet(tag_chip_style())
                chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                chips_row.addWidget(chip)
            if len(tags) > 3:
                more = QLabel(f"+{len(tags) - 3}")
                more.setStyleSheet(tag_chip_style())
                more.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                chips_row.addWidget(more)
            chips_row.addStretch()
            basic_layout.addLayout(chips_row)

        main_layout.addWidget(self.basic_info)

        # 勾选框悬浮封面左上角（8,8）
        self.checkbox = QCheckBox(self.cover_container)
        self.checkbox.setGeometry(T.CHECKBOX_CORNER, T.CHECKBOX_CORNER,
                                  T.CHECKBOX_OVERLAY, T.CHECKBOX_OVERLAY)
        self.checkbox.setStyleSheet("QCheckBox { background: transparent; }")
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.checkbox.raise_()

    def _load_cover(self):
        """封面：cover_path → frames_path 首帧；16:9 扩展裁切铺满。"""
        cover_path = self.file_data.get('cover_path')
        if not cover_path:
            frames_path = self.file_data.get('frames_path')
            if frames_path and os.path.isdir(frames_path):
                frames = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
                if frames:
                    cover_path = os.path.join(frames_path, sorted(frames)[0])

        target_w = self.cover_label.width() or self._card_w - 2
        target_h = self.cover_label.height() or self._card_w * 9 // 16
        if not (cover_path and os.path.exists(cover_path)):
            self.cover_label.setText("No Cover")
            return
        pixmap = None
        try:
            import cv2
            img_bgr = cv2.imread(str(cover_path))
            if img_bgr is not None and cv2.cvtColor(
                    img_bgr, cv2.COLOR_BGR2GRAY).mean() >= 10:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                q_img = QImage(img_rgb.data.tobytes(), w, h, ch * w,
                               QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(q_img)
        except ImportError:
            pixmap = QPixmap(cover_path)
            if pixmap.isNull():
                pixmap = None
        except Exception:
            pixmap = None
        if pixmap is not None and not pixmap.isNull():
            pixmap = pixmap.scaled(target_w, target_h,
                                   Qt.KeepAspectRatioByExpanding,
                                   Qt.SmoothTransformation)
            self.cover_label.setPixmap(pixmap)

    def contextMenuEvent(self, event):
        """右键菜单：在文件夹中显示 / 删除该文件记录。"""
        menu = QMenu(self)
        reveal_action = menu.addAction("在文件夹中显示")
        menu.addSeparator()
        delete_action = menu.addAction("🗑 删除")
        action = menu.exec_(event.globalPos())
        if action == reveal_action:
            file_path = self.file_data.get('file_path') or self.file_data.get('folder_path', '')
            if file_path and os.path.exists(file_path):
                subprocess.run(['open', '-R', file_path])
        elif action == delete_action:
            self.delete_requested.emit(self.file_data.get('id'))

    def _on_checkbox_changed(self, state):
        """Handle checkbox state change."""
        # Qt.CheckState: Unchecked=0, PartiallyChecked=1, Checked=2
        self.is_selected = (state == 2)  # Checked
        self.selection_changed.emit(self.file_data.get('id'), self.is_selected)

    def set_selected(self, selected: bool):
        """Set selection state programmatically."""
        self.is_selected = selected
        self.checkbox.setChecked(selected)


class ActorAvatarWidget(QFrame):
    """Widget to display an actor's face and code/name."""
    def __init__(self, actor_data: dict, small=False, parent=None):
        super().__init__(parent)
        self.actor_data = actor_data
        size = 30 if small else 35
        self.setFixedSize(size, size + 20)  # ui-lint:allow 遗留债（迁移批清零）
        self.setStyleSheet(f"background: transparent; color: {theme().get('text_primary')}; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  # ui-lint:allow 遗留债（迁移批清零）

        # Face Image
        self.lbl_face = QLabel()
        self.lbl_face.setFixedSize(size, size)
        self.lbl_face.setAlignment(Qt.AlignCenter)
        t = theme()
        self.lbl_face.setStyleSheet(f"""
            QLabel {{
                background: {t.get('bg_hover')};
                border-radius: {size//2}px;
                border: 1px solid {t.get('border')};
            }}
        """)

        face_path = actor_data.get('cover_face') or actor_data.get('face_path')
        pixmap = _load_face_pixmap(face_path)
        if pixmap is not None:
            self.lbl_face.setPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self.lbl_face.setText("👤")

        layout.addWidget(self.lbl_face, alignment=Qt.AlignHCenter)

        if not small:
            name = actor_data.get('name')
            code = actor_code_display(actor_data.get('code', '?'))
            display_text = name if name else code
            lbl_name = QLabel(display_text[:5]) # Truncate for small space
            t = theme()
            lbl_name.setStyleSheet(f"color: {t.get('text_primary')}; font-size: 8px; font-weight: bold;")  # ui-lint:allow 遗留债（迁移批清零）
            lbl_name.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl_name)



class _KVEditor(QTableWidget):
    """项目自定义字段编辑器：key-value 自由行，可增/删/改。"""

    def __init__(self, parent=None, fields=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["字段", "值"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setMinimumHeight(120)
        for k, v in (fields or {}).items():
            self.add_row(k, v)

    def add_row(self, key="", value=""):
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QTableWidgetItem(str(key)))
        self.setItem(r, 1, QTableWidgetItem(str(value)))

    def remove_selected_rows(self):
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)

    def to_dict(self):
        out = {}
        for r in range(self.rowCount()):
            k = (self.item(r, 0).text() if self.item(r, 0) else "").strip()
            v = (self.item(r, 1).text() if self.item(r, 1) else "").strip()
            if k:
                out[k] = v
        return out


class ProjectDialog(QDialog):
    """新建/编辑项目：客户 + 名称 + 类型 + 日期 + 状态 + 备注 + 自定义字段。"""

    STATUSES = ["拍摄中", "选片中", "修图中", "已交付", "已归档"]

    def __init__(self, api, parent=None, project=None, preset_client=""):
        super().__init__(parent)
        self.api = api
        self._project = project or {}
        self.setWindowTitle("编辑项目" if project else "新建项目")
        self.resize(520, 560)

        layout = QVBoxLayout(self)
        form = QHBoxLayout()
        form_left = QVBoxLayout()
        form_right = QVBoxLayout()

        # 客户（可下拉选既有客户，也可输入新名字创建）
        form_left.addWidget(QLabel("客户（可输入新名字创建）"))
        self.client_combo = QComboBox()
        self.client_combo.setEditable(True)
        self.client_combo.setMinimumWidth(200)
        form_left.addWidget(self.client_combo)

        # 活动类型
        form_right.addWidget(QLabel("活动类型"))
        self.type_combo = QComboBox()
        self.type_combo.setEditable(True)
        form_right.addWidget(self.type_combo)

        # 项目名（留空自动生成）
        form_left.addWidget(QLabel("项目名（留空自动生成）"))
        self.name_edit = QLineEdit()
        form_left.addWidget(self.name_edit)

        # 拍摄日期
        form_right.addWidget(QLabel("拍摄日期"))
        from PySide6.QtCore import QDate
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_check = QCheckBox("填写拍摄日期")
        self.date_check.setChecked(True)
        form_right.addWidget(self.date_edit)
        form_right.addWidget(self.date_check)

        form.addLayout(form_left)
        form.addLayout(form_right)
        layout.addLayout(form)

        # 状态
        srow = QHBoxLayout()
        srow.addWidget(QLabel("状态"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(self.STATUSES)
        srow.addWidget(self.status_combo)
        srow.addStretch()
        layout.addLayout(srow)

        # 备注
        layout.addWidget(QLabel("备注"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(70)
        layout.addWidget(self.notes_edit)

        # 自定义字段（key-value 自由行）
        cf_row = QHBoxLayout()
        cf_label = QLabel("自定义字段（可自行增加/修改/删除）")
        cf_row.addWidget(cf_label)
        cf_row.addStretch()
        btn_add_cf = QPushButton("+ 添加行")
        btn_add_cf.setFixedWidth(90)  # ui-lint:allow 遗留债（迁移批清零）
        btn_add_cf.clicked.connect(lambda: self.kv.add_row())
        btn_del_cf = QPushButton("删除选中行")
        btn_del_cf.setFixedWidth(110)  # ui-lint:allow 遗留债（迁移批清零）
        btn_del_cf.clicked.connect(self.kv_remove_selected)
        cf_row.addWidget(btn_add_cf)
        cf_row.addWidget(btn_del_cf)
        layout.addLayout(cf_row)
        self.kv = _KVEditor()
        layout.addWidget(self.kv)

        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("保存")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        btns.addStretch()
        layout.addLayout(btns)

        self._loaders = []  # AsyncLoader 引用，防 GC
        # macOS HIG 响应性：先弹窗给反馈，客户/类型词表后台加载（原先是
        # 构造时同步调两个接口——接口慢/未响应时主线程挂住，点击转彩球）
        self._pending_project = project
        self._pending_preset = preset_client
        # 加载态用**占位提示**而非占位条目——可编辑下拉里塞"加载中…"条目
        # 会被当成已输入文本：加载完成后恢复逻辑把它填回，保存时轻则静默
        # 选中第一个客户、重则建出名为"加载中…"的客户（2026-09-30 用户截图）
        self.client_combo.setPlaceholderText("加载中…")
        self.type_combo.setPlaceholderText("加载中…")
        self._load_master_data_async()
        self._fill(project, preset_client)

    def _load_master_data_async(self):
        """客户与类型词表：后台线程拉取，主线程回填（对话框先立即可见）。"""

        def fetch():
            # size 上限 100（超限 422）：曾用 200 导致词表永远加载失败
            clients = self.api.get(
                "/api/studio/clients", params={"size": 100}
            ).get("clients", [])
            types = self.api.get("/api/studio/event-types").get("event_types", [])
            return clients, types

        loader = AsyncLoader(fetch)
        loader.finished.connect(self._on_master_data_loaded)
        loader.error.connect(self._on_master_data_failed)
        self._loaders.append(loader)
        loader.run()

    def _on_master_data_loaded(self, data):
        """词表到位 → 回填下拉；同时恢复编辑态/预填客户的选择。"""
        try:
            clients, types = data
            cur_client = self.client_combo.currentText()
            self.client_combo.clear()
            for c in clients:
                self.client_combo.addItem(c.get("name") or f"客户{c.get('id')}", c.get("id"))
            self.type_combo.clear()
            self.type_combo.addItem("")
            for et in types:
                if et.get("active", 1):
                    self.type_combo.addItem(et.get("name"), et.get("id"))
            # 恢复用户可能已输入的文字 / 编辑态的既有选择
            # （占位文本不算输入——editable combo 空时 currentText 为 ""，
            #   防御性再挡一次旧占位字面量）
            if cur_client and cur_client != "加载中…" and cur_client != "加载中...":
                self.client_combo.setEditText(cur_client)
            self._apply_identity_selection()
            self.client_combo.setPlaceholderText("选择或输入新客户名")
            self.type_combo.setPlaceholderText("可留空")
        except RuntimeError:
            pass  # 对话框已被关闭销毁

    def _on_master_data_failed(self, msg: str):
        """词表加载失败：下拉退化为自由输入，不打断填写。"""
        try:
            self.client_combo.clear()
            self.client_combo.setEditText("")
            self.type_combo.clear()
            self.client_combo.setPlaceholderText("词表加载失败，直接输入客户名")
            self.type_combo.setPlaceholderText("可留空")
        except RuntimeError:
            pass

    def _apply_identity_selection(self):
        """按编辑态项目/预填客户设置客户与类型下拉。"""
        project = self._pending_project or {}
        if project.get("client_name"):
            idx = self.client_combo.findText(project["client_name"])
            if idx >= 0:
                self.client_combo.setCurrentIndex(idx)
            else:
                self.client_combo.setEditText(project["client_name"])
        elif self._pending_preset:
            self.client_combo.setEditText(self._pending_preset)
        if project.get("event_type"):
            tidx = self.type_combo.findText(project["event_type"])
            if tidx >= 0:
                self.type_combo.setCurrentIndex(tidx)

    def kv_remove_selected(self):
        for r in sorted({i.row() for i in self.kv.selectedIndexes()}, reverse=True):
            self.kv.removeRow(r)

    def _fill(self, project, preset_client):
        """立即可见的静态字段回填；客户/类型下拉待词表后台加载后由
        _apply_identity_selection 二次应用。"""
        if preset_client:
            self.client_combo.setEditText(preset_client)
        if not project:
            return
        self.name_edit.setText(project.get("name") or "")
        if project.get("event_date"):
            from PySide6.QtCore import QDate
            y, m, d = (int(x) for x in project["event_date"].split("-"))
            self.date_edit.setDate(QDate(y, m, d))
            self.date_check.setChecked(True)
        else:
            self.date_check.setChecked(False)
        status = project.get("status") or "拍摄中"
        sidx = self.status_combo.findText(status)
        self.status_combo.setCurrentIndex(sidx if sidx >= 0 else 0)
        self.notes_edit.setPlainText(project.get("notes") or "")
        for k, v in (project.get("custom_fields") or {}).items():
            self.kv.add_row(k, v)

    def _resolve_client_id(self):
        """下拉选中既有客户，或按输入名新建客户。返回 client_id。"""
        name = self.client_combo.currentText().strip()
        if not name:
            raise ValueError("客户名不能为空")
        idx = self.client_combo.currentIndex()
        cid = self.client_combo.itemData(idx)
        if cid:
            return int(cid)
        # 新客户
        r = self.api.post("/api/studio/clients", json={"name": name})
        return int(r.get("id"))

    def payload(self):
        if not self.date_check.isChecked():
            event_date = None
        else:
            event_date = self.date_edit.date().toString("yyyy-MM-dd")
        return {
            "client_name_input": self.client_combo.currentText().strip(),
            "name": self.name_edit.text().strip(),
            "event_type_name": self.type_combo.currentText().strip(),
            "event_date": event_date,
            "status": self.status_combo.currentText(),
            "notes": self.notes_edit.toPlainText(),
            "custom_fields": self.kv.to_dict(),
        }

    def save(self):
        """创建/更新项目（含新客户新建与类型解析）。返回 project_id。"""
        data = self.payload()
        # 类型名 → id（空则 None）
        type_name = data.pop("event_type_name", "")
        type_id = self.type_combo.itemData(self.type_combo.currentIndex()) if type_name else None
        client_id = self._resolve_client_id()
        body = {
            "client_id": client_id,
            "name": data["name"],
            "event_type_id": type_id,
            "event_date": data["event_date"],
            "status": data["status"],
            "notes": data["notes"],
            "custom_fields": data["custom_fields"],
        }
        if self._project.get("id"):
            self.api.patch(f"/api/studio/projects/{self._project['id']}",
                           json=body)
            return self._project["id"]
        r = self.api.post("/api/studio/projects", json=body)
        return int(r.get("id"))

class ResultsPage(QWidget):
    """Main Results Page with Scroll Area."""

    go_tasks_requested = Signal()  # 空态「去导入任务」→ 切回任务页（19§D4.14）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_videos = set()  # 存储选中的视频 ID
        self._selected_only = False  # 「已选中」筛选按钮点亮 = 只看勾选的文件
        self._scope_filter = None  # None=全部; 'unassigned'=未归类; int=project_id
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 分页状态
        self._current_page = 1
        self._page_size = 20  # 默认值，首次显示时会重新计算
        self._last_total = 0
        self._all_videos = []  # 所有视频数据缓存
        self._list_table = None  # 列表视图表格（宫格视图下为 None）
        self._list_checkboxes = []  # [(FileAnalysis.id, QCheckBox)]，列表视图行勾选框
        self._data_loaded = False  # 数据是否已加载
        self._page_size_calculated = False  # page_size 是否已计算（防止重复计算）

        # 动态布局防抖定时器
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_resize_timeout)

        # 当前布局参数
        self._current_columns = 0
        self._current_rows = 0

        # AsyncLoader 引用，防 GC（项目树后台拉取等；_load_data 用单独的 _loader）
        self._loaders = []

        self._setup_ui()
        self._setup_content()
        self.api = APIClient(base_url="http://localhost:8000") if 'APIClient' in globals() else None
        self._load_data()
        self._refresh_scope_tree()

    def _setup_ui(self):
        t = theme()
        self.setStyleSheet(f"background: {t.get('bg_primary')}; color: {t.get('text_primary')};")

        # 项目管理树迁入主窗口统一侧栏（12 号：与红绿灯头部区方案一致），
        # 本页只剩内容列。面板无父级构建，由 take_project_panel() 交接。
        self.project_panel = QWidget()
        # 实色垫底（D8）：面板不涂色时透明树直接透出 macOS 27 窗口背面
        # 玻璃（树列发黑）——任务页 source_panel 同款 bg_secondary
        self.project_panel.setStyleSheet(sidebar_panel_style())
        pp_v = QVBoxLayout(self.project_panel)
        pp_v.setContentsMargins(T.TREE_WRAP_X, 0, T.TREE_WRAP_X, T.TREE_WRAP_BOTTOM)  # 18§3.8
        pp_v.setSpacing(0)  # 标题行与树零缝（D7，同任务页配方）

        from PySide6.QtWidgets import QTreeWidget

        # 19§D3.1 标题行「项 目」：高 28、字 11/500/txt2 字距 0.5。
        # （原 4 小钮「新增项目/按名称/展开/收拢」按预览-17 撤出 Chrome——
        # 功能收敛到右键菜单：新建项目/新建子文件夹/编辑/删除/排序切换）
        title_row = QFrame()
        title_row.setFixedHeight(T.TREE_TITLE_H)
        _th = QHBoxLayout(title_row)
        _th.setContentsMargins(10, 0, 8, 0)  # ui-lint:allow 遗留债（迁移批清零）
        _th.setSpacing(4)  # ui-lint:allow 遗留债（迁移批清零）
        _lbl = QLabel("项 目")
        _lbl.setStyleSheet(
            f"font-size: {t.get('font_hint')}px; font-weight: 500; "
            f"color: {t.get('text_secondary')}; letter-spacing: 0.5px;")
        _th.addWidget(_lbl)
        _th.addStretch()
        # project-assets-roster：常驻新建入口（2026-09-30 用户令：建项目
        # 不依赖右键菜单；空库时这是唯一显眼入口）
        self.btn_new_project = QPushButton("＋")
        self.btn_new_project.setFixedSize(T.CTRL_SM_H, T.CTRL_SM_H)
        self.btn_new_project.setCursor(Qt.PointingHandCursor)
        self.btn_new_project.setToolTip("新建项目（客户可输入新名字即时创建）")
        self.btn_new_project.setStyleSheet(
            f"QPushButton {{ border: none; background: transparent;"
            f" color: {t.get('text_tertiary')};"
            f" font-size: {t.get('font_section')}px; font-weight: 600; }}"
            f" QPushButton:hover {{ color: {t.get('text_primary')}; }}")
        self.btn_new_project.clicked.connect(self._new_project)
        _th.addWidget(self.btn_new_project)
        pp_v.addWidget(title_row)

        # 空态引导（无项目时显示；project-assets-roster 问题1：初始化
        # 无项目且无处可建 → 现在有常驻入口 + 显式引导）
        self._scope_empty_hint = QLabel(
            "还没有项目\n\n点上方「＋」新建项目\n再把素材归集进来")
        self._scope_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scope_empty_hint.setStyleSheet(
            f"color: {t.get('text_tertiary')};"
            f" font-size: {t.get('font_secondary')}px; background: transparent;")
        self._scope_empty_hint.hide()
        pp_v.addWidget(self._scope_empty_hint)

        self.scope_tree = QTreeWidget()
        self.scope_tree.setHeaderHidden(True)  # 标题即上方「项 目」行
        # 19§D3.3：行规格同 18§3.3-3.7（委托单点实现），本树有真实层级
        # → 子级缩进 26；branch 装饰关（缩进本身表达层级）
        self.scope_tree.setIndentation(T.TREE_INDENT_PROJECT)
        self.scope_tree.setRootIsDecorated(False)
        self.scope_tree.setUniformRowHeights(True)
        # D7 零缝真机制=标题行与树 spacing 0（下方 pp_v）；不要再用视口
        # 负内缩——QSS polish 会把 QAbstractScrollArea 的程序化
        # contentsMargins 清零（实测 show 后读回 0），留了也是死代码。
        self.scope_tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.scope_tree.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; }")
        self._scope_delegate = TreeCountDelegate(self.scope_tree)
        self.scope_tree.setItemDelegate(self._scope_delegate)
        self.scope_tree.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.scope_tree.setMouseTracking(True)
        self.scope_tree.itemClicked.connect(self._on_scope_clicked)
        self.scope_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scope_tree.customContextMenuRequested.connect(self._on_scope_context_menu)
        pp_v.addWidget(self.scope_tree, 1)

    def take_project_panel(self):
        """把项目管理树面板交给主窗口统一侧栏（重复调用返回 None）。"""
        panel, self.project_panel = self.project_panel, None
        return panel

    def take_detail_bar(self):
        """把顶带操作面板（刷新/搜索/视图/排序）交给主窗口（重复调用返回 None）。"""
        bar, self._detail_bar = self._detail_bar, None
        return bar

    def restyle_band_bar(self):
        """顶带换肤钩子：五控件已迁出页面，页面级样式表够不到它们。

        _apply_toolbar_button_styles 直接对控件 setStyleSheet，与父级无关，
        重放即可（顺带刷新内容区工具栏，无副作用）。"""
        self._apply_toolbar_button_styles()

    def _setup_content(self):
        """页面内容列：操作行 + 素材面板（滚动区+分页底栏）。

        19§D-4：零边距满铺；素材面板=bg_card 直角（左右贴边）；
        分页栏并入面板底栏（D4.17，非独立行）。
        """
        from PySide6.QtWidgets import QButtonGroup
        t = theme()
        right_container = QWidget()
        right_container.setStyleSheet(f"background: {t.get('bg_primary')}; color: {t.get('text_primary')};")
        main_layout = QVBoxLayout(right_container)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 满铺零边距（18§4.0）
        main_layout.setSpacing(0)  # 操作行贴面板（19§D4.1 贴素材面板；契约在守）
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(right_container)

        # Action Bar（流式工具栏：窄窗口自动折行，按钮保持自然可读尺寸。
        # 承载方式与标签管理页审核工具栏一致：FlowLayout 放进 QScrollArea，
        # 由滚动区接管 heightForWidth——直接挂在普通 QWidget 上时，折行高度
        # 与子项定位在本机环境不可靠（实测按钮整体下坠被面板裁切））
        self._bar_inner = QWidget()
        self._bar_inner.setStyleSheet("background: transparent; border: none;")
        action_layout = FlowLayout(self._bar_inner, margin=0, spacing=T.SP_8)
        # 左右 10 呼吸、上下 4：容器高 28+8=36 与顶带槽等高（用户 2026-09-29：
        # 两行带高度须一致、大 padding 空白无意义）
        action_layout.setContentsMargins(T.TOOLBAR_MARGIN, T.SP_4,
                                         T.TOOLBAR_MARGIN, T.SP_4)

        # 顺序=D4.1 定稿：全选/已选中 N/归集到项目/删除/重新识别人脸
        # +（spring）+ 清理失效（末尾，中性）；刷新/搜索/视图/排序在顶带（D6）

        # 顶带操作面板（无父级构建，经 take_detail_bar() 交接主窗口）
        # 顶带控件统一 sm 档：高 26、padding 0 12、字 12（19§D1.2）
        self._detail_bar = QWidget()
        db_h = QHBoxLayout(self._detail_bar)
        db_h.setContentsMargins(0, 0, 0, 0)
        db_h.setSpacing(8)  # ui-lint:allow 遗留债（迁移批清零）

        # 刷新（上移顶带）
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self._load_data)
        self.btn_refresh.setFixedHeight(T.CTRL_SM_H)
        db_h.addWidget(self.btn_refresh)

        # 搜索框（上移顶带；回车搜索，可清除）
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 搜索文件名")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedWidth(T.SEARCH_W)
        self.search_box.setFixedHeight(T.CTRL_SM_H)
        self.search_box.setContentsMargins(0, 0, 0, 0)
        self.search_box.returnPressed.connect(self._on_search_box_return)
        db_h.addWidget(self.search_box)

        # 全选（内容区保留）
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self._toggle_select_all)
        action_layout.addWidget(self.btn_select_all)

        # 已选中 N 项 → 可点击的筛选按钮：点亮后只看勾选的文件（内容区保留）
        self.btn_selected_filter = QPushButton("已选中 0 项")
        self.btn_selected_filter.setCheckable(True)
        self.btn_selected_filter.setCursor(Qt.PointingHandCursor)
        self.btn_selected_filter.setToolTip("点击后只显示勾选的文件；再点一次恢复全部")
        self.btn_selected_filter.toggled.connect(self._on_selected_filter_toggled)
        action_layout.addWidget(self.btn_selected_filter)

        # 归集到项目（影楼 A 方案：多选素材 → 挂到项目；内容区保留）
        self.btn_assign_project = QPushButton("归集到项目")
        self.btn_assign_project.setEnabled(False)
        self.btn_assign_project.setToolTip("把选中的素材挂到项目（可新建项目）")
        self.btn_assign_project.clicked.connect(self._assign_to_project)
        action_layout.addWidget(self.btn_assign_project)

        # 导出（影楼 S4：勾选/当前项目范围 → 命名模板 + JPG 规格 + XMP）
        self.btn_export = QPushButton("导出")
        self.btn_export.setToolTip(
            "导出到文件夹（复制语义，源文件与库内登记不动）：\n"
            "勾选内容或当前项目范围 · 命名模板 · 照片转码规格 · XMP sidecar")
        self.btn_export.clicked.connect(self._export_items)
        action_layout.addWidget(self.btn_export)

        # 自动归集（影楼 S4.5：未归集内容按文件名厂牌分组 → 预览确认 → 建项目）
        self.btn_autoclass = QPushButton("自动归集")
        self.btn_autoclass.setToolTip(
            "按文件名中的厂牌自动分组未归集内容：\n"
            "预览分组与命名（可编辑、可取消勾选），确认后建项目并归集")
        self.btn_autoclass.clicked.connect(self._auto_classify_by_brand)
        action_layout.addWidget(self.btn_autoclass)

        # 视图切换（上移顶带）：宫格/列表双钮互斥组（19§D1.4）
        self.btn_view_grid = QPushButton("宫格")
        self.btn_view_list = QPushButton("列表")
        self.btn_view_grid.setToolTip("宫格视图（封面卡）")
        self.btn_view_list.setToolTip("列表视图")
        self.btn_view_grid.setCheckable(True)
        self.btn_view_list.setCheckable(True)
        self.btn_view_grid.setFixedHeight(T.CTRL_SM_H)
        self.btn_view_list.setFixedHeight(T.CTRL_SM_H)
        self.btn_view_grid.clicked.connect(lambda: self._set_view_mode("grid"))
        self.btn_view_list.clicked.connect(lambda: self._set_view_mode("list"))
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        self._view_group.addButton(self.btn_view_grid)
        self._view_group.addButton(self.btn_view_list)
        self.btn_view_grid.setChecked(True)
        db_h.addWidget(self.btn_view_grid)
        db_h.addWidget(self.btn_view_list)

        # 排序控件（08 方案：多维度 × 升降序，选择持久化；上移顶带）
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["默认", "文件名", "大小", "时长", "导入时间"])
        self.sort_combo.setFixedWidth(T.SORT_COMBO_W)
        self.sort_combo.setFixedHeight(T.CTRL_SM_H)
        self.sort_combo.setToolTip("排序依据")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        db_h.addSpacing(T.SP_8)
        db_h.addWidget(self.sort_combo)
        self.btn_sort_dir = QPushButton("↓ 降序")
        self.btn_sort_dir.setToolTip("切换升序/降序")
        self.btn_sort_dir.setFixedHeight(T.CTRL_SM_H)
        self.btn_sort_dir.clicked.connect(self._toggle_sort_dir)
        db_h.addWidget(self.btn_sort_dir)
        self._view_mode = "grid"  # grid / list

        # 删除按钮（内容区保留）
        self.btn_batch_delete = QPushButton("删除")
        self.btn_batch_delete.setEnabled(False)
        self.btn_batch_delete.clicked.connect(self._batch_delete)
        action_layout.addWidget(self.btn_batch_delete)

        # 重新识别人脸按钮（内容区保留）
        self.btn_rebuild_faces = QPushButton("重新识别人脸")
        self.btn_rebuild_faces.setEnabled(False)
        self.btn_rebuild_faces.setToolTip("重新识别选中视频的人脸数据")
        self.btn_rebuild_faces.clicked.connect(self._rebuild_faces)
        action_layout.addWidget(self.btn_rebuild_faces)

        # 标签编辑入口是「双击列表里的标签列」（针对单个文件）——
        # 标签需定位到帧，批量追加文字没有意义，故不设工具栏按钮。

        # 清理失效（D4.1：末尾、字 12/txt2 中性；与主组隔 16——
        # FlowLayout 无 addSpacing，用透明垫片）
        self.btn_cleanup = QPushButton("清理失效")
        self.btn_cleanup.clicked.connect(self._cleanup_stale)
        _spring_gap = QWidget()
        _spring_gap.setFixedSize(T.SP_8, T.SP_4)
        _spring_gap.setStyleSheet("background: transparent;")
        action_layout.addWidget(_spring_gap)
        action_layout.addWidget(self.btn_cleanup)

        self._apply_toolbar_button_styles()

        # 载入持久化的排序偏好
        self._load_sort_pref()

        self.action_bar = _FlowToolbar()
        self.action_bar.setWidgetResizable(True)
        self.action_bar.setFrameShape(QFrame.Shape.NoFrame)
        self.action_bar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.action_bar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.action_bar.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.action_bar.setWidget(self._bar_inner)
        self.action_bar.setVisible(True)
        main_layout.addWidget(self.action_bar)

        # 素材面板（D4.8：底 bg_card、直角满铺；D4.17：分页栏并入面板底栏）
        self.media_panel = QWidget()
        self.media_panel.setObjectName("MediaPanel")
        self.media_panel.setStyleSheet(
            f"QWidget#MediaPanel {{ background-color: {t.get('bg_card')}; }}")
        mp_v = QVBoxLayout(self.media_panel)
        mp_v.setContentsMargins(0, 0, 0, 0)
        mp_v.setSpacing(0)

        # Scroll Area（面板内滚动区：宫格/列表/空态 hero 共用此容器）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background: transparent;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)  # ui-lint:allow 遗留债（迁移批清零）

        self.scroll_area.setWidget(self.content_widget)
        mp_v.addWidget(self.scroll_area, 1)

        # 分页控件（D4.17：面板底栏，高 30、顶部 1px line、空库隐藏）
        self.pagination = PaginationWidget(compact=True)
        self.pagination.page_changed.connect(self._on_page_changed)
        self.pagination.page_size_changed.connect(self._on_page_size_changed)
        mp_v.addWidget(self.pagination)

        main_layout.addWidget(self.media_panel, 1)

    # 顶带 sm 档样式统一点：theme.sm_button_style / sm_combo_style /
    # search_box_style（样式构建器全部走 token，本页不再自带 QSS）
    def _sm_btn_style(self, checked: bool = False) -> str:
        return sm_button_style(checked)

    def _sm_combo_style(self) -> str:
        return sm_combo_style()

    def _enforce_one_tier_controls(self):
        """一档制钳高（17 号规范）：必须在样式设置之后调用——
        QSS polish 会清掉程序化 min/max。
        顶带控件=sm 档 26（19§D1.2）；内容区控件=28 唯一档。
        注意：构造期内容区按钮还挂在无父级的 _bar_inner 上，
        findChildren 够不到——内容区按属性显式钳，findChildren 只作
        showEvent 之后的兜底清扫。"""
        for _w in (self.btn_refresh, self.search_box, self.btn_view_grid,
                   self.btn_view_list, self.sort_combo, self.btn_sort_dir):
            try:
                _w.setFixedHeight(T.CTRL_SM_H)
            except RuntimeError:
                pass
        for _w in (self.btn_select_all, self.btn_selected_filter,
                   self.btn_assign_project, self.btn_batch_delete,
                   self.btn_rebuild_faces, self.btn_cleanup):
            try:
                _w.setFixedHeight(T.CTRL_H)
            except RuntimeError:
                pass
        band = {id(w) for w in (self.btn_refresh, self.search_box,
                                self.btn_view_grid, self.btn_view_list,
                                self.sort_combo, self.btn_sort_dir)}
        for _w in (self.findChildren(QPushButton) + self.findChildren(QComboBox)):
            if id(_w) in band:
                continue
            try:
                _w.setFixedHeight(T.CTRL_H)
            except RuntimeError:
                pass

    def _apply_toolbar_button_styles(self):
        """工具栏按钮统一样式（主题中性色按钮、自然尺寸），换肤时重放。"""
        t = theme()
        # 顶带四件套：sm 档（视图切换的选中态由 _set_view_mode 单独刷）
        for btn in (self.btn_refresh, self.btn_sort_dir):
            btn.setStyleSheet(self._sm_btn_style())
        self.search_box.setStyleSheet(search_box_style())
        self.sort_combo.setStyleSheet(self._sm_combo_style())
        # 内容区工具栏：28 中性钮
        for btn in (self.btn_select_all, self.btn_rebuild_faces,
                    self.btn_assign_project):
            btn.setStyleSheet(button_secondary_style())
        # 筛选点亮态：accent.14 底 + accent 字/边（19§D4.2，跟主题 accent）
        self.btn_selected_filter.setStyleSheet(button_secondary_style() + f"""
            QPushButton:checked {{
                color: {t.get('accent')};
                border-color: {rgba_str(t.get('accent'), 0.4)};
                background: {rgba_str(t.get('accent'), 0.14)};
            }}
        """)
        # 删除=danger 字（4.4）；清理失效=末尾中性次级（D4.1）
        self.btn_batch_delete.setStyleSheet(button_danger_style())
        self.btn_cleanup.setStyleSheet(button_secondary_style() + f"""
            QPushButton {{ font-size: {t.get('font_secondary')}px; color: {t.get('text_secondary')};
                border-color: transparent; background: transparent; }}
            QPushButton:hover {{ color: {t.get('text_primary')};
                background: {t.get('btn_glass_hover')}; }}
        """)
        self._set_view_mode(getattr(self, "_view_mode", "grid"), _rerender=False)
        self._enforce_one_tier_controls()

    def _set_view_mode(self, mode: str, _rerender: bool = True):
        """视图切换（宫格/列表双钮互斥，19§D1.4）+ 选中态样式重放。"""
        self._view_mode = mode
        for btn, on in ((self.btn_view_grid, mode == "grid"),
                        (self.btn_view_list, mode == "list")):
            btn.setChecked(on)
            btn.setStyleSheet(self._sm_btn_style(checked=on))
        if _rerender:
            # 切换视图只影响"怎么画"，数据已在 _all_videos 里。
            # 原先调 _load_data() 会重新联网拉全量、重置勾选并跳回第 1 页。
            self._render_current_page()

    def apply_theme(self):
        """换肤：仅重放样式不重建控件，保留页面状态。"""
        from .theme import theme
        t = theme()
        self.setStyleSheet(f"background: {t.get('bg_primary')}; color: {t.get('text_primary')};")
        # 操作行=透明行（D4.1：无卡片底）；素材面板=bg_card 直角满铺（D4.8）
        self._bar_inner.setStyleSheet("background: transparent; border: none;")
        self.media_panel.setStyleSheet(
            f"QWidget#MediaPanel {{ background-color: {t.get('bg_card')}; }}")
        # 树列面板实色垫底（D8：透明会透出窗口背面玻璃）；交接后引用置 None
        if self.project_panel is not None:
            self.project_panel.setStyleSheet(sidebar_panel_style())
        self._apply_toolbar_button_styles()
        self.pagination.apply_theme()
        self.content_widget.setStyleSheet("background: transparent;")
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        # 卡片/委托色随渲染取主题——重渲染当前页让存量卡片吃到新色
        self._render_current_page()

    def _grid_card_size(self):
        """按当前视口推算宫格卡尺寸（与 _render_current_page 同一公式）。"""
        vw = self.scroll_area.viewport().width() or self.width() or 0
        cols = max(1, (vw - 2 * T.GRID_MARGIN + T.GRID_GAP)
                   // (T.GRID_CARD_MIN_W + T.GRID_GAP))
        card_w = max(T.GRID_CARD_MIN_W,
                     (vw - 2 * T.GRID_MARGIN - T.GRID_GAP * (cols - 1)) // cols)
        card_h = card_w * 9 // 16 + 64  # 封面 16:9 + meta（名+chips+padding）
        return vw, cols, card_w, card_h

    def _calculate_columns(self):
        """计算可容纳的列数（基于面板视口，非整窗）"""
        vw, cols, _cw, _ch = self._grid_card_size()
        return cols

    def _calculate_rows(self):
        """计算可容纳的行数：ceil——至少填满一屏（用户 2026-09-29 裁定），
        视口富余时同屏更多，尾部超出部分页面内滚动。"""
        vw, cols, card_w, card_h = self._grid_card_size()
        vh = self.scroll_area.viewport().height() or self.height() or 0
        avail = max(0, vh - 2 * T.GRID_MARGIN)
        rows = max(1, -(-avail // (card_h + T.GRID_GAP)))  # ceil
        return rows

    def _calculate_page_size(self):
        """根据面板视口计算每页卡片数（列 × ceil 行）。"""
        vh = self.scroll_area.viewport().height()
        vw = self.scroll_area.viewport().width()
        if vw <= 0 or vh <= 0:
            return None
        page_size = self._calculate_columns() * self._calculate_rows()
        return max(1, min(400, page_size))

    def showEvent(self, event):
        """页面显示时重置选择状态并计算 page_size。"""
        super().showEvent(event)
        # 一档制终检（17 号规范）：异步回填/QSS polish 可能清掉钳制
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._enforce_one_tier_controls)
        
        # 每次页面显示时重置选择状态，防止切换页面后勾选状态残留
        if self._data_loaded:
            self._reset_selection()
        
        if self._page_size_calculated:
            return

        page_size = self._calculate_page_size()
        if page_size is None:
            return

        self._page_size = page_size
        self._page_size_calculated = True


        if self._data_loaded:
            self._render_current_page()

    def _search_videos(self):
        """按文件名搜索视频（弹输入框，支持再次搜索清除）。"""
        from PySide6.QtWidgets import QInputDialog

        current = getattr(self, '_search_query', '')
        text, ok = QInputDialog.getText(
            self, "搜索视频", "输入文件名关键词（留空清除搜索）:",
            text=current,
        )
        if not ok:
            return
        self._search_query = text.strip().lower()
        self._current_page = 1
        self._render_current_page()

    # ── S3.4 素材工作台：范围树 ────────────────────────────────────

    def _refresh_scope_tree(self):
        """从 /api/studio 拉项目（文件夹），按 parent_id 重建嵌套范围树。

        「项目」= 文件夹语义（12 号方案）：parent_id 为空即顶层；
        每个节点计数 = 含全部子级的素材总数。
        拉取走后台线程（D5）：原先在构造/首切时于主线程同步发两个请求
        （其一结果还被人丢弃），API 慢时详情页首切卡顿数秒。
        """
        if getattr(self, "_scope_loading", False):
            return
        self._scope_loading = True

        def fetch():
            return self.api.get(
                "/api/studio/projects", params={"size": 100}
            ).get("projects", [])

        loader = AsyncLoader(fetch)
        loader.finished.connect(self._on_scope_loaded)
        loader.error.connect(self._on_scope_load_failed)
        self._loaders.append(loader)
        loader.run()

    def _on_scope_load_failed(self, _message):
        self._scope_loading = False

    def _on_scope_loaded(self, projects):
        self._scope_loading = False
        if not isinstance(projects, list):
            projects = []
        from PySide6.QtWidgets import QTreeWidgetItem

        # 排序模式：name（按名称）/ created（按创建时间），同级内排序
        mode = getattr(self, "_scope_sort_mode", "name")
        projects = sorted(
            projects,
            key=lambda p: ((p.get("name") or "").lower(), p.get("id") or 0)
            if mode == "name"
            else (p.get("created_at") or "", p.get("id") or 0),
            reverse=mode != "name",
        )
        self._projects_flat = projects

        self.scope_tree.clear()

        # 常设项计数：全部素材=当前文件表总数；未归集=不在任何项目清单
        # （多归属口径按 project_ids；文件表与项目表分接口异步加载，先到先显）
        videos = getattr(self, "_all_videos", []) or []
        top_all = QTreeWidgetItem(["全部素材"])
        top_all.setData(0, Qt.ItemDataRole.UserRole, None)
        if videos:
            top_all.setData(0, Qt.ItemDataRole.UserRole + 1, len(videos))
        self.scope_tree.addTopLevelItem(top_all)

        top_un = QTreeWidgetItem(["未归集"])
        top_un.setData(0, Qt.ItemDataRole.UserRole, "unassigned")
        if videos:
            top_un.setData(0, Qt.ItemDataRole.UserRole + 1,
                           sum(1 for v in videos if not v.get("project_ids")))
        self.scope_tree.addTopLevelItem(top_un)

        # 空态引导（project-assets-roster：初始化无项目 → 显式入口引导）
        if hasattr(self, "_scope_empty_hint"):
            self._scope_empty_hint.setVisible(not projects)

        # 直接素材数 → 向上合计（含全部子级）
        direct = {p.get("id"): (p.get("asset_count") or 0) for p in projects}
        parent_of = {p.get("id"): p.get("parent_id") for p in projects}

        def rollup(pid):
            total = direct.get(pid, 0)
            for p in projects:
                if p.get("parent_id") == pid:
                    total += rollup(p["id"])
            return total

        nodes = {}

        def make_node(prj):
            # 计数走 UserRole+1，由 TreeCountDelegate 右对齐绘制（18§3.5）
            node = QTreeWidgetItem([prj.get("name") or ""])
            node.setData(0, Qt.ItemDataRole.UserRole, prj.get("id"))
            node.setData(0, Qt.ItemDataRole.UserRole + 1, rollup(prj.get("id")))
            return node

        # 先建所有节点再挂父子，避免父项排在后面时丢层级
        for prj in projects:
            nodes[prj.get("id")] = make_node(prj)
        for prj in projects:
            pid = prj.get("id")
            par = parent_of.get(pid)
            if par is not None and par in nodes:
                nodes[par].addChild(nodes[pid])
            else:
                self.scope_tree.addTopLevelItem(nodes[pid])

        self.scope_tree.expandAll()

    def _toggle_scope_sort(self):
        """排序切换：按名称 ↔ 按创建时间（同级内排序，嵌套不变）。

        （入口由原「按名称」小钮改为树右键菜单——19§D3.1 预览稿无小钮行）"""
        self._scope_sort_mode = (
            "created" if getattr(self, "_scope_sort_mode", "name") == "name" else "name"
        )
        self._refresh_scope_tree()

    def _scope_project_ids(self, root_id):
        """文件夹点击 → 该文件夹及全部子级的 id 集合（素材按归属过滤）。"""
        ids = {root_id}
        for p in getattr(self, "_projects_flat", []):
            if p.get("parent_id") in ids and p.get("id") not in ids:
                # 父级已在集合 → 子级加入；扁平列表顺序不定，多轮收敛
                ids.add(p["id"])
        # 多轮收敛（深层嵌套）
        for _ in range(3):
            for p in getattr(self, "_projects_flat", []):
                if p.get("parent_id") in ids:
                    ids.add(p["id"])
        return ids

    def _on_scope_clicked(self, item, col):
        """范围树点击 → 设置过滤条件 → 重渲染当前页（客户端过滤，零请求）。"""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            self._scope_filter = None
        elif data == "unassigned":
            self._scope_filter = "unassigned"
        else:
            self._scope_filter = data
        self._current_page = 1
        self._render_current_page()

    # ── 排序（08 方案）────────────────────────────────────────────

    def _sort_videos(self, videos):
        """按操作条所选维度排序；缺省维度保持原始顺序。"""
        key = self.sort_combo.currentIndex() if hasattr(self, "sort_combo") else 0
        desc = getattr(self, "_sort_desc", True)
        if key == 0:
            return videos

        def k_name(v):
            return (v.get("file_name") or "").lower()

        def k_size(v):
            return float(v.get("file_size") or 0)

        def k_duration(v):
            return float(v.get("duration") or 0)

        def k_created(v):
            return v.get("created_at") or ""

        funcs = {1: k_name, 2: k_size, 3: k_duration, 4: k_created}
        return sorted(videos, key=funcs.get(key, k_name), reverse=desc)

    def _on_sort_changed(self):
        self._save_sort_pref()
        self._current_page = 1
        self._render_current_page()

    def _toggle_sort_dir(self):
        self._sort_desc = not getattr(self, "_sort_desc", True)
        self.btn_sort_dir.setText("↓ 降序" if self._sort_desc else "↑ 升序")
        self._save_sort_pref()
        self._render_current_page()

    def _save_sort_pref(self):
        from PySide6.QtCore import QSettings
        st = QSettings()
        st.setValue("results/sort_key", self.sort_combo.currentIndex())
        st.setValue("results/sort_desc", bool(getattr(self, "_sort_desc", True)))

    def _load_sort_pref(self):
        from PySide6.QtCore import QSettings
        st = QSettings()
        try:
            key = int(st.value("results/sort_key", 0))
        except Exception:
            key = 0
        self.sort_combo.setCurrentIndex(max(0, min(key, self.sort_combo.count() - 1)))
        raw = st.value("results/sort_desc", "true")
        self._sort_desc = raw in (True, "true", "1")
        self.btn_sort_dir.setText("↓ 降序" if self._sort_desc else "↑ 升序")

    # ── 归集到项目（影楼 A 方案）──────────────────────────────────

    def _assign_to_project(self):
        """把选中素材挂到项目（可顺手新建项目）。"""
        if not self.selected_videos:
            return
        task_ids = sorted(self.selected_videos)
        try:
            # 服务端 size 上限 100（le=100，size=200 会 422）——分页取全，
            # 项目超过一页时菜单不静默截断
            projects: list = []
            page = 1
            while True:
                resp = self.api.get(
                    "/api/studio/projects", params={"size": 100, "page": page})
                batch = resp.get("projects", [])
                projects.extend(batch)
                if not batch or page * 100 >= resp.get("total", 0):
                    break
                page += 1
        except Exception as exc:
            QMessageBox.warning(self, "读取项目失败", str(exc))
            return

        menu = QMenu(self)
        act_new = menu.addAction("＋ 新建项目…")
        if projects:
            menu.addSeparator()
        chosen_pid = None
        for prj in projects:
            act = menu.addAction(f'{prj.get("name")}（{prj.get("asset_count", 0)}）')
            act.setData(prj.get("id"))
        chosen = menu.exec(self.btn_assign_project.mapToGlobal(
            self.btn_assign_project.rect().topLeft()))
        if chosen is None:
            return
        try:
            if chosen is act_new:
                dlg = ProjectDialog(self.api, self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                chosen_pid = dlg.save()
            else:
                chosen_pid = chosen.data()
            r = self.api.post(
                f"/api/studio/projects/{chosen_pid}/assign",
                json={"task_ids": task_ids},
            )
            # 多归属语义：重复加入幂等——命中已在清单的内容给出标注反馈
            dups = r.get("duplicates", 0)
            if dups:
                QMessageBox.information(
                    self, "归集到项目",
                    f"新增归集 {r.get('assigned', 0)} 个内容，"
                    f"{dups} 个已在该项目中（未重复计入）")
        except Exception as exc:
            QMessageBox.warning(self, "归集失败", str(exc))
            return
        self.selected_videos.clear()
        self._update_action_bar_state()
        self._refresh_scope_tree()
        self._load_data(silent=True)

    def _export_items(self):
        """导出（影楼 S4）：勾选内容 / 当前项目范围 → 命名模板+JPG 规格+XMP。

        复制语义：源文件与库内登记不动；范围候选与详情页过滤同口径
        （项目树选中项=子树，多归属交集）。"""
        from .project_export import run_export

        scope = getattr(self, "_scope_filter", None)
        if scope == "unassigned":
            scope_name = "未归集"
            candidates = [v for v in self._all_videos
                          if not v.get("project_ids")]
        elif scope is not None and not isinstance(scope, str):
            ids = self._scope_project_ids(scope)
            candidates = [v for v in self._all_videos
                          if set(v.get("project_ids") or []) & ids]
            scope_name = next(
                (p.get("name") or "项目" for p in getattr(self, "_projects_flat", [])
                 if p.get("id") == scope), "项目")
        else:
            scope_name = "全部素材"
            candidates = list(self._all_videos)

        run_export(self, candidates, scope_name,
                   checked_count=len(self.selected_videos))

    def _auto_classify_by_brand(self):
        """按文件名厂牌自动归集（S4.5）：未归集 → 分组预览 → 确认后建项目。"""
        from .brand_autoclass import BrandAutoclassDialog, execute_autoclass
        from src.core.brand_classifier import group_by_brand

        candidates = [v for v in self._all_videos
                      if not v.get("project_ids")]
        if not candidates:
            QMessageBox.information(self, "自动归集", "没有未归集的内容")
            return
        groups = group_by_brand(candidates)
        named_members = sum(v["count"] for k, v in groups.items()
                            if k != "__singletons__")
        unclassified = len(candidates) - named_members

        dlg = BrandAutoclassDialog(self, groups, unclassified=unclassified)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected_groups()
        if not selected:
            return
        execute_autoclass(self, self.api, selected)
        self._refresh_scope_tree()
        self._load_data(silent=True)

    # ── 项目管理：新建 / 编辑字段 / 删除 ─────────────────────────

    def _new_project(self):
        """新建项目（表单含自定义字段 key-value 行）。"""
        try:
            dlg = ProjectDialog(self.api, self)
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            dlg.save()
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._refresh_scope_tree()

    def _find_scope_project(self, item):
        """树项 → 项目 dict（含 id/name）；非项目节点返回 None。"""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, int):
            for p in getattr(self, "_projects_flat", []):
                if p.get("id") == data:
                    return p
            return {"id": data, "name": item.text(0)}
        return None

    def _on_scope_context_menu(self, pos):
        """文件夹树右键：新增子文件夹 / 编辑字段 / 删除（空白处=新增顶层）。"""
        item = self.scope_tree.itemAt(pos)
        prj = self._find_scope_project(item) if item is not None else None

        menu = QMenu(self)
        _sort_label = ("按创建时间排序" if getattr(self, "_scope_sort_mode", "name") == "name"
                       else "按名称排序")
        if prj is None:
            # 空白处 / 「全部素材」节点：顶层新增 + 排序切换
            act_new = menu.addAction("＋ 新增项目")
            act_sort = menu.addAction(_sort_label)
            chosen = menu.exec(self.scope_tree.mapToGlobal(pos))
            if chosen == act_new:
                self._new_project()
            elif chosen == act_sort:
                self._toggle_scope_sort()
            return

        act_sub = menu.addAction("＋ 新增子文件夹")
        act_edit = menu.addAction("✎ 编辑字段…")
        act_sort2 = menu.addAction(_sort_label)
        menu.addSeparator()
        act_del = menu.addAction("🗑 删除（含子级）")
        chosen = menu.exec(self.scope_tree.mapToGlobal(pos))
        if chosen == act_sub:
            self._new_subfolder(prj)
        elif chosen == act_edit:
            self._edit_project_fields(prj["id"])
        elif chosen == act_sort2:
            self._toggle_scope_sort()
        elif chosen == act_del:
            self._delete_project_flow(prj["id"], item.text(0))

    def _new_subfolder(self, parent_prj: dict):
        """在某文件夹下新增子文件夹（轻量命名输入，客户继承根活动）。"""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self,
            "新增子文件夹",
            f"在「{parent_prj.get('name')}」下新建文件夹：",
        )
        if not ok or not name.strip():
            return
        try:
            self.api.post(
                "/api/studio/projects",
                json={"parent_id": parent_prj["id"], "name": name.strip()},
            )
        except Exception as exc:
            QMessageBox.warning(self, "新增失败", str(exc))
            return
        self._refresh_scope_tree()
        self._load_data(silent=True)

    def _edit_project_fields(self, project_id: int):
        """编辑项目：基础字段 + 自定义字段 key-value 行。"""
        try:
            project = self.api.get(f"/api/studio/projects/{project_id}")
        except Exception as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        dlg = ProjectDialog(self.api, self, project=project)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            dlg.save()
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._refresh_scope_tree()
        self._load_data(silent=True)

    def _delete_project_flow(self, project_id: int, label: str):
        """删除项目（含子级）：仅摘除清单行——内容回未归集，本体与结果保留。"""
        choice = ConfirmDialog.delete_choice(
            self,
            "删除项目",
            f"确定删除「{label}」吗？\n\n将移除该项目（含子级）的成员清单：\n"
            "其中的内容会回到「未归集」，文件与分析结果不受影响；\n"
            "同时属于其他项目的内容，其在其他项目中的归属也不受影响。",
            confirm_text="删除",
        )
        if choice is None:
            return
        try:
            self.api.delete(f"/api/studio/projects/{project_id}")
        except Exception as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self._refresh_scope_tree()
        self._load_data(silent=True)

    def _load_data(self, silent: bool = False):
        """触发数据加载（**异步**，不阻塞 UI）。

        silent=False（默认，用户主动：刷新 / 删除编辑后 / 切页）：
            清屏 + 显示「正在加载数据…」+ 重置勾选与页码。
        silent=True（**10 秒自动刷新**）：**不动界面** —— 不清屏、不丢勾选、
            不跳页；取回后若数据没变就**不重渲染**。否则每 10 秒清空重建一次，
            用户看到的就是"频繁刷新、多次重新加载"。

        重入保护：上一轮未完成时直接跳过。
        """
        if getattr(self, "_loading_data", False):
            return
        self._loading_data = True
        self._silent_load = silent

        if not self.api:
            self._loading_data = False
            if not silent:
                self._show_load_error("API Client 未初始化")
            return

        if not silent:
            # 只有用户主动重载才重置勾选（防止勾选状态残留）；
            # 后台自动刷新绝不能动用户已勾选的内容。
            self._reset_selection()
            self._show_loading_state()

        self._loader = AsyncLoader(self.api.get, "/videos")
        self._loader.finished.connect(self._on_videos_loaded)
        self._loader.error.connect(self._on_load_failed)
        self._loader.run()

    @staticmethod
    def _videos_signature(videos) -> tuple:
        """判断"数据是否真的变了"的轻量指纹（纯内存计算，无 IO）。

        只覆盖列表/卡片会显示出来的字段：状态、人脸状态、标签、更新时间。
        """
        return tuple(
            (v.get("id"), v.get("updated_at"), v.get("status"),
             v.get("face_scan_status"), tuple(v.get("mapped_tags") or []))
            for v in videos
        )

    def _show_loading_state(self):
        """清屏并显示「正在加载数据...」。"""
        t = theme()
        self._clear_content()
        loading_label = QLabel("正在加载数据...")
        loading_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: 15px;")  # ui-lint:allow 遗留债（迁移批清零）
        loading_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(loading_label)

    def _clear_content(self):
        """清空内容区并重置容器引用（加载前 / 渲染前都要做）。"""
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                # 移除卡片前 block checkbox 信号，防止 stateChanged 在销毁时
                # 触发选中回调污染 selected_videos（删除后顺延任务被选中的根因之一）
                for child in widget.findChildren(QCheckBox):
                    child.blockSignals(True)
                # 必须先 hide()：setParent(None) 会让控件变成**顶层窗口**，
                # 若它当时可见就会以无标题小窗形式留在屏幕上（实测：重渲染后
                # 出现一个装着「🎬 系列: …」内容的 512x475 孤儿窗）。
                # 再 deleteLater() 交给事件循环回收，避免悬挂引用。
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
            else:
                self.content_layout.removeItem(item)

        # 必须连同 grid_container/grid_layout 一起清掉 —— 它们是容器的子对象,
        # 容器被 setParent(None)+delattr 后 C++ 对象即销毁, 若这里留下悬空引用,
        # 下次切回宫格时 _get_all_cards() 访问它会抛
        # "Internal C++ object already deleted" → 渲染中断, 表现为"切不回宫格"。
        for attr in ('_single_works_container', '_no_actor_container',
                     '_single_works_cards', '_no_actor_cards', '_page_cards',
                     'grid_container', 'grid_layout', 'grid_container_no_actor', 'grid_layout_no_actor'):
            if hasattr(self, attr):
                delattr(self, attr)

    def _on_videos_loaded(self, resp):
        """工作线程取回 /videos 后，回到主线程缓存并渲染。"""
        if not isinstance(resp, list):
            self._on_load_failed(f"API 返回格式异常: {type(resp).__name__}")
            return

        silent = getattr(self, "_silent_load", False)
        signature = self._videos_signature(resp)
        if silent and signature == getattr(self, "_videos_sig", None):
            # 后台自动刷新但内容没变 → 什么都不做（这正是"频繁刷新"的来源）
            self._loading_data = False
            return
        self._videos_sig = signature

        prev_scroll = 0
        if silent and self._list_table is not None:
            prev_scroll = self._list_table.verticalScrollBar().value()

        self._all_videos = resp
        self._data_loaded = True
        self._loading_data = False
        if not silent:
            self._current_page = 1      # 只有用户主动重载才回到第一页

        self._render_current_page()

        if prev_scroll:
            # 重渲染会重建表格 → 尽量把滚动位置还回去，避免"看着看着跳回顶部"
            QTimer.singleShot(0, lambda: self._restore_list_scroll(prev_scroll))

    def _restore_list_scroll(self, value: int) -> None:
        table = self._list_table
        if table is None:
            return
        try:
            table.verticalScrollBar().setValue(value)
        except RuntimeError:
            pass  # 表已被销毁（切页/重渲染）

    def _on_load_failed(self, message: str):
        """加载失败：用户主动加载 → 错误页；后台静默刷新 → 保持界面不动。"""
        self._loading_data = False
        if getattr(self, "_silent_load", False):
            return
        self._show_load_error(message)

    def _show_load_error(self, message: str):
        """加载失败：清屏 + 错误提示 + 重试按钮。

        页面禁止直出原始异常（URL/Max retries 数百字符——标签管理/设置页
        同款治理，用户 2026-09-28 截图指正）：连接类错误显示一句可行动提示，
        原始信息走 stdout。"""
        print(f"[results] 加载失败: {message}", flush=True)
        if "Connection" in message or "Max retries" in message:
            message = "无法连接服务，请确认后端已启动（./start-api.sh）"
        else:
            message = "加载失败，请点「刷新」重试"
        self._loading_data = False
        self._clear_content()
        t = theme()
        error_label = QLabel(f"⚠ {message}")
        error_label.setStyleSheet(f"color: {t.get('danger')}; font-size: 15px; padding: 20px;")  # ui-lint:allow 遗留债（迁移批清零）
        error_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(error_label)

        retry_btn = QPushButton("重试")
        retry_btn.setFixedWidth(100)  # ui-lint:allow 遗留债（迁移批清零）
        retry_btn.clicked.connect(self._load_data)
        retry_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px 16px;  # ui-lint:allow 遗留债（迁移批清零）
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{ background: {t.get('bg_hover')}; }}
        """)
        retry_container = QWidget()
        retry_layout = QVBoxLayout(retry_container)
        retry_layout.setAlignment(Qt.AlignCenter)
        retry_layout.addWidget(retry_btn)
        self.content_layout.addWidget(retry_container)
        self.content_layout.addStretch()
        self._all_videos = []
        self._data_loaded = False

    def _render_current_page(self):
        """渲染当前页的视频（根据 _current_page 和 _page_size 截取数据）。

        注意：不要在 resizeEvent 中调用此方法，否则会导致无限循环。
        """
        t = theme()
        # 清空现有内容并重置容器引用（与加载时共用同一套清理）
        self._clear_content()

        if not self._data_loaded:
            self.content_layout.addStretch()
            return

        all_videos = self._all_videos
        # 按文件名搜索过滤
        search_query = getattr(self, '_search_query', '')
        if search_query:
            all_videos = [
                v for v in all_videos
                if search_query in (v.get('file_name') or '').lower()
                or search_query in (v.get('file_path') or '').lower()
            ]

        # S3.4 按素材范围过滤（左栏项目树选中项；多归属口径：project_ids
        # 与子树集合有交集即命中）
        scope = getattr(self, '_scope_filter', None)
        if scope == 'unassigned':
            all_videos = [v for v in all_videos if not v.get('project_ids')]
        elif scope is not None and not isinstance(scope, str):
            ids = self._scope_project_ids(scope)
            all_videos = [v for v in all_videos
                          if set(v.get('project_ids') or []) & ids]
        # 「已选中」筛选按钮点亮时 = 只看勾选的文件
        if getattr(self, '_selected_only', False):
            all_videos = [v for v in all_videos if v.get('id') in self.selected_videos]
        # 排序（08 方案：排序控件，选择持久化）
        all_videos = self._sort_videos(all_videos)
        total = len(all_videos)
        self._last_total = total

        # 截取当前页数据
        start_idx = (self._current_page - 1) * self._page_size
        end_idx = start_idx + self._page_size
        page_videos = all_videos[start_idx:end_idx]

        # 空数据状态
        if total == 0:
            # 空态 hero（19§D4.14）：面板（media_panel bg_card）已就位，
            # hero 本体透明、垂直居中；图标容器 64×64 圆角 16 底 glass（18§4.14）
            _hero = QWidget()
            _hero.setStyleSheet("background: transparent;")
            _hv = QVBoxLayout(_hero)
            _hv.setContentsMargins(0, 0, 0, 0)
            _hv.addStretch(2)
            _art = QLabel("🎬")
            _art.setAlignment(Qt.AlignCenter)
            _art.setFixedSize(T.HERO_ICON, T.HERO_ICON)
            _art.setStyleSheet(
                f"font-size: {t.get('hero_emoji_fs')}px;"
                f" background: {t.get('btn_glass_bg')}; "
                f"border-radius: {t.get('hero_icon_radius')}px;")
            _hv.addWidget(_art, 0, Qt.AlignHCenter)
            _t1 = QLabel("这里还没有素材")
            _t1.setAlignment(Qt.AlignCenter)
            _t1.setStyleSheet(
                f"font-size: {t.get('font_section')}px; font-weight: 600;"
                f" color: {t.get('text_primary')}; background: transparent;")
            _hv.addWidget(_t1, 0, Qt.AlignHCenter)
            for _line in ("到「任务」页导入视频或照片即可。导入的素材会自动保存到文件表，",
                          "并按左侧项目树显示在这里——无需手动导入到本页。"):
                _tl = QLabel(_line)
                _tl.setAlignment(Qt.AlignCenter)
                _tl.setStyleSheet(
                    f"font-size: {t.get('font_secondary')}px;"
                    f" color: {t.get('text_secondary')}; background: transparent;")
                _hv.addWidget(_tl, 0, Qt.AlignHCenter)
            _hv.addSpacing(T.HERO_BTN_GAP)
            _btnrow = QHBoxLayout()
            _btnrow.addStretch(1)
            _go = QPushButton("去导入任务")
            _go.setCursor(Qt.CursorShape.PointingHandCursor)
            _go.setFixedHeight(T.CTRL_H)
            _go.setStyleSheet(button_secondary_style())
            _go.clicked.connect(self.go_tasks_requested.emit)
            _btnrow.addWidget(_go)
            _btnrow.addStretch(1)
            _hv.addLayout(_btnrow)
            _hv.addStretch(3)
            self.content_layout.addWidget(_hero, 1)  # 拉伸填满（用户 2026-09-28：内容区稳定填充）
            self.pagination.update_pagination(self._current_page, 0, self._page_size)
            self.pagination.setVisible(False)  # 空库隐藏（§D4.17）
            return

        # ── 列表视图：直接渲染统一表格（不按人物分组） ──────────
        # 每页 N 条 = 表格 N 行，分页条数直观生效；表格占满剩余空间
        if getattr(self, "_view_mode", "grid") == "list":
            table = self._build_list_table(page_videos, stretch=True, track_selection=True)
            self.content_layout.addWidget(table, 1)  # 占满剩余高度
            self.pagination.update_pagination(self._current_page, total, self._page_size)
            self.pagination.setVisible(True)
            return

        # 进入宫格视图：丢弃列表视图的行勾选框登记，避免残留引用
        # 让 _update_action_bar_state / _toggle_select_all 走错分支。
        self._list_table = None
        self._list_checkboxes = []

        # 扁平网格（08 方案：移除按人物自动归集系列——实现效果不佳，
        # 连拍/相似归组留待期 2 AI culling 以数据驱动方式重做）
        # D4.8a：列宽≥190 auto-fill、间距 10、内边距 10（面板 bg_card
        # 由 media_panel 承担，容器透明）
        viewport_w = self.scroll_area.viewport().width() or self.width() or 0
        _unit = T.GRID_CARD_MIN_W + T.GRID_GAP
        columns = max(1, int((viewport_w - 2 * T.GRID_MARGIN) // _unit))
        self._current_columns = columns
        card_w = max(T.GRID_CARD_MIN_W,
                     (viewport_w - 2 * T.GRID_MARGIN - T.GRID_GAP * (columns - 1)) // columns)
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(T.GRID_MARGIN, T.GRID_MARGIN,
                                            T.GRID_MARGIN, T.GRID_MARGIN)
        self.grid_layout.setSpacing(T.GRID_GAP)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.content_layout.addWidget(self.grid_container, 1)  # 拉伸填满面板

        self._page_cards = []
        for idx, task in enumerate(page_videos):
            card = FileCardWidget(task, card_width=card_w)
            card.delete_requested.connect(self._delete_video)
            card.selection_changed.connect(self._on_card_selection_changed)
            self._page_cards.append(card)
            row = idx // columns
            col = idx % columns
            self.grid_layout.addWidget(card, row, col)

        # 更新分页控件
        self.pagination.update_pagination(self._current_page, total, self._page_size)
        self.pagination.setVisible(True)  # 有素材恢复（空库隐藏，§D4.17）

    def _build_actor_cell(self, actors: list) -> QWidget:
        """人物单元格：每个人物单独一个标签，悬浮显示其人脸封面。

        用单元格控件而非纯文本行，才能区分鼠标落在**哪一个**人物上
        （纯文本无法逐个悬浮）。用 FlowLayout 让多个人物自动折到下一行，
        而不是被一行裁掉。
        """
        holder = QWidget()
        layout = FlowLayout(holder, margin=0, spacing=8)
        layout.setContentsMargins(4, 2, 4, 2)  # ui-lint:allow 遗留债（迁移批清零）
        for a in actors or []:
            text = a.get("name") or a.get("code")
            if not text:
                continue
            layout.addWidget(_ActorHoverLabel(text, a.get("cover_face")))
        return holder

    def _build_tags_cell(self, tags: list, analysis_id=None) -> QWidget:
        """标签单元格（19§D4.8b chips）：chip 11/圆角4/底 glass/字 txt2，
        FlowLayout 自动折行；双击可编辑该文件标签。"""
        cell = _TagsChipsCell(analysis_id)
        cell.setStyleSheet("background: transparent;")
        layout = FlowLayout(cell, margin=0, spacing=T.CHIP_GAP)
        layout.setContentsMargins(T.TABLE_CELL_PAD_X, T.TABLE_CELL_PAD_Y,
                                  T.TABLE_CELL_PAD_X, T.TABLE_CELL_PAD_Y)
        for tag in tags or []:
            chip = QLabel(str(tag))
            chip.setStyleSheet(tag_chip_style())
            chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(chip)
        cell.double_clicked.connect(self._edit_tags_for_file)
        return cell

    def _edit_tags_for_file(self, analysis_id) -> None:
        """双击标签单元格 → 编辑这一个文件的标签（无需先勾选）。

        标签列表直接取自已加载的缓存（后端没有单条 GET /videos/{id}）。
        """
        from .tag_edit_dialog import TagEditDialog
        tags = next(
            (v.get("mapped_tags") or [] for v in self._all_videos if v.get("id") == analysis_id),
            [],
        )
        dialog = TagEditDialog(analysis_id, tags, parent=self)
        if dialog.exec():
            self._load_data()

    def _build_list_table(self, files: list, stretch: bool = False,
                          track_selection: bool = False) -> QTableWidget:
        """构建列表视图表格（勾选/文件名/人物/标签/大小/人脸/状态）。

        stretch=True 时表格高度伸展填满内容区（列表主视图用）。
        track_selection=True 时登记本表的行勾选框，使其与工具栏
        「归集到项目 / 重新识别人脸 / 删除」联动。

        列契约（19§D4.8b）：复选36 / 文件名(左,弹性) / 大小(右) / 时长(右)
        / 标签(chips) / ID(右)；表头 32、行 28、无竖线。
        """
        table = _WrappingListTable()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["", "文件名", "大小", "时长", "标签", "ID"]
        )
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(False)  # 行分隔 1px line，不用斑马纹
        table.setStyleSheet(table_style())
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(T.ROW_H_TABLE)  # 行高一档制
        table.horizontalHeader().setFixedHeight(T.HEADER_H)  # 表头 32（19§D4.8b）

        header = table.horizontalHeader()
        # 不开 stretchLastSection：最后一列（ID）定宽，剩余宽度归标签列
        header.setStretchLastSection(False)
        for _col, _mode in (
            (0, QHeaderView.ResizeMode.Fixed),             # 复选
            (1, QHeaderView.ResizeMode.Stretch),           # 文件名
            (2, QHeaderView.ResizeMode.ResizeToContents),  # 大小（右）
            (3, QHeaderView.ResizeMode.ResizeToContents),  # 时长（右）
            (4, QHeaderView.ResizeMode.Stretch),           # 标签 chips
            (5, QHeaderView.ResizeMode.ResizeToContents),  # ID（右）
        ):
            header.setSectionResizeMode(_col, _mode)
        table.setColumnWidth(0, T.CHECK_COL_W)  # 复选列 36（19§D4.8b）

        checkboxes = []
        table.setRowCount(len(files))
        for row, f in enumerate(files):
            analysis_id = f.get("id")

            # 0: 勾选框（普通 QCheckBox，不设样式，保留原生对勾）
            holder = QWidget()
            holder_layout = QHBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox = QCheckBox()
            checkbox.setToolTip("勾选后可对选中项批量「归集到项目 / 重新识别人脸 / 删除」")
            checkbox.setChecked(analysis_id in self.selected_videos)
            holder_layout.addWidget(checkbox)
            table.setCellWidget(row, 0, holder)
            if track_selection:
                checkbox.stateChanged.connect(
                    lambda state, aid=analysis_id: self._on_row_checkbox_changed(aid, state)
                )
                checkboxes.append((analysis_id, checkbox))

            name = f.get("file_name") or Path(f.get("file_path", "")).name or str(analysis_id)
            # 标签来自 mapped_tags —— /videos 并不返回 "tags" 键。
            tags = f.get("mapped_tags") or f.get("tags") or []

            name_item = QTableWidgetItem(name)
            table.setItem(row, 1, name_item)
            name_item.setToolTip(f.get("file_path") or name)

            size_item = QTableWidgetItem(format_file_size(f.get("file_size")))
            size_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 2, size_item)

            duration = f.get("duration") or 0
            try:
                dur = int(duration)
                dur_text = f"{dur // 60}:{dur % 60:02d}" if dur > 0 else "—"
            except (TypeError, ValueError):
                dur_text = "—"
            dur_item = QTableWidgetItem(dur_text)
            dur_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 3, dur_item)

            # 标签 chips（19§D4.8b）：折行单元格 → 登记按列宽重算行高
            tags_cell = self._build_tags_cell(tags, analysis_id)
            table.setCellWidget(row, 4, tags_cell)
            table.register_wrap_cell(row, 4, tags_cell)

            id_item = QTableWidgetItem(str(analysis_id))
            id_item.setForeground(QColor(theme().get("text_tertiary")))
            id_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 5, id_item)

        if track_selection:
            self._list_table = table
            self._list_checkboxes = checkboxes

        # 列表主视图：表格高度伸展填满内容区，避免下方大片空白
        if stretch:
            table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        return table

    def _on_row_checkbox_changed(self, analysis_id, state):
        """列表视图行勾选变化 → 同步 selected_videos。

        存的是 FileAnalysis.id，与宫格卡片保持一致（_edit_tags_for_file/_rebuild_faces
        都从卡片数据解析 task_id，两条路径口径相同）。
        """
        if analysis_id is None:
            return
        if state:
            self.selected_videos.add(analysis_id)
        else:
            self.selected_videos.discard(analysis_id)
        self._update_action_bar_state()

    def _toggle_view_mode(self):
        """切换宫格/列表视图（保留旧入口：单钮时代 API）。"""
        self._set_view_mode("list" if getattr(self, "_view_mode", "grid") == "grid"
                            else "grid")

    def _on_page_changed(self, page: int):
        """分页页码变化时重新渲染。"""
        self._current_page = page
        self.selected_videos.clear()
        self._update_action_bar_state()
        self._render_current_page()

    def _on_page_size_changed(self, page_size: int):
        """每页条数变化 → 更新 page_size 并回到第 1 页重新渲染。

        列表视图下生效；宫格视图由布局动态计算 page_size，
        切换每页条数时以用户选择为准。
        """
        self._page_size = page_size
        self._page_size_calculated = True
        self._current_page = 1
        self.selected_videos.clear()
        self._update_action_bar_state()
        self._load_data()


    def _delete_video(self, analysis_id: int):
        """移除工单（project-assets-roster：删除=仅移除工单，内容与分析结果保留）。"""
        dlg = ConfirmDialog(
            self,
            "确认移除工单",
            "移除该任务的分析工单？\n\n"
            "内容与分析结果会保留在库中，磁盘文件不受影响；\n"
            "同内容再次添加时直接复用既有结果，不会重新分析。",
            confirm_text="移除",
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # 从缓存里找 task_id
        task_id = None
        for v in self._all_videos:
            if v.get("id") == analysis_id:
                task_id = v.get("task_id")
                break

        if self.api:
            try:
                if task_id is not None:
                    resp = self.api.post("/videos/delete", json={
                        "task_ids": [task_id],
                    })
                else:
                    # 无 task_id 时退回旧端点(仅删记录)
                    resp = self.api.delete(f"/videos/{analysis_id}")
                if resp.get("success") or resp.get("status") == "success":
                    msg = resp.get("message", "已移除工单")
                    if resp.get("file_errors"):
                        msg += "\n" + "\n".join(resp["file_errors"])
                    QMessageBox.information(self, "删除完成", msg)
                    self._reset_selection()
                    self._load_data()
                else:
                    QMessageBox.warning(self, "删除失败", resp.get("message", "未知错误"))
            except Exception as e:
                QMessageBox.warning(self, "删除失败", str(e))

    def _cleanup_stale(self):
        """Clean up all stale analysis records where files no longer exist."""
        reply = ConfirmDialog.question(
            self,
            "确认清理",
            "将扫描所有分析记录，清理文件已不存在的失效记录。\n\n此操作不可恢复，是否继续？",
        )
        if reply != QMessageBox.Yes:
            return

        if self.api:
            try:
                resp = self.api.post("/videos/cleanup-stale")
                deleted = resp.get("deleted_count", 0)
                QMessageBox.information(self, "清理完成", f"已清理 {deleted} 条失效记录")
                self._load_data()
            except Exception as e:
                QMessageBox.warning(self, "清理失败", str(e))

    def _toggle_select_all(self):
        """Toggle select/deselect all visible rows."""
        if self._list_checkboxes:
            # 列表视图：全选/取消全选所有行勾选框
            any_unselected = any(not cb.isChecked() for _, cb in self._list_checkboxes)
            for _, cb in self._list_checkboxes:
                try:
                    cb.blockSignals(True)
                    cb.setChecked(any_unselected)
                finally:
                    cb.blockSignals(False)
            if any_unselected:
                self.selected_videos = {aid for aid, _ in self._list_checkboxes if aid is not None}
            else:
                self.selected_videos.clear()
            self._update_action_bar_state()
            return

        all_cards = self._get_all_cards()
        
        # If any card is unselected, select all; otherwise deselect all
        any_unselected = any(not card.is_selected for card in all_cards)
        
        for card in all_cards:
            card.set_selected(any_unselected)
        
        # Update selection set
        if any_unselected:
            self.selected_videos = {card.file_data.get('id') for card in all_cards}
        else:
            self.selected_videos.clear()
        
        self._update_action_bar_state()

    def _get_all_cards(self):
        """Get all FileCardWidget instances in the view.

        防御: grid_layout / grid_layout_no_actor 可能指向**已销毁的 C++ 对象**
        （其父容器在 _load_data/_render_current_page 里被拆除）。此时访问
        .count() 会抛 RuntimeError; 这里吞掉并视为"无卡片"，避免整个渲染中断
        （历史 bug: 切到列表后切不回宫格, 就是这里抛异常导致渲染没跑完）。
        """
        cards = []

        for attr in ('grid_layout', 'grid_layout_no_actor'):
            layout = getattr(self, attr, None)
            if layout is None:
                continue
            try:
                count = layout.count()
            except RuntimeError:
                # 底层布局已删除 → 清掉悬空引用, 视为无卡片
                try:
                    delattr(self, attr)
                except Exception:
                    pass
                continue
            for i in range(count):
                item = layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), FileCardWidget):
                    cards.append(item.widget())

        return cards

    def _reset_selection(self):
        """彻底重置选中状态：清空集合 + 所有现存卡片取消勾选。

        删除任务后必须调用——否则旧卡片的勾选状态可能在新列表渲染时
        残留/串位（用户反馈：删除 10 个任务后，顺延出现的 10 个任务仍被选中）。
        用 blockSignals 防止 setChecked 触发 stateChanged 污染 selected_videos。
        """
        self.selected_videos.clear()
        for card in self._get_all_cards():
            try:
                card.checkbox.blockSignals(True)
                card.checkbox.setChecked(False)
                card.is_selected = False
            finally:
                card.checkbox.blockSignals(False)
        # 列表视图的行勾选框同样要清 —— 否则删除/刷新后旧勾选会残留
        for _, cb in self._list_checkboxes:
            try:
                cb.blockSignals(True)
                cb.setChecked(False)
            finally:
                cb.blockSignals(False)
        self._update_action_bar_state()

    def _on_search_box_return(self):
        """搜索框回车 → 按文件名过滤（客户端过滤，回车即生效）。"""
        self._search_query = (self.search_box.text() or "").strip().lower()
        self._current_page = 1
        self._render_current_page()

    def _on_selected_filter_toggled(self, checked: bool):
        """「已选中」筛选按钮点亮/取消 → 回到第 1 页按勾选过滤渲染。"""
        self._selected_only = checked
        self._current_page = 1
        self._render_current_page()

    def _update_action_bar_state(self):
        """Update action bar visibility and button states based on selection."""
        count = len(self.selected_videos)
        
        # Always show action bar, just enable/disable buttons
        self.action_bar.setVisible(True)
        
        # Update count label
        self.btn_selected_filter.setText(f"已选中 {count} 项")
        
        # Enable/disable buttons
        self.btn_batch_delete.setEnabled(count > 0)
        self.btn_assign_project.setEnabled(count > 0)
        self.btn_rebuild_faces.setEnabled(count > 0)
        
        # Update select all button text
        if self._list_checkboxes:
            # 列表视图：以行勾选框为准
            all_selected = all(cb.isChecked() for _, cb in self._list_checkboxes)
        else:
            all_cards = self._get_all_cards()
            all_selected = all_cards and all(card.is_selected for card in all_cards)
        self.btn_select_all.setText("取消全选" if all_selected else "全选")

    def _on_card_selection_changed(self, analysis_id, is_selected):
        """Handle card selection change."""
        if is_selected:
            self.selected_videos.add(analysis_id)
        else:
            self.selected_videos.discard(analysis_id)
        
        self._update_action_bar_state()

    def _batch_delete(self):
        """Batch delete selected video records."""
        if not self.selected_videos:
            return
        
        count = len(self.selected_videos)
        dlg = ConfirmDialog(
            self,
            "确认批量移除工单",
            f"移除选中的 {count} 个分析工单？\n\n"
            "内容与分析结果会保留在库中，磁盘文件不受影响；\n"
            "同内容再次添加时直接复用既有结果，不会重新分析。",
            confirm_text="移除",
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if self.api:
            try:
                # 把 analysis_id 映射成 task_id 列表
                id_to_task = {v.get("id"): v.get("task_id") for v in self._all_videos}
                task_ids = []
                for aid in self.selected_videos:
                    tid = id_to_task.get(aid)
                    if tid is not None:
                        task_ids.append(tid)
                # 无 task_id 映射的退回旧端点
                orphan_ids = [aid for aid in self.selected_videos if id_to_task.get(aid) is None]

                deleted = 0
                failed = 0
                msgs = []
                if task_ids:
                    resp = self.api.post("/videos/delete", json={
                        "task_ids": task_ids,
                    })
                    deleted = resp.get("deleted_count", 0)
                    failed = resp.get("failed_count", 0)
                    if resp.get("file_errors"):
                        msgs.extend(resp["file_errors"])
                if orphan_ids:
                    resp2 = self.api.post("/videos/batch-delete", json={"ids": orphan_ids})
                    deleted += resp2.get("deleted_count", 0)
                    failed += resp2.get("failed_count", 0)

                msg = f"成功删除 {deleted} 条记录"
                if failed > 0:
                    msg += f"，{failed} 条失败"
                if msgs:
                    msg += "\n" + "\n".join(msgs[:5])

                QMessageBox.information(self, "批量删除完成", msg)
                
                # Clear selection and refresh
                self._reset_selection()
                self._load_data()
            except Exception as e:
                QMessageBox.warning(self, "删除失败", str(e))

    def _rebuild_faces(self):
        """Rebuild face recognition for selected videos."""
        if not self.selected_videos:
            return

        count = len(self.selected_videos)
        reply = ConfirmDialog.question(
            self,
            "确认重新识别人脸",
            f"将为选中的 {count} 个视频重新识别人脸。\n\n"
            "这会重新提取人脸数据并更新人物关联。",
        )
        if reply != QMessageBox.Yes:
            return

        if not self.api:
            QMessageBox.warning(self, "错误", "API 客户端未初始化")
            return

        # 重建人脸需要 Task.id（/tasks/{task_id}/rebuild-faces）。
        # selected_videos 存的是 FileAnalysis.id，从缓存里解析 task_id —— 列表视图
        # 没有宫格卡片，走 _get_all_cards() 会一个都取不到、静默不执行。
        id_to_task = {v.get("id"): v.get("task_id") for v in self._all_videos}
        selected_task_ids = []
        missing_task = []
        for fid in self.selected_videos:
            tid = id_to_task.get(fid)
            if tid:
                selected_task_ids.append(tid)
            else:
                missing_task.append(fid)

        if missing_task:
            QMessageBox.warning(
                self, "无法重建",
                f"{len(missing_task)} 个视频没有关联的任务（可能已被删除），无法重建人脸。\n"
                f"ID: {missing_task[:5]}{'...' if len(missing_task) > 5 else ''}",
            )

        if not selected_task_ids:
            return

        # Disable button and show progress; init concurrent queue state
        self.btn_rebuild_faces.setEnabled(False)
        self.btn_rebuild_faces.setText("识别中...")
        self._rebuild_task_ids = list(selected_task_ids)
        self._rebuild_pending = deque(selected_task_ids)
        self._rebuild_active = set()
        self._rebuild_failed = []
        self._rebuild_done = 0
        try:
            from src.core.config import ConfigManager as _CfgMgrUI
            _mc = int(_CfgMgrUI().get("face_recognition.max_concurrent", 0))
        except Exception:
            _mc = 0
        if _mc and _mc > 0:
            self._rebuild_concurrency = min(_mc, 8)
        else:
            try:
                import multiprocessing as _mp
                self._rebuild_concurrency = min(max(2, _mp.cpu_count() // 2), 8)
            except Exception:
                self._rebuild_concurrency = 4
        logger.info("批量重建人脸: %d 个任务, 并发 %d", len(selected_task_ids), self._rebuild_concurrency)
        self._rebuild_pump()

    def _rebuild_pump(self):
        """并发调度：active 未满则从 pending 补位发起任务。"""
        if not self.api:
            return
        while len(self._rebuild_active) < self._rebuild_concurrency and self._rebuild_pending:
            task_id = self._rebuild_pending.popleft()
            self._rebuild_active.add(task_id)
            self._rebuild_start_one(task_id)

        total = len(self._rebuild_task_ids)
        done = self._rebuild_done
        running = len(self._rebuild_active)
        self.btn_rebuild_faces.setText(f"识别中 {done + running}/{total} (进行{running})")

        if not self._rebuild_pending and not self._rebuild_active:
            self.btn_rebuild_faces.setEnabled(True)
            self.btn_rebuild_faces.setText("重新识别人脸")
            if self._rebuild_failed:
                QMessageBox.warning(
                    self, "完成(部分失败)",
                    f"已完成 {self._rebuild_done}/{total} 个视频的人脸重建，"
                    f"{len(self._rebuild_failed)} 个失败: {self._rebuild_failed[:5]}"
                )
            else:
                QMessageBox.information(
                    self, "完成",
                    f"已完成 {total} 个视频的人脸重建"
                )
            self.selected_videos.clear()
            self._update_action_bar_state()
            self._load_data()

    def _rebuild_start_one(self, task_id: int):
        """发起单个任务的人脸重建，启动轮询链。"""
        if not self.api:
            self._rebuild_finish_one(task_id, "failed", "api 不可用")
            return
        try:
            resp = self.api.post(f"/tasks/{task_id}/rebuild-faces")
            st = resp.get("status", "unknown")
            if st in ("completed", "failed"):
                self._rebuild_finish_one(task_id, st, resp.get("message", ""))
            else:
                QTimer.singleShot(1000, lambda: self._rebuild_poll_one(task_id))
        except Exception as e:
            self._rebuild_finish_one(task_id, "failed", str(e))

    def _rebuild_poll_one(self, task_id: int):
        """Poll rebuild status for a single task (per-task 1s poll chain)."""
        if not self.api or task_id not in self._rebuild_active:
            return
        try:
            resp = self.api.get(f"/tasks/{task_id}/rebuild-faces/status")
            status = resp.get("status", "unknown")
            if status == "running":
                total = resp.get("total", 0)
                processed = resp.get("processed", 0)
                done = self._rebuild_done
                active_n = len(self._rebuild_active)
                batch_total = len(self._rebuild_task_ids)
                self.btn_rebuild_faces.setText(
                    f"识别中 {done + active_n}/{batch_total} "
                    f"(任务{task_id}: {processed}/{total})"
                )
                QTimer.singleShot(1000, lambda: self._rebuild_poll_one(task_id))
            elif status == "completed":
                self._rebuild_finish_one(task_id, "completed", "")
            elif status == "failed":
                msg = resp.get("message", "重建失败")
                self._rebuild_finish_one(task_id, "failed", msg)
            else:
                self._rebuild_finish_one(task_id, "failed", f"异常状态: {status}")
        except Exception as e:
            self._rebuild_finish_one(task_id, "failed", str(e))

    def _rebuild_finish_one(self, task_id: int, status: str, msg: str = ""):
        """单个任务结束：记录结果并从 active 移除，随后补位下一个。"""
        if task_id not in self._rebuild_active:
            return
        self._rebuild_active.discard(task_id)
        self._rebuild_done += 1
        if status != "completed":
            self._rebuild_failed.append(task_id)
            logger.warning("任务 %d 人脸重建失败: %s", task_id, msg or status)
        self._rebuild_pump()

    def resizeEvent(self, event):
        """窗口大小改变时触发，启动防抖定时器"""
        super().resizeEvent(event)
        # 只标记需要重新计算，不直接渲染
        self._resize_timer.start(200)  # 200ms 防抖

    def _on_resize_timeout(self):
        """防抖定时器触发后执行"""
        # 列表视图下保持用户选的每页条数，不按窗口重算
        if getattr(self, "_view_mode", "grid") == "list":
            return
        # 计算新的列数和行数
        new_columns = self._calculate_columns()
        new_rows = self._calculate_rows()
        new_page_size = new_columns * new_rows

        # 检查是否真的变化
        if (new_columns == self._current_columns and
            new_rows == self._current_rows):
            return  # 无变化，不重新渲染

        # 更新布局参数
        self._current_columns = new_columns
        self._current_rows = new_rows
        self._page_size = new_page_size
        self.pagination.update_pagination(self._current_page,
                                          self._last_total
                                          if hasattr(self, "_last_total")
                                          else 0, self._page_size)
        self._page_size_calculated = True


        # 重新渲染当前页
        if self._data_loaded and self._all_videos:
            total = len(self._all_videos)
            max_page = max(1, (total + self._page_size - 1) // self._page_size)
            if self._current_page > max_page:
                self._current_page = max_page

            self._render_current_page()

    def _relayout_grids(self):
        """重新布局所有网格（仅在数据已加载时执行）。

        resize 时调用此方法，不重新渲染页面，避免无限循环。
        """
        if not self._data_loaded:
            return
        self._relayout_single_grid()
        self._relayout_no_actor_grid()

    def _relayout_single_grid(self):
        """Relayout single works grid based on available width."""
        if not hasattr(self, '_single_works_cards') or not hasattr(self, 'grid_layout'):
            return
        
        available_width = self.grid_container.width()
        if available_width <= 0:
            return
        
        spacing = 12
        card_min_width = 200
        cols = max(1, (available_width + spacing) // (card_min_width + spacing))
        
        # Clear grid
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        # Re-add cards with new column count
        for idx, card in enumerate(self._single_works_cards):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col, Qt.AlignTop)
        
        # Update cover sizes
        self._update_card_cover_sizes(cols)

    def _relayout_no_actor_grid(self):
        """Relayout no-actor grid based on available width."""
        if not hasattr(self, '_no_actor_cards') or not hasattr(self, 'grid_layout_no_actor'):
            return
        
        available_width = self.grid_container_no_actor.width()
        if available_width <= 0:
            return
        
        spacing = 12
        card_min_width = 200
        cols = max(1, (available_width + spacing) // (card_min_width + spacing))
        
        # Clear grid
        for i in reversed(range(self.grid_layout_no_actor.count())):
            item = self.grid_layout_no_actor.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        # Re-add cards with new column count
        for idx, card in enumerate(self._no_actor_cards):
            row = idx // cols
            col = idx % cols
            self.grid_layout_no_actor.addWidget(card, row, col, Qt.AlignTop)
        
        # Update cover sizes
        self._update_card_cover_sizes(cols)

    def _update_card_cover_sizes(self, cols):
        """Update cover image sizes based on column count."""
        # Calculate card width based on columns
        if hasattr(self, 'grid_container'):
            container_width = self.grid_container.width()
            spacing = 12
            card_width = (container_width - (cols - 1) * spacing) // cols
            cover_height = int(card_width * 0.64)  # Maintain aspect ratio
            
            # Update all cards
            all_cards = []
            if hasattr(self, '_single_works_cards'):
                all_cards.extend(self._single_works_cards)
            if hasattr(self, '_no_actor_cards'):
                all_cards.extend(self._no_actor_cards)
            
            for card in all_cards:
                card.setFixedWidth(card_width)
                card.cover_container.setFixedHeight(cover_height)
                # Update cover image
                if card.cover_label.pixmap():
                    pixmap = card.cover_label.pixmap()
                    card.cover_label.setPixmap(
                        pixmap.scaled(card_width, cover_height, 
                                     Qt.KeepAspectRatioByExpanding, 
                                     Qt.SmoothTransformation)
                    )
