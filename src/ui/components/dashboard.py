"""DashboardPage — Task Dashboard widget for ai-video-factory."""

import json
import os
import re
import subprocess
import threading
import logging
import concurrent.futures
from collections import deque
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Emoji sanitizer ──────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\u2600-\u27BF"
    "\u2300-\u23FF"
    "\uFE00-\uFE0F"
    "\U00002702-\U000027B0"
    "\U0001F000-\U0001F02F"
    "\U0001F0A0-\U0001F0FF"
    "\U0001F100-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0000200D"
    "\U0000FE0F"
    "\u200C-\u200D"
    "\U0001F1E6-\U0001F1FF"
    "\U0001F201-\U0001F2FF"
    "]",
)

def sanitize_for_qt(text: str) -> str:
    """Remove all emoji/unicode symbols safe for Qt rendering."""
    if not isinstance(text, str):
        text = str(text)
    return _EMOJI_RE.sub("", text)


def _extract_import_error(e: Exception) -> str:
    """从导入异常提取用户可读信息（额度超限等后端 message 会藏在 response body 里）。"""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            detail = resp.json().get("detail")
            if isinstance(detail, dict) and detail.get("message"):
                return str(detail["message"])
            if detail:
                return str(detail)
        except Exception:
            pass
    return str(e)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTableWidget, QTableWidgetItem, QMessageBox,
    QHeaderView, QFrame, QFileDialog, QCheckBox,
    QSizePolicy, QComboBox, QDialog, QDialogButtonBox,
    QLineEdit, QListWidget, QListWidgetItem, QStyledItemDelegate,
    QTextEdit, QStackedWidget,
)
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QRect, QEvent, QPointF
from PySide6.QtGui import (QColor, QPainter, QStandardItemModel, QStandardItem,
                           QPolygonF, QPen)
from PySide6.QtWidgets import QStyle

from .api_client import APIClient

try:
    from ..widgets.tree_row_delegate import TreeCountDelegate
    from ..widgets.video_grid_widget import PaginationWidget
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.widgets.tree_row_delegate import TreeCountDelegate
    from ui.widgets.video_grid_widget import PaginationWidget
    from ui.tokens import T


class CenteredCheckboxDelegate(QStyledItemDelegate):
    """Center the checkbox in column 0 cells."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # 让 Qt 知道这是一个可勾选的项
        option.state |= QStyle.State_Enabled

    def paint(self, painter, option, index):
        painter.save()
        # 绘制选中高亮背景
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # macOS 27 新样式对 CE_CheckBox 的指示器自行放大/偏移/丢状态
        # （Checked 可画成未选样），与表头同理改三态自绘，不经样式。
        # 注意：此 PySide6 的 data(CheckStateRole) 返回裸 int，与枚举直接
        # 比较恒 False，必须先经 Qt.CheckState() 归一。
        raw = index.data(Qt.ItemDataRole.CheckStateRole)
        check_state = Qt.CheckState(raw) if raw is not None else Qt.CheckState.Unchecked
        t = theme()
        size = 15
        box = QRect(option.rect.x() + (option.rect.width() - size) // 2,
                    option.rect.y() + (option.rect.height() - size) // 2,
                    size, size)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if check_state == Qt.CheckState.Checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(t.get('accent')))
        elif check_state == Qt.CheckState.PartiallyChecked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(t.get('accent')))
        else:
            painter.setPen(QColor(t.get('text_secondary')))
            painter.setBrush(QColor(t.get('bg_card')))
        painter.drawRoundedRect(box, 4, 4)
        if check_state == Qt.CheckState.Checked:
            painter.setPen(QPen(QColor('#FFFFFF'), 1.8))
            painter.drawPolyline(QPolygonF([
                QPointF(box.x() + 3.5, box.y() + 7.8),
                QPointF(box.x() + 6.3, box.y() + 10.6),
                QPointF(box.x() + 11.4, box.y() + 4.4),
            ]))
        elif check_state == Qt.CheckState.PartiallyChecked:
            painter.setPen(QPen(QColor('#FFFFFF'), 1.8))
            painter.drawPolyline(QPolygonF([
                QPointF(box.x() + 3.8, box.y() + 7.5),
                QPointF(box.x() + 11.2, box.y() + 7.5),
            ]))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        from PySide6.QtCore import QEvent as QE
        if event.type() in (QE.Type.MouseButtonRelease, QE.Type.MouseButtonDblClick):
            # 扩大点击区域到整个单元格（更容易点击）
            if option.rect.contains(event.pos()):
                flags = model.flags(index)
                if flags & Qt.ItemFlag.ItemIsUserCheckable:
                    # data() 可能返回裸 int 或枚举，统一归一后比较
                    raw = index.data(Qt.ItemDataRole.CheckStateRole)
                    state = Qt.CheckState(raw) if raw is not None else Qt.CheckState.Unchecked
                    new_state = Qt.CheckState.Unchecked if state == Qt.CheckState.Checked else Qt.CheckState.Checked
                    model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
                    return True
        return super().editorEvent(event, model, option, index)

from .theme import theme
from .display_names import actor_code_display
# ── Status mapping ──────────────────────────────────────────────────────
# 面向用户的收敛状态(2026-09 用户确认):
#   - 待处理 = QUEUED(已勾选并点击「分析」, 进入调度队列等待处理)
#   - 未启动/已暂停 = PENDING(新导入未激活) + PAUSED(历史手动暂停); 筛选合并一档, 不计入「待处理」
#   - 处理中 = CHECKING/SIFTING/ANALYZING/CLUSTERING/PROCESSING/WRITING
#     (后台细分保留, 前端统一"处理中")
#   - 已完成 = COMPLETE; 错误 = ERROR(+FAILED); 重复 = DUPLICATE
# 死状态(SUSPENDED/RESUMABLE/SIFTED/CONFIRMED-tasks)不再展示。
STATUS_MAP = {
    "PENDING":     ("未启动", "text_secondary", ""),
    "QUEUED":      ("待处理", "accent", ""),
    "CHECKING":    ("处理中", "accent", ""),
    "SIFTING":     ("处理中", "accent", ""),
    "SIFTED":      ("处理中", "accent", ""),  # 历史数据兜底(worker 不再写)
    "ANALYZING":   ("处理中", "accent", ""),
    "CLUSTERING":  ("处理中", "accent", ""),
    "PROCESSING":  ("处理中", "accent", ""),
    "WRITING":     ("处理中", "accent", ""),
    "PAUSED":      ("已暂停", "text_secondary", ""),
    "COMPLETE":    ("已完成", "success", ""),
    "ERROR":       ("错误", "danger", "⚠"),
    "FAILED":      ("错误", "danger", "⚠"),
    "DUPLICATE":   ("重复文件", "text_secondary", "⊘"),
}

# 媒体扩展名已统一至 src/core/media_types.py（影楼基础包 S2）
try:
    from src.core.media_types import (  # noqa: E402
        ALL_MEDIA_EXTENSIONS,
        PHOTO_EXTENSIONS,
        VIDEO_EXTENSIONS,
    )
except ImportError:  # 脚本直跑时 src 已在 sys.path
    from core.media_types import (  # noqa: E402,F401
        ALL_MEDIA_EXTENSIONS,
        PHOTO_EXTENSIONS,
        VIDEO_EXTENSIONS,
    )

# 抽帧目录（<视频名>_frames/）判定：与后端 cache_manager.is_frames_dir 同一规则（方案 B）。
# 遍历目录时必须剪枝，否则每一帧（.jpg）都会被当成照片导入（2026-09 用户报告）。
try:  # 打包/常规启动：项目根在 path
    from src.core.cache_manager import is_frames_dir as _is_frames_dir
except ImportError:  # 开发态直跑（src 在 path）
    from core.cache_manager import is_frames_dir as _is_frames_dir

# 状态筛选下拉(可多选)。value 与后端 /task?status= 的取值一致;
# 多选时以逗号连接传给后端(OR 语义)。不设「全部」项 —— 默认全部勾选即不过滤。
STATUS_FILTER_ITEMS = [
    ("pending", "待处理"),
    ("unqueued", "未排队"),
    ("not_started", "未启动/已暂停"),
    ("processing", "处理中"),
    ("complete", "已完成"),
    ("error", "错误"),
    ("duplicate", "重复"),
]

# 统计条 → 快捷筛选键（点卡片即按该状态筛选；「总任务」= 不过滤）
_QUICK_FILTER_ALL_KEYS = [v for v, _ in STATUS_FILTER_ITEMS]


def quick_filter_values(key: str) -> list:
    """统计条快捷筛选键 → 筛选下拉应勾选的 value 列表。

    「总任务」/未知键 → 全部勾选(= 不过滤)。
    """
    if key in ("", "all") or key not in _QUICK_FILTER_ALL_KEYS:
        return list(_QUICK_FILTER_ALL_KEYS)
    return [key]


def normalize_status_filter(values) -> list:
    """归一化筛选值: 空、或包含全部状态 → [](上层按"不过滤"处理)。

    这样"点总任务(全勾)"与"下拉手动全勾"得到完全一致的结果。
    """
    vals = [v for v in (values or []) if v in _QUICK_FILTER_ALL_KEYS]
    if not vals or set(vals) == set(_QUICK_FILTER_ALL_KEYS):
        return []
    return vals


def active_quick_filter_key(current_filter: str) -> str:
    """当前筛选下哪个统计卡片该高亮；多选或无匹配 → ""(都不亮)。"""
    cur = current_filter or "all"
    return cur if "," not in cur else ""

# Maximum display length for filename in table
MAX_FILENAME_DISPLAY = 200


def truncate_filename(name: str, max_len: int = MAX_FILENAME_DISPLAY) -> str:
    """Truncate filename for display, keeping extension visible."""
    if len(name) <= max_len:
        return name
    # Try to keep extension visible
    dot_idx = name.rfind('.')
    if dot_idx > 0:
        ext = name[dot_idx:]
        base = name[:dot_idx]
        # Reserve space for ext + "..."
        available = max_len - len(ext) - 3
        if available > 10:
            return base[:available] + "..." + ext
    # Fallback: simple truncation
    return name[:max_len - 3] + "..."


def shorten_folder_path(folder_path: str, max_len: int = 40) -> str:
    """Shorten folder path for display by collapsing middle segments."""
    if not folder_path:
        return "-"
    if len(folder_path) <= max_len:
        return folder_path
    # Try to show ~/.../last_two_segments
    parts = folder_path.replace('\\', '/').split('/')
    if len(parts) > 3:
        # Show first + last 2 segments
        short = parts[0] + '/.../' + '/'.join(parts[-2:])
        if len(short) <= max_len:
            return short
    # Fallback: show last portion
    return '...' + folder_path[-(max_len - 3):]


def format_file_size(size_bytes: float | int | None) -> str:
    """格式化文件大小"""
    if size_bytes is None or size_bytes == 0:
        return "-"
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def format_duration(seconds: float | int | None) -> str:
    """格式化视频时长"""
    if seconds is None or seconds == 0:
        return "-"
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_file_size_short(size_bytes: float | int | None) -> str:
    """Short file size format for compact display."""
    if size_bytes is None or size_bytes == 0:
        return "-"
    size = float(size_bytes)
    if size < 1024:
        return f"{size:.0f}B"
    size /= 1024
    if size < 1024:
        return f"{size:.1f}KB"
    size /= 1024
    if size < 1024:
        return f"{size:.1f}MB"
    size /= 1024
    return f"{size:.2f}GB"


class ScanImportDialog(QDialog):
    """细粒度导入对话框：扫描目录 → 显示文件列表 → 多选 → 确认导入。

    Workflow:
    1. 用户选择目录后自动扫描（或手动点击扫描）
    2. 显示所有找到的视频/照片文件（文件名、大小、时长、路径）
    3. 支持全选/取消全选/反选
    4. 支持按文件名搜索过滤
    5. 点击"导入选中"执行导入
    """

    scan_finished = Signal(list)  # list of file paths found
    import_finished = Signal(int, int)  # (imported, duplicate)
    import_error = Signal(str)

    def __init__(self, api: APIClient, folder_path: str = "", parent=None):
        super().__init__(parent)
        self.api = api
        self.folder_path = folder_path
        self._all_files: list[dict] = []  # [{path, name, size, exists, in_db}]
        self._set_up_ui()
        if folder_path:
            self._scan_directory(folder_path)

    def _set_up_ui(self):
        t = theme()
        self.setWindowTitle("扫描并导入媒体")
        self.setMinimumSize(750, 500)
        self.resize(800, 550)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Top bar: folder path + scan button ──
        top_bar = QHBoxLayout()
        self.folder_label = QLabel("目录: " + (sanitize_for_qt(self.folder_path) if self.folder_path else "未选择"))
        self.folder_label.setStyleSheet(f"color: {t.get('text_primary')}; font-size: {t.get('font_size_md')}px;")
        top_bar.addWidget(self.folder_label)
        top_bar.addStretch()

        self.btn_select_folder = QPushButton("更换目录")
        self.btn_select_folder.setFixedWidth(80)
        self.btn_select_folder.clicked.connect(self._select_folder)
        top_bar.addWidget(self.btn_select_folder)

        self.btn_scan = QPushButton("扫描")
        self.btn_scan.setFixedWidth(60)
        self.btn_scan.clicked.connect(self._scan_current_folder)
        top_bar.addWidget(self.btn_scan)
        layout.addLayout(top_bar)

        # ── Search/filter bar ──
        filter_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索文件名...")
        self.search_input.textChanged.connect(self._filter_files)
        filter_bar.addWidget(self.search_input)

        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.setFixedWidth(50)
        self.btn_select_all.clicked.connect(self._select_all)
        filter_bar.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("取消全选")
        self.btn_deselect_all.setFixedWidth(70)
        self.btn_deselect_all.clicked.connect(self._deselect_all)
        filter_bar.addWidget(self.btn_deselect_all)

        self.btn_invert = QPushButton("反选")
        self.btn_invert.setFixedWidth(50)
        self.btn_invert.clicked.connect(self._invert_selection)
        filter_bar.addWidget(self.btn_invert)

        self.count_label = QLabel("0 / 0 已选择")
        self.count_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")
        self.count_label.setFixedWidth(100)
        self.count_label.setAlignment(Qt.AlignRight)
        filter_bar.addWidget(self.count_label)
        layout.addLayout(filter_bar)

        # ── File table ──
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(5)
        self.file_table.setHorizontalHeaderLabels(["选择", "文件名", "大小", "状态", "路径"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.file_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.file_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.file_table.setColumnWidth(0, 50)
        self.file_table.setColumnWidth(2, 80)
        self.file_table.setColumnWidth(3, 70)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.file_table)

        # ── Bottom bar: stats + action buttons ──
        bottom_bar = QHBoxLayout()
        self.stats_label = QLabel("共 0 个文件")
        self.stats_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")
        bottom_bar.addWidget(self.stats_label)
        bottom_bar.addStretch()

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedWidth(60)
        self.btn_cancel.clicked.connect(self.reject)
        bottom_bar.addWidget(self.btn_cancel)

        self.btn_import = QPushButton("导入选中文件")
        self.btn_import.setFixedWidth(110)
        self.btn_import.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px 16px;
                font-size: {t.get('font_size_md')}px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background: {t.get('bg_hover')}; }}
            QPushButton:disabled {{ background: {t.get('border')}; color: {t.get('text_secondary')}; }}
        """)
        self.btn_import.clicked.connect(self._import_selected)
        self.btn_import.setEnabled(False)
        bottom_bar.addWidget(self.btn_import)
        layout.addLayout(bottom_bar)

    # ── Scan logic ──

    def _select_folder(self):
        """Pick a folder using native macOS dialog for full HDD support."""
        # Use Native dialog to ensure sidebar and external drives are visible
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self._scan_directory(folder)
            self.folder_label.setText("目录: " + sanitize_for_qt(folder))

    def _scan_current_folder(self):
        if self.folder_path:
            self._scan_directory(self.folder_path)

    def _scan_directory(self, folder: str):
        """Scan directory for video/photo files (runs in thread to avoid UI freeze)."""
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("扫描中...")
        self.file_table.setRowCount(0)
        self._all_files = []

        def do_scan():
            found = []
            for root, dirs, files in os.walk(folder):
                # 剪枝：跳过抽帧目录，否则每一帧（.jpg）都会出现在扫描列表里
                dirs[:] = [d for d in dirs if not _is_frames_dir(os.path.join(root, d))]
                for f in files:
                    fp = os.path.join(root, f)
                    if Path(f).suffix.lower() in ALL_MEDIA_EXTENSIONS:
                        try:
                            size = os.path.getsize(fp)
                            exists = True
                        except OSError:
                            size = 0
                            exists = False
                        found.append({
                            "path": fp,
                            "name": f,
                            "size": size,
                            "exists": exists,
                        })
            self.scan_finished.emit(found)

        threading.Thread(target=do_scan, daemon=True).start()

    def _on_scan_finished(self, files: list[dict]):
        """Called on main thread after scan completes."""
        self._all_files = files
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("扫描")
        self._render_table(files)
        self.stats_label.setText(f"共 {len(files)} 个文件")
        self._update_count()

    def _render_table(self, files: list[dict]):
        """Render file list into the table."""
        self.file_table.setRowCount(len(files))
        for i, f in enumerate(files):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            cb.setStyleSheet("margin-left: 12px;")
            cb.stateChanged.connect(self._update_count)
            self.file_table.setCellWidget(i, 0, cb)

            # Filename
            name_item = QTableWidgetItem(sanitize_for_qt(f["name"]))
            name_item.setToolTip(f["name"])
            self.file_table.setItem(i, 1, name_item)

            # Size
            size_item = QTableWidgetItem(format_file_size_short(f["size"]))
            size_item.setTextAlignment(Qt.AlignCenter)
            self.file_table.setItem(i, 2, size_item)

            # Status
            status = "可导入" if f["exists"] else "不存在"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            if not f["exists"]:
                status_item.setForeground(QColor("red"))
            self.file_table.setItem(i, 3, status_item)

            # Path (shortened)
            path_item = QTableWidgetItem(shorten_folder_path(f["path"], 50))
            path_item.setToolTip(f["path"])
            self.file_table.setItem(i, 4, path_item)

        self.btn_import.setEnabled(len(files) > 0)

    # ── Filter / selection ──

    def _filter_files(self, text: str):
        """Show/hide rows based on search text."""
        text = text.lower()
        for i in range(self.file_table.rowCount()):
            name = self.file_table.item(i, 1).text().lower() if self.file_table.item(i, 1) else ""
            self.file_table.setRowHidden(i, text not in name if text else False)
        self._update_count()

    def _select_all(self):
        for i in range(self.file_table.rowCount()):
            if not self.file_table.isRowHidden(i):
                cb = self.file_table.cellWidget(i, 0)
                if cb:
                    cb.setChecked(True)
        self._update_count()

    def _deselect_all(self):
        for i in range(self.file_table.rowCount()):
            if not self.file_table.isRowHidden(i):
                cb = self.file_table.cellWidget(i, 0)
                if cb:
                    cb.setChecked(False)
        self._update_count()

    def _invert_selection(self):
        for i in range(self.file_table.rowCount()):
            if not self.file_table.isRowHidden(i):
                cb = self.file_table.cellWidget(i, 0)
                if cb:
                    cb.setChecked(not cb.isChecked())
        self._update_count()

    def _update_count(self):
        selected = 0
        total = 0
        for i in range(self.file_table.rowCount()):
            if not self.file_table.isRowHidden(i):
                total += 1
                cb = self.file_table.cellWidget(i, 0)
                if cb and cb.isChecked():
                    selected += 1
        self.count_label.setText(f"{selected} / {total} 已选择")
        self.btn_import.setEnabled(selected > 0)

    # ── Import ──

    def _get_selected_files(self) -> list[str]:
        """Return list of selected file paths."""
        selected = []
        for i in range(self.file_table.rowCount()):
            if self.file_table.isRowHidden(i):
                continue
            cb = self.file_table.cellWidget(i, 0)
            if cb and cb.isChecked():
                selected.append(self._all_files[i]["path"])
        return selected

    def _import_selected(self):
        files = self._get_selected_files()
        if not files:
            QMessageBox.warning(self, "提示", "请至少选择一个文件。")
            return

        self.btn_import.setEnabled(False)
        self.btn_import.setText("导入中...")

        def do_import():
            try:
                resp = self.api.post("/task/import", json={"files": files}, timeout=60)
                imported = resp.get("imported", 0) if isinstance(resp, dict) else 0
                duplicate = resp.get("duplicate", 0) if isinstance(resp, dict) else 0
                self.import_finished.emit(imported, duplicate)
            except Exception as e:
                self.import_error.emit(_extract_import_error(e))

        threading.Thread(target=do_import, daemon=True).start()

    def _on_import_done(self, imported: int, duplicate: int):
        msg = f"导入完成：{imported} 个新任务"
        if duplicate > 0:
            msg += f"，{duplicate} 个重复已跳过"
        self.stats_label.setText(msg)
        self.btn_import.setEnabled(True)
        self.btn_import.setText("导入选中文件")
        # Auto-close after short delay
        QTimer.singleShot(1500, self.accept)

    def _on_import_error(self, error_msg: str):
        QMessageBox.critical(self, "错误", sanitize_for_qt(f"导入失败: {error_msg}"))
        self.btn_import.setEnabled(True)
        self.btn_import.setText("导入选中文件")
        if "免费额度已用完" in error_msg:
            self._offer_activation()

    def exec(self):
        """Override to wire up signal handlers before showing dialog."""
        self.scan_finished.connect(self._on_scan_finished)
        self.import_finished.connect(self._on_import_done)
        self.import_error.connect(self._on_import_error)
        return super().exec()


class _StatusWorker:
    """Fetch status/tasks synchronously in main thread (avoid daemon thread + QTimer.singleShot issues)."""
    
    def __init__(self, api: APIClient, action: str, callback=None, error_callback=None, **kwargs):
        """
        callback: callable invoked on main thread
        error_callback: callable(str) invoked on main thread
        """
        self.api = api
        self.action = action
        self.kwargs = kwargs
        self._callback = callback
        self._error_callback = error_callback
        # Execute synchronously in main thread
        self._run()
    
    def _run(self):
        # 轮询在主线程同步执行（见 _poll_status 注释）：必须带短超时，
        # 否则 API 繁忙时 30s 默认超时 ×3 重试 = 主线程转圈（2026-09-23
        # 用户实测首点卡顿）。此处的 3s 上限只影响轮询，不影响导入等长操作。
        import time as _t
        _t0 = _t.perf_counter()
        try:
            if self.action == "status":
                data = self.api.get("/status", timeout=3)
                if self._callback:
                    self._callback(data)
            elif self.action == "tasks":
                params = self.kwargs.get("params", {})
                data = self.api.get("/task", params=params, timeout=3)
                tasks = data.get("tasks", data if isinstance(data, list) else [])
                if self._callback:
                    self._callback(tasks)
            elif self.action == "status_and_tasks":
                status = self.api.get("/status", timeout=3)
                # Fetch paginated tasks
                params = self.kwargs.get("params", {})
                task_resp = self.api.get("/task", params=params, timeout=3)
                
                # Handle both paginated and non-paginated responses
                if isinstance(task_resp, dict) and "tasks" in task_resp:
                    task_list = task_resp["tasks"]
                    pagination_info = {
                        "total": task_resp.get("total", len(task_list)),
                        "page": task_resp.get("page", 1),
                        "page_size": task_resp.get("page_size", len(task_list)),
                        "total_pages": task_resp.get("total_pages", 1),
                    }
                else:
                    task_list = task_resp if isinstance(task_resp, list) else []
                    pagination_info = {
                        "total": len(task_list),
                        "page": 1,
                        "page_size": len(task_list),
                        "total_pages": 1,
                    }
                
                if self._callback:
                    self._callback(status, task_list, pagination_info)
        except Exception as e:
            if self._error_callback:
                self._error_callback(str(e))
        finally:
            # TEMP-PERF：轮询耗时打点（定位首点转圈后移除）
            print(f"[PERF] 轮询 {self.action}: {(_t.perf_counter() - _t0) * 1000:.0f}ms")


class StatCard(QFrame):
    """统计小卡片 —— 同时作为**快捷筛选**入口：点击即按该状态筛选任务列表。

    与顶部筛选下拉是同一套条件（点「待处理」= 下拉里只勾「待处理」），
    所以统计条的计数与列表内容始终对得上。
    """

    clicked = Signal(str)  # 发出筛选键: all / pending / processing / complete

    def __init__(self, title: str, initial_value: int = 0, filter_key: str = "", parent=None):
        super().__init__(parent)
        self._filter_key = filter_key
        self._active = False
        self.setFixedHeight(32)  # 统计卡一档制（根参数钉板 2026-09-28）
        # 自定义 QWidget 子类需启用，否则 QSS 的 background/border 不绘制
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        # 标签对鼠标透明：点击整个卡片命中 StatCard 本体（顶部拖动条的
        # widgetAt 探测也据此返回卡片而非标签，转发点击才成立）
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label)

        self.value_label = QLabel(str(initial_value))
        self.value_label.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: {theme().get('text_primary')};")
        self.value_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.value_label)

        if filter_key:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip(f"点击筛选：只显示「{title}」的任务")
        self._apply_style()

    def _apply_style(self):
        """按当前选中态重放样式（padding 3 + border 1 = 原 4，尺寸不跳动）。"""
        t = theme()
        if self._active:
            bg = "rgba(10,132,255,0.16)"
            border = "rgba(10,132,255,0.45)"
        else:
            bg = t.get('bg_card')
            border = "transparent"
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 3px 6px;
            }}
            StatCard:hover {{
                border: 1px solid {t.get('accent')};
            }}
        """)
        self.title_label.setStyleSheet(f"""
            font-size: 13px;
            color: {t.get('text_secondary')};
            background: transparent;
        """)
        self.value_label.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 600;
            color: {t.get('text_primary')};
            background: transparent;
        """)
        # 双保险：显式透明底 + 非 不透明 绘制标志（选中态 tint 底上曾现暗盒）
        self.value_label.setAutoFillBackground(False)

    def set_active(self, active: bool):
        """高亮当前生效的快捷筛选。"""
        if self._active == active:
            return
        self._active = active
        self._apply_style()

    def set_value(self, value: int):
        self.value_label.setText(str(value))
        # 空值灰显（18§1.5）：0 值降为次三级色，非 0 回主色
        self.value_label.setStyleSheet(
            f"font-size: 15px; font-weight: 600; background: transparent; "
            f"color: {theme().get('text_tertiary') if value == 0 else theme().get('text_primary')};")

    def mousePressEvent(self, event):
        """左键点击 → 发出快捷筛选请求。"""
        if self._filter_key and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._filter_key)
        super().mousePressEvent(event)

    def apply_theme(self):
        """换肤：重放本卡片样式（背景/标题/数值色），不重建控件。"""
        self._apply_style()


class _HeaderCheckboxView(QHeaderView):
    """Custom QHeaderView that renders a real clickable checkbox in the first section."""

    clicked_in_header = Signal()

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._check_state = Qt.CheckState.Unchecked
        self.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.setSectionsClickable(True)

    def set_check_state(self, state: int):
        self._check_state = state
        if self.count() > 0:
            self.updateSection(0)

    def paintSection(self, painter, rect, logicalIndex):
        if logicalIndex == 0:
            # macOS 27 新样式对 CE_CheckBox 指示器自行放大/偏移，小矩形内
            # 经 QStyle 绘制必然错位 → 三态框直接自绘，不经样式。
            # 背景/边框同步自绘（绕过默认 paintSection 后 QSS 不再代画）。
            painter.save()
            t = theme()
            painter.fillRect(rect, QColor(t.get('table_header_bg')))
            painter.setPen(QColor(t.get('border')))
            painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
            painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            size = 15
            box = QRect(rect.x() + (rect.width() - size) // 2,
                        rect.y() + (rect.height() - size) // 2, size, size)
            state = self._check_state
            if state == Qt.CheckState.Unchecked:
                painter.setPen(QColor(t.get('text_secondary')))
                painter.setBrush(QColor(t.get('bg_card')))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(t.get('accent')))
            painter.drawRoundedRect(box, 4, 4)
            if state == Qt.CheckState.Checked:
                painter.setPen(QPen(QColor('#FFFFFF'), 1.8))
                painter.drawPolyline(QPolygonF([
                    QPointF(box.x() + 3.5, box.y() + 7.8),
                    QPointF(box.x() + 6.3, box.y() + 10.6),
                    QPointF(box.x() + 11.4, box.y() + 4.4),
                ]))
            elif state == Qt.CheckState.PartiallyChecked:
                painter.setPen(QPen(QColor('#FFFFFF'), 1.8))
                painter.drawPolyline(QPolygonF([
                    QPointF(box.x() + 3.8, box.y() + 7.5),
                    QPointF(box.x() + 11.2, box.y() + 7.5),
                ]))
            painter.restore()
        else:
            super().paintSection(painter, rect, logicalIndex)

    def mousePressEvent(self, event):
        # 获取点击位置的 x 坐标
        x = event.position().x()
        
        # 直接检查是否在第 0 列范围内（复选框列宽度 30px）
        section_size = self.sectionSize(0)
        if 0 <= x <= section_size:
            self._toggle_check()
        else:
            super().mousePressEvent(event)

    def _toggle_check(self):
        # PartiallyChecked or Unchecked → check all; Checked → uncheck all
        if self._check_state == Qt.CheckState.Checked:
            self._check_state = Qt.CheckState.Unchecked
        else:
            self._check_state = Qt.CheckState.Checked
        self.updateSection(0)
        self.clicked_in_header.emit()


class MultiSelectComboBox(QComboBox):
    """可多选的筛选下拉: 每项带勾选框。

    交互约定(按使用反馈调整):
    - 点选项只切换勾选, **弹层保持展开**, 可连续勾选多项; 不会"没再点就消失"。
    - 关闭方式: 再点一次下拉框本身, 或点击别处。
    - 列表里不再有「全部」这一项, 只有真实状态; 默认**全部勾选**(= 不过滤)。
      全部取消时按"不过滤"处理(而非零结果空筛选)。
    """

    changed = Signal(list)  # 勾选变化 → 选中的 value 列表

    def __init__(self, items, parent=None):
        super().__init__(parent)
        model = QStandardItemModel(self)
        self.setModel(model)
        for value, label in items:
            item = QStandardItem(label)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(value, Qt.ItemDataRole.UserRole)
            # 默认全选: 不做任何筛选
            item.setCheckState(Qt.CheckState.Checked)
            model.appendRow(item)

        # 只读的行编辑区用来显示"已选摘要"(QComboBox 本身不支持自定义当前文本)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lineEdit().setStyleSheet("background: transparent; border: none; padding: 0;")
        # editable 下拉点文本区不会展开弹层(实测), 用事件过滤器补上, 并实现"再点一次收起"
        self.lineEdit().installEventFilter(self)
        # 关键: 拦掉弹层内的"松开"事件 —— Qt 默认据此关闭弹层, 拦掉即可保持展开
        self.view().viewport().installEventFilter(self)

        self.view().pressed.connect(self._on_item_pressed)
        self._updating = False
        self._refresh_text()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt 命名
        etype = event.type()

        # 弹层内松开鼠标 → 吃掉, 弹层不关闭(可连续勾选)
        if obj is self.view().viewport() and etype == QEvent.Type.MouseButtonRelease:
            return True

        # 点下拉框文本区: 展开 / 再点收起
        if obj is self.lineEdit() and etype == QEvent.Type.MouseButtonPress:
            if self.view().isVisible():
                self.hidePopup()
            else:
                self.showPopup()
            return True

        return super().eventFilter(obj, event)

    # ── 内部 ────────────────────────────────────────────────
    def _items(self):
        for i in range(self.model().rowCount()):
            yield self.model().item(i)

    def _on_item_pressed(self, index):
        """点某项 → 切换勾选(弹层不关)。"""
        if self._updating:
            return
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        self._updating = True
        try:
            checked = item.checkState() == Qt.CheckState.Checked
            item.setCheckState(
                Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked
            )
        finally:
            self._updating = False
        self._refresh_text()
        self.changed.emit(self.checked_values())

    # ── 对外 ────────────────────────────────────────────────
    def checked_values(self) -> list:
        """勾选的 value 列表(空列表表示"未选择", 上层按不过滤处理)。"""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._items()
            if item.checkState() == Qt.CheckState.Checked
        ]

    def set_checked_values(self, values) -> list:
        """程序化设置勾选项（用于统计条快捷筛选）。

        不触发 changed 信号 —— 调用方自己走筛选处理路径，避免一次点击刷两次。
        返回实际勾选的列表。
        """
        want = set(values or [])
        self._updating = True
        try:
            for item in self._items():
                v = item.data(Qt.ItemDataRole.UserRole)
                item.setCheckState(
                    Qt.CheckState.Checked if v in want else Qt.CheckState.Unchecked
                )
        finally:
            self._updating = False
        self._refresh_text()
        return self.checked_values()

    def _refresh_text(self):
        values = self.checked_values()
        if not values:
            text = "未选择（不过滤）"
        else:
            labels = [
                item.text() for item in self._items()
                if item.data(Qt.ItemDataRole.UserRole) in values
            ]
            text = " + ".join(labels)
        # 全选时文本较长 → 按当前宽度省略, 完整内容放 tooltip
        fm = self.lineEdit().fontMetrics()
        avail = max(60, self.width() - 28)
        self.lineEdit().setText(fm.elidedText(text, Qt.TextElideMode.ElideRight, avail))
        self.setToolTip(text)



class DashboardPage(QWidget):
    """Page 1: Task Dashboard — progress overview and task list."""
    
    # 类级别信号，用于从后台线程安全地回调主线程
    import_finished = Signal(int, int)  # (imported, duplicate)
    import_error = Signal(str)  # error message
    import_progress = Signal(str)  # 实时进度文案（如「导入中 12/661 ...」）
    # 轮询后台线程化（2026-09-23 治本主线程转圈）：HTTP 在 _executor 线程，
    # 经信号回到主线程改 UI
    _poll_finished = Signal(object, object, object)  # (status, tasks, pagination)
    _poll_failed = Signal(str)
    _folders_finished = Signal(object)  # /task/folders 聚合列表

    def __init__(self, api_base: str = "http://localhost:8000", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.api = APIClient(base_url=api_base)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._selected_folders: dict[str, list[str]] = {}  # folder_path -> [file_paths]
        self._current_filter = "all"
        self._current_face_filter = "all"  # 人脸状态筛选: all/complete/none/skipped/unscanned
        
        # Pagination state
        self._current_page = 1
        self._page_size = 50  # Default 50 items per page
        self._total_pages = 1
        self._total_items = 0

        # 连接导入信号
        self.import_finished.connect(self._on_import_done)
        self.import_error.connect(self._on_import_error)
        self.import_progress.connect(self._on_import_progress)
        # 轮询后台线程 → 主线程信号
        self._poll_finished.connect(self._on_poll_status_done)
        self._poll_failed.connect(self._on_poll_error)
        self._folders_finished.connect(self._on_folders_fetched)
        self._poll_inflight = False

        self._build_ui()
        self._source_folder = None
        self._sync_stat_cards()   # 初始无筛选 → 高亮「总任务」
        self._start_polling()
        QTimer.singleShot(1500, self._refresh_source_tree)  # 首批数据到达后建树

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        t = theme()
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── 左侧：任务来源树（09 方案：文件夹 ≥2 单列 + 零散文件虚拟分组）──
        from PySide6.QtWidgets import QTreeWidget
        self.source_panel = QWidget()
        # 宽度随统一侧栏填满（此前写死 210 导致四页树列不等宽）
        self.source_panel.setStyleSheet(
            f"background: {t.get('bg_secondary')}; color: {t.get('text_primary')};"
        )
        sp_v = QVBoxLayout(self.source_panel)
        sp_v.setContentsMargins(T.TREE_WRAP_X, 0, T.TREE_WRAP_X, T.TREE_WRAP_BOTTOM)
        sp_v.setSpacing(0)  # 镜片行与树行零缝（用户 2026-09-28：消镜片下空白）
        # T1 双视图：tab 式二选一（用户 2026-09-28：两等宽 tab，选中=底部
        # accent 指示线+主色字；不拥挤不割裂）。行底一条通长线贯穿两 tab。
        self._lens_is_source = True
        self._lens_src = QLabel("来源")
        self._lens_proj = QLabel("项目")
        _lens = QHBoxLayout()
        _lens.setContentsMargins(0, 0, 0, 0)
        _lens.setSpacing(0)
        for _l, _cb in ((self._lens_src, self._set_lens_source),
                        (self._lens_proj, self._set_lens_project)):
            _l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _l.setFixedHeight(30)
            _l.setCursor(Qt.CursorShape.PointingHandCursor)
            _l.mousePressEvent = (lambda ev, cb=_cb, lab=_l: cb())
            _lens.addWidget(_l, 1)
        from PySide6.QtWidgets import QFrame as _QFrame
        self._lens_row = _QFrame()  # 分段控件容器（LM 视觉）
        self._lens_row.setObjectName("LensSeg")
        # macOS 分段控件样式：容器圆角 6 底 glass，选中段=bg_card 填充
        self._lens_row.setStyleSheet(
            f"QFrame#LensSeg {{ background: {t.get('glass')};"
            f" border-radius: 6px; }}")
        self._lens_row.setLayout(_lens)
        self._lens_row.setFixedHeight(28)
        sp_v.addWidget(self._lens_row)
        # （原「任务来源」标题已删——说明性文字，树自见即可）
        self.source_tree = QTreeWidget()
        # P12②③：纯平列表（仅顶层项）——缩进归零让选中条全宽对称；
        # 行规格走 TreeCountDelegate（18§3.3-3.7 单点实现：行 28/圆角 6/
        # 选中半透+指示线/计数右对齐）
        self.source_tree.setIndentation(0)
        self.source_tree.setHeaderHidden(True)
        self.source_tree.setUniformRowHeights(True)
        # 视口顶部内缩 -6：贴住镜片行底线（用户 2026-09-28：消镜片下空白）
        # 关闭 macOS 原生聚焦环（蓝色圆角环，顶端露出即用户看到的"蓝线"）
        self.source_tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.source_tree.setStyleSheet("QTreeWidget { background: transparent; border: none; }")
        self._source_delegate = TreeCountDelegate(self.source_tree)
        self.source_tree.setItemDelegate(self._source_delegate)
        self.source_tree.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.source_tree.setMouseTracking(True)
        self.source_tree.setContentsMargins(0, -6, 0, 0)
        self.source_tree.itemClicked.connect(self._on_source_clicked)
        sp_v.addWidget(self.source_tree, 1)
        # T1：项目树（同一规格，切换显示；数据=/api/studio/projects）
        self.project_tree = QTreeWidget()
        self.project_tree.setIndentation(26)  # 项目树有真实层级（父→子项目）
        self.project_tree.setRootIsDecorated(False)
        self.project_tree.setUniformRowHeights(True)
        self.project_tree.setHeaderHidden(True)
        self.project_tree.setContentsMargins(0, -6, 0, 0)  # 同源树：贴镜片行
        self.project_tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.project_tree.setStyleSheet(self.source_tree.styleSheet())
        self._project_delegate = TreeCountDelegate(self.project_tree)
        self.project_tree.setItemDelegate(self._project_delegate)
        self.project_tree.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.project_tree.setMouseTracking(True)
        self.project_tree.itemClicked.connect(self._on_project_clicked)
        self.project_tree.setVisible(False)
        sp_v.addWidget(self.project_tree, 1)
        self._project_filter = None  # None=全部; int=project_id
        # 10 号方案切片 2：来源树面板不加入本页布局——由 MainWindow 通过
        # take_source_panel() 取走、reparent 到主窗口统一侧栏。
        # （面板保持无父级构建，取走前不可 show，否则会闪成独立窗口）

        # 右侧：原页面内容
        right = QWidget()
        right_layout = QVBoxLayout(right)
        # 底部留 14pt：让分页行脱离窗口底缘（系统会把贴底栏识别为底栏并
        # 套玻璃容器——脱离底缘后胶囊消失，实验验证）
        right_layout.setContentsMargins(0, 0, 0, 0)  # 满铺零边距（18§4.0）：贴树/贴顶带/贴窗口
        right_layout.setSpacing(0)  # 面板贴操作行（同详情页，2026-09-29）
        outer.addWidget(right, 1)
        root = right_layout

        # ── Stat cards (compact inline) ──────────────────────────────────────
        # （原「总进度 xx%」百分比进度条已按 2026-09-24 反馈移除——统计卡已表达）
        # 13 号：统计行独立成面板，由 MainWindow 经 take_title_cards() 取走，
        # 放入统一标题栏（红绿灯右侧）。
        self.title_cards = QWidget()
        cards_h = QHBoxLayout(self.title_cards)
        cards_h.setContentsMargins(0, 0, 0, 0)
        cards_h.setSpacing(8)
        # 统计卡片同时是快捷筛选入口（filter_key 对应筛选下拉里的取值）
        self.stat_total = StatCard("总任务", 0, filter_key="all")
        self.stat_completed = StatCard("已完成", 0, filter_key="complete")
        self.stat_processing = StatCard("处理中", 0, filter_key="processing")
        self.stat_pending = StatCard("待处理", 0, filter_key="pending")
        # 未排队 = 已添加但从未激活排队（PENDING）：不手动激活永远不会被处理
        self.stat_unqueued = StatCard("未排队", 0, filter_key="unqueued")  # 兜底桶：出错/重复/暂停均计入
        for card in (self.stat_total, self.stat_completed, self.stat_processing,
                     self.stat_pending, self.stat_unqueued):
            cards_h.addWidget(card)
            card.clicked.connect(self._apply_quick_filter)
        
        # 调度器状态和控制
        cards_h.addSpacing(16)
        self.scheduler_status_label = QLabel("调度器: 停止")
        cards_h.addWidget(self.scheduler_status_label)

        # ── Task list (no group box, flat) ───────────────────────────────────
        task_layout = QVBoxLayout()
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(0)  # 操作行贴内容栈（同详情页 36，2026-09-29 用户令对齐）

        # ── Action bar: import + batch operations ──────────────────────
        # 操作行固定在内容区顶部（用户裁定 2026-09-27：顶带全 app 单行 48pt、
        # 只允许一行内容，工具栏一律内容区；顶带两行方案已被推翻）
        self.action_panel = QWidget()
        action_bar = QHBoxLayout(self.action_panel)
        action_bar.setContentsMargins(T.TOOLBAR_MARGIN, T.SP_4,
                                      T.TOOLBAR_MARGIN, T.SP_4)  # 容器 36 与顶带槽等高（同详情页操作行）
        action_bar.setSpacing(8)
        # 「导入」单钮（14 号定稿）：NSOpenPanel 混选文件+文件夹；
        # 无扩展名文件自动嗅探补名走收件箱——三合一无功能损失
        self.btn_import = QPushButton("导入")
        self.btn_import.setToolTip(
            "选择视频/图片文件或文件夹（可多选、可混选）\n"
            "无扩展名文件将自动识别类型并导入")
        # 主钮（18§4.2）：全页唯一 accent 实底，白字 600
        self.btn_import.setStyleSheet(
            "QPushButton { background: #0A84FF; color: #FFFFFF;"
            " border: 1px solid #0A84FF; border-radius: 6px;"
            " padding: 0 14px; font-weight: 600; }"
            " QPushButton:hover { background: #2492FF; }")
        self.btn_import.clicked.connect(self.import_mixed)
        action_bar.addWidget(self.btn_import)
        self.batch_start_btn = QPushButton("分析")
        self.batch_start_btn.setEnabled(False)
        self.batch_start_btn.clicked.connect(self._batch_start)
        self.batch_delete_btn = QPushButton("删除")
        self.batch_delete_btn.setEnabled(False)
        self.batch_delete_btn.clicked.connect(self._batch_delete)
        action_bar.addWidget(self.batch_start_btn)
        action_bar.addWidget(self.batch_delete_btn)

        # 「归到项目」（project-assets-roster）：选中工单 → 逻辑分类进项目。
        # 写的是内容归属（清单），工单删了重加分类自动跟回；本页不提供
        # 项目结构管理（建/删/移项目在文件详情页）。
        self.btn_assign_project = QPushButton("归到项目")
        self.btn_assign_project.setEnabled(False)
        self.btn_assign_project.setToolTip(
            "把选中任务对应的内容加入项目（逻辑分类，不动磁盘文件）\n"
            "项目的新建/删除/整理在「文件详情」页侧栏")
        self.btn_assign_project.clicked.connect(self._assign_selected_to_project)
        action_bar.addWidget(self.btn_assign_project)



        # 「优先」置顶按钮（14 号定稿）：所选任务派发恒最前，跨页有效
        self.btn_pin_top = QPushButton("优先")
        self.btn_pin_top.setEnabled(False)
        self.btn_pin_top.setToolTip(
            "置顶所选任务：派发时恒排最前（对已置顶的再点则取消）")
        self.btn_pin_top.clicked.connect(self._toggle_pin_selected)
        action_bar.addWidget(self.btn_pin_top)

        # 「取消排队」：所选 QUEUED 任务退回「未启动」；对话框内可一键取消
        # 全部排队（启动时自动恢复排队的反向操作，不影响运行中任务）
        self.batch_unqueue_btn = QPushButton("取消排队")
        self.batch_unqueue_btn.setEnabled(False)
        self.batch_unqueue_btn.setToolTip(
            "把排队中的任务退回「未启动」（不会自动分析）\n"
            "不影响正在运行的任务；对话框内也可一键取消全部排队")
        self.batch_unqueue_btn.clicked.connect(self._batch_unqueue)
        action_bar.addWidget(self.batch_unqueue_btn)

        # 2026-09-29 用户裁定：与详情页操作行同构——控件组靠左连排
        # （原中段 stretch 两端分布造成两页布局割裂）
        # 筛选下拉（可多选勾选框）: 勾选多项 = OR 过滤; 点选项不会收起弹层。
        # 例如只留「待处理 + 处理中」即"整个在跑管线"一屏可见——
        # 此前两项被拆成单选, 排队任务散落在上千条已完成里, 难以查看。
        self.filter_combo = MultiSelectComboBox(STATUS_FILTER_ITEMS)
        self.filter_combo.setFixedWidth(210)
        self.filter_combo.setFixedHeight(28)  # 控件一档制 28（根参数钉板）
        # 状态筛选下拉已按定稿移出界面——统计卡即筛选；combo 保留为
        # 筛选状态持有者（统计卡点击经它走同一条过滤链路）
        self.filter_combo.setVisible(False)
        # 防抖: 连续勾选多项时只在停下来后刷新一次。
        # 之前每次勾选都同步拉取+重建表格, 与紧随其后的点击相互干扰,
        # 表现为"勾了好几下但只生效一部分"(手感=不稳定)。
        self._filter_refresh_timer = QTimer(self)
        self._filter_refresh_timer.setSingleShot(True)
        self._filter_refresh_timer.setInterval(250)
        self._filter_refresh_timer.timeout.connect(self.refresh_data)
        # 人脸状态筛选下拉(叠加在状态筛选之上)
        self.face_filter_combo = QComboBox()
        self.face_filter_combo.addItems([
            "全部人脸",
            "✓ 已识别",
            "🙅 无人脸",
            "⏭ 跳过",
            "— 未扫描",
        ])
        self.face_filter_combo.setFixedWidth(110)
        self.face_filter_combo.setFixedHeight(28)  # 控件一档制 28（根参数钉板）
        action_bar.addWidget(self.face_filter_combo)
        self.face_filter_combo.currentIndexChanged.connect(self._on_face_filter_changed)
        # 刷新按钮（右侧）
        self.refresh_api_btn = QPushButton("刷新")
        self.refresh_api_btn.clicked.connect(self._on_refresh_api)
        action_bar.addWidget(self.refresh_api_btn)
        action_bar.addStretch()  # 尾 stretch：控件组靠左（同详情页操作行）
        # 操作按钮统一 26px（顶部区域紧凑）
        for _i in range(action_bar.count()):
            _it = action_bar.itemAt(_i)
            if _it.widget() and isinstance(_it.widget(), QPushButton):
                _it.widget().setFixedHeight(28)
                _it.widget().setMinimumWidth(60)  # 两字钮统一宽（18§4.1）  # 控件一档制 28（根参数钉板）
        task_layout.addWidget(self.action_panel)

        # Task table (9 columns: checkbox | filename | folder | file_size | duration | face | status | progress | action)
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(9)
        self.task_table.setHorizontalHeaderLabels(["", "文件名", "文件夹", "文件大小", "视频时长", "人脸", "标签分析", "标签进度", "ID"])
        # 表头按列对齐（18§4.10）：文本左/数值右/状态居中
        from PySide6.QtCore import Qt as _Qt
        _halign = [_Qt.AlignmentFlag.AlignCenter, _Qt.AlignmentFlag.AlignLeft | _Qt.AlignmentFlag.AlignVCenter,
                   _Qt.AlignmentFlag.AlignLeft | _Qt.AlignmentFlag.AlignVCenter,
                   _Qt.AlignmentFlag.AlignRight | _Qt.AlignmentFlag.AlignVCenter,
                   _Qt.AlignmentFlag.AlignRight | _Qt.AlignmentFlag.AlignVCenter,
                   _Qt.AlignmentFlag.AlignCenter, _Qt.AlignmentFlag.AlignCenter,
                   _Qt.AlignmentFlag.AlignLeft | _Qt.AlignmentFlag.AlignVCenter,
                   _Qt.AlignmentFlag.AlignRight | _Qt.AlignmentFlag.AlignVCenter]
        for _i, _al in enumerate(_halign):
            _hi = self.task_table.horizontalHeaderItem(_i)
            if _hi is not None:
                _hi.setTextAlignment(_al)
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setAlternatingRowColors(False)  # 无斑马（18§4.11）
        self.task_table.setShowGrid(False)  # 无竖线
        self.task_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.task_table.setMinimumHeight(400)
        # 禁止单元格内联编辑(任务列表是只读展示; 编辑走弹窗/按钮)。
        # 历史: 默认 editTriggers 允许双击文件名进入编辑态, 误改文件名。
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.task_table.verticalHeader().setDefaultSectionSize(28)  # 表格行高一档制
        self.task_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.task_table.customContextMenuRequested.connect(self._show_task_context_menu)
        # Install custom header with checkbox in column 0
        self._header_view = _HeaderCheckboxView(self.task_table)
        # 列宽模式：文件名自动扩充，其他列固定宽度，允许用户拖拽调整
        self._header_view.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # 复选框列固定宽度
        self._header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # 文件名自动扩展
        self._header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)  # 文件夹可调整
        self._header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # 文件大小固定宽度
        self._header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # 视频时长固定宽度
        self._header_view.setMinimumSectionSize(30)
        self.task_table.setHorizontalHeader(self._header_view)

        # 列宽持久化：加载保存的列宽或使用默认值
        self._column_widths_path = Path.home() / ".hermes" / "profiles" / "creative" / "config" / "table_widths.json"
        self._default_column_widths = {0: 30, 1: 400, 2: 200, 3: 100, 4: 100, 5: 100, 6: 150, 7: 60, 8: 60}
        self._load_column_widths()

        # 列宽变化时延迟保存（防抖）
        self._column_width_save_timer = QTimer()
        self._column_width_save_timer.setSingleShot(True)
        self._column_width_save_timer.setInterval(500)  # 500ms 防抖
        self._column_width_save_timer.timeout.connect(self._save_column_widths)
        self._header_view.sectionResized.connect(self._on_column_width_changed)

        # 应用居中复选框 delegate 到第 0 列
        self._checkbox_delegate = CenteredCheckboxDelegate(self.task_table)
        self.task_table.setItemDelegateForColumn(0, self._checkbox_delegate)
        
        self._header_view.clicked_in_header.connect(self._on_header_cb_clicked)
        self.task_table.cellChanged.connect(self._on_row_cb_toggled)

        # 表头点击排序（第 0 列复选框除外——由 _HeaderCheckboxView 拦截）
        self._sort_column = None       # 当前排序列
        self._sort_desc = False        # 是否降序
        self._header_view.sectionClicked.connect(self._on_header_sort_clicked)

        # 表格移入双页堆叠（空态/表格页）——见分页段构建处


        # 分页栏：全 app 唯一形态 PaginationWidget（30 底栏、每页+上一页/页码/
        # 下一页；无跳转/统计，绝不折行——2026-09-29 用户裁定收编手写行）
        self.pagination_bar = PaginationWidget(
            default_page_size=self._page_size,
            page_size_options=["20", "50", "100", "200"])
        self.pagination_bar.page_changed.connect(self._on_pager_page_changed)
        self.pagination_bar.page_size_changed.connect(self._on_pager_size_changed)
        self.pagination_bar.setVisible(False)  # 空库隐藏（18§4.18）

        # 表格页（表格 + 分页底栏，同一卡片底色——17 号 7b：分页栏与面板同色）
        self._page_table = QWidget()
        # plain QWidget 的 QSS 底色不可靠——用 palette+autoFillBackground 确定性绘制
        self._page_table.setAutoFillBackground(True)
        _tc_pal = self._page_table.palette()
        _tc_pal.setColor(self._page_table.backgroundRole(), QColor(t.get('bg_card')))
        self._page_table.setPalette(_tc_pal)
        _pt = QVBoxLayout(self._page_table)
        _pt.setContentsMargins(0, 0, 0, 0)
        _pt.setSpacing(0)
        _pt.addWidget(self.task_table, 1)
        _pt.addWidget(self.pagination_bar)

        # 空态 hero（18§4.14）：0 任务时接管表格区。
        # 空态页必须铺与表格页一致的面板底色（bg_card）——否则整块区域
        # 露页面深底，读起来像"内容没有填充"（用户 2026-09-28 指正）
        self._page_empty = QWidget()
        self._page_empty.setStyleSheet(
            f"QWidget#PageEmpty {{ background-color: {t.get('bg_card')}; }}")
        self._page_empty.setObjectName("PageEmpty")
        _he = QVBoxLayout(self._page_empty)
        _he.setContentsMargins(0, 0, 0, 0)
        _he.addStretch(2)
        self._hero_art = QLabel("🎬")
        self._hero_art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hero_art.setFixedSize(64, 64)
        self._hero_art.setStyleSheet(
            f"font-size: 30px; background: {t.get('glass')}; border-radius: 16px;")
        _he.addWidget(self._hero_art, 0, Qt.AlignmentFlag.AlignHCenter)
        _t1 = QLabel("导入第一批素材")
        _t1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _t1.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {t.get('text_primary')}; background: transparent;")
        _he.addWidget(_t1, 0, Qt.AlignmentFlag.AlignHCenter)
        # 项目同时支持视频与照片（core/media_types.py：视频 mp4/mov/mkv…
        # 照片 jpg/png/heic…）——空态文案不再只提视频（用户 2026-09-29 指正）
        # 提升为实例属性：apply_theme 需重放（局部变量换肤够不到，
        # 浅色下 hero 文字白字白底不可见——2026-09-29 实测）
        self._hero_title = _t1
        self._hero_lines = []
        for _line in ("支持视频与照片（MP4 / MOV / JPG / PNG / HEIC 等常见格式），"
                      "可多选文件、整个文件夹，或从照片库导入。",
                      "导入后自动分析：场景切分 → 标签 → 封面。"):
            _tl = QLabel(_line)
            _tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _tl.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')}; background: transparent;")
            _he.addWidget(_tl, 0, Qt.AlignmentFlag.AlignHCenter)
            self._hero_lines.append(_tl)
        _he.addSpacing(10)
        _btnrow = QHBoxLayout()
        _btnrow.addStretch(1)
        self.btn_empty_import = QPushButton("＋ 选择文件或文件夹")
        self.btn_empty_import.setStyleSheet(
            "QPushButton { background: " + t.get('accent') + "; color: #FFFFFF;"
            " border: none; border-radius: 6px; padding: 0 16px; height: 28px;"
            " font-size: 13px; font-weight: 600; }"
            " QPushButton:hover { background: #2492FF; }")
        self.btn_empty_import.clicked.connect(self.import_mixed)
        _btnrow.addWidget(self.btn_empty_import)
        _btnrow.addStretch(1)
        _he.addLayout(_btnrow)
        _he.addStretch(3)

        self._task_stack = QStackedWidget()
        self._task_stack.addWidget(self._page_empty)
        self._task_stack.addWidget(self._page_table)
        task_layout.addWidget(self._task_stack, 1)

        # Footer row: stats + import status（移到分页上方：分页器保持贴底，
        # 空文本时隐藏避免占位空白——用户定案：分页器贴底、内容占满）
        footer_h = QHBoxLayout()
        footer_h.setContentsMargins(0, 0, 0, 0)
        footer_h.addStretch(1)  # 同上：默认 0 不扩展，导入状态需右贴
        self.import_status_label = QLabel("")
        self.import_status_label.setVisible(False)
        footer_h.addWidget(self.import_status_label)
        task_layout.addLayout(footer_h)

        # 导入进度轮询：导入后台线程在跑时，每 1s 拉一次 /task/import/status，
        # 把静态的「导入中...」换成「N/M」。导入结束（_on_import_done/_error）停表。
        self._import_poll_busy = False
        self._import_poll_timer = QTimer(self)
        self._import_poll_timer.setInterval(1000)
        self._import_poll_timer.timeout.connect(self._poll_import_progress)

        # Create a container widget for task_layout
        self.task_container = QWidget()
        self.task_container.setLayout(task_layout)
        self.task_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.task_container.setMinimumHeight(600)

        root.addWidget(self.task_container, 1)

        # 样式统一由 apply_theme 单点维护（本方法不再内联 QSS），构建完成后立即套用
        self.apply_theme()

    def apply_theme(self):
        """本页全部样式的单一维护点：_build_ui 不写内联 QSS，构建完成时调用本方法；
        换肤时由 MainWindow._toggle_theme 再次调用——仅重放样式，不重建控件，
        保留任务进度/勾选/分页等状态。"""
        from .theme import theme
        t = theme()
        self.setStyleSheet(f"background-color: {t.get('bg_primary')}; color: {t.get('text_primary')};")
        # 分页栏唯一形态组件换肤（QLabel/底色随主题，防深色残留）
        self.pagination_bar.apply_theme()
        # 双页堆叠（空态/表格页）底色重放——palette 型必须在换肤时重设
        # （构造烘焙，漏重放=浅色下内容区残留深色，2026-09-29 实测）
        self._page_empty.setStyleSheet(
            f"QWidget#PageEmpty {{ background-color: {t.get('bg_card')}; }}")
        _tc_pal2 = self._page_table.palette()
        _tc_pal2.setColor(self._page_table.backgroundRole(), QColor(t.get('bg_card')))
        self._page_table.setPalette(_tc_pal2)
        self._page_table.update()
        self.scheduler_status_label.setStyleSheet(f"""
            font-size: 13px;
            font-weight: 500;
            color: {t.get('text_secondary')};
        """)
        self.btn_import.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px 12px;
                font-size: {t.get('font_size_sm')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        self.batch_start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px 12px;
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
            QPushButton:disabled {{
                background: {t.get('border')};
                color: {t.get('text_tertiary')};
            }}
        """)
        self.batch_delete_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {t.get('danger')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px 12px;
                font-size: {t.get('font_size_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
            QPushButton:disabled {{
                background: {t.get('border')};
                color: {t.get('text_tertiary')};
            }}
        """)
        self.task_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {t.get('bg_card')};
                                color: {t.get('text_primary')};
                                            }}
            QTableWidget::item {{
                border-bottom: 1px solid {t.get('border')};
                background-color: {t.get('bg_secondary')};
            }}
            QTableWidget::item:alternate {{
                background-color: {t.get('bg_primary')};
            }}
            QHeaderView::section {{
                background-color: {t.get('bg_card')};
                color: {t.get('text_primary')};
                padding: 4px 6px;  /* 表头高一档制 28（17 号规范） */
                border: none;
                border-right: 1px solid {t.get('border')};
            }}
        """)
        # 表头钳高须在样式设置之后（QSS polish 清程序化 min/max，17 号规范）
        self.task_table.horizontalHeader().setFixedHeight(32)  # 表头 32（18§4.9）
        # 旧手写分页控件样式已随收编删除（唯一形态组件自带样式）
        self.import_status_label.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            color: {t.get('accent')};
            padding: 4px 0;
        """)
        self.task_container.setStyleSheet(f"background-color: {t.get('bg_primary')}; color: {t.get('text_primary')};")
        # 子组件(统计卡片)换肤
        for card in (self.stat_total, self.stat_completed, self.stat_processing,
                     self.stat_pending, self.stat_unqueued):
            if hasattr(card, "apply_theme"):
                card.apply_theme()
        # 空态 hero 重放（浅色下白字白底不可见的根因：局部变量无法重放）
        if hasattr(self, "_hero_art"):
            self._hero_art.setStyleSheet(
                f"font-size: 30px; background: {t.get('glass')};"
                f" border-radius: 16px;")
            self._hero_title.setStyleSheet(
                f"font-size: 15px; font-weight: 600;"
                f" color: {t.get('text_primary')}; background: transparent;")
            for _tl in self._hero_lines:
                _tl.setStyleSheet(
                    f"font-size: 12px;"
                    f" color: {t.get('text_secondary')}; background: transparent;")

    # ── Column width persistence ──────────────────────────────────────

    def _load_column_widths(self):
        """Load saved column widths from JSON, or apply defaults."""
        widths = None
        try:
            if self._column_widths_path.exists():
                with open(self._column_widths_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                widths = {int(k): int(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError, OSError):
            widths = None

        if widths is None:
            widths = self._default_column_widths

        # Apply widths to all columns
        for col, width in widths.items():
            if 0 <= col < self.task_table.columnCount():
                self.task_table.setColumnWidth(col, width)

        # 文件名列始终 Stretch 吃掉剩余宽度——持久化的 setColumnWidth 会把
        # Stretch 降级为 Interactive，表格右侧留大片空白（08 细调：用户报告）
        self._header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def _on_column_width_changed(self, logical_index: int, old_size: int, new_size: int):
        """Column width changed — schedule a debounced save."""
        # Skip Fixed columns (0: checkbox, 5: file_size, 6: duration, 7: action) — they don't change
        if logical_index in (0, 5, 6, 7):
            return
        self._column_width_save_timer.start()

    def _save_column_widths(self):
        """Persist current column widths to JSON."""
        widths = {}
        for col in range(self.task_table.columnCount()):
            widths[str(col)] = self.task_table.columnWidth(col)
        try:
            self._column_widths_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._column_widths_path, "w", encoding="utf-8") as f:
                json.dump(widths, f, indent=2)
        except OSError:
            pass  # Silently ignore save failures

    # ── Selection sync ──────────────────────────────────────────────

    def _on_header_cb_clicked(self):
        """Header checkbox clicked → 全选/取消当前页。

        语义与全选按钮一致(2026-09 确认):
          - 勾选表头 = 选择集重置为仅当前页全部(清其他页)
          - 取消表头 = 仅取消当前页(其他页保留)
        """
        # _toggle_check has already updated the state, so we match the new state
        to_check = (self._header_view._check_state == Qt.CheckState.Checked)
        new_state = Qt.CheckState.Checked if to_check else Qt.CheckState.Unchecked
        
        # 通过模型更新数据，确保 delegate 能正确重绘
        model = self.task_table.model()
        self.task_table.blockSignals(True)
        try:
            for i in range(self.task_table.rowCount()):
                index = model.index(i, 0)
                model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
        finally:
            self.task_table.blockSignals(False)

        # 同步选择集(blockSignals 期间 cellChanged 不触发)
        if not hasattr(self, '_checked_ids'):
            self._checked_ids = set()
        page_ids = set()
        for i in range(self.task_table.rowCount()):
            tid_item = self.task_table.item(i, 1)
            if not tid_item:
                continue
            tid = tid_item.data(Qt.ItemDataRole.UserRole)
            if tid:
                page_ids.add(tid)
        if to_check:
            # 勾选表头 = 累加当前页(不清其他页)
            for tid in page_ids:
                self._checked_ids.add(tid)
        else:
            # 取消表头 = 仅移除当前页
            for tid in page_ids:
                self._checked_ids.discard(tid)

        # 强制刷新整个表格以更新 delegate 绘制
        self.task_table.viewport().update()
        self._update_selection_count()

    def _on_row_cb_toggled(self, row: int, column: int):
        """A row checkbox toggled → sync header checkbox state + global checked set."""
        # Only process checkbox column (column 0)
        if column == 0:
            # 同步全局勾选集合(_checked_ids, 跨页持久)
            if not hasattr(self, '_checked_ids'):
                self._checked_ids = set()
            item = self.task_table.item(row, 0)
            tid_item = self.task_table.item(row, 1)
            if item and tid_item:
                tid = tid_item.data(Qt.ItemDataRole.UserRole)
                if tid:
                    if item.checkState() == Qt.CheckState.Checked:
                        self._checked_ids.add(tid)
                    else:
                        self._checked_ids.discard(tid)
            self._sync_header_from_rows()
            self._update_selection_count()

    def _invert_selection(self):
        """反选 = 仅反转当前页勾选(其他页勾选保留)。"""
        if not hasattr(self, '_checked_ids'):
            self._checked_ids = set()
        for i in range(self.task_table.rowCount()):
            item = self.task_table.item(i, 0)
            tid_item = self.task_table.item(i, 1)
            if not (item and tid_item):
                continue
            tid = tid_item.data(Qt.ItemDataRole.UserRole)
            if not tid:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.Unchecked)
                self._checked_ids.discard(tid)
            else:
                item.setCheckState(Qt.CheckState.Checked)
                self._checked_ids.add(tid)
        self._sync_header_from_rows()
        self._update_selection_count()

    def _select_all(self):
        """全选 = 在当前勾选基础上追加勾选当前页全部(跨页累加)。

        业务语义(2026-09 最终确认): 全选/表头勾选 = 累加当前页, 不清其他页
        勾选。跨页操作通过逐页全选或逐行勾选累积选择集。
        """
        if not hasattr(self, '_checked_ids'):
            self._checked_ids = set()
        for i in range(self.task_table.rowCount()):
            item = self.task_table.item(i, 0)
            tid_item = self.task_table.item(i, 1)
            if item and tid_item:
                tid = tid_item.data(Qt.ItemDataRole.UserRole)
                if tid:
                    item.setCheckState(Qt.CheckState.Checked)
                    self._checked_ids.add(tid)
        self._sync_header_from_rows()
        self._update_selection_count()

    def _deselect_all(self):
        """取消全选 = 仅取消当前页的勾选(其他页勾选保留)。"""
        if not hasattr(self, '_checked_ids'):
            self._checked_ids = set()
        for i in range(self.task_table.rowCount()):
            item = self.task_table.item(i, 0)
            tid_item = self.task_table.item(i, 1)
            if item and tid_item:
                item.setCheckState(Qt.Unchecked)
                tid = tid_item.data(Qt.ItemDataRole.UserRole)
                if tid:
                    self._checked_ids.discard(tid)
        self._sync_header_from_rows()
        self._update_selection_count()

    def _update_selection_count(self):
        # 全局勾选集合(跨页) + 当前页行数
        global_checked = getattr(self, '_checked_ids', set())
        count = len(global_checked)

        # 当前页中处于勾选态的行数(用于表头三态显示, 与全局不同)
        page_checked = 0
        for i in range(self.task_table.rowCount()):
            item = self.task_table.item(i, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                page_checked += 1

        page_rows = self.task_table.rowCount()
        # 统计迁至程序底部状态栏（UI 重构 08 方案）
        w = self.window()
        if hasattr(w, "update_task_stats"):
            if global_checked and len(global_checked) > page_checked:
                w.update_task_stats(
                    f"共 {page_rows} 项(本页)，选中 {len(global_checked)} 项"
                    f"(含其他页 {len(global_checked) - page_checked} 项)"
                )
            else:
                w.update_task_stats(f"共 {page_rows} 项，选中 {count} 项")

        # 根据选中数量启用/禁用批量操作按钮(全局选中数驱动)
        has_selected = count > 0

        if hasattr(self, 'batch_start_btn'):
            self.batch_start_btn.setEnabled(has_selected)
        if hasattr(self, 'batch_delete_btn'):
            self.batch_delete_btn.setEnabled(has_selected)

        if hasattr(self, 'btn_pin_top'):
            self.btn_pin_top.setEnabled(has_selected)
        if hasattr(self, 'btn_assign_project'):
            self.btn_assign_project.setEnabled(has_selected)
        if hasattr(self, 'batch_unqueue_btn'):
            self.batch_unqueue_btn.setEnabled(has_selected)

    def _sync_header_from_rows(self):
        """Sync the header checkbox to match row states."""
        has_rows = self.task_table.rowCount() > 0
        if not has_rows:
            self._header_view.set_check_state(Qt.Unchecked)
            return
        all_checked = True
        any_checked = False
        for i in range(self.task_table.rowCount()):
            item = self.task_table.item(i, 0)
            if item:
                is_checked = (item.checkState() == Qt.CheckState.Checked)
                if is_checked:
                    any_checked = True
                else:
                    all_checked = False
        if all_checked:
            self._header_view.set_check_state(Qt.CheckState.Checked)
        elif any_checked:
            self._header_view.set_check_state(Qt.PartiallyChecked)
        else:
            self._header_view.set_check_state(Qt.Unchecked)

    # ── Polling ─────────────────────────────────────────────────────

    def _start_polling(self):
        """Start QTimer polling every 10 seconds — balances responsiveness and API load."""
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(10000)
        self.poll_timer.timeout.connect(self._poll_status)
        self.poll_timer.start()
        self._poll_status()  # immediate first poll

    def cleanup(self):
        """Stop all background threads and timers. Must be called before app exit."""
        # Stop poll timer first to prevent new workers from being created
        if hasattr(self, 'poll_timer') and self.poll_timer is not None:
            self.poll_timer.stop()
        if hasattr(self, '_import_poll_timer') and self._import_poll_timer is not None:
            self._import_poll_timer.stop()
        
        # 同步 worker 无残留线程；仅复位重入标志
        self._poll_inflight = False
        
        # Shutdown thread pool executor
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)

    # ── Poll via _StatusWorker (daemon thread, no QThread) ──

    # ── 任务来源树（09 方案）────────────────────────────────────────

    def take_title_cards(self):
        """13 号：统计行面板交由主窗口放入统一标题栏（重复调用返回 None）。"""
        panel, self.title_cards = self.title_cards, None
        return panel

    def take_source_panel(self):
        """10 号方案：交出来源树面板，由 MainWindow reparent 到统一侧栏。

        面板在 _build_ui 中即为无父级构建（未加入本页布局），此处只负责
        交接引用；self.source_tree 等子控件随面板走，本页方法照常可用。
        返回 None 表示已被取走（重复调用防御）。
        """
        panel, self.source_panel = getattr(self, "source_panel", None), None
        if panel is not None:
            panel.setParent(None)
        return panel

    def _refresh_source_tree(self):
        """后台线程拉 /task/folders，经信号回主线程重建来源树。"""
        self._executor.submit(self._fetch_source_folders)

    def _fetch_source_folders(self):
        try:
            resp = self.api.get("/task/folders", timeout=3)
        except Exception:
            return
        self._folders_finished.emit(resp.get("folders", []))

    def _on_folders_fetched(self, folders):
        """主线程：重建来源树（名称+计数，计数由委托右对齐绘制 18§3.5）。"""
        from PySide6.QtWidgets import QTreeWidgetItem
        self.source_tree.clear()
        total = sum(f["count"] for f in folders)
        top_all = QTreeWidgetItem(["全部"])
        top_all.setData(0, Qt.ItemDataRole.UserRole, None)
        top_all.setData(0, Qt.ItemDataRole.UserRole + 1, total)
        self.source_tree.addTopLevelItem(top_all)
        for f in folders:
            d = f["dir"]
            name = d.rstrip("/").rsplit("/", 1)[-1] or d
            node = QTreeWidgetItem([name])
            node.setToolTip(0, d)
            node.setData(0, Qt.ItemDataRole.UserRole, d)
            node.setData(0, Qt.ItemDataRole.UserRole + 1, f["count"])
            self.source_tree.addTopLevelItem(node)
        if self._source_folder is None:
            self.source_tree.setCurrentItem(top_all)

    def _on_source_clicked(self, item, col):
        """来源节点点击 → 设置目录过滤 → 重拉（服务端过滤，跨页生效）。"""
        self._source_folder = item.data(0, Qt.ItemDataRole.UserRole)
        self._project_filter = None  # 两过滤器互斥（T1）
        self._current_page = 1
        self._poll_status()

    # ── T1 双视图（§3.1）：来源/项目镜片 ──

    def _apply_lens_style(self):
        """T1 镜片分段控件样式（构造/切换/换肤共用）。
        选中段=bg_card 填充+主色字 600；未选段=透明+txt2。"""
        t = theme()
        _on = ("QLabel { color: " + t.get('text_primary') + "; font-weight: 600;"
               " background: " + t.get('bg_card') + "; border-radius: 5px; }")
        _off = ("QLabel { color: " + t.get('text_secondary') + ";"
                " background: transparent; }")
        self._lens_src.setStyleSheet(_on if self._lens_is_source else _off)
        self._lens_proj.setStyleSheet(_off if self._lens_is_source else _on)


    def _set_lens_source(self):
        self._lens_is_source = True
        self._apply_lens_style()
        self._lens_proj.setStyleSheet(
            "QLabel { padding: 3px 10px; border-radius: 5px; font-size: 11px;"
            " color: " + theme().get('text_secondary') + "; }")
        self.source_tree.setVisible(True)
        self.project_tree.setVisible(False)
        self._project_filter = None
        self._current_page = 1
        self._poll_status()

    def _set_lens_project(self):
        self._lens_is_source = False
        self._apply_lens_style()
        self._lens_src.setStyleSheet(
            "QLabel { padding: 3px 10px; border-radius: 5px; font-size: 11px;"
            " color: " + theme().get('text_secondary') + "; }")
        self.project_tree.setVisible(True)
        self.source_tree.setVisible(False)
        self._source_folder = None
        # 每次切换都刷新：项目可在文件详情页被新建/删除（原懒加载一次
        # 导致他页建的项目在本页永远不可见）
        self._executor.submit(self._fetch_project_tree)
        self._current_page = 1
        self._poll_status()

    def _fetch_project_tree(self):
        """后台线程：拉项目清单（纯数据），回主线程由 _apply_project_tree 套树。"""
        try:
            resp = self.api.get("/api/studio/projects", params={"size": 100}, timeout=3)
            projects = resp.get("projects", [])
        except Exception:
            projects = []
        self._project_tree_data = [
            (pj.get("id"), pj.get("parent_id"), pj.get("name", "?"))
            for pj in projects if pj.get("id") is not None]
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._apply_project_tree)

    @Slot()
    def _apply_project_tree(self):
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QTreeWidgetItem
        flat = getattr(self, "_project_tree_data", [])
        self.project_tree.clear()

        def fill(parent_id, parent_item):
            for pid, ppid, name in flat:
                if ppid == parent_id:
                    it = QTreeWidgetItem([name])
                    it.setData(0, _Qt.ItemDataRole.UserRole, pid)
                    parent_item.addChild(it)
                    fill(pid, it)

        top = QTreeWidgetItem(["全部项目"])
        top.setData(0, _Qt.ItemDataRole.UserRole, None)
        self.project_tree.addTopLevelItem(top)
        fill(None, self.project_tree)
        known = {pid for pid, _, _ in flat}
        for pid, ppid, name in flat:
            if ppid is not None and ppid not in known:
                it = QTreeWidgetItem([name])
                it.setData(0, _Qt.ItemDataRole.UserRole, pid)
                self.project_tree.addTopLevelItem(it)

    def _on_project_clicked(self, item, col):
        """项目节点点击 → 设置项目过滤 → 重拉（服务端过滤，跨页生效）。"""
        self._project_filter = item.data(0, Qt.ItemDataRole.UserRole)
        self._source_folder = None
        self._current_page = 1
        self._poll_status()

    # ── 归到项目（project-assets-roster：任务页逻辑分类写动作）──

    def _fetch_all_projects(self):
        """拉全量项目清单（服务端 size 上限 100，分页取全）。"""
        projects, page = [], 1
        while True:
            resp = self.api.get(
                "/api/studio/projects",
                params={"size": 100, "page": page}, timeout=5)
            batch = resp.get("projects", [])
            projects.extend(batch)
            if not batch or page * 100 >= resp.get("total", 0):
                return projects
            page += 1

    def _assign_selected_to_project(self):
        """选中工单 → 归到项目（写内容归属；重复加入幂等并提示）。"""
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            return
        try:
            projects = self._fetch_all_projects()
        except Exception as exc:
            QMessageBox.warning(self, "读取项目失败", str(exc))
            return

        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        act_new = menu.addAction("＋ 新建项目…")
        act_remove = menu.addAction("移出所有项目")
        if projects:
            menu.addSeparator()
        for prj in projects:
            act = menu.addAction(f'{prj.get("name")}（{prj.get("asset_count", 0)}）')
            act.setData(prj.get("id"))
        chosen = menu.exec(self.btn_assign_project.mapToGlobal(
            self.btn_assign_project.rect().bottomLeft()))
        if chosen is None:
            return
        try:
            if chosen is act_new:
                # 无项目时可直接新建：全量弹窗（客户可输入新名即时创建）
                from .results_page import ProjectDialog
                dlg = ProjectDialog(self.api, self.window())
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                new_pid = dlg.save()
                r = self.api.post(
                    f"/api/studio/projects/{new_pid}/assign",
                    json={"task_ids": task_ids}, timeout=10)
            elif chosen is act_remove:
                r = self.api.post(
                    "/api/studio/unassign",
                    json={"task_ids": task_ids}, timeout=10)
            else:
                r = self.api.post(
                    f"/api/studio/projects/{chosen.data()}/assign",
                    json={"task_ids": task_ids}, timeout=10)
            dups = r.get("duplicates", 0)
            if dups:
                QMessageBox.information(
                    self, "归到项目",
                    f"已归集 {r.get('assigned', 0)} 个内容，{dups} 个已在该项目中")
        except Exception as exc:
            QMessageBox.warning(self, "归到项目失败", str(exc))
            return
        # 项目镜片激活时列表会随归属变化 → 刷新；树计数同步重拉
        if not self._lens_is_source:
            self.refresh_data()
            self._executor.submit(self._fetch_project_tree)

    def _poll_status(self):
        """Poll /status and /task（后台线程执行，信号回主线程，不阻塞 UI）。"""
        # Build pagination params
        params = {
            "page": self._current_page,
            "page_size": self._page_size,
        }

        # Add status filter if not "all"
        if self._current_filter != "all":
            params["status"] = self._current_filter
        # 人脸扫描状态过滤(服务端过滤, 跨页生效)
        if self._current_face_filter != "all":
            params["face_scan_status"] = self._current_face_filter
        # 任务来源树：目录过滤（服务端前缀匹配，跨页生效）
        if getattr(self, "_source_folder", None):
            params["folder"] = self._source_folder
        # T1 项目视图：项目过滤（服务端 project_id 精确匹配）
        if getattr(self, "_project_filter", None) is not None:
            params["project_id"] = self._project_filter

        # 重入守卫：上一轮网络未返回前跳过本轮（API 偶发卡顿时防止堆积）
        if getattr(self, "_poll_inflight", False):
            return
        self._poll_inflight = True
        self._executor.submit(self._poll_fetch, params)

    def _poll_fetch(self, params):
        """后台线程：拉取状态+任务页，经信号回主线程。禁动任何 UI。"""
        try:
            status = self.api.get("/status", timeout=3)
            task_resp = self.api.get("/task", params=params, timeout=3)
            if isinstance(task_resp, dict) and "tasks" in task_resp:
                task_list = task_resp["tasks"]
                pagination_info = {
                    "total": task_resp.get("total", len(task_list)),
                    "page": task_resp.get("page", 1),
                    "page_size": task_resp.get("page_size", len(task_list)),
                    "total_pages": task_resp.get("total_pages", 1),
                }
            else:
                task_list = task_resp if isinstance(task_resp, list) else []
                pagination_info = {
                    "total": len(task_list),
                    "page": 1,
                    "page_size": len(task_list),
                    "total_pages": 1,
                }
            self._poll_finished.emit(status, task_list, pagination_info)
        except Exception as e:
            self._poll_failed.emit(str(e))

    def _on_poll_status_done(self, status, task_list, pagination_info):
        """Called on main thread via signal connection after poll completes.

        Dedup optimization: if task signature (id, status, progress) is identical
        to the previous poll, skip table rebuild — only update stat card values.
        """
        self._poll_inflight = False

        # 来源树为空（首次建树时 API 未就绪被吞）→ 补建一次
        if self.source_tree.topLevelItemCount() == 0:
            self._refresh_source_tree()
        
        # Check if widget is still valid (not deleted)
        try:
            if not self.isVisible():
                return
        except RuntimeError:
            # Widget already deleted
            return
        
        try:
            self._connected = True
            self._last_task_list = task_list  # Save for filter updates
            self._last_pagination_info = pagination_info  # Save pagination info
            
            # Clear error message if connection restored
            if hasattr(self, '_had_error') and self._had_error:
                self._had_error = False
                self._update_selection_count()
            
            # Update pagination state
            self._total_items = pagination_info.get("total", 0)
            self._total_pages = pagination_info.get("total_pages", 1)
            self._update_pagination_ui()

            # ── Dedup: compare task signature with last poll ─────────
            # Include status + progress + file_size + duration + face_scan_status
            # so metadata changes (e.g. face scan result) trigger table refresh
            new_signature = tuple(
                (t.get("id", ""), t.get("status", ""), t.get("progress", 0), 
                 t.get("file_size", 0), t.get("duration", 0),
                 t.get("face_scan_status", ""))
                for t in task_list
            )
            if hasattr(self, '_last_task_signature') and new_signature == self._last_task_signature:
                # Data unchanged — only update stat card values (lightweight)
                self._update_stats(status, task_list)
                self._restore_refresh_btn()
                return
            self._last_task_signature = new_signature

            # Data changed — do full update
            self._update_stats(status, task_list)
            self._update_task_table(task_list)
            self._restore_refresh_btn()
        except RuntimeError:
            # Widget was deleted during update
            pass
        except Exception as e:
            self._connected = False
            # Don't spam errors during startup
            if hasattr(self, '_silent'):
                return
            w = self.window()
            if hasattr(w, "update_task_stats"):
                w.update_task_stats(f"API 连接错误: {e}")

    def _on_poll_error(self, error_msg: str):
        """Handle polling errors — update UI without blocking."""
        self._poll_inflight = False
        self._connected = False
        self._had_error = True
        self._restore_refresh_btn()
        if hasattr(self, '_silent'):
            return
        w = self.window()
        if hasattr(w, "update_task_stats"):
            w.update_task_stats(f"轮询错误: {error_msg}")

    # ── Stats & progress update ─────────────────────────────────────

    def _update_stats(self, status_data: dict, task_list: list):
        """Update stat cards.

        统计卡基于 /status 的全量统计(所有导入文件, 非当前页),
        task_list(当前页)只用于表格展示。
        """
        # 全量统计(后端 /status): total_tasks / completed / processing / pending / unqueued
        total = int(status_data.get("total_tasks", status_data.get("total", len(task_list))) or 0)
        completed = int(status_data.get("completed", 0) or 0)
        processing = int(status_data.get("processing", 0) or 0)
        pending = int(status_data.get("pending", 0) or 0)
        unqueued = int(status_data.get("unqueued", 0) or 0)

        self.stat_total.set_value(total)
        self.stat_completed.set_value(completed)
        self.stat_processing.set_value(processing)
        self.stat_pending.set_value(pending)
        self.stat_unqueued.set_value(unqueued)

        # 更新调度器状态
        try:
            scheduler_status = self.api.get("/scheduler/status")
            self._update_scheduler_ui(
                scheduler_status.get("status", "stopped"),
                scheduler_status
            )
        except Exception:
            pass  # 忽略调度器状态获取失败

    def _on_header_sort_clicked(self, column: int):
        """表头点击：切换排序（再次点击切换升降序）。"""
        if column == 0:
            return  # 第 0 列是复选框，不排序
        if self._sort_column == column:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_column = column
            self._sort_desc = False
        # 显示排序指示箭头
        try:
            self._header_view.setSortIndicatorShown(True)
            self._header_view.setSortIndicator(
                column,
                Qt.SortOrder.DescendingOrder if self._sort_desc else Qt.SortOrder.AscendingOrder,
            )
        except Exception:
            pass
        # 清除去重签名，强制表格按新排序重绘（否则数据未变时 Dedup 会跳过刷新）
        if hasattr(self, '_last_task_signature'):
            delattr(self, '_last_task_signature')
        self.refresh_data()

    def _apply_sort(self, task_list: list) -> list:
        """按当前排序列对任务列表排序（数据层排序，避免与增量更新冲突）。

        行位置稳定(2026-09): 未点表头排序时(col is None), 不跟随后端
        updated_at 动态序——否则任务状态变化会触发重排导致行漂移。
        改为: 用本页首次加载的 task_id 顺序快照(_stable_page_order)固定行序,
        状态/进度变化只原位更新单元格, 行位置不动。
        """
        col = self._sort_column
        if col is None or col <= 0:
            # 默认稳定序: 与 _stable_page_order 快照对齐
            ids_now = [t.get("id", "") for t in task_list]
            if ids_now and hasattr(self, '_stable_page_order') and self._stable_page_order:
                snap = self._stable_page_order
                # 快照与当前集合一致 → 按快照排(保持位置); 否则重建快照
                if set(ids_now) == set(snap):
                    by_id = {t.get("id", ""): t for t in task_list}
                    ordered = [by_id[i] for i in snap if i in by_id]
                    # 补齐新出现(快照外的)任务, 追加到尾部保持可见
                    seen = set(snap)
                    ordered += [t for t in task_list if t.get("id") not in seen]
                    return ordered
            self._stable_page_order = ids_now
            return task_list

        # 列号 → 排序键函数
        def sort_key(t):
            if col == 1:    # 文件名
                return (t.get("file_name") or "").lower()
            elif col == 2:  # 文件夹
                return (t.get("folder_path") or "").lower()
            elif col == 3:  # 文件大小
                return t.get("file_size") or 0
            elif col == 4:  # 时长
                return t.get("duration") or 0
            elif col == 5:  # 人脸状态：未扫描排最前（方便筛选未识别），其余按状态排序
                scan = t.get("face_scan_status")
                order = {"complete": 1, "none": 2, "skipped": 3, None: 0}
                return order.get(scan, 0)
            elif col == 6:  # 状态
                return t.get("status") or ""
            elif col == 7:  # 进度
                return t.get("progress") or 0
            else:
                return t.get("id") or 0

        try:
            return sorted(task_list, key=sort_key, reverse=self._sort_desc)
        except Exception:
            return task_list

    def _update_task_table(self, task_list: list):
        """Incremental update of the task table.

        Key optimizations:
        1. Reuse QTableWidgetItem objects (only update text/checkState), don't recreate.
        2. Cache cell widgets (progress bars, action buttons) — reuse when status unchanged.
        3. Only add/remove rows for actual changes.
        4. Store checkbox states by task_id so they survive filter changes.

        全局勾选语义(2026-09): _checked_ids 是按 task_id 的**跨页全局集合**,
        不随当前页行快照覆盖——翻页/筛选/刷新/状态变化后勾选保持。
        """
        if not hasattr(self, '_checked_ids'):
            self._checked_ids = set()

        # ── 过滤已全部在服务端完成(/task?status=&face_scan_status=&page=) ──
        # 服务端按筛选条件对全量任务过滤后重新分页, 返回当前页 task_list。
        # 客户端不再二次过滤(历史 bug: 客户端对当前页过滤 → 当前页无匹配即空,
        # 且分页 total 与实际不符)。直接渲染服务端返回的当前页。
        filtered = task_list

        # 表头排序（数据层排序，保持增量更新逻辑不变）
        filtered = self._apply_sort(filtered)

        # ── Build current task_id set & lookup (use list order, not set) ──
        # ── 统一全量重建 ──
        # 历史: 行数相同时走增量路径(原位复用 item + 后设 checkState), 实测在本表
        # 上不可靠——checkbox 状态设了但界面不显示(勾选丢失, 表头仍有效)。
        # 全量重建新建 item(先 Unchecked), 随后用 model.setData 设置勾选——
        # setData 走模型正常信号路径, 视图可靠刷新(实测 blockSignals 期间
        # setCheckState 不触发重绘, 导致状态在但界面不显示)。
        checked_by_id = getattr(self, '_checked_ids', set())

        # Block signals during rebuild to prevent cellChanged firing for every item
        self.task_table.blockSignals(True)
        try:
            self._rebuild_table_rows(filtered, checked_by_id)
        finally:
            self.task_table.blockSignals(False)

        # 用 model.setData 恢复勾选(信号已恢复, 触发正常 dataChanged → 视图刷新)
        if checked_by_id:
            model = self.task_table.model()
            for i in range(self.task_table.rowCount()):
                tid_item = self.task_table.item(i, 1)
                tid = tid_item.data(Qt.ItemDataRole.UserRole) if tid_item else None
                if tid and tid in checked_by_id:
                    idx = model.index(i, 0)
                    model.setData(idx, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)

        # Manually sync header checkbox after rebuild
        self._sync_header_from_rows()

        # Store for batch operations
        self._task_rows = {}

        # Footer
        self._update_selection_count()


    def _set_task_empty(self, empty: bool):
        """空态收敛（18§4.18）：空/非空两页切换 + 分页栏联动隐藏。"""
        if hasattr(self, "_task_stack"):
            self._task_stack.setCurrentIndex(1 if empty is False else 0)
        if hasattr(self, "pagination_bar"):
            self.pagination_bar.setVisible(not empty)

    def _rebuild_table_rows(self, filtered: list, checked_by_id: set):
        """Clear and repopulate all table rows (called with signals blocked)."""
        new_row_count = len(filtered)
        self.task_table.setRowCount(new_row_count)

        # 空态收敛（18§4.18）：0 任务 → hero 接管表格区、分页栏隐藏
        self._set_task_empty(new_row_count == 0)

        for i, task in enumerate(filtered):
            tid = task.get("id", "")
            status = task.get("status", "PENDING")
            info = STATUS_MAP.get(status, ("未知", "text_secondary", "?"))
            progress = task.get("progress", 0)
            if isinstance(progress, (int, float)):
                pass
            elif isinstance(progress, str):
                try:
                    progress = float(progress.replace("%", ""))
                except ValueError:
                    progress = 0
            else:
                progress = 0

            # Checkbox: 先统一 Unchecked 建 item; 勾选状态在 rebuild 结束后由
            # model.setData 统一设置(走正常信号路径, 确保视图刷新显示)。
            cb = QTableWidgetItem()
            cb.setFlags(cb.flags() | Qt.ItemIsUserCheckable)
            cb.setFlags(cb.flags() & ~Qt.ItemIsEditable)
            cb.setCheckState(Qt.CheckState.Unchecked)
            cb.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐
            cb.setData(Qt.ItemDataRole.UserRole, tid)
            self.task_table.setItem(i, 0, cb)

            # File name (column 1) — truncate for display, store full name in tooltip
            full_name = sanitize_for_qt(task.get("file_name", str(tid)))
            display_name = truncate_filename(full_name)
            name_item = QTableWidgetItem(display_name)
            name_item.setToolTip(full_name)  # Show full name on hover
            name_item.setData(Qt.ItemDataRole.UserRole, tid)
            self.task_table.setItem(i, 1, name_item)

            # Folder path (column 2) — shorten for display
            folder_path = task.get("folder_path", "")
            display_folder = shorten_folder_path(folder_path)
            folder_item = QTableWidgetItem(display_folder)
            folder_item.setToolTip(folder_path)  # Show full path on hover
            folder_item.setForeground(QColor(theme().get("text_secondary")))  # Gray color for folder
            self.task_table.setItem(i, 2, folder_item)

            # File size (column 3)
            file_size_item = QTableWidgetItem(format_file_size(task.get("file_size")))
            file_size_item.setForeground(QColor(theme().get("text_secondary")))
            file_size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.task_table.setItem(i, 3, file_size_item)

            # Duration (column 4)
            duration_item = QTableWidgetItem(format_duration(task.get("duration")))
            duration_item.setForeground(QColor(theme().get("text_secondary")))
            duration_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.task_table.setItem(i, 4, duration_item)

            # Face scan status (column 5) — 文件维度人脸识别状态
            face_scan = task.get("face_scan_status")
            if face_scan == "complete":
                face_text, face_color = "✓ 已识别", "success"
            elif face_scan == "none":
                face_text, face_color = "🙅 无人脸", "text_secondary"
            elif face_scan == "skipped":
                face_text, face_color = "⏭ 跳过", "warning"
            elif face_scan == "low_confidence":
                face_text, face_color = "⚠ 低置信", "warning"
            else:
                face_text, face_color = "—", "text_secondary"
            face_item = QTableWidgetItem(face_text)
            face_item.setForeground(QColor(theme().get(face_color)))
            face_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if face_scan in ("none", "skipped", "complete"):
                face_item.setToolTip("人脸识别结果（文件维度，任务清空后仍保留）")
            self.task_table.setItem(i, 5, face_item)

            # Status (column 6)
            # Show fine-grained scene progress during ANALYZING stage
            error_msg = task.get("error_message")
            if status == "ANALYZING" and error_msg and "场景" in error_msg:
                display_status = sanitize_for_qt(error_msg)
            else:
                display_status = sanitize_for_qt(f"{info[2]} {info[0]}")
            status_item = QTableWidgetItem(display_status)
            status_item.setForeground(QColor(theme().get(info[1])))
            # Add tooltip for error status showing error message
            if error_msg and status in ("ERROR", "FAILED"):
                status_item.setToolTip(sanitize_for_qt(error_msg))
            self.task_table.setItem(i, 6, status_item)

            # Progress bar widget (column 7)
            prog_widget = QWidget()
            prog_layout = QHBoxLayout(prog_widget)
            prog_layout.setContentsMargins(2, 0, 2, 0)
            bar = QProgressBar()
            bar.setValue(int(progress))
            bar.setTextVisible(True)
            bar.setMinimumWidth(80)
            t = theme()
            bar.setStyleSheet(f"""
                QProgressBar {{ border: 1px solid {t.get('border')}; border-radius: 3px; text-align: center; }}
                QProgressBar::chunk {{ background: {t.get('bg_hover')}; border-radius: 2px; }}
            """)
            # Show fine-grained scene progress tooltip during ANALYZING
            _err = task.get("error_message")
            if status == "ANALYZING" and _err and "场景" in _err:
                bar.setToolTip(sanitize_for_qt(_err))
            prog_layout.addWidget(bar)
            self.task_table.setCellWidget(i, 7, prog_widget)

            # Task ID (column 8) — 显示任务ID，便于排序时核对行对应关系
            id_item = QTableWidgetItem(str(tid))
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            id_item.setForeground(QColor(theme().get("text_secondary")))
            id_item.setData(Qt.ItemDataRole.UserRole, tid)
            self.task_table.setItem(i, 8, id_item)

    def _show_task_context_menu(self, pos):
        """右键任务行菜单：在文件夹中显示视频（人工核对人脸识别结果）。"""
        from PySide6.QtWidgets import QMenu

        item = self.task_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        # 获取该行 task_id（从 checkbox 或文件名列 UserRole）
        tid = None
        for col in (0, 1):
            it = self.task_table.item(row, col)
            if it and it.data(Qt.ItemDataRole.UserRole):
                tid = it.data(Qt.ItemDataRole.UserRole)
                break
        if tid is None:
            return

        # 从最近的任务列表找 file_path
        file_path = None
        task_list = getattr(self, '_last_task_list', [])
        for t in task_list:
            if t.get("id") == tid:
                file_path = t.get("file_path")
                break

        menu = QMenu(self)
        t = theme()
        menu.setStyleSheet(f"""
            QMenu {{ background: {t.get('bg_card')}; color: {t.get('text_primary')};
                       border: 1px solid {t.get('border')}; }}
            QMenu::item:selected {{ background: {t.get('bg_hover')}; }}
        """)
        act_show = menu.addAction("📂 在文件夹中显示")
        act_show.setEnabled(bool(file_path))
        act_detail = menu.addAction("查看任务详情")

        chosen = menu.exec(self.task_table.viewport().mapToGlobal(pos))
        if chosen == act_show and file_path:
            self._reveal_in_finder(file_path)
        elif chosen == act_detail:
            self._view_task_detail(tid)

    @staticmethod
    def _reveal_in_finder(file_path: str):
        """在 macOS Finder 中显示文件（open -R）。"""
        import subprocess
        try:
            subprocess.Popen(["open", "-R", file_path])
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("在 Finder 中显示失败: %s", e)

    # ── Folder picker ───────────────────────────────────────────────

    def _pick_folder(self):
        """Pick a folder via osascript (Finder) with QFileDialog fallback."""
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 "tell application \"Finder\" to POSIX path of "
                 "(choose folder with prompt \"选择要导入的文件夹\")"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                folder_path = result.stdout.strip().rstrip("/")
                self._scan_folder(folder_path)
                return
        except Exception:
            pass
        # Fallback to QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self._scan_folder(folder)

    def _scan_folder(self, folder_path: str):
        """Scan a folder and store file paths. Returns list of video files."""
        try:
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
            files = []
            for root, _dirs, fnames in os.walk(folder_path):
                for f in fnames:
                    if Path(f).suffix.lower() in video_exts:
                        files.append(os.path.join(root, f))
            if not files:
                return []
            self._selected_folders[folder_path] = files
            return files
        except Exception as e:
            QMessageBox.critical(self, "错误", f"扫描文件夹失败: {e}")
            return []

    # ── Import ──────────────────────────────────────────────────────

    def _import_selected(self):
        """Import selected folders/files via POST /task/import."""
        if not self._selected_folders:
            QMessageBox.warning(self, "无文件夹", "请先选择文件夹。")
            return

        total_files = sum(len(fs) for fs in self._selected_folders.values())
        if QMessageBox.question(self, "确认",
                                f"将导入 {total_files} 个视频文件?\n\n"
                                + "\n".join(self._selected_folders.keys()),
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        fps = [fp for files in self._selected_folders.values() for fp in files]
        try:
            resp = self.api.post("/task/import", json={"files": fps}, timeout=1800)
            if resp.get("imported", 0) > 0 or resp.get("updated", 0) > 0:
                imported = resp.get('imported', 0)
                updated = resp.get('updated', 0)
                note = ""
                if resp.get("quota_limited"):
                    note = f"\n\n已达免费额度上限（100），本次导入 {imported} 个。激活后可无限制处理。"
                QMessageBox.information(
                    self, "导入成功",
                    f"已导入 {imported} 个文件（{updated} 个已更新）{note}")
            else:
                QMessageBox.information(self, "导入完成", "无新文件")
            # 导入后清空选择，恢复可继续导入状态
            self._clear_selection()
        except Exception as e:
            msg = _extract_import_error(e)
            QMessageBox.critical(self, "导入失败", sanitize_for_qt(msg))
            if "免费额度已用完" in msg:
                self._offer_activation()

    def _offer_activation(self):
        """免费额度用尽 → 引导激活（docs/界面设计/07-授权管理.md §4.3）。"""
        if QMessageBox.question(
            self, "免费额度已用完",
            "免费版累计可处理 100 个文件，已用完。\n已有数据不受影响。是否立即激活？",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            from licensing.activation_dialog import ActivationDialog
        except ImportError:
            from src.licensing.activation_dialog import ActivationDialog
        ActivationDialog(self).exec()

    def _clear_selection(self):
        self._selected_folders.clear()

    # ── Import (migrated from desktop_app.py) ────────────────────────

    def import_mixed(self):
        """「导入」单钮（14 号定稿）：NSOpenPanel 文件+文件夹混选。

        选区统一处理：文件夹→递归扫描（跳过抽帧目录）；有媒体扩展名→直接导入；
        无扩展名/陌生扩展名→文件头嗅探补名+收件箱暂存（与照片导入同管线）。
        面板起始目录=上次导入目录（目录记忆）。
        """
        from PySide6.QtCore import QSettings
        try:
            from AppKit import NSOpenPanel, NSURL
        except Exception as e:
            QMessageBox.warning(self, "导入", f"无法打开文件选择面板: {e}")
            return

        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(True)
        panel.setMessage_("选择要导入的视频/照片文件或文件夹（可多选、可混选）")
        last_dir = QSettings().value("ui/last_import_dir", "", type=str)
        if last_dir and os.path.isdir(last_dir):
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(last_dir))

        if panel.runModal() != 1:
            return

        urls = panel.URLs() or []
        paths = [str(u.path()) for u in urls]
        if not paths:
            return
        QSettings().setValue("ui/last_import_dir", str(Path(paths[0]).parent))
        self._import_paths_unified(paths)

    def _import_paths_unified(self, paths):
        """混选选区统一处理：文件夹递归扫描 + 媒体文件直进 + 无扩展名嗅探暂存。"""
        import logging
        logger = logging.getLogger(__name__)
        folders, files = [], []
        for p in paths or []:
            if os.path.isdir(p):
                folders.append(p)
            else:
                files.append(p)

        found = []
        for folder in folders:
            for root, dirs, fs in os.walk(folder):
                # 剪枝：跳过抽帧目录，否则每一帧（.jpg）都会被当照片导入
                dirs[:] = [d for d in dirs if not _is_frames_dir(os.path.join(root, d))]
                for f in fs:
                    if Path(f).suffix.lower() in ALL_MEDIA_EXTENSIONS:
                        found.append(os.path.join(root, f))

        # 无扩展名/陌生扩展名 → 文件头嗅探补名+收件箱暂存（与照片导入同管线）
        direct = [f for f in files if Path(f).suffix.lower() in ALL_MEDIA_EXTENSIONS]
        sniff_candidates = [f for f in files if Path(f).suffix.lower() not in ALL_MEDIA_EXTENSIONS]
        staged = []
        if sniff_candidates:
            staged, errors = self._stage_files_to_inbox(sniff_candidates)
            for err in errors or []:
                logger.warning(f"嗅探暂存失败: {err}")

        all_paths = found + direct + list(staged)
        if not all_paths:
            QMessageBox.warning(self, "无有效文件", "未找到有效的视频或照片文件。")
            return

        logger.info(f"导入混选：文件夹扫描 {len(found)}，直接 {len(direct)}，嗅探暂存 {len(staged)}")
        self._async_import(all_paths)

    def add_files(self):
        """Select video/photo files and import them via POST /task/import."""
        try:
            # Use Native dialog to support external drives and sidebar
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择视频或照片文件",
                "",
                "媒体文件 (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.jpg *.jpeg *.png *.webp *.heic *.heif *.gif *.bmp)"
            )
            if not files:
                return

            # 过滤有效的媒体后缀（视频 + 照片）
            vf = [f for f in files if Path(f).suffix.lower() in ALL_MEDIA_EXTENSIONS]
            if not vf:
                QMessageBox.warning(self, "无有效文件", "未选择有效的视频或照片文件。")
                return

            self._async_import(vf)
        except Exception as e:
            QMessageBox.critical(self, "错误", sanitize_for_qt(f"添加文件失败: {e}"))

    def add_folder(self):
        """Select a folder and import all video/photo files recursively."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("开始选择文件夹...")
            
            # 使用 QFileDialog 选择文件夹
            folder = QFileDialog.getExistingDirectory(
                self, 
                "选择视频文件夹",
                "",
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
            )
            
            if not folder:
                logger.info("用户取消了文件夹选择")
                return
            
            logger.info(f"选择的文件夹: {folder}")

            # 扫描视频 + 照片文件
            logger.info("开始扫描媒体文件...")
            found = []
            for root, dirs, files in os.walk(folder):
                # 剪枝：跳过抽帧目录，否则每一帧（.jpg）都会被当照片导入
                dirs[:] = [d for d in dirs if not _is_frames_dir(os.path.join(root, d))]
                for f in files:
                    if Path(f).suffix.lower() in ALL_MEDIA_EXTENSIONS:
                        found.append(os.path.join(root, f))
            
            logger.info(f"找到 {len(found)} 个媒体文件")
            
            if not found:
                QMessageBox.warning(
                    self, 
                    "无媒体文件",
                    sanitize_for_qt(f"在 '{os.path.basename(folder)}' 中未找到有效视频或照片文件。")
                )
                return
            
            # 开始异步导入
            logger.info(f"准备导入 {len(found)} 个文件...")
            self._async_import(found)
            logger.info("已启动异步导入")
            
        except Exception as e:
            logger.error(f"add_folder 异常: {e}", exc_info=True)
            QMessageBox.critical(
                self, 
                "错误", 
                sanitize_for_qt(f"添加文件夹失败: {e}")
            )

    def scan_and_import(self):
        """Open the granular scan/import dialog for selective file import."""
        dialog = ScanImportDialog(self.api, parent=self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            # Refresh dashboard after successful import
            self.refresh_data()

    def import_from_photos(self):
        """从系统相册导入：标准文件选择框（可预览内容、可多选）。

        照片库 originals/ 的文件多为无扩展名 UUID 名——旧实现 .jpg 过滤器
        导致视频（.mov 等）报扩展名错误且无法导入。现在过滤器覆盖全部媒体
        （另提供"所有文件"兜底），选完后按文件头嗅探真实类型补扩展名，
        复制到收件箱进入常规导入管线。
        """
        photos_dir = self._find_photos_library_dir()
        if not photos_dir:
            # 找不到照片库时回退到「图片」目录
            pictures_dir = str(Path.home() / "Pictures")
            photos_dir = pictures_dir if os.path.isdir(pictures_dir) else str(Path.home())

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "从照片库导入（照片与视频，可多选）",
            photos_dir,
            "媒体文件 (*.jpg *.jpeg *.png *.webp *.heic *.heif *.gif *.bmp *.mp4 *.mov *.m4v *.avi *.mkv);;所有文件 (*)",
        )
        if not files:
            return

        staged, errors = self._stage_files_to_inbox(files)
        if errors:
            tail = "\n…" if len(errors) > 8 else ""
            QMessageBox.warning(self, "部分文件无法导入", "\n".join(errors[:8]) + tail)
        if staged:
            self._async_import([str(p) for p in staged])

    @staticmethod
    def _find_photos_library_dir() -> Optional[str]:
        """查找 macOS 照片库位置（供文件选择框定位起点）。

        照片 App 库通常位于 ~/Pictures/Photos Library.photoslibrary，
        原始素材在其 originals/ 子目录。返回最深的可读目录。
        """
        try:
            pictures = Path.home() / "Pictures"
            if not pictures.is_dir():
                return None

            # 1. 直接搜 ~/Pictures 下的 .photoslibrary
            libs = list(pictures.glob("*.photoslibrary"))
            if libs:
                originals = libs[0] / "originals"
                if originals.is_dir():
                    return str(originals)
                return str(libs[0])

            # 2. 常见名称兜底
            candidates = [
                pictures / "Photos Library.photoslibrary" / "originals",
                pictures / "照片图库.photoslibrary" / "originals",
            ]
            for c in candidates:
                if c.is_dir():
                    return str(c)
        except Exception:
            pass
        return None

    def _stage_files_to_inbox(self, files):
        """复制所选文件到收件箱：无扩展名/错误扩展名按文件头嗅探修正。

        返回 (成功路径列表, 错误字符串列表)。嗅探逻辑在 media_types.sniff_media_ext。
        """
        import logging
        import shutil

        logger = logging.getLogger(__name__)
        try:  # 打包/常规启动：项目根在 path
            from src.core.media_types import ALL_MEDIA_EXTENSIONS, RAW_EXTENSIONS, sniff_media_ext
            from src.core.path_manager import get_base_dir
        except ImportError:  # 脚本直跑
            from core.media_types import ALL_MEDIA_EXTENSIONS, RAW_EXTENSIONS, sniff_media_ext
            from core.path_manager import get_base_dir

        dest = get_base_dir() / "photos_inbox"
        dest.mkdir(parents=True, exist_ok=True)
        known_exts = ALL_MEDIA_EXTENSIONS | RAW_EXTENSIONS
        staged, errors = [], []
        for f in files:
            src = Path(f)
            try:
                ext = src.suffix.lower()
                if ext not in known_exts:
                    sniffed = sniff_media_ext(src)
                    if sniffed:
                        ext = sniffed
                    elif not ext:
                        errors.append(f"{src.name}: 无法识别文件类型")
                        continue
                out = dest / f"{src.stem}{ext}"
                n = 1
                while out.exists():
                    out = dest / f"{src.stem}_{n}{ext}"
                    n += 1
                shutil.copy2(src, out)
                staged.append(out)
            except Exception as exc:
                logger.warning("暂存 %s 失败: %s", src, exc)
                errors.append(f"{src.name}: {exc}")
        return staged, errors

    def _async_import(self, files):
        """Import files in background thread without blocking UI."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"开始异步导入 {len(files)} 个文件...")
        self.import_status_label.setText("导入中...")
        self._import_poll_timer.start()
        self._poll_import_progress()  # 立即拉一次，避免首秒空档

        def do_import():
            try:
                logger.info("发送导入请求到后端...")
                # 增加超时时间到 30 分钟（1800 秒），处理大量文件需要更长时间
                resp = self.api.post("/task/import", json={"files": files}, timeout=1800)
                logger.info(f"后端响应: {resp}")
                imported = resp.get('imported', 0) if isinstance(resp, dict) else 0
                duplicate = resp.get('duplicate', 0) if isinstance(resp, dict) else 0
                # 使用类级别信号确保在主线程调用
                self.import_finished.emit(imported, duplicate)
            except Exception as e:
                logger.error(f"导入失败: {e}", exc_info=True)
                self.import_error.emit(str(e))

        threading.Thread(target=do_import, daemon=True).start()

    def _poll_import_progress(self):
        """QTimer（主线程）→ 丢到后台线程拉进度，避免同步 HTTP 阻塞 UI。

        上一次还没回来就跳过（导入时后端偶有慢响应），避免线程堆积。
        """
        if self._import_poll_busy:
            return
        self._import_poll_busy = True
        threading.Thread(target=self._fetch_import_progress, daemon=True).start()

    def _fetch_import_progress(self):
        """后台线程：拉 /task/import/status，算好文案后经信号回主线程。"""
        try:
            st = self.api.get("/task/import/status", timeout=5)
        except Exception:
            st = None
        finally:
            self._import_poll_busy = False
        if not isinstance(st, dict) or not st.get("active"):
            return
        total = int(st.get("total") or 0)
        processed = int(st.get("processed") or 0)
        imported = int(st.get("imported") or 0)
        duplicate = int(st.get("duplicate") or 0)
        if total:
            text = f"导入中 {processed}/{total}（新增 {imported}，跳过 {duplicate}）..."
        else:
            text = f"导入中（新增 {imported}）..."
        self.import_progress.emit(text)

    def _on_import_progress(self, text):
        """主线程槽：更新进度文案（导入已结束时忽略迟到的那一次）。"""
        if self._import_poll_timer.isActive():
            self.import_status_label.setText(text)
        self.import_status_label.setVisible(bool(text))

    def _on_import_done(self, imported, duplicate):
        """Called on main thread via signal after import completes."""
        self._import_poll_timer.stop()
        if duplicate > 0:
            self.import_status_label.setText(
                f"已导入 {imported} 个文件，跳过 {duplicate} 个重复文件")
        else:
            self.import_status_label.setText(f"已导入 {imported} 个文件")
        self.refresh_data()
    
    def _on_import_error(self, error_msg):
        """Called on main thread via signal when import fails."""
        self._import_poll_timer.stop()
        self.import_status_label.setText(sanitize_for_qt(f"导入失败: {error_msg}"))

    # ── Task actions ────────────────────────────────────────────────

    def _start_task(self, task_id: str):
        try:
            self.api.post(f"/task/{task_id}/start")
            # Let QTimer poll timer handle the update (no blocking _poll_status here)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {e}")

    def _pause_task(self, task_id: str):
        try:
            self.api.post(f"/task/{task_id}/pause")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"暂停失败: {e}")

    def _resume_task(self, task_id: str):
        try:
            self.api.post(f"/task/{task_id}/resume")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"恢复失败: {e}")

    def _delete_task(self, task_id: str):
        if QMessageBox.question(self, "确认", f"删除任务 #{task_id}?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self.api.delete(f"/task/{task_id}")
            self._clear_selection_after_delete()
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _view_task_detail(self, task_id: str):
        """查看任务详情（可滚动对话框，结构化展示）。

        历史 bug：直接把 API 返回 dict str() 塞进 QMessageBox，任务带超长
        analysis_details（数万字符）时弹窗内容爆炸/不可读（如 task 20 详情 3.3 万字符）。
        """
        import json as _json
        try:
            detail = self.api.get(f"/task/{task_id}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取详情失败: {e}")
            return

        lines = []
        lines.append(f"任务 ID: {detail.get('id')}")
        lines.append(f"文件: {detail.get('file_path') or '(无)'}")
        lines.append(f"状态: {detail.get('status') or '未知'}")
        lines.append(f"帧目录: {detail.get('frames_path') or '(无)'}")

        # 人物
        actors = detail.get("actors") or []
        if actors:
            lines.append(f"\n【人物】({len(actors)} 个)")
            for a in actors:
                if isinstance(a, dict):
                    lines.append(
                        f"  - {actor_code_display(a.get('code', ''))} {a.get('name') or ''} "
                        f"({a.get('sample_count', 0)} 张人脸)"
                    )
        else:
            lines.append("\n【人物】无")

        # mapped_tags（JSON 字符串 → 列表）
        mapped_tags = detail.get("mapped_tags")
        try:
            tags = _json.loads(mapped_tags) if isinstance(mapped_tags, str) else (mapped_tags or [])
        except Exception:
            tags = []
        if tags:
            lines.append(f"\n【标签】({len(tags)} 个)")
            lines.append("  " + "、".join(str(t) for t in tags))

        # analysis_details（JSON 字符串 → 结构摘要）
        analysis_details = detail.get("analysis_details")
        if analysis_details:
            lines.append("\n【分析详情】")
            try:
                ad = _json.loads(analysis_details) if isinstance(analysis_details, str) else analysis_details
                if isinstance(ad, dict):
                    md5 = ad.get("md5") or {}
                    if md5:
                        lines.append(f"  MD5: {md5.get('hash', '')} ({md5.get('time', 0)}s)")
                    scenes = ad.get("scenes") or []
                    if scenes:
                        lines.append(f"  场景数: {len(scenes)}")
                        for i, s in enumerate(scenes[:50]):
                            st, en = s.get("start"), s.get("end")
                            lines.append(f"    场景 {i+1}: 帧 {st} - {en}")
                        if len(scenes) > 50:
                            lines.append(f"    ... 共 {len(scenes)} 个场景")
                    frames = ad.get("frames") or []
                    if frames:
                        lines.append(f"  抽帧: {len(frames)} 组")
                    vlm_raw = ad.get("vlm_raw")
                    if vlm_raw:
                        lines.append(f"  VLM 原始输出: {str(vlm_raw)[:500]}")
                    timing = ad.get("timing") or {}
                    if timing:
                        lines.append(f"  耗时: {_json.dumps(timing, ensure_ascii=False)}")
                    # 其他键（不深展开）
                    extra_keys = [k for k in ad.keys()
                                  if k not in ("md5", "scenes", "frames", "vlm_raw", "timing")]
                    if extra_keys:
                        for k in extra_keys:
                            v = ad[k]
                            sv = _json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                            lines.append(f"  {k}: {str(sv)[:300]}")
                else:
                    lines.append(f"  {str(analysis_details)[:500]}")
            except Exception:
                # 非 JSON 时直接展示原始文本（截断到可读长度）
                lines.append(f"  {str(analysis_details)[:3000]}")

        text = "\n".join(lines)

        # 可滚动只读对话框（替代 QMessageBox——超长文本会撑爆弹窗）
        dialog = QDialog(self)
        dialog.setWindowTitle(f"任务 #{task_id} 详情")
        dialog.setMinimumSize(720, 560)
        dialog.resize(820, 640)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(text)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        t = theme()
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Menlo', monospace;
                font-size: 13px;
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 10px;
            }}
        """)
        layout.addWidget(text_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px 20px;
                font-size: {t.get('font_size_md')}px;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dialog.exec()

    # ── Batch actions ───────────────────────────────────────────────

    def _toggle_pin_selected(self):
        """优先按钮：切换所选任务的置顶状态（逐任务翻转）。

        - 选中任务在派发队列中恒排最前（tasks.pinned，跨页有效）
        - 服务端逐任务翻转：已置顶的取消、未置顶的置顶
        - 刷新后置顶任务按排序规则排在列表最前
        """
        ids = self._get_selected_task_ids()
        if not ids:
            return
        try:
            resp = self.api.post("/tasks/pin/toggle", json={"task_ids": ids})
            changed = resp.get("changed", [])
            now_pinned = [c["id"] for c in changed if c.get("pinned")]
            QMessageBox.information(
                self, "优先",
                f"已置顶 {len(now_pinned)} 个任务（取消 {len(changed) - len(now_pinned)} 个）"
                if now_pinned else
                f"已取消 {len(changed)} 个任务的置顶",
            )
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"置顶操作失败: {e}")

    def _batch_start(self):
        """分析选中的任务

        语义(2026-09 用户确认): 「分析」= 激活所选任务的分析。
        - 全部未完成(PENDING/PAUSED/ERROR 等) → 直接激活, 不弹窗
        - 含已完成(COMPLETE) → 弹三按钮: 仅分析未完成 / 全部重跑
          (已完成也重跑, 范围=主流程标签→人脸) / 取消
        """
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            QMessageBox.warning(self, "提示", "请先选择要分析的任务。")
            return

        # 按状态分类（与后端 /task/batch-start 的接受集合对齐）：
        # - 可入队：未启动/已暂停/出错/待处理
        # - 处理中：force=False 时后端会跳过，需选"全部重跑"才会重排
        # - 已完成：默认不重跑（VLM 分析耗时较长）
        # 跨页勾选、不在当前列表里的任务按"可入队"处理（后端会自行判定）。
        _PROCESSING = {"CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "WRITING", "PROCESSING"}
        id_to_status = {t.get("id"): t.get("status") for t in (self._last_task_list or [])}
        startable_ids, processing_ids, completed_ids = [], [], []
        for tid in task_ids:
            st = id_to_status.get(tid)
            if st == "COMPLETE":
                completed_ids.append(tid)
            elif st in _PROCESSING:
                processing_ids.append(tid)
            else:
                startable_ids.append(tid)

        force = False
        if completed_ids or processing_ids:
            lines = [f"选中 {len(task_ids)} 个任务。"]
            if completed_ids:
                lines.append(f"· {len(completed_ids)} 个已完成（默认不重跑）。")
            if processing_ids:
                lines.append(f"· {len(processing_ids)} 个正在处理中（如需重排，请选「全部重跑」）。")
            lines.append(f"· 可直接加入队列：{len(startable_ids)} 个（未启动/已暂停/出错/待处理）。")

            box = QMessageBox(self)
            box.setWindowTitle("分析")
            box.setText("\n".join(lines))
            btn_skip = box.addButton("仅分析未完成", QMessageBox.ButtonRole.AcceptRole)
            btn_force = box.addButton("全部重跑", QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            if not startable_ids:
                btn_skip.setEnabled(False)
            box.exec()
            clicked = box.clickedButton()
            if clicked == btn_cancel or clicked is None:
                return
            force = (clicked == btn_force)
        else:
            startable_ids = task_ids

        if not startable_ids and not force:
            QMessageBox.information(
                self, "分析",
                "没有可直接加入队列的任务。\n处理中/已完成的任务请选择「全部重跑」。",
            )
            return

        try:
            send_ids = task_ids if force else startable_ids
            resp = self.api.post(
                "/task/batch-start",
                json={"task_ids": send_ids, "force": force},
            )
            queued = resp.get("queued", 0)
            skipped = max(0, len(send_ids) - queued)
            msg = f"已将 {queued} 个任务加入{'重新' if force else ''}分析队列。"
            if skipped:
                msg += f"\n（{skipped} 个任务当前状态无需/不能重排，已跳过）"
            QMessageBox.information(self, "分析", msg)
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析失败: {e}")

    def _batch_unqueue(self):
        """取消排队：所选 QUEUED 任务退回「未启动」，可一键取消全部排队。

        「分析」的反向操作：QUEUED → PENDING，启动应用不再自动恢复；
        运行中任务不受影响（中止运行走「暂停」）。后端只翻真正的
        QUEUED，跨页勾选/状态已变的 id 原样上送即可。
        """
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            QMessageBox.warning(self, "提示", "请先选择任务。")
            return

        # 全库排队数（「取消全部」按钮文案；调度器状态本页轮询已在用同一接口）
        total_queued = 0
        try:
            total_queued = int(self.api.get("/scheduler/status").get("queued") or 0)
        except Exception:
            pass

        # 所选中已在排队的数量（仅当前页可见任务可判；跨页的交后端判定）
        id_to_status = {t.get("id"): t.get("status") for t in (self._last_task_list or [])}
        queued_n = sum(1 for tid in task_ids if id_to_status.get(tid) == "QUEUED")

        if not queued_n and not total_queued:
            QMessageBox.information(self, "取消排队", "当前没有排队中的任务。")
            return

        box = QMessageBox(self)
        box.setWindowTitle("取消排队")
        box.setText(
            f"选中 {len(task_ids)} 个任务，其中 {queued_n} 个在排队中。\n"
            "取消后排回「未启动」，启动应用不再自动分析；不影响正在运行的任务。"
        )
        btn_selected = None
        if queued_n:
            btn_selected = box.addButton(
                f"取消所选排队 ({queued_n})", QMessageBox.ButtonRole.AcceptRole)
        btn_all = None
        if total_queued > 0:
            btn_all = box.addButton(
                f"取消全部排队 ({total_queued})", QMessageBox.ButtonRole.DestructiveRole)
        btn_cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked == btn_cancel:
            return
        if btn_all is not None and clicked == btn_all:
            payload = {"all_queued": True}
        elif btn_selected is not None and clicked == btn_selected:
            payload = {"task_ids": task_ids}
        else:
            return

        try:
            resp = self.api.post("/task/batch-unqueue", json=payload)
            n = resp.get("unqueued", 0)
            QMessageBox.information(
                self, "取消排队", f"已取消 {n} 个排队任务（退回「未启动」）。")
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"取消排队失败: {e}")

    def _batch_delete(self):
        task_ids = self._get_selected_task_ids()
        if not task_ids:
            QMessageBox.warning(self, "警告", "请先选择要删除的任务")
            return
        if QMessageBox.question(self, "确认", f"确定删除选中的 {len(task_ids)} 个任务?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self.api.post("/task/batch-delete", json={"task_ids": task_ids})
            QMessageBox.information(self, "删除", f"已删除 {len(task_ids)} 个任务。")
            self._clear_selection_after_delete()
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _batch_delete_done(self):
        """Delete all completed tasks."""
        if QMessageBox.question(self, "确认", "批量删除所有已完成任务?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            completed_ids = [t["id"] for t in self._last_task_list if t.get("status") == "COMPLETE"]
            if not completed_ids:
                QMessageBox.information(self, "提示", "没有已完成的任务可删除。")
                return
            self.api.post("/task/batch-delete", json={"task_ids": completed_ids})
            QMessageBox.information(self, "删除", f"已删除 {len(completed_ids)} 个已完成任务。")
            self._clear_selection_after_delete()
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")
    
    def _update_scheduler_ui(self, status: str, scheduler_info: dict):
        """更新调度器 UI 状态"""
        running = scheduler_info.get("running", 0)
        queued = scheduler_info.get("queued", 0)
        
        if status == "running":
            self.scheduler_status_label.setText(f"调度 {running}/{scheduler_info.get('max_concurrent', 3)}")
            self.scheduler_status_label.setStyleSheet(f"color: {theme().get('success')}; font-weight: 500;")
        elif status == "paused":
            self.scheduler_status_label.setText(f"调度 已暂停 · 队列 {queued}")
            self.scheduler_status_label.setStyleSheet(f"color: {theme().get('warning')}; font-weight: 500;")
        else:  # stopped
            self.scheduler_status_label.setText("调度 停止")
            self.scheduler_status_label.setStyleSheet(f"color: {theme().get('text_secondary')}; font-weight: 500;")

    def _edit_task_tags(self, task_id):
        """打开编辑标签对话框"""
        try:
            # 获取任务详情
            task = self.api.get(f"/task/{task_id}")
            if not task:
                QMessageBox.warning(self, "警告", "任务不存在")
                return
            
            # 解析当前标签
            import json
            current_tags = []
            if task.get("mapped_tags"):
                try:
                    current_tags = json.loads(task["mapped_tags"])
                except:
                    current_tags = []
            
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"编辑标签 - {task.get('file_name', '未知')}")
            dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # 当前标签列表
            layout.addWidget(QLabel("当前标签:"))
            tag_list = QListWidget()
            for tag in current_tags:
                item = QListWidgetItem(tag)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                tag_list.addItem(item)
            layout.addWidget(tag_list)
            
            # 按钮区域
            btn_layout = QHBoxLayout()
            
            add_btn = QPushButton("添加标签")
            add_btn.clicked.connect(lambda: self._add_tag_to_list(tag_list))
            btn_layout.addWidget(add_btn)
            
            remove_btn = QPushButton("删除标签")
            remove_btn.clicked.connect(lambda: self._remove_tag_from_list(tag_list))
            btn_layout.addWidget(remove_btn)
            
            btn_layout.addStretch()
            layout.addLayout(btn_layout)
            
            # 确定/取消按钮
            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            button_box.accepted.connect(lambda: self._save_task_tags(task_id, tag_list, dialog))
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取任务详情失败: {e}")
    
    def _add_tag_to_list(self, tag_list):
        """添加新标签到列表"""
        from PySide6.QtWidgets import QInputDialog
        tag, ok = QInputDialog.getText(self, "添加标签", "请输入标签名称:")
        if ok and tag.strip():
            tag_list.addItem(tag.strip())
    
    def _remove_tag_from_list(self, tag_list):
        """从列表删除选中的标签"""
        current_item = tag_list.currentItem()
        if current_item:
            tag_list.takeItem(tag_list.row(current_item))
    
    def _save_task_tags(self, task_id, tag_list, dialog):
        """保存修改后的标签"""
        try:
            # 收集所有标签
            tags = []
            for i in range(tag_list.count()):
                item = tag_list.item(i)
                if item and item.text().strip():
                    tags.append(item.text().strip())
            
            # 调用 API 更新
            self.api.patch(f"/task/{task_id}/tags", json={"tags": tags})
            
            QMessageBox.information(self, "成功", "标签已更新")
            dialog.accept()
            
            # 刷新列表
            self.refresh_data()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存标签失败: {e}")
    
    def _force_reanalyze_task(self, task_id):
        """强制重新分析单个任务"""
        if QMessageBox.question(
            self, "确认强制重新分析",
            f"确定要强制重新分析任务 #{task_id} 吗？\n\n"
            "这将清除已有的分析结果并重新开始分析。",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        
        try:
            self.api.post(f"/task/{task_id}/force-reanalyze")
            QMessageBox.information(self, "成功", "任务已加入强制重新分析队列")
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"强制重新分析失败: {e}")
    
    def _batch_rebuild_faces(self):
        """批量识别人脸（带进度轮询）"""
        task_ids = self._get_selected_task_ids()
        
        if not task_ids:
            QMessageBox.warning(self, "警告", "请先选择要识别人脸的任务")
            return

        # 统计选中任务的识别状态，提示已识别/无人脸/跳过的（避免重复触发）
        task_list = getattr(self, '_last_task_list', [])
        id_to_scan = {t.get("id"): t.get("face_scan_status") for t in task_list}
        already_scanned = []
        for tid in task_ids:
            scan = id_to_scan.get(tid)
            if scan in ("complete", "none", "skipped"):
                already_scanned.append(tid)

        # 过滤掉已识别的，只保留未扫描的
        scan_set = set(already_scanned)
        pending_ids = [t for t in task_ids if t not in scan_set]

        if already_scanned:
            # 有已识别任务 → 三键对话框
            msg = (
                f"选中的 {len(task_ids)} 个任务中，"
                f"{len(already_scanned)} 个已有识别结果：\n"
                + "、".join(str(t) for t in already_scanned[:8])
                + ("..." if len(already_scanned) > 8 else "")
                + f"\n\n未识别的有 {len(pending_ids)} 个。\n"
            )
            box = QMessageBox(self)
            box.setWindowTitle("确认识别人脸")
            box.setText(msg)
            box.setIcon(QMessageBox.Icon.Warning)
            btn_skip = box.addButton("仅识别未扫描的", QMessageBox.ButtonRole.AcceptRole)
            if pending_ids:
                btn_skip.setEnabled(True)
            else:
                btn_skip.setEnabled(False)  # 全部已识别，不能跳过
            box.addButton("全部重新识别", QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()

            if box.clickedButton() == btn_cancel:
                return
            if box.clickedButton() == btn_skip and pending_ids:
                task_ids = pending_ids
            # btn_all → 保持全部 task_ids

            if not task_ids:
                QMessageBox.information(self, "提示", "所选任务均已有识别结果，无需重新识别")
                return
        else:
            # 无已识别任务 → 普通确认
            if QMessageBox.question(
                self, "确认识别人脸",
                f"确定要对选中的 {len(task_ids)} 个任务识别人脸吗？\n\n"
                "这将检测视频中的人脸并创建人物记录。",
                QMessageBox.Yes | QMessageBox.No
            ) != QMessageBox.Yes:
                return

        # 禁用按钮，显示进度；初始化并发队列状态
        # 识别人脸按钮已随业务域精简移除（人脸页承载）
        self.btn_rebuild_faces.setText("识别中...")
        self._rebuild_task_ids = list(task_ids)
        self._rebuild_pending = deque(task_ids)   # 待处理
        self._rebuild_active = set()              # 运行中
        self._rebuild_failed = []                 # 失败的任务
        self._rebuild_done = 0
        # 并发数: 优先读后端 config face_recognition.max_concurrent;
        # 0/未配置 → min(4, CPU核数)。无法读到时默认 4。
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
        logger.info("批量识别人脸: %d 个任务, 并发 %d", len(task_ids), self._rebuild_concurrency)
        self._rebuild_pump()

    def _rebuild_pump(self):
        """并发调度：active 未满则从 pending 补位发起任务。"""
        # 先清理 active 中已不存在的(防御)
        while len(self._rebuild_active) < self._rebuild_concurrency and self._rebuild_pending:
            task_id = self._rebuild_pending.popleft()
            self._rebuild_active.add(task_id)
            self._rebuild_start_one(task_id)

        # 更新进度文案
        total = len(self._rebuild_task_ids)
        done = self._rebuild_done
        running = len(self._rebuild_active)
        self.btn_rebuild_faces.setText(
            f"识别中 {done + running}/{total} (进行{running})"
        )

        # 全部结束(无 pending、无 active)
        if not self._rebuild_pending and not self._rebuild_active:
            self.btn_rebuild_faces.setEnabled(True)
            self.btn_rebuild_faces.setText("识别人脸")
            if self._rebuild_failed:
                QMessageBox.warning(
                    self, "完成(部分失败)",
                    f"已完成 {self._rebuild_done}/{total} 个任务的人脸识别，"
                    f"{len(self._rebuild_failed)} 个失败: {self._rebuild_failed[:5]}"
                )
            else:
                QMessageBox.information(
                    self, "完成",
                    f"已完成 {total} 个任务的人脸识别"
                )
            self.refresh_data()

    def _rebuild_start_one(self, task_id: int):
        """发起单个任务的人脸重建，启动轮询链。"""
        try:
            resp = self.api.post(f"/tasks/{task_id}/rebuild-faces")
            st = resp.get("status", "unknown")
            if st in ("completed", "failed"):
                self._rebuild_finish_one(task_id, st, resp.get("message", ""))
            else:
                # running / 排队中 → 进入轮询
                QTimer.singleShot(1000, lambda: self._rebuild_poll_one(task_id))
        except Exception as e:
            self._rebuild_finish_one(task_id, "failed", str(e))

    def _rebuild_poll_one(self, task_id: int):
        """轮询单个任务进度(每任务独立 1s 轮询链)。"""
        if task_id not in self._rebuild_active:
            return  # 已被收尾(如用户中断)，丢弃
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
                # unknown/idle 等异常态 → 视为失败收尾，避免卡死队列
                self._rebuild_finish_one(task_id, "failed", f"异常状态: {status}")
        except Exception as e:
            self._rebuild_finish_one(task_id, "failed", str(e))

    def _rebuild_finish_one(self, task_id: int, status: str, msg: str = ""):
        """单个任务结束：记录结果并从 active 移除，随后补位下一个。"""
        if task_id not in self._rebuild_active:
            return
        self._rebuild_active.discard(task_id)
        if status == "completed":
            self._rebuild_done += 1
        else:
            self._rebuild_failed.append(task_id)
            self._rebuild_done += 1
            logger.warning("任务 %d 人脸重建失败: %s", task_id, msg or status)
        # 补位下一个（可能触发全部完成收尾）
        self._rebuild_pump()
    
    def _get_selected_task_ids(self):
        """获取选中的任务ID列表(全局勾选集合, 跨页有效)。"""
        checked = getattr(self, '_checked_ids', set())
        return sorted(int(t) for t in checked if str(t).isdigit())

    def _restore_refresh_btn(self):
        """刷新完成后恢复按钮状态。"""
        try:
            if hasattr(self, 'refresh_api_btn'):
                self.refresh_api_btn.setText("刷新")
                self.refresh_api_btn.setEnabled(True)
        except RuntimeError:
            pass

    def _clear_selection_after_delete(self):
        """删除任务后清空选中状态，避免顺延任务被误选中。

        清空 _checked_ids 集合 + 重置表格复选框 + 删除去重签名强制重建。
        """
        if hasattr(self, '_checked_ids'):
            self._checked_ids = set()
        # 重置所有行复选框
        for i in range(self.task_table.rowCount()):
            cb_item = self.task_table.item(i, 0)
            if cb_item:
                cb_item.setCheckState(Qt.CheckState.Unchecked)
        # 强制表格重建（跳过 Dedup 签名）
        if hasattr(self, '_last_task_signature'):
            delattr(self, '_last_task_signature')
        # 同步表头复选框
        self._sync_header_from_rows()
        self._update_selection_count()

    # ── Filters & refresh ───────────────────────────────────────────

    def refresh_data(self):
        """Immediately refresh dashboard data (stop timer → poll → restart timer)."""
        self.poll_timer.stop()
        self._poll_status()
        self.poll_timer.start()

    def _apply_quick_filter(self, key: str):
        """统计条点击 → 快捷筛选（与筛选下拉同一套条件，双向保持同步）。

        key: all / pending / processing / complete ...；点「总任务」= 不过滤。
        """
        values = quick_filter_values(key)
        combo = getattr(self, "filter_combo", None)
        if combo is not None:
            combo.set_checked_values(values)     # 同步下拉的勾选与摘要文字
        self._on_status_filter_changed(values)   # 走同一条处理路径(归一/防抖/刷新)

    def _sync_stat_cards(self):
        """按当前筛选高亮对应的统计卡片；多选或无匹配时不点亮任何卡片。"""
        single = active_quick_filter_key(getattr(self, "_current_filter", "all"))
        for card in (getattr(self, "stat_total", None), getattr(self, "stat_completed", None),
                     getattr(self, "stat_processing", None), getattr(self, "stat_pending", None),
                     getattr(self, "stat_unqueued", None)):
            if card is None:
                continue
            try:
                card.set_active(getattr(card, "_filter_key", "") == single)
            except Exception:
                pass

    def _on_status_filter_changed(self, values: list):
        """状态筛选(可多选)变化 → 回到第 1 页, 防抖后重新拉取。

        空选或全选都归一为"不过滤"(all) —— 见 normalize_status_filter。

        刷新用 250ms 防抖: 连续勾选多项时只在停下来后刷新一次。若每次勾选都
        立刻同步刷新, 会与紧随其后的点击互相干扰(丢勾选, 手感不稳定)。
        """
        vals = normalize_status_filter(values)
        self._current_filter = ",".join(vals) if vals else "all"
        self._current_page = 1
        # 强制表格重建(签名去重会跳过"看起来相同"的数据)
        if hasattr(self, '_last_task_signature'):
            delattr(self, '_last_task_signature')
        self._sync_stat_cards()
        timer = getattr(self, "_filter_refresh_timer", None)
        if timer is not None:
            timer.start()          # 已在跑则重新计时(合并连续操作)
        else:
            self.refresh_data()


    def _on_face_filter_changed(self, index: int):
        """人脸状态筛选下拉变化: 叠加在状态筛选之上(AND)。"""
        self._current_face_filter = ["all", "complete", "none", "skipped", "unscanned"][index]
        self._current_page = 1
        if hasattr(self, '_last_task_signature'):
            delattr(self, '_last_task_signature')
        self.refresh_data()

    def _on_refresh_api(self):
        """刷新按钮：强制全表重建并重新拉取任务列表和统计。"""
        try:
            # 清除去重签名 + 强制全量重建（即使数据未变也刷新显示）
            if hasattr(self, '_last_task_signature'):
                delattr(self, '_last_task_signature')
            self._force_full_rebuild = True
            # 按钮反馈
            self.refresh_api_btn.setText("刷新中...")
            self.refresh_api_btn.setEnabled(False)
            self.refresh_data()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("刷新失败: %s", e)
            self.refresh_api_btn.setText("刷新")
            self.refresh_api_btn.setEnabled(True)

    # ── Pagination ──────────────────────────────────────────────────

    def _refresh_page(self):
        """翻页刷新: 强制全量重建表格(增量路径原位复用 item + 后设 checkState
        在本表上不可靠——checkbox 状态设了但界面不显示; 全量重建新建 item 可靠)。"""
        self._force_full_rebuild = True
        self.refresh_data()

    def _on_pager_size_changed(self, size: int) -> None:
        """每页条数变化（PaginationWidget 信号，int）。"""
        self._on_page_size_changed(str(size))

    def _on_pager_page_changed(self, page: int) -> None:
        """页码切换（PaginationWidget 信号）。"""
        if 1 <= page <= self._total_pages and page != self._current_page:
            self._current_page = page
            self._refresh_page()

    def _update_pagination_ui(self):
        """分页状态回填唯一形态组件（计算/使能/页码都在组件内）。"""
        self.pagination_bar.update_pagination(
            self._current_page, self._total_items, self._page_size)

    # ── Cleanup ─────────────────────────────────────────────────────

    def __del__(self):
        try:
            self.poll_timer.stop()
            self._executor.shutdown(wait=False)
        except Exception:
            pass


# Import Path here for _scan_folder
