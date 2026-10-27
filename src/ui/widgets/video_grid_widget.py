"""VideoGridWidget — 视频宫格展示组件。

可复用的视频网格组件，支持分页、复选框选择、点击播放、显示封面与标题。
被 TagBrowsePage 和 ActorDetailDialog 复用。
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFontMetricsF
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu,
    QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from ..components.api_client import APIClient
from ..components.theme import theme
from ..tokens import T


# ── 异步数据加载 ──────────────────────────────────────────────────────────
# 通用加载器已抽到 widgets/async_loader.py 共用（原来这里有一份私有副本）。
from .async_loader import AsyncLoader as _AsyncLoader

# 满屏容量的业务下限：布局未落位时的瞬时视口会算出极小值并被锁死
# （实测首次进页 9/页、918 页），低于下限的一律钳回（2026-09-30 用户令）
MIN_FILL_PAGE_SIZE = 20


# ── 宫格几何单点 ──────────────────────────────────────────────────────────
# 列宽模型唯一来源：auto_page_size（算容量拉多少条）与 _relayout（实际
# 排几列）必须同口径——此前两套模型（200px 列 vs 160px 列+8 列帽）各自
# 为政，实排行数少于容量计划行数，第一屏底部大片留白（2026-09-30 用户实测）。
# 常量与 _setup_ui 的 grid_layout margins/spacing 同源。

GRID_MARGIN = 8      # grid_layout contentsMargins
GRID_SPACING = 12    # grid_layout spacing
GRID_COL_MIN = 190   # 列宽下限（= T.GRID_CARD_MIN_W）


def grid_columns(viewport_w: int):
    """视口宽 → (列数, 卡宽)。容量计算与重排共用。"""
    cols = max(1, (viewport_w - 2 * GRID_MARGIN + GRID_SPACING)
               // (GRID_COL_MIN + GRID_SPACING))
    card_w = max(GRID_COL_MIN, (viewport_w - 2 * GRID_MARGIN
                                - GRID_SPACING * (cols - 1)) // cols)
    return cols, card_w


def grid_card_height(card_w: int) -> int:
    """卡高 = 封面 16:9 + 标题行 + 起播行（与 VideoCard 实际结构一致）。"""
    return card_w * 9 // 16 + 66  # ui-lint:allow 16:9 宽高比


# ── 视频卡片 ──────────────────────────────────────────────────────────────

class VideoCard(QFrame):
    """封面卡（19§D4.8a + 21§TB4.2/4.3）：16:9 封面+时长角标+文件名
    单行省略+「从封面帧 hh:mm:ss 起播」行；hover 显示播放钮，
    点击卡片=play_requested（配置播放器时间轴跳转）。"""

    play_requested = Signal(str, int)
    checked_changed = Signal(str, bool)
    delete_requested = Signal(dict)

    def __init__(self, video: Dict, card_width: int = 232,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.video = video
        self.video_id: str = str(video.get("id", ""))
        self.file_path: str = video.get("file_path", "")
        self.frame_number: int = (video.get("video_frame_number")
                                  or video.get("frame_number", 0))
        self._card_w = max(T.GRID_CARD_MIN_W, int(card_width))
        self._cover_pm = None  # 未缩放原图（卡宽变化时按新尺寸重缩放）
        self.setObjectName("VideoCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(self._card_w)
        self._setup_ui()

    def _setup_ui(self):
        t = theme()
        cover_h = self._card_w * 9 // 16  # ui-lint:allow 16:9 宽高比
        self.setStyleSheet(f"""
            QFrame#VideoCard {{
                background: {t.get('bg_primary')};
                border-radius: {t.get('radius_card')}px;
                border: 1px solid transparent;
            }}
            QFrame#VideoCard:hover {{
                border: 1px solid {t.get('accent')}33;
            }}
        """)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── 封面 16:9 ──
        self.cover_container = QFrame()
        self.cover_container.setObjectName("CoverContainer")
        self.cover_container.setFixedSize(self._card_w - 2, cover_h)  # ui-lint:allow -2 边框补偿
        self.cover_container.setStyleSheet(f"""
            QFrame#CoverContainer {{
                background: {t.get('bg_secondary')};
                border-top-left-radius: {t.get('radius_card')}px;
                border-top-right-radius: {t.get('radius_card')}px;
            }}
        """)
        cover_layout = QVBoxLayout(self.cover_container)
        cover_layout.setContentsMargins(0, 0, 0, 0)
        self.cover_label = QLabel()
        self.cover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_label.setFixedSize(self._card_w - 2, cover_h)  # ui-lint:allow -2 边框补偿
        self.cover_label.setStyleSheet(
            f"color: {t.get('text_tertiary')};"
            f" font-size: {t.get('dur_badge_fs')}px; background: transparent;")
        self._load_cover()
        cover_layout.addWidget(self.cover_label)

        # 时长角标（右下）
        duration = self.video.get("duration") or 0
        try:
            dur = int(duration)
        except (TypeError, ValueError):
            dur = 0
        if dur > 0:
            self._badge = QLabel(f"{dur // 60}:{dur % 60:02d}", self.cover_container)
            self._badge.setStyleSheet(
                f"QLabel {{ font-size: {t.get('dur_badge_fs')}px; color: #FFFFFF;"
                f" background: rgba(0,0,0,0.6);"
                f" border-radius: {t.get('dur_badge_r')}px;"
                f" padding: 1px {t.get('dur_badge_pad_x')}px; }}")
            self._badge.adjustSize()
            self._badge.move(self.cover_container.width() - self._badge.width() - T.DUR_BADGE_CORNER,
                             self.cover_container.height() - self._badge.height() - T.DUR_BADGE_CORNER)
            self._badge.show()
            self._badge.raise_()

        # TB4.3 hover 播放钮：40px accent 圆+白三角，hover 显形
        self.play_btn = QLabel("▶", self.cover_container)
        self.play_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_btn.setFixedSize(40, 40)  # ui-lint:allow 圆钮直径（TB4.3 定稿）
        self.play_btn.setStyleSheet(
            f"QLabel {{ color: #FFFFFF; font-size: 16px;"
            f" background: {t.get('accent')}; border-radius: 20px; }}")  # ui-lint:allow 圆角=半径
        self.play_btn.move((self.cover_container.width() - 40) // 2,
                           (self.cover_container.height() - 40) // 2)
        self.play_btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.play_btn.hide()
        self.cover_container.setStyleSheet(self.cover_container.styleSheet() + """
            QFrame#CoverContainer:hover ~ #PlayHover { }
        """)

        main.addWidget(self.cover_container)

        # ── 文字区 ──
        meta = QWidget()
        meta.setStyleSheet("background: transparent;")
        meta_v = QVBoxLayout(meta)
        meta_v.setContentsMargins(T.CARD_META_PAD, T.CARD_META_PAD_TOP,
                                  T.CARD_META_PAD, T.CARD_META_PAD)
        meta_v.setSpacing(T.CARD_META_LINE_GAP)

        title_text = self.video.get("title") or Path(self.file_path).stem or "未知标题"
        self.title_text = title_text
        self.title_label = QLabel(title_text)
        self.title_label.setStyleSheet(
            f"color: {t.get('text_primary')};"
            f" font-size: {t.get('font_small')}px; background: transparent;")
        _tw = self._card_w - 2 * T.CARD_META_PAD
        self.title_label.setFixedWidth(_tw)
        _fm = QFontMetricsF(self.title_label.font())
        self.title_label.setText(_fm.elidedText(
            title_text, Qt.TextElideMode.ElideRight, _tw))
        meta_v.addWidget(self.title_label)

        # TB4.2 第二行：从封面帧 hh:mm:ss 起播
        start_label = QLabel(f"从封面帧 {self._fmt_frame_ts(self.frame_number, dur)} 起播")
        start_label.setStyleSheet(
            f"color: {t.get('text_secondary')};"
            f" font-size: {t.get('font_hint')}px; background: transparent;")
        meta_v.addWidget(start_label)

        main.addWidget(meta)

        # 勾选框悬浮封面左上（保留多选能力，标签浏览默认隐藏由页面控制）
        self.checkbox = QCheckBox(self.cover_container)
        self.checkbox.setGeometry(T.CHECKBOX_CORNER, T.CHECKBOX_CORNER,
                                  T.CHECKBOX_OVERLAY, T.CHECKBOX_OVERLAY)
        self.checkbox.setStyleSheet("QCheckBox { background: transparent; }")
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        self.checkbox.raise_()

    @staticmethod
    def _fmt_frame_ts(frame_number: int, duration: int) -> str:
        """帧号→hh:mm:ss（无帧/时长时显示 00:00:00）。"""
        if frame_number > 0 and duration > 0:
            sec = frame_number % max(1, duration)  # 帧在时长内取模（无 fps 信息时的近似）
        else:
            sec = 0
        return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"

    def enterEvent(self, event):  # noqa: N802 - Qt 命名
        self.play_btn.show()
        self.play_btn.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802 - Qt 命名
        self.play_btn.hide()
        super().leaveEvent(event)

    def set_card_width(self, width: int) -> None:
        """重排统一改卡宽：封面/角标/播放钮/标题全部跟随。

        旧实现只 setFixedWidth 卡框——内部定宽停留构造尺寸，卡宽随窗口
        变化时封面被裁或留缝。封面位图按新尺寸重缩放（Smooth）。"""
        width = max(T.GRID_CARD_MIN_W, int(width))
        if width == self._card_w:
            return
        self._card_w = width
        self.setFixedWidth(width)
        cover_w = width - 2  # ui-lint:allow -2 边框补偿
        cover_h = width * 9 // 16  # ui-lint:allow 16:9 宽高比
        self.cover_container.setFixedSize(cover_w, cover_h)
        self.cover_label.setFixedSize(cover_w, cover_h)
        if getattr(self, "_badge", None) is not None:
            self._badge.move(cover_w - self._badge.width() - T.DUR_BADGE_CORNER,
                             cover_h - self._badge.height() - T.DUR_BADGE_CORNER)
        self.play_btn.move((cover_w - self.play_btn.width()) // 2,
                           (cover_h - self.play_btn.height()) // 2)
        title_w = width - 2 * T.CARD_META_PAD
        self.title_label.setFixedWidth(title_w)
        _fm = QFontMetricsF(self.title_label.font())
        self.title_label.setText(_fm.elidedText(
            self.title_text, Qt.TextElideMode.ElideRight, title_w))
        self._apply_cover_scale()

    def _load_cover(self):
        """封面：cover_path 优先；OpenCV 读（Qt 无 JPEG 插件环境）。"""
        cover_path = self.video.get("cover_path", "")
        pixmap = None
        if cover_path and Path(cover_path).exists():
            try:
                import cv2
                img_bgr = cv2.imread(str(cover_path))
                if img_bgr is not None and cv2.cvtColor(
                        img_bgr, cv2.COLOR_BGR2GRAY).mean() >= 10:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h, w2, ch = img_rgb.shape
                    q_img = QImage(img_rgb.data.tobytes(), w2, h, ch * w2,
                                   QImage.Format.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(q_img)
            except Exception:
                pixmap = None
            if pixmap is None:
                pm = QPixmap(cover_path)
                if not pm.isNull():
                    pixmap = pm
        self._cover_pm = pixmap
        self._apply_cover_scale()

    def _apply_cover_scale(self):
        """按当前封面标签尺寸缩放已加载位图（加载/改卡宽共用单点）。"""
        if self._cover_pm is not None and not self._cover_pm.isNull():
            self.cover_label.setPixmap(self._cover_pm.scaled(
                self.cover_label.width(), self.cover_label.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation))
        else:
            self.cover_label.setText("暂无封面")

    def mousePressEvent(self, event):
        """点击卡片=播放（TB4.3）。"""
        if self.file_path:
            self.play_requested.emit(self.file_path, self.frame_number)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """右键菜单：在文件夹中显示 / 删除。"""
        menu = QMenu(self)
        show_action = menu.addAction("📁 在文件夹中显示")
        menu.addSeparator()
        delete_action = menu.addAction("🗑 删除")
        chosen = menu.exec(event.globalPos())
        if chosen == show_action:
            self._show_in_finder()
        elif chosen == delete_action:
            self.delete_requested.emit(self.video)

    def _show_in_finder(self):
        if not self.file_path:
            return
        file_path = Path(self.file_path)
        if file_path.exists():
            try:
                subprocess.run(["open", "-R", str(file_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"打开 Finder 失败: {e}", flush=True)

    def _on_checkbox_changed(self, state):
        self.checked_changed.emit(self.video_id,
                                  state == Qt.CheckState.Checked.value)


# ── 分页组件 ──────────────────────────────────────────────────────────────

class PaginationWidget(QWidget):
    """分页控件，支持首页、末页、页码跳转、每页条数设置。"""

    page_changed = Signal(int)  # 切换页面，传递新的页码
    page_size_changed = Signal(int)  # 每页条数变化，传递新的 page_size

    def __init__(self, parent: Optional[QWidget] = None, default_page_size: int = 20,
                 compact: bool = True, page_size_options: Optional[List[str]] = None):
        """全 app 唯一分页形态（2026-09-29 用户裁定）：高 30 底栏单行、
        每页条数 + 首页/上一页/页码/下一页/末页/跳转页码/共 N——全 app
        统一形态（2026-10-07 用户要求恢复跳转）。
        绝不折行（QHBoxLayout 单行 + 固定高）。compact 参数保留兼容旧调用，
        逻辑上已无差别。"""
        super().__init__(parent)
        self.current_page: int = 1
        self.total_pages: int = 1
        self.total_count: int = 0
        self.page_size: int = default_page_size
        self._compact = True  # 单一形态；compact 参数仅兼容签名

        t = theme()

        layout = QHBoxLayout(self)
        if compact:
            layout.setContentsMargins(0, 0, 0, 0)
            self.setFixedHeight(T.PAGER_H)
        else:
            layout.setContentsMargins(8, 8, 8, 8)  # ui-lint:allow 遗留债（迁移批清零）
        layout.setSpacing(8)  # ui-lint:allow 遗留债（迁移批清零）

        # 每页条数选择
        self.page_size_label = QLabel("每页")
        self.page_size_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(page_size_options or ["10", "20", "50", "100"])
        self.page_size_combo.setCurrentText(str(default_page_size))
        self.page_size_combo.setFixedWidth(60)  # ui-lint:allow 遗留债（迁移批清零）
        self.page_size_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t.get('bg_card')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: 4px;  # ui-lint:allow 遗留债（迁移批清零）
                padding: 2px 6px;  # ui-lint:allow 遗留债（迁移批清零）
                font-size: {t.get('font_size_sm')}px;
            }}
        """)
        self.page_size_combo.currentIndexChanged.connect(self._on_page_size_changed)
        self.page_size_suffix = QLabel("条")
        self.page_size_suffix.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")

        # 首页按钮
        self.first_btn = QToolButton()
        self.first_btn.setText("首页")
        self.first_btn.setAutoRaise(True)
        self.first_btn.clicked.connect(lambda: self._go_to(1))
        self.first_btn.setStyleSheet(f"color: {t.get('accent')}; font-size: {t.get('font_size_sm')}px;")

        # 上一页按钮
        self.prev_btn = QToolButton()
        self.prev_btn.setText("上一页")
        self.prev_btn.setAutoRaise(True)
        self.prev_btn.clicked.connect(lambda: self._go_to(self.current_page - 1))
        self.prev_btn.setStyleSheet(f"color: {t.get('accent')}; font-size: {t.get('font_size_sm')}px;")

        # 页码显示
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")

        # 下一页按钮
        self.next_btn = QToolButton()
        self.next_btn.setText("下一页")
        self.next_btn.setAutoRaise(True)
        self.next_btn.clicked.connect(lambda: self._go_to(self.current_page + 1))
        self.next_btn.setStyleSheet(f"color: {t.get('accent')}; font-size: {t.get('font_size_sm')}px;")

        # 末页按钮
        self.last_btn = QToolButton()
        self.last_btn.setText("末页")
        self.last_btn.setAutoRaise(True)
        self.last_btn.clicked.connect(lambda: self._go_to(self.total_pages))
        self.last_btn.setStyleSheet(f"color: {t.get('accent')}; font-size: {t.get('font_size_sm')}px;")

        # 页码跳转
        self.jump_label = QLabel("跳至")
        self.jump_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")

        self.page_jump_input = QSpinBox()
        self.page_jump_input.setMinimum(1)
        self.page_jump_input.setMaximum(1)
        self.page_jump_input.setFixedWidth(60)  # ui-lint:allow 遗留债（迁移批清零）
        self.page_jump_input.setButtonSymbols(QSpinBox.NoButtons)
        self.page_jump_input.setStyleSheet(f"""
            QSpinBox {{
                background: {t.get('bg_card')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: 4px;  # ui-lint:allow 遗留债（迁移批清零）
                padding: 2px 6px;  # ui-lint:allow 遗留债（迁移批清零）
                font-size: {t.get('font_size_sm')}px;
            }}
            QSpinBox:focus {{
                border: 1px solid {t.get('accent')};
            }}
        """)
        # 回车跳转
        self.page_jump_input.lineEdit().returnPressed.connect(self._on_jump)
        # 值改变也跳转（点击上下箭头时）
        self.page_jump_input.valueChanged.connect(self._on_jump_value_changed)

        self.page_suffix_label = QLabel("页")
        self.page_suffix_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: {t.get('font_size_sm')}px;")

        # 总数显示
        self.count_label = QLabel()
        self.count_label.setStyleSheet(f"color: {t.get('text_tertiary')}; font-size: {t.get('font_size_sm')}px;")

        layout.addStretch()
        layout.addWidget(self.page_size_label)
        layout.addWidget(self.page_size_combo)
        layout.addWidget(self.page_size_suffix)
        layout.addWidget(self.first_btn)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.page_label)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.last_btn)
        layout.addWidget(self.jump_label)
        layout.addWidget(self.page_jump_input)
        layout.addWidget(self.page_suffix_label)
        layout.addWidget(self.count_label)
        layout.addStretch()

        # 18§4.17 单一形态于 2026-10-07 用户裁定推翻：首页/末页恢复
        # （256 页场景连点"下一页"不可用），跳转页码输入随后恢复
        # （用户要求"制定页面跳转"）；统计仍隐藏。
        # 所有可见控件钳 30 高 + 单行布局 → 结构性不可能折行。
        for _w in (self.first_btn, self.last_btn, self.page_size_combo,
                   self.prev_btn, self.next_btn, self.page_jump_input,
                   self.page_size_label, self.page_size_suffix, self.count_label):
            _w.setFixedHeight(T.PAGER_BTN_H)
        self.page_label.setFixedHeight(T.PAGER_BTN_H)
        self.page_label.setStyleSheet(
            f"color: {t.get('text_primary')};"
            f" font-size: {t.get('font_secondary')}px; font-weight: 600;")
        self._apply_style()

        self._update_ui()

    def _apply_style(self):
        """bar 底色与控件色（构造/换肤共用单点）。

        底色=bg_card 与所在面板同色；内嵌 QLabel 全透明——透出 bar 底，
        任何主题下 QLabel 背景都与页面一致（2026-09-29 用户要求）。"""
        t = theme()
        self.setStyleSheet(
            f"PaginationWidget {{ background-color: {t.get('bg_card')};"
            f" border-top: 1px solid {t.get('border')}; }}")
        for _w in (self.page_size_label, self.page_size_suffix, self.count_label):
            _w.setStyleSheet(
                f"color: {t.get('text_secondary')};"
                f" font-size: {t.get('font_secondary')}px; background: transparent;")
        self.page_label.setStyleSheet(
            f"color: {t.get('text_primary')};"
            f" font-size: {t.get('font_secondary')}px; font-weight: 600;"
            f" background: transparent;")
        for _btn in (self.first_btn, self.prev_btn, self.next_btn, self.last_btn):
            _btn.setStyleSheet(
                f"color: {t.get('accent')};"
                f" font-size: {t.get('font_secondary')}px;"
                f" background: transparent; border: none;")
        self.page_size_combo.setStyleSheet(
            f"QComboBox {{ background: {t.get('btn_glass_bg')};"
            f" color: {t.get('text_primary')};"
            f" border: 1px solid {t.get('border')};"
            f" border-radius: {t.get('radius_pill')}px;"
            f" font-size: {t.get('font_secondary')}px; }}")
        self.page_jump_input.setStyleSheet(
            f"QSpinBox {{ background: {t.get('btn_glass_bg')};"
            f" color: {t.get('text_primary')};"
            f" border: 1px solid {t.get('border')};"
            f" border-radius: {t.get('radius_pill')}px;"
            f" font-size: {t.get('font_secondary')}px; }}")

    def apply_theme(self):
        """宿主页换肤链调用点（dashboard/actor/各页 apply_theme）。"""
        self._apply_style()

    def update_pagination(self, current_page: int, total_count: int, page_size: int):
        """更新分页状态。"""
        self.current_page = max(1, current_page)
        self.total_count = total_count
        self.page_size = page_size or self.page_size
        self.total_pages = max(1, (total_count + self.page_size - 1) // self.page_size)
        self.page_jump_input.setMaximum(self.total_pages)
        # 同步每页条数下拉（blockSignals 避免触发重新加载）
        self.page_size_combo.blockSignals(True)
        # 满屏自适应容量可能不在固定档位（如 24、36）——动态插入选项，
        # 保证「每页 N」显示与实际拉取一致（此前显示 20 实际 24 的两张皮）
        want = str(self.page_size)
        if self.page_size_combo.findText(want) < 0:
            opts = [int(self.page_size_combo.itemText(i))
                    for i in range(self.page_size_combo.count())]
            opts.append(self.page_size)
            opts.sort()
            for i, v in enumerate(opts):
                self.page_size_combo.setItemText(i, str(v))
            if len(opts) > self.page_size_combo.count():
                self.page_size_combo.insertItems(self.page_size_combo.count(),
                                                 [str(v) for v in opts[self.page_size_combo.count():]])
        self.page_size_combo.setCurrentText(want)
        self.page_size_combo.blockSignals(False)
        self._update_ui()

    def _update_ui(self):
        self.first_btn.setEnabled(self.current_page > 1)
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < self.total_pages)
        self.last_btn.setEnabled(self.current_page < self.total_pages)
        self.page_label.setText(f"第 {self.current_page} / {self.total_pages} 页")
        self.count_label.setText(f"共 {self.total_count} 个作品")
        # 同步跳转输入框的当前值，避免触发信号
        self.page_jump_input.blockSignals(True)
        self.page_jump_input.setValue(self.current_page)
        self.page_jump_input.blockSignals(False)

    def _go_to(self, page: int):
        if 1 <= page <= self.total_pages and page != self.current_page:
            self.current_page = page
            self._update_ui()
            self.page_changed.emit(page)

    def _on_jump(self):
        """回车键触发跳转。"""
        page = self.page_jump_input.value()
        self._go_to(page)

    def _on_jump_value_changed(self, page: int):
        """SpinBox 值改变时触发跳转（仅当用户交互时）。"""
        # 检查是否有键盘焦点（用户主动修改）
        if self.page_jump_input.hasFocus():
            self._go_to(page)

    def _on_page_size_changed(self, _index: int):
        """每页条数变化 → 重置到第 1 页并发出信号。"""
        self.page_size = int(self.page_size_combo.currentText())
        self.current_page = 1
        self.page_size_changed.emit(self.page_size)


# ── 主组件 ────────────────────────────────────────────────────────────────

class VideoGridWidget(QWidget):
    """
    视频宫格展示组件。

    特性：
    - 响应式网格布局，根据宽度自适应每行卡片数量
    - 支持复选框选择
    - 点击卡片触发播放事件
    - 内建分页功能

    信号：
    - play_requested(str, int): 点击播放，传递 file_path 和 frame_number
    - selection_changed(list): 选择变化，传递 video_id 列表
    - data_loaded(int): 数据加载完成，传递总数
    """

    play_requested = Signal(str, int)
    selection_changed = Signal(list)
    data_loaded = Signal(int)
    page_changed = Signal(int)
    reload_requested = Signal()  # 删除后请求外部刷新

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        page_size: int = 20,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # 自定义 QWidget 子类的 QSS background 必须开此属性才绘制
        # （标签浏览页内容区卡色不生效的根因）
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.api_client = api_client or APIClient()
        self.page_size = page_size

        self.videos: List[Dict] = []
        self.cards: List[VideoCard] = []
        self.selected_ids: List[str] = []

        # 分页
        self.current_page: int = 1
        self.total_count: int = 0

        # 加载器引用
        self._loader: Optional[_AsyncLoader] = None
        self._last_loader = None  # 最近一次加载闭包（容量变化重拉用）
        from collections import deque
        self._loader_refs = deque(maxlen=5)  # 保活：QThread 运行中被 GC 会 abort
        self._size_timer = QTimer(self)
        self._size_timer.setSingleShot(True)
        self._size_timer.timeout.connect(self._apply_auto_page_size)

        # 重排防抖 + 结果缓存（同 actor_grid：resize 里直接重排会无限递归卡死）
        self._layout_key = None
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self._relayout)

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────
    def _setup_ui(self):
        t = theme()

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        # 级联透明（同 label_mgmt）：网格容器/视口不吞组件底色
        self.scroll_area.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(12)  # ui-lint:allow 遗留债（迁移批清零）
        self.grid_layout.setContentsMargins(8, 8, 8, 8)  # ui-lint:allow 遗留债（迁移批清零）
        self.grid_layout.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area, 0, 0)

        # 空状态
        self.empty_label = QLabel("暂无视频数据")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            f"color: {t.get('text_tertiary')}; font-size: {t.get('font_size_lg')}px; padding: 40px;"  # ui-lint:allow 遗留债（迁移批清零）
        )
        self.empty_label.hide()
        layout.addWidget(self.empty_label, 0, 0)

        # 分页器
        self.pagination = PaginationWidget(compact=True)
        self.pagination.page_changed.connect(self._on_page_changed)
        layout.addWidget(self.pagination, 1, 0)
        # 空态/少数据时让滚动区吃掉全部剩余高度，分页器贴底不悬空
        layout.setRowStretch(0, 1)

    # ── 公开方法 ──────────────────────────────────────────────────

    def auto_page_size(self, refetch: bool = False):
        """按视口计算满屏页容量（FM：至少填满一屏，ceil 允许尾部滚动）。

        列宽模型走 grid_columns 单点——与 _relayout 实排同口径，容量计划
        几列实排就几列。resize 防抖后调用；refetch=True 时容量变化会以
        新 page_size 重拉当前页（浏览页数据走 API 分页，容量必须与服务
        端一致）。"""
        vw = self.scroll_area.viewport().width()
        vh = self.scroll_area.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        cols, card_w = grid_columns(vw)
        card_h = grid_card_height(card_w)
        avail = max(0, vh - 2 * GRID_MARGIN)
        rows = max(1, -(-avail // (card_h + GRID_SPACING)))
        new_size = max(MIN_FILL_PAGE_SIZE, min(400, cols * rows))
        changed = new_size != self.page_size
        if changed:
            self.page_size = new_size
            self.pagination.page_size = new_size
            self.pagination.update_pagination(self.current_page,
                                              self.total_count, new_size)
        if refetch and changed and hasattr(self, "_last_loader") and self._last_loader:
            self._last_loader()
        return changed

    def _apply_auto_page_size(self):
        """resize 防抖落点：容量变化且最近有加载 → 以新容量重拉当前页。"""
        changed = self.auto_page_size()
        if changed and self._last_loader:
            self._last_loader()

    def apply_theme(self):
        """换肤重放：组件底色/网格区随主题（标签浏览页 apply_theme 链
        调用——缺失时切浅色内容区残留深色，2026-09-29 实测）。"""
        t = theme()
        self.setStyleSheet(
            f"VideoGridWidget {{ background-color: {t.get('bg_card')}; }}")
        self.scroll_area.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }")
        self.empty_label.setStyleSheet(
            f"color: {t.get('text_tertiary')};"
            f" font-size: {t.get('font_section')}px;"
            f" background: transparent; padding: 40px;")  # ui-lint:allow 遗留债
        self.pagination.apply_theme()


    def load_by_tag(self, tag: str, page: int = 1):
        """通过标签加载视频列表。"""
        self.auto_page_size()  # 拉取前先按视口算满屏容量
        self.current_page = page
        self._show_loading()

        def fetch():
            return self.api_client.get(
                "/api/videos/by-tag",
                params={"tag": tag, "page": page, "size": self.page_size},
            )

        self._last_loader = lambda: self.load_by_tag(tag, page=1)
        self._start_loader(fetch)

    def load_all(self, page: int = 1):
        """加载所有视频（分页）。"""
        self.auto_page_size()
        self.current_page = page
        self._show_loading()

        def fetch():
            return self.api_client.get(
                "/api/videos/all",
                params={"page": page, "size": self.page_size},
            )

        self._last_loader = lambda: self.load_all(page=1)
        self._start_loader(fetch)

    def load_by_actor(self, actor_id: int, page: int = 1):
        """通过人物 ID 加载视频列表。"""
        self.auto_page_size()
        self.current_page = page
        self._show_loading()

        def fetch():
            return self.api_client.get(
                f"/api/actors/{actor_id}/videos",
                params={"page": page, "size": self.page_size},
            )

        # 与 load_by_tag/load_all 同机制：登记最近加载闭包，容量随视口
        # 变化后由防抖落点补拉（缺失时弹窗网格 resize 永远不跟随）
        self._last_loader = lambda: self.load_by_actor(actor_id, page=1)
        self._start_loader(fetch)

    def set_videos(self, videos: List[Dict], total_count: Optional[int] = None):
        """直接设置视频数据（不走 API）。"""
        self.total_count = total_count if total_count is not None else len(videos)
        self._set_videos_internal(videos)

    # ── 选择操作 ──────────────────────────────────────────────────

    def select_all(self):
        """全选当前页所有视频。"""
        for card in self.cards:
            card.checkbox.setChecked(True)

    def invert_selection(self):
        """反选当前页所有视频。"""
        for card in self.cards:
            card.checkbox.setChecked(not card.checkbox.isChecked())

    def clear_selection(self):
        """清空选择。"""
        for card in self.cards:
            card.checkbox.blockSignals(True)
            card.checkbox.setChecked(False)
            card.checkbox.blockSignals(False)
        self.selected_ids.clear()
        self.selection_changed.emit([])

    def get_selected_ids(self) -> List[str]:
        """获取当前选中的 video_id 列表。"""
        return list(self.selected_ids)

    # ── 内部方法 ──────────────────────────────────────────────────

    def _start_loader(self, fetch_func):
        """启动异步加载。"""
        if self._loader:
            try:
                self._loader.finished.disconnect()
                self._loader.error.disconnect()
            except RuntimeError:
                pass

        self._loader = _AsyncLoader(fetch_func)
        self._loader_refs.append(self._loader)  # 防运行中 GC abort
        self._loader.finished.connect(self._on_load_finished)
        self._loader.error.connect(self._on_load_error)
        self._loader.run()

    def _on_load_finished(self, data):
        """API 加载完成回调。"""
        videos = data.get("videos", data.get("items", []))
        total = data.get("total", len(videos))
        self.total_count = total
        self._set_videos_internal(videos)
        self.data_loaded.emit(total)

    def _on_load_error(self, message: str):
        """API 加载失败回调（不直出原始异常——URL/重试细节走 stdout）。"""
        print(f"[video_grid] 加载失败: {message}", flush=True)
        self._clear_cards()
        self.scroll_area.hide()
        brief = ("无法连接服务，请确认后端已启动"
                 if ("Connection" in message or "Max retries" in message)
                 else "加载失败，请点「刷新」重试")
        self.empty_label.setText(brief)
        self.empty_label.setStyleSheet(
            f"color: {theme().get('text_tertiary')};"
            f" font-size: {theme().get('font_section')}px;"
            f" background: transparent; padding: {T.SP_12 * 3}px;")
        self.empty_label.show()

    def _set_videos_internal(self, videos: List[Dict]):
        """内部方法：将视频数据渲染到网格。"""
        self.videos = videos
        self._clear_cards()

        if not videos:
            self.scroll_area.hide()
            self.empty_label.setText("暂无视频数据")
            self.empty_label.show()
            self.pagination.update_pagination(self.current_page, 0, self.page_size)
            return

        self.empty_label.hide()
        self.scroll_area.show()

        for video in videos:
            card = VideoCard(video)
            card.play_requested.connect(self.play_requested.emit)
            card.checked_changed.connect(self._on_card_checked)
            card.delete_requested.connect(self._on_delete_requested)
            self.cards.append(card)

        self._relayout()
        self.pagination.update_pagination(self.current_page, self.total_count, self.page_size)

    def _clear_cards(self):
        for card in self.cards:
            self.grid_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()
        self.selected_ids.clear()
        # 让重排缓存在下一轮失效，否则"同宽同数量"时会漏排新卡片
        self._layout_key = None

    def _relayout(self):
        """根据当前宽度重新布局。

        **可用宽度与卡片数都没变则跳过** —— 终止递归的关键：重排会让布局失效
        并再次触发 resize，若每次都重排则永不收敛（实测会卡死主线程）。

        列宽模型与 auto_page_size 同源（grid_columns 单点）——容量计划
        几列、实排就几列，行数一致，满屏不留白（2026-09-30 修复）。"""
        if not self.cards:
            return

        available_width = self.scroll_area.viewport().width()
        if available_width <= 0:
            return

        layout_key = (available_width, len(self.cards))
        if layout_key == self._layout_key:
            return
        self._layout_key = layout_key

        cols, card_w = grid_columns(available_width)

        for card in self.cards:
            self.grid_layout.removeWidget(card)
            card.set_card_width(card_w)

        for idx, card in enumerate(self.cards):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)

    def _show_loading(self):
        self._clear_cards()
        self.scroll_area.hide()
        self.empty_label.setText("加载中…")
        self.empty_label.show()

    def _on_page_changed(self, page: int):
        """分页器切换页码事件。"""
        self.page_changed.emit(page)

    def _on_card_checked(self, video_id: str, checked: bool):
        if checked:
            if video_id not in self.selected_ids:
                self.selected_ids.append(video_id)
        else:
            if video_id in self.selected_ids:
                self.selected_ids.remove(video_id)
        self.selection_changed.emit(list(self.selected_ids))

    def _on_delete_requested(self, video: Dict):
        """删除单张卡片对应的视频记录(弹三选对话框)。"""
        from ..components.results_page import ConfirmDialog
        from PySide6.QtWidgets import QMessageBox

        video_id = video.get("id")
        file_path = video.get("file_path", "")
        name = file_path.split("/")[-1] if file_path else f"ID {video_id}"

        choice = ConfirmDialog.delete_choice(
            self,
            "确认删除",
            f"确定要删除视频「{name}」吗？\n\n"
            "「仅删除记录」会移除数据库登记，保留磁盘视频文件；\n"
            "「删除记录和文件」会连同磁盘上的原视频文件一起删除。\n\n此操作不可恢复。",
        )
        if choice is None:
            return

        try:
            resp = self.api_client.post(
                "/videos/delete",
                json={
                    "task_ids": [video_id],
                    "delete_file": choice == "file",
                },
            )
            if resp.get("success") or resp.get("status") == "success":
                msg = resp.get("message", "已删除")
                if resp.get("file_errors"):
                    msg += "\n" + "\n".join(resp["file_errors"])
                QMessageBox.information(self, "删除完成", msg)
                # 从当前列表移除并刷新
                self.reload_requested.emit()
            else:
                QMessageBox.warning(self, "删除失败", resp.get("message", "未知错误"))
        except Exception as e:
            QMessageBox.warning(self, "删除失败", str(e))

    # ── 事件 ──────────────────────────────────────────────────────

    def resizeEvent(self, event):
        # 防抖后统一落点：重排+重算容量。不在此处直接算——resizeEvent 到达时
        # 子件几何尚未更新，按瞬时视口算容量曾锁死 9/页；逐次 resize 连发
        # 重拉也拖慢展示（防抖 200ms 后只补拉一次）
        super().resizeEvent(event)
        self._relayout_timer.start(200)
        self._size_timer.start(200)
