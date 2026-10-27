"""Actor management page with split layout and detail panel.

Provides a browsable table of recognized actors (face cover, code, name,
source file count, creation date) with merge / rename / delete operations.
Split layout: left side shows actor list, right side shows actor details.
"""

import requests
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog,
    QMessageBox, QScrollArea, QGridLayout, QFrame, QInputDialog,
    QLineEdit, QComboBox, QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QRunnable, QThreadPool
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor,
)
from PySide6.QtCore import QRectF, QPointF

from .theme import (
    theme,
    button_primary_style,
    button_secondary_style,
    button_danger_style,
)
from .display_names import actor_code_display

try:
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.tokens import T
from ..widgets.video_grid_widget import PaginationWidget
from .api_client import APIClient
from ..widgets.video_grid_widget import VideoGridWidget


# ── Actor Card Widget ──────────────────────────────────────────────

# 卡片右上角勾选框尺寸(actor_grid 计算卡片高度时需与之一致)
CARD_CHECKBOX_SIZE = 18

# 满屏页容量业务下限（与初始 _page_size=24 同源）：布局未落位时的瞬时
# 视口会算出极小值并被锁死（实测首次进页 9/页），低于下限一律钳回
ACTOR_MIN_FILL_PAGE_SIZE = 24


class CardCheckBox(QCheckBox):
    """人脸卡片上的高对比度勾选框。

    原生 QCheckBox 的指示框由当前 Qt 风格(Fusion)+ 深色调色板绘制, 颜色取自
    调色板的 Base/Window, 在卡片底色 bg_card(#38383A) 上几乎同色 —— 用户看不出
    「这里有个可以勾的框」, 自然只会去点人脸看详情。这里改为自绘指示框:
    未勾选画清晰描边方框, 勾选用 accent 填充 + 白色对勾, 保证深色卡片上一眼可见。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CARD_CHECKBOX_SIZE, CARD_CHECKBOX_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("勾选以选择该人物（用于合并 / 批量操作）")

    def paintEvent(self, event):  # noqa: N802 - Qt 命名
        t = theme()
        s = float(self.width())
        checked = self.isChecked()
        hovered = self.underMouse()
        enabled = self.isEnabled()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(1.0, 1.0, s - 2.0, s - 2.0)

        if checked:
            # 勾选态: accent 填充 + 白色对勾(与卡片底色反差强)
            fill = QColor(t.get("accent_hover") if hovered else t.get("accent"))
            border = fill
        elif not enabled:
            fill = QColor(t.get("bg_primary"))
            border = QColor(t.get("text_tertiary"))
        else:
            # 未勾选态: 凹入底色 + 高对比描边。
            # 只画一道细描边在深色卡片上仍然"看不见", 因此底色要和卡片底色
            # bg_card(#38383A) 明显不同(bg_primary 更暗), 才能一眼看出"这里有个框"。
            fill = QColor(t.get("bg_secondary") if hovered else t.get("bg_primary"))
            border = QColor(t.get("text_primary") if hovered else t.get("text_secondary"))

        painter.setBrush(QBrush(fill))
        pen = QPen(border)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 4.0, 4.0)

        if checked:
            check_pen = QPen(QColor(t.get("text_on_accent")))
            check_pen.setWidthF(2.0)
            check_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            p1 = QPointF(s * 0.27, s * 0.52)
            p2 = QPointF(s * 0.44, s * 0.69)
            p3 = QPointF(s * 0.75, s * 0.33)
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)

        painter.end()

    def enterEvent(self, event):  # noqa: N802 - Qt 命名
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):  # noqa: N802 - Qt 命名
        super().leaveEvent(event)
        self.update()


class ActorCardWidget(QWidget):
    """人物卡（FM4.2-4.4）：底 win 色、圆角 8、hover 描边 accent.5；
    勾选框悬浮封面右上（14px 不占行）；封面 4:3 + 左下「N 张脸」角标；
    meta=人物名 13/600 + 编号「人物 N」11/txt3 + 「出现于 N 个文件」11/txt2。
    点击卡（非勾选框）=详情抽屉；右键菜单保留。"""

    clicked = Signal(int)  # 点击信号，传递 actor_id
    checked = Signal(int, bool)  # 复选框状态变化信号
    annotate_requested = Signal(int, str)  # 标注请求: (actor_id, face_type)

    def __init__(self, actor_data: dict, parent=None):
        super().__init__(parent)
        self.actor_data = actor_data
        self.actor_id = actor_data["id"]
        self._selected = False
        self._checked = False

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._update_style()

        t = theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 封面 4:3（勾选框悬浮右上、角标左下）──
        self.cover_container = QFrame()
        self.cover_container.setObjectName("CoverContainer")
        self.cover_container.setStyleSheet(f"""
            QFrame#CoverContainer {{
                background: {t.get('bg_secondary')};
                border-top-left-radius: {t.get('radius_card')}px;
                border-top-right-radius: {t.get('radius_card')}px;
            }}
        """)
        cover_h = 180  # ui-lint:allow 4:3 基准（卡宽 240 时）
        self.cover_container.setFixedHeight(cover_h)
        cover_layout = QVBoxLayout(self.cover_container)
        cover_layout.setContentsMargins(0, 0, 0, 0)

        self.cover_label = QLabel()
        self.cover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.cover_label.setStyleSheet("background: transparent;")
        self._load_cover()
        cover_layout.addWidget(self.cover_label)

        # FM4.2 勾选框悬浮封面右上（14px，自绘高对比）
        self.checkbox = CardCheckBox(self.cover_container)
        self.checkbox.move(self.cover_container.width() - 14 - 8, 8)
        self.checkbox.raise_()
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)

        # FM4.4 左下「N 张脸」角标（黑 60% 底白字圆角 4）
        face_count = (actor_data.get("face_count")
                      or actor_data.get("face_total")
                      or actor_data.get("source_count") or 0)
        self.badge = QLabel(f"{face_count} 张脸", self.cover_container)
        self.badge.setStyleSheet(
            f"QLabel {{ font-size: {t.get('dur_badge_fs')}px; color: #FFFFFF;"
            f" background: rgba(0,0,0,0.6);"
            f" border-radius: {t.get('dur_badge_r')}px;"
            f" padding: 1px {t.get('dur_badge_pad_x')}px; }}")
        self.badge.adjustSize()
        self.badge.move(6, self.cover_container.height() - self.badge.height() - 6)
        self.badge.show()
        self.badge.raise_()

        layout.addWidget(self.cover_container)

        # ── meta 区（FM4.3，padding 8 10 10）──
        meta = QWidget()
        meta.setStyleSheet("background: transparent;")
        meta_v = QVBoxLayout(meta)
        meta_v.setContentsMargins(10, 8, 10, 10)  # ui-lint:allow 遗留债（对话框批清零）
        meta_v.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）

        primary_name = actor_data.get("primary_name")
        name = primary_name or actor_data.get("name") or ""
        self.name_label = QLabel(name if name else "未命名")
        self.name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        if name:
            self.name_label.setStyleSheet(
                f"font-size: {t.get('font_body')}px; font-weight: 600;"
                f" color: {t.get('text_primary')}; background: transparent;")
        else:  # 未命名=txt2 斜体
            self.name_label.setStyleSheet(
                f"font-size: {t.get('font_body')}px; font-style: italic;"
                f" color: {t.get('text_secondary')}; background: transparent;")
        aliases = actor_data.get("aliases", [])
        tip = []
        if primary_name and actor_data.get("name"):
            tip.append(f"原始名: {actor_data['name']}")
        if aliases:
            tip.append(f"别名: {', '.join(aliases)}")
        if tip:
            self.name_label.setToolTip("\n".join(tip))
        meta_v.addWidget(self.name_label)

        # 人物编号 11/txt3（展示层去 actor 术语）
        self.code_label = QLabel(actor_code_display(actor_data.get("code", "")))
        self.code_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.code_label.setStyleSheet(
            f"font-size: {t.get('font_hint')}px;"
            f" color: {t.get('text_tertiary')}; background: transparent;")
        meta_v.addWidget(self.code_label)

        # 「出现于 N 个文件」11/txt2
        source_count = actor_data.get("source_count", 0)
        source_files = actor_data.get("source_files", [])
        self.source_label = QLabel(f"出现于 {source_count} 个文件")
        self.source_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.source_label.setStyleSheet(
            f"font-size: {t.get('font_hint')}px;"
            f" color: {t.get('text_secondary')}; background: transparent;")
        if source_files:
            self.source_label.setToolTip("来源文件:\n" + "\n".join(source_files))
        meta_v.addWidget(self.source_label)

        # 人脸完整度标注（训练数据）——meta 行尾小徽标
        face_type = actor_data.get("face_type")
        if face_type in ("complete", "partial"):
            ft_text = "✓ 完整脸" if face_type == "complete" else "⚠ 局部/误检"
            ft_color = t.get("success") if face_type == "complete" else t.get("danger")
            self.face_type_label = QLabel(ft_text)
            self.face_type_label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.face_type_label.setStyleSheet(
                f"font-size: {t.get('font_hint')}px; font-weight: 600;"
                f" color: {ft_color}; background: transparent;")
            self.face_type_label.setToolTip(
                "已标注的人脸完整度（用于训练识别分类器）。\n"
                "右键人物卡片可修改标注。")
            meta_v.addWidget(self.face_type_label)

        layout.addWidget(meta)

    def _show_context_menu(self, pos):
        """右键菜单：标记人脸完整度（训练数据积累）。"""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {theme().get('bg_card')}; color: {theme().get('text_primary')};
                       border: 1px solid {theme().get('border')}; }}
            QMenu::item:selected {{ background: {theme().get('bg_hover')}; }}
        """)
        current = self.actor_data.get("face_type")

        act_complete = menu.addAction("✓ 完整脸" + ("  ✓" if current == "complete" else ""))
        act_partial = menu.addAction("⚠ 局部/误检" + ("  ✓" if current == "partial" else ""))
        act_clear = menu.addAction("清除标注")
        menu.addSeparator()
        act_detail = menu.addAction("打开详情")

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == act_complete:
            self.annotate_requested.emit(self.actor_id, "complete")
        elif chosen == act_partial:
            self.annotate_requested.emit(self.actor_id, "partial")
        elif chosen == act_clear:
            self.annotate_requested.emit(self.actor_id, "unknown")
        elif chosen == act_detail:
            self.clicked.emit(self.actor_id)

    def _load_cover(self):
        """加载封面图片"""
        cover_path = self.actor_data.get("cover_face")
        if not cover_path:
            return
            
        full_path = Path(cover_path)
        if not full_path.exists():
            # 尝试用path_manager转换路径
            try:
                from src.core.path_manager import get_path_manager
                pm = get_path_manager()
                full_path = pm.base_dir / cover_path.replace("data/", "")
            except Exception:
                pass
        
        if not full_path.exists():
            return
        
        try:
            import cv2
            
            img = cv2.imread(str(full_path))
            if img is None:
                return
            
            # BGR转RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            
            # 复制数据以防止 numpy 数组被回收后 QImage 数据失效
            q_img = QImage(img_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # 转换为QPixmap：保存原始图，显示尺寸在 _apply_cover_pixmap 中
            # 按 cover_label 实际大小缩放（卡片宽度不同时人脸都尽量占满封面区）
            pixmap = QPixmap.fromImage(q_img)
            if pixmap.isNull():
                return
            self._cover_pixmap = pixmap
            self._apply_cover_pixmap()
        except Exception:
            pass

    def _apply_cover_pixmap(self):
        """按 cover_label 当前尺寸缩放封面图（KeepAspectRatio 居中）。"""
        try:
            if getattr(self, "_cover_pixmap", None) is None or self._cover_pixmap.isNull():
                return
            cw = self.cover_label.width()
            ch = self.cover_label.height()
            if cw <= 1 or ch <= 1:
                return
            scaled = self._cover_pixmap.scaled(
                cw, ch,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.cover_label.setPixmap(scaled)
        except Exception:
            pass
    
    def _update_style(self):
        """更新卡片样式

        状态优先级：点击查看详情(_selected) > 勾选(_checked) > 普通。
        勾选态用 accent 边框 + 浅色底，让「勾的是哪张卡」一目了然；
        点击查看详情态保持原有 accent 底高亮。
        """
        from .theme import theme
        t = theme()
        if self._selected:
            self.setStyleSheet(f"""
                ActorCardWidget {{
                    background-color: {t.get('bg_selected')};
                    border: 2px solid {t.get('accent')};
                    border-radius: {t.get('radius_lg')}px;
                }}
                ActorCardWidget:hover {{
                    background-color: {t.get('bg_hover')};
                }}
            """)
        elif self._checked:
            # 勾选态：accent 边框即可, 不整块涂蓝——选中与否由右上角
            # 原生 QCheckBox 的「✓」表达(符合习惯), 整卡保持卡片底色
            self.setStyleSheet(f"""
                ActorCardWidget {{
                    background-color: {t.get('bg_card')};
                    border: 2px solid {t.get('accent')};
                    border-radius: {t.get('radius_lg')}px;
                }}
                ActorCardWidget:hover {{
                    border: 2px solid {t.get('accent_hover')};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                ActorCardWidget {{
                    background-color: {t.get('bg_card')};
                    border: 2px solid {t.get('border')};
                    border-radius: {t.get('radius_lg')}px;
                }}
                ActorCardWidget:hover {{
                    border: 2px solid {t.get('accent')};
                }}
            """)
    
    def _on_checkbox_changed(self, state):
        """复选框状态变化"""
        is_checked = state == Qt.CheckState.Checked.value
        self._checked = is_checked
        self._update_style()
        self.checked.emit(self.actor_id, is_checked)
    
    def mousePressEvent(self, event):
        """点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.actor_id)
        super().mousePressEvent(event)
    
    def set_selected(self, selected: bool):
        """设置选中状态（点击查看详情）"""
        self._selected = selected
        self._update_style()

    def resizeEvent(self, event):  # noqa: N802 - Qt 命名
        """卡片尺寸变化 → 悬浮元素全部归位（勾选框右上、角标左下）。

        搜索/翻页重建后 _relayout 与子件布局激活存在时序竞速
        （actor_grid.py 同日修复注），无论谁触发 resize、时序如何，
        这里都强制归位，保证勾选框/角标可见可点、不与 meta 行叠字。"""
        super().resizeEvent(event)
        cover_h = self._cover_height_for(self.width())
        self.checkbox.move(self.width() - 14 - 8, 8)
        if getattr(self, "badge", None) is not None:
            self.badge.move(6, max(0, cover_h - self.badge.height() - 6))

    def _cover_height_for(self, card_width: int) -> int:
        """封面高度唯一公式（4:3）——_relayout 与悬浮元素归位共用,
        避免两处口径漂移（历史:构造写死 180 vs _relayout 按卡宽计算,
        卡宽<240 时容器吃 180 而 meta 被顶进封面,角标与名字叠字）。"""
        return int(card_width * 3 / 4)


    def apply_theme(self):
        """换肤：重放封面占位背景与文字透明背景，并刷新卡片边框/底色。
        不重建控件，勾选状态保留。"""
        from .theme import theme
        t = theme()
        # 封面占位背景与卡片同色
        self.cover_label.setStyleSheet(f"""
        self.pagination.apply_theme()
            QLabel {{
                background-color: {t.get('bg_card')};
            }}
        """)
        # 文字标签显式透明，避免 QLabel 样式表导致 palette 底色色块
        for lbl in (self.code_label, self.name_label, self.source_label):
            try:
                cur = lbl.styleSheet()
                if cur and "background" not in cur:
                    lbl.setStyleSheet(cur + "; background: transparent;")
            except Exception:
                pass
        if hasattr(self, "face_type_label"):
            try:
                cur = self.face_type_label.styleSheet()
                if cur and "background" not in cur:
                    self.face_type_label.setStyleSheet(cur + "; background: transparent;")
            except Exception:
                pass
        # 自绘勾选框按 theme() 取色, 换肤后需重绘
        self.checkbox.update()
        self._update_style()


def _fetch_actors(
    api_base: str, page: int, page_size: int, search: str, only_with_cover: bool,
    ids: str = "",
) -> dict:
    """同步取一次人物列表（在工作线程执行——只做网络 I/O，不碰 Qt 控件）。

    ids: 逗号分隔的人物 id（"只看已勾选"用）。勾选集合是**跨页**的，所以按 id
    向服务端要数据，而不是在本地过滤当前页——否则翻页就看不到其余勾选项。
    """
    params: dict = {"page": page, "page_size": page_size}
    if search:
        params["search"] = search
    if only_with_cover:
        params["only_with_cover"] = "true"
    if ids:
        params["ids"] = ids
    resp = requests.get(f"{api_base}/actors", params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.json()


def _fetch_actor_ids(api_base: str) -> Optional[list]:
    """同步取全部人物 id（工作线程执行；供剔除失效勾选）。

    失败返回 None（主线程据此**跳过**剔除，避免误清用户勾选）；
    成功返回 id 列表（可能是空列表）。
    """
    try:
        resp = requests.get(f"{api_base}/actors/ids", timeout=10)
        if resp.status_code != 200:
            return None
        return list(resp.json().get("ids") or [])
    except Exception:
        return None


class _ActorsFetchTask(QRunnable):
    """在线程池里取一次人物列表，结果经页面的信号回到主线程。

    生命周期刻意**不自持 QThread**：线程由页面的 QThreadPool 管理，从根上
    避免"每次加载新建 QThread 并赋给控件属性、运行中被回收 → 析构 qFatal"的崩溃。

    latest-wins：任务启动与结果回主线程时都用**请求序号**比对，过期请求既不
    发网络也不渲染——快速输入时不会被旧结果覆盖，也不会为作废请求白跑一次。
    """

    def __init__(self, page, seq: int, api_base: str, page_no: int,
                 page_size: int, search: str, only_with_cover: bool, ids: str = ""):
        super().__init__()
        self._page = page
        self._seq = seq
        self._api_base = api_base
        self._page_no = page_no
        self._page_size = page_size
        self._search = search
        self._only_with_cover = only_with_cover
        self._ids = ids
        self.setAutoDelete(True)

    def run(self) -> None:
        page = self._page
        if page is None or self._seq != page._req_seq:
            return  # 已被更新的请求取代，连网络都不发
        try:
            data = _fetch_actors(
                self._api_base, self._page_no, self._page_size,
                self._search, self._only_with_cover, self._ids,
            )
        except Exception as e:  # noqa: BLE001 - 交页面统一提示
            page._load_failed.emit(self._seq, str(e))
            return
        # 顺带取一次全量有效 id（轻量），供主线程剔除已删除人物的残留勾选。
        # 取失败返回 None → 主线程跳过剔除，不误清用户勾选。
        if isinstance(data, dict):
            data["_valid_ids"] = _fetch_actor_ids(self._api_base)
        page._load_ready.emit(self._seq, data)


# ── Actor Management Page ────────────────────────────────────────────

class AsyncLoader(QObject):
    """Generic async data loader to prevent UI blocking."""
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
        # Keep self alive until thread finishes
        self._thread.finished.connect(self.deleteLater)
        self._thread.start()
    
    def _execute(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self._thread:
                self._thread.quit()


# ── Actor Management Page ────────────────────────────────────────────

class ActorManagementPage(QWidget):
    """Full-page actor grid with toolbar, pagination, and batch operations."""

    # 加载任务 → 主线程（带请求序号，见 _ActorsFetchTask / _on_actors_loaded）
    _load_ready = Signal(int, object)
    _load_failed = Signal(int, str)

    def __init__(self, api_base: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.api_base = api_base
        self._current_page: int = 1
        self._page_size: int = 24  # 满屏容量在 showEvent 按视口计算（FM：内容不超一屏）
        self._total_items: int = 0
        self._detail_dialog: Optional[ActorDetailDialog] = None  # 跟踪弹窗实例
        self._selected_actor_ids_set: set = set()  # 选中的人物 ID 集合
        # 「只看已勾选」筛选态（跨页：按 id 向服务端取数，而非过滤当前页）
        self._only_checked: bool = False
        # 加载：单线程池 + 请求序号（latest-wins）。线程归池管理、页面不自持，
        # 从根上避免"运行中 QThread 被回收 → 析构 qFatal"的崩溃。
        self._req_seq: int = 0
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(1)
        self._load_ready.connect(self._on_actors_loaded)
        self._load_failed.connect(self._on_actors_error)
        # 满屏容量防抖重算：resizeEvent/showEvent 到达时子件几何可能尚未
        # 落位，直接按瞬时视口算容量曾锁死 9/页（2026-09-30 修复），且
        # 逐次 resize 连发重拉拖慢展示——200ms 防抖后只算一次、只补拉一次
        self._fill_timer = QTimer(self)
        self._fill_timer.setSingleShot(True)
        self._fill_timer.timeout.connect(self._apply_fill_page_size)
        self._build_ui()
        self._load_actors_async()

    # ── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        t = theme()
        
        # 设置页面背景色
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 满铺（18§4.0）：面板贴边
        layout.setSpacing(0)

        # （原「人脸管理」标题已删——页名由 rail 图标标识，页内重复多余）

        # ── 批量操作七钮：迁往主窗口顶带（F1）——面板无父级构建，
        # 经 take_face_bar() 交接。按钮槽函数仍在本页，信号不受换父影响。
        self._face_bar = QWidget()
        fb = QHBoxLayout(self._face_bar)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.setSpacing(T.SP_12)  # FM1.1：每钮独立间隔 12（用户 2026-09-28 裁定）

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet(button_secondary_style())
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.refresh_btn.clicked.connect(self.load_actors)
        fb.addWidget(self.refresh_btn)

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setStyleSheet(button_secondary_style())
        self.select_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_all_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.select_all_btn.clicked.connect(self._on_select_all)
        fb.addWidget(self.select_all_btn)

        # 只看已勾选（可切换，跨页）：勾选集合本身跨页保留，因此按 id 向服务端取数，
        # 把各页勾出来的项集中显示——而不是只过滤当前页。
        # 按钮文本实时携带勾选总数（见 _update_checked_count），0 表示尚未勾选。
        self.only_checked_btn = QPushButton("已勾选（0）")
        self.only_checked_btn.setCheckable(True)
        self.only_checked_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.only_checked_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.only_checked_btn.setToolTip(
            "只看已勾选的人物（跨页）：把各页勾选出来的项集中显示；再点一次恢复全部"
        )
        self.only_checked_btn.toggled.connect(self._on_only_checked_toggled)
        fb.addWidget(self.only_checked_btn)
        self._apply_only_checked_style()

        self.clear_selection_btn = QPushButton("取消勾选")
        self.clear_selection_btn.setStyleSheet(button_secondary_style())
        self.clear_selection_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_selection_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.clear_selection_btn.setToolTip("取消全部勾选（跨页生效）")
        self.clear_selection_btn.clicked.connect(self._on_clear_selection)
        fb.addWidget(self.clear_selection_btn)

        self.merge_btn = QPushButton("合并选中")
        self.merge_btn.setStyleSheet(button_secondary_style())
        self.merge_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.merge_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.merge_btn.clicked.connect(self.merge_selected)
        fb.addWidget(self.merge_btn)

        self.delete_selected_btn = QPushButton("删除选中")
        self.delete_selected_btn.setStyleSheet(button_danger_style())
        self.delete_selected_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_selected_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.delete_selected_btn.clicked.connect(self.batch_delete_selected)
        fb.addWidget(self.delete_selected_btn)

        self.rebuild_faces_btn = QPushButton("重建选中人物人脸")
        self.rebuild_faces_btn.setStyleSheet(button_secondary_style())
        self.rebuild_faces_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rebuild_faces_btn.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.rebuild_faces_btn.clicked.connect(self._rebuild_faces)
        self.rebuild_faces_btn.setToolTip("从已有帧文件重建选中人物的人脸特征")
        # 交互对齐合并/删除钮：常可点、空选弹提示（用户 2026-10-01 裁定，
        # 废除"勾选后才激活"的硬禁用门槛——禁用态视觉弱，悬停不可点显得诡异）
        fb.addWidget(self.rebuild_faces_btn)
        fb.addStretch()  # 无 stretch 时 band_stack 会把宽度均摊给七个钮（实测拉成宽条）

        # 搜索框入顶带（用户指令）：七钮左组、搜索框右置。
        # F2：回车直接搜索——独立「搜索」「清除」钮已删，
        # textEdited 实时过滤 + returnPressed 显式搜索双通道保留。
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索人物名称...")
        self.search_input.setFixedWidth(200)  # ui-lint:allow 遗留债（对话框批清零）
        self.search_input.setFixedHeight(28)  # 控件一档制 28（根参数钉板）  # ui-lint:allow 遗留债（对话框批清零）
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                font-size: {t.get('font_size_sm')}px;
                padding: 3px 10px;  # ui-lint:allow 遗留债（对话框批清零）
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                background: {t.get('bg_primary')};
                color: {t.get('text_primary')};
            }}
            QLineEdit:focus {{
                border: 1px solid {t.get('accent')};
            }}
        """)
        self.search_input.textEdited.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._on_search)
        fb.addWidget(self.search_input)

        # （原页内 toolbar 行删除：七钮+搜索在顶带，统计与分页行信息重复
        # ——重复元素与空浮条一并撤除，2026-09-29 用户令。
        # 加载中提示由 actor_grid.set_loading 占位承担）

        # Grid view
        from .actor_grid import ActorGridWidget
        self.actor_grid = ActorGridWidget()
        self.actor_grid.card_clicked.connect(self._on_card_clicked)
        self.actor_grid.selection_changed.connect(self._on_selection_changed)
        self.actor_grid.annotate_requested.connect(self._on_annotate_actor)
        layout.addWidget(self.actor_grid, 1)

        # ── 分页栏：全 app 唯一形态 PaginationWidget（30 底栏纯分页控制；
        # 2026-09-29 用户裁定收编手写行）──
        self.pagination = PaginationWidget(default_page_size=self._page_size)
        self.pagination.page_changed.connect(self._go_to_page)
        self.pagination.page_size_changed.connect(self._on_page_size_changed)
        layout.addWidget(self.pagination)
        # （统计「共 N 项，选中 N 项」删除——与分页行「共 N 个作品」+顶带
        # 「已勾选（N）」信息三重重复，2026-09-29 用户令去重）

        # 换肤链路收口：apply_theme 内含样式重放+一档制钳 28（17 号规范）
        self.apply_theme()

    def _enforce_one_tier(self):
        """一档制终检（17 号规范）：显示与样式重放后钳 28。"""
        for _w in (self.findChildren(QPushButton) + self.findChildren(QComboBox)):
            try:
                _w.setFixedHeight(28)  # ui-lint:allow 遗留债（对话框批清零）
            except RuntimeError:
                pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 视口变化 → 防抖后重算满屏容量（落点统一在 _apply_fill_page_size）
        if hasattr(self, "actor_grid") and self.actor_grid.isVisible():
            self._fill_timer.start(200)

    def _compute_fill_page_size(self) -> int:
        """按视口计算满屏页容量（人物卡 4:3 + meta ≈66；ceil 填满）。

        业务下限 24：布局未落位时的瞬时视口会算出极小值并被锁死
        （实测首次进页 9/页），低于下限的一律钳回（2026-09-30 用户令）。"""
        vw = self.actor_grid.scroll_area.viewport().width()
        vh = self.actor_grid.scroll_area.viewport().height()
        if vw <= 0 or vh <= 0:
            return self._page_size
        cols = max(1, (vw - 20 + 10) // (190 + 10))
        card_w = max(180, (vw - 20 - 10 * (cols - 1)) // cols)
        card_h = int(card_w * 3 / 4) + 66
        avail = max(0, vh - 20)
        rows = max(1, -(-avail // (card_h + 10)))
        return max(ACTOR_MIN_FILL_PAGE_SIZE, min(200, cols * rows))

    def _apply_fill_page_size(self) -> None:
        """防抖落点：按已落位视口重算满屏容量；变化则重载当前页。"""
        if not hasattr(self, "pagination"):
            return
        new_size = self._compute_fill_page_size()
        if new_size != self._page_size:
            self._page_size = new_size
            self.pagination.page_size = new_size
            self._load_actors_async()

    def showEvent(self, e):
        super().showEvent(e)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._enforce_one_tier)
        # FM：满屏容量在布局落位后计算（防抖落点统一走 _apply_fill_page_size）
        self._fill_timer.start(200)

    def apply_theme(self):
        """换肤：仅重放样式不重建控件，保留勾选/分页/批量重建状态。"""
        from .theme import theme, button_secondary_style, button_danger_style
        t = theme()
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        self.refresh_btn.setStyleSheet(button_secondary_style())
        self.select_all_btn.setStyleSheet(button_secondary_style())
        # 「已勾选」按钮带 :checked 规则，换肤后需重放（否则开关态看不出）
        self._apply_only_checked_style()
        self.clear_selection_btn.setStyleSheet(button_secondary_style())
        self.merge_btn.setStyleSheet(button_secondary_style())
        self.delete_selected_btn.setStyleSheet(button_danger_style())
        self.rebuild_faces_btn.setStyleSheet(button_secondary_style())
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                font-size: {t.get('font_size_sm')}px;
                padding: 6px 12px;  # ui-lint:allow 遗留债（对话框批清零）
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                background: {t.get('bg_primary')};
                color: {t.get('text_primary')};
            }}
            QLineEdit:focus {{
                border: 1px solid {t.get('accent')};
            }}
        """)
        # 旧手写分页控件已收编 PaginationWidget（样式随组件 apply_theme）
        self.pagination.apply_theme()
        if hasattr(self, "actor_grid") and hasattr(self.actor_grid, "apply_theme"):
            self.actor_grid.apply_theme()
        self._enforce_one_tier()
        self.restyle_band_bar()

    def restyle_band_bar(self) -> None:
        """顶带换肤钩子：七钮面板已迁出页面，页面级样式表够不到它。"""
        bar = getattr(self, "_face_bar", None)
        if bar is None:
            return
        self.refresh_btn.setStyleSheet(button_secondary_style())
        self.select_all_btn.setStyleSheet(button_secondary_style())
        self.clear_selection_btn.setStyleSheet(button_secondary_style())
        self.merge_btn.setStyleSheet(button_secondary_style())
        self.delete_selected_btn.setStyleSheet(button_danger_style())
        self.rebuild_faces_btn.setStyleSheet(button_secondary_style())
        self._apply_only_checked_style()

    def take_face_bar(self) -> QWidget | None:
        """把批量操作七钮面板交给主窗口顶带（重复调用返回 None）。"""
        bar, self._face_bar = self._face_bar, None
        return bar

    # ── Data loading ───────────────────────────────────────────────

    def _apply_only_checked_style(self) -> None:
        """「已勾选」按钮样式：中性色（不引入彩色背景），用**填充 vs 描边**区分开关态。

        全局 QPushButton 样式没有 `:checked` 规则，不显式画的话开关态看不出来，
        用户无法判断自己是否处于筛选态。
        """
        t = theme()
        base = button_secondary_style()
        self.only_checked_btn.setStyleSheet(base + f"""
            QPushButton:checked {{
                background: {t.get('btn_glass_hover')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('text_secondary')};
            }}
            QPushButton:checked:hover {{
                background: {t.get('btn_glass_bg')};
            }}
        """)

    def _on_only_checked_toggled(self, checked: bool) -> None:
        """切换「只看已勾选」：回到第 1 页并按当前勾选集合重新取数。"""
        self._only_checked = bool(checked)
        self._current_page = 1
        self._load_actors_async()

    def _load_actors_async(self) -> None:
        """发起一次加载：递增请求序号 → 丢给单线程池（latest-wins）。"""
        # 筛选态但一个都没勾：不必打网络，直接说明原因（否则会被误读成"人物没了"）
        if self._only_checked and not self._selected_actor_ids_set:
            self._req_seq += 1  # 作废在途请求，避免旧结果盖住空状态
            self._end_loading()
            self._total_items = 0
            self._update_stats()
            self._update_pagination_ui()
            self.actor_grid.show_empty("还没有勾选任何人物")
            return

        self._req_seq += 1
        self._begin_loading()
        self._load_pool.start(_ActorsFetchTask(
            self, self._req_seq, self.api_base,
            self._current_page, self._page_size,
            self.search_input.text().strip(), True,
            self._checked_ids_param(),
        ))

    def _checked_ids_param(self) -> str:
        """跨页勾选集合 → 逗号分隔的 id 串（仅筛选态下发；升序便于日志比对）。"""
        if not self._only_checked:
            return ""
        return ",".join(str(i) for i in sorted(self._selected_actor_ids_set))

    def _begin_loading(self) -> None:
        """进入加载态：不清空已有卡片（避免搜索时闪成空白）。

        加载提示由 actor_grid 的加载占位承担（首载无卡时显示）。"""
        # 仅首次（还没有任何卡片）才显示"加载中"占位；已有结果时保留
        if not self.actor_grid.cards:
            self.actor_grid.set_loading(True)

    def _end_loading(self) -> None:
        self.actor_grid.set_loading(False)

    def _on_actors_loaded(self, seq: int, data: object) -> None:
        """Handle successful actors load (paginated response)."""
        if seq != self._req_seq:
            return  # 过期结果：已被更新的请求取代，丢弃不渲染

        self._end_loading()

        if not isinstance(data, dict) or "actors" not in data:
            self.actor_grid.show_empty("返回数据格式错误")
            return

        actors: list = data["actors"]
        # 剔除已不存在的 actor 勾选：跨页持久集合只增不减，若某人物被删除
        # （本页删除 / 后端识别自动合并 / 重建），其 id 会残留成"幽灵勾选"，
        # 之后合并会因 id 不存在而整体 404。这里按服务端返回的全量有效 id
        # 收敛；拉取失败(None)时跳过，避免误清用户的勾选。
        _valid_ids = data.get("_valid_ids")
        if _valid_ids is not None:
            self.actor_grid.prune_selection(_valid_ids)
        self._total_items = data.get("total", 0)
        self._current_page = data.get("page", self._current_page)

        # 设置人物数据到宫格视图
        self.actor_grid.set_actors(actors)
        if not self.actor_grid.actors:
            # FM4.6：无人物数据 → hero 引导+「去导入任务」（与文件表口径一致）
            self.actor_grid.show_empty(
                "还没有人物数据", hero=True, button_text="去导入任务")

        # 同步勾选: set_actors 会从持久集回填勾选(blockSignals 不触发
        # _on_selection_changed), 这里把页面副本同步为回填后的结果——
        # 保证 reload/翻页/搜索后勾选状态跨刷新保留。
        self._selected_actor_ids_set = set(self.actor_grid.get_selected_ids())
        self._update_checked_count()
        self._update_stats()

        self._update_pagination_ui()
        self._update_stats()

    def _on_actors_error(self, seq: int, error: str) -> None:
        """Handle actors load error."""
        if seq != self._req_seq:
            return
        self._end_loading()
        print(f"[actors] 加载失败: {error}", flush=True)
        brief = ("无法连接服务，请确认后端已启动"
                 if ("Connection" in error or "Max retries" in error)
                 else "加载失败，请点「刷新」重试")
        self.actor_grid.show_empty(brief)  # 错误态保持纯文本（非引导）

    def load_actors(self) -> None:
        """Reload actors with current page params."""
        self._load_actors_async()

    # ── Search ─────────────────────────────────────────────────────

    def _on_search_changed(self, text: str = "") -> None:
        """输入变化 → 防抖后按**服务端搜索**重载（作用于全部人物，而非仅当前页）。

        绑定的是 textEdited（仅用户输入触发）——程序 clear() 不会走这里。
        """
        if not hasattr(self, "_search_timer"):
            self._search_timer = QTimer(self)
            self._search_timer.setSingleShot(True)
            self._search_timer.setInterval(350)
            self._search_timer.timeout.connect(self._apply_search)
        self._search_timer.start()

    def _apply_search(self) -> None:
        """回到第 1 页并按当前关键词重载。"""
        self._current_page = 1
        self._load_actors_async()

    def _on_search(self) -> None:
        """搜索按钮 / 回车：立即执行。"""
        if hasattr(self, "_search_timer"):
            self._search_timer.stop()
        self._apply_search()

    def _on_clear_search(self) -> None:
        """清空关键词并立即重载（textEdited 不会被程序 clear 触发，无需去抖）。"""
        if hasattr(self, "_search_timer"):
            self._search_timer.stop()
        self.search_input.clear()
        self._apply_search()

    # ── Selection helpers ──────────────────────────────────────────

    def _checked_actor_ids(self) -> List[int]:
        """Return actor IDs for all checked cards."""
        return list(self._selected_actor_ids_set)

    def _selected_actor_ids(self) -> List[int]:
        """Return actor IDs from selection (used by merge)."""
        return list(self._selected_actor_ids_set)

    def _update_stats(self) -> None:
        """（已并入顶带「已勾选（N）」钮与分页行计数——保留空壳防旧调用崩。）"""
        pass

    def _total_pages(self) -> int:
        if self._total_items <= 0:
            return 1
        return max(1, (self._total_items + self._page_size - 1) // self._page_size)

    def _update_pagination_ui(self) -> None:
        """分页状态回填唯一形态组件。"""
        self.pagination.update_pagination(
            self._current_page, self._total_items, self._page_size)

    def _go_to_page(self, page: int) -> None:
        total_pg = self._total_pages()
        page = max(1, min(page, total_pg))
        if page == self._current_page:
            return
        self._current_page = page
        self._load_actors_async()

    def _on_page_size_changed(self, text) -> None:
        if isinstance(text, int):
            text = str(text)
        try:
            new_size = int(text)
        except ValueError:
            return
        if new_size == self._page_size:
            return
        self._page_size = new_size
        self._current_page = 1
        self._load_actors_async()

    # ── Cell click handler ─────────────────────────────────────────

    def _on_annotate_actor(self, actor_id: int, face_type: str) -> None:
        """标注人物人脸完整度（训练数据积累）。"""
        try:
            import requests
            payload = {"face_type": face_type}
            resp = requests.post(
                f"{self.api_base}/actors/{actor_id}/face-type",
                json=payload, timeout=10,
            )
            resp.raise_for_status()
            # 刷新列表显示标注徽标
            self.load_actors()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "标注失败", f"标记人物 {actor_id} 失败: {e}")

    def _on_card_clicked(self, actor_id: int) -> None:
        """Handle card click to show actor details."""
        # 如果弹窗已打开，切换内容并提到前台；否则打开新弹窗
        if self._detail_dialog and self._detail_dialog.isVisible():
            self._detail_dialog.load_actor(actor_id)
            # 非模态弹窗会被主窗口压到后面:同 id 早退时若不提前台,
            # 用户视角就是"点了没反应,详情出不来"(2026-10-07 实测)
            self._detail_dialog.raise_()
            self._detail_dialog.activateWindow()
        else:
            self._detail_dialog = ActorDetailDialog(self.api_base, actor_id, parent=self)
            self._detail_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._detail_dialog.destroyed.connect(self._on_detail_dialog_destroyed)
            self._detail_dialog.show()  # 使用 show() 而非 exec() 使其非模态
    
    def _update_checked_count(self) -> None:
        """「已勾选」按钮文本实时显示勾选总数（跨页持久集合的大小）。

        用户以按钮上的数字判断自己勾了多少人；单卡勾选/全选/清空/幽灵剔除
        都会经 selection_changed 到这里，加载回填走 _on_actors_loaded。
        """
        self.only_checked_btn.setText(f"已勾选（{len(self._selected_actor_ids_set)}）")

    def _on_selection_changed(self) -> None:
        """Handle selection changes in the grid."""
        selected_ids = self.actor_grid.get_selected_ids()
        self._selected_actor_ids_set = set(selected_ids)
        self._update_checked_count()
        self._update_stats()
    
    def _on_select_all(self) -> None:
        """Toggle select all / deselect all."""
        current_selected = len(self._selected_actor_ids_set)
        total_cards = len(self.actor_grid.cards)
        
        if current_selected == total_cards:
            # 已全选，取消全选
            self.actor_grid.clear_selection()
        else:
            # 未全选，执行全选
            self.actor_grid.select_all()

    def _on_clear_selection(self) -> None:
        """取消全部勾选（跨页生效：清空持久勾选集合）。

        勾选集合 _ACTOR_SELECTION 是进程级持久的，本操作把它整体清空并
        取消当前页卡片勾选，翻到其它页面也不会再回填勾选。
        """
        self.actor_grid.clear_selection()

    def _on_detail_dialog_destroyed(self) -> None:
        """弹窗关闭时清理引用并刷新列表"""
        self._detail_dialog = None
        self.load_actors()  # 重命名/删除后刷新列表

    # ── Actions ────────────────────────────────────────────────────

    def _delete_single(self, actor_id: int) -> None:
        """Delete a single actor via the batch endpoint."""
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除此人物吗？\n（不会影响视频文件，仅清除关联）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            resp = requests.delete(
                f"{self.api_base}/actors/batch",
                json={"actor_ids": [actor_id]},
                timeout=10,
            )
            if resp.status_code == 200:
                QMessageBox.information(self, "成功", "删除完成")
                # 移除被删人物的勾选残留
                self._selected_actor_ids_set.discard(actor_id)
                self.actor_grid.selected_ids.discard(actor_id)
                self.load_actors()
            else:
                QMessageBox.critical(self, "错误", f"删除失败: {resp.text}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"删除失败: {exc}")

    def _rebuild_faces(self) -> None:
        """从已有帧文件重建选中人物的人脸特征"""
        actor_ids = self._checked_actor_ids()
        if not actor_ids:
            QMessageBox.warning(self, "提示", "请先勾选要重建人脸的人物")
            return

        reply = QMessageBox.question(
            self,
            "确认重建",
            f"将为选中的 {len(actor_ids)} 个人物重建人脸特征。\n"
            "这会重新提取人脸数据并更新人物记录。\n\n"
            "确定要继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # 启动批量重建任务
            resp = requests.post(
                f"{self.api_base}/actors/batch-rebuild-faces",
                json={"actor_ids": actor_ids},
                timeout=30,
            )
            if resp.status_code != 200:
                QMessageBox.critical(self, "错误", f"启动重建失败: {resp.text}")
                return
            
            # 禁用按钮，显示进度
            self.rebuild_faces_btn.setEnabled(False)
            self.rebuild_faces_btn.setText("重建中...")
            
            # 轮询进度
            self._poll_rebuild_progress()
            
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"重建失败: {exc}")
            self.rebuild_faces_btn.setEnabled(True)
            self.rebuild_faces_btn.setText("重建选中人物人脸")

    def _poll_rebuild_progress(self):
        """轮询重建进度"""
        try:
            resp = requests.get(
                f"{self.api_base}/actors/batch-rebuild-faces/status",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                
                if status == "running":
                    total = data.get("total", 0)
                    processed = data.get("processed", 0)
                    
                    self.rebuild_faces_btn.setText(f"重建中 {processed}/{total}")
                    
                    # 1秒后继续轮询
                    QTimer.singleShot(1000, self._poll_rebuild_progress)
                    
                elif status == "completed":
                    message = data.get("message", "重建完成")
                    self.rebuild_faces_btn.setEnabled(True)
                    self.rebuild_faces_btn.setText("重建选中人物人脸")
                    QMessageBox.information(self, "重建完成", message)
                    self.load_actors()
                    
                elif status == "failed":
                    message = data.get("message", "重建失败")
                    self.rebuild_faces_btn.setEnabled(True)
                    self.rebuild_faces_btn.setText("重建选中人物人脸")
                    QMessageBox.critical(self, "错误", message)
                    
                else:
                    self.rebuild_faces_btn.setEnabled(True)
                    self.rebuild_faces_btn.setText("重建选中人物人脸")
            else:
                self.rebuild_faces_btn.setEnabled(True)
                self.rebuild_faces_btn.setText("重建选中人物人脸")
                
        except Exception:
            self.rebuild_faces_btn.setEnabled(True)
            self.rebuild_faces_btn.setText("重建选中人物人脸")

    def batch_delete_selected(self) -> None:
        """Delete all checked actors via batch endpoint."""
        ids = self._checked_actor_ids()
        if not ids:
            QMessageBox.warning(self, "提示", "请先勾选要删除的人物")
            return

        reply = QMessageBox.question(
            self,
            "确认批量删除",
            f"确定要删除选中的 {len(ids)} 个人物吗？\n（不会影响视频文件，仅清除关联）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            resp = requests.delete(
                f"{self.api_base}/actors/batch",
                json={"actor_ids": ids},
                timeout=30,
            )
            if resp.status_code == 200:
                QMessageBox.information(self, "成功", f"已删除 {len(ids)} 个人物")
                self.load_actors()
            else:
                QMessageBox.critical(self, "错误", f"删除失败: {resp.text}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"删除失败: {exc}")

    def merge_selected(self) -> None:
        ids = self._selected_actor_ids()
        if len(ids) < 2:
            QMessageBox.warning(self, "提示", "请至少选择 2 个人物进行合并")
            return

        # 确定"主记录"并**固定顺序**：勾选集合是 set（顺序随机），若不排序，
        # 后端拿到的第一个 id 每次都可能不同 → 合并结果不可预期
        # （历史 bug：人工重命名的那个若不是主记录，名字会随源记录一起消失）。
        # 规则：优先已命名的人物（命名更可能是有意保留的），否则取 id 最小者。
        by_id = {a.get("id"): a for a in (getattr(self.actor_grid, "actors", None) or [])}

        def _display_name(aid: int) -> str:
            a = by_id.get(aid) or {}
            code = actor_code_display(a.get("code")) or f"id={aid}"
            label = (a.get("primary_name") or a.get("name") or "").strip()
            return f"{code}（{label}）" if label else code

        def _is_named(aid: int) -> bool:
            a = by_id.get(aid) or {}
            return bool((a.get("primary_name") or a.get("name") or "").strip())

        named = sorted(i for i in ids if _is_named(i))
        target = (named or sorted(ids))[0]
        ordered = [target] + [i for i in sorted(ids) if i != target]

        reply = QMessageBox.question(
            self,
            "确认合并",
            f"确定要将选中的 {len(ids)} 个人物合并为一个吗？\n\n"
            f"主记录: {_display_name(target)}\n"
            "其余人物的名称与别名会一并保留（不会丢失）。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            resp = requests.post(
                f"{self.api_base}/actors/merge",
                json={"actor_ids": ordered},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = f"合并完成（主记录 {_display_name(target)}）"
                _missing = data.get("missing_files") or 0
                if _missing:
                    msg += f"\n\n注意: {_missing} 条人脸记录的图片文件已缺失，记录已保留但可能无图。"
                QMessageBox.information(self, "成功", msg)
                # 合并后清空持久勾选(涉及的人物 id 已变化, 避免残留)
                self.actor_grid.clear_selection()
                self._selected_actor_ids_set = set()
                self.load_actors()
            else:
                QMessageBox.critical(self, "错误", f"合并失败: {resp.text}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"合并失败: {exc}")


# ── 拆分人脸对话框 ───────────────────────────────────────────────────

class _FaceThumb(QLabel):
    """拆分对话框的可点人脸缩略图。

    左键点 = 翻转选中;Shift+点 = 从锚点连选到此处(由对话框处理)。
    大目标交互:点图本身,不必瞄准小勾选框。"""

    clicked = Signal(int)        # cell idx(普通点击,携带锚点语义)
    shift_clicked = Signal(int)  # cell idx(Shift 连选)

    def __init__(self, idx: int, parent=None):
        super().__init__(parent)
        self._idx = idx

    def mousePressEvent(self, event):  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton:
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self.shift_clicked.emit(self._idx)
            else:
                self.clicked.emit(self._idx)
        super().mousePressEvent(event)


class _SplitFacesDialog(QDialog):
    """手动拆分簇对话框：展示人物全部人脸（可勾选），拆分为新人物。"""

    def __init__(self, parent, faces: list, api_base: str):
        super().__init__(parent)
        self.api_base = api_base
        self.faces = faces
        self.face_checkboxes = {}  # face_id -> QCheckBox
        self._cells = []           # cell idx -> face_id(faces 顺序,Shift 连选用)
        self._thumbs = {}          # cell idx -> _FaceThumb(选中态描边刷新)
        self._anchor = None        # Shift 连选锚点(上次点击的 cell idx)
        self.setWindowTitle("拆分簇 - 选择要拆分的人脸")
        self.resize(720, 560)
        t = theme()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)  # ui-lint:allow 遗留债（对话框批清零）

        # 提示
        hint = QLabel(
            f"共 {len(faces)} 张人脸。勾选要拆分到新人物的人脸，\n"
            "未勾选的人脸保留在当前人物。\n"
            "点人脸图=选中/取消；Shift+点=从上次点击处连选到此处。"
        )
        hint.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: 12px;")  # ui-lint:allow 遗留债（对话框批清零）
        layout.addWidget(hint)

        # 滚动网格
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(8)  # ui-lint:allow 遗留债（对话框批清零）

        cols = 6
        for idx, face in enumerate(faces):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(2, 2, 2, 2)  # ui-lint:allow 遗留债（对话框批清零）
            cell_layout.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）
            self._cells.append((idx, face["id"]))

            # 人脸缩略图（可点：点图=翻转选中;Shift+点=连选;选中态描边）
            img_label = _FaceThumb(idx)
            img_label.setFixedSize(80, 80)  # ui-lint:allow 遗留债（对话框批清零）
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setCursor(Qt.CursorShape.PointingHandCursor)
            img_label.setToolTip("点图=选中/取消;Shift+点=从上次点击处连选")
            self._thumbs[idx] = img_label
            face_path = face.get("face_path", "")
            if face_path and Path(face_path).exists():
                pixmap = self._load_image(face_path)
                if pixmap and not pixmap.isNull():
                    img_label.setPixmap(pixmap.scaled(
                        80, 80,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    ))
            else:
                img_label.setText("缺失")
            img_label.setStyleSheet(self._thumb_style(False))
            img_label.clicked.connect(self._on_thumb_clicked)
            img_label.shift_clicked.connect(self._on_thumb_shift_clicked)
            cell_layout.addWidget(img_label, alignment=Qt.AlignmentFlag.AlignCenter)

            # 勾选框（状态与点图互通）
            cb = QCheckBox(f"#{face.get('id')}")
            cb.setStyleSheet(f"font-size: 10px; color: {t.get('text_secondary')};")  # ui-lint:allow 遗留债（对话框批清零）
            self.face_checkboxes[face["id"]] = cb
            cb.stateChanged.connect(lambda _s, i=idx: self._refresh_thumb_style(i))
            cell_layout.addWidget(cb, alignment=Qt.AlignmentFlag.AlignCenter)

            row, col = divmod(idx, cols)
            grid.addWidget(cell, row, col, Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        # 底部按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        select_all_btn = QPushButton("全选")
        select_all_btn.setStyleSheet(button_secondary_style())
        select_all_btn.clicked.connect(
            lambda: [cb.setChecked(True) for cb in self.face_checkboxes.values()]
        )
        btn_row.addWidget(select_all_btn)

        clear_btn = QPushButton("清空")
        clear_btn.setStyleSheet(button_secondary_style())
        clear_btn.clicked.connect(self._on_clear_selection)
        btn_row.addWidget(clear_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(button_secondary_style())
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        split_btn = QPushButton("拆分为新人物")
        split_btn.setStyleSheet(button_primary_style())
        split_btn.clicked.connect(self._do_split)
        btn_row.addWidget(split_btn)
        layout.addLayout(btn_row)

    def _thumb_style(self, checked: bool) -> str:
        from .theme import theme
        t = theme()
        if checked:
            return (f"border: 2px solid {t.get('accent')}; border-radius: 4px; "  # ui-lint:allow 遗留债（对话框批清零）
                    f"background: {t.get('bg_secondary')};")
        return (f"border: 1px solid {t.get('border_light')}; border-radius: 4px; "  # ui-lint:allow 遗留债（对话框批清零）
                f"background: {t.get('bg_secondary')}; "
                f"color: {t.get('text_tertiary')}; font-size: 10px;")  # ui-lint:allow 遗留债（对话框批清零）

    def _refresh_thumb_style(self, idx: int) -> None:
        thumb = self._thumbs.get(idx)
        if thumb is None or idx >= len(self._cells):
            return
        fid = self._cells[idx][1]
        cb = self.face_checkboxes.get(fid)
        if cb is not None:
            thumb.setStyleSheet(self._thumb_style(cb.isChecked()))

    def _refresh_all_thumb_styles(self) -> None:
        for idx, fid in self._cells:
            cb = self.face_checkboxes.get(fid)
            if cb is not None:
                self._thumbs[idx].setStyleSheet(self._thumb_style(cb.isChecked()))

    def _on_thumb_clicked(self, idx: int) -> None:
        """点图 = 翻转该张选中,并记为 Shift 连选锚点。"""
        fid = self._cells[idx][1]
        cb = self.face_checkboxes[fid]
        cb.setChecked(not cb.isChecked())
        self._anchor = idx
        self._refresh_thumb_style(idx)

    def _on_thumb_shift_clicked(self, idx: int) -> None:
        """Shift+点 = 从锚点连选到此处(只勾选不取消,宁多勿丢)。"""
        if self._anchor is None:
            self._on_thumb_clicked(idx)
            return
        lo, hi = min(self._anchor, idx), max(self._anchor, idx)
        for i in range(lo, hi + 1):
            self.face_checkboxes[self._cells[i][1]].setChecked(True)
        self._anchor = idx
        self._refresh_all_thumb_styles()

    def _on_clear_selection(self) -> None:
        for cb in self.face_checkboxes.values():
            cb.setChecked(False)
        self._anchor = None
        self._refresh_all_thumb_styles()

    def _load_image(self, path):
        """加载图片（OpenCV 优先）。"""
        try:
            import cv2
            from PySide6.QtGui import QImage
            img_bgr = cv2.imread(str(path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(q_img)
        except Exception:
            pass
        return QPixmap(path)

    def get_selected_face_ids(self) -> list:
        return [fid for fid, cb in self.face_checkboxes.items() if cb.isChecked()]

    def _do_split(self):
        selected = self.get_selected_face_ids()
        if not selected:
            QMessageBox.warning(self, "提示", "请先勾选要拆分的人脸。")
            return
        reply = QMessageBox.question(
            self, "确认拆分",
            f"确定把选中的 {len(selected)} 张人脸拆分为新人物吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            import requests as _req
            resp = _req.post(
                f"{self.api_base}/api/actors/{self._current_actor_id()}/split",
                json={"face_ids": selected},
                timeout=10,
            )
            if resp.status_code == 200:
                self.accept()
            else:
                QMessageBox.critical(self, "错误", f"拆分失败: {resp.text[:200]}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"拆分失败: {exc}")

    def _current_actor_id(self):
        """从父对话框取当前 actor_id。"""
        parent = self.parent()
        if hasattr(parent, "actor_id"):
            return parent.actor_id
        return 0


class _ClickableCard(QFrame):
    """可点击的卡片（用于"点这张人脸 → 设为封面"）。

    QFrame 本身不发点击信号，这里补一个。
    """

    clicked = Signal()

    def mousePressEvent(self, event):  # noqa: N802 - Qt 命名
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ActorDetailDialog(QDialog):
    """Modal dialog showing actor info, collaborators, and face images."""

    def __init__(
        self,
        api_base: str,
        actor_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.api_base = api_base
        self.actor_id = actor_id
        self.api_client = APIClient(base_url=api_base)
        self._display_code = ""  # _load_detail 里赋值（设封面确认框引用）
        self.setWindowTitle("人物详情")

        self.resize(800, 750)
        self._build_ui()
        self._load_detail()
        # 作品列表加载涉及工作线程（AsyncLoader），必须在事件循环运行后发起
        # （__init__ 阶段直发会在 show 时触发 QThread 生命周期 abort，实测）
        QTimer.singleShot(0, lambda: self._load_actor_videos(1))

    def _load_image_with_opencv(self, image_path: str) -> Optional[QPixmap]:
        """使用OpenCV加载JPEG图片并转换为QPixmap（Qt原生不支持JPEG）"""
        import cv2
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        # 转换为QImage
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 转换为QPixmap
        pixmap = QPixmap.fromImage(q_img)
        return pixmap if not pixmap.isNull() else None

    def _build_ui(self) -> None:
        t = theme()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)  # ui-lint:allow 遗留债（对话框批清零）

        # ── 主标题层级：人物编号为主标题 ────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(16)  # ui-lint:allow 遗留债（对话框批清零）
        self.cover_label = QLabel()
        self.cover_label.setFixedSize(96, 96)  # ui-lint:allow 遗留债（对话框批清零）
        self.cover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_label.setStyleSheet(
            f"border: 1px solid {t.get('border_light')}; "
            f"border-radius: {t.get('radius_lg')}px; "
            f"background: {t.get('bg_secondary')};"
        )
        header.addWidget(self.cover_label)

        info_col = QVBoxLayout()
        info_col.setSpacing(4)  # ui-lint:allow 遗留债（对话框批清零）
        self.code_label = QLabel()
        self.code_label.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {t.get('text_primary')};"  # ui-lint:allow 遗留债（对话框批清零）
        )
        self.code_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.name_label = QLabel()
        self.name_label.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; color: {t.get('text_secondary')};"
        )
        meta_row = QHBoxLayout()
        meta_row.setSpacing(16)  # ui-lint:allow 遗留债（对话框批清零）
        self.created_label = QLabel()
        self.created_label.setStyleSheet(
            f"font-size: {t.get('font_size_sm')}px; color: {t.get('text_secondary')};"
        )
        self.quality_label = QLabel()
        self.quality_label.setStyleSheet(
            f"font-size: {t.get('font_size_sm')}px; color: {t.get('text_secondary')};"
        )
        meta_row.addWidget(self.created_label)
        meta_row.addWidget(self.quality_label)
        meta_row.addStretch()
        info_col.addWidget(self.code_label)
        info_col.addWidget(self.name_label)
        info_col.addLayout(meta_row)
        info_col.addStretch()
        header.addLayout(info_col, 1)
        layout.addLayout(header)

        # ── 3.2 人脸图片区（主要特征）──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.faces_container = QWidget()
        self.faces_layout = QVBoxLayout(self.faces_container)
        self.faces_layout.setSpacing(12)  # ui-lint:allow 遗留债（对话框批清零）

        # 主要特征标题
        self.primary_face_title = QLabel("主要特征")
        self.primary_face_title.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; font-weight: 600;"
        )
        self.faces_layout.addWidget(self.primary_face_title)

        # 主要特征封面
        self.primary_face_label = QLabel()
        self.primary_face_label.setFixedSize(120, 120)  # ui-lint:allow 遗留债（对话框批清零）
        self.primary_face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.primary_face_label.setStyleSheet(
            f"border: 1px solid {t.get('border_light')}; "
            f"border-radius: {t.get('radius_lg')}px; "
            f"background: {t.get('bg_secondary')};"
        )
        self.faces_layout.addWidget(self.primary_face_label)

        # 相似特征来源标题
        self.similar_faces_title = QLabel("相似特征来源")
        self.similar_faces_title.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; font-weight: 600;"
        )
        self.faces_layout.addWidget(self.similar_faces_title)

        # 相似特征网格
        self.similar_faces_grid = QGridLayout()
        self.similar_faces_grid.setSpacing(8)  # ui-lint:allow 遗留债（对话框批清零）
        self.faces_layout.addLayout(self.similar_faces_grid)

        # 全部人脸标题（含计数：张数 / 来源视频数）
        # 说明: "相似特征来源"每个源文件只展示 1 张代表脸; 合并两个同源 actor 时
        # 界面看起来"一张都没多"，易被误认为丢了脸。此处列出全部人脸以便核对。
        self.all_faces_title = QLabel("全部人脸")
        self.all_faces_title.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; font-weight: 600;"
        )
        self.faces_layout.addWidget(self.all_faces_title)

        self.all_faces_grid = QGridLayout()
        self.all_faces_grid.setSpacing(6)  # ui-lint:allow 遗留债（对话框批清零）
        self.faces_layout.addLayout(self.all_faces_grid)

        self.faces_layout.addStretch()

        scroll.setWidget(self.faces_container)
        layout.addWidget(scroll, 1)

        # ── 3.3 合作人物区 ──────────────────────────────────────────
        self.collaborators_title = QLabel("合作人物 (0 人)")
        self.collaborators_title.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; font-weight: 600;"
        )
        self.faces_layout.addWidget(self.collaborators_title)

        self.collaborators_container = QWidget()
        self.collaborators_layout = QGridLayout(self.collaborators_container)
        self.collaborators_layout.setSpacing(8)  # ui-lint:allow 遗留债（对话框批清零）
        self.faces_layout.addWidget(self.collaborators_container)

        # ── 3.4 作品列表区 ─────────────────────────────────────────
        self.videos_title = QLabel("作品列表 (0 个)")
        self.videos_title.setStyleSheet(
            f"font-size: {t.get('font_size_md')}px; font-weight: 600;"
        )
        self.faces_layout.addWidget(self.videos_title)

        self.video_grid = VideoGridWidget(api_client=self.api_client, page_size=20)
        self.video_grid.setMinimumHeight(200)
        self.video_grid.setMaximumHeight(400)
        self.video_grid.play_requested.connect(self._play_video)
        self.video_grid.pagination.page_changed.connect(self._load_actor_videos)
        self.video_grid.data_loaded.connect(self._update_videos_title)
        self.faces_layout.addWidget(self.video_grid)

        # ── 底部横排动词按钮 ────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.rename_btn = QPushButton("重命名")
        self.rename_btn.setStyleSheet(button_secondary_style())
        self.rename_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rename_btn.clicked.connect(self._rename_actor)
        btn_row.addWidget(self.rename_btn)

        self.split_btn = QPushButton("拆分簇")
        self.split_btn.setStyleSheet(button_secondary_style())
        self.split_btn.setToolTip("把部分人脸拆分为新的人物（手动拆分簇）")
        self.split_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.split_btn.clicked.connect(self._split_faces)
        btn_row.addWidget(self.split_btn)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.setStyleSheet(button_danger_style())
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.clicked.connect(self._delete_actor)
        btn_row.addWidget(self.delete_btn)
        layout.addLayout(btn_row)

    # ── Data loading ───────────────────────────────────────────────

    def load_actor(self, actor_id: int) -> None:
        """动态切换人物，重新加载详情"""
        if actor_id == self.actor_id:
            return  # 已经是当前人物，无需重新加载
        self.actor_id = actor_id
        self._load_detail()

    def _load_detail(self) -> None:
        t = theme()
        try:
            resp = requests.get(
                f"{self.api_base}/actors/{self.actor_id}", timeout=10
            )
            if resp.status_code == 404:
                # 幽灵卡:列表刷新前该人物已被拆分/删除——自愈而非死胡同报错
                QMessageBox.information(
                    self, "人物已不存在",
                    "该人物已不存在（可能已被拆分或删除），列表将自动刷新。")
                _p = self.parent()
                if hasattr(_p, "load_actors"):
                    _p.load_actors()
                self.close()
                return
            if resp.status_code != 200:
                QMessageBox.critical(self, "错误", "加载人物详情失败")
                return
            actor = resp.json()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"加载失败: {exc}")
            return

        # 基本信息（编号展示层转文案：ACTOR_122 → 人物 122）
        self._display_code = actor_code_display(actor.get("code", ""))
        self.code_label.setText(self._display_code)
        primary_name = actor.get("primary_name")
        name = primary_name or actor.get("name") or "(未命名)"
        self.name_label.setText(name)
        # 显示别名信息
        aliases = actor.get("aliases", [])
        tooltip_parts = [f"名称: {name}"]
        if primary_name and actor.get("name"):
            tooltip_parts.append(f"原始名: {actor['name']}")
        if aliases:
            tooltip_parts.append(f"别名: {', '.join(aliases)}")
        self.name_label.setToolTip("\n".join(tooltip_parts))
        created = actor.get("created_at") or ""
        if created and "T" in created:
            created = created[:10]
        self.created_label.setText(f"创建时间: {created}")

        # 人脸质量指标（face-quality-filter）：优先用质量接口（有动态 fallback）
        try:
            qr = requests.get(
                f"{self.api_base}/api/actors/{self.actor_id}/quality", timeout=10
            )
            if qr.status_code == 200:
                qdata = qr.json()
                avg_sharpness = qdata.get("avg_face_sharpness")
                if avg_sharpness:
                    if avg_sharpness >= 300:
                        level_text, level_color = "高", t.get("success")
                    elif avg_sharpness >= 150:
                        level_text, level_color = "中", t.get("warning")
                    else:
                        level_text, level_color = "低", t.get("danger")
                    self.quality_label.setText(
                        f"人脸质量: {avg_sharpness:.0f}（{level_text}）"
                    )
                    self.quality_label.setStyleSheet(
                        f"font-size: {t.get('font_size_sm')}px; color: {level_color};"
                    )
                else:
                    self.quality_label.setText("人脸质量: 暂无数据")
        except Exception:
            self.quality_label.setText("人脸质量: 暂无数据")

        # Cover
        cover_path = actor.get("cover_face")
        if cover_path and Path(cover_path).exists():
            pixmap = self._load_image_with_opencv(cover_path)
            if pixmap and not pixmap.isNull():
                pixmap = pixmap.scaled(
                    96, 96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.cover_label.setPixmap(pixmap)

        # ── 合作人物区 ──────────────────────────────────────────────
        # 清空现有合作人物
        while self.collaborators_layout.count() > 0:
            item = self.collaborators_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        collaborators = actor.get("collaborators", [])
        self.collaborators_title.setText(f"合作人物 ({len(collaborators)} 人)")

        if not collaborators:
            empty_label = QLabel("暂无合作记录")
            empty_label.setStyleSheet(
                f"font-size: {t.get('font_size_sm')}px; color: {t.get('text_tertiary')};"
            )
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.collaborators_layout.addWidget(empty_label, 0, 0, 1, 6)
        else:
            for idx, collab in enumerate(collaborators):
                card = self._create_collaborator_card(collab, t)
                row, col = divmod(idx, 6)
                self.collaborators_layout.addWidget(card, row, col)

        # ── 人脸图片区 ──────────────────────────────────────────────
        # 主要特征
        faces = actor.get("faces", {})
        primary_face = faces.get("primary")
        if primary_face:
            face_path = primary_face.get("face_path", "")
            if face_path and Path(face_path).exists():
                pixmap = self._load_image_with_opencv(face_path)
                if pixmap and not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        120, 120,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.primary_face_label.setPixmap(pixmap)
                    
                    # Tooltip
                    yaw = primary_face.get("yaw", 0) or 0
                    pitch = primary_face.get("pitch", 0) or 0
                    roll = primary_face.get("roll", 0) or 0
                    eyes_open = primary_face.get("eyes_open_score", 0) or 0
                    eyes_status = "👁️ 睁眼" if eyes_open > 0.5 else "😑 闭眼"
                    tooltip = f"Yaw: {yaw:.1f}°  Pitch: {pitch:.1f}°  Roll: {roll:.1f}°\n{eyes_status} ({eyes_open:.2f})"
                    self.primary_face_label.setToolTip(tooltip)
        else:
            self.primary_face_label.setText("无")
            self.primary_face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 清空现有相似特征
        while self.similar_faces_grid.count() > 0:
            item = self.similar_faces_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # 相似特征来源（同样可点击设为封面，覆盖"只有 1 张脸"的场景）
        similar_faces = faces.get("similar", [])
        for idx, face in enumerate(similar_faces):
            card = self._create_similar_face_card(
                face, t, on_click=self._set_face_as_cover,
                is_cover=bool(face.get("is_cover")),
            )
            row, col = divmod(idx, 6)
            self.similar_faces_grid.addWidget(card, row, col)

        # ── 全部人脸（含计数）──────────────────────────────────────
        for grid in (self.all_faces_grid,):
            while grid.count() > 0:
                item = grid.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

        all_faces = faces.get("all", [])
        total = faces.get("total", len(all_faces))
        file_count = faces.get("file_count", 0)
        self.all_faces_title.setText(
            f"全部人脸 ({total} 张，来自 {file_count} 个视频) — 点击任意一张可设为封面"
        )
        # 全部人脸可能与"相似特征"重复展示；仅在数量多于相似特征时补充，
        # 避免同一个文件只有 1 张脸时出现无意义的重复区块。
        if total > len(similar_faces):
            self.all_faces_title.setVisible(True)
            for idx, face in enumerate(all_faces):
                card = self._create_similar_face_card(
                    face, t, on_click=self._set_face_as_cover,
                    is_cover=bool(face.get("is_cover")),
                )
                row, col = divmod(idx, 8)
                self.all_faces_grid.addWidget(card, row, col)
        else:
            self.all_faces_title.setVisible(False)

        # ── 作品列表 ──────────────────────────────────────────────
        self.video_grid.load_by_actor(self.actor_id, page=1)

    def _update_videos_title(self, total: int) -> None:
        """更新作品列表标题"""
        self.videos_title.setText(f"作品列表 ({total} 个)")

    def _load_actor_videos(self, page: int) -> None:
        """加载人物作品列表（分页）"""
        self.video_grid.load_by_actor(self.actor_id, page=page)

    def _create_collaborator_card(self, collab: dict, t: dict) -> QWidget:
        """创建合作人物卡片"""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_secondary')}; "
            f"border: 1px solid {t.get('border_light')}; "
            f"border-radius: {t.get('radius_sm')}px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(4, 4, 4, 4)  # ui-lint:allow 遗留债（对话框批清零）
        card_layout.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）

        # 封面
        cover_label = QLabel()
        cover_label.setFixedSize(60, 60)  # ui-lint:allow 遗留债（对话框批清零）
        cover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cover_label.setStyleSheet(
            f"border: 1px solid {t.get('border_light')}; "
            f"border-radius: {t.get('radius_sm')}px;"
        )
        
        cover_path = collab.get("cover_face")
        if cover_path:
            full_path = Path(cover_path)
            if not full_path.exists():
                from src.core.path_manager import get_path_manager
                pm = get_path_manager()
                full_path = pm.base_dir / cover_path.replace("data/", "")
            
            if full_path.exists():
                pixmap = self._load_image_with_opencv(str(full_path))
                if pixmap and not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        60, 60,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    cover_label.setPixmap(pixmap)
        
        card_layout.addWidget(cover_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 名称
        name = collab.get("name") or "(未命名)"
        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-size: {t.get('font_size_xs')}px;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        card_layout.addWidget(name_label)

        # 合作次数
        count = collab.get("collaboration_count", 0)
        count_label = QLabel(f"{count} 次合作")
        count_label.setStyleSheet(
            f"font-size: {t.get('font_size_xs')}px; color: {t.get('text_tertiary')};"
        )
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(count_label)

        # Tooltip - 合作作品列表
        files = collab.get("collaboration_files", [])
        if files:
            file_names = []
            for f in files:
                if isinstance(f, dict):
                    path = f.get("file_path") or f.get("file_md5", "")
                    file_names.append(path.split("/")[-1] if path else "")
                else:
                    file_names.append(str(f))
            file_names = [n for n in file_names if n]
            if file_names:
                tooltip = "合作作品:\n" + "\n".join(file_names)
                card.setToolTip(tooltip)

        return card

    def _create_similar_face_card(self, face: dict, t: dict,
                                  on_click=None, is_cover: bool = False) -> QWidget:
        """创建人脸卡片。

        on_click: 传入回调则卡片可点击（用于「把这 张脸设为人物封面」）。
        is_cover: 是否为当前封面 → 强调描边 + 「封面」角标。
        """
        card = _ClickableCard() if on_click is not None else QFrame()
        _border = (f"2px solid {t.get('accent')}" if is_cover
                   else f"1px solid {t.get('border_light')}")
        card.setStyleSheet(
            f"QFrame {{ background: {t.get('bg_secondary')}; "
            f"border: {_border}; "
            f"border-radius: {t.get('radius_sm')}px; }}"
        )
        if on_click is not None:
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.clicked.connect(lambda f=face: on_click(f))

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(4, 4, 4, 4)  # ui-lint:allow 遗留债（对话框批清零）
        card_layout.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）

        # 封面角标
        if is_cover:
            cover_badge = QLabel("封面")
            cover_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cover_badge.setStyleSheet(
                f"font-size: {t.get('font_size_xs')}px; font-weight: 600; "
                f"color: {t.get('text_on_accent')}; "
                f"background: {t.get('accent')}; "
                f"border-radius: 3px; padding: 1px 4px;"  # ui-lint:allow 遗留债（对话框批清零）
            )
            card_layout.addWidget(cover_badge, alignment=Qt.AlignmentFlag.AlignCenter)

        # 人脸图
        face_label = QLabel()
        face_label.setFixedSize(60, 60)  # ui-lint:allow 遗留债（对话框批清零）
        face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        face_label.setStyleSheet(
            f"border: 1px solid {t.get('border_light')}; "
            f"border-radius: {t.get('radius_sm')}px;"
        )
        
        face_path = face.get("face_path", "")
        if face_path and Path(face_path).exists():
            pixmap = self._load_image_with_opencv(face_path)
            if pixmap and not pixmap.isNull():
                pixmap = pixmap.scaled(
                    60, 60,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                face_label.setPixmap(pixmap)
        
        card_layout.addWidget(face_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 文件名（截断）
        file_name = face.get("file_name", "")
        if not file_name:
            file_path = face.get("file_path", "")
            file_name = Path(file_path).name if file_path else "未知"
        display_name = file_name[:10] + "..." if len(file_name) > 10 else file_name
        name_label = QLabel(display_name)
        name_label.setStyleSheet(f"font-size: {t.get('font_size_xs')}px;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(name_label)

        # Tooltip - 完整文件名 + 角度信息 + 睁眼状态
        yaw = face.get("yaw", 0) or 0
        pitch = face.get("pitch", 0) or 0
        roll = face.get("roll", 0) or 0
        eyes_open = face.get("eyes_open_score", 0) or 0
        eyes_status = "👁️ 睁眼" if eyes_open > 0.5 else "😑 闭眼"
        tooltip = f"文件: {file_name}\nYaw: {yaw:.1f}°  Pitch: {pitch:.1f}°  Roll: {roll:.1f}°\n{eyes_status} ({eyes_open:.2f})"
        if is_cover:
            tooltip += "\n\n✅ 当前封面"
        elif on_click is not None:
            tooltip += "\n\n点击可设为该人物的封面"
        card.setToolTip(tooltip)

        return card

    # ── Actions ────────────────────────────────────────────────────

    def _set_face_as_cover(self, face: dict) -> None:
        """把指定人脸设为该人物封面（点击人脸卡片触发）。

        后端会同步 actor_faces.is_cover 与 actors.cover_face（单一事实来源），
        成功后重新拉取详情，使「主要特征」「封面角标」同步更新。
        """
        face_id = face.get("id")
        if not face_id:
            QMessageBox.warning(self, "无法设置", "这张人脸缺少记录 ID，无法设为封面。")
            return
        if face.get("is_cover"):
            return  # 已经是封面

        reply = QMessageBox.question(
            self,
            "设为封面",
            f"将这张人脸设为 {self._display_code or f'人物 {self.actor_id}'} 的封面吗？\n\n"
            f"来源文件: {face.get('file_name') or '未知'}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            resp = requests.post(
                f"{self.api_base}/actors/{self.actor_id}/cover",
                json={"face_id": face_id},
                timeout=15,
            )
            if resp.status_code == 200:
                self._load_detail()
                self._load_actor_videos(1)
            else:
                try:
                    _detail = resp.json().get("detail", "")
                except Exception:
                    _detail = resp.text
                QMessageBox.critical(self, "设置失败", _detail or f"HTTP {resp.status_code}")
        except Exception as exc:
            QMessageBox.critical(self, "设置失败", str(exc))

    def _rename_actor(self) -> None:
        current = self.name_label.text().replace("名称: ", "")
        if current == "(未命名)":
            current = ""

        new_name, ok = QInputDialog.getText(
            self, "重命名", "请输入新名称:",
            text=current,
        )
        if not ok:
            return

        try:
            resp = requests.put(
                f"{self.api_base}/actors/{self.actor_id}/rename",
                json={"name": new_name},
                timeout=10,
            )
            if resp.status_code == 200:
                QMessageBox.information(self, "成功", "重命名完成")
                self._load_detail()
            else:
                QMessageBox.critical(self, "错误", f"重命名失败: {resp.text}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"重命名失败: {exc}")

    def _delete_actor(self) -> None:
        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除此人物吗？\n（不会影响视频文件，仅清除关联）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            resp = requests.delete(
                f"{self.api_base}/actors/{self.actor_id}", timeout=10
            )
            if resp.status_code == 200:
                QMessageBox.information(self, "成功", "删除完成")
                self.accept()
            else:
                QMessageBox.critical(self, "错误", f"删除失败: {resp.text}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"删除失败: {exc}")

    def _split_faces(self) -> None:
        """手动拆分簇：展示全部人脸，勾选后拆分为新人物。"""
        try:
            resp = requests.get(
                f"{self.api_base}/api/actors/{self.actor_id}/faces", timeout=10
            )
            if resp.status_code != 200:
                QMessageBox.critical(self, "错误", f"加载人脸失败: {resp.text[:200]}")
                return
            data = resp.json()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"加载人脸失败: {exc}")
            return

        faces = data.get("faces", [])
        if not faces:
            QMessageBox.information(self, "提示", "该人物没有可拆分的人脸。")
            return

        dialog = _SplitFacesDialog(self, faces, self.api_base)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_ids = dialog.get_selected_face_ids()
            if selected_ids:
                self._load_detail()
                QMessageBox.information(self, "成功", "拆分完成，已创建新人物。")

    def _play_video(self, file_path: str, frame_number: int = 0) -> None:
        """播放视频：使用 VideoPlayerService，支持从指定帧开始播放。"""
        
        import sys
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "播放失败", f"文件不存在:\n{file_path}")
            return
        
        try:
            # 确保项目根目录在 sys.path 中
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from src.core.video_player import VideoPlayerService
            from src.core.config import ConfigManager
            from src.db.models import get_db_engine, get_session, TagFrameMapping, FileLocation
            
            # 读取播放器配置
            config_manager = ConfigManager()
            player_config = config_manager.get("player", {})
            player_type = player_config.get("type", "系统默认")
            player_path = player_config.get("path", "")
            
            # 创建播放器服务
            player_service = VideoPlayerService(player_type, player_path)
            
            # 从数据库查询时间戳
            timestamp = None
            if frame_number > 0:
                engine = get_db_engine()
                db = get_session(engine)
                try:
                    # 根据视频文件路径找到 entity_id（通过 file_locations 表）
                    location = db.query(FileLocation).filter(
                        FileLocation.file_path == str(path)
                    ).first()
                    
                    
                    if location:
                        mapping = db.query(TagFrameMapping).filter(
                            TagFrameMapping.video_frame_number == frame_number,
                            TagFrameMapping.entity_id == location.entity_id
                        ).first()
                        
                        
                        if mapping and mapping.timestamp:
                            timestamp = mapping.timestamp
                finally:
                    db.close()
            
            
            # 播放视频
            player_service.play(str(path), timestamp)
            # S13 ⑩：选中播放器未安装已自动回退系统默认——把原因告诉用户
            notice = getattr(player_service, "fallback_notice", "")
            if notice:
                QMessageBox.information(self, "播放器提示", notice)

            
        except Exception as e:
            # 出错时回退到系统默认播放器
            print(f"[ERROR] 播放失败，回退到系统默认: {e}")
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    
    def _get_video_fps(self, video_path: str) -> float:
        """获取视频帧率"""
        try:
            import subprocess
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # 解析帧率，格式可能是 "30/1" 或 "30000/1001"
                fps_str = result.stdout.strip()
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    return float(num) / float(den)
                return float(fps_str)
        except Exception:
            pass
        return 30.0  # 默认值
