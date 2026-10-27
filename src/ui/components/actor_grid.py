"""Actor grid widget for displaying actor cards in a responsive grid layout."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout,
    QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # 懒导入避免循环依赖，仅类型检查可见
    from .actor_mgmt import ActorCardWidget

from .theme import theme

try:
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.tokens import T


# ── 跨刷新/跨页持久的人脸勾选集合 ─────────────────────────────────────
# 需求: 人脸合并的勾选状态切页/刷新后仍保留。ActorManagementPage 是唯一
# 使用方(单例页), 因此进程级模块变量即可; 勾选以 actor_id 为 key。
# 手动清空(clear_selection)或合并/删除成功后才会真正移除。
_ACTOR_SELECTION: set = set()


class ActorGridWidget(QWidget):
    """
    宫格布局组件，用于展示人物卡片
    
    特性：
    - 响应式布局，根据窗口宽度自动调整每行卡片数
    - 支持卡片点击、选择、批量操作
    - 自动滚动和分页
    """
    
    # 信号定义
    card_clicked = Signal(int)  # 卡片点击，传递 actor_id
    selection_changed = Signal(list)  # 选择变化，传递选中的 actor_id 列表
    annotate_requested = Signal(int, str)  # 标注请求: (actor_id, face_type)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 自定义 QWidget 子类必须启用，否则 QSS background/border 不绘制
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.actors: List[Dict] = []
        self.cards: List[ActorCardWidget] = []
        # 勾选集合指向模块级持久集(跨刷新/跨页保留)
        self.selected_ids: set = _ACTOR_SELECTION

        # 重排防抖 + 结果缓存：resize 里直接重排会让布局失效 → 再次 resize
        # → 再次重排，无限递归卡死主线程（见 _relayout / resizeEvent 注释）。
        self._layout_key = None
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self._relayout)

        self._setup_ui()

    def apply_theme(self):
        """换肤：重放网格衬底背景色并让现有卡片重上色，不重建网格。"""
        from .theme import theme
        t = theme()
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        self.scroll_area.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        self.grid_container.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        # 已创建的卡片也重上色（ActorCardWidget._update_style 读取当前 theme）
        for card in self.cards:
            try:
                card.apply_theme()
            except Exception:
                try:
                    card._update_style()
                except Exception:
                    pass
    
    def _setup_ui(self):
        """初始化 UI"""
        from .theme import theme
        t = theme()
        
        # 网格衬底用 bg_secondary（比页面 bg_primary 略深/略灰），
        # 让白色/亮色卡片在浅灰衬底上成为一个个独立的整体——
        # 上下行卡片之间靠「衬底色差 + 间距 + 卡片边框」三重分隔，
        # 勾选框与所属人脸一眼可辨。
        self.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        
        # 设置 size policy 确保组件可以扩展
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        
        # 安装事件过滤器监听 viewport 的 resize
        self.scroll_area.viewport().installEventFilter(self)
        
        # 创建网格容器
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet(f"background-color: {t.get('bg_card')}; color: {t.get('text_primary')};")
        self.grid_layout = QGridLayout(self.grid_container)
        # 卡片间距：行/列都留出空隙，让每张「勾选框+人脸+文字」的边框
        # 形成独立整体——避免相邻卡片(尤其上下行)贴在一起，勾选框归属被误读
        self.grid_layout.setSpacing(10)  # ui-lint:allow 遗留债（对话框批清零）
        self.grid_layout.setContentsMargins(10, 10, 10, 10)  # ui-lint:allow 遗留债（对话框批清零）
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area)
        
        # 空状态提示
        # FM4.6 空态 hero 轻规格：图标容器 64 + 标题 15/600 + 副文案 12/txt2
        # + 单钮（文案/可见性由 show_empty 配置）
        from .theme import theme
        t = theme()
        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setTextFormat(Qt.TextFormat.RichText)
        self.empty_label.setStyleSheet("background: transparent;")
        self.empty_label.hide()
        layout.addWidget(self.empty_label)
        self.go_tasks_btn = None  # 惰性：show_empty(button=...) 时建
        # go_tasks_requested 是类级 Signal——此处曾被置 None 覆盖信号定义，
        # 主窗口 connect 时报 'NoneType' has no attribute 'connect'（实测）
        
        # 初始化时延迟触发一次布局
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._relayout)
    
    def set_actors(self, actors: List[Dict]):
        """
        设置人物数据并刷新网格

        前端过滤：无封面（cover_face 为空）的人物不展示——
        人脸卡片以封面图为主视觉，无封面卡片只有占位背景，没有展示价值。

        Args:
            actors: 人物数据列表
        """
        # 过滤无封面人物（cover_face 已由服务端做存在性回退，空即无封面）
        self.actors = [a for a in (actors or []) if a.get("cover_face")]
        # 不清空 _ACTOR_SELECTION —— 勾选需跨刷新保留
        self.selected_ids = _ACTOR_SELECTION
        self._refresh_grid()
    
    def _refresh_grid(self):
        """刷新网格布局"""
        # 清空现有卡片
        self._clear_cards()
        
        if not self.actors:
            self.empty_label.show()
            self.scroll_area.hide()
            return
        
        self.empty_label.hide()
        self.scroll_area.show()
        
        # 延迟导入避免循环依赖
        from .actor_mgmt import ActorCardWidget
        
        # 创建新卡片
        for actor in self.actors:
            card = ActorCardWidget(actor)
            card.clicked.connect(self._on_card_clicked)
            card.checked.connect(self._on_card_checked)
            card.annotate_requested.connect(self.annotate_requested.emit)
            # 跨刷新回填勾选状态(持久集里有该人物则恢复勾选)
            if actor.get("id") in _ACTOR_SELECTION:
                card.checkbox.blockSignals(True)
                card.checkbox.setChecked(True)
                card.checkbox.blockSignals(False)
                card._checked = True
                card._update_style()
            self.cards.append(card)
        
        # 延迟重新布局，确保容器宽度已计算
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, self._relayout)
    
    def _clear_cards(self):
        """清空所有卡片"""
        for card in self.cards:
            self.grid_layout.removeWidget(card)
            card.deleteLater()
        self.cards.clear()
        # 卡片已清空 → 让重排缓存在下一轮失效，否则"同宽同数量"时会
        # 误判为"无需重排"而漏排（新卡片永远不进网格 → 界面空白）
        self._layout_key = None
    
    def _relayout(self):
        """根据当前宽度重新布局卡片，让卡片填满整行。

        **可用宽度与卡片数都没变则直接跳过** —— 这是终止递归的关键：重排本身
        会让布局失效并再次触发 resize/事件过滤器，若每次都重排则永不收敛
        （实测：人脸管理页会把主线程 100% 卡死）。
        """
        if not self.cards:
            return

        available_width = self.scroll_area.viewport().width()
        if available_width <= 0:
            return

        layout_key = (available_width, len(self.cards))
        if layout_key == self._layout_key:
            return
        self._layout_key = layout_key

        # 清空当前布局（removeWidget 保留父子关系，避免 setParent 式级联）
        for card in self.cards:
            self.grid_layout.removeWidget(card)

        # FM4.1：列宽≥190 auto-fill、间距 10（内边距 10 由面板 margins 承担）
        ideal_card_width = T.GRID_CARD_MIN_W + T.GRID_GAP
        cards_per_row = max(1, available_width // ideal_card_width)

        spacing = self.grid_layout.spacing()
        card_width = (available_width - spacing * (cards_per_row - 1)) // cards_per_row

        # FM4.2：封面 4:3（勾选框悬浮右上不再占行）；meta padding 8 10 10
        inner_w = card_width  # 卡无内 padding（封面满宽、圆角由容器裁）
        cover_height = int(inner_w * 3 / 4)
        # 卡高 = 封面(4:3) + meta(8 + 名~18 + 2 + 编号~14 + 2 + 出现于~14 + 10)
        card_height = cover_height + 8 + 18 + 2 + 14 + 2 + 14 + 10

        # 设置每张卡片的宽度和高度
        for card in self.cards:
            card.setFixedWidth(card_width)
            card.setFixedHeight(card_height)
            # 封面宽度和高度都跟随卡片（扣除 padding 后的内宽）
            if hasattr(card, 'cover_container'):
                # 容器高度与 cover_height 同源(历史:构造写死 180,卡宽<240 时
                # meta 被顶进封面,角标与名字叠字——任务 0.3 同族几何竞速)
                card.cover_container.setFixedHeight(cover_height)
            if hasattr(card, 'cover_label'):
                card.cover_label.setFixedWidth(inner_w - 2)  # ui-lint:allow -2 边框补偿
                card.cover_label.setFixedHeight(cover_height)
                # 封面尺寸确定后，按新尺寸重绘人脸图（尽量占满封面区）
                if hasattr(card, '_apply_cover_pixmap'):
                    card._apply_cover_pixmap()
            # FM4.2 勾选框悬浮封面右上（跟随卡宽）。
            # 用本循环刚算出的 card_width，不读 cover_container.width()：
            # setFixedWidth 后子件布局是异步激活的，此刻读到的可能是构建
            # 默认宽（640），勾选框会被钉出卡片右侧被裁剪隐形（实测：
            # 搜索重建后 cb_x=618 而卡宽 215，点右上角穿透成开详情）。
            if hasattr(card, 'checkbox'):
                card.checkbox.move(card_width - 14 - 8, 8)
        
        # 重新排列卡片
        for idx, card in enumerate(self.cards):
            row = idx // cards_per_row
            col = idx % cards_per_row
            self.grid_layout.addWidget(card, row, col)
    
    def _calculate_cards_per_row(self) -> int:
        """
        根据窗口宽度计算每行显示的卡片数
        
        Returns:
            每行卡片数
        """
        # 获取可用宽度
        available_width = self.scroll_area.viewport().width() - 32  # 减去左右边距
        
        # 卡片宽度 150px + 间距 16px = 166px
        card_width = 150 + 16
        
        # 计算可以放多少张卡片
        cards_per_row = max(1, available_width // card_width)
        
        # 限制在 3-6 之间
        return min(6, max(3, cards_per_row))
    
    def resizeEvent(self, event):
        """窗口尺寸变化 → 防抖后重排（不在此处直接重排，见 _relayout 注释）。"""
        super().resizeEvent(event)
        self._relayout_timer.start(200)

    def showEvent(self, event):
        """首次显示 → 防抖后重排（短暂延迟，确保容器已拿到真实宽度）。"""
        super().showEvent(event)
        self._relayout_timer.start(10)

    def eventFilter(self, obj, event):
        """监听 viewport 的 resize → 防抖重排。"""
        from PySide6.QtCore import QEvent
        if obj == self.scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self._relayout_timer.start(200)
        return super().eventFilter(obj, event)
    
    def _on_card_clicked(self, actor_id: int):
        """卡片点击事件"""
        self.card_clicked.emit(actor_id)
    
    def _on_card_checked(self, actor_id: int, is_checked: bool):
        """卡片复选框变化事件"""
        if is_checked:
            self.selected_ids.add(actor_id)
        else:
            self.selected_ids.discard(actor_id)

        # 卡片自身已根据 checkbox 状态联动勾选高亮样式（ActorCardWidget._on_checkbox_changed），
        # 此处只维护选中集合，避免重复设置状态。
        self.selection_changed.emit(list(self.selected_ids))
    
    def get_selected_ids(self) -> List[int]:
        """获取选中的人物 ID 列表"""
        return list(self.selected_ids)

    def prune_selection(self, valid_ids) -> None:
        """剔除已不存在的 actor id（跨页持久勾选集合的自动清理）。

        valid_ids = 服务端返回的全量有效 id。任何来源的删除（本页删除 /
        后端识别自动合并 / 重建）都会让该 id 从服务端消失，这里把它从勾选
        集合里收敛掉，避免残留成"幽灵勾选"。调用方须保证 valid_ids 是一次
        成功拉取的结果（拉取失败应传 None 并跳过调用），否则会误清勾选。
        """
        valid = set(valid_ids or [])
        before = len(self.selected_ids)
        self.selected_ids.intersection_update(valid)
        if len(self.selected_ids) != before:
            # 勾选随条目 id（跨页）保留；被删除的 id 由此真正移除，计数同步。
            self.selection_changed.emit(list(self.selected_ids))

    def clear_selection(self):
        """清空选择"""
        self.selected_ids.clear()
        for card in self.cards:
            card.checkbox.setChecked(False)
        self.selection_changed.emit([])
    
    def select_all(self):
        """全选所有卡片"""
        self.selected_ids.clear()
        for card in self.cards:
            card.checkbox.setChecked(True)
            self.selected_ids.add(card.actor_id)
        self.selection_changed.emit(list(self.selected_ids))
    
    def set_loading(self, show: bool):
        """显示/隐藏加载状态"""
        if show:
            self._clear_cards()
            self.scroll_area.hide()
            self.empty_label.setText("加载中...")
            self.empty_label.show()
        else:
            self.empty_label.hide()
            # 如果有数据，显示 scroll_area；如果没有数据，保持隐藏（由 _refresh_grid 处理）
            if self.actors:
                self.scroll_area.show()
    
    def show_empty(self, message: str = "暂无人物数据",
                   hero: bool = False, button_text: str = ""):
        """显示空状态。hero=True 走 FM4.6 轻规格（图标+标题+副文案+单钮）。"""
        self._clear_cards()
        self.scroll_area.hide()
        t = theme()
        if hero:
            sub = ("对素材执行人脸识别后，同一张脸会自动聚合成一个人物，"
                   "出现在这里")
            html = (
                f"<div style='text-align:center;'>"
                f"<span style='font-size:{T.HERO_EMOJI_FS}px;'>👤</span><br>"
                f"<span style='font-size:{t.get('font_section')}px;"
                f" font-weight:600; color:{t.get('text_primary')};'>"
                f"{message}</span><br>"
                f"<span style='font-size:{t.get('font_secondary')}px;"
                f" color:{t.get('text_secondary')};'>{sub}</span></div>")
            self.empty_label.setText(html)
            if button_text and self.go_tasks_btn is None:
                from PySide6.QtWidgets import QPushButton, QVBoxLayout
                from .theme import button_secondary_style
                self.go_tasks_btn = QPushButton(button_text)
                self.go_tasks_btn.setFixedHeight(T.CTRL_H)
                self.go_tasks_btn.setStyleSheet(button_secondary_style())
                self.go_tasks_btn.clicked.connect(self._emit_go_tasks)
                wrap = QWidget()
                _h = QVBoxLayout(wrap)
                _h.setContentsMargins(0, 0, 0, 0)
                _h.addStretch(1)
                _row = QHBoxLayout()
                _row.addStretch(1)
                _row.addWidget(self.go_tasks_btn)
                _row.addStretch(1)
                _h.addLayout(_row)
                self.layout().addWidget(wrap)
                self._empty_btn_wrap = wrap
            if hasattr(self, "_empty_btn_wrap") and self._empty_btn_wrap is not None:
                self._empty_btn_wrap.show()
        else:
            self.empty_label.setText(message)
            if hasattr(self, "_empty_btn_wrap") and self._empty_btn_wrap is not None:
                self._empty_btn_wrap.hide()
        self.empty_label.show()

    go_tasks_requested = Signal()  # FM4.6 空态「去导入任务」

    def _emit_go_tasks(self):
        self.go_tasks_requested.emit()
