"""TagTreeWidget — 标签树形列表组件。

展示标签分类树，支持动态加载，包含"全部"选项，
点击标签节点时发出 tag_selected 信号。
"""

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal, QThreadPool
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..components.api_client import APIClient
from ..components.theme import theme
from .tree_row_delegate import TreeCountDelegate

try:
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.tokens import T


# ── 异步加载器 ────────────────────────────────────────────────────────────

from PySide6.QtCore import QRunnable


class _TagLoadSignals(QObject):
    """信号容器，用于跨线程通信"""
    finished = Signal(object)
    error = Signal(str)


class _AsyncLoader(QRunnable):
    """使用 QThreadPool 的异步加载器"""
    
    def __init__(self, api_client, endpoint):
        super().__init__()
        self.api_client = api_client
        self.endpoint = endpoint
        self.signals = _TagLoadSignals()

    def run(self):
        """在线程池中执行 API 调用"""
        try:
            result = self.api_client.get(self.endpoint)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ── 主组件 ────────────────────────────────────────────────────────────────

class TagTreeWidget(QWidget):
    """
    标签树形列表组件。

    特性：
    - 动态加载标签分类树
    - 包含 "全部" 顶级选项
    - 点击标签节点触发 tag_selected 信号
    - 点击 "全部" 触发 all_selected 信号

    信号：
    - tag_selected(str): 点击具体标签时发出，传递 tag_name
    - all_selected(): 点击 "全部" 时发出
    """

    tag_selected = Signal(str)
    all_selected = Signal()

    # 树节点角色
    ROLE_TYPE = Qt.ItemDataRole.UserRole
    ROLE_VALUE = Qt.ItemDataRole.UserRole + 2  # +1 让给委托计数角色（全 app 约定）

    TYPE_ALL = "all"
    TYPE_CATEGORY = "category"
    TYPE_TAG = "tag"

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.api_client = api_client or APIClient()
        self._loader: Optional[_AsyncLoader] = None

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self):
        t = theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(T.TREE_WRAP_X, 0, T.TREE_WRAP_X, T.TREE_WRAP_BOTTOM)
        layout.setSpacing(0)

        # TB3.1 标题行「标 签」：高 28、字 11/500/txt2（同详情页「项 目」）
        title_row = QFrame()
        title_row.setFixedHeight(T.TREE_TITLE_H)
        _th = QHBoxLayout(title_row)
        _th.setContentsMargins(T.TREE_PAD_X - T.TREE_ROW_INSET, 0, T.SP_8, 0)
        _lbl = QLabel("标 签")
        _lbl.setStyleSheet(
            f"font-size: {t.get('font_hint')}px; font-weight: 500; "
            f"color: {t.get('text_secondary')}; letter-spacing: 0.5px;")
        _th.addWidget(_lbl)
        _th.addStretch()
        layout.addWidget(title_row)

        # TB3.3 标签树：行规格走共享委托（18§3.3 单点；分类行无计数，
        # 标签行计数=文件数右对齐），子级缩进 26，装饰关
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(T.TREE_INDENT_PROJECT)
        self.tree.setUniformRowHeights(True)
        self.tree.setAnimated(False)
        self.tree.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.tree.setStyleSheet("QTreeWidget { background: transparent; border: none; }")
        self._delegate = TreeCountDelegate(self.tree)
        self.tree.setItemDelegate(self._delegate)
        self.tree.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.tree.setMouseTracking(True)

        # 默认添加 "全部" 节点
        self._add_all_node()

        self.tree.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

        # 空状态
        self.loading_label = QLabel("加载标签中…")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet(
            f"color: {t.get('text_tertiary')}; padding: {T.SP_12}px;"
        )
        self.loading_label.hide()
        layout.addWidget(self.loading_label)


    def _add_all_node(self):
        """添加 "全部" 顶级节点。"""
        all_item = QTreeWidgetItem(self.tree)
        all_item.setText(0, "全部")
        all_item.setData(0, self.ROLE_TYPE, self.TYPE_ALL)
        all_item.setData(0, self.ROLE_VALUE, None)
        all_item.setExpanded(True)

    # ── 公开方法 ──────────────────────────────────────────────────

    def load_tags(self):
        """从 API 加载标签树（后台线程拉取，台账 B5）。

        原先在主线程同步请求（注释自称"简化版本，先验证数据流"），
        API 慢时首次进入标签浏览页整窗冻结数秒——与 D5 同根因。
        文件内本就有 _AsyncLoader(QRunnable)，改用之。"""
        self.loading_label.show()
        self.tree.hide()

        loader = _AsyncLoader(self.api_client, "/api/tags/tree")
        loader.signals.finished.connect(self._on_tags_loaded)
        loader.signals.error.connect(self._on_tags_error)
        self._loader = loader  # 持引用防 GC
        QThreadPool.globalInstance().start(loader)

    def set_tags(self, tag_tree: Dict[str, List[tuple]]):
        """
        直接设置标签树数据（不走 API）。

        Args:
            tag_tree: {category: [(tag_name, file_count), ...]}
        """
        self._populate_tree(tag_tree)

    def clear_selection(self):
        """清除选择。"""
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        self.tree.blockSignals(False)

    def select_all(self):
        """选中 "全部" 节点。"""
        root = self.tree.topLevelItem(0)
        if root and root.data(0, self.ROLE_TYPE) == self.TYPE_ALL:
            self.tree.setCurrentItem(root)

    # ── 内部方法 ──────────────────────────────────────────────────

    def _on_tags_loaded(self, data):
        """API 返回标签树数据。"""
        
        # API 返回格式: {"categories": [{"name": "人数", "tags": [...]}, ...]}
        categories = data.get("categories", [])
        
        # 转换为前端期望的格式: {category_name: [(tag_name, file_count), ...]}
        tag_tree = {}
        for category in categories:
            cat_name = category.get("name")
            tags = category.get("tags", [])
            tag_tree[cat_name] = [(tag.get("name"), tag.get("file_count", 0)) for tag in tags]
        
        if tag_tree:
            self._populate_tree(tag_tree)
        else:
            self._on_tags_error("标签数据为空")

    def _on_tags_error(self, message: str):
        """API 加载失败（窄侧栏只放一句友好提示，细节走 stdout）。"""
        print(f"[tag_tree] 加载失败: {message}", flush=True)
        brief = ("服务未连接" if ("Connection" in message or "Max retries" in message)
                 else "加载失败")
        self.loading_label.setText(brief)
        self.loading_label.show()
        self.tree.hide()

    def _populate_tree(self, tag_tree: Dict[str, List[tuple]]):
        """构建树结构。
        
        Args:
            tag_tree: {category_name: [(tag_name, file_count), ...]}
        """
        # 保留 "全部" 节点
        self.tree.clear()
        self._add_all_node()

        for category, tags in tag_tree.items():
            cat_item = QTreeWidgetItem(self.tree)
            cat_item.setText(0, category)
            cat_item.setData(0, self.ROLE_TYPE, self.TYPE_CATEGORY)
            cat_item.setData(0, self.ROLE_VALUE, category)
            cat_item.setExpanded(True)

            for tag_name, file_count in tags:
                tag_item = QTreeWidgetItem(cat_item)
                tag_item.setText(0, tag_name)
                # 计数走 UserRole+1，委托右对齐绘制（同任务/详情/标签管理树）
                tag_item.setData(0, Qt.ItemDataRole.UserRole + 1, file_count)
                tag_item.setData(0, self.ROLE_TYPE, self.TYPE_TAG)
                tag_item.setData(0, self.ROLE_VALUE, tag_name)

        self.loading_label.hide()
        self.tree.show()
        self.tree.repaint()

    def _on_item_changed(self, current: Optional[QTreeWidgetItem], previous):
        """树节点选中变化。"""
        if current is None:
            return

        item_type = current.data(0, self.ROLE_TYPE)
        item_value = current.data(0, self.ROLE_VALUE)

        if item_type == self.TYPE_ALL:
            self.all_selected.emit()
        elif item_type == self.TYPE_TAG:
            self.tag_selected.emit(item_value)
        # 点击分类节点时不触发事件，仅展开/折叠
