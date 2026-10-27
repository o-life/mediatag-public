"""LabelManagementPage — Label Management widget for ai-video-factory."""


from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QTreeWidget, QTreeWidgetItem, QMessageBox,
    QDialog, QFormLayout, QFrame, QTextEdit, QInputDialog, QFileDialog,
    QCheckBox, QScrollArea, QProgressBar, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject

from .api_client import APIClient
from .theme import (
    theme, button_secondary_style, button_danger_style,
    sm_button_style, sm_combo_style, search_box_style,
)
from ..widgets.flow_layout import FlowLayout
from ..widgets.tree_row_delegate import TreeCountDelegate
from ..widgets.video_grid_widget import PaginationWidget

try:
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.tokens import T


# ── Async data loader ───────────────────────────────────────────────────

class AsyncLoader(QObject):
    """Generic async data loader to prevent UI blocking."""
    finished = Signal(object)  # Emits result data
    error = Signal(str)  # Emits error message
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._thread: QThread | None = None
    
    def run(self):
        """Run the function in a separate thread."""
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        # Keep self alive until thread finishes
        self._thread.finished.connect(self.deleteLater)
        self._thread.start()
    
    def _execute(self):
        """Execute the function and emit signals."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self._thread:
                self._thread.quit()


# ── Dictionary card widget ──────────────────────────────────────────────


class TagCard(QFrame):
    """A single tag card showing name, aliases, and edit button."""

    def __init__(self, name: str, category: str, file_count: int, parent=None):
        super().__init__(parent)
        self.tag_name = name
        self.tag_category = category
        self.setMinimumHeight(50)
        self.setFrameShape(QFrame.StyledPanel)
        t = theme()
        self.setStyleSheet(f"""
            QFrame {{
                background: {t.get('bg_primary')};
                border: 1px solid transparent;
                border-radius: 8px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
            QFrame:hover {{
                border: 1px solid rgba(10,132,255,0.5);
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 满铺零边距（18§4.0）
        layout.setSpacing(8)  # ui-lint:allow 遗留债（对话框批清零）

        # Category badge
        badge = QLabel(f"[{category}]")
        badge.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            color: {t.get('text_secondary')};
            background: {t.get('bg_secondary')};
            padding: 2px 6px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_sm')}px;
        """)
        layout.addWidget(badge)

        # Name
        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            font-size: {t.get('font_size_md')}px;
            font-weight: bold;
            color: {t.get('text_primary')};
        """)
        layout.addWidget(name_label)

        # File count
        count_label = QLabel(f"({file_count} 个文件)")
        count_label.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            color: {t.get('text_secondary')};
        """)
        layout.addWidget(count_label)

        layout.addStretch()

        # Edit button
        edit_btn = QPushButton("编辑别名")
        edit_btn.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            padding: 4px 10px;  # ui-lint:allow 遗留债（对话框批清零）
            background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
            border: none;
            border-radius: {t.get('radius_sm')}px;
        """)
        edit_btn.clicked.connect(self._on_edit)
        layout.addWidget(edit_btn)

    def _on_edit(self):
        # Signal that edit was requested — parent handles via slot
        if hasattr(self, 'edit_signal'):
            self.edit_signal.emit(self.tag_name)

    def set_edit_callback(self, callback):
        self._on_edit = callback


# ── Audit queue item widget ─────────────────────────────────────────────


class AuditItemWidget(QFrame):
    """审核卡（LM4.2）：底 win 色、圆角 8；横向布局=
    勾选框 14 → 原始标签（13/600）+meta 行 → 箭头 → AI 推荐区 → 操作区。

    无匹配卡（ai_rec=="无匹配"）：AI 区显示 txt3 说明，操作区只留 拒绝/删除
    （LM4.3：无匹配无入库语义）。"""

    confirm_signal = None
    reject_signal = None
    manual_signal = None
    delete_signal = None
    selection_changed_signal = None

    def __init__(self, item_id: str, raw_label: str, count: int, ai_rec: str,
                 parent=None):
        super().__init__(parent)
        self.item_id = item_id
        t = theme()

        self.setObjectName("AuditCard")
        self.setStyleSheet(f"""
            QFrame#AuditCard {{
                background: {t.get('bg_primary')};
                border: 1px solid transparent;
                border-radius: {t.get('radius_card')}px;
            }}
            QFrame#AuditCard:hover {{
                border: 1px solid {t.get('border_light')};
            }}
        """)
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)  # ui-lint:allow 遗留债（对话框批清零）
        row.setSpacing(T.SP_8)

        # 勾选框 14（LM4.2）
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        row.addWidget(self.checkbox)

        # 原始标签 + meta 行（LM4.2）
        name_col = QVBoxLayout()
        name_col.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）
        raw_lbl = QLabel(raw_label)
        raw_lbl.setStyleSheet(
            f"font-size: {t.get('font_body')}px; font-weight: 600;"
            f" color: {t.get('text_primary')}; background: transparent;")
        name_col.addWidget(raw_lbl)
        meta_lbl = QLabel(f"出现于 {count} 个文件")
        meta_lbl.setStyleSheet(
            f"font-size: {t.get('font_hint')}px;"
            f" color: {t.get('text_secondary')}; background: transparent;")
        name_col.addWidget(meta_lbl)
        row.addLayout(name_col, 2)

        # 箭头（LM4.2）
        arrow = QLabel("→")
        arrow.setStyleSheet(
            f"color: {t.get('text_tertiary')}; background: transparent;")
        row.addWidget(arrow)

        # AI 推荐区（LM4.3）
        has_rec = bool(ai_rec) and ai_rec != "无匹配"
        if has_rec:
            rec_lbl = QLabel(f"AI 推荐：{ai_rec}")
            rec_lbl.setWordWrap(True)
            rec_lbl.setStyleSheet(
                f"font-size: {t.get('font_secondary')}px;"
                f" color: {t.get('accent')}; background: transparent;")
            row.addWidget(rec_lbl, 2)
        else:
            nom_lbl = QLabel("无匹配——疑似设备名/无意义词")
            nom_lbl.setWordWrap(True)
            nom_lbl.setStyleSheet(
                f"font-size: {t.get('font_secondary')}px;"
                f" color: {t.get('text_tertiary')}; background: transparent;")
            row.addWidget(nom_lbl, 2)

        # 操作区（LM4.4）：入库=accent.14/accent 字；拒绝=danger 字中性底；
        # 无匹配卡无入库
        row.addSpacing(T.SP_8)
        if has_rec:
            self.confirm_btn = QPushButton("入库")
            self.confirm_btn.setCursor(Qt.PointingHandCursor)
            self.confirm_btn.setFixedHeight(T.CTRL_SM_H)
            self.confirm_btn.setStyleSheet(sm_button_style(checked=True) +
                "QPushButton { font-weight: 600; }")
            self.confirm_btn.clicked.connect(self._on_confirm)
            row.addWidget(self.confirm_btn)

        self.reject_btn = QPushButton("拒绝")
        self.reject_btn.setCursor(Qt.PointingHandCursor)
        self.reject_btn.setFixedHeight(T.CTRL_SM_H)
        self.reject_btn.setStyleSheet(button_danger_style())
        self.reject_btn.clicked.connect(self._on_reject)
        row.addWidget(self.reject_btn)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setFixedHeight(T.CTRL_SM_H)
        self.delete_btn.setStyleSheet(button_secondary_style())
        self.delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self.delete_btn)

    def _on_checkbox_changed(self, state):
        if self.selection_changed_signal:
            self.selection_changed_signal()

    def is_selected(self) -> bool:
        return self.checkbox.isChecked()

    def set_selected(self, selected: bool):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(selected)
        self.checkbox.blockSignals(False)

    def _on_confirm(self):
        if self.confirm_signal:
            self.confirm_signal(self.item_id)

    def _on_reject(self):
        if self.reject_signal:
            self.reject_signal(self.item_id)

    def _on_delete(self):
        if self.delete_signal:
            self.delete_signal(self.item_id)


# ── Edit Dialog ─────────────────────────────────────────────────────────


class EditTagDialog(QDialog):
    """Modal dialog for editing a tag's aliases."""

    def __init__(self, tag_name: str, current_aliases: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"编辑标签: {tag_name}")
        self.resize(400, 200)
        t = theme()
        layout = QFormLayout(self)

        self.name_edit = QLineEdit(tag_name)
        self.name_edit.setReadOnly(True)
        self.name_edit.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_md')}px;
            padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
        """)
        layout.addRow("标签名称:", self.name_edit)

        self.alias_text = QTextEdit()  # type: ignore[assignment]
        self.alias_text.setPlainText("\n".join(current_aliases))
        self.alias_text.setPlaceholderText("每个别名一行")
        self.alias_text.setStyleSheet(f"""
            background: {t.get('bg_input')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
        """)
        layout.addRow("别名:", self.alias_text)

        btn_h = QHBoxLayout()
        ok_btn = QPushButton("保存")
        ok_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
            border: none;
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_h.addWidget(ok_btn)
        btn_h.addWidget(cancel_btn)
        layout.addRow(btn_h)

    def get_aliases(self) -> list:
        return [a.strip() for a in self.alias_text.toPlainText().split("\n") if a.strip()]


# ── Add New Tag Dialog ────────────────────────────────────────────────


class AddTagDialog(QDialog):
    """Modal dialog for creating a new dictionary tag with name, category, and aliases."""

    def __init__(self, parent=None, default_category=None, themes=None):
        super().__init__(parent)
        self.setWindowTitle("添加新标签")
        self.resize(450, 350)
        t = theme()
        layout = QFormLayout(self)
        self.default_category = default_category
        self.themes = themes or []  # [{id, name, is_default}]

        # Tag name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入标签名称")
        self.name_edit.setStyleSheet(f"""
            background: {t.get('bg_input')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
        """)
        layout.addRow("标签名称:", self.name_edit)

        # Category - 动态从后端获取，只读下拉
        self.category_combo = QComboBox()
        self.category_combo.setEditable(False)
        self.category_combo.addItem("加载中...")
        self.category_combo.setEnabled(False)
        self.category_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
        """)
        layout.addRow("标签大类:", self.category_combo)

        # Aliases
        self.alias_text = QTextEdit()
        self.alias_text.setPlaceholderText("每行一个别名（可选）")
        self.alias_text.setMaximumHeight(120)
        self.alias_text.setStyleSheet(f"""
            background: {t.get('bg_input')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
        """)
        layout.addRow("别名:", self.alias_text)


        # Buttons
        btn_h = QHBoxLayout()
        ok_btn = QPushButton("添加")
        ok_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
            border: none;
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_h.addWidget(ok_btn)
        btn_h.addWidget(cancel_btn)
        layout.addRow(btn_h)
        
        # 异步加载分类
        self._load_categories()

    def _load_categories(self):
        """从后端动态加载标签分类"""
        try:
            api = APIClient()
            response = api.get("/dictionary/categories")
            categories = response.get("categories", [])
            
            self.category_combo.clear()
            if categories:
                self.category_combo.addItems(categories)
                self.category_combo.setEnabled(True)
                # 如果有预设分类，设置为默认值
                if self.default_category and self.default_category in categories:
                    self.category_combo.setCurrentText(self.default_category)
            else:
                self.category_combo.addItem("其他")
                self.category_combo.setEnabled(True)
        except Exception:
            # 加载失败时使用默认分类
            self.category_combo.clear()
            default_categories = ["场景", "体位", "动作", "人数", "属性", "服装", "视角", "身材", "颜值", "其他"]
            self.category_combo.addItems(default_categories)
            self.category_combo.setEnabled(True)
            # 如果有预设分类，设置为默认值
            if self.default_category and self.default_category in default_categories:
                self.category_combo.setCurrentText(self.default_category)

    def get_data(self) -> dict:
        """Return the tag data entered by the user."""
        return {
            "name": self.name_edit.text().strip(),
            "category": self.category_combo.currentText(),
            "aliases": [
                a.strip()
                for a in self.alias_text.toPlainText().split("\n")
                if a.strip()
            ],
            # 每主题一个数据文件：不再需要 theme_id 归属
        }


# Fix import above


class ConfirmAuditDialog(QDialog):
    """Dialog for confirming an audit item with manual target tag selection."""

    def __init__(self, raw_label: str, ai_recommendation: str, existing_tags: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"确认标签: {raw_label}")
        self.resize(500, 300)
        t = theme()
        layout = QFormLayout(self)

        self.raw_label = raw_label
        self.existing_tags = existing_tags

        # Show raw label
        raw_lbl = QLabel(raw_label)
        raw_lbl.setStyleSheet(f"""
            font-size: {t.get('font_size_md')}px;
            font-weight: bold;
            color: {t.get('text_primary')};
            padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            background: {t.get('bg_secondary')};
            border-radius: {t.get('radius_md')}px;
        """)
        layout.addRow("待审核标签:", raw_lbl)

        # Show AI recommendation
        if ai_recommendation and ai_recommendation != "无匹配":
            rec_lbl = QLabel(ai_recommendation)
            rec_lbl.setStyleSheet(f"""
                font-size: {t.get('font_size_sm')}px;
                color: {t.get('accent')};
                padding: 4px;  # ui-lint:allow 遗留债（对话框批清零）
            """)
            layout.addRow("AI 推荐:", rec_lbl)

        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["添加到现有标签", "创建新标签"])
        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
        """)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addRow("操作模式:", self.mode_combo)

        # Target tag selection (for existing tags)
        self.target_container = QWidget()
        target_layout = QVBoxLayout(self.target_container)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(4)  # ui-lint:allow 遗留债（对话框批清零）
        
        # Search box for filtering tags
        self.target_search = QLineEdit()
        self.target_search.setPlaceholderText("搜索标签...")
        self.target_search.textChanged.connect(self._filter_tags)
        self.target_search.setStyleSheet(f"""
            QLineEdit {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
        """)
        target_layout.addWidget(self.target_search)
        
        # Scrollable tree of tags grouped by category
        self.target_tree = QTreeWidget()
        self.target_tree.setHeaderLabels(["标签名称", "所属大类"])
        self.target_tree.setMaximumHeight(200)
        self.target_tree.setColumnWidth(0, 250)  # ui-lint:allow 遗留债（对话框批清零）
        self.target_tree.setRootIsDecorated(True)
        self.target_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
            }}
            QTreeWidget::item {{
                padding: 0px 8px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
            QTreeWidget::item:selected {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
            }}
            QTreeWidget::item:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        self.target_tree.itemClicked.connect(self._on_tag_selected)
        target_layout.addWidget(self.target_tree)

        self._all_tags = [
            tag if isinstance(tag, dict) else {"name": str(tag), "category": "其他"}
            for tag in existing_tags
        ]
        self._populate_tag_tree(self._all_tags)

        layout.addRow("目标标签:", self.target_container)
        self.selected_target_tag = None

        # Set AI recommendation as default if it exists
        if ai_recommendation and ai_recommendation != "无匹配":
            for i in range(self.target_tree.topLevelItemCount()):
                cat_item = self.target_tree.topLevelItem(i)
                for j in range(cat_item.childCount()):
                    child = cat_item.child(j)
                    if child.text(0) == ai_recommendation:
                        self.target_tree.setCurrentItem(child)
                        self.target_tree.scrollToItem(child)
                        self.selected_target_tag = ai_recommendation
                        break
                if self.selected_target_tag:
                    break

        # New tag name (for creating new tags)
        self.new_tag_edit = QLineEdit()
        self.new_tag_edit.setPlaceholderText("输入新标签名称")
        self.new_tag_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
        """)
        self.new_tag_edit.setVisible(False)
        layout.addRow("新标签名称:", self.new_tag_edit)

        # Category selection (for creating new tags)
        self.category_combo = QComboBox()
        self.category_combo.addItems(["场景", "体位", "动作", "人数", "属性", "服装", "视觉重点", "其他"])
        self.category_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 6px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
        """)
        self.category_combo.setVisible(False)
        layout.addRow("标签维度:", self.category_combo)

        # Buttons
        btn_h = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
            border: none;
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(f"""
            background: {t.get('bg_secondary')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            padding: 8px 20px;  # ui-lint:allow 遗留债（对话框批清零）
            border-radius: {t.get('radius_md')}px;
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_h.addWidget(ok_btn)
        btn_h.addWidget(cancel_btn)
        layout.addRow(btn_h)

    def _populate_tag_tree(self, tags):
        """Populate the tag tree widget grouped by category."""
        self.target_tree.clear()
        categories = {}
        for tag in tags:
            name = tag["name"] if isinstance(tag, dict) else str(tag)
            category = tag.get("category", "其他") if isinstance(tag, dict) else "其他"
            if category not in categories:
                cat_item = QTreeWidgetItem(self.target_tree)
                cat_item.setText(0, category)
                cat_item.setText(1, f"{len([t for t in tags if (t.get('category', '其他') if isinstance(t, dict) else '其他') == category])} 个标签")
                cat_item.setExpanded(True)
                cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
                categories[category] = cat_item
            else:
                cat_item = categories[category]
            child = QTreeWidgetItem(cat_item)
            child.setText(0, name)
            child.setText(1, category)
            child.setData(0, Qt.UserRole, name)

    def _filter_tags(self, text):
        """Filter tag tree based on search text."""
        text = text.strip().lower()
        if not text:
            self._populate_tag_tree(self._all_tags)
        else:
            filtered = [
                t for t in self._all_tags
                if text in (t["name"] if isinstance(t, dict) else str(t)).lower()
            ]
            self._populate_tag_tree(filtered)

    def _on_tag_selected(self, item, column):
        """Handle tag selection from the tree."""
        # Only allow selecting leaf items (actual tags, not category headers)
        if item.childCount() == 0:
            self.selected_target_tag = item.data(0, Qt.UserRole)

    def _on_mode_changed(self, index):
        """Toggle visibility of fields based on mode."""
        is_existing = (index == 0)
        # Hide/show the target container (not the whole dialog!)
        self.target_container.setVisible(is_existing)
        self.new_tag_edit.setVisible(not is_existing)
        self.category_combo.setVisible(not is_existing)
        if not is_existing and not self.new_tag_edit.text():
            self.new_tag_edit.setText(self.raw_label)

    def get_result(self):
        """Return the confirmation result."""
        if self.mode_combo.currentIndex() == 0:
            # Add to existing tag
            return {
                "mode": "add_to_existing",
                "target_tag": self.selected_target_tag or ""
            }
        else:
            # Create new tag
            return {
                "mode": "create_new",
                "tag_name": self.new_tag_edit.text().strip(),
                "category": self.category_combo.currentText()
            }


class BatchAuditProgressDialog(QDialog):
    """Dialog showing real-time progress of batch confirm/reject operations."""

    def __init__(self, items: list, action: str, api: "APIClient", parent=None):
        super().__init__(parent)
        self.items = items
        self.action = action  # "confirm" or "reject" or "delete"
        self.api = api
        self._success_count = 0
        self._fail_count = 0
        self._skip_count = 0

        action_text = {"confirm": "确认", "reject": "拒绝", "delete": "删除"}.get(action, action)
        self.setWindowTitle(f"批量{action_text}进度")
        self.resize(550, 420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        t = theme()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)  # ui-lint:allow 遗留债（对话框批清零）
        layout.setContentsMargins(16, 16, 16, 16)  # ui-lint:allow 遗留债（对话框批清零）

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(items))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                text-align: center;
                background: {t.get('bg_input')};
                color: {t.get('text_primary')};
                height: 24px;  # ui-lint:allow 遗留债（对话框批清零）
            }}
            QProgressBar::chunk {{
                background: {t.get('bg_hover')};
                border-radius: {t.get('radius_md')}px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("准备中...")
        self.status_label.setStyleSheet(f"""
            font-size: {t.get('font_size_sm')}px;
            color: {t.get('text_secondary')};
        """)
        layout.addWidget(self.status_label)

        # Scrollable results log
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                background: {t.get('bg_input')};
            }}
        """)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(8, 8, 8, 8)  # ui-lint:allow 遗留债（对话框批清零）
        self.results_layout.setSpacing(2)  # ui-lint:allow 遗留债（对话框批清零）
        self.results_layout.addStretch()
        self.results_area.setWidget(self.results_widget)
        layout.addWidget(self.results_area, 1)

        # Summary label
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet(f"""
            font-size: {t.get('font_size_sm')}px;
            font-weight: bold;
            color: {t.get('text_primary')};
        """)
        layout.addWidget(self.summary_label)

        # Close button (hidden until done)
        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: none;
                padding: 8px 24px;  # ui-lint:allow 遗留债（对话框批清零）
                border-radius: {t.get('radius_md')}px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setVisible(False)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    # ── Public API ──────────────────────────────────────────────────

    def start_processing(self) -> dict:
        """Call the batch API once, then display per-item results.

        Returns:
            dict with counts: {"success": int, "failed": int, "skipped": int}
        """
        self.close_btn.setVisible(False)

        # Collect item IDs for the batch call
        item_ids = [item["id"] for item in self.items]
        id_to_label = {item["id"]: item["raw_label"] for item in self.items}

        self.status_label.setText(f"正在批量处理 {len(item_ids)} 项...")
        self.progress_bar.setValue(0)

        try:
            response = self.api.post(
                "/audit-queue/batch",
                json={"item_ids": item_ids, "action": self.action},
            )
        except Exception as exc:
            self.status_label.setText("批量请求失败")
            self._add_result_row("(全部)", False, f"请求失败: {exc}")
            self._fail_count = len(self.items)
            self._update_summary()
            self.progress_bar.setValue(len(self.items))
            self.close_btn.setVisible(True)
            return {
                "success": 0,
                "failed": self._fail_count,
                "skipped": 0,
            }

        results = response.get("results", [])

        # 对于批量删除，不显示详细日志，只显示总体进度
        if self.action == "delete":
            # 直接显示完成状态
            self._success_count = sum(1 for r in results if r.get("status") == "deleted")
            self._fail_count = sum(1 for r in results if r.get("status") == "error")
            self._skip_count = sum(1 for r in results if r.get("status") == "skipped")
            
            self.progress_bar.setValue(len(results))
            self._update_summary()
            self.status_label.setText(f"已删除 {self._success_count} 项")
        else:
            # 对于确认和拒绝操作，显示详细日志
            for i, result in enumerate(results):
                item_id = result.get("item_id")
                status = result.get("status", "error")
                message = result.get("message", "")
                raw_label = result.get("raw_label") or id_to_label.get(item_id, str(item_id))

                if status == "confirmed" or status == "rejected":
                    self._success_count += 1
                    ok = True
                elif status == "skipped":
                    self._skip_count += 1
                    ok = False
                    message = "skipped"
                else:
                    self._fail_count += 1
                    ok = False

                self._add_result_row(raw_label, ok, message)
                self.progress_bar.setValue(i + 1)
                self._update_summary()
                self.status_label.setText(
                    f"处理中 ({i + 1}/{len(results)}): {raw_label}"
                )

        self.status_label.setText("处理完成")
        self.close_btn.setVisible(True)
        return {
            "success": self._success_count,
            "failed": self._fail_count,
            "skipped": self._skip_count,
        }

    # ── Internal helpers ────────────────────────────────────────────

    def _add_result_row(self, label: str, ok: bool, message: str):
        """Append one result line to the scrollable log."""
        t = theme()
        if ok:
            icon, color = "✓", t.get("success")
        elif message == "skipped":
            icon, color = "⊘", t.get("warning")
        else:
            icon, color = "✗", t.get("danger")

        row = QLabel(f'{icon}  {label}  — {message}')
        row.setStyleSheet(f"""
            font-size: {t.get('font_size_xs')}px;
            color: {color};
            padding: 2px 4px;  # ui-lint:allow 遗留债（对话框批清零）
        """)

        # Insert before the trailing stretch
        self.results_layout.insertWidget(self.results_layout.count() - 1, row)
        # Auto-scroll to bottom
        self.results_area.verticalScrollBar().setValue(
            self.results_area.verticalScrollBar().maximum()
        )

    def _update_summary(self):
        parts = []
        if self._success_count:
            parts.append(f"成功: {self._success_count}")
        if self._skip_count:
            parts.append(f"跳过: {self._skip_count}")
        if self._fail_count:
            parts.append(f"失败: {self._fail_count}")
        self.summary_label.setText("  |  ".join(parts))


class LabelManagementPage(QWidget):
    """Page 5: Label Management — standard dictionary + audit queue."""

    # 切换分析主题后广播给主窗口，让所有已构建页面刷新（提示词/人物/看板等）
    analysis_theme_changed = Signal(str)

    def __init__(self, api_base: str = "http://localhost:8000", parent=None):
        super().__init__(parent)
        self.api = APIClient(base_url=api_base)
        self._audit_items: list[dict] = []          # cache for batch operations
        self._audit_widgets: list["AuditItemWidget"] = []  # parallel widget list
        self._audit_all: list[dict] = []            # 全量待审项（渲染层分页，LM7）
        self._audit_page: int = 1
        self._audit_page_size: int = 50
        self._audit_count_value = 0                 # LM1.1 计数 N（富文本渲染用）
        self._updating_checkboxes = False            # guard for select-all toggle
        self._loaders = []  # Keep references to async loaders to prevent GC
        self._build_ui()
        # Use async loading to prevent UI blocking
        self._load_themes()
        self._load_dictionary()
        self._load_audit_queue()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        """11 号三栏：中栏=标准标签库面板（MainWindow 经 take_dictionary_panel()
        取走入统一侧栏 212px），本页右栏=审核总览条 + 待审核标签卡片流。
        原两段可收起区与顶部重复按钮已按三栏定稿移除。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        t = theme()
        self.setStyleSheet(f"background-color: {t.get('bg_primary')}; color: {t.get('text_primary')};")

        # ── 中栏面板：标准标签库（交由主窗口统一侧栏承载）──
        # LM3：标题行+搜索行+两级树+底部工具行（原 4 小钮两行收敛）
        self.dict_panel = QWidget()
        self.dict_panel.setStyleSheet(
            f"background: {t.get('bg_secondary')}; color: {t.get('text_primary')};")
        dp = QVBoxLayout(self.dict_panel)
        dp.setContentsMargins(T.TREE_WRAP_X, 0, T.TREE_WRAP_X, T.TREE_WRAP_BOTTOM)
        dp.setSpacing(0)

        # LM3.1 标题行「标签库」：高 28、字 11/500/txt2
        title_row = QFrame()
        title_row.setFixedHeight(T.TREE_TITLE_H)
        _th = QHBoxLayout(title_row)
        _th.setContentsMargins(T.TREE_PAD_X - T.TREE_ROW_INSET, 0, T.SP_8, 0)
        _lbl = QLabel("标签库")
        _lbl.setStyleSheet(
            f"font-size: {t.get('font_hint')}px; font-weight: 500; "
            f"color: {t.get('text_secondary')}; letter-spacing: 0.5px;")
        _th.addWidget(_lbl)
        _th.addStretch()
        dp.addWidget(title_row)

        # LM3.2 搜索框：高 26、上下距 8/6、圆角 6、边 1px line、占位含别名
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 搜索标签或别名")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedHeight(T.CTRL_SM_H)
        self.search_box.textChanged.connect(self._search_dictionary)
        self.search_box.setStyleSheet(search_box_style())
        dp.addSpacing(T.SP_8)
        dp.addWidget(self.search_box)
        dp.addSpacing(T.SP_8 - T.SP_4)  # 搜索与树呼吸 4

        # LM3.5 排序（全部维度/名称）收进搜索框旁一行：维度下拉+排序下拉
        filter_row = QHBoxLayout()
        filter_row.setSpacing(T.SP_4)
        self.category_filter = QComboBox()
        self.category_filter.addItems(["全部维度", "场景", "体位", "动作", "人数", "属性", "服装", "视觉重点"])
        self.category_filter.currentTextChanged.connect(self._filter_by_category)
        self.category_filter.setStyleSheet(sm_combo_style())
        self.category_filter.setFixedHeight(T.CTRL_SM_H)
        filter_row.addWidget(self.category_filter, 1)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["名称排序", "文件数排序"])
        self.sort_combo.currentTextChanged.connect(self._sort_dictionary)
        self.sort_combo.setStyleSheet(sm_combo_style())
        self.sort_combo.setFixedHeight(T.CTRL_SM_H)
        filter_row.addWidget(self.sort_combo, 1)
        dp.addLayout(filter_row)
        dp.addSpacing(T.SP_4)

        def _small_btn(text, tip, slot):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(T.CTRL_SM_H)
            b.setStyleSheet(
                f"QPushButton {{ background: {t.get('btn_glass_bg')};"
                f" color: {t.get('text_secondary')};"
                f" border: 1px solid {t.get('border')};"
                f" border-radius: {t.get('radius_ctrl')}px;"
                f" font-size: {t.get('font_hint')}px; padding: 0 {T.SP_8}px; }}"
                f"QPushButton:hover {{ background: {t.get('btn_glass_hover')};"
                f" color: {t.get('text_primary')}; }}")
            b.clicked.connect(slot)
            return b

        # LM3.4 底部工具行：＋新增/导入/导出小钮+「共 N 个标签」右对齐；
        # 删除收敛进标签右键菜单（防误删整库）——不设按钮
        self.dictionary_list = QTreeWidget()
        self.dictionary_list.setHeaderHidden(True)
        self.dictionary_list.setIndentation(T.TREE_INDENT_PROJECT)  # LM3.3 子级缩进 26
        self.dictionary_list.setRootIsDecorated(False)
        self.dictionary_list.setUniformRowHeights(True)
        self.dictionary_list.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.dictionary_list.setStyleSheet("QTreeWidget { background: transparent; border: none; }")
        self._dict_delegate = TreeCountDelegate(self.dictionary_list)
        self.dictionary_list.setItemDelegate(self._dict_delegate)
        self.dictionary_list.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.dictionary_list.setMouseTracking(True)
        self.dictionary_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dictionary_list.customContextMenuRequested.connect(self._show_context_menu)
        self.dictionary_list.itemDoubleClicked.connect(self._edit_tag)
        self.dictionary_list.itemClicked.connect(self._toggle_category)
        dp.addWidget(self.dictionary_list, 1)

        # LM3.4 底部工具行：顶 1px line
        footer_row = QFrame()
        footer_row.setFixedHeight(T.TREE_TITLE_H)
        footer_row.setStyleSheet(
            f"border-top: 1px solid {t.get('border')}; background: transparent;")
        _fh = QHBoxLayout(footer_row)
        _fh.setContentsMargins(0, 0, 0, 0)
        _fh.setSpacing(T.SP_4)
        self.add_tag_btn = _small_btn("＋新增", "新增标准标签", self._add_new_tag)
        _fh.addWidget(self.add_tag_btn)
        self.import_dict_btn = _small_btn("导入", "导入标准库", self._import_dictionary)
        _fh.addWidget(self.import_dict_btn)
        self.export_btn = _small_btn("导出", "导出标准库", self._export_dictionary)
        _fh.addWidget(self.export_btn)
        _fh.addStretch()
        self.dict_footer = QLabel("共 0 个标签")
        self.dict_footer.setStyleSheet(
            f"font-size: {t.get('font_hint')}px;"
            f" color: {t.get('text_tertiary')}; background: transparent;")
        _fh.addWidget(self.dict_footer)
        dp.addWidget(footer_row)

        # ── 审核操作条：迁移到主窗口顶带（L1/L2）——本页内容区不再放 ──
        # 面板无父级构建，经 take_audit_bar() 交接给顶带 band_stack。
        # LM1：计数+全选+已选+确认入库（次强调）+拒绝+删除/刷新（中性），sm 档
        self._audit_bar = QWidget()
        ab = QHBoxLayout(self._audit_bar)
        ab.setContentsMargins(0, 0, 0, 0)
        ab.setSpacing(T.SP_8)

        # LM1.1 计数：「待审核 N 条」字 12/txt2，N 用 13/600/warn（0 时 txt3）
        self.audit_footer = QLabel("")
        self.audit_footer.setTextFormat(Qt.TextFormat.RichText)
        ab.addWidget(self.audit_footer)

        # LM1.7 无选中时三操作钮 disabled ——顺序：计数|全选 已选|入库 拒绝 删除|刷新
        ab.addSpacing(T.SP_16)

        self.select_all_checkbox = QCheckBox("全选")
        self.select_all_checkbox.toggled.connect(self._select_all_items)
        ab.addWidget(self.select_all_checkbox)

        self.selection_count_label = QLabel("已选 0")
        ab.addWidget(self.selection_count_label)
        ab.addSpacing(T.SP_16)


        def _op_btn(text, slot, danger=False):
            b = QPushButton(text)
            b.setEnabled(False)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(T.CTRL_SM_H)  # LM1.6 sm 档 26
            b.setStyleSheet(button_danger_style() if danger
                            else button_secondary_style())
            b.clicked.connect(slot)
            return b

        # LM1.4 确认入库=次强调：accent.14 底+accent.4 边+accent 字 600
        self.batch_confirm_btn = _op_btn("确认入库", self._batch_confirm)
        self.batch_confirm_btn.setStyleSheet(sm_button_style(checked=True) +
            "QPushButton { font-weight: 600; }")
        ab.addWidget(self.batch_confirm_btn)
        # LM1.5 拒绝=danger 字中性底
        self.batch_reject_btn = _op_btn("拒绝", self._batch_reject, danger=True)
        ab.addWidget(self.batch_reject_btn)
        # LM1.6 删除=中性（原 danger 字降为中性，danger 语义留给单卡拒绝）
        self.batch_delete_btn = _op_btn("删除", self._batch_delete)
        ab.addWidget(self.batch_delete_btn)

        ab.addSpacing(T.SP_16)
        self.audit_refresh_btn = QPushButton("刷新")
        self.audit_refresh_btn.setFixedHeight(T.CTRL_SM_H)
        self.audit_refresh_btn.setStyleSheet(button_secondary_style())
        self.audit_refresh_btn.setCursor(Qt.PointingHandCursor)
        self.audit_refresh_btn.clicked.connect(self._load_audit_queue)
        ab.addWidget(self.audit_refresh_btn)
        ab.addStretch()  # 控件组紧凑靠左（顶带 band_stack 拉伸布局下必须收尾）
        self._style_audit_bar()

        # 卡片流滚动区（LM4.2：卡距 8）
        self.audit_scroll = QScrollArea()
        self.audit_scroll.setWidgetResizable(True)
        # 级联透明：页面级裸 background 声明会渗入滚动区子件盖住面板卡色
        # （QueuePanel 只剩头行可见的根因），子代显式透明放行面板底色
        self.audit_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }")

        self.audit_flow_widget = QWidget()
        self.audit_flow = FlowLayout(self.audit_flow_widget,
                                     margin=T.SP_12, spacing=T.SP_8)
        self.audit_scroll.setWidget(self.audit_flow_widget)

        # LM4.5 空态轻提示（非 hero）：🎉 28 + 标题 15/600 + 副文案 12/txt2
        self.audit_empty_label = QLabel(
            f"<div style='text-align:center;'>"
            f"<span style='font-size:{T.HERO_EMOJI_FS + 0}px;'>🎉</span><br>"
            f"<span style='font-size:{t.get('font_section')}px; font-weight:600;"
            f" color:{t.get('text_primary')};'>待审核队列已清空</span><br>"
            f"<span style='font-size:{t.get('font_secondary')}px;"
            f" color:{t.get('text_secondary')};'>"
            f"分析产出的新标签会先到这里，等你确认后再进入左侧标签库</span></div>")
        self.audit_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.audit_empty_label.setStyleSheet("background: transparent;")
        self.audit_empty_label.hide()
        # 审核队列面板（LM4.1）：底 card 满铺直角 + 头行 32（标题+口径说明）
        qpanel = QWidget()
        qpanel.setObjectName("QueuePanel")
        qpanel.setStyleSheet(
            f"QWidget#QueuePanel {{ background-color: {t.get('bg_card')}; }}")
        qp = QVBoxLayout(qpanel)
        qp.setContentsMargins(0, 0, 0, 0)
        qp.setSpacing(0)
        qhead = QLabel(
            f"&nbsp;&nbsp;<span style='font-size:{t.get('font_secondary')}px;"
            f" font-weight:600; color:{t.get('text_primary')};'>待审核标签</span>"
            f"&nbsp;&nbsp;<span style='font-size:{t.get('font_hint')}px;"
            f" color:{t.get('text_secondary')};'>"
            f"分析产出的新标签先到这里，确认后进入左侧标签库</span>")
        qhead.setTextFormat(Qt.TextFormat.RichText)
        qhead.setFixedHeight(T.HEADER_H)
        qhead.setStyleSheet(
            f"border-bottom: 1px solid {t.get('border')}; background: transparent;")
        qp.addWidget(qhead)
        qp.addWidget(self.audit_empty_label, 1)
        qp.addWidget(self.audit_scroll, 1)
        # LM7 队列分页：待审项可能上千，一次性全量渲染卡曾拖慢整页——
        # 渲染层分页（API 不动），全 app 唯一形态 PaginationWidget
        self.audit_pagination = PaginationWidget(
            default_page_size=self._audit_page_size,
            page_size_options=["20", "50", "100", "200"])
        self.audit_pagination.page_changed.connect(self._on_audit_page_changed)
        self.audit_pagination.page_size_changed.connect(
            self._on_audit_page_size_changed)
        qp.addWidget(self.audit_pagination)
        root.addWidget(qpanel, 1)

    def _style_audit_bar(self):
        """审核条控件样式（LM1；构造与换肤共用一份，避免两处漂移）。"""
        t = theme()
        self.select_all_checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: {t.get('font_secondary')}px;
                color: {t.get('text_secondary')};
            }}
            QCheckBox::indicator {{
                width: {int(T.CHECKBOX_SM)}px;
                height: {int(T.CHECKBOX_SM)}px;
                border: 1.5px solid {t.get('text_secondary')};
                border-radius: {t.get('radius_pill')}px;
                background: {t.get('bg_primary')};
            }}
            QCheckBox::indicator:checked {{
                background: {t.get('accent')};
                border-color: {t.get('accent')};
            }}
        """)
        # LM1.3 已选 N：字 12/txt2
        self.selection_count_label.setStyleSheet(
            f"font-size: {t.get('font_secondary')}px;"
            f" color: {t.get('text_secondary')}; background: transparent;")
        # LM1.4 确认入库=次强调（accent 跟主题）
        self.batch_confirm_btn.setStyleSheet(sm_button_style(checked=True) +
            "QPushButton { font-weight: 600; }")
        # LM1.5 拒绝=danger 字中性底
        self.batch_reject_btn.setStyleSheet(button_danger_style())
        # LM1.6 删除/刷新=中性钮
        self.batch_delete_btn.setStyleSheet(button_secondary_style())
        self.audit_refresh_btn.setStyleSheet(button_secondary_style())
        self._render_audit_count()

    def _render_audit_count(self):
        """LM1.1 计数：「待审核 N 条」12/txt2；N=13/600/warn（0→txt3）。

        只依据 _audit_count_value 渲染（不读当前 text 判断态——空文本
        首渲染会被误判成纯文本态）。加载中/失败态由 setText 直接覆盖，
        下次刷新数据时本方法重写富文本。"""
        t = theme()
        n = self._audit_count_value
        color = t.get('warning') if n else t.get('text_tertiary')
        self.audit_footer.setStyleSheet(
            f"font-size: {t.get('font_secondary')}px;"
            f" color: {t.get('text_secondary')}; background: transparent;")
        self.audit_footer.setText(
            f"待审核 <span style=\"font-size:{t.get('font_body')}px;"
            f" font-weight:600; color:{color};\">{n}</span> 条")

    def restyle_band_bar(self):
        """顶带换肤钩子：审核条已迁出页面，页面级样式表够不到它。"""
        if getattr(self, "_audit_bar", None) is not None:
            self._style_audit_bar()

    def take_audit_bar(self):
        """把审核操作条交给主窗口顶带（重复调用返回 None）。"""
        bar, self._audit_bar = self._audit_bar, None
        return bar

    def take_dictionary_panel(self):
        """把标准标签库面板交给主窗口统一侧栏（11 号三栏）。"""
        return self.dict_panel

    def apply_theme(self):
        """换肤：仅重放样式不重建控件，保留页面状态（LM 全量样式单点）。"""
        t = theme()
        self.setStyleSheet(
            f"background-color: {t.get('bg_primary')}; color: {t.get('text_primary')};")
        # 侧栏面板（实色垫底防透窗背玻璃——D8 教训）
        if self.dict_panel is not None:
            self.dict_panel.setStyleSheet(
                f"background: {t.get('bg_secondary')};"
                f" color: {t.get('text_primary')};")
        self.search_box.setStyleSheet(search_box_style())
        for _c in (self.category_filter, self.sort_combo):
            _c.setStyleSheet(sm_combo_style())
        # 树面板 QSS 不带行样式（行规格走 TreeCountDelegate 单点）
        self.dictionary_list.setStyleSheet(
            "QTreeWidget { background: transparent; border: none; }")
        self.dict_footer.setStyleSheet(
            f"font-size: {t.get('font_hint')}px;"
            f" color: {t.get('text_tertiary')}; background: transparent;")
        # 级联透明：页面级裸 background 声明会渗入滚动区子件盖住面板卡色
        # （QueuePanel 只剩头行可见的根因），子代显式透明放行面板底色
        self.audit_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }")
        # 队列面板/头行
        qpanel = self.findChild(QWidget, "QueuePanel")
        if qpanel is not None:
            qpanel.setStyleSheet(
                f"QWidget#QueuePanel {{ background-color: {t.get('bg_card')}; }}")
        self.audit_pagination.apply_theme()
        self._style_audit_bar()
        self._style_footer_buttons()
        # 重渲染审核卡与队列头行（色值在样式里）
        self._load_dictionary()
        self._load_audit_queue()

    def _style_footer_buttons(self):
        """LM3.4 底部工具行小钮样式（构造/换肤共用）。"""
        t = theme()
        small_qss = (
            f"QPushButton {{ background: {t.get('btn_glass_bg')};"
            f" color: {t.get('text_secondary')};"
            f" border: 1px solid {t.get('border')};"
            f" border-radius: {t.get('radius_ctrl')}px;"
            f" font-size: {t.get('font_hint')}px; padding: 0 {T.SP_8}px; }}"
            f"QPushButton:hover {{ background: {t.get('btn_glass_hover')};"
            f" color: {t.get('text_primary')}; }}")
        for _b in (self.add_tag_btn, self.import_dict_btn, self.export_btn):
            _b.setStyleSheet(small_qss)

    def _toggle_category(self, item, column):
        """Toggle expand/collapse for category items."""
        # Only toggle if it's a category item (has children)
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())

    def _toggle_all_categories(self):
        """Toggle expand/collapse for all categories."""
        # Check if any category is expanded
        any_expanded = False
        for i in range(self.dictionary_list.topLevelItemCount()):
            item = self.dictionary_list.topLevelItem(i)
            if item.isExpanded():
                any_expanded = True
                break
        
        # Toggle all categories
        for i in range(self.dictionary_list.topLevelItemCount()):
            item = self.dictionary_list.topLevelItem(i)
            item.setExpanded(not any_expanded)
        
        # （展开/收起钮已按 LM3 收敛——本方法保留给右键菜单复用）

    def _load_themes(self):
        """Load themes from GET /themes and populate the theme combo (async)."""
        def fetch_themes():
            resp = self.api.get("/themes")
            return resp.get("themes", [])

        loader = AsyncLoader(fetch_themes)
        loader.finished.connect(self._on_themes_loaded)
        loader.error.connect(lambda msg: print(f"加载主题失败: {msg}"))
        self._loaders.append(loader)
        loader.run()

    def _on_themes_loaded(self, themes):
        """Populate theme combo with themes (default first)."""
        self._themes = themes
        # 读取前端配置的当前分析主题, 优先选中它
        active = getattr(self, "_active_analysis_theme", "") or ""
        if not active:
            try:
                cfg = self.api.get("/config") or {}
                if isinstance(cfg, dict):
                    active = ((cfg.get("analysis") or {}).get("active_theme") or "")
            except Exception:
                active = ""
        self._active_analysis_theme = active


        # 标签树只反映当前主题的库（不再有"全部分析主题"过滤维度）
        self._load_dictionary()

    def _current_theme_id(self):
        """当前**生效**分析主题的 id（无则 None）。

        不再有"下拉选择"——主题切换已移至顶栏常驻切换器；这里只反映当前主题。
        """
        active = getattr(self, "_active_analysis_theme", "") or ""
        if not active:
            return None
        for th in getattr(self, "_themes", []) or []:
            if th.get("name") == active:
                return th.get("id")
        return None

    def _load_dictionary(self):
        """Load dictionary from GET /dictionary and populate tree (async)。

        **每主题一个数据文件**之后不再按 theme_id 过滤：当前库里的标签就是本主题的标签
        （旧的多主题共存 + 降级为通用机制已退役）。
        """
        self.dict_footer.setText("加载中...")

        def fetch_dictionary():
            # 获取所有分类（包括空分类）
            categories_response = self.api.get("/dictionary/categories")
            all_categories = categories_response.get("categories", [])

            # 获取标签（本主题库的全部标签）
            tags_response = self.api.get("/dictionary")
            tags = tags_response.get("tags", [])

            return {"categories": all_categories, "tags": tags}
        
        loader = AsyncLoader(fetch_dictionary)
        loader.finished.connect(self._on_dictionary_loaded)
        loader.error.connect(self._on_dictionary_error)
        self._loaders.append(loader)  # Prevent GC
        loader.run()
    
    def _on_dictionary_loaded(self, data):
        """Handle successful dictionary load."""
        all_categories = data.get("categories", [])
        tags = data.get("tags", [])
        
        self.dictionary_list.clear()
        categories = {}
        
        # 先创建所有分类节点（包括空分类）
        for category_name in all_categories:
            cat_item = QTreeWidgetItem(self.dictionary_list)
            cat_item.setText(0, category_name)
            cat_item.setExpanded(True)
            categories[category_name] = cat_item
        
        # 归组 + 按当前排序方式排列（名称 / 文件数）
        grouped: dict = {}
        for tag in tags:
            if isinstance(tag, dict):
                name = tag.get("name", tag.get("label", "未知"))
                category = tag.get("category", tag.get("type", "其他"))
                file_count = tag.get("file_count", tag.get("count", 0)) or 0
                aliases = tag.get("aliases", tag.get("synonyms", []))
            else:
                name, category, file_count, aliases = str(tag), "其他", 0, []
            grouped.setdefault(category, []).append((name, aliases, file_count))

        sort_by = getattr(self, "_dict_sort", "名称排序")
        for items in grouped.values():
            if sort_by == "文件数排序":
                items.sort(key=lambda x: (-x[2], x[0]))
            else:
                items.sort(key=lambda x: x[0])

        for category, items in grouped.items():
            # 如果分类不在列表中，创建它（防止数据不一致）
            if category not in categories:
                cat_item = QTreeWidgetItem(self.dictionary_list)
                cat_item.setText(0, category)
                cat_item.setExpanded(True)
                categories[category] = cat_item
            cat_item = categories[category]
            # LM3.3 分类行计数=类下标签数（UserRole+1 委托右对齐）
            cat_item.setData(0, Qt.ItemDataRole.UserRole + 1, len(items))

            for name, aliases, file_count in items:
                child = QTreeWidgetItem(cat_item)
                child.setText(0, name)
                # 计数走 UserRole+1，委托右对齐绘制（18§3.5 同任务页/详情页）
                child.setData(0, Qt.ItemDataRole.UserRole + 1, file_count)
                # 侧栏宽度有限名称会省略，悬停显示全名
                child.setToolTip(0, name)
                child.setData(0, Qt.ItemDataRole.UserRole, {"name": name, "category": category, "aliases": aliases, "file_count": file_count})
        
        self.dict_footer.setText(f"共 {len(tags)} 个标签")
    
    def _on_dictionary_error(self, error: str):
        """Handle dictionary load error（侧栏底行窄，只放一句友好提示）。"""
        print(f"[dictionary] 加载失败: {error}", flush=True)
        brief = ("服务未连接" if ("Connection" in error or "Max retries" in error)
                 else "加载失败")
        self.dict_footer.setText(brief)

    def _search_dictionary(self, query: str):
        """Client-side filter of dictionary tree."""
        query = query.strip().lower()
        if not query:
            self._load_dictionary()
            return

        # Re-load and filter
        try:
            data = self.api.get("/dictionary")
            tags = data.get("tags", data if isinstance(data, dict) else [])

            matched = [
                t for t in tags
                if isinstance(t, dict) and (
                    query in t.get("name", "").lower()
                    or any(query in a.lower() for a in t.get("aliases", []))
                )
            ]
            if getattr(self, "_dict_sort", "名称排序") == "文件数排序":
                matched.sort(key=lambda t: (-(t.get("file_count") or 0), t.get("name", "")))
            else:
                matched.sort(key=lambda t: t.get("name", ""))

            self.dictionary_list.clear()
            categories = {}

            for tag in matched:
                name = tag.get("name", tag.get("label", ""))
                category = tag.get("category", tag.get("type", "其他"))
                aliases = tag.get("aliases", tag.get("synonyms", []))
                file_count = tag.get("file_count", tag.get("count", 0))

                if category not in categories:
                    cat_item = QTreeWidgetItem(self.dictionary_list)
                    cat_item.setText(0, category)
                    cat_item.setExpanded(True)
                    categories[category] = cat_item
                else:
                    cat_item = categories[category]

                child = QTreeWidgetItem(cat_item)
                child.setText(0, name)
                # 计数走 UserRole+1，委托右对齐绘制（18§3.5 同任务页/详情页）
                child.setData(0, Qt.ItemDataRole.UserRole + 1, file_count)
                # 侧栏宽度有限名称会省略，悬停显示全名
                child.setToolTip(0, name)
                child.setData(0, Qt.ItemDataRole.UserRole, {"name": name, "category": category, "aliases": aliases, "file_count": file_count})

            self.dict_footer.setText(f"搜索 '{query}': {len(matched)} 个标签")
        except Exception as e:
            self.dict_footer.setText(f"搜索失败: {e}")

    def _filter_by_category(self, category: str):
        """Filter dictionary by category."""
        if category == "全部维度":
            self._load_dictionary()
            return
        try:
            data = self.api.get("/dictionary")
            tags = data.get("tags", data if isinstance(data, dict) else [])

            self.dictionary_list.clear()
            filtered = [t for t in tags if isinstance(t, dict) and t.get("category", t.get("type", "")) == category]
            if getattr(self, "_dict_sort", "名称排序") == "文件数排序":
                filtered.sort(key=lambda t: (-(t.get("file_count") or 0), t.get("name", "")))
            else:
                filtered.sort(key=lambda t: t.get("name", ""))

            for tag in filtered:
                name = tag.get("name", tag.get("label", "未知"))
                aliases = tag.get("aliases", tag.get("synonyms", []))
                file_count = tag.get("file_count", tag.get("count", 0))
                child = QTreeWidgetItem(self.dictionary_list)
                child.setText(0, name)
                child.setData(0, Qt.ItemDataRole.UserRole + 1, file_count)
                child.setToolTip(0, name)
                child.setData(0, Qt.ItemDataRole.UserRole, {"name": name, "category": category, "aliases": aliases, "file_count": file_count})

            self.dict_footer.setText(f"[{category}]: {len(filtered)} 个标签")
        except Exception as e:
            self.dict_footer.setText(f"筛选失败: {e}")

    def _sort_dictionary(self, sort_by: str):
        """切换排序方式后重新加载（实际排序在 _on_dictionary_loaded 中生效）。"""
        self._dict_sort = sort_by
        self._load_dictionary()

    def _edit_tag(self, item: QTreeWidgetItem, _column: int):
        """Open edit dialog for a tag."""
        if item.childCount() == 0:
            data = item.data(0, Qt.UserRole)
        else:
            # Category header clicked, find first child
            if item.childCount() > 0:
                child = item.child(0)
                data = child.data(0, Qt.UserRole)
            else:
                return
        if not data or "name" not in data:
            return

        name = data["name"]
        aliases = data.get("aliases", [])

        dialog = EditTagDialog(name, aliases, self)
        if dialog.exec() == QDialog.Accepted:
            new_aliases = dialog.get_aliases()
            try:
                self.api.patch(f"/dictionary/{name}", json={"aliases": new_aliases})
                self._load_dictionary()
                QMessageBox.information(self, "成功", f"标签 '{name}' 已更新。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"更新失败: {e}")

    def _add_new_tag(self):
        """Open dialog to add a new tag with name, category, and aliases."""
        dialog = AddTagDialog(self, themes=getattr(self, "_themes", []))
        if dialog.exec() != QDialog.Accepted:
            return

        data = dialog.get_data()
        if not data["name"]:
            QMessageBox.warning(self, "警告", "请输入标签名称")
            return

        try:
            self.api.post(
                "/dictionary",
                json={
                    "name": data["name"],
                    "category": data["category"],
                    "aliases": data["aliases"],
                    "theme_id": data.get("theme_id"),
                },
            )
            self._load_dictionary()
            QMessageBox.information(
                self,
                "成功",
                f"标签 '{data['name']}' 已添加到 [{data['category']}] 大类。",
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加失败: {e}")

    def _add_new_category(self):
        """Open dialog to add a new category."""
        category_name, ok = QInputDialog.getText(
            self,
            "新增分类",
            "请输入分类名称:",
            QLineEdit.EchoMode.Normal,
            ""
        )
        
        if not ok or not category_name.strip():
            return
        
        category_name = category_name.strip()
        
        # 检查分类是否已存在
        try:
            response = self.api.get("/dictionary/categories")
            categories = response.get("categories", [])
            if category_name in categories:
                QMessageBox.warning(self, "警告", f"分类 '{category_name}' 已存在")
                return
            
            # 创建分类（虚拟的，只需要在创建标签时指定即可）
            self.api.post("/dictionary/categories", json={"name": category_name})
            self._load_dictionary()
            QMessageBox.information(
                self,
                "成功",
                f"分类 '{category_name}' 已创建。\n\n现在可以在该分类下新增标签。"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建分类失败: {e}")

    def _add_new_tag_with_category(self, category_name):
        """Open dialog to add a new tag with pre-selected category."""
        dialog = AddTagDialog(self, default_category=category_name, themes=getattr(self, "_themes", []))
        if dialog.exec() != QDialog.Accepted:
            return

        data = dialog.get_data()
        if not data["name"]:
            QMessageBox.warning(self, "警告", "请输入标签名称")
            return

        try:
            self.api.post(
                "/dictionary",
                json={
                    "name": data["name"],
                    "category": data["category"],
                    "aliases": data["aliases"],
                    "theme_id": data.get("theme_id"),
                },
            )
            self._load_dictionary()
            QMessageBox.information(
                self,
                "成功",
                f"标签 '{data['name']}' 已添加到 [{data['category']}] 大类。",
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加失败: {e}")

    def _delete_tag(self):
        """Delete selected tag from dictionary."""
        item = self.dictionary_list.currentItem()
        if not item:
            QMessageBox.warning(self, "提示", "请先选择要删除的标签")
            return

        # Get tag data
        if item.childCount() > 0:
            QMessageBox.warning(self, "提示", "请选择具体的标签（不是分类）")
            return

        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or "name" not in data:
            return

        tag_name = data["name"]
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除标签 '{tag_name}' 吗？\n\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.api.delete(f"/dictionary/{tag_name}")
            self._load_dictionary()
            QMessageBox.information(self, "成功", f"标签 '{tag_name}' 已删除。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _show_context_menu(self, pos):
        """显示右键菜单"""
        item = self.dictionary_list.itemAt(pos)
        if not item:
            return

        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction

        # 右键所在分类作为「新增标签」默认选中项：
        # 标签行取其 category，分类行取行文本（分类行无 UserRole 数据）
        _d = item.data(0, Qt.ItemDataRole.UserRole)
        context_category = _d.get("category") if _d and "name" in _d else item.text(0)

        menu = QMenu(self)

        # 新增分类
        add_category_action = QAction("新增分类...", self)
        add_category_action.triggered.connect(self._add_new_category)
        menu.addAction(add_category_action)

        # 新增标签（预选右键所在分类，而非对话框下拉首项）
        add_tag_action = QAction("新增标签...", self)
        add_tag_action.triggered.connect(
            lambda: self._add_new_tag_with_category(context_category))
        menu.addAction(add_tag_action)
        
        # 获取标签数据
        data = item.data(0, Qt.ItemDataRole.UserRole)
        
        if data and "name" in data:
            # 点击的是标签项，添加编辑、移动、删除选项
            menu.addSeparator()
            
            # 编辑别名
            edit_action = QAction("编辑别名", self)
            edit_action.triggered.connect(lambda: self._edit_tag(item, 0))
            menu.addAction(edit_action)
            
            # 移动到其他分类
            move_action = QAction("移动到其他分类...", self)
            move_action.triggered.connect(lambda: self._move_tag_to_category(data))
            menu.addAction(move_action)
            
            menu.addSeparator()
            
            # 删除
            delete_action = QAction("删除标签", self)
            delete_action.triggered.connect(lambda: self._delete_tag_by_data(data))
            menu.addAction(delete_action)
        
        menu.exec(self.dictionary_list.mapToGlobal(pos))

    def _move_tag_to_category(self, tag_data):
        """移动标签到其他分类"""
        tag_name = tag_data["name"]
        current_category = tag_data.get("category", "其他")
        
        # 获取所有分类
        try:
            response = self.api.get("/dictionary/categories")
            categories = response.get("categories", [])
        except Exception:
            categories = ["场景", "体位", "动作", "人数", "属性", "服装", "视角", "身材", "颜值", "其他"]
        
        # 创建选择对话框
        new_category, ok = QInputDialog.getItem(
            self,
            "移动标签分类",
            f"将标签 '{tag_name}' 从 [{current_category}] 移动到:",
            categories,
            0,
            False
        )
        
        if not ok or new_category == current_category:
            return
        
        try:
            self.api.patch(f"/dictionary/{tag_name}", json={"category": new_category})
            self._load_dictionary()
            QMessageBox.information(
                self, 
                "成功", 
                f"标签 '{tag_name}' 已从 [{current_category}] 移动到 [{new_category}]。"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"移动失败: {e}")

    def _delete_tag_by_data(self, tag_data):
        """通过标签数据删除标签"""
        tag_name = tag_data["name"]
        
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除标签 '{tag_name}' 吗？\n\n此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            self.api.delete(f"/dictionary/{tag_name}")
            self._load_dictionary()
            QMessageBox.information(self, "成功", f"标签 '{tag_name}' 已删除。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _import_dictionary(self):
        """Import dictionary from an Obsidian .md file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择字典文件", "", "Markdown (*.md);;All Files (*)"
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            result = self.api.post("/dictionary/import", json={"content": content})
            self._load_dictionary()
            QMessageBox.information(
                self, "成功", f"已导入 {result.get('imported', 0)} 个标签。"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")

    def _export_dictionary(self):
        """Export dictionary as Markdown (same format as import — round-trippable)."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出标准库", "标准标签库.md", "Markdown (*.md);;All Files (*)"
        )
        if not filepath:
            return
        try:
            result = self.api.get("/dictionary/export")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result.get("content", ""))
            QMessageBox.information(
                self, "成功", f"已导出 {result.get('count', 0)} 个标签到 {filepath}"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

    # ── Audit queue operations ──────────────────────────────────────

    def _load_audit_queue(self):
        """Load audit queue from GET /audit-queue (async)."""
        self.audit_footer.setText("加载中...")

        def fetch_audit_queue():
            return self.api.get("/audit-queue", params={"status": "pending"})

        loader = AsyncLoader(fetch_audit_queue)
        loader.finished.connect(self._on_audit_queue_loaded)
        loader.error.connect(self._on_audit_queue_error)
        self._loaders.append(loader)  # Prevent GC
        loader.run()

    def _on_audit_queue_loaded(self, data):
        """Handle successful audit queue load."""
        items = data.get("items", data.get("queue", data if isinstance(data, list) else []))
        parsed: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                iid = item.get("id", item.get("item_id", ""))
                raw_label = item.get("raw_label", item.get("label", item.get("raw", "未知")))
                count = item.get("count", item.get("num", 1))
                ai_rec = item.get("ai_recommendation", item.get("recommendation", item.get("suggested_tag", "无匹配")))
            else:
                iid = str(item)
                raw_label = str(item)
                count = 1
                ai_rec = "无推荐"
            parsed.append({"id": iid, "raw_label": raw_label, "count": count, "ai_rec": ai_rec})
        self._audit_all = parsed
        self._render_audit_page()

    def _render_audit_page(self):
        """渲染当前页审核卡（全量在 _audit_all，渲染层分页——LM7）。

        API /audit-queue 无分页参数，保持不动；页面只渲染当前页切片，
        千级待审队列不再一次性建卡拖慢整页。"""
        # 清掉上一页卡片
        while self.audit_flow.count():
            item = self.audit_flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._audit_items = []
        self._audit_widgets = []
        self._updating_checkboxes = True
        self.select_all_checkbox.setChecked(False)
        self._updating_checkboxes = False

        total = len(self._audit_all)
        pages = max(1, (total + self._audit_page_size - 1) // self._audit_page_size)
        self._audit_page = max(1, min(self._audit_page, pages))
        start = (self._audit_page - 1) * self._audit_page_size
        for item in self._audit_all[start:start + self._audit_page_size]:
            iid = item["id"]
            widget = AuditItemWidget(iid, item["raw_label"], item["count"], item["ai_rec"])
            widget.confirm_signal = lambda iid=iid: self._confirm_tag(iid)
            widget.reject_signal = lambda iid=iid: self._reject_tag(iid)
            widget.manual_signal = lambda iid=iid: self._on_manual_select(iid)
            widget.delete_signal = lambda iid=iid: self._delete_audit_item(iid)
            widget.selection_changed_signal = self._on_item_selection_changed
            self._audit_items.append(item)
            self._audit_widgets.append(widget)
            self.audit_flow.addWidget(widget)

        self._audit_count_value = total
        self._render_audit_count()
        empty = total == 0
        self.audit_empty_label.setVisible(empty)
        self.audit_scroll.setVisible(not empty)
        self.audit_pagination.setVisible(not empty)
        self.audit_pagination.update_pagination(
            self._audit_page, total, self._audit_page_size)
        self._update_selection_label()

    def _on_audit_page_changed(self, page: int):
        """队列翻页：渲染目标页切片。"""
        self._audit_page = page
        self._render_audit_page()

    def _on_audit_page_size_changed(self, size):
        """每页条数变化：按页首条目位置换算页码，保持阅读位置不跳顶。"""
        if isinstance(size, str):
            try:
                size = int(size)
            except ValueError:
                return
        if size == self._audit_page_size:
            return
        first = (self._audit_page - 1) * self._audit_page_size
        self._audit_page_size = size
        self._audit_page = first // size + 1
        self._render_audit_page()
    
    def _on_audit_queue_error(self, error: str):
        """Handle audit queue load error.

        顶带计数位只显示一句友好提示（原始异常含 URL/重试细节，又长又吓人——
        用户 2026-09-28 截图指正）；细节留在 stdout 供排查。连接断开时提示
        「无法连接服务」而非裸错误。"""
        print(f"[audit] 队列加载失败: {error}", flush=True)
        brief = ("无法连接服务，请确认后端已启动"
                 if "Connection" in error or "Max retries" in error
                 else "加载失败，请点「刷新」重试")
        self.audit_footer.setText(
            f"<span style=\"color:{theme().get('danger')};\">{brief}</span>")
        self._audit_count_value = 0
        self.audit_empty_label.setVisible(True)
        self.audit_scroll.setVisible(False)
        self.audit_pagination.setVisible(False)

    def _select_all_items(self, checked):
        """Toggle all checkboxes when the 'select all' checkbox changes."""
        if self._updating_checkboxes:
            return
        # toggled signal passes bool value directly
        self._updating_checkboxes = True
        for widget in self._audit_widgets:
            widget.set_selected(checked)
        # Keep _updating_checkboxes True while updating the label
        self._update_selection_label()
        self._updating_checkboxes = False

    def _on_item_selection_changed(self):
        """Called when any individual item checkbox changes."""
        self._update_selection_label()

    def _update_selection_label(self):
        """Update the selection count label and enable/disable batch buttons."""
        selected = self._get_selected_items()
        count = len(selected)
        self.selection_count_label.setText(f"已选择 {count} 项")

        has_selection = count > 0
        self.batch_confirm_btn.setEnabled(has_selection)
        self.batch_reject_btn.setEnabled(has_selection)
        self.batch_delete_btn.setEnabled(has_selection)

        # Sync select-all checkbox state without triggering signal
        all_checked = count == len(self._audit_widgets) and count > 0
        self._updating_checkboxes = True
        self.select_all_checkbox.setChecked(all_checked)
        self._updating_checkboxes = False

    def _get_selected_items(self) -> list:
        """Return audit item dicts for all checked widgets."""
        return [
            item for item, widget in zip(self._audit_items, self._audit_widgets)
            if widget.is_selected()
        ]

    def _get_selected_ids(self) -> list:
        """Return IDs of all checked audit items."""
        return [item["id"] for item in self._get_selected_items()]

    def _confirm_tag(self, item_id: str):
        """Confirm a single audit item via POST /audit-queue/{id}/confirm (uses AI recommendation)."""
        # Find the audit item
        item = next((i for i in self._audit_items if i["id"] == item_id), None)
        if not item:
            return
        
        # If no AI recommendation, use manual select
        if not item["ai_rec"] or item["ai_rec"] == "无匹配":
            self._on_manual_select(item_id)
            return
        
        try:
            self.api.post(f"/audit-queue/{item_id}/confirm")
            self._load_audit_queue()
            QMessageBox.information(self, "成功", f"标签已确认: {item_id}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"确认失败: {e}")

    def _reject_tag(self, item_id: str):
        """Reject a single audit item via POST /audit-queue/{id}/reject."""
        try:
            self.api.post(f"/audit-queue/{item_id}/reject")
            self._load_audit_queue()
            QMessageBox.information(self, "成功", f"标签已拒绝: {item_id}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"拒绝失败: {e}")

    def _delete_audit_item(self, item_id: str):
        """Permanently delete an audit item via DELETE /audit-queue/{id}. No confirmation dialog."""
        try:
            self.api.delete(f"/audit-queue/{item_id}")
            self._load_audit_queue()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _on_manual_select(self, item_id: str):
        """Open dialog for manual target tag selection or new tag creation."""
        if not self._audit_items:
            return
        
        item = next((i for i in self._audit_items if i["id"] == item_id), None)
        if not item:
            return
        
        try:
            # Get existing tags for the dropdown
            data = self.api.get("/dictionary")
            tags = data.get("tags", data if isinstance(data, dict) else [])
            existing_tags = []
            for tag in tags:
                if isinstance(tag, dict):
                    name = tag.get("name", tag.get("label", ""))
                    category = tag.get("category", tag.get("type", "其他"))
                    if name:
                        existing_tags.append({"name": name, "category": category})
            
            dialog = ConfirmAuditDialog(
                raw_label=item["raw_label"],
                ai_recommendation=item["ai_rec"],
                existing_tags=existing_tags,
                parent=self
            )
            
            # Show dialog and wait for user action
            result_code = dialog.exec()
            
            if result_code != QDialog.Accepted:
                return
            
            result = dialog.get_result()
            
            if result["mode"] == "add_to_existing":
                target_tag = result["target_tag"]
                if not target_tag:
                    QMessageBox.warning(self, "警告", "请选择或输入目标标签")
                    return
                self.api.post(f"/audit-queue/{item_id}/confirm", params={"target_tag": target_tag})
                self._load_audit_queue()
                self._load_dictionary()
                QMessageBox.information(self, "成功", f"已将 '{item['raw_label']}' 添加为 '{target_tag}' 的别名")
            
            elif result["mode"] == "create_new":
                tag_name = result["tag_name"]
                category = result["category"]
                if not tag_name:
                    QMessageBox.warning(self, "警告", "请输入新标签名称")
                    return
                # Create new tag first
                self.api.post("/dictionary", json={
                    "name": tag_name,
                    "category": category,
                    "aliases": [item["raw_label"]],
                    "source": "audit"
                })
                # Then confirm the audit item
                self.api.post(f"/audit-queue/{item_id}/confirm", params={"target_tag": tag_name})
                self._load_audit_queue()
                self._load_dictionary()
                QMessageBox.information(self, "成功", f"已创建新标签 '{tag_name}' 并将 '{item['raw_label']}' 添加为别名")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"操作失败: {e}")

    def _batch_confirm(self):
        """Batch confirm selected audit items, showing a progress dialog."""
        selected = self._get_selected_items()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择要确认的审核项")
            return

        if QMessageBox.question(
            self, "确认",
            f"批量确认已选中的 {len(selected)} 条审核项？\n\n"
            "注意: 无 AI 推荐标签的项目将被跳过，需要手动处理。",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        dialog = BatchAuditProgressDialog(selected, "confirm", self.api, self)
        dialog.exec()
        result = dialog.start_processing()

        # Refresh views
        self._load_audit_queue()
        self._load_dictionary()

        # Summary
        parts = [f"成功: {result['success']}"]
        if result["skipped"]:
            parts.append(f"跳过: {result['skipped']} (无 AI 推荐，需手动处理)")
        if result["failed"]:
            parts.append(f"失败: {result['failed']}")
        QMessageBox.information(self, "批量确认完成", "  |  ".join(parts))

    def _batch_reject(self):
        """Batch reject selected audit items, showing a progress dialog."""
        selected = self._get_selected_items()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择要拒绝的审核项")
            return

        if QMessageBox.question(
            self, "确认",
            f"批量拒绝已选中的 {len(selected)} 条审核项？",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        dialog = BatchAuditProgressDialog(selected, "reject", self.api, self)
        dialog.exec()
        result = dialog.start_processing()

        # Refresh views
        self._load_audit_queue()

        # Summary
        parts = [f"成功: {result['success']}"]
        if result["failed"]:
            parts.append(f"失败: {result['failed']}")
        QMessageBox.information(self, "批量拒绝完成", "  |  ".join(parts))

    def _batch_delete(self):
        """Batch delete selected audit items, showing a progress dialog."""
        selected = self._get_selected_items()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择要删除的审核项")
            return

        if QMessageBox.question(
            self, "确认删除",
            f"批量删除已选中的 {len(selected)} 条审核项？\n\n"
            "注意: 删除操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        dialog = BatchAuditProgressDialog(selected, "delete", self.api, self)
        # 先处理，再显示对话框
        result = dialog.start_processing()
        dialog.exec()

        # Refresh views
        self._load_audit_queue()

        # Summary
        parts = [f"成功: {result['success']}"]
        if result["failed"]:
            parts.append(f"失败: {result['failed']}")
        QMessageBox.information(self, "批量删除完成", "  |  ".join(parts))

    def _on_refresh(self):
        """Refresh both dictionary and audit queue."""
        self._load_dictionary()
        self._load_audit_queue()


