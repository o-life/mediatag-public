"""分析主题管理对话框。

「分析主题」= 业务主题（标签映射文件 + 提示词 + 人脸资源库），
区别于「外观」（界面深/浅色皮肤）。

左侧主题列表 + 右侧表单：选中任意已有主题 → 字段全部回填 → 修改保存。
支持新增 / 删除 / 设默认 / 载入预置主题。
"""
import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from ..components.theme import theme

logger = logging.getLogger(__name__)


class AnalysisThemeManagerDialog(QDialog):
    """分析主题管理弹窗（仿「VLM 供应商管理」的管理型 CRUD 布局）。"""

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self._themes = {}
        self._current_theme_id = None

        self.setWindowTitle("分析主题管理")
        self.resize(760, 560)
        t = theme()
        self.setStyleSheet(f"background: {t.get('bg_primary')}; color: {t.get('text_primary')};")

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── 左侧：主题列表 ──────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("分析主题"))
        self.theme_list = QListWidget()
        self.theme_list.setStyleSheet(f"""
            QListWidget {{
                background: {t.get('bg_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 4px;
            }}
            QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
            QListWidget::item:selected {{
                background: {t.get('bg_hover')};
                color: {t.get('text_primary')};
            }}
        """)
        self.theme_list.currentItemChanged.connect(self._on_list_selected)
        left.addWidget(self.theme_list, 1)

        self.add_btn = QPushButton("＋ 新增分析主题")
        self.add_btn.setStyleSheet(self._btn_style(t))
        self.add_btn.clicked.connect(self._new_theme)
        left.addWidget(self.add_btn)

        self.seed_btn = QPushButton("载入预置主题")
        self.seed_btn.setStyleSheet(self._btn_style(t))
        self.seed_btn.setToolTip("补建缺失的预置分析主题（幂等，不影响已有主题）")
        self.seed_btn.clicked.connect(self._seed_presets)
        left.addWidget(self.seed_btn)
        root.addLayout(left, 1)

        # ── 右侧：表单 ──────────────────────────────────────────
        form_wrap = QWidget()
        form_wrap.setStyleSheet(
            f"background: {t.get('bg_secondary')}; border-radius: {t.get('radius_md')}px;"
        )
        form = QFormLayout(form_wrap)
        form.setContentsMargins(16, 16, 16, 16)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("如：日常生活 / 成人内容 / 影视")
        self.name_edit.setStyleSheet(self._input_style(t))
        form.addRow("名称:", self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("这个分析主题适用于什么内容")
        self.desc_edit.setStyleSheet(self._input_style(t))
        form.addRow("说明:", self.desc_edit)

        self.default_check = QCheckBox("设为默认分析主题")
        self.default_check.setStyleSheet(f"color: {t.get('text_primary')};")
        form.addRow("", self.default_check)

        self.mapping_edit = QLineEdit()
        self.mapping_edit.setPlaceholderText("标签映射表 .md 路径（切换主题时导入，标签归属该主题）")
        self.mapping_edit.setStyleSheet(self._input_style(t))
        form.addRow("标签映射:", self._with_browse(self.mapping_edit, self._browse_mapping, t))

        # 数据文件与人脸目录由「主题工作区」按主题管理、路径取默认值——
        # 不在此暴露（见 openspec/changes/per-theme-vault design D5）。
        hint = QLabel("数据文件与人脸资源库由该分析主题的工作区独立管理，无需在此设置。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')};")
        form.addRow("内容存储:", hint)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(130)
        self.prompt_edit.setPlaceholderText("该主题的 VLM 提示词（分析时按当前分析主题取用）")
        self.prompt_edit.setStyleSheet(self._input_style(t))
        form.addRow("提示词:", self.prompt_edit)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"font-size: 12px; color: {t.get('text_secondary')};")
        form.addRow("", self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setStyleSheet(self._btn_style(t))
        self.delete_btn.clicked.connect(self._delete_theme)
        btn_row.addWidget(self.delete_btn)
        self.save_btn = QPushButton("保存")
        self.save_btn.setStyleSheet(self._btn_style(t))
        self.save_btn.clicked.connect(self._save_theme)
        btn_row.addWidget(self.save_btn)
        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(self._btn_style(t))
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        form.addRow(btn_row)

        root.addWidget(form_wrap, 2)

        self._refresh()

    # ── 样式 ──────────────────────────────────────────────────
    @staticmethod
    def _btn_style(t) -> str:
        return f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 5px 12px;
            }}
            QPushButton:hover {{ background: {t.get('bg_hover')}; }}
        """

    @staticmethod
    def _input_style(t) -> str:
        return f"""
            background: {t.get('bg_primary')};
            border: 1px solid {t.get('border')};
            padding: 5px 8px;
            border-radius: {t.get('radius_md')}px;
            color: {t.get('text_primary')};
        """

    def _with_browse(self, edit: QLineEdit, handler, t) -> QWidget:
        """给输入框配一个「浏览…」按钮。"""
        wrap = QWidget()
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(edit, 1)
        btn = QPushButton("浏览…")
        btn.setStyleSheet(self._btn_style(t))
        btn.clicked.connect(handler)
        h.addWidget(btn)
        return wrap

    # ── 数据 ──────────────────────────────────────────────────
    def _refresh(self):
        """拉取主题列表并刷新左侧列表。"""
        try:
            data = self.api.get("/themes")
        except Exception as e:
            self.status_label.setText(f"加载失败: {e}")
            return
        themes = data.get("themes", []) or []
        self._themes = {th.get("id"): th for th in themes}

        self.theme_list.blockSignals(True)
        self.theme_list.clear()
        for th in themes:
            label = th.get("name", "")
            if th.get("is_default"):
                label += "  ★默认"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, th.get("id"))
            if th.get("is_default"):
                item.setForeground(QColor(theme().get("accent")))
            self.theme_list.addItem(item)
        self.theme_list.blockSignals(False)

        if not themes:
            self.status_label.setText("暂无分析主题，点「＋ 新增分析主题」或「载入预置主题」")

    def _select_theme(self, theme_id):
        """按 id 选中列表项。"""
        for i in range(self.theme_list.count()):
            if self.theme_list.item(i).data(Qt.ItemDataRole.UserRole) == theme_id:
                self.theme_list.setCurrentRow(i)
                return

    def _on_list_selected(self, current, _previous):
        """选中列表项 → 表单全部回填（用户要求的"可调整已有值"）。"""
        if current is None:
            return
        tid = current.data(Qt.ItemDataRole.UserRole)
        data = self._themes.get(tid)
        if not data:
            return
        self._current_theme_id = tid
        self.name_edit.setText(data.get("name", ""))
        self.desc_edit.setText(data.get("description") or "")
        self.default_check.setChecked(bool(data.get("is_default")))
        self.mapping_edit.setText(data.get("label_mapping_path") or "")
        self.prompt_edit.setPlainText(data.get("prompt") or "")
        self.status_label.setText("")

    def _new_theme(self):
        """新增：清空表单，等待填写。"""
        self._current_theme_id = None
        self.name_edit.setText("")
        self.desc_edit.setText("")
        self.default_check.setChecked(False)
        self.mapping_edit.setText("")
        self.prompt_edit.setPlainText("")
        self.status_label.setText("填写后点「保存」创建新分析主题")
        self.theme_list.clearSelection()

    # ── 动作 ──────────────────────────────────────────────────
    def _save_theme(self):
        """保存表单（新增或更新已有主题）。"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请填写分析主题名称")
            return

        payload = {
            "name": name,
            "description": self.desc_edit.text().strip() or None,
            "label_mapping_path": self.mapping_edit.text().strip() or None,
        }
        try:
            if self._current_theme_id is None:
                resp = self.api.post("/themes", json=payload)
                self._current_theme_id = resp.get("id")

            payload["is_default"] = 1 if self.default_check.isChecked() else 0
            self.api.put(f"/themes/{self._current_theme_id}", json=payload)

            prompt = self.prompt_edit.toPlainText()
            if prompt:
                self.api.put(f"/themes/{self._current_theme_id}/prompt", json={"prompt": prompt})

            self.status_label.setText(f"已保存「{name}」")
            self._refresh()
            self._select_theme(self._current_theme_id)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _delete_theme(self):
        """删除当前主题（默认主题不可删）。"""
        if self._current_theme_id is None:
            QMessageBox.warning(self, "提示", "请先在列表中选择要删除的分析主题")
            return
        data = self._themes.get(self._current_theme_id, {})
        name = data.get("name", "")
        if data.get("is_default"):
            QMessageBox.information(self, "提示", "默认分析主题不可删除，请先把别的主题设为默认")
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除分析主题「{name}」吗？\n该主题下的标签会变为「通用标签」（其他主题仍可见）。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self.api.delete(f"/themes/{self._current_theme_id}")
            self.status_label.setText(f"已删除「{name}」")
            self._current_theme_id = None
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, "删除失败", str(e))

    def _seed_presets(self):
        """载入/补建预置分析主题（幂等）。"""
        try:
            resp = self.api.post("/themes/seed-presets")
            created = resp.get("created", 0)
            self.status_label.setText(
                f"已补建 {created} 个预置主题" if created else "预置主题均已存在，无需补建"
            )
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, "载入失败", str(e))

    # ── 文件浏览 ──────────────────────────────────────────────
    def _browse_mapping(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择标签映射文件", self.mapping_edit.text().strip() or "",
            "Markdown 文件 (*.md)",
        )
        if path:
            self.mapping_edit.setText(path)

