"""TagEditDialog — 编辑**单个文件**的标签列表。

为什么只有"整表编辑"这一种操作：
标签需要能定位到对应帧（tag_frame_mappings），单独追加一段文字没有意义
（没有帧归属的标签无法定位），因此不做"添加标签"；"替换标签"同理。
删除是编辑列表的自然结果（把那一行删掉），无需单独功能。

对应后端 POST /api/videos/batch-tags（整表覆盖，接口本身只有这一种语义）。
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QTextEdit
)

try:
    from .theme import theme
    from .api_client import APIClient
except ImportError:
    def theme():
        return {
            'bg_primary': '#1e1e1e',
            'bg_secondary': '#252525',
            'bg_card': '#2a2a2a',
            'text_primary': '#ffffff',
            'text_secondary': '#aaaaaa',
            'accent': '#007aff',
            'border': '#3a3a3a',
            'bg_hover': '#333333',
        }
    APIClient = None


class TagEditDialog(QDialog):
    """编辑一个文件的标签：一行一个标签，确认后整表覆盖。"""

    def __init__(self, analysis_id, tags=None, parent=None):
        """Args:
            analysis_id: FileAnalysis.id
            tags: 该文件当前的标签列表。由调用方从已加载的列表数据传入
                  （后端没有单条 GET /videos/{id}）。
        """
        super().__init__(parent)
        self.analysis_id = analysis_id
        self._tags = list(tags or [])
        self.api = APIClient(base_url="http://localhost:8000") if APIClient else None

        self.setWindowTitle("编辑标签")
        self.setFixedWidth(500)
        self.setMinimumHeight(380)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        t = theme()
        self.setStyleSheet(f"background: {t.get('bg_card')};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel(f"编辑标签（共 {len(self._tags)} 个）")
        title.setStyleSheet(
            f"color: {t.get('text_primary')}; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(title)

        hint = QLabel(
            "一行一个标签。直接增、删、改，确认后以此列表整体覆盖。\n"
            "标签需能定位到对应帧，请使用已有标签（或删掉不需要的）。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: 12px;")
        layout.addWidget(hint)

        self.tags_input = QTextEdit()
        self.tags_input.setPlaceholderText("一行一个标签…")
        self.tags_input.setPlainText("\n".join(self._tags))
        self.tags_input.setStyleSheet(f"""
            QTextEdit {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: 4px;
                padding: 6px;
                font-size: 13px;
            }}
            QTextEdit:focus {{
                border: 1px solid {t.get('accent')};
            }}
        """)
        layout.addWidget(self.tags_input)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedWidth(80)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_hover')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background: {t.get('border')};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.setFixedWidth(80)
        self.confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        self.confirm_btn.clicked.connect(self._save)
        btn_layout.addWidget(self.confirm_btn)

        layout.addLayout(btn_layout)

    def _parsed_tags(self):
        """文本区 → 标签列表（去空行、去首尾空格、去重、保序）。"""
        tags = []
        seen = set()
        for line in self.tags_input.toPlainText().splitlines():
            tag = line.strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags

    def _save(self):
        """整表覆盖该文件的标签。"""
        if not self.api:
            QMessageBox.warning(self, "错误", "API 客户端未初始化")
            return

        tags = self._parsed_tags()
        try:
            resp = self.api.post("/api/videos/batch-tags", json={
                "analysis_ids": [self.analysis_id],
                "tags": tags,
            })
            updated = resp.get("updated_count", 0)
            if updated:
                QMessageBox.information(self, "成功", f"已更新为 {len(tags)} 个标签")
            else:
                QMessageBox.information(self, "无变化", "标签内容未变化")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存标签失败: {str(e)}")
