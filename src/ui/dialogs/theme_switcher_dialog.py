"""分析主题切换器（仿 Obsidian 的 vault switcher）。

为什么要有它：`分析主题`是**工作区级**概念（换库 + 换人脸目录 + 换标签/提示词/人物/任务，
且需重启），原来的入口却藏在「标签管理」页的一个下拉里，看起来像"只是切标签"——
用户明确反馈这种关联性让人迷惑。这里把它抬成**常驻入口**：

- 顶栏常驻「分析主题: X ▾」→ 点开本对话框
- 列表**只显示名称**（路径进 tooltip；路径不对用户暴露）
- 动作：切换 / 新建（从模板）/ 在访达中显示 / 管理主题…
- 切换后**明确提示需重启**（冷切）

决策记录（用户 2026-09-19）：不显示绝对路径、**不**纳管外部已有目录（只保留"新建"）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..app_restart import prompt_restart
from ..components.theme import theme


class ThemeSwitcherDialog(QDialog):
    """主题切换面板。切换成功后发出 `switched(主题名)`。"""

    switched = Signal(str)

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self._themes: list = []
        self._current = ""

        self.setWindowTitle("切换分析主题")
        self.setMinimumSize(460, 420)
        self._build_ui()
        self._reload()

    # ── UI ──────────────────────────────────────────────────────
    def _build_ui(self):
        t = theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hint = QLabel("每个分析主题有独立的数据文件与人脸目录；切换后需重启应用生效。")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: 12px;")
        layout.addWidget(hint)

        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索主题…")
        self.search.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _i: self._switch())
        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.switch_btn = QPushButton("切换到该主题")
        self.switch_btn.setDefault(True)
        self.switch_btn.clicked.connect(self._switch)
        row.addWidget(self.switch_btn)

        self.new_btn = QPushButton("新建（从模板）")
        self.new_btn.clicked.connect(self._create)
        row.addWidget(self.new_btn)

        self.reveal_btn = QPushButton("在访达中显示")
        self.reveal_btn.clicked.connect(self._reveal)
        row.addWidget(self.reveal_btn)

        self.manage_btn = QPushButton("管理主题…")
        self.manage_btn.clicked.connect(self._manage)
        row.addWidget(self.manage_btn)

        row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        row.addWidget(close_btn)
        layout.addLayout(row)

    # ── 数据 ────────────────────────────────────────────────────
    def _reload(self):
        try:
            data = self.api.get("/themes") or {}
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", f"读取主题列表失败：{exc}")
            return
        self._themes = data.get("themes") or []
        self._current = data.get("current") or ""
        self._fill()

    def _fill(self):
        self.list.clear()
        for th in self._themes:
            name = th.get("name") or "未命名"
            label = name + (" ★" if th.get("is_default") else "")
            if th.get("is_current") or name == self._current:
                label = f"✓ {label}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, name)
            # 路径**只进 tooltip**（用户决策：列表只显示名称）
            item.setToolTip(f"路径：{self._theme_dir(th)}")
            self.list.addItem(item)
        self._apply_filter(self.search.text())
        self._update_buttons()

    @staticmethod
    def _theme_dir(th: dict) -> str:
        faces = (th.get("faces_dir") or "").rstrip("/")
        if faces:
            p = Path(faces)
            return str(p.parent if p.name == "faces" else p)
        return "（未指定）"

    def _apply_filter(self, text: str):
        needle = (text or "").strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())
        self._update_buttons()

    def _selected_name(self) -> str:
        item = self.list.currentItem()
        if item is None or item.isHidden():
            return ""
        return item.data(Qt.UserRole) or ""

    def _selected_theme(self) -> dict:
        name = self._selected_name()
        for th in self._themes:
            if th.get("name") == name:
                return th
        return {}

    def _update_buttons(self):
        name = self._selected_name()
        is_current = bool(name) and name == self._current
        self.switch_btn.setEnabled(bool(name) and not is_current)
        self.switch_btn.setText("当前主题" if is_current else "切换到该主题")
        self.reveal_btn.setEnabled(bool(name))

    # ── 动作 ────────────────────────────────────────────────────
    def _switch(self):
        name = self._selected_name()
        if not name or name == self._current:
            return
        th = self._selected_theme()
        reply = QMessageBox.question(
            self, "切换分析主题",
            f"切换到「{name}」？\n\n"
            "将切换：该主题**独立的数据文件 + 人脸目录**（任务/文件/关联/标签/人脸信息）。\n"
            "⚠️ 需重启应用后生效。",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        theme_id = th.get("id")
        if theme_id is None:
            QMessageBox.warning(self, "切换失败", "该主题缺少标识，无法激活")
            return
        try:
            resp = self.api.post(f"/themes/{theme_id}/activate") or {}
        except Exception as exc:
            QMessageBox.warning(self, "切换失败", f"切换分析主题失败：{exc}")
            return
        self._current = name
        self.switched.emit(name)
        if resp.get("restart_required"):
            prompt_restart(self, name)
        self._fill()

    def _create(self):
        name, ok = QInputDialog.getText(self, "新建分析主题", "主题名称：")
        if not ok or not (name or "").strip():
            return
        name = name.strip()
        try:
            self.api.post("/themes", json={"name": name, "description": ""})
        except Exception as exc:
            QMessageBox.warning(self, "新建失败", f"新建主题失败：{exc}")
            return
        self._reload()
        QMessageBox.information(
            self, "已新建",
            f"「{name}」已创建（空白主题：数据文件与人脸目录会在重启后初始化）。\n"
            "如需使用，请切换过去并重启。",
        )

    def _reveal(self):
        path = self._theme_dir(self._selected_theme())
        if not Path(path).exists():
            QMessageBox.information(self, "尚无目录",
                                    f"该主题的目录还没创建（重启后初始化）：\n{path}")
            return
        subprocess.Popen(["open", path])

    def _manage(self):
        """打开完整的管理对话框（增删改/设默认/绑定映射/提示词）。"""
        try:
            from .analysis_theme_dialog import AnalysisThemeManagerDialog

            AnalysisThemeManagerDialog(self.api, self).exec()
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        self._reload()
