"""按项目批量导出 UI（影楼基础包 S4）—— 对话框 + 后台导出线程。

纯逻辑在 core/asset_export.py（可单测）；本模块只做参数收集、线程与进度。

入口：ResultsPage 操作行「导出」→ ExportDialog（范围/命名模板/JPG 规格/
XMP/目标目录）→ run_export()（模态进度 + 取消 + 结果汇总）。
"""

from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
    QVBoxLayout, QWidget,
)

from src.core.asset_export import DEFAULT_NAME_TEMPLATE, export_asset, render_name
from src.ui.tokens import T

JPEG_EDGE_OPTIONS = [("原始尺寸", 0), ("4096px", 4096), ("2048px", 2048),
                     ("1600px", 1600), ("1024px", 1024)]


class ExportDialog(QDialog):
    """导出参数收集：范围 / 命名模板 / 照片规格 / XMP / 目标目录。"""

    def __init__(self, parent, *, checked_count: int, scope_count: int,
                 scope_name: str,
                 default_template: str = DEFAULT_NAME_TEMPLATE):
        super().__init__(parent)
        self.setWindowTitle("按项目导出")
        self.setMinimumWidth(480)
        self._checked_count = checked_count
        self._scope_count = scope_count
        self._scope_name = scope_name

        layout = QVBoxLayout(self)

        # ── 范围 ──
        layout.addWidget(QLabel("导出范围"))
        self.rb_checked = QRadioButton(f"勾选的内容（{checked_count} 个）")
        self.rb_scope = QRadioButton(
            f"当前范围「{scope_name}」的全部（{scope_count} 个）")
        if checked_count > 0:
            self.rb_checked.setChecked(True)
        else:
            self.rb_scope.setChecked(True)
            self.rb_checked.setEnabled(False)
        if scope_count <= 0:
            self.rb_scope.setEnabled(False)
        layout.addWidget(self.rb_checked)
        layout.addWidget(self.rb_scope)

        # ── 命名模板 ──
        form = QFormLayout()
        self.template_edit = QLineEdit(default_template)
        self.template_edit.setFixedHeight(T.CTRL_H)
        form.addRow("命名模板", self.template_edit)
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet(
            f"color: {self._theme().get('text_tertiary')};")
        form.addRow("文件名示例", self.preview_label)
        layout.addLayout(form)
        self.template_edit.textChanged.connect(self._update_preview)
        self._update_preview()

        # ── 照片规格 ──
        layout.addWidget(QLabel("照片规格（视频一律原样复制）"))
        row = QHBoxLayout()
        self.rb_photo_original = QRadioButton("原图")
        self.rb_photo_jpeg = QRadioButton("转码 JPEG")
        self.rb_photo_jpeg.setChecked(True)
        row.addWidget(self.rb_photo_original)
        row.addWidget(self.rb_photo_jpeg)
        self.edge_combo = QComboBox()
        for label, _v in JPEG_EDGE_OPTIONS:
            self.edge_combo.addItem(label)
        self.edge_combo.setCurrentIndex(2)  # 2048px
        self.quality_combo = QComboBox()
        for q in (100, 95, 90, 85, 80):
            self.quality_combo.addItem(f"质量 {q}")
        self.quality_combo.setCurrentIndex(3)  # 85
        row.addWidget(QLabel("最长边"))
        row.addWidget(self.edge_combo)
        row.addWidget(QLabel("质量"))
        row.addWidget(self.quality_combo)
        row.addStretch()
        layout.addLayout(row)

        # ── XMP sidecar ──
        self.xmp_check = QCheckBox(
            "随照片写 XMP sidecar（AI 标签作为关键词，LR/C1 直接可读）")
        self.xmp_check.setChecked(True)
        layout.addWidget(self.xmp_check)

        # ── 目标目录 ──
        drow = QHBoxLayout()
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("选择导出目标文件夹")
        self.dest_edit.setFixedHeight(T.CTRL_H)
        btn_browse = QPushButton("浏览…")
        btn_browse.setFixedHeight(T.CTRL_H)
        btn_browse.clicked.connect(self._browse)
        drow.addWidget(self.dest_edit, 1)
        drow.addWidget(btn_browse)
        layout.addLayout(drow)

        # ── 按钮 ──
        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("导出")
        ok.setFixedHeight(T.CTRL_H)
        ok.setDefault(True)
        ok.clicked.connect(self._on_accept)
        cancel = QPushButton("取消")
        cancel.setFixedHeight(T.CTRL_H)
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    @staticmethod
    def _theme():
        from .theme import theme
        return theme()

    def _update_preview(self):
        from src.core.asset_export import render_name
        name = render_name(self.template_edit.text(),
                           self._scope_name, 1, "IMG_0001.JPG")
        self.preview_label.setText(f"{name}.jpg")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出目标文件夹")
        if d:
            self.dest_edit.setText(d)

    def _on_accept(self):
        if not self.dest_edit.text().strip():
            QMessageBox.warning(self, "缺少目标目录", "请先选择导出目标文件夹")
            return
        self.accept()

    # ── 参数导出 ──

    @property
    def use_checked(self) -> bool:
        return self.rb_checked.isChecked()

    @property
    def name_template(self) -> str:
        return self.template_edit.text().strip() or DEFAULT_NAME_TEMPLATE

    @property
    def photo_mode(self) -> str:
        return "original" if self.rb_photo_original.isChecked() else "jpeg"

    @property
    def longest_edge(self) -> int:
        return JPEG_EDGE_OPTIONS[self.edge_combo.currentIndex()][1]

    @property
    def jpeg_quality(self) -> int:
        return (100, 95, 90, 85, 80)[self.quality_combo.currentIndex()]

    @property
    def write_xmp(self) -> bool:
        return self.xmp_check.isChecked()

    @property
    def dest_dir(self) -> str:
        return self.dest_edit.text().strip()


class ExportWorker(QThread):
    """后台导出线程：逐个资产调 core.export_asset，进度/警告信号回主线程。"""

    progress = Signal(int, int, str)      # done, total, 当前文件名
    warning = Signal(str)
    done = Signal(int, int, int)          # 成功, 跳过(取消), 失败

    def __init__(self, items: List[dict], dest_dir: str, name_template: str,
                 project_name: str, *, photo_mode: str, longest_edge: int,
                 jpeg_quality: int, write_xmp: bool, parent=None):
        super().__init__(parent)
        self._items = items
        self._dest_dir = dest_dir
        self._template = name_template
        self._project_name = project_name
        self._photo_mode = photo_mode
        self._edge = longest_edge
        self._quality = jpeg_quality
        self._xmp = write_xmp
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        dest_dir = Path(self._dest_dir)
        total = len(self._items)
        ok = skip = fail = 0
        for i, item in enumerate(self._items, start=1):
            if self._cancelled:
                skip = total - i + 1
                break
            self.progress.emit(i - 1, total, item.get("file_name") or "")
            try:
                # 命名统一走 core.render_name（清洗非法字符单点，勿绕过）
                base_name = render_name(
                    self._template, self._project_name, i,
                    item.get("file_name") or "")
                export_asset(
                    item.get("file_path") or "", dest_dir,
                    base_name,
                    is_video=(item.get("media_type") == "video"),
                    photo_mode=self._photo_mode,
                    longest_edge=self._edge,
                    jpeg_quality=self._quality,
                    write_xmp=self._xmp,
                    keywords=list(item.get("mapped_tags") or []),
                    project_name=self._project_name,
                )
                ok += 1
            except FileNotFoundError as exc:
                self.warning.emit(str(exc))
                fail += 1
            except Exception as exc:  # 单文件失败不中断整批
                self.warning.emit(f"{item.get('file_name')}: {exc}")
                fail += 1
        self.done.emit(ok, skip, fail)


def run_export(parent: QWidget, items: List[dict], scope_name: str,
               checked_count: int) -> None:
    """导出入口：弹参数对话框 → 模态进度 → 结果汇总。items 为 /videos 字典列表。"""
    if not items and checked_count <= 0:
        QMessageBox.information(parent, "导出", "没有可导出的内容")
        return
    dlg = ExportDialog(parent, checked_count=checked_count,
                       scope_count=len(items), scope_name=scope_name)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    chosen = (parent.selected_videos if dlg.use_checked
              else {v.get("id") for v in items})
    by_id = {v.get("id"): v for v in items}
    export_items = [by_id[i] for i in chosen if i in by_id]
    if not export_items:
        QMessageBox.information(parent, "导出", "没有可导出的内容")
        return

    worker = ExportWorker(
        export_items, dlg.dest_dir, dlg.name_template, scope_name,
        photo_mode=dlg.photo_mode, longest_edge=dlg.longest_edge,
        jpeg_quality=dlg.jpeg_quality, write_xmp=dlg.write_xmp)

    prog = QDialog(parent)
    prog.setWindowTitle("正在导出")
    prog.setMinimumWidth(420)
    v = QVBoxLayout(prog)
    lbl = QLabel("准备中…")
    bar = QProgressBar()
    bar.setRange(0, len(export_items))
    v.addWidget(lbl)
    v.addWidget(bar)

    state = {"ok": 0, "skip": 0, "fail": 0, "warns": []}

    def on_progress(done, total, name):
        bar.setValue(done)
        lbl.setText(f"{done}/{total}　{name}")

    def on_warning(msg):
        state["warns"].append(msg)

    def on_done(ok, skip, fail):
        state.update({"ok": ok, "skip": skip, "fail": fail})
        prog.accept()

    worker.progress.connect(on_progress)
    worker.warning.connect(on_warning)
    worker.done.connect(on_done)
    prog.rejected.connect(worker.cancel)  # 关闭进度窗=取消
    worker.start()
    prog.exec()
    worker.wait()

    msg = f"成功导出 {state['ok']} 个（目标：{dlg.dest_dir}）"
    if state["skip"]:
        msg += f"\n取消跳过 {state['skip']} 个"
    if state["fail"]:
        msg += f"\n失败 {state['fail']} 个"
    if state["warns"]:
        msg += "\n" + "\n".join(state["warns"][:8])
    QMessageBox.information(parent, "导出完成", msg)
