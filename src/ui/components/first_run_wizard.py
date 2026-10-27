"""首次运行引导向导 - 帮助用户了解并下载必要的 AI 模型"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QWidget, QCheckBox
)
from PySide6.QtCore import Qt, Signal


class FirstRunWizard(QDialog):
    """首次运行引导对话框

    触发条件: 人脸模型(必选组件)缺失。
    界面分两区:
      - 必选: 人脸识别模型(缺失才显示, 提供下载)
      - 可选: MiniCPM-V VLM(用于本地 AI 分析; 可用设置页其他 VLM 供应商替代)
    """

    download_required_requested = Signal()  # 用户请求下载必选组件(人脸模型)
    download_vlm_requested = Signal()  # 用户请求下载可选 VLM(MiniCPM-V)
    skip_requested = Signal()  # 用户跳过(暂不下载)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("欢迎使用 MediaTag")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setMinimumHeight(360)
        self.setWindowFlags(
            self.windowFlags() 
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        
        from src.ui.components.theme import theme
        t = theme()
        
        self.setStyleSheet(f"""
            QDialog {{
                background: {t.get('bg_primary')};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(32, 32, 32, 32)
        
        # ── 标题区域 ─────────────────────────────────────────
        title_layout = QVBoxLayout()
        title_layout.setSpacing(8)
        
        title_label = QLabel("👋 欢迎使用 MediaTag")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {t.get('text_primary')};
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("首次使用，请选择需要下载的组件")
        subtitle_label.setStyleSheet(f"""
            font-size: 15px;
            color: {t.get('text_secondary')};
        """)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(subtitle_label)
        
        layout.addLayout(title_layout)
        layout.addSpacing(10)

        # ── 下载项勾选列表 ───────────────────────────────────
        # 人脸模型(必选, 默认勾选且不可取消)
        self.face_checkbox = QCheckBox("人脸识别模型（约 600 MB）")
        self.face_checkbox.setChecked(True)
        self.face_checkbox.setToolTip(
            "人脸检测与识别所必需（AuraFace，Apache-2.0 可商用）"
        )
        self.face_checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: 15px;
                font-weight: 500;
                color: {t.get('text_primary')};
                spacing: 10px;
                padding: 8px 4px;
            }}
        """)
        layout.addWidget(self.face_checkbox)

        # VLM 视觉模型(可选, 默认不勾选)
        self.vlm_checkbox = QCheckBox("MiniCPM-V 视觉模型（约 5 GB，可选）")
        self.vlm_checkbox.setChecked(False)
        self.vlm_checkbox.setToolTip(
            "用于本地 AI 视频内容分析；也可在「设置 → VLM 供应商」"
            "连接其他模型/云端服务，非必需"
        )
        self.vlm_checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: 15px;
                font-weight: 500;
                color: {t.get('text_primary')};
                spacing: 10px;
                padding: 8px 4px;
            }}
        """)
        layout.addWidget(self.vlm_checkbox)

        # VLM 可选说明
        vlm_hint = QLabel("可选：VLM 也可在设置页连接其他模型/云端服务，人脸识别不受影响")
        vlm_hint.setWordWrap(True)
        vlm_hint.setStyleSheet(f"""
            font-size: 11px;
            color: {t.get('text_tertiary')};
            padding-left: 12px;
        """)
        layout.addWidget(vlm_hint)

        layout.addStretch()

        # ── 按钮区域 ─────────────────────────────────────────
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # 暂不下载
        skip_btn = QPushButton("暂不下载")
        skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {t.get('text_secondary')};
                border: none;
                padding: 8px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                color: {t.get('text_primary')};
            }}
        """)
        skip_btn.clicked.connect(self._on_skip)
        button_layout.addWidget(skip_btn)

        # 开始下载
        start_btn = QPushButton("开始下载")
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                padding: 10px 32px;
                font-size: 15px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        start_btn.clicked.connect(self._on_start)
        button_layout.addWidget(start_btn)

        layout.addLayout(button_layout)
    
    def _selected_components(self):
        """返回用户勾选要下载的组件集合。"""
        sel = []
        if self.face_checkbox.isChecked():
            sel.append("face")
        if self.vlm_checkbox.isChecked():
            sel.append("vlm")
        return sel

    def _on_start(self):
        """用户点击「开始下载」: 按勾选项分别触发下载。

        - 勾了人脸模型 → download_required_requested(下载必选组件)
        - 勾了 MiniCPM-V → download_vlm_requested(下载可选 VLM)
        两者都勾则按序先必选后可选; 一个都没勾则仅关闭向导。
        """
        sel = self._selected_components()
        self.accept()
        if "face" in sel:
            self.download_required_requested.emit()
        if "vlm" in sel:
            self.download_vlm_requested.emit()

    def _on_skip(self):
        """用户选择暂不下载"""
        self.skip_requested.emit()
        self.reject()


class DownloadProgressDialog(QDialog):
    """下载进度对话框 - 显示详细的下载进度"""
    
    cancel_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("正在下载模型")
        # 非模态 + 允许关闭: 提示语承诺「下载过程可继续使用」, modal 却锁死
        # 主窗口是矛盾的。关闭 = 隐藏(下载后台继续, 进度在主窗口状态栏可见)。
        self.setModal(False)
        self.setMinimumWidth(500)
        
        from src.ui.components.theme import theme
        t = theme()
        
        self.setStyleSheet(f"""
            QDialog {{
                background: {t.get('bg_primary')};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 标题
        self.title_label = QLabel("正在下载 AI 模型...")
        self.title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {t.get('text_primary')};
        """)
        layout.addWidget(self.title_label)
        
        # 状态说明
        self.status_label = QLabel("正在下载 MiniCPM-V 模型，请稍候...")
        self.status_label.setStyleSheet(f"""
            font-size: 13px;
            color: {t.get('text_secondary')};
        """)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {t.get('bg_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                text-align: center;
                font-size: 12px;
                color: {t.get('text_primary')};
                height: 28px;
            }}
            QProgressBar::chunk {{
                background: {t.get('bg_hover')};
                border-radius: {t.get('radius_md')}px;
            }}
        """)
        self.progress_bar.setFixedHeight(28)
        layout.addWidget(self.progress_bar)
        
        # 详细信息
        details_widget = QWidget()
        details_layout = QHBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(24)
        
        self.downloaded_label = QLabel("已下载: --")
        self.downloaded_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.downloaded_label)
        
        self.total_label = QLabel("总计: 5 GB")
        self.total_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.total_label)
        
        self.speed_label = QLabel("速度: --")
        self.speed_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
        """)
        details_layout.addWidget(self.speed_label)
        
        details_layout.addStretch()
        layout.addWidget(details_widget)
        
        # 提示信息
        hint_label = QLabel("💡 下载过程中可以继续使用，模型会在后台下载")
        hint_label.setStyleSheet(f"""
            font-size: 12px;
            color: {t.get('text_secondary')};
            font-style: italic;
        """)
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)
        
        layout.addStretch()
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("取消下载")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {t.get('text_secondary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
                color: {t.get('text_primary')};
            }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("完成")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setVisible(False)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('bg_secondary')};
                color: {t.get('text_primary')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {t.get('bg_hover')};
            }}
        """)
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def closeEvent(self, event):
        """关闭窗口 = 隐藏, 不取消下载(下载后台继续, 进度在状态栏可见)。"""
        event.ignore()
        self.hide()

    def _on_cancel(self):
        """用户取消下载"""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("正在取消...")
        self.status_label.setText("正在取消下载...")
        self.cancel_requested.emit()
    
    def update_progress(self, percent: float, downloaded: str = "", 
                       total: str = "", speed: str = "", status: str = ""):
        """更新进度显示

        percent=0 且为状态行(manifest/verify 等)时不动进度条——
        避免「pulling manifest」等行把总进度条打回 0。
        """
        if percent > 0:
            self.progress_bar.setValue(int(percent))
        
        if downloaded:
            self.downloaded_label.setText(f"已下载: {downloaded}")
        if total:
            self.total_label.setText(f"总计: {total}")
        if speed:
            self.speed_label.setText(f"速度: {speed}")
        if status and status not in ("pulling manifest", "verifying", "writing"):
            self.status_label.setText(status)
    
    def set_completed(self, success: bool, message: str):
        """设置完成状态"""
        from src.ui.components.theme import theme
        t = theme()
        
        if success:
            self.title_label.setText("✅ 下载完成")
            self.status_label.setText(message)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("完成")
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {t.get('bg_secondary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_md')}px;
                    text-align: center;
                    font-size: 12px;
                    color: {t.get('text_primary')};
                    height: 28px;
                }}
                QProgressBar::chunk {{
                    background: {t.get('bg_hover')};
                    border-radius: {t.get('radius_md')}px;
                }}
            """)
            self.cancel_btn.setVisible(False)
            self.close_btn.setVisible(True)
        else:
            self.title_label.setText("❌ 下载失败")
            self.status_label.setText(message)
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {t.get('bg_secondary')};
                    border: 1px solid {t.get('border')};
                    border-radius: {t.get('radius_md')}px;
                    text-align: center;
                    font-size: 12px;
                    color: {t.get('text_primary')};
                    height: 28px;
                }}
                QProgressBar::chunk {{
                    background: {t.get('bg_hover')};
                    border-radius: {t.get('radius_md')}px;
                }}
            """)
            self.cancel_btn.setText("关闭")
            self.cancel_btn.clicked.disconnect()
            self.cancel_btn.clicked.connect(self.reject)
