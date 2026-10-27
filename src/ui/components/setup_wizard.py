"""
依赖安装向导 - 图形化界面
提供一键安装所有依赖项的功能
"""

import subprocess
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread


class InstallWorker(QThread):
    """后台安装工作线程"""
    progress = Signal(str)  # 输出日志
    finished = Signal(int)  # 退出码
    
    def __init__(self, script_args: list):
        super().__init__()
        self.script_args = script_args
        
    def run(self):
        """执行安装脚本"""
        try:
            # 使用 subprocess 执行安装脚本
            process = subprocess.Popen(
                ['bash'] + self.script_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取输出
            if process.stdout:
                for line in process.stdout:
                    self.progress.emit(line.rstrip())
            
            process.wait()
            self.finished.emit(process.returncode)
            
        except Exception as e:
            self.progress.emit(f"错误: {str(e)}")
            self.finished.emit(-1)


class SetupWizardDialog(QDialog):
    """安装向导对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("依赖安装向导")
        self.setMinimumSize(700, 500)
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 标题
        title = QLabel("MediaTag 依赖安装向导")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 说明文本
        desc = QLabel(
            "此向导将帮助您安装 MediaTag 所需的所有依赖项：\n"
            "• FFmpeg - 视频处理\n"
            "• Ollama - 本地 AI 模型运行时\n"
            "• MiniCPM-V 模型 - 视频分析\n"
            "• 人脸识别 - 自动人物识别与聚类（可选，约 200MB）\n\n"
            "安装过程可能需要 10-30 分钟，请保持网络连接。"
        )
        desc.setWordWrap(True)
        from .theme import theme
        t = theme()
        desc.setStyleSheet(f"padding: 10px; background: {t.get('bg_secondary')}; border-radius: {t.get('radius_md')}px;")
        layout.addWidget(desc)
        
        # 可选组件选择
        options_group = QGroupBox("可选组件")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        from PySide6.QtWidgets import QCheckBox
        self.face_recognition_checkbox = QCheckBox("人脸识别（自动识别人物并聚类，约 200MB）")
        self.face_recognition_checkbox.setChecked(True)  # 默认启用
        self.face_recognition_checkbox.setToolTip("安装 insightface 和 onnxruntime，用于自动识别视频中的人脸")
        options_layout.addWidget(self.face_recognition_checkbox)
        
        layout.addWidget(options_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志输出区域
        log_group = QGroupBox("安装日志")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            f"QTextEdit {{ "
            f"font-family: 'Monaco', 'Menlo', monospace; "
            f"font-size: 11px; "
            f"background: {t.get('bg_card')}; "
            f"color: {t.get('text_primary')}; "
            f"padding: 5px; "
            f"}}"
        )
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.install_btn = QPushButton("开始安装")
        self.install_btn.setStyleSheet(
            f"QPushButton {{ "
            f"background: {t.get('bg_hover')}; "
            f"color: {t.get('text_on_accent')}; "
            f"padding: 8px 20px; "
            f"font-weight: bold; "
            f"border-radius: {t.get('radius_md')}px; "
            f"}} "
            f"QPushButton:hover {{ background: {t.get('bg_hover')}; }} "
            f"QPushButton:disabled {{ background: {t.get('text_tertiary')}; }}"
        )
        self.install_btn.clicked.connect(self.start_installation)
        btn_layout.addWidget(self.install_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            f"QPushButton {{ "
            f"padding: 8px 20px; "
            f"border-radius: {t.get('radius_md')}px; "
            f"}} "
            f"QPushButton:hover {{ background: {t.get('bg_hover')}; }}"
        )
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
    def start_installation(self):
        """开始安装"""
        # 查找安装脚本
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "install_dependencies.sh"
        
        if not script_path.exists():
            QMessageBox.critical(
                self,
                "错误",
                f"找不到安装脚本:\n{script_path}"
            )
            return
        
        # 禁用按钮
        self.install_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        
        # 构建安装参数
        script_args = [str(script_path)]
        if not self.face_recognition_checkbox.isChecked():
            script_args.append("--skip-face")
        
        # 启动工作线程
        self.worker = InstallWorker(script_args)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_install_finished)
        self.worker.start()
        
    def append_log(self, text: str):
        """追加日志"""
        self.log_text.append(text)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def on_install_finished(self, exit_code: int):
        """安装完成"""
        self.progress_bar.setVisible(False)
        self.install_btn.setEnabled(True)
        
        if exit_code == 0:
            QMessageBox.information(
                self,
                "安装成功",
                "所有依赖项已成功安装！\n\n"
                "您现在可以开始使用 MediaTag 了。"
            )
        else:
            QMessageBox.warning(
                self,
                "安装完成（有警告）",
                f"安装过程已完成，但可能有一些问题。\n\n"
                f"退出码: {exit_code}\n\n"
                f"请查看上方日志了解详情。"
            )


def run_setup_wizard(parent=None):
    """运行安装向导（便捷函数）"""
    dialog = SetupWizardDialog(parent)
    dialog.exec()
