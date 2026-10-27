"""
依赖安装向导 - 首次启动时检查并引导安装依赖

功能：
1. 显示依赖检测进度
2. 提供一键安装功能
3. 显示安装日志
4. 安装完成后进入主界面
"""

import re
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QFont, QDesktopServices

from src.core.dependency_checker import DependencyChecker, DependencyStatus


def clean_ansi(text: str) -> str:
    """清理 ANSI 转义序列"""
    # 移除颜色代码
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # 移除光标控制序列
    text = re.sub(r'\x1b\[\?[0-9;]*[a-zA-Z]', '', text)
    # 移除其他转义序列
    text = re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', text)
    # 移除回车符
    text = text.replace('\r', '')
    return text.strip()


class InstallWorker(QThread):
    """安装工作线程"""
    
    log_output = Signal(str)
    install_finished = Signal(bool, str)
    
    def __init__(self, command: str, name: str):
        super().__init__()
        self.command = command
        self.name = name
        self.process: Optional[subprocess.Popen] = None
    
    def run(self):
        try:
            self.log_output.emit(f"开始安装 {self.name}...")
            self.log_output.emit(f"执行命令: {self.command}")
            self.log_output.emit("-" * 50)
            
            # 执行安装命令
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 读取输出（清理 ANSI 字符）
            for line in self.process.stdout:
                cleaned_line = clean_ansi(line)
                if cleaned_line:  # 只输出非空行
                    self.log_output.emit(cleaned_line)
            
            # 等待完成
            return_code = self.process.wait()
            
            if return_code == 0:
                self.log_output.emit("-" * 50)
                self.log_output.emit(f"✓ {self.name} 安装成功")
                self.install_finished.emit(True, f"{self.name} 安装成功")
            else:
                self.log_output.emit("-" * 50)
                self.log_output.emit(f"✗ {self.name} 安装失败 (返回码: {return_code})")
                self.install_finished.emit(False, f"{self.name} 安装失败")
        
        except Exception as e:
            self.log_output.emit(f"✗ 安装出错: {str(e)}")
            self.install_finished.emit(False, str(e))


class DependencyWizard(QDialog):
    """依赖安装向导对话框"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("依赖安装向导")
        self.setMinimumSize(600, 500)
        self.setModal(True)
        
        self.checker = DependencyChecker()
        self.current_worker: Optional[InstallWorker] = None
        
        self._init_ui()
        self._check_dependencies()
    
    def _init_ui(self):
        """初始化 UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 标题
        title = QLabel("依赖检查")
        title.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 说明
        desc = QLabel("MediaTag 需要以下依赖才能正常运行")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #666; font-size: 13px;")
        layout.addWidget(desc)
        
        # 依赖列表
        self.dep_list_widget = QWidget()
        self.dep_list_layout = QVBoxLayout(self.dep_list_widget)
        self.dep_list_layout.setSpacing(8)
        self.dep_list_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dep_list_widget)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Menlo", 11))
        from .theme import theme
        t = theme()
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {t.get('bg_card')};
                color: {t.get('text_primary')};
                border: 1px solid {t.get('border')};
                border-radius: {t.get('radius_md')}px;
                padding: 8px;
            }}
        """)
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.install_btn = QPushButton("一键安装")
        self.install_btn.setMinimumHeight(36)
        self.install_btn.setMinimumWidth(120)
        self.install_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.get('accent')};
                color: {t.get('text_on_accent')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                font-size: 14px;
                font-weight: 500;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {t.get('accent_hover')};
            }}
            QPushButton:disabled {{
                background-color: {t.get('text_tertiary')};
            }}
        """)
        self.install_btn.clicked.connect(self._install_all)
        btn_layout.addWidget(self.install_btn)
        
        self.continue_btn = QPushButton("继续")
        self.continue_btn.setMinimumHeight(36)
        self.continue_btn.setMinimumWidth(120)
        self.continue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.get('success')};
                color: {t.get('text_on_accent')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                font-size: 14px;
                font-weight: 500;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {t.get('success_hover')};
            }}
        """)
        self.continue_btn.clicked.connect(self.accept)
        self.continue_btn.setVisible(False)
        btn_layout.addWidget(self.continue_btn)
        
        layout.addLayout(btn_layout)
    
    def _check_dependencies(self):
        """检查依赖"""
        # 清空列表
        for i in reversed(range(self.dep_list_layout.count())):
            widget = self.dep_list_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 检查依赖
        deps = self.checker.check_all()
        
        # 添加依赖项
        for key, dep in deps.items():
            item_widget = self._create_dep_item(dep)
            self.dep_list_layout.addWidget(item_widget)
        
        # 更新按钮状态
        all_installed = self.checker.all_dependencies_installed()
        self.install_btn.setEnabled(not all_installed)
        self.install_btn.setVisible(not all_installed)
        self.continue_btn.setVisible(all_installed)
        
        if all_installed:
            self._append_log("✓ 所有依赖已安装，可以继续使用")
    
    def _create_dep_item(self, dep: DependencyStatus) -> QWidget:
        """创建依赖项组件"""
        from .theme import theme
        t = theme()
        widget = QWidget()
        widget.setStyleSheet(f"""
            QWidget {{
                background-color: {t.get('bg_secondary')};
                border-radius: {t.get('radius_lg')}px;
                padding: 12px;
            }}
        """)
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(12, 8, 12, 8)
        
        # 状态图标
        icon_label = QLabel("✓" if dep.installed else "✗")
        icon_label.setFont(QFont("SF Pro Display", 16))
        icon_label.setStyleSheet(f"color: {t.get('success') if dep.installed else t.get('danger')};")
        icon_label.setFixedWidth(24)
        layout.addWidget(icon_label)
        
        # 名称和消息
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        name_label = QLabel(dep.name)
        name_label.setFont(QFont("SF Pro Display", 14, QFont.Weight.Medium))
        text_layout.addWidget(name_label)
        
        msg_label = QLabel(dep.message)
        msg_label.setStyleSheet(f"color: {t.get('text_secondary')}; font-size: 12px;")
        text_layout.addWidget(msg_label)
        
        layout.addLayout(text_layout, 1)
        
        # 安装命令或下载链接（如果未安装）
        if not dep.installed:
            if dep.download_url:
                # 显示下载链接按钮
                download_btn = QPushButton("下载")
                download_btn.setFont(QFont("SF Pro Display", 12))
                download_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {t.get('accent')};
                        color: {t.get('text_on_accent')};
                        border: none;
                        border-radius: {t.get('radius_md')}px;
                        padding: 4px 12px;
                    }}
                    QPushButton:hover {{
                        background-color: {t.get('accent_hover')};
                    }}
                """)
                download_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(dep.download_url)))
                layout.addWidget(download_btn)
            elif dep.install_command:
                # 显示安装命令
                cmd_label = QLabel(dep.install_command)
                cmd_label.setFont(QFont("Menlo", 11))
                cmd_label.setStyleSheet(f"""
                    background-color: {t.get('bg_hover')};
                    color: {t.get('text_primary')};
                    padding: 4px 8px;
                    border-radius: {t.get('radius_md')}px;
                """)
                layout.addWidget(cmd_label)
        
        return widget
    
    def _install_all(self):
        """一键安装所有缺失依赖"""
        missing = self.checker.get_missing_dependencies()
        
        if not missing:
            self._append_log("所有依赖已安装")
            return
        
        # 检测是否是打包环境
        if self.checker.is_bundled_app():
            # 打包环境：使用专门的安装脚本（只下载模型，不安装 Homebrew）
            self._install_bundled_app()
            return
        
        # 开发环境：使用原有的安装逻辑
        # 显示日志区域
        self.log_text.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(missing))
        self.progress_bar.setValue(0)
        
        # 禁用安装按钮
        self.install_btn.setEnabled(False)
        self.install_btn.setText("安装中...")
        
        # 依次安装
        self._install_next(missing, 0)
    
    def _install_bundled_app(self):
        """打包环境下的安装（只下载模型）"""
        import sys
        
        # 显示日志区域
        self.log_text.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        
        # 禁用安装按钮
        self.install_btn.setEnabled(False)
        self.install_btn.setText("安装中...")
        
        # 查找打包后的安装脚本
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            script_path = Path(meipass) / "scripts" / "install_bundled_app.sh"
        else:
            script_path = Path(sys.executable).parent / "scripts" / "install_bundled_app.sh"
        
        if not script_path.exists():
            self._append_log(f"✗ 找不到安装脚本: {script_path}")
            self.install_btn.setText("重试")
            self.install_btn.setEnabled(True)
            return
        
        # 创建工作线程
        self.current_worker = InstallWorker(
            f'bash "{script_path}"',
            "AI 模型"
        )
        self.current_worker.log_output.connect(self._append_log)
        self.current_worker.install_finished.connect(
            lambda success, msg: self._on_bundled_install_finished(success, msg)
        )
        self.current_worker.start()
    
    def _on_bundled_install_finished(self, success: bool, message: str):
        """打包环境安装完成回调"""
        self.progress_bar.setValue(1)
        
        if success:
            self._append_log("\n" + "=" * 50)
            self._append_log("✓ 所有依赖安装完成！")
            self.install_btn.setText("重新检查")
            self.install_btn.setEnabled(True)
            self.continue_btn.setVisible(True)
        else:
            self._append_log(f"\n✗ 安装失败: {message}")
            self.install_btn.setText("重试")
            self.install_btn.setEnabled(True)
        
        # 重新检查依赖
        self.checker.check_all()
        self._check_dependencies()
    
    def _install_next(self, missing: list, index: int):
        """安装下一个依赖"""
        if index >= len(missing):
            # 所有安装完成
            self._append_log("\n" + "=" * 50)
            self._append_log("所有依赖安装完成！")
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.install_btn.setText("重新检查")
            self.install_btn.setEnabled(True)
            self.continue_btn.setVisible(True)
            return
        
        dep = missing[index]
        self.progress_bar.setValue(index)
        
        # 创建工作线程
        self.current_worker = InstallWorker(dep.install_command, dep.name)
        self.current_worker.log_output.connect(self._append_log)
        self.current_worker.install_finished.connect(
            lambda success, msg: self._on_install_finished(success, msg, missing, index)
        )
        self.current_worker.start()
    
    def _on_install_finished(self, success: bool, message: str, missing: list, index: int):
        """安装完成回调"""
        # 重新检查依赖
        self.checker.check_all()
        
        # 更新 UI
        self._check_dependencies()
        
        # 继续安装下一个
        self._install_next(missing, index + 1)
    
    def _append_log(self, text: str):
        """追加日志"""
        self.log_text.append(text)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
