"""
关于对话框 - 展示应用信息和版权
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QTextBrowser
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from .theme import theme, get_system_font

# 版本号兜底: 优先读 .app bundle 的 CFBundleShortVersionString,
# 裸 python 运行(无 bundle)时才用此常量。
# 导入需兼容两种运行形态: dev 脚本模式(src 在 sys.path, 顶层 version),
# 包模式 / 打包入口(仓库根在 sys.path, src.version)。
try:
    from version import APP_VERSION
except ImportError:
    from src.version import APP_VERSION


def _runtime_version() -> str:
    """打包版/开发壳返回 Info.plist 里的版本号, 与发布版本保持同步。"""
    try:
        from Foundation import NSBundle
        v = NSBundle.mainBundle().objectForInfoDictionaryKey_(
            "CFBundleShortVersionString")
        if v:
            return str(v)
    except Exception:
        pass
    return APP_VERSION


# 第三方开源组件声明(商业分发合规提示)
# 注: 许可证为各组件官方主版本常见许可, 完整文本请以组件自带 LICENSE / 官方仓库为准。
THIRD_PARTY_NOTICES = [
    ("PySide6 / Qt", "LGPL-3.0", "桌面 GUI 框架(Qt 经 PySide6 动态链接使用)"),
    ("PyTorch", "BSD-3-Clause", "深度学习框架(CLIP/特征提取)"),
    ("OpenCV (opencv-python)", "Apache-2.0",
     "图像/视频处理; wheel 内嵌 FFmpeg 等组件为 LGPL-2.1, 许可文本随包携带(third_party_licenses)"),
    ("Ultralytics YOLO", "AGPL-3.0", "目标检测; 注意: AGPL 具传染性, 商业闭源分发需评估或购买企业授权"),
    ("OpenAI CLIP / open_clip", "MIT", "图文特征模型"),
    ("DeepFace", "MIT", "人脸属性分析"),
    ("InsightFace", "MIT", "人脸检测/识别"),
    ("ONNX Runtime", "MIT", "推理运行时"),
    ("opennsfw-onnx", "Apache-2.0", "NSFW 预筛推理封装(ONNX / CoreML)"),
    ("open_nsfw (Yahoo)", "BSD-2-Clause", "NSFW 预筛模型权重"),
    ("scikit-learn", "BSD-3-Clause", "机器学习工具"),
    ("MediaPipe", "Apache-2.0", "人脸关键点等"),
    ("FastAPI", "MIT", "后端 API 框架"),
    ("Uvicorn", "BSD-3-Clause", "ASGI 服务器"),
    ("SQLAlchemy", "MIT", "ORM(部分旧接口)"),
    ("Pydantic", "MIT", "数据校验"),
    ("Streamlit", "Apache-2.0", "Web 备用界面"),
    ("SQLite", "Public Domain", "本地数据库"),
    ("Ollama", "MIT", "本地大模型运行时"),
    ("MiniCPM-V (经 Ollama 调用)", "模型许可", "VLM 视觉语言模型; 模型权重许可独立于代码, 商用前需按其发布页确认"),
    ("FFmpeg", "LGPL-2.1+(内置)",
     "视频解码/探测; 内置 LGPL 构建 binaries (bundled_bin/ffmpeg_lgpl, 不含 x264/x265 等 GPL 组件), 许可文本见 third_party_licenses/ffmpeg"),
]


class ThirdPartyDialog(QDialog):
    """第三方开源组件声明对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("第三方组件声明")
        self.setModal(True)
        self.resize(620, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        lines = [
            "MediaTag 基于以下开源组件构建, 按各组件许可证要求声明:\n"
        ]
        for name, license_name, usage in THIRD_PARTY_NOTICES:
            lines.append(f"■ {name}\n   许可证: {license_name}\n   用途: {usage}\n")
        lines.append(
            "以上为摘要信息, 各组件完整的版权与许可证文本, "
            "请以发行包内 LICENSE 文件或官方仓库为准。\n"
            "若将本软件用于商业分发, 请依据实际调用与分发方式 "
            "(尤其涉及 AGPL/GPL/LGPL 组件)评估相应合规义务。"
        )
        browser.setPlainText("\n".join(lines))
        layout.addWidget(browser)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(100, 32)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)


class AboutDialog(QDialog):
    """关于对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("关于 MediaTag")
        self.setFixedSize(520, 500)
        self.setModal(True)
        
        t = theme()
        
        # 主布局
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 应用图标
        icon_label = QLabel()
        icon_pixmap = QPixmap("assets/app_icon.png")
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(
                80, 80, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # 应用名称
        title_label = QLabel("MediaTag")
        title_label.setFont(get_system_font(24, "bold"))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {t.get('text_primary')};")
        layout.addWidget(title_label)
        
        # 版本号
        version_label = QLabel(f"版本 {_runtime_version()}")
        version_label.setFont(get_system_font(12))
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet(f"color: {t.get('text_secondary')};")
        layout.addWidget(version_label)
        
        # 分隔线
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {t.get('border')};")
        layout.addWidget(separator)
        
        # 描述信息
        desc_label = QLabel("智能媒体标签管理系统（视频 + 照片）")
        desc_label.setFont(get_system_font(14))
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet(f"color: {t.get('text_primary')};")
        layout.addWidget(desc_label)
        
        # 技术栈
        tech_label = QLabel(
            "基于 AI 的视频分析工具\n"
            "使用 MiniCPM-V 视觉语言模型"
        )
        tech_label.setFont(get_system_font(11))
        tech_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tech_label.setStyleSheet(f"color: {t.get('text_secondary')};")
        tech_label.setWordWrap(True)
        layout.addWidget(tech_label)
        
        # 分隔线
        separator2 = QLabel()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet(f"background-color: {t.get('border')};")
        layout.addWidget(separator2)
        
        # 版权信息
        copyright_label = QLabel("© 2024-2026 MediaTag. All rights reserved.")
        copyright_label.setFont(get_system_font(10))
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet(f"color: {t.get('text_tertiary')};")
        layout.addWidget(copyright_label)
        
        # 邮箱联系
        email_label = QLabel()
        email_label.setText('<a href="mailto:support@cloudcraft.xin" style="color: {accent}; text-decoration: none;">support@cloudcraft.xin</a>'.format(accent=t.get('accent')))
        email_label.setFont(get_system_font(10))
        email_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        email_label.setOpenExternalLinks(True)
        email_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(email_label)
        
        # 许可信息
        license_label = QLabel("本软件及其文档受版权法保护")
        license_label.setFont(get_system_font(10))
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_label.setStyleSheet(f"color: {t.get('text_tertiary')};")
        layout.addWidget(license_label)
        
        # 第三方组件声明入口
        notice_label = QLabel()
        notice_label.setText(
            '<a href="third-party" style="color: {accent}; text-decoration: none;">第三方组件声明</a>'.format(accent=t.get('accent')))
        notice_label.setFont(get_system_font(10))
        notice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice_label.linkActivated.connect(self._show_third_party)
        layout.addWidget(notice_label)
        
        layout.addStretch()
        
        # 关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.setFixedSize(100, 32)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {t.get('accent')};
                color: {t.get('text_on_accent')};
                border: none;
                border-radius: {t.get('radius_md')}px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {t.get('accent_hover')};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)

    def _show_third_party(self):
        """打开第三方组件声明对话框。"""
        dialog = ThirdPartyDialog(self)
        dialog.exec()
