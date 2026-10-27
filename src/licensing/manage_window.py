"""授权管理窗口（应用菜单「授权管理…」/ 设置页入口）。

免费版：额度说明 + 激活入口；完整版：方案/设备管理（内嵌 LicenseStatusPanel 同源面板）。
"""

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .gate import SUPPORT_EMAIL
from .status_panel import LicenseStatusPanel


class LicenseManageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("授权管理")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)

        layout.addWidget(LicenseStatusPanel(self))

        buttons = QHBoxLayout()
        buttons.addStretch()
        support = QLabel(f"如有疑问请联系 {SUPPORT_EMAIL}")
        buttons.addWidget(support)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
