"""激活对话框（按需弹出：设置页授权区 / 授权管理窗口 / 导入额度引导）。

v1.2 免费额度模式下不再有启动门、锁定屏与等待窗——激活是唯一需要的授权对话，
激活成功即进程内切换为完整版（无需重启）。
错误文案严格按规范 §3.5 映射表；关闭窗口即放弃（免费版继续可用，无任何阻断）。
"""

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from . import client, fingerprint, state, store
from .gate import SUPPORT_EMAIL
from .qt_worker import AsyncWorker

_log = logging.getLogger("licensing.activation")

# 规范 §3.5 错误文案映射表（error code → 用户提示）
ACTIVATION_ERROR_TEXT = {
    "invalid_code": "激活码无效，请检查后重试",
    "license_revoked": "激活码已被吊销，请联系 " + SUPPORT_EMAIL,
    "license_expired": "激活码已过期",
    "license_refunded": "该订单已退款",
    "device_limit_reached": "激活码已绑定到最大设备数，请先在旧设备解绑",
    "rate_limited": "尝试过于频繁，请稍后再试",
}
FALLBACK_ERROR_TEXT = "网络异常，请稍后重试"


def _app_icon_path() -> Path | None:
    candidates = []
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", ""))
        candidates.append(base / "assets" / "app_icon.png")
    candidates.append(Path(__file__).resolve().parents[2] / "assets" / "app_icon.png")
    candidates.append(Path("assets/app_icon.png"))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _theme_token(key: str, fallback: str) -> str:
    try:
        from ui.components.theme import theme
    except ImportError:
        try:
            from src.ui.components.theme import theme
        except ImportError:
            return fallback
    return str(theme().get(key, fallback))


def _activation_snapshot(res: client.ActivateResult) -> dict:
    return {
        "source": "activate",
        "status": "active",
        "plan": res.plan,
        "plan_type": res.plan_type,
        "features": res.features,
        "expires_at": res.expires_at,
        "days_to_expiry": None,
        "devices_used": res.devices_used,
        "devices_max": res.devices_max,
        "verified_at": datetime.now().isoformat(timespec="seconds"),
    }


class ActivationDialog(QDialog):
    """激活对话框：输码 → POST /v1/activate → 存 token → 切完整版。"""

    def __init__(self, hint: str | None = None, parent=None):
        super().__init__(parent)
        self.activated = False
        self._worker: AsyncWorker | None = None
        self.setWindowTitle("激活 MediaTag")
        self.setModal(True)
        self.setFixedWidth(420)
        self.setStyleSheet(self._base_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(10)

        icon_path = _app_icon_path()
        if icon_path:
            from PySide6.QtGui import QIcon

            icon_label = QLabel()
            icon_label.setPixmap(QIcon(str(icon_path)).pixmap(56, 56))
            icon_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(icon_label)

        title = QLabel("激活 MediaTag")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        subtitle = QLabel("输入购买时获得的激活码以解除免费版额度限制")
        subtitle.setProperty("role", "secondary")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(subtitle)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setProperty("role", "hint")
            hint_label.setWordWrap(True)
            layout.addWidget(hint_label)

        self.code_edit = QLineEdit()
        self.code_edit.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
        layout.addWidget(self.code_edit)

        self.error_label = QLabel("")
        self.error_label.setProperty("role", "error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.activate_btn = QPushButton("激活")
        self.activate_btn.setDefault(True)
        self.activate_btn.clicked.connect(self._on_activate)
        buttons.addWidget(self.activate_btn)
        layout.addLayout(buttons)

        support = QLabel(f"激活遇到问题？请联系 {SUPPORT_EMAIL}")
        support.setProperty("role", "secondary")
        support.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(support)

    def _base_style(self) -> str:
        return (
            f"QDialog {{ background-color: {_theme_token('bg_primary', '#ffffff')}; }}"
            f"QLabel {{ color: {_theme_token('text_primary', '#1d1d1f')}; }}"
            f"QLabel[role='secondary'] {{ color: {_theme_token('text_secondary', '#6e6e73')}; font-size: 12px; }}"
            f"QLabel[role='error'] {{ color: {_theme_token('danger', '#d70015')}; font-size: 12px; }}"
            f"QLabel[role='hint'] {{ color: {_theme_token('warning', '#b25000')}; font-size: 12px; }}"
            f"QLineEdit {{ padding: 8px 10px; border: 1px solid {_theme_token('border', '#d2d2d7')};"
            f" border-radius: 6px; font-size: 14px; font-family: Menlo, monospace;"
            f" background: {_theme_token('bg_input', '#ffffff')}; color: {_theme_token('text_primary', '#1d1d1f')}; }}"
            f"QPushButton {{ padding: 8px 18px; border-radius: 6px;"
            f" background-color: {_theme_token('accent', '#0a84ff')}; color: white; font-size: 13px; }}"
            "QPushButton:disabled { background-color: #a0c4ff; }"
        )

    def _show_error(self, text: str) -> None:
        self.error_label.setText(text)
        self.error_label.show()

    def _on_activate(self) -> None:
        code = self.code_edit.text().strip()  # 去首尾空格；连字符差异由服务端处理
        if not code:
            self._show_error("请输入激活码")
            return
        self.activate_btn.setEnabled(False)
        self.activate_btn.setText("正在激活…")
        self.error_label.hide()

        def _do_activate():
            fp = fingerprint.get_fingerprint()
            return client.activate(code, fp, platform.node() or "")

        self._worker = AsyncWorker(_do_activate, self)
        self._worker.done.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.start()

    def _on_success(self, res) -> None:
        store.save_token(res.token)
        store.save_license_info(_activation_snapshot(res))
        state.set_mode(state.Mode.PRO, "activated")
        self.activated = True
        self.accept()

    def _on_failure(self, exc: Exception) -> None:
        self.activate_btn.setEnabled(True)
        self.activate_btn.setText("激活")
        if isinstance(exc, client.LicenseError):
            text = ACTIVATION_ERROR_TEXT.get(exc.error_code, FALLBACK_ERROR_TEXT)
            if exc.error_code == "device_limit_reached":
                devices = exc.data.get("devices")
                if isinstance(devices, list) and devices:
                    names = []
                    for item in devices:
                        name = item.get("device_name") if isinstance(item, dict) else None
                        names.append(str(name or "未知设备"))
                    text += "：\n· " + "\n· ".join(names)
            self._show_error(text)
        else:
            _log.warning("activate 失败: %s", exc)
            self._show_error(FALLBACK_ERROR_TEXT)
