"""设置页「授权」状态区（免费版/完整版双态）。

免费版：方案=免费版、额度（累计 x/100，≤10 时橙色警示）、[输入激活码…]。
完整版：方案/到期/设备/最近验证/功能 + 刷新 + 解绑本设备（解绑=主动取消激活，
做库存快照后切免费版，同窗体完成）。
"""

import logging
from datetime import datetime

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import client, quota, state, store
from .qt_worker import AsyncWorker

_log = logging.getLogger("licensing.status")


def _fmt_datetime(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso)


def _theme_token(key: str, fallback: str) -> str:
    try:
        from ui.components.theme import theme
    except ImportError:
        try:
            from src.ui.components.theme import theme
        except ImportError:
            return fallback
    return str(theme().get(key, fallback))


class LicenseStatusPanel(QWidget):
    """嵌入设置页/授权管理窗口的授权状态区块。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: AsyncWorker | None = None
        self._rows: dict[str, tuple[QLabel, QLabel]] = {}
        self._build_ui()
        self.refresh_local()

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        text_primary = _theme_token("text_primary", "#1d1d1f")
        text_secondary = _theme_token("text_secondary", "#6e6e73")

        self.setStyleSheet(
            "LicenseStatusPanel { background: transparent; }"
            f"QLabel {{ color: {text_primary}; font-size: 13px; }}"
            f"QLabel[role='secondary'] {{ color: {text_secondary}; font-size: 12px; }}"
            "QPushButton { padding: 4px 14px; border-radius: 6px; font-size: 12px;"
            f" background-color: {_theme_token('accent', '#0a84ff')}; color: white; }}"
            "QPushButton:disabled { background-color: #a0c4ff; }"
            f"QPushButton[role='danger'] {{ background-color: {_theme_token('danger', '#d70015')}; color: white; }}"
            f"QPushButton[role='secondary'] {{ background-color: {_theme_token('bg_hover', '#e5e5ea')}; color: {text_primary}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(8)

        title = QLabel("授权")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        root.addWidget(title)

        self.plan_value = self._add_row(root, "plan", "方案：")
        self.quota_value = self._add_row(root, "quota", "额度：")
        self.expiry_value = self._add_row(root, "expiry", "到期：")
        self.devices_value = self._add_row(root, "devices", "设备：")
        self.verified_value = self._add_row(root, "verified", "最近验证：")
        self.features_value = self._add_row(root, "features", "功能开关：")

        self.device_list_label = QLabel("")
        self.device_list_label.setProperty("role", "secondary")
        self.device_list_label.setWordWrap(True)
        root.addWidget(self.device_list_label)

        buttons = QHBoxLayout()
        self.activate_btn = QPushButton("输入激活码…")
        self.activate_btn.clicked.connect(self._on_activate)
        buttons.addWidget(self.activate_btn)
        self.refresh_btn = QPushButton("刷新授权状态")
        self.refresh_btn.setProperty("role", "secondary")
        self.refresh_btn.clicked.connect(self.refresh_status)
        buttons.addWidget(self.refresh_btn)
        self.unbind_btn = QPushButton("解绑本设备")
        self.unbind_btn.setProperty("role", "danger")
        self.unbind_btn.clicked.connect(self._on_deactivate)
        buttons.addWidget(self.unbind_btn)
        buttons.addStretch()
        root.addLayout(buttons)

    def _add_row(self, layout: QVBoxLayout, name: str, label: str) -> QLabel:
        row = QHBoxLayout()
        key = QLabel(label)
        key.setProperty("role", "secondary")
        key.setStyleSheet("QLabel[role='secondary'] { font-size: 12px; }")
        row.addWidget(key)
        value = QLabel("—")
        row.addWidget(value)
        row.addStretch()
        layout.addLayout(row)
        self._rows[name] = (key, value)
        return value

    def _set_row_visible(self, name: str, visible: bool) -> None:
        key, value = self._rows[name]
        key.setVisible(visible)
        value.setVisible(visible)

    def _set_free_mode(self, free: bool) -> None:
        """按模式显隐行与按钮。"""
        self._set_row_visible("quota", free)
        for name in ("expiry", "devices", "verified", "features"):
            self._set_row_visible(name, not free)
        self.device_list_label.setVisible(not free)
        self.activate_btn.setVisible(free)
        self.refresh_btn.setVisible(not free)
        self.unbind_btn.setVisible(not free)

    # ── 数据 ──────────────────────────────────────────────────
    def refresh_local(self) -> None:
        """用本地状态立即刷新（不联网）。"""
        free = store.load_token() is None
        self._set_free_mode(free)
        if free:
            self.plan_value.setText("免费版")
            processed = quota.processed()
            remaining = quota.remaining()
            self.quota_value.setText(f"已处理 {processed}/100 · 剩 {remaining}")
            if remaining <= 10:
                self.quota_value.setStyleSheet(
                    f"color: {_theme_token('warning', '#ff9500')}; font-size: 13px;"
                )
            else:
                self.quota_value.setStyleSheet("")
            self.features_value.setText("全部功能可用")
            return

        self.quota_value.setStyleSheet("")
        info = store.load_license_info()
        if not info:
            self.plan_value.setText("已激活")
            return
        self.plan_value.setText(str(info.get("plan") or "—"))
        if info.get("plan_type") == "perpetual" or not info.get("expires_at"):
            self.expiry_value.setText("永久有效")
        else:
            self.expiry_value.setText(_fmt_datetime(info.get("expires_at")))
        used, mx = info.get("devices_used"), info.get("devices_max")
        self.devices_value.setText(
            f"{used}/{mx}" if used is not None and mx is not None else "—"
        )
        self.verified_value.setText(_fmt_datetime(info.get("verified_at")))
        features = info.get("features") or {}
        self.features_value.setText(
            "、".join(str(k) for k in features) if features else "—"
        )

    def refresh_status(self) -> None:
        """GET /v1/status 联网刷新（完整版：方案 + 设备列表）。"""
        token = store.load_token()
        if not token:
            return
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("正在获取…")
        self._start_worker(
            lambda: client.get_status(token), self._on_status, self._on_status_error
        )

    def _on_status(self, res) -> None:
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("刷新授权状态")
        self.plan_value.setText(str(res.plan or "—"))
        if res.expires_at:
            self.expiry_value.setText(_fmt_datetime(res.expires_at))
        else:
            self.expiry_value.setText("永久有效")
        self.devices_value.setText(f"{res.devices_used}/{res.devices_max}")
        lines = []
        for device in res.devices:
            name = device.get("device_name") or "未知设备"
            mark = "（本机）" if device.get("is_current") else ""
            lines.append(f"· {name}{mark}　最近活跃: {_fmt_datetime(device.get('last_seen_at'))}")
        self.device_list_label.setText("\n".join(lines) if lines else "")

    def _on_status_error(self, exc: Exception) -> None:
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("刷新授权状态")
        if isinstance(exc, client.LicenseError):
            QMessageBox.warning(self, "授权状态", f"获取失败：{exc.message}")
        else:
            _log.warning("get_status 失败: %s", exc)
            QMessageBox.warning(self, "授权状态", "网络异常，请稍后重试")

    # ── 激活 / 解绑 ───────────────────────────────────────────
    def _on_activate(self) -> None:
        try:
            from .activation_dialog import ActivationDialog
        except ImportError:
            from src.licensing.activation_dialog import ActivationDialog
        dlg = ActivationDialog(parent=self)
        if dlg.exec() == ActivationDialog.DialogCode.Accepted and dlg.activated:
            self.refresh_local()

    def _on_deactivate(self) -> None:
        token = store.load_token()
        if not token:
            return
        confirm = QMessageBox.question(
            self,
            "解绑本设备",
            "解绑后本设备将切换为免费版（累计可处理 100 个文件、1 个 vault）。\n"
            "已有数据不受影响。确定解绑本设备？",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.unbind_btn.setEnabled(False)
        self.unbind_btn.setText("正在解绑…")
        self._start_worker(
            lambda: client.deactivate(token), self._on_deactivated, self._on_deactivate_error
        )

    def _on_deactivated(self, _res) -> None:
        self.unbind_btn.setEnabled(True)
        self.unbind_btn.setText("解绑本设备")
        try:
            quota.sync_from_stock()  # 取消激活：库存快照
        except Exception as exc:
            _log.warning("额度库存快照失败（忽略）: %s", exc)
        store.clear()
        state.set_mode(state.Mode.FREE, "unbound")
        QMessageBox.information(
            self,
            "已解绑",
            "本设备已解绑，席位已释放。\n当前为免费版：累计可处理 100 个文件、1 个 vault。\n已有数据不受影响。",
        )
        self.refresh_local()

    def _on_deactivate_error(self, exc: Exception) -> None:
        self.unbind_btn.setEnabled(True)
        self.unbind_btn.setText("解绑本设备")
        if isinstance(exc, client.LicenseError):
            QMessageBox.warning(self, "解绑失败", exc.message or "解绑失败，请稍后重试")
        else:
            _log.warning("deactivate 失败: %s", exc)
            QMessageBox.warning(self, "解绑失败", "网络异常，请稍后重试")

    # ── 后台执行 ──────────────────────────────────────────────
    def _start_worker(self, fn, on_done, on_fail) -> None:
        self._worker = AsyncWorker(fn, self)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(on_fail)
        self._worker.start()
