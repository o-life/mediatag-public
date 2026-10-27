"""应用重启助手。

为什么需要：**切换分析主题 = 切换数据文件**（每主题一个 `data.db` + 人脸目录），
而数据库/人脸目录是**启动期解析**的（`PathManager` / `ConfigManager` 构造时定路径）
→ 切换必须**冷切**：写指针后重启进程才生效。

- 打包态：`open -n <bundle>` 重新拉起，再由调用方退出当前进程。
- 开发态：后端是**独立进程**（start-api.sh），无法自动重开 → 返回 False，
  由调用方提示手动重启。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def bundle_path() -> Path:
    """打包态的 .app 路径（`Contents/MacOS/<exe>` 上溯三层）。"""
    return Path(sys.executable).parent.parent.parent


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def restart_application() -> bool:
    """重启应用。返回 True 表示**已发起重启**（调用方随后应退出进程）。"""
    if not is_packaged():
        return False
    try:
        subprocess.Popen(["open", "-n", str(bundle_path())])
        return True
    except Exception as exc:  # pragma: no cover - 依赖打包环境
        logger.warning("自动重启失败: %s", exc)
        return False


def manual_restart_hint() -> str:
    """开发态的手动重启指引。"""
    return ("./start-api.sh\n"
            "uv run --active python src/ui/desktop_app.py")


def prompt_restart(parent, theme_name: str) -> bool:
    """提示"切换主题需重启"并提供「退出并重启」。返回是否**已发起**重启。

    切换主题 = 换库 + 换人脸目录，二者都是启动期解析的 → 必须冷切。
    """
    from PySide6.QtWidgets import QApplication, QMessageBox

    box = QMessageBox(parent)
    box.setWindowTitle("需重启生效")
    box.setText(f"已切换分析主题为「{theme_name}」。\n\n"
                "数据库与人脸目录在启动时解析，需重启应用后生效。")
    restart_now = box.addButton("退出并重启", QMessageBox.AcceptRole)
    box.addButton("稍后手动重启", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is not restart_now:
        return False
    if restart_application():
        QApplication.instance().quit()
        return True
    QMessageBox.information(parent, "请手动重启", manual_restart_hint())
    return False
