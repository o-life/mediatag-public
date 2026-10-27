"""token 与授权信息的本地文件存储。

隐私要求（2026-09-22 用户定案）：本项目**不访问系统钥匙串**（不使用 keyring）——
授权数据只写本地文件 <应用数据目录>/license.json（权限 0600）。
任何"清除授权"操作 = 删除该文件。
"""

import json
import logging
import os
from pathlib import Path

KEY_TOKEN = "license_token"
KEY_INFO = "license_info"

_log = logging.getLogger("licensing.store")

# 测试可覆盖；None = 使用应用数据目录（path_manager.app_support_base）
_DIR_OVERRIDE: Path | None = None


def _store_file() -> Path:
    if _DIR_OVERRIDE is not None:
        base: Path | str = _DIR_OVERRIDE
    else:
        try:
            from ..core.path_manager import app_support_base
        except ImportError:
            from core.path_manager import app_support_base
        base = app_support_base()
    return Path(base) / "license.json"


def _read_all() -> dict:
    path = _store_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_all(data: dict) -> None:
    path = _store_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        _log.error("授权文件写入失败: %s", exc)


def _set(key: str, value: str) -> None:
    data = _read_all()
    data[key] = value
    _write_all(data)


def _get(key: str) -> str | None:
    return _read_all().get(key)


def _delete(key: str) -> None:
    data = _read_all()
    if key in data:
        data.pop(key, None)
        if data:
            _write_all(data)
        else:
            _remove_all()


def _remove_all() -> None:
    try:
        _store_file().unlink(missing_ok=True)
    except OSError as exc:
        _log.error("授权文件删除失败: %s", exc)


def save_token(token: str) -> None:
    _set(KEY_TOKEN, token)


def load_token() -> str | None:
    return _get(KEY_TOKEN)


def save_license_info(info: dict) -> None:
    _set(KEY_INFO, json.dumps(info, ensure_ascii=False))


def load_license_info() -> dict | None:
    raw = _get(KEY_INFO)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def clear() -> None:
    """清除授权：删除授权文件（token 与 license_info 一并清除）。"""
    _remove_all()
