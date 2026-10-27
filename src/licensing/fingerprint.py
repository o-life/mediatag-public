"""设备指纹：IOPlatformUUID + 固定盐值 → SHA-256 十六进制（64 位）。

同一台机器跨版本/重装稳定不变（UUID 来自主板固件）；不同机器 UUID 全局唯一。
"""

import hashlib
import re
import subprocess

# 固定盐值：换盐 = 全部设备指纹变化 = 所有激活码在服务端失配，绝不可更改。
_SALT = "MediaTag/license/v1"

_UUID_RE = re.compile(r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f-]+)"')


class FingerprintError(RuntimeError):
    """无法读取设备指纹（ioreg 不可用或输出异常）。"""


def get_fingerprint() -> str:
    """返回 64 位十六进制设备指纹。失败抛 FingerprintError（调用方按锁定处理）。"""
    try:
        proc = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FingerprintError(f"无法读取 IOPlatformUUID: {exc}") from exc
    match = _UUID_RE.search(proc.stdout)
    if not match:
        raise FingerprintError("ioreg 输出中未找到 IOPlatformUUID")
    raw = (match.group(1) + _SALT).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
