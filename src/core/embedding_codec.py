"""人脸特征向量（embedding）的存储编解码。

为什么改：`actor_faces.embedding` 与 `actors.face_embedding` 原以 **JSON 文本**存储——
每个 float 约 19~21 字符（`-0.2939862012863159, `），512 维 ≈ **10 KB/行**，
37k 行 ≈ **435 MB**（实测占库 36%）。改用 **float32 BLOB** 后 512 维 = **2 KB/行**，
整体约 **5.7×** 更小。（SQLite 原生支持 BLOB；项目里较新的 `face_embeddings.embedding`
本来就是 BLOB，这次是把旧表对齐过来。）

读取**双向兼容**：BLOB 与 JSON 文本都能解，因此"旧库仍是 JSON、新库已是 BLOB"
的过渡期不会崩——这也是本改动可以安全分步落地的前提。

数值精度：JSON 往返是 float64，BLOB 为 float32（人脸匹配的阈值间距约 0.05，
float32 足够）；转换后需做一次匹配对照验证。
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np


def encode_embedding(vec: Any) -> Optional[bytes]:
    """向量 → float32 BLOB（``None`` 原样返回 ``None``）。"""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return arr.tobytes()


def decode_embedding(value: Any) -> Optional[np.ndarray]:
    """存储值 → float32 向量。

    兼容三种形态：
    - ``bytes`` / ``bytearray`` / ``memoryview`` → float32 BLOB（新格式）
    - ``str`` → JSON 文本（旧格式）
    - 其它（list / ndarray 等）→ 直接转数组

    无法解析时返回 ``None``（调用方原有的 try/except 语义得以保留）。
    **空值一律返回 None**——空 BLOB / 空数组不会抛错，若放行会让相似度计算静默出错。

    返回**可写**数组：调用方可能就地归一化，而 ``np.frombuffer`` 的结果是只读的。
    """
    if value is None:
        return None
    arr: Optional[np.ndarray]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            return None
        try:
            arr = np.frombuffer(raw, dtype=np.float32).copy()
        except Exception:
            return None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            arr = np.asarray(json.loads(text), dtype=np.float32)
        except Exception:
            return None
    else:
        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            return None

    return arr if arr.size else None
