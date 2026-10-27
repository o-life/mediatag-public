"""RAW 内嵌预览提取（rawpy/LibRaw）—— 影楼基础包 S2。

产品语义：RAW 原文件归档保存不转码（精修仍在 LR/PS），管理/初筛用**内嵌预览**：
rawpy.extract_thumb() 直出 JPEG（相机写入的快速路径，S1 spike 实证 ~1ms/张）。

- 部分格式无 JPEG 预览时 LibRaw 给位图（BITMAP），此处转码为 JPEG 兜底；
- 无任何预览 → RawPreviewError（调用方决定跳过/占位）；
- 本模块只做提取，不做缓存策略与导入挂接（见导入管线）。
"""

import logging
from pathlib import Path

import rawpy

_log = logging.getLogger(__name__)


class RawPreviewError(Exception):
    """RAW 内嵌预览提取失败（无预览/格式不支持/文件损坏）。"""


def extract_preview(raw_path, out_jpeg_path, jpeg_quality: int = 90) -> Path:
    """提取 RAW 内嵌预览并写入 out_jpeg_path（JPEG）。成功返回输出路径。

    - 内嵌 JPEG 预览：直接落盘（零转码，最快路径）；
    - 仅有位图预览：cv2 转码 JPEG（jpeg_quality）；
    - 失败抛 RawPreviewError，不产生半截文件。
    """
    raw_path = Path(raw_path)
    out_jpeg_path = Path(out_jpeg_path)
    if not raw_path.exists():
        raise RawPreviewError(f"RAW 文件不存在: {raw_path}")

    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
    except rawpy.LibRawError as exc:
        raise RawPreviewError(f"{raw_path.name}: 内嵌预览提取失败（{exc}）") from exc

    if thumb.format == rawpy.ThumbFormat.JPEG:
        data = bytes(thumb.data)
        if not data:
            raise RawPreviewError(f"{raw_path.name}: 内嵌 JPEG 预览为空")
        _write_atomic(out_jpeg_path, data)
        _log.debug("RAW 预览(JPEG 直出): %s -> %s", raw_path.name, out_jpeg_path.name)
        return out_jpeg_path

    if thumb.format == rawpy.ThumbFormat.BITMAP:
        import cv2

        bitmap = thumb.data  # ndarray RGB
        if bitmap is None or bitmap.size == 0:
            raise RawPreviewError(f"{raw_path.name}: 位图预览为空")
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RawPreviewError(f"{raw_path.name}: 位图预览转码 JPEG 失败")
        _write_atomic(out_jpeg_path, encoded.tobytes())
        _log.debug("RAW 预览(位图转码): %s -> %s", raw_path.name, out_jpeg_path.name)
        return out_jpeg_path

    raise RawPreviewError(f"{raw_path.name}: 未知预览格式 {thumb.format}")


def _write_atomic(out_path: Path, data: bytes) -> None:
    """先写临时文件再 rename，避免调用方读到半截文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out_path)
