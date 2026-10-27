"""媒体类型常量与判定 —— 唯一定义点。

背景：PHOTO_EXTENSIONS/VIDEO_EXTENSIONS 原先在 api/main.py 与 dashboard.py 各自
定义（双份维护债）。影楼基础包（S2）起统一从本模块导入；两处旧定义在并行
改动收敛后替换为 from core.media_types import ...。

RAW（影楼基础包 S2）：CR3/ARW/NEF/RAF/DNG 五家（ORF 等长尾进 backlog）。
RAW 归档保存不转码，管理/初筛用内嵌预览（见 core.raw_preview）；
因此 RAW 不并入 PHOTO_EXTENSIONS（现有照片管线按 cv2 可读设计），
由导入流显式分流：is_raw() → 提取预览 → 以"photo"语义进入展示层。
"""

from pathlib import Path
from typing import Optional

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".ts", ".m4v"}

# 常规照片（cv2/Pillow 可直接解码）
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif", ".bmp"}

# 相机 RAW（影楼基础包 S2）：原格式归档 + 内嵌预览展示
RAW_EXTENSIONS = {".cr3", ".arw", ".nef", ".raf", ".dng"}

# 兼容别名：与既有 api/main.py 行为一致（不含 RAW，导入扫描行为零变化）
ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | PHOTO_EXTENSIONS

# S2 起的完整扫描面（含 RAW；导入管线挂接 RAW 分流后使用）
ALL_SUPPORTED_EXTENSIONS = ALL_MEDIA_EXTENSIONS | RAW_EXTENSIONS


def _suffix(path) -> str:
    return Path(path).suffix.lower()


def is_video(path) -> bool:
    return _suffix(path) in VIDEO_EXTENSIONS


def is_photo(path) -> bool:
    """常规照片（不含 RAW——RAW 需走预览提取路径）。"""
    return _suffix(path) in PHOTO_EXTENSIONS


def is_raw(path) -> bool:
    return _suffix(path) in RAW_EXTENSIONS


def is_media(path) -> bool:
    """属于本项目处理的媒体（含 RAW——RAW 经预览分流进入展示层）。"""
    return _suffix(path) in (ALL_MEDIA_EXTENSIONS | RAW_EXTENSIONS)


# ── 内容嗅探 ────────────────────────────────────────────────────────────
# 照片库 originals/ 里的文件是 UUID 名、无扩展名；文件对话框选完后需要
# 按内容还原真实类型，否则无扩展名/错误扩展名的文件进不了导入管线。

_FTYP_BRAND_EXT = {
    "heic": ".heic", "mif1": ".heic", "msf1": ".heif", "heif": ".heif",
    "hevc": ".heic", "avif": ".avif",
    "mp42": ".mp4", "mp41": ".mp4", "isom": ".mp4", "mp4v": ".mp4",
    "MSNV": ".mp4", "avc1": ".mp4", "dash": ".mp4",
    "qt": ".mov",
}


def sniff_media_ext(path) -> Optional[str]:
    """按文件头嗅探媒体真实扩展名（小写含点）；不认识返回 None。

    读取前 16 字节：JPEG/PNG/GIF/WEBP/BMP 魔数，ISO-BMFT（HEIC/MP4/MOV）
    看偏移 4-8 的 brand。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if len(head) < 12:
        return None
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if head[:4] == b"GIF8":
        return ".gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:2] == b"BM":
        return ".bmp"
    # ISO-BMFT 容器：HEIC / MP4 / MOV（brand 在偏移 4-8）
    if head[4:8] == b"ftyp":
        brand = head[8:12].decode("ascii", errors="replace").strip()
        return _FTYP_BRAND_EXT.get(brand)
    return None
