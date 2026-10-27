"""缓存管理器（Phase 6）。

管理帧图片缓存、VLM 结果缓存的统计与清理。
设计原则：
- 帧目录在原视频旁（视频名_frames/），不复制到统一缓存（避免磁盘翻倍）
- VLM 结果缓存存数据库（FileAnalysis.analysis_details 的 vlm_raw），不额外存文件
- 提供统计/清理入口，供设置页和调度器使用
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# 抽帧目录约定：<视频同目录>/<视频名>_frames/（见 scan_frame_caches 与 api/main.py 抽帧）。
# 导入时必须排除它，否则每一帧（.jpg）都会被当成照片逐张登记。
_FRAMES_SUFFIX = "_frames"
_DEFAULT_VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v",
}


def is_frames_dir(dir_path, video_exts=None) -> bool:
    """目录是否为某视频的抽帧目录。

    方案 B：目录名以 ``_frames`` 结尾，**且**同目录下存在同名视频文件
    （``<stem>.mp4`` 等）才算 —— 避免误伤用户真叫 ``xxx_frames`` 的相册目录。
    """
    p = Path(dir_path)
    name = p.name
    if not name.endswith(_FRAMES_SUFFIX):
        return False
    stem = name[: -len(_FRAMES_SUFFIX)]
    if not stem:
        return False
    exts = {e.lower() for e in (video_exts or _DEFAULT_VIDEO_EXTS)}
    parent = p.parent
    for ext in exts:
        if (parent / f"{stem}{ext}").exists() or (parent / f"{stem}{ext.upper()}").exists():
            return True
    return False


def is_inside_frames_dir(file_path, video_exts=None) -> bool:
    """文件是否位于某个抽帧目录之下（用于过滤直接传入的文件列表）。"""
    for parent in Path(file_path).parents:
        if parent.name.endswith(_FRAMES_SUFFIX) and is_frames_dir(parent, video_exts):
            return True
    return False


def scan_frame_caches(scan_dir: Optional[str] = None) -> List[Dict]:
    """扫描所有帧缓存目录（视频名_frames/）。

    Args:
        scan_dir: 扫描根目录;None 时扫描工作区数据根下的视频库。

    Returns:
        [{"path", "size_mb", "file_count", "video_name"}, ...]
    """
    roots = []
    if scan_dir:
        roots.append(Path(scan_dir))
    else:
        # 默认扫描工作区数据根(历史版本曾硬编码外部卷,卷不存在时
        # 静默空转;改为跟随 PathManager,打包/开发两态一致)
        try:
            from .path_manager import get_path_manager
            roots.append(Path(get_path_manager().base_dir))
        except Exception:
            roots.append(Path("data"))

    result = []
    seen = set()
    for root in roots:
        try:
            for d in root.rglob("*_frames"):
                if not d.is_dir():
                    continue
                # 去重（多根目录可能重叠）
                if str(d) in seen:
                    continue
                seen.add(str(d))
                result.append({
                    "path": str(d),
                    "video_name": d.name[:-7],  # 去掉 _frames 后缀
                })
        except Exception as e:
            logger.warning(f"扫描帧缓存失败 {root}: {e}")
    return result


def get_frame_cache_cleanup_stats(max_age_days: int = 30, scan_dir: Optional[str] = None) -> Dict:
    """一次性统计帧缓存：总量 + 超过保留期（可清理）的数量与占用。

    供设置页「帧缓存 → 统计」使用（按"清理阈值(天)"给出可清理量）。

    Returns:
        {"total_count", "total_size_mb", "cleanable_count", "cleanable_size_mb"}
    """
    cutoff = time.time() - max_age_days * 86400
    total_count = 0
    total_bytes = 0
    cleanable_count = 0
    cleanable_bytes = 0
    for d in scan_frame_caches(scan_dir):
        p = Path(d["path"])
        size = estimate_dir_size(p)
        total_count += 1
        total_bytes += size
        try:
            expired = p.stat().st_mtime <= cutoff
        except Exception:
            expired = False
        if expired:
            cleanable_count += 1
            cleanable_bytes += size
    return {
        "total_count": total_count,
        "total_size_mb": round(total_bytes / 1e6, 2),
        "cleanable_count": cleanable_count,
        "cleanable_size_mb": round(cleanable_bytes / 1e6, 2),
    }


def clean_frame_cache(video_path: Optional[str] = None, max_age_days: int = 30) -> Dict:
    """清理帧缓存目录。

    Args:
        video_path: 指定视频的帧目录；None = 清理所有超过保留期的缓存
        max_age_days: 保留天数（默认 30 天）

    Returns:
        {"removed": [路径], "freed_mb": 0}
    """
    removed = []
    freed = 0
    cutoff = time.time() - max_age_days * 86400

    if video_path:
        # 指定单个视频的帧目录
        p = Path(video_path)
        frames_dir = p.parent / f"{p.stem}_frames"
        candidates = [frames_dir] if frames_dir.exists() else []
    else:
        # 扫描所有，按过期时间清理
        candidates = [Path(d["path"]) for d in scan_frame_caches()]

    for d in candidates:
        try:
            # 检查是否过期（mtime）
            mtime = d.stat().st_mtime
            if mtime > cutoff:
                continue
            size = estimate_dir_size(d)
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
            freed += size
            logger.info(f"清理帧缓存: {d} ({size/1e6:.1f}MB)")
        except Exception as e:
            logger.warning(f"清理帧缓存失败 {d}: {e}")

    return {"removed": removed, "freed_mb": round(freed / 1e6, 1)}


def estimate_dir_size(path: Path) -> int:
    """估算目录大小（字节）。"""
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception as e:
        logger.warning(f"估算目录大小失败 {path}: {e}")
    return total
