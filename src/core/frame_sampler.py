"""Frame sampling within montage units.

Extracts N frames per scene unit for VLM analysis.
"""

import cv2
import os
import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# 延迟初始化的NSFW检测器
_nsfw_detector = None

def _get_nsfw_detector():
    """获取NSFW检测器实例（延迟初始化）"""
    global _nsfw_detector
    if _nsfw_detector is None:
        try:
            from src.core.nsfw_detector import create_nsfw_detector
            _nsfw_detector = create_nsfw_detector(threshold=0.4)
            if _nsfw_detector is None:
                logger.info("NSFW检测器已停用（nsfw.detector=none），不过滤帧")
                return None
            logger.info("NSFW检测器初始化成功")
        except Exception as e:
            logger.warning(f"NSFW检测器初始化失败，将不过滤帧: {e}")
            return None
    return _nsfw_detector


def sample_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    n_frames: int = 5,
    output_dir: str = None,
    min_kept_frames: int = 3,
    max_resample_attempts: int = 2,
) -> List[str]:
    """Sample N equidistant frames from a frame range and save to disk.

    Returns list of saved file paths.
    
    Args:
        min_kept_frames: 最少保留帧数，低于此值会自动重采
        max_resample_attempts: 最大重采次数，避免无限循环
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    # Ensure output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Clamp frame range
    start_frame = max(0, start_frame)
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, end_frame)

    if start_frame > end_frame:
        cap.release()
        return []

    # Get video FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default fallback

    _get_nsfw_detector()  # 预热：确保 NSFW 模型已加载
    current_n_frames = n_frames
    attempt = 0
    frame_paths = []
    frame_mappings = {}
    
    while attempt <= max_resample_attempts:
        # Calculate equidistant intervals
        step = (end_frame - start_frame) / max(1, current_n_frames - 1) if current_n_frames > 1 else 0

        frame_paths = []
        frame_mappings = {}  # frame_path -> {video_frame_number, timestamp}
        filtered_count = 0
        
        for i in range(current_n_frames):
            frame_num = int(start_frame + i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            fname = f"frame_{frame_num}.jpg"
            fpath = os.path.join(output_dir, fname) if output_dir else fname
            cv2.imwrite(fpath, frame)
            
            # 保留所有帧，不在采样阶段进行 NSFW 过滤
            # NSFW 过滤在后续的两阶段检测器中进行，避免帧目录为空导致人脸识别失败
            frame_paths.append(fpath)
            
            # Calculate timestamp for this frame
            timestamp = frame_num / fps
            frame_mappings[fpath] = {
                "video_frame_number": frame_num,
                "timestamp": timestamp
            }
        
        # 检查保留帧数是否足够
        if len(frame_paths) >= min_kept_frames or attempt >= max_resample_attempts:
            if filtered_count > 0:
                logger.info(f"NSFW过滤: {filtered_count} 帧被过滤，保留 {len(frame_paths)} 帧")
            break
        
        # 保留帧不足，跳过一段时间后重新抽帧
        attempt += 1
        
        # 根据场景单元时长动态计算跳过时间
        scene_duration = (end_frame - start_frame) / fps
        if scene_duration > 300:  # 场景超过5分钟
            skip_seconds = 60 * attempt  # 每次跳过1分钟
        elif scene_duration > 120:  # 场景超过2分钟
            skip_seconds = 30 * attempt  # 每次跳过30秒
        else:
            skip_seconds = 10 * attempt  # 每次跳过10秒
        
        skip_frames = int(skip_seconds * fps)
        
        # 调整起始帧位置（跳过一段时间）
        new_start_frame = min(start_frame + skip_frames, end_frame - 100)  # 确保还有足够空间采样
        
        if new_start_frame >= end_frame:
            logger.warning(f"跳过 {skip_seconds} 秒后超出场景范围，停止重采")
            break
        
        logger.warning(f"保留帧数不足 ({len(frame_paths)} < {min_kept_frames})，第{attempt}次重采，跳过 {skip_seconds} 秒")
        
        # 清理已保存的帧文件
        for fpath in frame_paths:
            try:
                os.remove(fpath)
            except Exception:
                pass
        
        # 更新起始帧位置
        start_frame = new_start_frame

    cap.release()
    
    # Save frame mappings to JSON file in the parent directory (video_output_dir)
    if output_dir and frame_mappings:
        video_output_dir = str(Path(output_dir).parent)
        mappings_file = os.path.join(video_output_dir, "frame_mappings.json")
        
        # Load existing mappings if file exists
        all_mappings = {}
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    all_mappings = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing frame_mappings.json: {e}")
        
        # Merge new mappings
        all_mappings.update(frame_mappings)
        
        # Save updated mappings
        try:
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(all_mappings, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(frame_mappings)} frame mappings to {mappings_file}")
        except Exception as e:
            logger.error(f"Failed to save frame_mappings.json: {e}")
    
    logger.info(f"Sampled {len(frame_paths)} frames from [{start_frame}, {end_frame}]")
    return frame_paths


def sample_frames_per_unit(
    video_path: str,
    units: List[Dict],
    n_frames: int = 5,
    output_base_dir: str = None,
) -> Dict[int, List[str]]:
    """Sample frames for all montage units.

    Returns dict mapping unit_index → list of frame paths.
    """
    if not output_base_dir:
        output_base_dir = "output/frames"

    results = {}
    for unit in units:
        ui = unit["unit_index"]
        unit_dir = os.path.join(output_base_dir, f"unit_{ui}")
        paths = sample_frames(
            video_path,
            unit["start_frame"],
            unit["end_frame"],
            n_frames=n_frames,
            output_dir=unit_dir,
        )
        results[ui] = paths
        # Update unit with frame paths
        unit["frame_paths"] = paths

    return results
