"""Scene detection via frame-difference analysis.

Uses OpenCV to detect montage unit boundaries (scene changes) in a video.
"""

import subprocess
import cv2
from typing import List, Dict, Tuple
import logging

# Suppress opencv/numpy import warnings (installed at runtime via uv)

logger = logging.getLogger(__name__)


def _probe_duration_fps(video_path: str) -> Tuple[float, float]:
    """Get duration and FPS via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        duration = 0.0

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        fps_str = result.stdout.strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)
    except Exception:
        fps = 30.0

    return duration, fps


def detect_scenes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_frames: int = 15,
    sample_rate: int = 1,
) -> List[Dict[str, int]]:
    """Detect scene boundaries in a video file.

    Returns a list of montage unit dicts:
      [{"unit_index": 0, "start_frame": 0, "end_frame": 45, "duration_seconds": 1.5}, ...]

    Algorithm:
    1. Open video with OpenCV
    2. Downsample frames (every `sample_rate`th frame)
    3. Compute frame-to-frame difference via HSV histogram comparison
    4. When difference exceeds threshold, mark a new scene boundary
    5. Filter out scenes shorter than min_scene_frames
    6. 检测视频读取完整性，如果覆盖不足则返回空列表触发 fallback
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration, fps = _probe_duration_fps(video_path)
    logger.info(f"Video: {total_frames} frames, sampling every {sample_rate}")

    scenes = []
    scene_start = 0
    prev_hist = None
    frame_idx = 0
    processed = 0
    last_successful_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_successful_frame = frame_idx

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # Resize for faster histogram computation
        frame_resized = cv2.resize(frame, (64, 36))
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = 1.0 - score  # higher = more different

            if diff > (threshold / 100.0) and (frame_idx - scene_start) >= min_scene_frames:
                scenes.append({
                    "unit_index": len(scenes),
                    "start_frame": scene_start,
                    "end_frame": frame_idx - 1,
                    "duration_seconds": round((frame_idx - scene_start) / fps, 2),
                })
                scene_start = frame_idx

        prev_hist = hist
        frame_idx += 1
        processed += 1

    # Close last scene if any frames remain
    if scene_start < frame_idx and (frame_idx - scene_start) >= min_scene_frames:
        scenes.append({
            "unit_index": len(scenes),
            "start_frame": scene_start,
            "end_frame": frame_idx - 1,
            "duration_seconds": round((frame_idx - scene_start) / fps, 2),
        })

    cap.release()
    
    # 检测视频读取完整性
    coverage_ratio = last_successful_frame / total_frames if total_frames > 0 else 0
    logger.info(f"Detected {len(scenes)} scenes, coverage: {coverage_ratio:.1%} ({last_successful_frame}/{total_frames} frames)")
    
    # 如果覆盖率低于 80%，说明视频读取有问题，返回空列表触发 fallback
    if coverage_ratio < 0.8:
        logger.warning(f"视频覆盖率不足 ({coverage_ratio:.1%})，将使用均匀抽帧 fallback")
        return []
    
    return scenes


def estimate_duration_from_frames(start_frame: int, end_frame: int, fps: float = 30.0) -> float:
    """Estimate duration in seconds from frame range."""
    if fps <= 0:
        fps = 30.0
    return round((end_frame - start_frame) / fps, 2)
