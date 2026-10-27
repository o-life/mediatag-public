"""Video quality analysis module.

Analyzes video quality by extracting resolution, frame rate, bitrate,
and computing average sharpness via Laplacian variance. Provides dynamic
threshold calculation based on video resolution.
"""

import cv2
import numpy as np
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoQualityInfo:
    """Container for video quality analysis results."""

    resolution: Tuple[int, int]   # (width, height)
    fps: float
    bitrate: int
    avg_sharpness: float
    threshold: float
    is_low_quality: bool


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance.

    Algorithm:
        1. Convert to grayscale
        2. Apply Laplacian operator (cv2.Laplacian)
        3. Compute variance (np.var)

    Higher variance indicates a sharper image.

    Args:
        image: BGR image as a numpy array (H, W, 3).

    Returns:
        Sharpness score (float). Returns 0.0 if the image is invalid.
    """
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_sharpness_optimized(image: np.ndarray, target_size: int = 256) -> float:
    """Calculate sharpness with downscaling for speed.

    缩放到 target_size×target_size 后再算 Laplacian 方差。
    缩放后纹理细节减少，但相对清晰度排序基本不变（适合批量筛选/缓存场景），
    速度约为原始计算的 5-10 倍。

    Args:
        image: BGR image as a numpy array (H, W, 3).
        target_size: 缩放的目标边长（默认 256）。

    Returns:
        Sharpness score (float). Returns 0.0 if the image is invalid.
    """
    if image is None or image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    # 只在图像大于目标尺寸时缩放（小图直接算）
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return calculate_sharpness(image)


def calculate_sharpness_batch(
    images: List[np.ndarray],
    max_workers: int = 4,
    optimized: bool = True,
) -> List[float]:
    """Calculate sharpness for multiple images in parallel.

    Args:
        images: List of BGR images.
        max_workers: 并行线程数。
        optimized: 是否使用缩放的快速算法（默认 True）。

    Returns:
        List of sharpness scores, same order as input. 无效图像得 0.0。
    """
    if not images:
        return []

    fn = calculate_sharpness_optimized if optimized else calculate_sharpness
    if len(images) == 1 or max_workers <= 1:
        return [fn(img) for img in images]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(images))) as executor:
        return list(executor.map(fn, images))


class SharpnessCache:
    """内存缓存：文件路径 → 清晰度分数（避免重复计算）。

    简单 LRU：超出 max_size 时清空最老的一半。
    线程安全：读写用锁保护。
    """

    def __init__(self, max_size: int = 4096):
        self._cache: Dict[str, float] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, path: str) -> Optional[float]:
        with self._lock:
            return self._cache.get(path)

    def put(self, path: str, sharpness: float) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size:
                # 简单 LRU 近似：清掉一半最旧的（dict 保持插入序）
                for k in list(self._cache.keys())[: self._max_size // 2]:
                    del self._cache[k]
            self._cache[path] = sharpness

    def compute(self, path: str, optimized: bool = True) -> float:
        """查缓存，未命中则读图计算并缓存。"""
        cached = self.get(path)
        if cached is not None:
            return cached
        try:
            img = cv2.imread(path)
        except Exception as e:
            logger.warning(f"cv2.imread failed for sharpness compute: {path}, error: {e}")
            return 0.0
        if img is None:
            return 0.0
        score = calculate_sharpness_optimized(img) if optimized else calculate_sharpness(img)
        self.put(path, score)
        return score

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def get_threshold_by_resolution(width: int, height: int) -> float:
    """Determine sharpness threshold based on video resolution.

    Strategy:
        - 4K  (3840×2160): threshold ≈ 500
        - 1080p (1920×1080): threshold = 300  (reference)
        - 720p (1280×720):  threshold ≈ 150
        - Other resolutions: linearly interpolated by pixel count,
          with a minimum floor of 100.

    Formula:
        threshold = max(100, base_threshold * (total_pixels / reference_pixels))

    Args:
        width:  Video width in pixels.
        height: Video height in pixels.

    Returns:
        Recommended sharpness threshold (float).
    """
    total_pixels = width * height
    reference_pixels = 1920 * 1080  # 1080p as baseline
    base_threshold = 300.0

    threshold = max(100.0, base_threshold * (total_pixels / reference_pixels))
    return threshold


class VideoQualityAnalyzer:
    """Analyzes overall video quality by sampling frames."""

    # Number of frames to sample for average sharpness estimation
    DEFAULT_SAMPLE_FRAMES = 5

    def analyze(
        self,
        video_path: str,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES,
    ) -> VideoQualityInfo:
        """Analyze video quality.

        Opens the video, reads metadata (resolution, fps, bitrate), then
        randomly samples *sample_frames* frames to compute average sharpness.
        A dynamic threshold is derived from the resolution, and the video is
        flagged as low-quality when its average sharpness falls below that
        threshold.

        Args:
            video_path:    Path to the video file.
            sample_frames: How many frames to randomly sample for sharpness
                           estimation (default 5).

        Returns:
            VideoQualityInfo with all quality metrics populated.

        Raises:
            ValueError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            # --- Metadata ---
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Bitrate estimation: file_size / duration (bits per second)
            bitrate = self._estimate_bitrate(cap, video_path)

            # --- Sharpness sampling ---
            avg_sharpness = self._sample_sharpness(cap, total_frames, sample_frames)

            # --- Threshold & quality flag ---
            threshold = get_threshold_by_resolution(width, height)
            is_low_quality = avg_sharpness < threshold

            quality_info = VideoQualityInfo(
                resolution=(width, height),
                fps=fps,
                bitrate=bitrate,
                avg_sharpness=avg_sharpness,
                threshold=threshold,
                is_low_quality=is_low_quality,
            )

            logger.info(
                "Video quality: %dx%d @ %.2f fps, bitrate=%d, "
                "avg_sharpness=%.2f, threshold=%.2f, low_quality=%s",
                width, height, fps, bitrate,
                avg_sharpness, threshold, is_low_quality,
            )

            return quality_info

        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_bitrate(cap: cv2.VideoCapture, video_path: str) -> int:
        """Estimate bitrate in bits-per-second from file size and duration."""
        import os

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            return 0

        duration = total_frames / fps  # seconds
        if duration <= 0:
            return 0

        try:
            file_size = os.path.getsize(video_path)  # bytes
        except OSError:
            return 0

        bitrate = int((file_size * 8) / duration)
        return bitrate

    @staticmethod
    def _sample_sharpness(
        cap: cv2.VideoCapture,
        total_frames: int,
        sample_count: int,
    ) -> float:
        """Randomly sample frames and return average sharpness."""
        if total_frames <= 0:
            return 0.0

        # Clamp sample count to available frames
        actual_count = min(sample_count, total_frames)

        # Pick unique random frame indices
        frame_indices = random.sample(range(total_frames), actual_count)

        sharpness_values = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            sharpness_values.append(calculate_sharpness(frame))

        if not sharpness_values:
            return 0.0

        return sum(sharpness_values) / len(sharpness_values)
