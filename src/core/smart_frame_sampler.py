"""Smart frame sampling with blur detection and retry.

Implements the "intelligent frame sampling" promised by the
face-quality-filter change: instead of blindly picking equidistant frames,
each sampled frame is checked for sharpness (Laplacian variance). Blurry
frames are retried at nearby offsets (± retry_window_seconds) up to
``max_retries`` times, so the resulting frame set is much more likely to be
usable for face detection and VLM analysis.

Reuses:
    - ``src.core.video_quality_analyzer.calculate_sharpness``
    - ``src.core.video_quality_analyzer.get_threshold_by_resolution``
"""

import cv2
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.video_quality_analyzer import (
    calculate_sharpness,
    get_threshold_by_resolution,
)

logger = logging.getLogger(__name__)


class SmartFrameSampler:
    """Sample frames from a video range, retrying blurry samples nearby.

    Interface mirrors ``src/core/frame_sampler.py::sample_frames`` so it can
    be used as a drop-in upgrade. When sharpness filtering is disabled or the
    video cannot be analyzed, it falls back to plain equidistant sampling.
    """

    def __init__(
        self,
        sharpness_threshold: Optional[float] = None,
        retry_on_blur: bool = True,
        max_retries: int = 3,
        retry_window_seconds: float = 10.0,
        min_kept_frames: int = 3,
    ) -> None:
        self.sharpness_threshold = sharpness_threshold
        self.retry_on_blur = retry_on_blur
        self.max_retries = max_retries
        self.retry_window_seconds = retry_window_seconds
        self.min_kept_frames = min_kept_frames

    # ------------------------------------------------------------------
    # Public API (mirrors frame_sampler.sample_frames)
    # ------------------------------------------------------------------

    def sample_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        n_frames: int = 5,
        output_dir: Optional[str] = None,
        min_kept_frames: Optional[int] = None,
        max_resample_attempts: int = 2,
    ) -> List[str]:
        """Sample N frames from a frame range, retrying blurry samples.

        Args:
            video_path: Path to the video file.
            start_frame: Inclusive first frame index.
            end_frame: Inclusive last frame index.
            n_frames: Number of frames to sample.
            output_dir: Directory to save frames into.
            min_kept_frames: Minimum frames required before giving up on a
                range (defaults to ``self.min_kept_frames``).
            max_resample_attempts: Legacy param (kept for interface parity);
                blur retry is bounded by ``self.max_retries`` instead.

        Returns:
            List of saved frame file paths.
        """
        min_kept = min_kept_frames or self.min_kept_frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            start_frame = max(0, start_frame)
            end_frame = min(max(0, total_frames - 1), end_frame)
            if start_frame > end_frame:
                return []

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Resolution-based threshold, unless an explicit one was given.
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            threshold = self.sharpness_threshold
            if threshold is None:
                if width > 0 and height > 0:
                    threshold = get_threshold_by_resolution(width, height)
                else:
                    threshold = 300.0

            step = (
                (end_frame - start_frame) / max(1, n_frames - 1)
                if n_frames > 1
                else 0
            )

            frame_paths: List[str] = []
            frame_mappings: Dict[str, Dict] = {}
            for i in range(n_frames):
                target = int(start_frame + i * step)
                path, frame_num, retried = self._sample_one_sharp_frame(
                    cap, video_path, target, start_frame, end_frame,
                    fps, threshold, output_dir,
                )
                if path is None:
                    continue
                frame_paths.append(path)
                frame_mappings[path] = {
                    "video_frame_number": frame_num,
                    "timestamp": frame_num / fps,
                }
                if retried:
                    logger.debug(
                        "Sampled frame %d after %d blur retry(ies) -> %s",
                        frame_num, retried, path,
                    )

            if len(frame_paths) < min_kept:
                logger.warning(
                    "Smart sampling kept only %d/%d sharp frames "
                    "(threshold=%.1f); falling back to equidistant sampling",
                    len(frame_paths), n_frames, threshold,
                )
                frame_paths, frame_mappings = self._fallback_equidistant(
                    cap, video_path, start_frame, end_frame, n_frames,
                    fps, output_dir,
                )

            self._save_mappings(frame_mappings, output_dir)
            logger.info(
                "SmartSampled %d frames from [%d, %d] (threshold=%.1f)",
                len(frame_paths), start_frame, end_frame, threshold,
            )
            return frame_paths
        finally:
            cap.release()

    def sample_frames_per_unit(
        self,
        video_path: str,
        units: List[Dict],
        n_frames: int = 5,
        output_base_dir: Optional[str] = None,
    ) -> Dict[int, List[str]]:
        """Sample frames for all montage units (mirrors frame_sampler)."""
        if not output_base_dir:
            output_base_dir = "output/frames"

        results: Dict[int, List[str]] = {}
        for unit in units:
            ui = unit["unit_index"]
            unit_dir = os.path.join(output_base_dir, f"unit_{ui}")
            paths = self.sample_frames(
                video_path,
                unit["start_frame"],
                unit["end_frame"],
                n_frames=n_frames,
                output_dir=unit_dir,
            )
            results[ui] = paths
            unit["frame_paths"] = paths
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_one_sharp_frame(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        target: int,
        start_frame: int,
        end_frame: int,
        fps: float,
        threshold: float,
        output_dir: Optional[str],
    ) -> Tuple[Optional[str], int, int]:
        """Sample one sharp frame near ``target``, retrying within a window.

        Returns (path, frame_num, retry_count). ``path`` is None if no sharp
        frame was found in the window.
        """
        if not self.retry_on_blur:
            path = self._read_and_save(cap, target, output_dir)
            return path, target, 0

        # Candidate offsets: target first, then expanding ± window.
        candidates = self._candidate_offsets(target, start_frame, end_frame)

        for attempt, frame_num in enumerate(candidates):
            sharpness = self._read_sharpness(cap, frame_num)
            if sharpness is None:
                continue
            if sharpness >= threshold or attempt >= self.max_retries:
                # Accept: sharp enough, or out of retries — take the best seen.
                path = self._read_and_save(cap, frame_num, output_dir)
                return path, frame_num, attempt
            logger.debug(
                "Frame %d blurry (sharpness=%.1f < %.1f), retrying...",
                frame_num, sharpness, threshold,
            )

        return None, target, 0

    def _candidate_offsets(
        self,
        target: int,
        start_frame: int,
        end_frame: int,
    ) -> List[int]:
        """Ordered candidate frame numbers: target, then ±k within window."""
        max_offset = max(1, int(self.retry_window_seconds * 30))  # assume ~30fps
        seen = set()
        ordered = [target]
        seen.add(target)

        for k in range(1, max_offset + 1):
            for cand in (target - k, target + k):
                if start_frame <= cand <= end_frame and cand not in seen:
                    seen.add(cand)
                    ordered.append(cand)
            # Stop early once we have enough candidates for retries.
            if len(ordered) >= 1 + self.max_retries * 2:
                break
        return ordered

    def _read_sharpness(
        self,
        cap: cv2.VideoCapture,
        frame_num: int,
    ) -> Optional[float]:
        """Read a frame and return its sharpness, or None on read failure."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        sharpness = calculate_sharpness(frame)
        del frame
        return sharpness

    def _read_and_save(
        self,
        cap: cv2.VideoCapture,
        frame_num: int,
        output_dir: Optional[str],
    ) -> Optional[str]:
        """Read and save a frame. Returns the saved path or None."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        fname = f"frame_{frame_num}.jpg"
        fpath = os.path.join(output_dir, fname) if output_dir else fname
        cv2.imwrite(fpath, frame)
        del frame
        return fpath

    def _fallback_equidistant(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        start_frame: int,
        end_frame: int,
        n_frames: int,
        fps: float,
        output_dir: Optional[str],
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """Plain equidistant sampling (used when sharp filtering fails)."""
        from src.core.frame_sampler import sample_frames

        paths = sample_frames(
            video_path, start_frame, end_frame,
            n_frames=n_frames, output_dir=output_dir,
            min_kept_frames=self.min_kept_frames,
        )
        mappings: Dict[str, Dict] = {}
        for p in paths:
            try:
                frame_num = int(Path(p).stem.split("_")[-1])
            except (ValueError, IndexError):
                continue
            mappings[p] = {
                "video_frame_number": frame_num,
                "timestamp": frame_num / fps,
            }
        return paths, mappings

    @staticmethod
    def _save_mappings(
        frame_mappings: Dict[str, Dict],
        output_dir: Optional[str],
    ) -> None:
        """Merge frame mappings into the parent dir's frame_mappings.json."""
        if not output_dir or not frame_mappings:
            return

        video_output_dir = str(Path(output_dir).parent)
        mappings_file = os.path.join(video_output_dir, "frame_mappings.json")

        all_mappings: Dict = {}
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, "r", encoding="utf-8") as f:
                    all_mappings = json.load(f)
            except Exception as e:
                logger.warning("Failed to load existing frame_mappings.json: %s", e)

        all_mappings.update(frame_mappings)
        try:
            with open(mappings_file, "w", encoding="utf-8") as f:
                json.dump(all_mappings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save frame_mappings.json: %s", e)
