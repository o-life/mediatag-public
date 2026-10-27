"""Tests for SmartFrameSampler (blur detection + retry sampling).

Uses a synthetic video whose frames alternate sharp/blurry so the sampler's
retry logic can be exercised deterministically without real footage.
"""

import cv2
import numpy as np
import pytest

from src.core.smart_frame_sampler import SmartFrameSampler


@pytest.fixture
def mixed_quality_video(tmp_path):
    """Create a short video: even frames sharp, odd frames blurry.

    Returns the video path. 100 frames @ 10fps = 10s.
    """
    path = str(tmp_path / "mixed.mp4")
    writer = None
    for i in range(100):
        # Sharp: random noise with an edge pattern.
        sharp = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        # Ensure high-frequency content (sharp).
        cv2.rectangle(sharp, (10, 10), (150, 110), (255, 255, 255), 2)
        frame = sharp if i % 2 == 0 else cv2.GaussianBlur(sharp, (31, 31), 10)
        if writer is None:
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
            )
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sharp_video(tmp_path):
    """A video where every frame is sharp."""
    path = str(tmp_path / "sharp.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    for _ in range(100):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestSmartFrameSampler:
    """Tests for SmartFrameSampler."""

    def test_returns_all_sharp_frames(self, sharp_video, tmp_path):
        """All-sharp video: every sampled frame should be saved."""
        out = str(tmp_path / "out")
        sampler = SmartFrameSampler(
            sharpness_threshold=100.0,
            retry_on_blur=True,
            max_retries=3,
        )
        paths = sampler.sample_frames(sharp_video, 0, 99, n_frames=5, output_dir=out)

        assert len(paths) == 5
        for p in paths:
            assert p.endswith(".jpg")

    def test_recovers_blurry_frames(self, mixed_quality_video, tmp_path):
        """Blurry targets should be retried at nearby sharp frames."""
        out = str(tmp_path / "out")
        sampler = SmartFrameSampler(
            sharpness_threshold=100.0,
            retry_on_blur=True,
            max_retries=3,
        )
        paths = sampler.sample_frames(mixed_quality_video, 0, 99, n_frames=5, output_dir=out)

        # Sharpness filter should keep at least 3 sharp frames (min_kept default).
        assert len(paths) >= 3
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            assert img is not None
            # Saved frames should be the sharp ones (high Laplacian variance).
            assert cv2.Laplacian(img, cv2.CV_64F).var() >= 100.0

    def test_blur_retry_disabled_saves_original(self, mixed_quality_video, tmp_path):
        """With retry disabled, frames are saved as-is (even blurry)."""
        out = str(tmp_path / "out")
        sampler = SmartFrameSampler(
            sharpness_threshold=100.0,
            retry_on_blur=False,
        )
        paths = sampler.sample_frames(mixed_quality_video, 0, 99, n_frames=5, output_dir=out)
        assert len(paths) == 5

    def test_invalid_video_returns_empty(self, tmp_path):
        """Non-existent video should return an empty list."""
        sampler = SmartFrameSampler()
        assert sampler.sample_frames("/nonexistent.mp4", 0, 100, n_frames=3) == []

    def test_empty_range_returns_empty(self, sharp_video, tmp_path):
        """An empty frame range should return an empty list."""
        sampler = SmartFrameSampler()
        # end_frame < start_frame → empty
        assert sampler.sample_frames(sharp_video, 50, 10, n_frames=3) == []

    def test_sample_frames_per_unit(self, sharp_video, tmp_path):
        """sample_frames_per_unit should map units to frame path lists."""
        base = str(tmp_path / "frames")
        units = [
            {"unit_index": 0, "start_frame": 0, "end_frame": 49},
            {"unit_index": 1, "start_frame": 50, "end_frame": 99},
        ]
        sampler = SmartFrameSampler(sharpness_threshold=100.0)
        results = sampler.sample_frames_per_unit(sharp_video, units, n_frames=4, output_base_dir=base)

        assert set(results.keys()) == {0, 1}
        assert len(results[0]) == 4
        assert len(results[1]) == 4
        # Units should have been updated with their frame paths.
        assert units[0]["frame_paths"] == results[0]
