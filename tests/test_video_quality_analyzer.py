"""Unit tests for the video quality analysis module."""

import cv2
import numpy as np
import pytest

from src.core.video_quality_analyzer import (
    VideoQualityAnalyzer,
    VideoQualityInfo,
    calculate_sharpness,
    get_threshold_by_resolution,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sharp_image() -> np.ndarray:
    """Create a synthetic sharp image (checkerboard pattern).

    A checkerboard has maximum high-frequency content, which produces
    a large Laplacian variance — similar to a sharp photograph.
    """
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)
    block = 8
    for y in range(0, size, block):
        for x in range(0, size, block):
            if ((x // block) + (y // block)) % 2 == 0:
                img[y:y + block, x:x + block] = 255
    return img


@pytest.fixture
def blurry_image(sharp_image: np.ndarray) -> np.ndarray:
    """Create a blurry image by heavily blurring the sharp image."""
    return cv2.GaussianBlur(sharp_image, (51, 51), 30)


@pytest.fixture
def solid_image() -> np.ndarray:
    """A uniform gray image — the least sharp possible."""
    return np.full((256, 256, 3), 128, dtype=np.uint8)


@pytest.fixture
def sample_video(tmp_path) -> str:
    """Create a short synthetic video for integration-style tests."""
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 25.0, (320, 240))

    for i in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw a moving white rectangle (adds edges → non-zero sharpness)
        x = (i * 10) % 280
        cv2.rectangle(frame, (x, 50), (x + 40, 190), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


# -----------------------------------------------------------------------
# calculate_sharpness
# -----------------------------------------------------------------------

class TestCalculateSharpness:
    """Tests for the calculate_sharpness() function."""

    def test_sharp_image_high_score(self, sharp_image: np.ndarray):
        """Sharp (checkerboard) image should produce a high sharpness score."""
        score = calculate_sharpness(sharp_image)
        assert score > 300, f"Expected sharp score > 300, got {score:.2f}"

    def test_blurry_image_low_score(self, blurry_image: np.ndarray):
        """Heavily blurred image should produce a low sharpness score."""
        score = calculate_sharpness(blurry_image)
        assert score < 150, f"Expected blurry score < 150, got {score:.2f}"

    def test_solid_image_near_zero(self, solid_image: np.ndarray):
        """Uniform image should have near-zero sharpness."""
        score = calculate_sharpness(solid_image)
        assert score < 10, f"Expected near-zero for solid image, got {score:.2f}"

    def test_sharp_vs_blurry_ordering(self, sharp_image, blurry_image):
        """Sharp image must score higher than blurry image."""
        sharp_score = calculate_sharpness(sharp_image)
        blurry_score = calculate_sharpness(blurry_image)
        assert sharp_score > blurry_score

    def test_none_image_returns_zero(self):
        """None input should safely return 0.0."""
        assert calculate_sharpness(None) == 0.0

    def test_empty_array_returns_zero(self):
        """Empty numpy array should safely return 0.0."""
        assert calculate_sharpness(np.array([])) == 0.0

    def test_returns_float(self, sharp_image):
        """Return value must be a float."""
        score = calculate_sharpness(sharp_image)
        assert isinstance(score, float)

    def test_deterministic(self, sharp_image):
        """Same image should always produce the same score."""
        s1 = calculate_sharpness(sharp_image)
        s2 = calculate_sharpness(sharp_image)
        assert s1 == s2


# -----------------------------------------------------------------------
# get_threshold_by_resolution
# -----------------------------------------------------------------------

class TestGetThresholdByResolution:
    """Tests for the get_threshold_by_resolution() function."""

    def test_1080p_exact(self):
        """1080p (reference resolution) should return exactly 300."""
        threshold = get_threshold_by_resolution(1920, 1080)
        assert threshold == 300.0

    def test_4k_higher_than_1080p(self):
        """4K threshold should be significantly higher than 1080p."""
        threshold = get_threshold_by_resolution(3840, 2160)
        assert threshold > 400, f"Expected 4K threshold > 400, got {threshold:.1f}"

    def test_4k_expected_value(self):
        """4K threshold should be approximately 1200 (4× 1080p pixels)."""
        threshold = get_threshold_by_resolution(3840, 2160)
        # 3840*2160 / (1920*1080) = 4.0 → 300 * 4 = 1200
        assert abs(threshold - 1200.0) < 1.0

    def test_720p_lower_than_1080p(self):
        """720p threshold should be lower than 1080p."""
        threshold = get_threshold_by_resolution(1280, 720)
        assert threshold < 200, f"Expected 720p threshold < 200, got {threshold:.1f}"

    def test_720p_expected_value(self):
        """720p threshold: 1280*720 / (1920*1080) = 0.444 → 300*0.444 ≈ 133."""
        threshold = get_threshold_by_resolution(1280, 720)
        expected = 300.0 * (1280 * 720) / (1920 * 1080)
        assert abs(threshold - expected) < 1.0

    def test_minimum_floor_100(self):
        """Very low resolution should be clamped to a minimum of 100."""
        threshold = get_threshold_by_resolution(100, 50)
        assert threshold >= 100.0

    def test_low_resolution_floor(self):
        """240×180 (43200 px) → ratio 0.0194 → 300*0.0194 ≈ 5.8 → clamped to 100."""
        threshold = get_threshold_by_resolution(240, 180)
        assert threshold == 100.0

    def test_higher_resolution_higher_threshold(self):
        """Monotonically increasing: more pixels → higher threshold."""
        t_480p = get_threshold_by_resolution(854, 480)
        t_720p = get_threshold_by_resolution(1280, 720)
        t_1080p = get_threshold_by_resolution(1920, 1080)
        t_4k = get_threshold_by_resolution(3840, 2160)
        assert t_480p < t_720p < t_1080p < t_4k

    def test_returns_float(self):
        """Return value must be a float."""
        result = get_threshold_by_resolution(1920, 1080)
        assert isinstance(result, float)


# -----------------------------------------------------------------------
# VideoQualityAnalyzer
# -----------------------------------------------------------------------

class TestVideoQualityAnalyzer:
    """Tests for the VideoQualityAnalyzer.analyze() method."""

    def test_analyze_returns_video_quality_info(self, sample_video):
        """analyze() should return a VideoQualityInfo instance."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert isinstance(info, VideoQualityInfo)

    def test_analyze_resolution(self, sample_video):
        """Resolution should match the synthetic video (320×240)."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert info.resolution == (320, 240)

    def test_analyze_fps(self, sample_video):
        """FPS should match the synthetic video (25.0)."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert abs(info.fps - 25.0) < 0.5

    def test_analyze_avg_sharpness_positive(self, sample_video):
        """Average sharpness should be positive for a non-blank video."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert info.avg_sharpness > 0

    def test_analyze_threshold_set(self, sample_video):
        """Threshold should be computed and positive."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert info.threshold > 0
        # For 320×240 → very low res → clamped to 100
        assert info.threshold == 100.0

    def test_analyze_is_low_quality_bool(self, sample_video):
        """is_low_quality should be a boolean."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert isinstance(info.is_low_quality, bool)

    def test_analyze_invalid_video_raises(self):
        """analyze() should raise ValueError for a non-existent video."""
        analyzer = VideoQualityAnalyzer()
        with pytest.raises(ValueError, match="Cannot open video"):
            analyzer.analyze("/nonexistent/path/video.mp4")

    def test_analyze_bitrate_positive(self, sample_video):
        """Bitrate should be positive for a valid video file."""
        analyzer = VideoQualityAnalyzer()
        info = analyzer.analyze(sample_video)
        assert info.bitrate > 0

    def test_analyze_sample_frames_clamped(self, tmp_path):
        """Requesting more samples than total frames should not crash."""
        # Create a video with only 3 frames
        video_path = str(tmp_path / "short.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (160, 120))
        for _ in range(3):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        analyzer = VideoQualityAnalyzer()
        # Request 10 samples from a 3-frame video — should not crash
        info = analyzer.analyze(video_path, sample_frames=10)
        assert info.avg_sharpness >= 0
