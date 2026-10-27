"""Tests for Stage 3: Face Detection (two-stage pipeline).

Tests quality evaluation functions, database schema, and the
detection pipeline with synthetic test data.
"""

import cv2
import numpy as np
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/auraface/scrfd_10g_bnkps.onnx"),
    reason="需要 AuraFace 人脸模型(应用内「设置 → 环境初始化」下载)")


from src.core.face_detector import (
    evaluate_clarity,
    evaluate_pose,
    evaluate_brightness,
    calculate_overall_quality,
    FaceDetector,
    FaceSegment,
    DenseFrame,
    FaceQuality,
)


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sharp_face_image() -> np.ndarray:
    """Create a synthetic sharp face-like image (oval + features).

    High-frequency edges from facial features produce high clarity score.
    """
    img = np.full((200, 160, 3), 200, dtype=np.uint8)
    # Face oval
    cv2.ellipse(img, (80, 100), (60, 75), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (55, 80), 10, (80, 80, 80), -1)
    cv2.circle(img, (105, 80), 10, (80, 80, 80), -1)
    # Nose
    cv2.line(img, (80, 90), (80, 120), (100, 80, 70), 2)
    # Mouth
    cv2.ellipse(img, (80, 140), (20, 8), 0, 0, 180, (120, 60, 60), 2)
    return img


@pytest.fixture
def blurry_face_image(sharp_face_image: np.ndarray) -> np.ndarray:
    """Blurry version of the face image."""
    return cv2.GaussianBlur(sharp_face_image, (31, 31), 10)


@pytest.fixture
def dark_face_image(sharp_face_image: np.ndarray) -> np.ndarray:
    """Dark version of the face image."""
    return (sharp_face_image * 0.3).astype(np.uint8)


@pytest.fixture
def bright_face_image(sharp_face_image: np.ndarray) -> np.ndarray:
    """Overexposed face image."""
    return np.clip(sharp_face_image.astype(int) + 150, 0, 255).astype(np.uint8)


@pytest.fixture
def test_db(tmp_path) -> str:
    """Create a temporary database with all tables."""
    db_path = str(tmp_path / "test.db")
    from src.db.schema import init_db
    init_db(db_path)
    return db_path


@pytest.fixture
def synthetic_face_video(tmp_path) -> str:
    """Create a 10-second synthetic video with a face-like pattern from 6.0-6.5s.

    Frames 0-149 (0-5s): plain background
    Frames 150-164 (5.0-5.5s): face appears (for testing)
    Frames 165-299 (5.5-10s): plain background

    FPS=30, so face appears at ~5.0s and lasts ~0.5s.
    We put it at 5.0s to be safe for testing.
    """
    video_path = str(tmp_path / "test_face_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 320, 240
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    face_start_frame = 150  # 5.0s at 30fps
    face_end_frame = 165    # 5.5s at 30fps

    for i in range(300):  # 10 seconds at 30fps
        frame = np.ones((height, width, 3), dtype=np.uint8) * 180

        if face_start_frame <= i <= face_end_frame:
            # Draw a simple face-like pattern in the center
            cx, cy = 160, 120
            # Face oval
            cv2.ellipse(frame, (cx, cy), (50, 65), 0, 0, 360, (200, 180, 160), -1)
            # Eyes
            cv2.circle(frame, (cx - 20, cy - 15), 8, (60, 60, 60), -1)
            cv2.circle(frame, (cx + 20, cy - 15), 8, (60, 60, 60), -1)
            # Nose
            cv2.line(frame, (cx, cy - 5), (cx, cy + 15), (100, 80, 70), 2)
            # Mouth
            cv2.ellipse(frame, (cx, cy + 30), (15, 6), 0, 0, 180, (120, 60, 60), 2)

        writer.write(frame)

    writer.release()
    return video_path


# ───────────────────────────────────────────────────────────────────
# Quality evaluation: clarity
# ───────────────────────────────────────────────────────────────────

class TestEvaluateClarity:
    """Tests for the clarity evaluation function."""

    def test_sharp_face_high_clarity(self, sharp_face_image: np.ndarray):
        """A sharp face image should have high clarity."""
        score = evaluate_clarity(sharp_face_image)
        assert 0.0 <= score <= 1.0
        assert score > 0.3, f"Expected high clarity, got {score:.3f}"

    def test_blurry_face_low_clarity(self, blurry_face_image: np.ndarray):
        """A blurry face image should have low clarity."""
        score = evaluate_clarity(blurry_face_image)
        assert 0.0 <= score <= 1.0
        assert score < 0.3, f"Expected low clarity, got {score:.3f}"

    def test_sharp_higher_than_blurry(self, sharp_face_image, blurry_face_image):
        """Sharp image must score higher than blurry image."""
        sharp_score = evaluate_clarity(sharp_face_image)
        blurry_score = evaluate_clarity(blurry_face_image)
        assert sharp_score > blurry_score

    def test_none_image_returns_zero(self):
        """None input should safely return 0."""
        assert evaluate_clarity(None) == 0.0

    def test_empty_image_returns_zero(self):
        """Empty array should safely return 0."""
        assert evaluate_clarity(np.array([])) == 0.0

    def test_returns_float(self, sharp_face_image):
        """Return value should be a float."""
        assert isinstance(evaluate_clarity(sharp_face_image), float)


# ───────────────────────────────────────────────────────────────────
# Quality evaluation: pose
# ───────────────────────────────────────────────────────────────────

class TestEvaluatePose:
    """Tests for the pose quality evaluation function."""

    def test_frontal_face_perfect(self):
        """A perfectly frontal face should have pose score 1.0."""
        score = evaluate_pose(0.0, 0.0, 0.0)
        assert score == 1.0

    def test_high_yaw_poor(self):
        """A face turned 80 degrees to the side should have low pose score."""
        score = evaluate_pose(80.0, 0.0, 0.0)
        assert score < 0.3

    def test_high_pitch_poor(self):
        """A face tilted far up/down should have low pose score."""
        score = evaluate_pose(0.0, 50.0, 0.0)
        assert score < 0.3

    def test_high_roll_poor(self):
        """A heavily rotated face should have low pose score."""
        score = evaluate_pose(0.0, 0.0, 40.0)
        assert score < 0.3

    def test_score_range(self):
        """All pose scores should be in [0, 1]."""
        for yaw in [-90, -45, 0, 45, 90]:
            for pitch in [-60, -30, 0, 30, 60]:
                for roll in [-45, -20, 0, 20, 45]:
                    score = evaluate_pose(yaw, pitch, roll)
                    assert 0.0 <= score <= 1.0

    def test_moderate_pose_acceptable(self):
        """A slightly tilted face should still have decent score."""
        score = evaluate_pose(15.0, 10.0, 5.0)
        assert score > 0.5

    def test_returns_float(self):
        """Return value should be a float."""
        assert isinstance(evaluate_pose(0, 0, 0), float)


# ───────────────────────────────────────────────────────────────────
# Quality evaluation: brightness
# ───────────────────────────────────────────────────────────────────

class TestEvaluateBrightness:
    """Tests for the brightness evaluation function."""

    def test_normal_brightness_high_score(self, sharp_face_image: np.ndarray):
        """A normally lit face should have high brightness score."""
        score = evaluate_brightness(sharp_face_image)
        assert 0.0 <= score <= 1.0
        assert score > 0.7, f"Expected high brightness score, got {score:.3f}"

    def test_dark_face_low_score(self, dark_face_image: np.ndarray):
        """A dark face image should have low brightness score."""
        score = evaluate_brightness(dark_face_image)
        assert score < 0.7, f"Expected low brightness, got {score:.3f}"

    def test_bright_face_high_score(self, sharp_face_image: np.ndarray):
        """A well-lit face should have high score."""
        # Adjust to be in the good range
        img = np.clip(sharp_face_image.astype(int) + 30, 100, 200).astype(np.uint8)
        score = evaluate_brightness(img)
        assert score == 1.0

    def test_overexposed_lower_score(self, bright_face_image: np.ndarray):
        """An overexposed face should have reduced score."""
        score = evaluate_brightness(bright_face_image)
        assert score < 0.8

    def test_none_image_returns_zero(self):
        """None input should safely return 0."""
        assert evaluate_brightness(None) == 0.0

    def test_empty_image_returns_zero(self):
        """Empty array should safely return 0."""
        assert evaluate_brightness(np.array([])) == 0.0

    def test_returns_float(self, sharp_face_image):
        """Return value should be a float."""
        assert isinstance(evaluate_brightness(sharp_face_image), float)


# ───────────────────────────────────────────────────────────────────
# Quality evaluation: overall
# ───────────────────────────────────────────────────────────────────

class TestCalculateOverallQuality:
    """Tests for overall quality calculation."""

    def test_perfect_face_perfect_score(self):
        """All quality metrics perfect → overall = 1.0."""
        score = calculate_overall_quality(1.0, 1.0, 1.0)
        assert score == 1.0

    def test_zero_face_zero_score(self):
        """All quality metrics zero → overall = 0.0."""
        score = calculate_overall_quality(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_weights_sum_correctly(self):
        """Weighted sum should be computed correctly."""
        # clarity:0.5 * 0.4 = 0.2
        # pose:0.5 * 0.4 = 0.2
        # brightness:0.5 * 0.2 = 0.1
        # total = 0.5
        score = calculate_overall_quality(0.5, 0.5, 0.5)
        assert abs(score - 0.5) < 0.001

    def test_custom_weights(self):
        """Custom weights should be respected."""
        score = calculate_overall_quality(
            1.0, 0.0, 0.0,
            clarity_weight=1.0, pose_weight=0.0, brightness_weight=0.0,
        )
        assert score == 1.0

    def test_score_range(self):
        """Overall quality should always be in [0, 1]."""
        score = calculate_overall_quality(0.6, 0.7, 0.8)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        """Return value should be a float."""
        assert isinstance(calculate_overall_quality(0.5, 0.5, 0.5), float)


# ───────────────────────────────────────────────────────────────────
# Database CRUD for face tables
# ───────────────────────────────────────────────────────────────────

class TestFaceDatabaseCRUD:
    """Tests for face-related database operations."""

    def _insert_task(self, db, name="test.mp4"):
        """Helper: insert a test task."""
        from src.db import repository as repo
        return repo.upsert_task(db, f"/tmp/{name}", name, 1000, "md5_" + name)

    def test_add_and_get_face_segment(self, test_db: str):
        """Add a face segment and retrieve it."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)

        seg = repo.add_face_segment(
            db, task["id"], unit_index=0,
            start_time=5.0, end_time=5.5,
            face_count_prescan=1,
            prescan_status="completed",
        )

        assert seg["id"] > 0
        assert seg["task_id"] == task["id"]
        assert seg["start_time"] == 5.0
        assert seg["end_time"] == 5.5
        assert seg["face_count_prescan"] == 1
        assert seg["prescan_status"] == "completed"

        segments = repo.get_face_segments_for_task(db, task["id"])
        assert len(segments) == 1
        assert segments[0]["id"] == seg["id"]

        db.close()

    def test_update_face_segment_status(self, test_db: str):
        """Update segment status fields."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 0.0, 1.0)

        repo.update_face_segment_status(db, seg["id"], prescan_status="completed")
        segs = repo.get_face_segments_for_task(db, task["id"])
        assert segs[0]["prescan_status"] == "completed"
        assert segs[0]["dense_status"] == "pending"

        repo.update_face_segment_status(db, seg["id"], dense_status="completed")
        segs = repo.get_face_segments_for_task(db, task["id"])
        assert segs[0]["prescan_status"] == "completed"
        assert segs[0]["dense_status"] == "completed"

        db.close()

    def test_face_dense_frames_crud(self, test_db: str):
        """CRUD operations for face_dense_frames."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5)

        # Add dense frames
        frame1 = repo.add_face_dense_frame(
            db, task["id"], seg["id"], 0, 5.0, "/tmp/f1.jpg", "completed",
        )
        frame2 = repo.add_face_dense_frame(
            db, task["id"], seg["id"], 1, 5.2, "/tmp/f2.jpg", "completed",
        )

        assert frame1["id"] > 0
        assert frame2["frame_index"] == 1

        # Get by segment
        frames = repo.get_dense_frames_for_segment(db, seg["id"])
        assert len(frames) == 2
        assert frames[0]["timestamp"] == 5.0

        # Get by task
        task_frames = repo.get_dense_frames_for_task(db, task["id"])
        assert len(task_frames) == 2

        # Update status
        repo.update_dense_frame_status(db, frame1["id"], "failed", "test error")
        frames = repo.get_dense_frames_for_segment(db, seg["id"])
        assert frames[0]["detection_status"] == "failed"
        assert frames[0]["error_message"] == "test error"

        db.close()

    def test_face_detections_crud(self, test_db: str):
        """CRUD operations for face_detections."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5)
        frame = repo.add_face_dense_frame(db, task["id"], seg["id"], 0, 5.0)

        bbox = [100, 80, 200, 180]
        landmarks = {
            "left_eye": [130, 110],
            "right_eye": [170, 110],
            "nose": [150, 140],
            "left_mouth": [135, 160],
            "right_mouth": [165, 160],
        }

        det = repo.add_face_detection(
            db, frame["id"], task["id"], 0, 0.95,
            bbox, landmarks,
            quality_clarity=0.8, quality_pose=0.9,
            quality_brightness=0.7, quality_overall=0.82,
        )

        assert det["id"] > 0
        assert det["confidence"] == 0.95
        assert det["quality_overall"] == 0.82

        detections = repo.get_detections_for_frame(db, frame["id"])
        assert len(detections) == 1

        task_detections = repo.get_detections_for_task(db, task["id"])
        assert len(task_detections) == 1
        assert repo.count_faces_for_task(db, task["id"]) == 1

        db.close()

    def test_face_embeddings_crud(self, test_db: str):
        """CRUD operations for face_embeddings."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5)
        frame = repo.add_face_dense_frame(db, task["id"], seg["id"], 0, 5.0)
        bbox = [100, 80, 200, 180]
        det = repo.add_face_detection(
            db, frame["id"], task["id"], 0, 0.95, bbox, {},
            quality_overall=0.8,
        )

        # Save embedding as blob
        embedding = np.random.randn(512).astype(np.float32)
        emb = repo.add_face_embedding(
            db, det["id"], task["id"],
            embedding.tobytes(), embedding_dim=512,
            quality=0.8, face_image_path="/tmp/face.jpg",
        )

        assert emb["id"] > 0
        assert emb["embedding_dim"] == 512
        assert emb["quality"] == 0.8

        # Get embeddings for task
        embs = repo.get_embeddings_for_task(db, task["id"])
        assert len(embs) == 1

        # Verify blob is correct
        stored_emb = np.frombuffer(embs[0]["embedding"], dtype=np.float32)
        assert len(stored_emb) == 512
        np.testing.assert_array_almost_equal(stored_emb, embedding)

        db.close()

    def test_unprocessed_faces_query(self, test_db: str):
        """Query for faces without embeddings."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5)
        frame = repo.add_face_dense_frame(db, task["id"], seg["id"], 0, 5.0)
        bbox = [100, 80, 200, 180]

        det1 = repo.add_face_detection(db, frame["id"], task["id"], 0, 0.9, bbox, {})
        det2 = repo.add_face_detection(db, frame["id"], task["id"], 1, 0.8, bbox, {})

        # Initially both should be unprocessed
        unprocessed = repo.get_unprocessed_face_ids(db, task["id"])
        assert len(unprocessed) == 2
        assert det1["id"] in unprocessed
        assert det2["id"] in unprocessed

        # Add embedding for det1
        repo.add_face_embedding(db, det1["id"], task["id"], b"\x00" * 4, embedding_dim=1)

        unprocessed = repo.get_unprocessed_face_ids(db, task["id"])
        assert len(unprocessed) == 1
        assert det2["id"] in unprocessed

        db.close()

    def test_delete_face_data(self, test_db: str):
        """Delete all face data for a task."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task = self._insert_task(db)
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5)
        frame = repo.add_face_dense_frame(db, task["id"], seg["id"], 0, 5.0)
        bbox = [100, 80, 200, 180]
        det = repo.add_face_detection(db, frame["id"], task["id"], 0, 0.9, bbox, {})
        repo.add_face_embedding(db, det["id"], task["id"], b"\x00", 1)

        assert repo.count_faces_for_task(db, task["id"]) == 1

        result = repo.delete_face_data_for_task(db, task["id"])
        assert result == 1

        assert repo.count_faces_for_task(db, task["id"]) == 0
        assert len(repo.get_face_segments_for_task(db, task["id"])) == 0
        assert len(repo.get_dense_frames_for_task(db, task["id"])) == 0
        assert len(repo.get_embeddings_for_task(db, task["id"])) == 0

        db.close()


# ───────────────────────────────────────────────────────────────────
# FaceDetector: basic instantiation
# ───────────────────────────────────────────────────────────────────

class TestFaceDetector:
    """Tests for the FaceDetector class (instantiation and config)."""

    def test_detector_default_config(self):
        """FaceDetector should initialize with default config."""
        detector = FaceDetector()
        assert detector.prescan_fps == 2
        assert detector.dense_fps == 5
        # segment_margin_seconds 于 round11 调优为 15.0（见 config.yaml）
        assert detector.segment_margin == 15.0
        assert detector.confidence_threshold == 0.5

    def test_detector_custom_config(self):
        """FaceDetector should respect custom config."""
        config = {
            "face_detection": {
                "prescan_fps": 3,
                "dense_fps": 8,
                "segment_margin_seconds": 2.0,
                "confidence_threshold": 0.3,
                "quality_evaluation": {"enabled": True, "clarity_weight": 0.5},
                "quality_filter": {"enabled": False},
            }
        }
        detector = FaceDetector(config=config)
        assert detector.prescan_fps == 3
        assert detector.dense_fps == 8
        assert detector.segment_margin == 2.0
        assert detector.confidence_threshold == 0.3
        assert detector.clarity_weight == 0.5
        assert detector.quality_filter_enabled is False

    def test_face_segment_dataclass(self):
        """FaceSegment dataclass should work correctly."""
        seg = FaceSegment(start_time=1.0, end_time=2.0, face_count=3)
        assert seg.start_time == 1.0
        assert seg.end_time == 2.0
        assert seg.face_count == 3

    def test_dense_frame_dataclass(self):
        """DenseFrame dataclass should work correctly."""
        df = DenseFrame(frame_index=0, timestamp=5.0, frame_path="/tmp/f.jpg")
        assert df.frame_index == 0
        assert df.timestamp == 5.0
        assert len(df.faces) == 0

    def test_face_quality_dataclass(self):
        """FaceQuality dataclass should have correct defaults."""
        fq = FaceQuality()
        assert fq.clarity == 0.0
        assert fq.pose == 0.0
        assert fq.brightness == 0.0
        assert fq.overall == 0.0

        fq2 = FaceQuality(clarity=0.8, pose=0.9, overall=0.85)
        assert fq2.clarity == 0.8
        assert fq2.overall == 0.85

    def test_synthetic_video_creation(self, synthetic_face_video: str):
        """Verify test video can be created and read."""
        assert os.path.exists(synthetic_face_video)

        cap = cv2.VideoCapture(synthetic_face_video)
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert abs(fps - 30.0) < 1.0
        assert total_frames > 0

        # Check duration is ~10 seconds
        duration = total_frames / fps
        assert abs(duration - 10.0) < 1.0

        cap.release()

    def test_detector_smart_sampling_default_on(self):
        """Smart sampling should be enabled by default."""
        detector = FaceDetector()
        assert detector.smart_sampling is True

    def test_detector_smart_sampling_custom_config(self):
        """Smart sampling config should be respected."""
        config = {
            "face_detection": {
                "smart_sampling": False,
                "smart_frames_per_segment": 10,
                "smart_retry_window_seconds": 5,
                "smart_max_retries": 2,
            }
        }
        detector = FaceDetector(config=config)
        assert detector.smart_sampling is False
        assert detector.smart_frames_per_segment == 10
        assert detector.smart_retry_window_seconds == 5
        assert detector.smart_max_retries == 2


# ───────────────────────────────────────────────────────────────────
# Smart sampling (face-quality-filter scheme)
# ───────────────────────────────────────────────────────────────────

class TestSmartSampling:
    """Tests for the smart sampling path in _dense_sample_and_detect."""

    def test_smart_sampling_saves_sharp_frames(self, synthetic_face_video, tmp_path):
        """Smart sampling should save frames to the per-segment subdir."""
        detector = FaceDetector(config={
            "face_detection": {
                "smart_sampling": True,
                "smart_frames_per_segment": 10,
                "smart_max_retries": 3,
            }
        })
        out = str(tmp_path / "frames")
        # Segment covering the whole video (0-10s)
        dense = detector._dense_sample_and_detect(
            synthetic_face_video, 0.0, 10.0,
            task_id=1, segment_id=0, output_dir=out,
        )
        # Even with no real faces, frames should be produced and saved.
        assert len(dense) > 0
        seg_dir = os.path.join(out, "seg0")
        assert os.path.isdir(seg_dir)
        saved = os.listdir(seg_dir)
        assert len(saved) > 0

    def test_smart_sampling_disabled_uses_flat_dir(self, synthetic_face_video, tmp_path):
        """With smart sampling off, frames go directly into output_dir."""
        detector = FaceDetector(config={
            "face_detection": {
                "smart_sampling": False,
                "dense_fps": 5,
            }
        })
        out = str(tmp_path / "frames")
        dense = detector._dense_sample_and_detect(
            synthetic_face_video, 0.0, 10.0,
            task_id=1, segment_id=0, output_dir=out,
        )
        assert len(dense) > 0
        # No seg0 subdir in legacy mode — frames are flat in output_dir.
        assert not os.path.isdir(os.path.join(out, "seg0"))

    def test_smart_sampling_frame_count_limited(self, tmp_path):
        """Frames per segment should be capped by smart_frames_per_segment.

        Uses a noise video (high-frequency content) so smart sampling
        succeeds instead of falling back to the legacy path.
        """
        # 10s noise video @ 30fps
        video_path = str(tmp_path / "noise.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (320, 240))
        for _ in range(300):
            writer.write(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        writer.release()

        detector = FaceDetector(config={
            "face_detection": {
                "smart_sampling": True,
                "smart_frames_per_segment": 6,
                "dense_fps": 5,  # 10s segment → 50 frames at 5fps
            }
        })
        dense = detector._dense_sample_and_detect(
            video_path, 0.0, 10.0,
            task_id=1, segment_id=0, output_dir=str(tmp_path / "f"),
        )
        # target = min(6, 50) = 6; smart sampling succeeds on noise frames
        assert 0 < len(dense) <= 6


# ───────────────────────────────────────────────────────────────────
# BaseStageProcessor base class contract
# ───────────────────────────────────────────────────────────────────

class TestBaseStageProcessor:
    """Tests for the BaseStageProcessor abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """BaseStageProcessor cannot be instantiated directly."""
        from src.core.base_stage import BaseStageProcessor
        with pytest.raises(TypeError):
            BaseStageProcessor("test")

    def test_concrete_subclass_works(self):
        """A concrete subclass should work with process_all."""
        from src.core.base_stage import BaseStageProcessor

        class TestProcessor(BaseStageProcessor):
            def __init__(self):
                super().__init__("test")
                self._status = {1: "pending", 2: "pending", 3: "completed"}

            def get_processing_units(self, task_id):
                return [1, 2, 3]

            def get_unit_status(self, unit_id):
                return self._status.get(unit_id, "pending")

            def update_unit_status(self, unit_id, status, error_msg=None):
                self._status[unit_id] = status

            def process_unit(self, unit_id):
                return True  # all succeed

        proc = TestProcessor()
        result = proc.process_all(task_id=1)

        assert result["total"] == 3
        assert result["completed"] == 2  # 2 pending → 2 completed
        assert result["skipped"] == 1    # 1 already completed
        assert result["failed"] == 0

    def test_failed_units_tracked(self):
        """Failed units should be counted properly."""
        from src.core.base_stage import BaseStageProcessor

        class FailingProcessor(BaseStageProcessor):
            def __init__(self):
                super().__init__("test")
                self._status = {1: "pending", 2: "pending"}

            def get_processing_units(self, task_id):
                return [1, 2]

            def get_unit_status(self, unit_id):
                return self._status.get(unit_id, "pending")

            def update_unit_status(self, unit_id, status, error_msg=None):
                self._status[unit_id] = status

            def process_unit(self, unit_id):
                if unit_id == 1:
                    return True
                raise RuntimeError("test failure")

        proc = FailingProcessor()
        result = proc.process_all(task_id=1)

        assert result["completed"] == 1
        assert result["failed"] == 1
        assert proc._status[2] == "failed"

    def test_progress_callback(self):
        """Progress callback should be called with correct values."""
        from src.core.base_stage import BaseStageProcessor

        calls = []

        class SimpleProcessor(BaseStageProcessor):
            def __init__(self):
                super().__init__("test")
                self._status = {i: "pending" for i in range(5)}

            def get_processing_units(self, task_id):
                return list(range(5))

            def get_unit_status(self, unit_id):
                return self._status[unit_id]

            def update_unit_status(self, unit_id, status, error_msg=None):
                self._status[unit_id] = status

            def process_unit(self, unit_id):
                return True

        def callback(current, total, msg):
            calls.append((current, total))

        proc = SimpleProcessor()
        proc.process_all(task_id=1, progress_callback=callback)

        assert len(calls) == 5
        assert calls[0] == (1, 5)
        assert calls[-1] == (5, 5)


# ───────────────────────────────────────────────────────────────────
# ModelManager base class
# ───────────────────────────────────────────────────────────────────

class TestModelManager:
    """Tests for the ModelManager."""

    def test_register_and_get_model(self):
        """Register a model factory and get the model."""
        from src.core.model_manager import ModelManager, BaseModel

        class FakeModel(BaseModel):
            def __init__(self, config):
                self.config = config
                self._loaded = False

            def is_available(self):
                return self._loaded

            def load(self):
                self._loaded = True
                return True

            def unload(self):
                self._loaded = False

        manager = ModelManager()
        manager.register_factory("test_model", FakeModel)

        model = manager.get("test_model")
        assert model is not None
        assert model.is_available()

        # Second call should return same instance
        model2 = manager.get("test_model")
        assert model is model2

    def test_unload_all(self):
        """Unload all models should release them."""
        from src.core.model_manager import ModelManager, BaseModel

        class FakeModel(BaseModel):
            def __init__(self, config):
                self._loaded = False

            def is_available(self):
                return self._loaded

            def load(self):
                self._loaded = True
                return True

            def unload(self):
                self._loaded = False

        manager = ModelManager()
        manager.register_factory("fake", FakeModel)
        manager.get("fake")
        assert manager.is_available("fake")

        manager.unload_all()
        assert not manager.is_available("fake")

    def test_unknown_model_returns_none(self):
        """Getting an unregistered model should return None."""
        from src.core.model_manager import ModelManager
        manager = ModelManager()
        assert manager.get("nonexistent") is None
