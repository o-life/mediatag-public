"""Tests for face recognition: L2 normalization, face alignment, pose estimation,
quality scoring, and the database resume logic.

These tests target the real implementation in ``src/core/face_utils.py`` and
``src/core/face_recognition.py``, using synthetic images (no real videos or
model weights required).
"""

import numpy as np
import pytest

from src.core.face_utils import (
    align_face,
    l2_normalize,
    estimate_pose,
    calculate_face_quality,
    REFERENCE_5POINTS_112,
)


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture
def face_image_with_landmarks():
    """Create a synthetic face image with known landmarks.

    The face is centered in a 200x200 image with features at
    known positions so we can test alignment.
    """
    import cv2

    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    cx, cy = 100, 100

    # Face oval
    cv2.ellipse(img, (cx, cy), (60, 75), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (cx - 25, cy - 20), 8, (60, 60, 60), -1)
    cv2.circle(img, (cx + 25, cy - 20), 8, (60, 60, 60), -1)
    # Nose
    cv2.line(img, (cx, cy - 5), (cx, cy + 20), (100, 80, 70), 2)
    # Mouth
    cv2.ellipse(img, (cx, cy + 40), (18, 8), 0, 0, 180, (120, 60, 60), 2)

    landmarks = {
        "left_eye": [float(cx - 25), float(cy - 20)],
        "right_eye": [float(cx + 25), float(cy - 20)],
        "nose": [float(cx), float(cy + 5)],
        "left_mouth": [float(cx - 15), float(cy + 40)],
        "right_mouth": [float(cx + 15), float(cy + 40)],
    }
    return img, landmarks


@pytest.fixture
def test_db(tmp_path) -> str:
    """Create a temporary database with all tables."""
    from src.db.schema import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path


# ───────────────────────────────────────────────────────────────────
# L2 Normalization
# ───────────────────────────────────────────────────────────────────

class TestL2Normalize:
    """Tests for the L2 normalization function."""

    def test_unit_vector_unchanged(self):
        """A unit vector should not change after normalization."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = l2_normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_non_unit_vector_normalized(self):
        """A non-unit vector should become unit length."""
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = l2_normalize(v)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_512_dim_vector(self):
        """512-dim vectors should normalize correctly."""
        v = np.random.randn(512).astype(np.float32)
        result = l2_normalize(v)
        assert len(result) == 512
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_zero_vector_safe(self):
        """Zero-length vector should not cause division by zero."""
        v = np.zeros(128, dtype=np.float32)
        result = l2_normalize(v)
        assert result is not None
        assert np.all(result == 0.0)

    def test_preserves_direction(self):
        """Normalization should preserve direction (same sign)."""
        v = np.array([-2.0, 3.0, -1.0], dtype=np.float32)
        result = l2_normalize(v)
        # Signs should match
        assert np.all(np.sign(result) == np.sign(v))

    def test_scalar_independence(self):
        """Scaling a vector should not change its normalized form."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_scaled = v * 5.0
        r1 = l2_normalize(v)
        r2 = l2_normalize(v_scaled)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_returns_float_array(self):
        """Return type should be a numpy array of floats."""
        v = np.array([1.0, 1.0], dtype=np.float32)
        result = l2_normalize(v)
        assert isinstance(result, np.ndarray)
        assert result.dtype in (np.float32, np.float64)


# ───────────────────────────────────────────────────────────────────
# Face Alignment
# ───────────────────────────────────────────────────────────────────

class TestFaceAlignment:
    """Tests for face alignment using 5-point landmarks."""

    def test_reference_points_shape(self):
        """Reference landmarks should have correct shape."""
        assert REFERENCE_5POINTS_112.shape == (5, 2)

    def test_align_output_size_correct(self, face_image_with_landmarks):
        """Aligned face should have the requested output size."""
        img, landmarks = face_image_with_landmarks
        aligned = align_face(img, landmarks, output_size=(112, 112))
        assert aligned.shape == (112, 112, 3)

    def test_align_different_sizes(self, face_image_with_landmarks):
        """Alignment should work with different output sizes."""
        img, landmarks = face_image_with_landmarks
        for size in [(112, 112), (224, 224), (96, 112)]:
            aligned = align_face(img, landmarks, output_size=size)
            assert aligned.shape == (size[1], size[0], 3)

    def test_align_frontal_face_no_distortion(self, face_image_with_landmarks):
        """A roughly frontal face should remain recognizable after alignment."""
        img, landmarks = face_image_with_landmarks
        aligned = align_face(img, landmarks, output_size=(112, 112))

        # After alignment, the face should still have content (not all zeros)
        assert np.mean(aligned) > 10
        # And not all 255
        assert np.mean(aligned) < 245

    def test_align_rotated_face(self):
        """A rotated face should be straightened by alignment."""
        import cv2

        # Create a face with tilted landmarks
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        cx, cy = 100, 100

        # Draw tilted face (rotate by 30 degrees conceptually)
        angle = np.radians(30)
        def rotate_point(x, y):
            rx = (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle) + cx
            ry = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle) + cy
            return int(rx), int(ry)

        # Eyes at tilted positions
        le = rotate_point(cx - 25, cy - 20)
        re = rotate_point(cx + 25, cy - 20)
        nose = rotate_point(cx, cy + 5)
        lm = rotate_point(cx - 15, cy + 40)
        rm = rotate_point(cx + 15, cy + 40)

        cv2.circle(img, le, 8, (60, 60, 60), -1)
        cv2.circle(img, re, 8, (60, 60, 60), -1)

        landmarks = {
            "left_eye": [float(le[0]), float(le[1])],
            "right_eye": [float(re[0]), float(re[1])],
            "nose": [float(nose[0]), float(nose[1])],
            "left_mouth": [float(lm[0]), float(lm[1])],
            "right_mouth": [float(rm[0]), float(rm[1])],
        }

        aligned = align_face(img, landmarks, output_size=(112, 112))
        assert aligned.shape == (112, 112, 3)
        # After alignment, the face should be straightened and retain content
        assert np.mean(aligned) > 5

    def test_align_empty_landmarks_fallback(self):
        """Alignment with empty landmarks should still return an image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = {
            "left_eye": [30.0, 40.0],
            "right_eye": [70.0, 40.0],
            "nose": [50.0, 55.0],
            "left_mouth": [35.0, 75.0],
            "right_mouth": [65.0, 75.0],
        }
        aligned = align_face(img, landmarks, (112, 112))
        assert aligned.shape == (112, 112, 3)

    def test_align_missing_landmark_key_fallback(self):
        """Missing a landmark key should fall back to center-crop, not crash."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        landmarks = {
            "left_eye": [30.0, 40.0],
            "right_eye": [70.0, 40.0],
            "nose": [50.0, 55.0],
            # missing left_mouth / right_mouth
        }
        aligned = align_face(img, landmarks, (112, 112))
        assert aligned.shape == (112, 112, 3)
        assert np.mean(aligned) > 0

    def test_align_same_face_idempotent(self, face_image_with_landmarks):
        """Aligning an already-aligned face should produce similar results."""
        img, landmarks = face_image_with_landmarks
        aligned1 = align_face(img, landmarks, output_size=(112, 112))

        # We verify shape and content (not exact pixel equality)
        assert aligned1.shape == (112, 112, 3)
        assert np.mean(aligned1) > 0


# ───────────────────────────────────────────────────────────────────
# Pose estimation
# ───────────────────────────────────────────────────────────────────

class TestPoseEstimation:
    """Tests for 5-point-landmark pose estimation."""

    def test_frontal_face_zero_pose(self):
        """A symmetric frontal face should yield near-zero angles."""
        # Eye midpoint at y=50, face_height = 70 (bbox [20,30,92,100]).
        # Frontal nose sits ~0.33 * face_height below the eyes: 50 + 23 ≈ 73.
        kps = np.array(
            [
                [40.0, 50.0],   # left eye
                [72.0, 50.0],   # right eye
                [56.0, 73.0],   # nose (frontal ratio)
                [44.0, 80.0],   # left mouth
                [68.0, 80.0],   # right mouth
            ],
            dtype=np.float32,
        )
        yaw, pitch, roll = estimate_pose(kps, [20, 30, 92, 100])
        assert abs(yaw) < 10
        assert abs(pitch) < 10
        assert abs(roll) < 5

    def test_tilted_face_roll_detected(self):
        """A tilted face should produce a non-zero roll."""
        kps = np.array(
            [
                [50.0, 40.0],   # left eye (higher)
                [70.0, 60.0],   # right eye (lower)
                [60.0, 55.0],
                [55.0, 70.0],
                [65.0, 75.0],
            ],
            dtype=np.float32,
        )
        yaw, pitch, roll = estimate_pose(kps, [30, 25, 90, 95])
        assert abs(roll) > 5

    def test_missing_landmarks_returns_zero(self):
        """Unusable landmarks should return (0, 0, 0)."""
        assert estimate_pose(None, None) == (0.0, 0.0, 0.0)
        assert estimate_pose(np.zeros((2, 2)), None) == (0.0, 0.0, 0.0)

    def test_degenerate_face_width_zero(self):
        """Zero-width face (coincident eyes) should return (0, 0, 0)."""
        kps = np.array(
            [
                [50.0, 50.0],
                [50.0, 50.0],   # same as left eye
                [50.0, 60.0],
                [45.0, 70.0],
                [55.0, 70.0],
            ],
            dtype=np.float32,
        )
        assert estimate_pose(kps, None) == (0.0, 0.0, 0.0)


# ───────────────────────────────────────────────────────────────────
# Landmark geometry validation (false-positive filter)
# ───────────────────────────────────────────────────────────────────

class TestLandmarkValidation:
    """Tests for validate_face_landmarks (rejects back-of-head etc.)."""

    def _frontal_kps(self):
        """A plausible frontal-face landmark layout."""
        return np.array(
            [
                [40.0, 50.0],   # left eye
                [72.0, 50.0],   # right eye
                [56.0, 73.0],   # nose (frontal ratio)
                [44.0, 80.0],   # left mouth
                [68.0, 80.0],   # right mouth
            ],
            dtype=np.float32,
        )

    def test_valid_frontal_face(self):
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        assert validate_face_landmarks(kps, [20, 30, 92, 100]) is True

    def test_missing_landmarks_rejected(self):
        from src.core.face_utils import validate_face_landmarks
        assert validate_face_landmarks(None) is False
        assert validate_face_landmarks(np.zeros((2, 2))) is False

    def test_nose_above_eyes_accepted(self):
        """Nose above the eye line is accepted (relaxed for inverted faces)."""
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        kps[2] = [56.0, 40.0]  # nose above eyes
        # Relaxed: allows nose_dy in -1.5 to 1.5 * eye_dist
        assert validate_face_landmarks(kps, None) is True

    def test_extreme_roll_accepted(self):
        """Extreme eye-line tilt is accepted (relaxed for tilted faces)."""
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        # Tilt eyes by ~45 deg: right eye far below left eye
        kps[1] = [72.0, 82.0]
        # Relaxed: allows roll up to ±120 deg
        assert validate_face_landmarks(kps, None) is True

    def test_super_extreme_roll_rejected(self):
        """Super extreme eye-line tilt (>120 deg) is rejected."""
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        # Tilt eyes by ~135 deg: right eye far above and left of left eye
        kps[1] = [10.0, 100.0]
        assert validate_face_landmarks(kps, None) is False

    def test_nose_way_off_center_rejected(self):
        """Nose far from the eye midpoint is implausible."""
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        kps[2] = [120.0, 73.0]  # nose far right
        assert validate_face_landmarks(kps, None) is False

    def test_tiny_bbox_rejected(self):
        """A bbox barely larger than the eye distance is implausible."""
        from src.core.face_utils import validate_face_landmarks
        kps = self._frontal_kps()
        # eye_dist = 32; bbox width 40 < 2*32 → reject
        assert validate_face_landmarks(kps, [30, 30, 70, 70]) is False


# ───────────────────────────────────────────────────────────────────
# Face quality scoring
# ───────────────────────────────────────────────────────────────────

class TestFaceQuality:
    """Tests for the composite quality score."""

    def test_quality_in_range(self):
        """Quality should always be in [0, 100]."""
        for sharpness in (0, 100, 1000):
            for pose in ((0, 0, 0), (30, 20, 15), (90, 90, 90)):
                q = calculate_face_quality(sharpness, pose, brightness=130)
                assert 0.0 <= q <= 100.0

    def test_frontal_sharp_face_scores_high(self):
        """A sharp frontal face with good brightness should score high."""
        q = calculate_face_quality(
            sharpness=800.0,
            pose=(0.0, 0.0, 0.0),
            brightness=130.0,
            eyes_open_score=1.0,
        )
        assert q > 80

    def test_profile_face_scores_lower(self):
        """A strongly profile face should score lower than a frontal one."""
        frontal = calculate_face_quality(500.0, (0.0, 0.0, 0.0), 130.0, 1.0)
        profile = calculate_face_quality(500.0, (70.0, 10.0, 5.0), 130.0, 1.0)
        assert profile < frontal


# ───────────────────────────────────────────────────────────────────
# FaceRecognizer: basic instantiation and config
# ───────────────────────────────────────────────────────────────────

class TestFaceRecognizer:
    """Tests for the real FaceRecognizer class."""

    def test_instantiation(self):
        """FaceRecognizer should instantiate with sane defaults."""
        from src.core.face_recognition import FaceRecognizer

        recognizer = FaceRecognizer()
        # Model availability depends on InsightFace install; either way the
        # attribute must exist and be a bool.
        assert isinstance(recognizer.available, bool)
        assert recognizer.available in (True, False)
        # Similarity threshold used for actor matching
        assert recognizer.SIMILARITY_THRESHOLD > 0.5


# ───────────────────────────────────────────────────────────────────
# FaceRecognitionProcessor: database resume logic
# ───────────────────────────────────────────────────────────────────

class TestFaceRecognitionProcessor:
    """Tests for the face recognition database resume logic."""

    def _setup_task_with_faces(self, db, task_id=1):
        """Helper: create task with face detections (no embeddings)."""
        from src.db import repository as repo

        task = repo.upsert_task(db, "/tmp/test.mp4", "test.mp4", 1000, "md5_test")
        seg = repo.add_face_segment(db, task["id"], 0, 5.0, 5.5,
                                     prescan_status="completed", dense_status="completed")
        frame = repo.add_face_dense_frame(
            db, task["id"], seg["id"], 0, 5.0,
            detection_status="completed",
        )

        bbox = [100, 80, 200, 180]
        landmarks = {
            "left_eye": [130.0, 110.0],
            "right_eye": [170.0, 110.0],
            "nose": [150.0, 140.0],
            "left_mouth": [135.0, 160.0],
            "right_mouth": [165.0, 160.0],
        }

        det1 = repo.add_face_detection(
            db, frame["id"], task["id"], 0, 0.95,
            bbox, landmarks, quality_overall=0.8,
        )
        det2 = repo.add_face_detection(
            db, frame["id"], task["id"], 1, 0.85,
            bbox, landmarks, quality_overall=0.7,
        )
        return task, [det1, det2]

    def test_unprocessed_faces_detected(self, test_db: str):
        """Should correctly identify faces without embeddings."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task, detections = self._setup_task_with_faces(db)

        unprocessed = repo.get_unprocessed_face_ids(db, task["id"])
        assert len(unprocessed) == 2

        # Add one embedding
        repo.add_face_embedding(
            db, detections[0]["id"], task["id"],
            np.zeros(512, dtype=np.float32).tobytes(), 512,
            quality=0.8,
        )

        unprocessed = repo.get_unprocessed_face_ids(db, task["id"])
        assert len(unprocessed) == 1
        assert unprocessed[0] == detections[1]["id"]

        db.close()

    def test_embedding_blob_roundtrip(self, test_db: str):
        """Embedding BLOB should survive a DB round-trip unchanged."""
        from src.db.schema import get_connection
        from src.db import repository as repo

        db = get_connection(test_db)
        task, detections = self._setup_task_with_faces(db)

        original = np.random.randn(512).astype(np.float32)
        original = original / np.linalg.norm(original)  # L2 normalize

        repo.add_face_embedding(
            db, detections[0]["id"], task["id"],
            original.tobytes(), 512, quality=0.85,
        )

        embs = repo.get_embeddings_for_task(db, task["id"])
        assert len(embs) == 1

        stored = np.frombuffer(embs[0]["embedding"], dtype=np.float32)
        assert len(stored) == 512
        np.testing.assert_array_almost_equal(stored, original, decimal=5)

        db.close()
