"""Shared face-processing utilities.

Provides pure functions used by both the face detection pipeline
(``face_detector.py``) and the face recognition pipeline
(``face_recognition.py``):

- ``l2_normalize``: L2-normalize an embedding vector.
- ``align_face``:   Align a face crop to canonical 5-point landmarks.
- ``estimate_pose``: Estimate Euler angles (yaw/pitch/roll) from 5 landmarks.
- ``calculate_face_quality``: Composite quality score for a face.

These are kept dependency-light (numpy + cv2 only) so they can be unit-tested
with synthetic images and reused across modules without importing the heavy
InsightFace stack.
"""

import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


def get_insightface_model_root() -> str:
    """Return the InsightFace model root directory.

    Packaged (frozen) app  → ``~/.insightface``
    Development             → ``"."`` (project root, where ``./models/`` lives)
    """
    if getattr(sys, 'frozen', False):
        return str(Path.home() / '.insightface')
    return "."


def ensure_insightface_models_copied(name: str) -> None:
    """Copy bundled InsightFace models to ``~/.insightface/models/{name}``.

    In a PyInstaller-packaged app the ONNX files are inside ``sys._MEIPASS``.
    InsightFace only looks at ``{root}/models/{name}/``, so we copy them to
    the user home directory on first launch.  Subsequent calls are no-ops.
    """
    if not getattr(sys, 'frozen', False):
        return
    src = Path(sys._MEIPASS) / 'models' / name
    dst = Path.home() / '.insightface' / 'models' / name
    if src.exists() and not dst.exists():
        _logger.info("Copying bundled InsightFace model '%s' to %s", name, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dst))
        _logger.info("InsightFace model '%s' copied successfully", name)


def get_face_providers(full_pipeline: bool = True) -> list:
    """返回 InsightFace/ONNX 推理 providers（按硬件能力精细选择）。

    Apple Silicon(M 系列)实测结论(2026-09, M4 Max):
      - 完整管线(det scrfd640 + recog glintr100 + genderage):
        CoreML 38fps vs CPU 4.7fps → **8.1x**, CoreML 优先
      - glintr100(识别 embedding): CoreML 27.5x
      - scrfd 小图(320 det_size): CoreML 静态 shape 崩溃 → 纯检测路径必须 CPU
      - scrfd 640 单跑 CoreML 反而更慢 → 纯检测大图也用 CPU

    策略:
      full_pipeline=True (det+recog+genderage 一起):  CoreML + CPU
      full_pipeline=False (仅 detection, 尤其 320 小图): CPU

    config 可用 face_recognition.execution_provider 强制覆盖:
      "auto" (默认) | "cpu" | "coreml"
    """
    try:
        from src.core.config import ConfigManager
        mode = str(ConfigManager().get("face_recognition.execution_provider", "auto")).lower()
    except Exception:
        mode = "auto"

    import onnxruntime as ort
    available = set(ort.get_available_providers())
    has_coreml = "CoreMLExecutionProvider" in available

    if mode == "cpu":
        return ["CPUExecutionProvider"]
    if mode == "coreml":
        return (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                if has_coreml else ["CPUExecutionProvider"])
    # auto: 完整管线用 CoreML, 纯检测用 CPU
    if full_pipeline and has_coreml:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

# Canonical 5-point landmarks for a 112x112 aligned face (ArcFace convention):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
REFERENCE_5POINTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Key names accepted by align_face / estimate_pose.
_LANDMARK_KEYS = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector in place of a new array.

    Returns a zero vector unchanged (avoids division by zero).
    """
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _landmarks_to_array(landmarks: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """Convert a dict of 5 named landmarks to an (5, 2) float array.

    Returns None if any of the 5 canonical keys is missing.
    """
    if landmarks is None:
        return None
    try:
        pts = np.array(
            [[landmarks[k][0], landmarks[k][1]] for k in _LANDMARK_KEYS],
            dtype=np.float32,
        )
    except (KeyError, TypeError, IndexError):
        return None
    return pts


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a 2x3 similarity transform aligning ``src`` to ``dst``.

    Uses cv2.estimateAffinePartial2D (rotation + uniform scale + translation),
    which is the same alignment family InsightFace uses.
    """
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    # estimateAffinePartial2D needs at least 2 points; we always pass 5.
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        # Fall back to a plain translation (mean shift).
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        transform = np.array(
            [[1.0, 0.0, dst_mean[0] - src_mean[0]],
             [0.0, 1.0, dst_mean[1] - src_mean[1]]],
            dtype=np.float32,
        )
    return transform


def align_face(
    image: np.ndarray,
    landmarks: Dict[str, List[float]],
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Align a face crop using 5-point landmarks.

    Computes a similarity transform mapping the detected landmarks to the
    canonical ArcFace reference points, then warps the image.

    Args:
        image: BGR face image (H, W, 3).
        landmarks: Dict with keys left_eye/right_eye/nose/left_mouth/right_mouth.
        output_size: (width, height) of the aligned output.

    Returns:
        Aligned BGR image. Falls back to a center-crop resize if landmarks
        are unusable, so callers never receive an empty array.
    """
    if image is None or image.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    pts = _landmarks_to_array(landmarks)
    if pts is None:
        # No usable landmarks: center-crop + resize as a safe fallback.
        h, w = image.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = image[y0:y0 + side, x0:x0 + side]
        return cv2.resize(crop, output_size)

    transform = _similarity_transform(pts, REFERENCE_5POINTS_112)
    aligned = cv2.warpAffine(
        image,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def estimate_pose(
    kps: Optional[np.ndarray],
    bbox: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Estimate Euler angles (yaw, pitch, roll) in degrees from 5 landmarks.

    Uses the same geometry as the InsightFace pose estimator:
    - roll: tilt of the eye line
    - yaw: nose horizontal offset from the eye midpoint, scaled by face width
    - pitch: nose vertical offset ratio vs face height (0.33 == frontal)

    Args:
        kps: (5, 2) landmark array (left eye, right eye, nose, mouth corners),
             or None.
        bbox: Optional [x1, y1, x2, y2] used for face height in pitch.

    Returns:
        (yaw, pitch, roll). Returns (0.0, 0.0, 0.0) if landmarks are unusable.
    """
    if kps is None or len(kps) < 5:
        return 0.0, 0.0, 0.0

    kps = np.asarray(kps, dtype=np.float32)
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    face_width = float(np.linalg.norm(right_eye - left_eye))
    if face_width == 0:
        return 0.0, 0.0, 0.0

    # Roll: angle of the eye line.
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

    # Yaw: nose horizontal offset from eye midpoint, normalized by face width.
    eye_mid = (left_eye + right_eye) / 2
    nose_offset_x = nose[0] - eye_mid[0]
    yaw = float(np.degrees(np.arctan2(nose_offset_x, face_width))) * 2.0

    # Pitch: nose vertical offset ratio vs face height (0.33 is frontal).
    nose_offset_y = nose[1] - eye_mid[1]
    if bbox is not None and len(bbox) >= 4:
        face_height = float(abs(bbox[3] - bbox[1]))
    else:
        face_height = face_width
    if face_height > 0:
        ratio = nose_offset_y / face_height
        pitch = float((ratio - 0.33) * 180.0)
    else:
        pitch = 0.0

    return yaw, pitch, roll


def calculate_face_quality(
    sharpness: float,
    pose: Tuple[float, float, float],
    brightness: float,
    eyes_open_score: Optional[float] = None,
) -> float:
    """Composite face quality score in [0, 100].

    Weights (matching the pipeline defaults):
        pose 40% + clarity 25% + brightness 15% + eyes_open 20%

    Args:
        sharpness: Laplacian variance (higher is sharper).
        pose: (yaw, pitch, roll) in degrees.
        brightness: Mean pixel value (0-255).
        eyes_open_score: Optional 0-1 openness score; omitted → 0.5 neutral.

    Returns:
        Quality score in [0, 100].
    """
    yaw, pitch, roll = pose

    # Pose score: 100 when all angles are near 0, falls off with deviation.
    pose_score = max(
        0.0,
        100.0
        - 2.0 * abs(yaw)
        - 2.5 * abs(pitch)
        - 3.0 * abs(roll),
    )

    # Clarity: map Laplacian variance (typ. 100-1000) to 0-100.
    clarity_score = min(100.0, sharpness / 10.0)

    # Brightness: ideal around 120-140; penalize dark (<60) and blown (>200).
    brightness_score = 100.0
    if brightness < 60:
        brightness_score = brightness / 60.0 * 100.0
    elif brightness > 200:
        brightness_score = max(0.0, 100.0 - (brightness - 200) * 1.5)

    eyes = eyes_open_score if eyes_open_score is not None else 0.5
    eyes_score = float(np.clip(eyes, 0.0, 1.0)) * 100.0

    return round(
        0.40 * pose_score
        + 0.25 * clarity_score
        + 0.15 * brightness_score
        + 0.20 * eyes_score,
        2,
    )


def validate_face_landmarks(
    kps: Optional[np.ndarray],
    bbox: Optional[np.ndarray] = None,
    det_score: Optional[float] = None,
    min_eye_dist: float = 8.0,
    min_det_score: float = 0.2,
    debug: bool = False,
) -> bool:
    """Validate that 5-point landmarks form a plausible face geometry.

    Real faces have a stable landmark layout:
    - eye pair roughly horizontal (small roll)
    - nose tip below the eye midpoint (positive y offset), roughly centered
    - nose-to-eye distance a meaningful fraction of the eye distance
    - a minimum absolute eye distance (back-of-head false positives often
      produce tiny "imagined" landmarks, e.g. eye distance of a few px)
    - a minimum detection confidence

    False positives (back of head, skin patches, feet, etc.) usually produce
    degenerate landmark layouts that fail these checks.

    Args:
        kps: (5, 2) landmark array (left eye, right eye, nose, mouth corners).
        bbox: Optional [x1, y1, x2, y2] used as a size sanity check.
        det_score: Optional detection confidence; rejected if below threshold.
        min_eye_dist: Minimum eye-to-eye distance in pixels.
        min_det_score: Minimum detection confidence.

    Returns:
        True if the landmarks look like a real face.
    """
    if kps is None or len(kps) < 5:
        return False

    if det_score is not None and det_score < min_det_score:
        return False

    kps = np.asarray(kps, dtype=np.float32)
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < min_eye_dist:
        if debug:
            print(f"  FAIL: eye_dist={eye_dist:.1f} < {min_eye_dist}")
        return False

    eye_mid = (left_eye + right_eye) / 2

    # 1. Roll: eye line angle check (relaxed for tilted faces, allow up to ±120 deg).
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    roll_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
    # Normalize to -180 to 180 range, then check if within ±120 deg
    if abs(roll_angle) > 120:
        if debug:
            print(f"  FAIL: roll={roll_angle:.1f} > 120")
        return False

    # 2. Nose: roughly centered (|x| < 1.5 * eye_dist, very relaxed for extreme angles),
    #    and a meaningful vertical offset (-1.5 to 1.5 * eye_dist, allows inverted faces).
    nose_dx = nose[0] - eye_mid[0]
    nose_dy = nose[1] - eye_mid[1]
    if abs(nose_dy) > 1.5 * eye_dist:
        if debug:
            print(f"  FAIL: nose_dy={nose_dy:.1f}, eye_dist={eye_dist:.1f}, range=[{-1.5*eye_dist:.1f}, {1.5*eye_dist:.1f}]")
        return False
    if abs(nose_dx) > 1.5 * eye_dist:
        if debug:
            print(f"  FAIL: nose_dx={nose_dx:.1f} > {1.5*eye_dist:.1f}")
        return False

    # 3. Mouth: below the nose, on either side.
    mouth_dy = (left_mouth[1] + right_mouth[1]) / 2 - nose[1]
    if mouth_dy < 0:
        if debug:
            print(f"  FAIL: mouth_dy={mouth_dy:.1f} < 0")
        return False

    # 4. Face size sanity: bbox should be a reasonable multiple of eye dist.
    if bbox is not None and len(bbox) >= 4:
        bbox = np.asarray(bbox, dtype=np.float32)
        bbox_w = float(abs(bbox[2] - bbox[0]))
        bbox_h = float(abs(bbox[3] - bbox[1]))
        if bbox_w > 0 and bbox_w < 2.0 * eye_dist:
            if debug:
                print(f"  FAIL: bbox_w={bbox_w:.1f} < {2.0*eye_dist:.1f}")
            return False
        if bbox_h > 0 and bbox_h < 2.0 * eye_dist:
            if debug:
                print(f"  FAIL: bbox_h={bbox_h:.1f} < {2.0*eye_dist:.1f}")
            return False

    return True
