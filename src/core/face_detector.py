"""Stage 3: Face Detection (两阶段人脸检测).

Fast prescan (InsightFace low-res, ~5-10ms/frame) to locate face segments,
then dense sampling + high-quality detection (AuraFace, ~50ms/frame).

Independent sampling chain — does NOT reuse stage 0 uniform frames.
"""

import cv2
import numpy as np
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Graceful degradation: MediaPipe (fast prescan)
# ───────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp  # noqa: F401 — 可用性探测（真实导入验证依赖链）
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available — fast face prescan disabled")

# ───────────────────────────────────────────────────────────────────
# Graceful degradation: InsightFace/AuraFace (high-quality detection)
# ───────────────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available — high-quality face detection disabled")


# ───────────────────────────────────────────────────────────────────
# Data classes
# ───────────────────────────────────────────────────────────────────

@dataclass
class FaceSegment:
    """A time segment where faces were detected in prescan."""
    start_time: float
    end_time: float
    face_count: int = 0


@dataclass
class DenseFrame:
    """A densely-sampled frame from a face segment."""
    frame_index: int
    timestamp: float
    frame_path: str = ""
    faces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FaceQuality:
    """Face quality assessment scores."""
    clarity: float = 0.0
    pose: float = 0.0
    brightness: float = 0.0
    overall: float = 0.0


@dataclass
class FaceDetectionResult:
    """Complete face detection result for a video."""
    task_id: int
    face_segments: List[FaceSegment] = field(default_factory=list)
    dense_frames: List[DenseFrame] = field(default_factory=list)
    total_faces_detected: int = 0
    prescan_frame_count: int = 0
    dense_frame_count: int = 0


# ───────────────────────────────────────────────────────────────────
# Quality evaluation helpers
# ───────────────────────────────────────────────────────────────────

def evaluate_clarity(face_img: np.ndarray) -> float:
    """Evaluate face clarity using Laplacian variance.

    Returns a score in [0, 1] range (1 = very sharp).
    """
    if face_img is None or face_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize: 0-1000 → 0-1 (sigmoid-like mapping)
    score = min(1.0, laplacian_var / 500.0)
    return float(score)


def evaluate_pose(yaw: float, pitch: float, roll: float) -> float:
    """Evaluate face pose quality based on Euler angles.

    Frontal face = 1.0, heavily tilted = 0.0.
    Angles in degrees.
    """
    yaw_score = max(0.0, 1.0 - abs(yaw) / 90.0)
    pitch_score = max(0.0, 1.0 - abs(pitch) / 60.0)
    roll_score = max(0.0, 1.0 - abs(roll) / 45.0)

    # Pose is limited by the worst angle
    score = min(yaw_score, pitch_score, roll_score)
    return float(score)


def evaluate_brightness(face_img: np.ndarray) -> float:
    """Evaluate face brightness.

    Optimal brightness (100-200 on 0-255 scale) = 1.0.
    """
    if face_img is None or face_img.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    # Best at 128-180, linear falloff
    if 100 <= mean_brightness <= 200:
        return 1.0
    elif mean_brightness < 100:
        return mean_brightness / 100.0
    else:
        return (255.0 - mean_brightness) / 55.0


def calculate_overall_quality(
    clarity: float, pose: float, brightness: float,
    clarity_weight: float = 0.4, pose_weight: float = 0.4, brightness_weight: float = 0.2,
) -> float:
    """Calculate weighted overall quality score."""
    total = clarity * clarity_weight + pose * pose_weight + brightness * brightness_weight
    return float(total)


# ───────────────────────────────────────────────────────────────────
# Fast prescan: InsightFace (low-resolution, relaxed threshold)
# ───────────────────────────────────────────────────────────────────

class InsightFacePrescanner:
    """Fast face prescan using InsightFace (~5-10ms/frame).

    Scenes are sampled at 2 fps to quickly identify time segments
    containing faces, avoiding expensive full-resolution detection
    on every frame.
    
    Uses InsightFace with low resolution and relaxed thresholds for speed.
    """

    def __init__(self, min_detection_confidence: float = 0.3):
        self.min_detection_confidence = min_detection_confidence
        self._detector = None

    @property
    def available(self) -> bool:
        return _INSIGHTFACE_AVAILABLE

    def _init_detector(self):
        """Lazy-initialize the InsightFace detector for prescan."""
        if self._detector is not None:
            return
        if not _INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not available")

        # Use AuraFace (Apache-2.0, 检测与主链路同一模型) at low resolution for speed
        from src.core.face_utils import (
            get_insightface_model_root, ensure_insightface_models_copied,
            get_face_providers,
        )
        model_root = get_insightface_model_root()
        ensure_insightface_models_copied("auraface")
        # 模型缺失守卫: 不触发 insightface 自动下载(下载源不存在) —— 与 face_recognition 守卫一致
        _adir = Path(model_root) / "models" / "auraface"
        _aneed = ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx"]
        if any(not (_adir / f).exists() for f in _aneed):
            raise RuntimeError(
                f"人脸模型 auraface 缺失({_adir}), 请先在「设置 → 环境初始化」下载"
            )
        self._detector = FaceAnalysis(
            name='auraface',
            root=model_root,
            providers=get_face_providers(full_pipeline=False),
            allowed_modules=["detection"],  # 预扫描只需检测, 不加载识别/关键点权重
        )
        # Low resolution for speed, relaxed threshold for sensitivity
        self._detector.prepare(ctx_id=0, det_size=(320, 320), det_thresh=self.min_detection_confidence)
        logger.info("InsightFace prescanner initialized (320x320, thresh=%.2f)", self.min_detection_confidence)

    def detect_frame(self, frame: np.ndarray) -> Tuple[bool, int]:
        """Detect faces in a single frame.

        Returns:
            (has_face, face_count)
        """
        if frame is None or frame.size == 0:
            return False, 0

        self._init_detector()

        # Use InsightFace for detection
        try:
            faces = self._detector.get(frame)
            if len(faces) > 0:
                # 只把"带完整五官(有几何合理的关键点)"的框算作真脸，
                # 避免后脑勺/皮肤/模糊区等假阳性把无人脸片段误判为"人脸片段"。
                from src.core.face_utils import validate_face_landmarks
                true_faces = [
                    f for f in faces
                    if validate_face_landmarks(
                        getattr(f, "kps", None),
                        getattr(f, "bbox", None),
                        det_score=float(getattr(f, "det_score", 0.5)),
                    )
                ]
                if true_faces:
                    return True, len(true_faces)
            return False, 0
        except Exception as e:
            logger.warning("Face detection failed: %s", e)
            return False, 0

    def scan_video_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_fps: float = 2.0,
        offset: float = 0.0,
    ) -> List[FaceSegment]:
        """Scan a video time range at sample_fps to find face segments.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_fps: Sampling rate in frames per second
            offset: Sampling phase offset in seconds. 用该参数做"第二次偏移扫描"，
                避开第一次采样的时间格点，把恰好落在两格之间的短暂人脸兜住。

        Returns:
            List of FaceSegment where faces were detected
        """
        self._init_detector()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Clamp time range
        start_time = max(0.0, start_time)
        end_time = min(video_duration, end_time)

        if start_time >= end_time:
            cap.release()
            return []

        segments: List[FaceSegment] = []
        current_start: Optional[float] = None
        max_face_count = 0

        sample_interval = 1.0 / sample_fps
        t = start_time + offset
        frames_scanned = 0

        while t < end_time:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                t += sample_interval
                continue

            frames_scanned += 1
            has_face, face_count = self.detect_frame(frame)

            if has_face:
                if current_start is None:
                    current_start = t
                    max_face_count = face_count
                else:
                    max_face_count = max(max_face_count, face_count)
            else:
                if current_start is not None:
                    segments.append(FaceSegment(
                        start_time=current_start,
                        end_time=t,
                        face_count=max_face_count,
                    ))
                    current_start = None
                    max_face_count = 0

            t += sample_interval

        # Close any open segment
        if current_start is not None:
            segments.append(FaceSegment(
                start_time=current_start,
                end_time=end_time,
                face_count=max_face_count,
            ))

        cap.release()
        logger.info(
            "Prescan: %d frames scanned, %d face segments found (%.2f-%.2fs)",
            frames_scanned, len(segments), start_time, end_time,
        )
        return segments

    def close(self):
        """Release resources."""
        if self._detector is not None:
            del self._detector
            self._detector = None


# ───────────────────────────────────────────────────────────────────
# High-quality detection: AuraFace / InsightFace
# ───────────────────────────────────────────────────────────────────

class AuraFaceDetector:
    """High-quality face detection using AuraFace (InsightFace).

    Provides bounding boxes, 5-point landmarks, confidence scores,
    and pose estimation (via keypoint analysis).
    """

    def __init__(self, det_thresh: float = 0.5, det_size: Tuple[int, int] = (640, 640)):
        self.det_thresh = det_thresh
        self.det_size = det_size
        self._app = None

    @property
    def available(self) -> bool:
        return _INSIGHTFACE_AVAILABLE

    def _init_model(self):
        """Lazy-initialize the AuraFace detector."""
        if self._app is not None:
            return
        if not _INSIGHTFACE_AVAILABLE:
            raise RuntimeError("InsightFace is not available")

        # 复用 ModelManager 已加载的共享检测模型（只读查询，不触发加载，
        # 避免 wrapper.load() → _init_model() → get() 的递归初始化）。
        try:
            from src.core.model_manager import ModelManager
            mm = ModelManager.get_instance()
            wrapper = mm.peek("face_detection")
            if wrapper is not None and wrapper.is_available():
                inner = getattr(wrapper, "_detector", None)
                if inner is not None and getattr(inner, "_app", None) is not None:
                    self._app = inner._app
                    logger.info(
                        "AuraFaceDetector reusing shared model from ModelManager "
                        "(thresh=%.2f)", self.det_thresh,
                    )
                    return
        except Exception as e:
            logger.warning("ModelManager lookup failed, falling back to direct init: %s", e)

        from src.core.face_utils import (
            get_insightface_model_root, ensure_insightface_models_copied,
            get_face_providers,
        )
        model_root = get_insightface_model_root()
        ensure_insightface_models_copied("auraface")
        # 模型缺失守卫: 不触发 insightface 自动下载 —— 与 face_recognition 守卫一致
        _adir = Path(model_root) / "models" / "auraface"
        _aneed = ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx"]
        if any(not (_adir / f).exists() for f in _aneed):
            raise RuntimeError(
                f"人脸模型 auraface 缺失({_adir}), 请先在「设置 → 环境初始化」下载"
            )

        logger.info("Initializing AuraFace detector (root=%s)...", model_root)
        # 纯检测路径用 CPU（scrfd 在 CoreML 下 shape 崩溃/无收益）
        self._app = FaceAnalysis(
            name="auraface",
            root=model_root,
            providers=get_face_providers(full_pipeline=False),
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        logger.info("AuraFace detector initialized (det_thresh=%.2f)", self.det_thresh)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a frame.

        Returns:
            List of face dicts with: confidence, bbox, landmarks, pose
        """
        if frame is None or frame.size == 0:
            return []

        self._init_model()

        faces = self._app.get(frame)
        results = []

        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            kps = face.kps if hasattr(face, 'kps') else None

            # Extract 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
            landmarks_5pt = {}
            if kps is not None and len(kps) >= 5:
                landmarks_5pt = {
                    "left_eye": [float(kps[0][0]), float(kps[0][1])],
                    "right_eye": [float(kps[1][0]), float(kps[1][1])],
                    "nose": [float(kps[2][0]), float(kps[2][1])],
                    "left_mouth": [float(kps[3][0]), float(kps[3][1])],
                    "right_mouth": [float(kps[4][0]), float(kps[4][1])],
                }

            # Estimate pose from 5-point landmarks
            yaw, pitch, roll = self._estimate_pose(kps, bbox)

            results.append({
                "face_index": i,
                "confidence": float(face.det_score),
                "bbox": bbox,
                "landmarks": landmarks_5pt,
                "kps": np.asarray(kps, dtype=float).tolist() if kps is not None else None,
                "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
            })

        return results

    @staticmethod
    def _estimate_pose(kps, bbox) -> Tuple[float, float, float]:
        """Rough pose estimation from 5-point landmarks.

        Returns (yaw, pitch, roll) in degrees.
        This is a simplified estimation; accurate pose requires 3D model fitting.
        """
        from src.core.face_utils import estimate_pose
        return estimate_pose(kps, bbox)

    def close(self):
        """Release model resources."""
        if self._app is not None:
            del self._app
            self._app = None


# ───────────────────────────────────────────────────────────────────
# Main Face Detector (two-stage pipeline)
# ───────────────────────────────────────────────────────────────────

class FaceDetector:
    """Two-stage face detection pipeline.

    Stage 1: Fast prescan with InsightFace low-res (2 fps) to find face segments
    Stage 2: Dense sampling (5 fps) in face segments + AuraFace high-quality detection

    Usage:
        detector = FaceDetector()
        result = detector.detect_video(video_path, task_id=1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from src.core.config import ConfigManager
        cfg = config or ConfigManager().get_all()

        fd_cfg = cfg.get("face_detection", {})
        self.prescan_fps = fd_cfg.get("prescan_fps", 2)
        self.dense_fps = fd_cfg.get("dense_fps", 5)
        self.segment_margin = fd_cfg.get("segment_margin_seconds", 1.0)
        self.confidence_threshold = fd_cfg.get("confidence_threshold", 0.5)
        self.min_face_area_ratio = fd_cfg.get("min_face_area_ratio", 0.005)

        qe = fd_cfg.get("quality_evaluation", {})
        self.quality_enabled = qe.get("enabled", True)
        self.clarity_weight = qe.get("clarity_weight", 0.4)
        self.pose_weight = qe.get("pose_weight", 0.4)
        self.brightness_weight = qe.get("brightness_weight", 0.2)

        qf = fd_cfg.get("quality_filter", {})
        self.quality_filter_enabled = qf.get("enabled", True)
        self.min_quality = qf.get("min_quality", 0.3)

        # 片段内智能抽帧配置（face-quality-filter 方案）
        self.smart_sampling = fd_cfg.get("smart_sampling", True)
        self.smart_frames_per_segment = fd_cfg.get("smart_frames_per_segment", 20)
        self.smart_retry_window_seconds = fd_cfg.get("smart_retry_window_seconds", 10)
        self.smart_max_retries = fd_cfg.get("smart_max_retries", 3)

        # 本片生效值（按视频计算；配置值 confidence_threshold / min_face_area_ratio
        # **不再被改写**——见 _apply_video_thresholds）
        self._eff_confidence = self.confidence_threshold
        self._eff_area_ratio = self.min_face_area_ratio

        self._prescanner: Optional[InsightFacePrescanner] = None
        self._detector: Optional[AuraFaceDetector] = None

    @property
    def effective_confidence(self) -> float:
        """**本片生效**的检测置信度（按视频分辨率放宽；见 `_apply_video_thresholds`）。"""
        return self._eff_confidence

    @property
    def effective_min_face_area_ratio(self) -> float:
        """**本片生效**的人脸面积下限。"""
        return self._eff_area_ratio

    def _apply_video_thresholds(self, video_width: int, video_height: int) -> None:
        """按视频分辨率计算本片生效阈值，**不写回配置值**。

        低清片源（< 360p 基准像素）人脸天然小、置信度偏低：置信度 0.5→0.25、
        人脸面积下限 0.005→0.002（只放宽）。

        为什么要单独存"生效值"：此前直接改写 `self.confidence_threshold` /
        `self.min_face_area_ratio`，而下一部视频又拿它们当"基准值"读——
        **一次低清会让后续高清视频也按低阈值跑**（取值串味）；且共享实例在并发下会互相干扰。
        （见 openspec/changes/unify-face-thresholds §4.3）
        """
        eff_conf = self.confidence_threshold
        eff_area = self.min_face_area_ratio
        if video_width * video_height < 230400:  # 低于 360p 基准像素
            eff_conf = min(eff_conf, 0.25)
            eff_area = min(eff_area, 0.002)
            logger.info(
                "低分辨率视频 (%dx%d): 检测阈值 %.2f→%.2f, 人脸面积下限 %.4f→%.4f",
                video_width, video_height,
                self.confidence_threshold, eff_conf,
                self.min_face_area_ratio, eff_area,
            )
        if (eff_conf, eff_area) == (self._eff_confidence, self._eff_area_ratio):
            return
        self._eff_confidence = eff_conf
        self._eff_area_ratio = eff_area
        # 生效值变化 → 丢弃按旧值构建的缓存实例，下次访问按新值重建
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._prescanner is not None:
            self._prescanner.close()
            self._prescanner = None

    @property
    def prescanner(self) -> InsightFacePrescanner:
        if self._prescanner is None:
            self._prescanner = InsightFacePrescanner(
                min_detection_confidence=self._eff_confidence,
            )
        return self._prescanner

    @property
    def detector(self) -> AuraFaceDetector:
        if self._detector is None:
            self._detector = AuraFaceDetector(
                det_thresh=self._eff_confidence,
            )
        return self._detector

    def detect_video(
        self,
        video_path: str,
        task_id: int,
        output_dir: Optional[str] = None,
        scene_segments: Optional[List[Tuple[float, float]]] = None,
        progress_callback: Optional[callable] = None,
    ) -> FaceDetectionResult:
        """Run full two-stage face detection on a video.

        Args:
            video_path: Path to video file
            task_id: Task identifier for database tracking
            output_dir: Directory to save dense frame images
            scene_segments: Optional list of (start_time, end_time) scene ranges.
                If None, scans the entire video as one segment.
            progress_callback: Optional callback(current, total, message)

        Returns:
            FaceDetectionResult with all detection data
        """
        result = FaceDetectionResult(task_id=task_id)

        # Get video duration & resolution
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return result
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 低分辨率视频自适应：360p 以下小人脸置信度天然偏低（0.2-0.3），
        # 用统一 0.5 阈值会漏检。按分辨率算出**本片生效值**（低清才放宽）。
        # 注意：只写"生效值"，**不改写配置值**——否则一次低清会让后续高清也按低阈值跑。
        self._apply_video_thresholds(video_width, video_height)

        # If no scene segments provided, treat entire video as one scene
        if not scene_segments:
            scene_segments = [(0.0, duration)]

        # ── Stage 1: Fast prescan (or skip if scene_segments already have faces) ──
        all_segments: List[FaceSegment] = []
        prescan_frame_count = 0

        # 如果传入了 scene_segments，说明调用方已经判断过这些片段有人脸，
        # 直接作为 face_segments 使用，跳过预扫描（节省大量时间）
        if scene_segments and len(scene_segments) < 500:  # 有明确的场景列表
            for scene_start, scene_end in scene_segments:
                all_segments.append(FaceSegment(
                    start_time=scene_start,
                    end_time=scene_end,
                    face_count=1,
                ))
            logger.info("跳过预扫描，直接使用 %d 个已知有人脸的场景", len(all_segments))
        else:
            # 无场景信息，需要预扫描
            if progress_callback:
                progress_callback(0, 100, "阶段 1/2: 快速预扫描...")

            # 多轮偏移扫描的兜底
            rescan_attempts = 4
            for attempt in range(rescan_attempts):
                frac = 0.0 if attempt == 0 else (0.5 ** attempt)
                offset_sec = frac / self.prescan_fps

                if self.prescanner.available:
                    for scene_start, scene_end in scene_segments:
                        segments = self.prescanner.scan_video_segment(
                            video_path, scene_start, scene_end,
                            sample_fps=self.prescan_fps,
                            offset=offset_sec,
                        )
                        prescan_frame_count += int((scene_end - scene_start) * self.prescan_fps)
                        all_segments.extend(segments)
                else:
                    for scene_start, scene_end in scene_segments:
                        all_segments.append(FaceSegment(
                            start_time=scene_start,
                            end_time=scene_end,
                            face_count=1,
                        ))

                if all_segments:
                    break
                if not self.prescanner.available:
                    break
                if attempt == 0:
                    logger.info("Prescan 第一轮未发现人脸，开始偏移补扫抓取短暂露面...")

        result.face_segments = all_segments
        result.prescan_frame_count = prescan_frame_count
        logger.info("Prescan complete: %d face segments", len(all_segments))

        if not all_segments:
            logger.info("No faces found in prescan — skipping dense detection")
            if progress_callback:
                progress_callback(100, 100, "检测完成：未发现人脸")
            return result

        # ── Stage 2: Dense sampling + high-quality detection ──
        if progress_callback:
            progress_callback(50, 100, f"阶段 2/2: 密集采样检测 ({len(all_segments)}个片段)...")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        total_dense_frames = sum(
            int(max(1, (seg.end_time - seg.start_time + 2 * self.segment_margin) * self.dense_fps))
            for seg in all_segments
        )
        processed_dense = 0

        for seg_idx, segment in enumerate(all_segments):
            # Expand segment with margin
            seg_start = max(0.0, segment.start_time - self.segment_margin)
            seg_end = min(duration, segment.end_time + self.segment_margin)

            dense_frames = self._dense_sample_and_detect(
                video_path, seg_start, seg_end,
                task_id=task_id,
                segment_id=seg_idx,
                output_dir=output_dir,
            )
            result.dense_frames.extend(dense_frames)
            result.total_faces_detected += sum(len(f.faces) for f in dense_frames)

            processed_dense += len(dense_frames)
            if progress_callback:
                pct = 50 + int(processed_dense / max(1, total_dense_frames) * 50)
                progress_callback(pct, 100, f"检测中... {processed_dense}/{total_dense_frames} 帧")

        result.dense_frame_count = len(result.dense_frames)

        if progress_callback:
            progress_callback(100, 100, f"检测完成：{result.total_faces_detected}张人脸")

        logger.info(
            "Face detection complete: %d dense frames, %d faces detected",
            result.dense_frame_count, result.total_faces_detected,
        )
        return result

    def _dense_sample_and_detect(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        task_id: int,
        segment_id: int,
        output_dir: Optional[str] = None,
    ) -> List[DenseFrame]:
        """Densely sample a time segment and run high-quality face detection.

        When smart sampling is enabled (``face_detection.smart_sampling``),
        the segment is sampled with ``SmartFrameSampler`` (sharpness-aware
        sampling with blur retry) so blurry frames are replaced by nearby
        sharp ones. If smart sampling yields too few frames (e.g. a fully
        blurry segment), it falls back to the legacy fixed-rate sampling.

        Returns list of DenseFrame with face detections.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = frame_width * frame_height
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Frame range for this segment
        start_frame = int(start_time * fps)
        end_frame = min(total_frames - 1, int(end_time * fps))
        if start_frame > end_frame:
            return []

        # Expected number of frames from legacy fixed-rate sampling
        expected_frames = max(1, int((end_time - start_time) * self.dense_fps))

        dense_frames: List[DenseFrame] = []

        # ── Smart sampling path (sharpness-aware, with blur retry) ──
        if self.smart_sampling:
            try:
                from src.core.smart_frame_sampler import SmartFrameSampler

                target_frames = min(
                    self.smart_frames_per_segment,
                    expected_frames,
                )
                # Use a per-segment subdirectory to keep files separate.
                seg_output = os.path.join(output_dir, f"seg{segment_id}") if output_dir else None
                sampler = SmartFrameSampler(
                    retry_on_blur=True,
                    max_retries=self.smart_max_retries,
                    retry_window_seconds=self.smart_retry_window_seconds,
                    min_kept_frames=min(3, target_frames),
                )
                smart_paths = sampler.sample_frames(
                    video_path,
                    start_frame,
                    end_frame,
                    n_frames=target_frames,
                    output_dir=seg_output,
                )
                logger.info(
                    "Smart sampling segment %d: got %d/%d sharp frames "
                    "(range [%d, %d])",
                    segment_id, len(smart_paths), target_frames,
                    start_frame, end_frame,
                )

                # Only use smart results if we got enough frames.
                # min_smart is relative to target_frames (not expected_frames)
                # so a small smart_frames_per_segment cap doesn't force a
                # fallback every time.
                min_smart = max(1, int(target_frames * 0.5))
                if len(smart_paths) >= min_smart:
                    for idx, path in enumerate(smart_paths):
                        frame_num = int(Path(path).stem.split("_")[-1])
                        timestamp = frame_num / fps
                        try:
                            frame = cv2.imread(path)
                        except Exception as e:
                            logger.warning(f"cv2.imread failed for smart frame: {path}, error: {e}")
                            continue
                        if frame is None:
                            continue
                        faces = self._detect_frame_faces(frame, frame_area)
                        dense_frames.append(DenseFrame(
                            frame_index=idx,
                            timestamp=timestamp,
                            frame_path=path,
                            faces=faces,
                        ))
                    return dense_frames
                else:
                    logger.warning(
                        "Smart sampling segment %d too few frames (%d < %d), "
                        "falling back to fixed-rate sampling",
                        segment_id, len(smart_paths), min_smart,
                    )
            except Exception as e:
                logger.warning(
                    "Smart sampling segment %d failed (%s), falling back to "
                    "fixed-rate sampling", segment_id, e,
                )

        # ── Legacy fixed-rate sampling path ──
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        sample_interval = 1.0 / self.dense_fps
        frame_idx = 0
        t = start_time

        while t < end_time:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                t += sample_interval
                frame_idx += 1
                continue

            # Save frame if output dir specified
            frame_path = ""
            if output_dir:
                fname = f"task{task_id}_seg{segment_id}_f{frame_idx}.jpg"
                frame_path = os.path.join(output_dir, fname)
                cv2.imwrite(frame_path, frame)

            faces = self._detect_frame_faces(frame, frame_area)

            dense_frames.append(DenseFrame(
                frame_index=frame_idx,
                timestamp=t,
                frame_path=frame_path,
                faces=faces,
            ))

            frame_idx += 1
            t += sample_interval

        cap.release()
        return dense_frames

    def _detect_frame_faces(self, frame: np.ndarray, frame_area: int) -> List[Dict[str, Any]]:
        """Run AuraFace detection + quality evaluation on a single frame.

        Returns the list of accepted face dicts (after area & quality filters).
        """
        faces: List[Dict[str, Any]] = []
        if not self.detector.available:
            return faces

        raw_faces = self.detector.detect(frame)
        for face in raw_faces:
            # Filter by minimum face area
            bbox = face["bbox"]
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = face_area / max(1, frame_area)
            if area_ratio < self._eff_area_ratio:
                continue

            # Landmark-geometry gate: 后脑勺/皮肤/模糊区等假阳性通常关键点布局
            # 退化，无法通过几何校验。确保框内是带完整五官的真脸才继续。
            from src.core.face_utils import validate_face_landmarks
            if not validate_face_landmarks(
                face.get("kps"),
                bbox,
                det_score=face.get("confidence", 0.5),
            ):
                continue

            # Quality evaluation
            if self.quality_enabled:
                quality = self._evaluate_face_quality(frame, face)
                face["quality"] = {
                    "clarity": quality.clarity,
                    "pose": quality.pose,
                    "brightness": quality.brightness,
                    "overall": quality.overall,
                }

                # Quality filter
                if self.quality_filter_enabled and quality.overall < self.min_quality:
                    continue
            else:
                face["quality"] = {
                    "clarity": 0.0, "pose": 0.0,
                    "brightness": 0.0, "overall": 0.0,
                }

            faces.append(face)
        return faces

    def _evaluate_face_quality(self, frame: np.ndarray, face: Dict) -> FaceQuality:
        """Evaluate quality of a detected face."""
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return FaceQuality()

        face_img = frame[y1:y2, x1:x2]

        clarity = evaluate_clarity(face_img)

        pose_data = face.get("pose", {})
        pose = evaluate_pose(
            pose_data.get("yaw", 0.0),
            pose_data.get("pitch", 0.0),
            pose_data.get("roll", 0.0),
        )

        brightness = evaluate_brightness(face_img)

        overall = calculate_overall_quality(
            clarity, pose, brightness,
            self.clarity_weight, self.pose_weight, self.brightness_weight,
        )

        return FaceQuality(
            clarity=clarity,
            pose=pose,
            brightness=brightness,
            overall=overall,
        )

    def close(self):
        """Release all model resources."""
        if self._prescanner is not None:
            self._prescanner.close()
            self._prescanner = None
        if self._detector is not None:
            self._detector.close()
            self._detector = None


# ───────────────────────────────────────────────────────────────────
# Stage Processor (checkpoint/resume)
# ───────────────────────────────────────────────────────────────────

class FaceDetectionProcessor:
    """Stage 3 processor with database-backed checkpoint/resume.

    Processing units: face segments (prescan) → dense frames (detection).
    Resume granularity: per-segment prescan, per-frame detection.
    """

    def __init__(self, db_connection=None):
        self.db = db_connection
        self._detector: Optional[FaceDetector] = None

    def _ensure_db(self):
        if self.db is None:
            from src.db.schema import get_connection, init_db
            init_db()
            self.db = get_connection()

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    def run_for_task(
        self,
        task_id: int,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Run face detection for a task with resume support.

        Returns summary dict with segment/frame/face counts.
        """
        self._ensure_db()

        # Get scene information for this task (from montage_units)
        scenes = self._get_task_scenes(task_id)

        if not output_dir:
            from src.core.config import ConfigManager
            cfg = ConfigManager()
            base_dir = cfg.get("video.output_dir", "./data/frames")
            output_dir = os.path.join(base_dir, f"task_{task_id}", "face_dense")

        # Phase 1: Prescan (per-scene resume)
        segments = self._run_prescan(task_id, video_path, scenes, output_dir, progress_callback)

        if not segments:
            return {
                "task_id": task_id,
                "prescan_segments": 0,
                "dense_frames": 0,
                "faces_detected": 0,
            }

        # Phase 2: Dense detection (per-frame resume)
        total_faces = self._run_dense_detection(task_id, video_path, output_dir, progress_callback)

        return {
            "task_id": task_id,
            "prescan_segments": len(segments),
            "dense_frames": self._count_dense_frames(task_id),
            "faces_detected": total_faces,
        }

    def _get_task_scenes(self, task_id: int) -> List[Tuple[float, float]]:
        """Get scene time ranges for a task from montage_units."""
        self._ensure_db()

        rows = self.db.execute(
            "SELECT start_frame, end_frame FROM montage_units WHERE task_id = ? ORDER BY unit_index",
            (task_id,),
        ).fetchall()

        if not rows:
            return []

        # Need FPS to convert frames to seconds
        task = self.db.execute(
            "SELECT file_path, duration FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

        if not task:
            return []

        cap = cv2.VideoCapture(task["file_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        scenes = []
        for row in rows:
            start_time = row["start_frame"] / fps
            end_time = row["end_frame"] / fps
            scenes.append((start_time, end_time))

        return scenes

    def _run_prescan(
        self, task_id: int, video_path: str,
        scenes: List[Tuple[float, float]],
        output_dir: str,
        progress_callback: Optional[callable],
    ) -> List[int]:
        """Run prescan, skipping already-completed scenes.

        Returns list of segment IDs that have faces.
        """
        # Check which segments already exist for this task
        existing = self.db.execute(
            "SELECT id, prescan_status FROM face_segments WHERE task_id = ?",
            (task_id,),
        ).fetchall()

        if existing:
            logger.info("Task %d: %d prescan segments already exist", task_id, len(existing))
            # Return IDs of segments that have faces
            rows = self.db.execute(
                "SELECT id FROM face_segments WHERE task_id = ? AND face_count_prescan > 0",
                (task_id,),
            ).fetchall()
            return [r["id"] for r in rows]

        # Run prescan for each scene
        face_segment_ids = []
        for unit_idx, (start_time, end_time) in enumerate(scenes):
            segments = self.detector.prescanner.scan_video_segment(
                video_path, start_time, end_time,
                sample_fps=self.detector.prescan_fps,
            )

            for seg in segments:
                cur = self.db.execute(
                    """INSERT INTO face_segments
                       (task_id, unit_index, start_time, end_time, face_count_prescan, prescan_status)
                       VALUES (?, ?, ?, ?, ?, 'completed')""",
                    (task_id, unit_idx, seg.start_time, seg.end_time, seg.face_count),
                )
                if seg.face_count > 0:
                    face_segment_ids.append(cur.lastrowid)
            self.db.commit()

        return face_segment_ids

    def _run_dense_detection(
        self, task_id: int, video_path: str,
        output_dir: str,
        progress_callback: Optional[callable],
    ) -> int:
        """Run dense face detection on face segments, skipping completed frames.

        Returns total number of faces detected.
        """
        # Get segments that have faces
        segments = self.db.execute(
            "SELECT id, start_time, end_time FROM face_segments WHERE task_id = ? AND face_count_prescan > 0",
            (task_id,),
        ).fetchall()

        if not segments:
            return 0

        total_faces = 0

        for seg in segments:
            seg_id = seg["id"]

            # Check if we already have dense frames for this segment
            existing_frames = self.db.execute(
                "SELECT COUNT(*) as cnt FROM face_dense_frames WHERE segment_id = ? AND detection_status = 'completed'",
                (seg_id,),
            ).fetchone()["cnt"]

            if existing_frames > 0:
                # Count faces from existing detections
                face_count = self.db.execute(
                    """SELECT COUNT(*) as cnt FROM face_detections fd
                       JOIN face_dense_frames fdf ON fd.dense_frame_id = fdf.id
                       WHERE fdf.segment_id = ?""",
                    (seg_id,),
                ).fetchone()["cnt"]
                total_faces += face_count
                continue

            # Dense sample and detect
            start_time = max(0.0, seg["start_time"] - self.detector.segment_margin)
            end_time = seg["end_time"] + self.detector.segment_margin

            dense_frames = self.detector._dense_sample_and_detect(
                video_path, start_time, end_time,
                task_id=task_id,
                segment_id=seg_id,
                output_dir=output_dir,
            )

            # Save to database
            for df in dense_frames:
                cur = self.db.execute(
                    """INSERT INTO face_dense_frames
                       (task_id, segment_id, frame_index, timestamp, frame_path, detection_status)
                       VALUES (?, ?, ?, ?, ?, 'completed')""",
                    (task_id, seg_id, df.frame_index, df.timestamp, df.frame_path),
                )
                dense_frame_id = cur.lastrowid

                for face in df.faces:
                    quality = face.get("quality", {})
                    self.db.execute(
                        """INSERT INTO face_detections
                           (dense_frame_id, task_id, face_index, confidence,
                            bbox, landmarks, quality_clarity, quality_pose,
                            quality_brightness, quality_overall)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            dense_frame_id, task_id, face["face_index"],
                            face["confidence"],
                            json.dumps(face["bbox"]),
                            json.dumps(face.get("landmarks", {})),
                            quality.get("clarity", 0.0),
                            quality.get("pose", 0.0),
                            quality.get("brightness", 0.0),
                            quality.get("overall", 0.0),
                        ),
                    )
                    total_faces += 1

            self.db.execute(
                "UPDATE face_segments SET dense_status = 'completed' WHERE id = ?",
                (seg_id,),
            )
            self.db.commit()

        return total_faces

    def _count_dense_frames(self, task_id: int) -> int:
        self._ensure_db()
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM face_dense_frames WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return row["cnt"] if row else 0
