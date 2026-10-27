"""src/core/hybrid_analyzer.py — Hybrid model analysis pipeline

Implements a layered analysis strategy:
1. CLIP (fast, zero-shot scene/action classification)
2. YOLO (target detection for nudity/poses)
3. DeepFace (person attributes: gender, age, emotion)
4. VLM (fallback when confidence < threshold)

All tags are mapped via the dictionary table.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Import torch here so it's available at module level
import torch  # noqa: E402

# ── Enum: Silo category ──────────────────────────────────────────

class SiloCategory(str, Enum):
    ADULT = "adult"
    AI_GEN = "ai_gen"


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class CLIPResult:
    """CLIP classification result per dimension."""
    labels: Dict[str, float]  # {label_name: confidence}
    top_label: str = ""
    top_confidence: float = 0.0


@dataclass
class YOLOResult:
    """YOLO detection result."""
    detections: List[Dict[str, Any]]  # [{box, confidence, category}]
    top_category: str = ""
    top_confidence: float = 0.0


@dataclass
class DeepFaceResult:
    """DeepFace person attribute analysis result."""
    gender: Optional[str] = None
    age: Optional[int] = None
    emotion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class HybridAnalysisResult:
    """Combined hybrid analysis output."""
    clip: CLIPResult
    yolo: YOLOResult
    deepface: DeepFaceResult
    vlm_summary: str = ""  # Only if VLM was used as fallback
    vlm_used: bool = False
    processing_time: float = 0.0


# ── Label mappings ───────────────────────────────────────────────

# Adult silo labels for CLIP matching (normalized English)
ADULT_CLIP_LABELS = {
    "scene": [
        "bedroom", "living room", "bathroom", "hotel room", "office",
        "outdoor", "public place", "special location", "inside car"
    ],
    "position": [
        "oral sex", "deepthroat", "69 position", "doggy style",
        "missionary position", "cowgirl position", "double penetration",
        "foreplay", "solo masturbation", "facial cumshot", "internal creampie"
    ],
    "count": [
        "solo person", "two people", "three people", "group of people"
    ],
    "body": [
        "slim body", "muscular body", "chubby body", "bear type body",
        "mature man", "young man"
    ],
    "clothing": [
        "fully nude", "partially nude", "uniform", "with props"
    ],
    "visual_focus": [
        "close-up", "point of view", "full body shot"
    ]
}

# AI-Gen silo labels for CLIP matching
AI_GEN_CLIP_LABELS = {
    "generation_quality": [
        "flickering", "morphing", "physics error", "lost detail",
        "logical error", "temporal inconsistency"
    ],
    "visual_style": [
        "photorealistic", "anime style", "cinematic style", "surreal style"
    ],
    "cinematography": [
        "dynamic camera", "stable close-up", "wide shot", "point of view"
    ],
    "consistency": [
        "stable face", "outfit jump", "background drift"
    ]
}


# ── Hybrid Analyzer Core ─────────────────────────────────────────

class CLIPAnalyzer:
    """CLIP-based zero-shot classification."""

    def __init__(self, model_name: str = "ViT-B-16-SigLIP-256", precision: str = "fp16"):
        self.model_name = model_name
        self.precision = precision
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._is_initialized = False
        self._disabled = False

    def _init_model(self):
        """Lazy-load CLIP model.

        许可说明：默认使用 SigLIP（Apache 2.0，权重可商用）。
        SigLIP 的预训练名为 'webli'；其他模型（如 ViT-H-14）用 'openai'。
        模型加载失败时降级禁用（返回空结果），不影响其他分析层。
        """
        if self._is_initialized:
            return
        if self._disabled:
            return
        try:
            import open_clip
            # 权重预检：检查 HuggingFace 缓存是否已有模型权重。
            # 未缓存则禁用（不触发下载阻塞），提示用户手动下载。
            # SigLIP 权重来自 timm/ViT-B-16-SigLIP-256/（Apache 2.0）
            try:
                from huggingface_hub import try_to_load_from_cache
                _hf_repo = "timm/ViT-B-16-SigLIP-256" if "siglip" in self.model_name.lower() else None
                if _hf_repo:
                    cached = try_to_load_from_cache(_hf_repo, "open_clip_pytorch_model.bin")
                    if cached is None:
                        logger.warning(
                            "CLIP 权重未缓存（%s/open_clip_pytorch_model.bin），"
                            "场景分类已禁用。请网络可用时运行下载："
                            "hf download %s",
                            _hf_repo, _hf_repo,
                        )
                        self._disabled = True
                        return
            except ImportError:
                pass  # 无 huggingface_hub 时跳过预检，直接尝试加载

            # SigLIP 系列 → webli 预训练（Apache 2.0 可商用）
            pretrained = "webli" if "siglip" in self.model_name.lower() else "openai"
            self._model, _, self._processor = open_clip.create_model_and_transforms(
                self.model_name, pretrained=pretrained
            )
            # 用模型对应的 tokenizer（SigLIP 需要 HFTokenizer，通用 tokenize 不适配）
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            if hasattr(self._model, 'eval'):
                self._model.eval()
            self._is_initialized = True
            logger.info(f"CLIP model initialized: {self.model_name} (pretrained={pretrained})")
        except ImportError:
            logger.error("open-clip-torch not installed. Run: pip install open-clip-torch")
            self._disabled = True
        except Exception as e:
            logger.error(f"Failed to init CLIP model (降级禁用): {e}")
            self._disabled = True

    def classify(self, image_path: str, label_set: List[str]) -> Dict[str, float]:
        """
        Classify an image against a set of text labels using CLIP.

        Args:
            image_path: Path to the image file
            label_set: List of text labels to match against

        Returns:
            Dict mapping each label to its confidence score (0-1)
        """
        self._init_model()
        if self._disabled:
            return {label: 0.0 for label in label_set}

        # Encode image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Encode text labels
        text_tokens = self._tokenizer([f"a photo of {label}" for label in label_set])

        try:
            with torch.no_grad():
                image_features = self._model.encode_image(self._processor(image).unsqueeze(0))
                text_features = self._model.encode_text(text_tokens)

                # Compute cosine similarity
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = image_features @ text_features.T

                # SigLIP 用 logit_scale/logit_bias + sigmoid（输出维度 = num_labels）
                # 其他 CLIP 用 100×softmax
                if hasattr(self._model, "logit_scale") and hasattr(self._model, "logit_bias"):
                    logit_scale = self._model.logit_scale.exp()
                    logit_bias = self._model.logit_bias
                    logits = similarity * logit_scale + logit_bias
                    probs = torch.sigmoid(logits)
                    scores = probs.squeeze(0).cpu().numpy()
                else:
                    scores = (100.0 * similarity).softmax(dim=-1).squeeze(0).cpu().numpy()

            return {label: float(score) for label, score in zip(label_set, scores)}
        except Exception as e:
            logger.error(f"CLIP classification error: {e}")
            return {label: 0.0 for label in label_set}


class RFDETRAnalyzer:
    """RF-DETR-based object detection (Apache 2.0 license).

    Uses ONNX Runtime to load the RF-DETR-N model for fast,
    commercially-safe object detection.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._session = None

    def _init_model(self):
        if self._session is None:
            try:
                import onnxruntime as ort

                # Default model path if not specified
                if self.model_path is None:
                    from src.core.config import ConfigManager
                    cfg = ConfigManager()
                    self.model_path = cfg.get(
                        "models.detection.path",
                        "~/.video-tag/models/rf-detr-n.onnx",
                    )

                path = Path(self.model_path).expanduser()
                if not path.exists():
                    logger.warning(
                        "RF-DETR model not found at %s — detection disabled. "
                        "Download rf-detr-n.onnx and set models.detection.path in config.",
                        path,
                    )
                    self._session = None
                    return

                self._session = ort.InferenceSession(
                    str(path),
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self._output_names = [o.name for o in self._session.get_outputs()]
                logger.info("RF-DETR model initialized: %s", self.model_path)
            except ImportError:
                logger.error("onnxruntime not installed. Run: pip install onnxruntime")
                raise
            except Exception as e:
                logger.error("Failed to init RF-DETR model: %s", e)
                raise

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run RF-DETR detection on an image.

        Returns:
            List of detections: [{box, confidence, category}]
        """
        self._init_model()
        if self._session is None:
            return []

        try:
            import numpy as np
            from PIL import Image

            # Preprocess image: resize to 384×384, rescale /255（RF-DETR-N 规格）
            img = Image.open(image_path).convert("RGB")
            img = img.resize((384, 384), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            # RF-DETR expects [1, 3, H, W]
            img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]

            # Run inference
            outputs = self._session.run(
                self._output_names,
                {self._input_name: img_array},
            )

            # RF-DETR-N 输出: pred_boxes [1, 300, 4] + logits [1, 300, 91]
            boxes = outputs[0][0]  # [300, 4] 归一化 [cx, cy, w, h]
            logits = outputs[1][0]  # [300, 91]

            # 每个候选框取最高类别（跳过背景类 0）
            h, w = img.height, img.width
            detections = []
            for i in range(boxes.shape[0]):
                class_logits = logits[i][1:]  # 跳过背景类
                # softmax 归一化得到概率
                exp_logits = np.exp(class_logits - np.max(class_logits))
                probs = exp_logits / exp_logits.sum()
                class_id = int(np.argmax(probs))
                score = float(probs[class_id])
                if score < 0.3:
                    continue

                # RF-DETR 的 box 是 [cx, cy, w, h] 归一化 → [x1, y1, x2, y2] 像素
                cx, cy, bw, bh = boxes[i]
                x1 = float((cx - bw / 2) * w)
                y1 = float((cy - bh / 2) * h)
                x2 = float((cx + bw / 2) * w)
                y2 = float((cy + bh / 2) * h)

                category = self._coco_label(class_id + 1)  # +1 因为跳过了背景

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": score,
                    "category": category,
                })

            return detections
        except Exception as e:
            logger.error("RF-DETR detection error: %s", e)
            return []

    @staticmethod
    def _coco_label(class_id: int) -> str:
        """Map COCO class ID to human-readable label."""
        coco_labels = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light", 10: "fire hydrant",
            11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse",
            18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
            22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
            26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
            34: "baseball bat", 35: "baseball glove", 36: "skateboard",
            37: "surfboard", 38: "tennis racket", 39: "bottle",
            40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
            48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
            52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
            68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
            72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
            76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
        }
        return coco_labels.get(class_id, f"object_{class_id}")


class DeepFaceAnalyzer:
    """DeepFace-based person attribute analysis."""

    def __init__(self):
        self._analyzed = False

    def _init_model(self):
        if not self._analyzed:
            try:
                import deepface  # noqa: F401 — 可用性探测（真实导入验证依赖链）
                self._analyzed = True
                logger.info("DeepFace model initialized")
            except ImportError:
                logger.error("deepface not installed. Run: pip install deepface")
                raise

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze person attributes from an image.

        Returns:
            {gender, age, emotion, confidence}
        """
        self._init_model()
        try:
            # 懒加载（与 _init_model 同款）。此前缺失本行 → analyze 必 NameError，
            # 被 except 吞掉后整层静默返回空结果，DeepFace 人员属性从未真正生效。
            import deepface

            result = deepface.analyze(
                image_path,
                actions=["gender", "age", "emotion"],
                enforce_detection=False
            )
            # deepface returns list; take first person
            if result and isinstance(result, list):
                r = result[0]
                return {
                    "gender": r.get("dominant_gender"),
                    "age": r.get("age"),
                    "emotion": r.get("dominant_emotion"),
                    "confidence": r.get("emotion", {}).get(
                        r.get("dominant_emotion", ""), 0.0
                    ),
                }
            return {"gender": None, "age": None, "emotion": None, "confidence": 0.0}
        except Exception as e:
            logger.error(f"DeepFace analysis error: {e}")
            return {"gender": None, "age": None, "emotion": None, "confidence": 0.0}


# ── Main Hybrid Analyzer ─────────────────────────────────────────

class HybridAnalyzer:
    """Main entry point — orchestrates all analysis layers."""

    def __init__(
        self,
        clip_model: str = "ViT-B-16-SigLIP-256",
        detection_model_path: str = None,
        vllm_base_url: str = "http://localhost:11434",
        vllm_model: str = "miniCPM-V-4.5",
        confidence_threshold: float = 0.5,
    ):
        self.clip = CLIPAnalyzer(clip_model)
        self.detector = RFDETRAnalyzer(detection_model_path)
        self.deepface = DeepFaceAnalyzer()
        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.confidence_threshold = confidence_threshold
        self._vlm_client = None

    @property
    def vllm_client(self):
        if self._vlm_client is None:
            from src.core.vlm import MiniCPMVClient
            self._vlm_client = MiniCPMVClient(
                model=self.vllm_model,
                base_url=self.vllm_base_url,
            )
        return self._vlm_client

    def analyze_frames(
        self,
        frame_paths: List[str],
        silo: SiloCategory = SiloCategory.ADULT,
        use_clip: bool = True,
        use_yolo: bool = True,
        use_deepface: bool = True,
        use_vllm_fallback: bool = True,
    ) -> HybridAnalysisResult:
        """
        Run hybrid analysis on a list of frames.

        Strategy:
        1. CLIP classifies all frames against silo-specific label sets
        2. YOLO detects objects/targets in frames
        3. DeepFace extracts person attributes
        4. If highest confidence < threshold, VLLM is used as fallback

        Args:
            frame_paths: List of image file paths
            silo: Which label set to use
            use_clip: Enable CLIP classification
            use_yolo: Enable YOLO detection
            use_deepface: Enable DeepFace analysis
            use_vllm_fallback: Enable VLLM as fallback

        Returns:
            HybridAnalysisResult with combined outputs
        """
        if not frame_paths:
            logger.warning("No frames provided for analysis")
            empty_clip = CLIPResult(labels={}, top_label="", top_confidence=0.0)
            empty_yolo = YOLOResult(detections=[], top_category="", top_confidence=0.0)
            return HybridAnalysisResult(
                clip=empty_clip, yolo=empty_yolo, deepface=DeepFaceResult()
            )

        start_time = time.time()
        all_clip_scores = {}
        max_confidence = 0.0

        # ── Layer 1: CLIP ───────────────────────────────────────
        clip_labels = (
            ADULT_CLIP_LABELS if silo == SiloCategory.ADULT else AI_GEN_CLIP_LABELS
        )

        if use_clip:
            for dim, labels in clip_labels.items():
                dim_scores = {}
                for frame_path in frame_paths:
                    scores = self.clip.classify(frame_path, labels)
                    for label, score in scores.items():
                        dim_scores[label] = max(dim_scores.get(label, 0.0), score)

                # Normalize to 0-1
                max_score = max(dim_scores.values()) if dim_scores else 1.0
                if max_score > 0:
                    dim_scores = {k: v / max_score for k, v in dim_scores.items()}

                all_clip_scores[dim] = dim_scores
                top_label = max(dim_scores, key=dim_scores.get)
                top_conf = dim_scores[top_label]
                max_confidence = max(max_confidence, top_conf)

                logger.info(f"CLIP [{dim}]: top={top_label} ({top_conf:.3f})")
        else:
            all_clip_scores = {}

        # ── Layer 2: RF-DETR detection ───────────────────────────
        all_detections = []
        max_det_conf = 0.0

        if use_yolo:
            for frame_path in frame_paths:
                detections = self.detector.detect(frame_path)
                all_detections.extend(detections)
                for d in detections:
                    max_det_conf = max(max_det_conf, d["confidence"])

            max_confidence = max(max_confidence, max_det_conf)

            # Aggregate by category
            category_counts = {}
            for d in all_detections:
                cat = d["category"]
                category_counts[cat] = category_counts.get(cat, 0) + d["confidence"]

            top_cat = max(category_counts, key=category_counts.get) if category_counts else ""
            logger.info(f"YOLO: detected {len(all_detections)} objects, top={top_cat}")
        else:
            all_detections = []

        # ── Layer 3: DeepFace ───────────────────────────────────
        deepface_result = DeepFaceResult()
        if use_deepface:
            # Use first frame for person analysis
            deepface_result = self.deepface.analyze(frame_paths[0])
            max_confidence = max(max_confidence, deepface_result.confidence)
            if deepface_result.gender:
                logger.info(f"DeepFace: gender={deepface_result.gender}, "
                           f"age={deepface_result.age}, emotion={deepface_result.emotion}")

        # ── Layer 4: VLLM Fallback ──────────────────────────────
        vlm_summary = ""
        vlm_used = False

        if use_vllm_fallback and max_confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({max_confidence:.3f} < {self.confidence_threshold}), "
                       "triggering VLLM fallback")
            try:
                vlm_summary = self.vllm_client.analyze_unit(frame_paths)
                vlm_used = True
                # Update max confidence based on VLM output (we assume it's reliable)
                max_confidence = 0.9
            except Exception as e:
                logger.error(f"VLLM fallback failed: {e}")
        else:
            logger.info(f"High confidence ({max_confidence:.3f} >= {self.confidence_threshold}), "
                       "skipping VLLM")

        elapsed = time.time() - start_time

        clip_result = CLIPResult(
            labels={dim: labels for dim, labels in all_clip_scores.items()},
        )
        yolo_result = YOLOResult(
            detections=all_detections[:10],  # Top 10 detections
        )

        return HybridAnalysisResult(
            clip=clip_result,
            yolo=yolo_result,
            deepface=deepface_result,
            vlm_summary=vlm_summary,
            vlm_used=vlm_used,
            processing_time=elapsed,
        )

    def analyze_video(
        self,
        video_path: str,
        silo: SiloCategory = SiloCategory.ADULT,
        frame_count: int = 15,
    ) -> HybridAnalysisResult:
        """
        Full video analysis pipeline.

        1. Scene detect
        2. Sample frames
        3. Run hybrid analysis
        """
        from src.core.frame_sampler import sample_frames
        from src.core.scenedetect import detect_scenes

        # Scene detect
        units = detect_scenes(video_path)
        if not units:
            # Fallback: single unit
            units = [{"unit_index": 0, "start_frame": 0, "end_frame": 1000}]

        # Sample frames from all units
        all_frame_paths = []
        frames_per_unit = max(1, frame_count // max(1, len(units)))
        for unit in units:
            frame_paths = sample_frames(
                video_path,
                unit["start_frame"],
                unit["end_frame"],
                n_frames=frames_per_unit,
                output_dir=None,  # Return raw frames without saving
            )
            all_frame_paths.extend(frame_paths)

        logger.info(f"Sampled {len(all_frame_paths)} total frames for analysis")

        return self.analyze_frames(all_frame_paths, silo=silo)


    def analyze_frames_to_dict(
        self,
        frame_paths: List[str],
        silo: SiloCategory = SiloCategory.ADULT,
        use_clip: bool = True,
        use_yolo: bool = True,
        use_deepface: bool = True,
        use_vllm_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Run hybrid analysis and return result as a JSON-serializable dict."""
        result = self.analyze_frames(
            frame_paths, silo=silo,
            use_clip=use_clip, use_yolo=use_yolo,
            use_deepface=use_deepface, use_vllm_fallback=use_vllm_fallback,
        )
        return self._result_to_dict(result)

    def _result_to_dict(self, result: HybridAnalysisResult) -> Dict[str, Any]:
        """Convert HybridAnalysisResult to JSON-serializable dict."""
        clip_labels = {}
        for dim, scores in result.clip.labels.items():
            if isinstance(scores, dict) and scores:
                top_k = max(scores.keys(), key=lambda k: scores[k])
                clip_labels[dim] = {
                    "scores": scores,
                    "top_label": top_k,
                    "top_confidence": float(scores[top_k]),
                }
            elif isinstance(scores, (int, float)):
                clip_labels[dim] = {
                    "scores": {"score": scores},
                    "top_label": "score",
                    "top_confidence": float(scores),
                }

        yolo_detections = []
        for d in result.yolo.detections:
            yolo_detections.append({
                "box": d.get("box"),
                "confidence": d.get("confidence", 0.0),
                "category": d.get("category", ""),
            })

        return {
            "clip_labels": clip_labels,
            "yolo_detections": yolo_detections,
            "yolo_top_category": result.yolo.top_category,
            "deepface": {
                "gender": result.deepface.gender,
                "age": result.deepface.age,
                "emotion": result.deepface.emotion,
                "confidence": result.deepface.confidence,
            },
            "vlm_summary": result.vlm_summary if result.vlm_used else None,
            "vlm_used": result.vlm_used,
            "processing_time": result.processing_time,
        }
