"""Face recognition module using InsightFace/RetinaFace.

Extracts face embeddings from video frames, clusters them into actor identities,
and persists face images + embeddings for Jellyfin collection generation.
"""

import cv2
import numpy as np
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional

from src.core.embedding_codec import decode_embedding, encode_embedding
from src.core.face_thresholds import (
    DEFAULT_DISTANCE_THRESHOLD, DEFAULT_MERGE_THRESHOLD, DEFAULT_MIN_CLUSTER_SIZE,
)
from src.core.face_clustering import FaceClusterer
from src.core.face_utils import estimate_pose

logger = logging.getLogger(__name__)


def _maybe_commit(db_session, task_id=None, what: str = "") -> None:
    """中途提交当前事务以**缩短长写事务**（见 openspec/changes/fix-sqlite-write-lock，C）。

    识别/重建原本"整段一个事务"，写锁从首次写一直持到收尾 —— 长事务跨在写闸外，
    写闸帮不上，其它写者只能等或超时。按 actor/簇分批提交后，写锁只在写单个
    actor 时短暂持有。

    兼容外部 SQLAlchemy Session 与 raw sqlite3.Connection（两者都有 commit/rollback）。
    失败只记日志、不中断：已提交的部分保留，与"信息只增不减"一致。
    """
    if db_session is None:
        return
    try:
        db_session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("中途提交失败 task=%s %s: %s", task_id, what, exc)
        try:
            db_session.rollback()
        except Exception:
            pass

# 共享推理模型锁：多 worker 并发复用同一 FaceRecognizer/insightface app 时，
# det_size 动态切换(prepare)与推理(get)存在竞态(一个线程改 det_size 另一个
# 按旧尺寸推理)。用模块级 RLock 保护 prepare+get 序列——prepare 仅在 det_size
# 变化时短暂持锁，同 det_size 的并发推理不受影响(onnxruntime Session.run 线程安全)。
import threading as _threading
_APP_LOCK = _threading.RLock()

# Graceful degradation: if InsightFace is not installed, mark unavailable
try:
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available — face recognition disabled")


class FaceRecognizer:
    """Face recognizer: extract embeddings, save face crops, match actors."""

    # Frontal face thresholds (degrees)
    # 只关注roll角（头部歪斜程度），不限制yaw/pitch（摄像头俯拍/仰拍是正常的）
    YAW_THRESHOLD = 90.0  # 不限制左右转头（摄像头角度）
    PITCH_THRESHOLD = 90.0  # 不限制上下抬头（摄像头俯拍/仰拍）
    ROLL_THRESHOLD = 30.0  # 限制头部歪斜程度（人脸应该基本竖直）

    # Cosine similarity threshold for matching
    SIMILARITY_THRESHOLD = 0.75

    # 是否只保存正脸（严格模式）
    FRONTAL_ONLY = False

    # 清晰度阈值：低于此值的人脸不用于封面
    # 降低到 0.01，保留更多人脸（高清视频中人脸可能较小，导致清晰度分数偏低）
    MIN_CLARITY_FOR_COVER = 0.01

    def __init__(self, model_name: str = "antelopev2", detector: Optional[str] = None) -> None:
        """Initialize face recognition model.

        Args:
            model_name: InsightFace 模型名（仅作兼容保留；实际始终加载 AuraFace）
                       AuraFace: SCRFD-10GF + ResNet100@Glint360K, Apache-2.0 可商用
            detector: 保留参数，已废弃。InsightFace 内置检测器。
        """
        self.available = _INSIGHTFACE_AVAILABLE
        self.app = None
        self._gate_app = None  # 非人脸闸门专用检测器（小 det_size，惰性初始化）
        # CoreML 模式标志: 为 True 时 det_size 固定 640——
        # scrfd 检测在 CoreML 下仅 640 稳定, 切 960/1280(1080p/4K 帧)会静态
        # shape 崩溃导致整帧漏检(2026-09 排查 task1537: 1080p 视频 0 人脸根因)。
        self._coreml_active = False
        self.use_retinaface = False  # 保留字段，始终为 False（InsightFace 内置 RetinaFace）
        self.model_name = model_name
        self.detector_type = "insightface"  # 固定为 insightface

        # 从配置读取采样帧数
        from src.core.config import ConfigManager
        config = ConfigManager()
        self.MAX_SAMPLE_FRAMES = config.get("video.frames_per_scene", 5)
        
        # 从配置读取模型名称（如果未指定）
        if model_name == "antelopev2":
            model_name = config.get("face_recognition.model", "antelopev2")
            self.model_name = model_name

        # 从配置读取 AuraFace 模型路径
        model_path = config.get("face_recognition.model_path", "models/auraface")
        model_path = str(Path(model_path).expanduser().resolve())

        # 复用 ModelManager 已加载的共享识别模型（只读查询，不触发加载，
        # 避免 wrapper.load() → FaceRecognizer() → get() 的递归初始化）。
        if self.available:
            try:
                from src.core.model_manager import ModelManager
                mm = ModelManager.get_instance()
                wrapper = mm.peek("face_recognition")
                if wrapper is not None and wrapper.is_available():
                    inner = getattr(wrapper, "_recognizer", None)
                    if inner is not None and getattr(inner, "app", None) is not None:
                        self.app = inner.app
                        self.model_name = inner.model_name
                        # 共享模型的 CoreML 标志从 inner 继承(det_size 固定策略一致)
                        self._coreml_active = bool(getattr(inner, "_coreml_active", False))
                        logger.info(
                            "FaceRecognizer reusing shared AuraFace model from ModelManager (%s)",
                            self.model_name,
                        )
                        self.available = True
            except Exception as e:
                logger.warning("ModelManager lookup failed, falling back to direct init: %s", e)

        if self.available and self.app is not None:
            # Shared model obtained — skip the direct FaceAnalysis init below.
            pass
        elif self.available:
            # 模型根目录：打包环境用 ~/.insightface，开发环境用当前目录
            from src.core.face_utils import (
                get_insightface_model_root, ensure_insightface_models_copied,
                get_face_providers,
            )
            model_root = get_insightface_model_root()
            # 实际加载的模型目录名是 auraface(spec 内置 models/auraface)。
            # 这里显式复制内置模型(完整版); 精简版无内置源时该调用为 no-op。
            ensure_insightface_models_copied("auraface")

            # ── 模型缺失守卫(关键) ──────────────────────────────────
            # FaceAnalysis 在 root/models/<name> 不存在时会调用 insightface 的
            # download() 自动从 GitHub 拉 <name>.zip——AuraFace 在官方 release
            # 中不存在, 每次启动都会卡 30-60s 下载
            # 失败并遗留 .zip 残片(2026-09 VM 实测: 精简版未下载模型时反复失败)。
            # 这里先检查关键 onnx 是否就位, 缺失则跳过加载 → available=False,
            # 由上层提示走「设置 → 环境初始化」下载, 绝不触发 insightface 自动下载。
            def _model_dir(name_: str):
                return Path(model_root) / "models" / name_

            # 识别必需文件: AuraFace(检测 + 识别 + 性别年龄)
            _load_model_name = "auraface"  # 模型目录名(FaceAnalysis 的 name)
            _required_onnx = ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx"]
            _missing = [
                f for f in _required_onnx
                if not (_model_dir(_load_model_name) / f).exists()
            ]
            if _missing:
                logger.warning(
                    "人脸模型 %s 缺失(%s), 跳过初始化——请先在「设置 → 环境初始化」"
                    "下载模型(轻量版首次运行需要)。缺失文件: %s",
                    _load_model_name, _model_dir(_load_model_name), _missing,
                )
                self.available = False
                self.app = None
                # 不再回退尝试(回退同样会触发自动下载)
                return

            # 完整管线(det+recog+genderage)优先 CoreML(M4 实测 8x)，失败自动降级 CPU
            _providers_attempts = [get_face_providers(full_pipeline=True), ["CPUExecutionProvider"]]
            for _providers in _providers_attempts:
                try:
                    logger.info(
                        "Initializing InsightFace with model '%s' (root=%s) providers=%s...",
                        model_name, model_root, _providers,
                    )
                    self.app = FaceAnalysis(
                        name=_load_model_name,
                        root=model_root,
                        providers=_providers,
                        allowed_modules=["detection", "recognition", "genderage"],
                    )
                    # 降低检测阈值，提高检测灵敏度
                    logger.info("Calling app.prepare()...")
                    self.app.prepare(
                        ctx_id=0, det_size=(640, 640),
                        det_thresh=type(self)._recognition_det_thresh(),
                    )
                    logger.info("InsightFace model loaded successfully (AuraFace)")
                    logger.info("InsightFace app.det_size = %s", self.app.det_size)
                    logger.info("InsightFace app.det_thresh = %s", self.app.det_thresh)
                    self._coreml_active = any(
                        "CoreML" in p for p in (_providers or [])
                    )
                    break
                except Exception as exc:
                    logger.error(
                        "Failed to initialize InsightFace with providers=%s: %s",
                        _providers, exc,
                    )
                    logger.error("Exception type: %s", type(exc).__name__)
                    self.app = None
            if self.app is None:
                # 不再回退到 buffalo_l（非商用权重，已全量 AuraFace 化）
                self.available = False
                self.app = None

        # Face image storage directory（唯一解析入口，见 src/core/face_paths.py）
        from src.core.config import ConfigManager
        from src.core.face_paths import resolve_faces_dir

        config = ConfigManager()
        self.faces_dir = resolve_faces_dir(get_config=config.get)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Face storage directory: %s", self.faces_dir)

    # ------------------------------------------------------------------
    # Face quality filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _get_face_quality_config() -> dict:
        """获取人脸质量过滤配置。

        Returns:
            配置字典，包含 enabled, sharpness_threshold, min_faces_to_keep 等
        """
        from src.core.config import ConfigManager
        config = ConfigManager()
        return config.get("face_quality", {
            "enabled": True,
            "sharpness_threshold": 300,
            "min_faces_to_keep": 3,
            "min_sharpness_absolute": 50,  # sharpness 绝对值下限（上限，自适应不会超过它）
            "min_sharpness_absolute_low": 8,  # 兜底下限：低于此值视为垃圾帧
            "adaptive_enabled": True,  # 按本视频人脸分布自适应该下限
            "adaptive_percentile": 30,  # 自适应参考分位
            "adaptive_min_faces": 5,  # 人脸太少时不做自适应
            # 以下三项原为"被读取但不在默认字典内"的隐式默认（见 D6/§6.4），
            # 补入此处使单一来源成立；取值与调用点原有默认一致。
            "min_quality": 0.3,
            "cover_eyes_open_threshold": 0.5,
            "low_res_min_eyes_open": 0.9,
            # embedding 范数自适应（§3）：下限 = min(配置上限, max(绝对下限, 本片该分位))
            # ⚠️ 分位必须 ≥ p25：阈值 15.0 ≈ 库内 p12，取 p5/p10 会把所有片源都放宽
            "norm_absolute_low": 8.0,       # 绝对下限（库内范数 p5≈12.8，留余量）
            "norm_adaptive_percentile": 25,  # 参考分位（库内 p25≈17.2 > 15 → 清晰片源不变）
            "norm_low_quality_ratio": 0.7,   # 低质量/低清时上限按此比例放宽
        })

    @staticmethod
    def _adaptive_sharpness_floor(fqc: dict, faces: list, video_quality_info=None) -> float:
        """清晰度下限：以本视频自身人脸分布的低分位为基准，仅放宽、不收紧。

        实现已迁至 :func:`src.core.face_thresholds.build_profile`（阈值唯一来源），
        本方法保留为兼容入口，行为不变。

        固定绝对阈值对内容与裁剪尺寸敏感——Laplacian 方差随人脸裁剪的像素尺寸和
        纹理变化，实测已入库 32,991 张人脸 sharpness 跨度 9.3~8539（p5=25.5 /
        p50=115.4）。同一个 50 在总体只淘汰约 11%，但在"画面清楚、脸却小而软"的片源上
        会淘汰 85%，让整片聚不出人物、被误判成无人脸（2026-09-17 「一只牛蛙（下）」：
        1280x590、预检 0.80，52 张脸中位 sharpness 仅 26.3，只剩 8 张 → 0 人物）。

        因此下限取"本视频已检出人脸 sharpness 的低分位"，仅用于放宽这个绝对阈值，
        并以 min_sharpness_absolute_low 兜底防止垃圾帧被整体保留：
            floor = min(min_sharpness_absolute, max(min_sharpness_absolute_low, p(ref)))
        清晰片源上 p(ref) 高于绝对阈值 → 结果仍是 min_sharpness_absolute，行为不变。

        Args:
            fqc: face_quality 配置字典
            faces: 当前过滤阶段的人脸全集（需含 sharpness 字段）
            video_quality_info: VideoQualityInfo（可选），沿用原有的低质量兜底下调

        Returns:
            本次过滤使用的 sharpness 下限
        """
        from src.core.face_thresholds import build_profile

        return build_profile(fqc, faces, video_quality_info).sharpness_floor

    @staticmethod
    def _uniform_sampling_target(duration_seconds: float, interval_seconds: float,
                                 min_frames: int = 60, max_frames: int = 400) -> int:
        """按"目标采样间隔"推导均匀抽帧的帧数。

        为什么按时长推导、而不是固定帧数：**聚类要求同一个人的相邻样本落在"人脸外观
        变化尺度"之内**。实测（见 docs/架构/人脸识别-验证样本登记.md 的 M4 判定）：

        ==============  ==========  ==========  ============
        相邻样本间隔    同人距离    跨人距离    能否聚类
        ==============  ==========  ==========  ============
        约 2 秒        0.307       0.855       可分（阈值 0.6–0.7）
        7 秒（原实现）  0.807       0.855       **不可分**
        ==============  ==========  ==========  ============

        故间隔上限需远小于原实现的 8 秒；**具体取值由样本标定**，不在此硬编码
        （配置键 ``face_recognition.sampling_target_interval_seconds``，默认 8.0 即改动前行为）。

        Args:
            duration_seconds: 视频时长（秒）
            interval_seconds: 目标采样间隔（秒）
            min_frames: 帧数下限（短视频不因间隔大而少于该值）
            max_frames: 帧数上限（成本护栏；长视频会因此达不到目标间隔）

        Returns:
            目标帧数
        """
        if duration_seconds <= 0:
            return min_frames          # 与改动前一致（dur=0 → 下限）
        if interval_seconds <= 0:
            return max_frames          # 配置异常：宁可多抽，不静默降级为稀疏采样
        return int(max(min_frames, min(max_frames, duration_seconds / interval_seconds)))

    @staticmethod
    def _stage_det_thresh() -> float:
        """正脸/侧脸阶段使用的检测置信度门槛。

        历史问题：这两处调用 ``extract_faces_from_frames`` 时**未传** ``det_thresh``，
        落回签名默认值 **0.5**；而预筛用的是 **0.15** —— 两者口径不一致，导致
        "预筛说有正脸、正式检测却检出 0~1 张"（见 docs/架构/人脸识别-验证样本登记.md 的 M2，
        样本 1/3b/4/6/7/10 中招）。

        此值应与预筛口径一致；**具体取值由样本标定**，不在此硬编码
        （配置键 ``face_recognition.stage_det_thresh``，默认 0.5 = 改动前行为）。
        """
        from src.core.config import ConfigManager
        return float(ConfigManager().get("face_recognition.stage_det_thresh", 0.5))

    @staticmethod
    def _recognition_det_thresh() -> float:
        """识别器 ``self.app`` 的检测置信度门槛（单一来源，取代硬编码 0.15）。

        该值是 ``extract_faces_from_frames`` 未显式 ``prepare`` 时的生效值，
        也是 ``_extract_embeddings_from_detection`` 直接 ``app.get()`` 时**实际使用**的值——
        因后者不调用 ``prepare``，会沿用上一次 ``prepare`` 留下的阈值（"粘住"）。
        默认 0.15 = 改动前行为；配置键 ``face_recognition.det_thresh``。
        """
        from src.core.config import ConfigManager
        return float(ConfigManager().get("face_recognition.det_thresh", 0.15))

    @staticmethod
    def _gate_det_thresh() -> float:
        """非人脸闸门专用检测器的置信度门槛（默认 0.15 = 改动前行为）。"""
        from src.core.config import ConfigManager
        return float(ConfigManager().get("face_recognition.gate_det_thresh", 0.15))

    @staticmethod
    def _track_grouping_config() -> dict:
        """轨迹分组配置（见 ``face_clustering.group_faces_into_tracks``）。

        **默认关闭**（配置缺失或 enabled=false）→ 行为与改动前完全一致。
        """
        from src.core.config import ConfigManager
        cfg = ConfigManager().get("face_recognition.track_grouping", {}) or {}
        return dict(cfg)

    @staticmethod
    def _needs_prepare(current_size, target_size, current_thresh, target_thresh) -> bool:
        """是否需要重新 ``prepare()``：**检测尺寸或检测阈值任一变化**。

        历史漏洞：此前只在尺寸变化时才 prepare，于是 ``det_thresh`` 参数"传了却不生效"，
        同一进程内先前用过的阈值会粘住（见 M2）。
        同尺寸同阈值时返回 False，因此不会带来逐帧开销。
        """
        if current_size != target_size:
            return True
        if current_thresh is None:
            return True
        try:
            return abs(float(current_thresh) - float(target_thresh)) > 1e-6
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _filter_by_sharpness(
        faces: list,
        threshold: float = 300,
        min_keep: int = 3,
        min_sharpness_absolute: float = 50,
    ) -> list:
        """按清晰度过滤人脸。

        策略：
        1. 先过滤掉 sharpness < min_sharpness_absolute 的人脸（绝对值下限）
        2. 保留清晰度 >= threshold 的人脸
        3. 如果保留后少于 min_keep，则按清晰度排序保留最好的 min_keep 张
        4. 确保至少保留 1 张（避免无人脸可用）

        Args:
            faces: 人脸列表，每个包含 sharpness 字段
            threshold: 清晰度阈值（Laplacian方差）
            min_keep: 最少保留数量
            min_sharpness_absolute: sharpness 绝对值下限，低于此值直接丢弃

        Returns:
            过滤后的人脸列表
        """
        if not faces:
            return faces

        # 第一步：过滤掉 sharpness 绝对值过低的人脸（可能是误检）
        valid_faces = [f for f in faces if f.get("sharpness", 0) >= min_sharpness_absolute]
        if len(valid_faces) < len(faces):
            removed = len(faces) - len(valid_faces)
            logger.info(
                f"sharpness 绝对值过滤: {removed}/{len(faces)} 张人脸被丢弃 "
                f"(min_sharpness={min_sharpness_absolute:.0f})"
            )
            for f in faces:
                if f not in valid_faces:
                    try:
                        Path(f["face_path"]).unlink()
                    except Exception as e:
                        logger.warning(f"删除极低清晰度人脸失败: {f.get('face_path')}, {e}")

        if not valid_faces:
            return []

        # 第二步：按原有逻辑过滤
        good_faces = [f for f in valid_faces if f.get("sharpness", 0) >= threshold]

        if len(good_faces) >= min_keep:
            # 清晰人脸足够，直接返回
            removed = len(faces) - len(good_faces)
            if removed > 0:
                logger.info(
                    f"清晰度过滤: {removed}/{len(faces)} 张人脸被过滤 "
                    f"(阈值={threshold:.0f})"
                )
                # 删除被过滤的人脸文件
                for f in faces:
                    if f not in good_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception as e:
                            logger.warning(f"删除模糊人脸失败: {f.get('face_path')}, {e}")
            return good_faces

        # 清晰人脸不足，保留最好的 min_keep 张
        sorted_faces = sorted(faces, key=lambda f: f.get("sharpness", 0), reverse=True)
        kept = sorted_faces[:max(min_keep, 1)]
        removed = len(faces) - len(kept)
        logger.warning(
            f"清晰度过滤（容错模式）: {removed}/{len(faces)} 张被过滤，"
            f"仅 {len(kept)} 张清晰人脸 (阈值={threshold:.0f})"
        )
        # 删除被过滤的人脸文件
        for f in faces:
            if f not in kept:
                try:
                    Path(f["face_path"]).unlink()
                except Exception as e:
                    logger.warning(f"删除模糊人脸失败: {f.get('face_path')}, {e}")
        return kept

    # ------------------------------------------------------------------
    # Face extraction
    # ------------------------------------------------------------------

    def _get_optimal_det_size(self, img_width: int, img_height: int) -> tuple:
        """根据图像分辨率动态调整检测尺寸，优化内存使用和检测精度
        
        策略：
        - 4K (3840x2160) 及以上：使用 1280x1280，保证大分辨率下的人脸检测精度
        - 1080p (1920x1080)：使用 960x960，平衡精度和内存
        - 720p (1280x720) 及以下：使用 640x640，节省内存
        
        Args:
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            (width, height) 检测尺寸元组
        """
        max_dim = max(img_width, img_height)
        
        if max_dim >= 3840:  # 4K 及以上
            return (1280, 1280)
        elif max_dim >= 1920:  # 1080p
            return (960, 960)
        else:  # 720p 及以下
            return (640, 640)

    def extract_faces_from_frames(self, frames_path: str, det_thresh: float = 0.5, clarity_threshold: float = 0.01, frame_filter: List[str] = None, temp_dir: Optional[Path] = None) -> List[Dict]:
        """Extract faces from a frame directory.

        Uniformly samples up to MAX_SAMPLE_FRAMES frames, detects faces in each,
        and returns per-face metadata (embedding, pose, crop path).
        When FRONTAL_ONLY is enabled, filters to keep only near-frontal faces.

        Args:
            frames_path: Path to directory containing frame images (*.jpg).
            det_thresh: Detection confidence threshold (default 0.15).
            clarity_threshold: Minimum clarity score for faces to be kept (default 0.01).
            frame_filter: 可选，只处理这些帧路径（两阶段流水线：正脸/侧脸分批分析）。
            temp_dir: 可选裁剪临时目录。传入时人脸裁剪图写到该目录（用于 rebuild 等
                需要隔离 temp 命名空间的路径，避免多个任务/人物共享 faces_dir 根目录的
                temp_{frame}_{x}_{y}.jpg 撞名覆写）；不传保持原行为（写 self.faces_dir）。

        Returns:
            List of dicts with keys:
                frame_path, face_path, embedding (np.ndarray), yaw, pitch, roll
        """
        logger.info(f"extract_faces_from_frames called: frames_path={frames_path}")
        if not self.available:
            logger.info("Face recognition not available, returning empty list")
            return []

        frames_dir = Path(frames_path)
        if not frames_dir.exists():
            logger.warning("Frames directory does not exist: %s", frames_path)
            return []

        # 裁剪临时目录：隔离模式写入调用方指定目录，否则写 faces_dir 根（历史行为）
        crop_dir = temp_dir if temp_dir is not None else self.faces_dir
        if temp_dir is not None:
            try:
                crop_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning("extract_faces_from_frames: 创建 temp_dir 失败，回退 faces_dir: %s", e)
                crop_dir = self.faces_dir
        if frame_filter is not None:
            frame_filter_set = {str(Path(f)) for f in frame_filter}
            frame_files = sorted(
                f for f in frames_dir.rglob("*.jpg") if str(f) in frame_filter_set
            )
        else:
            frame_files = sorted(frames_dir.rglob("*.jpg"))
        logger.info("Searching for .jpg files in %s, found %d files", frames_dir, len(frame_files))
        if not frame_files:
            logger.warning("No .jpg files found in: %s", frames_path)
            return []

        # Process all frames (sampling is already done during frame extraction)
        import gc
        
        all_faces: List[Dict] = []
        for frame_idx, frame_path in enumerate(frame_files):
            logger.info(f"Processing frame {frame_idx}/{len(frame_files)}: {frame_path.name}")
            
            # 尝试用 cv2 读取，捕获损坏文件的异常
            img = None
            try:
                img = cv2.imread(str(frame_path))
            except Exception as cv2_err:
                logger.warning(f"cv2.imread exception for {frame_path.name}: {cv2_err}")
            
            if img is None:
                logger.warning(f"cv2.imread failed for {frame_path.name}, trying PIL fallback")
                # 尝试用 PIL 读取（避免 cv2.cvtColor 在打包环境中崩溃）
                try:
                    from PIL import Image
                    try:
                        import pillow_heif
                        pillow_heif.register_heif_opener()  # 支持 HEIC 照片
                    except ImportError:
                        pass
                    pil_img = Image.open(str(frame_path))
                    # 强制加载图片数据（触发可能的解压错误）
                    pil_img.load()
                    # 转换为 RGB 模式（如果是 RGBA 或其他模式）
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    # 用 numpy 进行 RGB -> BGR 转换，避免调用 cv2.cvtColor
                    img_array = np.array(pil_img)
                    img = img_array[:, :, ::-1].copy()  # RGB -> BGR
                    logger.info(f"PIL fallback success: shape={img.shape}, dtype={img.dtype}")
                except Exception as pil_err:
                    logger.warning("Both cv2 and PIL failed to read %s: %s", frame_path, pil_err)
                    continue
            else:
                logger.info(f"cv2.imread success: shape={img.shape}, dtype={img.dtype}")

            # 保存图像尺寸用于后续裁剪
            img_height, img_width = img.shape[:2]
            
            # 根据图像分辨率动态调整检测尺寸
            if getattr(self, '_coreml_active', False):
                # CoreML 模式固定 640: scrfd 检测在 CoreML 下仅 det_size=640 稳定,
                # 切 960/1280(1080p/4K 帧) 会 CoreML 静态 shape 崩溃 → 整帧漏检
                # (task1537 1080p 视频 0 人脸根因, 2026-09)。640 检测 1080p 人脸
                # 足够(实测 40/40 帧稳定检出), 损失极小。
                optimal_det_size = (640, 640)
            else:
                optimal_det_size = self._get_optimal_det_size(img_width, img_height)

            # prepare(det_size 切换)加锁串行化；推理 get 不加锁——实测同一 CoreML
            # session 4 线程并发 run 安全且近 2x 吞吐(35→73fps)，加锁反而杀并发。
            # 竞态窗口仅 prepare 瞬间(每视频首次/分辨率档切换, ~ms 级), 后果只是
            # 极少数帧按切换前尺寸检测，可接受。
            # 需要 prepare 的条件：检测尺寸变化，**或检测阈值与当前生效值不同**。
            # 后者补的是一个历史漏洞：det_thresh 此前只在 det_size 变化时才被应用，
            # 于是"传了阈值却不生效"，同一进程内先前用过的阈值会粘住（见 M2）。
            # 同尺寸同阈值时不重复 prepare，故无逐帧开销。
            _need_prepare = bool(self.app) and self._needs_prepare(
                self.app.det_size, optimal_det_size,
                getattr(self.app, "det_thresh", None), det_thresh,
            )
            if _need_prepare:
                with _APP_LOCK:
                    self.app.prepare(ctx_id=0, det_size=optimal_det_size, det_thresh=det_thresh)

            # 使用 InsightFace 检测（内置 RetinaFace）
            try:
                detected = self.app.get(img) if self.app else []
                logger.info(f"Frame {frame_idx}: {len(detected)} faces detected (det_size={optimal_det_size})")
            except Exception as e:
                logger.warning(f"InsightFace detection failed for frame {frame_idx}: {e}")
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                logger.warning(f"img type={type(img)}, shape={getattr(img, 'shape', 'NO SHAPE')}, dtype={getattr(img, 'dtype', 'NO DTYPE')}")
                detected = []

            if not detected:
                # 立即释放图像内存
                del img
                # 每处理 5 帧强制垃圾回收
                if frame_idx % 5 == 0:
                    gc.collect()
                continue

            for face in detected:
                embedding = face.embedding
                det_score = getattr(face, 'det_score', 1.0)
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                yaw, pitch, roll = self._extract_pose(face)

                if embedding is None:
                    continue

                # 检查检测置信度（支持动态调整）
                if det_score < det_thresh:
                    logger.debug(f"Skipping low confidence face: {det_score:.2f}")
                    continue

                # Crop with 20% margin
                w, h = x2 - x1, y2 - y1
                
                # 过滤太小的人脸（可能是误检）
                if w < 50 or h < 50:
                    logger.debug(f"Skipping small face: {w}x{h}")
                    continue
                
                margin_x, margin_y = int(w * 0.2), int(h * 0.2)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(img_width, x2 + margin_x)
                y2 = min(img_height, y2 + margin_y)

                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    del face_img
                    continue

                # 裁剪文件名：frame + bbox 左上角 + 短随机后缀。
                # 随机后缀避免不同视频/任务帧名相同（frame_000000.jpg 等）时
                # 在共享 faces_dir 根撞名覆写（历史 bug：temp 被后处理视频覆写，
                # rebuild 时旧 temp 指向错误内容 → 目录与 DB 不一致/缩水）。
                _rand = os.urandom(2).hex()
                temp_name = f"temp_{frame_path.stem}_{x1}_{y1}_{_rand}.jpg"
                temp_path = crop_dir / temp_name
                try:
                    cv2.imwrite(str(temp_path), face_img)
                except Exception as _imwrite_exc:
                    logger.warning("裁剪人脸写入失败 %s: %s", temp_path, _imwrite_exc)
                    del face_img
                    continue
                
                # 立即释放裁剪后的图像
                del face_img

                # 解析帧号（帧文件名 frame_N.jpg 或 supp_N.jpg，N 为视频内帧号）
                # 时间戳由帧号反推：extract_faces_from_frames 只有帧目录无视频上下文，
                # 无法拿 fps，故先留空；时序先验用 frame_id 的邻近关系即可
                frame_id = None
                try:
                    _stem = frame_path.stem
                    _m = re.match(r"^(?:frame|supp)_(\d+)", _stem)
                    if _m:
                        frame_id = int(_m.group(1))
                except Exception:
                    pass

                all_faces.append(
                    {
                        "frame_path": str(frame_path),
                        "face_path": str(temp_path),
                        "embedding": embedding,
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "roll": float(roll),
                        "det_score": float(det_score),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "frame_id": frame_id,
                        "timestamp": None,
                        "sex": getattr(face, "sex", None),
                        "age": getattr(face, "age", None),
                        "kps": np.asarray(face.kps, dtype=float).tolist() if hasattr(face, 'kps') and face.kps is not None else None,
                    }
                )
                
                # 显式释放临时变量
                del bbox

        # 为每个人脸计算质量分数
        for face in all_faces:
            # 先计算亮度（用于质量分数计算）
            face['brightness'] = self._calculate_brightness(face['face_path'])
            # 计算清晰度（Laplacian方差原始值）
            face['sharpness'] = self._calculate_raw_sharpness(face['face_path'])
            # 计算综合质量分数
            face['quality_score'] = self.calculate_face_quality(face)

        # ── face-quality-filter: 仅做绝对清晰度过滤（过滤明显误检） ──
        # 注意：sharpness_threshold（相对阈值）用于选封面，不在聚类前过滤
        fqc = self._get_face_quality_config()
        if fqc.get("enabled", False):
            from src.core.face_thresholds import build_profile, log_profile

            profile = build_profile(fqc, all_faces)
            log_profile(profile, where="extract_faces", frames_dir=Path(frames_path).name)
            min_sharp = profile.sharpness_floor
            # 只做绝对值过滤，不做相对阈值过滤（保留所有有效人脸用于聚类）
            if min_sharp > 0:
                before_count = len(all_faces)
                all_faces = [f for f in all_faces if f.get("sharpness", 0) >= min_sharp]
                removed = before_count - len(all_faces)
                if removed > 0:
                    logger.info(
                        "sharpness绝对值过滤: %d/%d 张人脸被丢弃 (min_sharpness=%.0f)",
                        removed, before_count, min_sharp,
                    )
                    for f in all_faces:
                        if f.get("sharpness", 0) < min_sharp:
                            try:
                                Path(f["face_path"]).unlink()
                            except Exception:
                                pass

        # 正脸过滤：只保留接近正脸的人脸
        if self.FRONTAL_ONLY and all_faces:
            # 筛选正脸
            frontal_faces = [
                f for f in all_faces
                if (abs(f["yaw"]) < self.YAW_THRESHOLD
                    and abs(f["pitch"]) < self.PITCH_THRESHOLD
                    and abs(f["roll"]) < self.ROLL_THRESHOLD)
            ]

            if frontal_faces:
                logger.info("Frontal filter: %d/%d faces passed", len(frontal_faces), len(all_faces))
                # 删除非正脸的图片文件
                for f in all_faces:
                    if f not in frontal_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception as e:
                            logger.warning("Failed to delete non-frontal face: %s", e)
                return frontal_faces
            else:
                # 没有正脸时，选择最接近正脸的几张
                logger.warning("No frontal faces found, using %d closest", min(3, len(all_faces)))
                sorted_faces = sorted(all_faces, key=lambda f: abs(f["yaw"]) + abs(f["pitch"]) + abs(f["roll"]))
                best_faces = sorted_faces[:min(3, len(sorted_faces))]
                for f in all_faces:
                    if f not in best_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception as e:
                            logger.warning("Failed to delete non-best face: %s", e)
                return best_faces

        return all_faces

    # ------------------------------------------------------------------
    # Frontal face selection
    # ------------------------------------------------------------------

    def select_best_frontal_face(self, faces: List[Dict]) -> Optional[Dict]:
        """Select the best frontal face from a list.

        Criterion: |yaw| < 15deg, |pitch| < 15deg, |roll| < 15deg + normal brightness.
        Falls back to the face with the smallest total angle if none qualify.

        Args:
            faces: List of face dicts (each with yaw/pitch/roll).

        Returns:
            The best face dict, or None if the list is empty.
        """
        if not faces:
            return None

        frontal = [
            f
            for f in faces
            if (
                abs(f["yaw"]) < self.YAW_THRESHOLD
                and abs(f["pitch"]) < self.PITCH_THRESHOLD
                and abs(f["roll"]) < self.ROLL_THRESHOLD
            )
        ]

        candidates = frontal if frontal else faces

        # Calculate brightness for candidates
        for face in candidates:
            brightness = self._calculate_brightness(face.get("face_path", ""))
            face["brightness"] = brightness

        # Prefer normal brightness (60-200)
        normal_brightness = [f for f in candidates if 60 <= f.get("brightness", 0) <= 200]
        if normal_brightness:
            return min(
                normal_brightness,
                key=lambda f: abs(f["yaw"]) + abs(f["pitch"]) + abs(f["roll"]),
            )

        # Fallback: closest to normal brightness
        return min(
            candidates,
            key=lambda f: abs(f.get("brightness", 0) - 130),
        )

    # ------------------------------------------------------------------
    # Similarity & matching
    # ------------------------------------------------------------------

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in [0, 1].
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def match_actor(self, embedding: np.ndarray, threshold: float = 0.65, db_session=None) -> Optional[int]:
        """Find the best matching actor for an embedding.

        Args:
            embedding: Face embedding to match.
            threshold: Minimum cosine similarity to consider a match.
            db_session: Optional shared database session.

        Returns:
            actor_id of the best match, or None if no match found.
        """
        from src.db.models import Actor, get_session
        from src.db.models import get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            import sqlite3
            if isinstance(db_session, sqlite3.Connection):
                engine = get_db_engine()
                db = get_session(engine)
                own_session = True
            else:
                db = db_session
        
        try:
            actors = db.query(Actor).all()

            best_match: Optional[int] = None
            best_score = 0.0

            for actor in actors:
                if not actor.face_embedding:
                    continue

                existing_emb = decode_embedding(actor.face_embedding)
                score = self.compute_similarity(embedding, existing_emb)

                if score >= threshold and score > best_score:
                    best_match = actor.id
                    best_score = score

            return best_match
        finally:
            if own_session:
                db.close()

    def match_actor_v2(self, cluster_faces: List[Dict], threshold: float = 0.55, db_session=None,
                       scope_actor_ids: Optional[List[int]] = None) -> Optional[int]:
        """Per-face max-similarity 匹配。

        对 cluster 中每张脸，与 actor 的所有 per-face embeddings 计算相似度，
        取 cluster-to-actor 的平均最大相似度作为匹配分数。
        比 avg-vs-avg 更鲁棒，对噪声和部分遮挡更容忍。

        Args:
            cluster_faces: 当前聚类中所有人脸的 dict 列表（需包含 embedding 字段）
            threshold: 最低匹配分数
            db_session: 可选数据库会话
            scope_actor_ids: 可选，仅在这些 actor 中匹配（如当前任务刚建立的 actor）。
                为 None 时匹配全部 actor。

        Returns:
            最佳匹配的 actor_id，或 None
        """
        from src.db.models import Actor, ActorFace, get_session
        from src.db.models import get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            import sqlite3
            if isinstance(db_session, sqlite3.Connection):
                engine = get_db_engine()
                db = get_session(engine)
                own_session = True
            else:
                db = db_session

        try:
            # 收集 cluster 的 embeddings
            cluster_embs = [np.array(f["embedding"]) for f in cluster_faces if f.get("embedding") is not None]
            if not cluster_embs:
                return None

            # 加载 actor 的 per-face embeddings；scope_actor_ids 限定候选范围
            actor_q = db.query(Actor)
            if scope_actor_ids:
                actor_q = actor_q.filter(Actor.id.in_(scope_actor_ids))
            actors = actor_q.all()
            best_match: Optional[int] = None
            best_score = 0.0

            for actor in actors:
                if not actor.face_embedding:
                    continue

                # 加载该 actor 所有人脸的 embeddings
                actor_faces = db.query(ActorFace).filter(
                    ActorFace.actor_id == actor.id,
                    ActorFace.embedding.isnot(None),
                ).all()

                actor_embs = []
                for af in actor_faces:
                    try:
                        actor_embs.append(decode_embedding(af.embedding))
                    except (json.JSONDecodeError, TypeError):
                        continue

                # 如果没有 per-face embeddings，回退到 actor 平均 embedding
                if not actor_embs:
                    try:
                        actor_embs = [decode_embedding(actor.face_embedding)]
                    except (json.JSONDecodeError, TypeError):
                        continue

                # 计算 cluster-to-actor 的匹配分数
                # 对 cluster 中每张脸，找与 actor 所有脸的最大相似度
                face_max_sims = []
                for c_emb in cluster_embs:
                    max_sim = 0.0
                    for a_emb in actor_embs:
                        sim = self.compute_similarity(c_emb, a_emb)
                        if sim > max_sim:
                            max_sim = sim
                    face_max_sims.append(max_sim)

                # 用 top-3 平均替代全平均：跨视频场景簇内常混入低质量脸
                # （低画质/运动模糊），全平均会被拉低导致漏匹配。
                # top-3 只要求"有足够多的脸高度匹配"即可判定同一人
                # （实测：Harry Louis 碎片 ACTOR_020 mean 0.51 漏匹配，
                # 但 top3=0.70 可正确匹配；而纯质量差的簇 top3 仍低，不会误并）
                face_max_sims = np.asarray(face_max_sims, dtype=float)
                if len(face_max_sims) >= 3:
                    score = float(np.sort(face_max_sims)[-3:].mean())
                else:
                    score = float(face_max_sims.mean())

                if score >= threshold and score > best_score:
                    best_match = actor.id
                    best_score = score

            return best_match
        finally:
            if own_session:
                db.close()

    # 编号生成说明(2026-09-11 起):
    # code 一律由 id 派生 —— `ACTOR_{id:03d}`, 见 create_actor。
    # 曾用"找未占用空洞/取最大 code+1"的独立编号策略, 会与自增 id 脱节
    # (实测 336/722 错位, 如 id=341 却叫 ACTOR_004), 且并发时撞
    # UNIQUE(actors.code)。改为从属 id 后, 两者永久一致且并发天然无冲突。

    # ------------------------------------------------------------------
    # Actor CRUD
    # ------------------------------------------------------------------

    def _calculate_brightness(self, image_path: str) -> float:
        """计算图片的平均亮度（0-255）

        Args:
            image_path: 图片路径

        Returns:
            平均亮度值，0表示全黑，255表示全白
        """
        import cv2
        try:
            img = cv2.imread(image_path)
        except Exception as e:
            logger.warning(f"cv2.imread failed for brightness calculation: {image_path}, error: {e}")
            return 0
        if img is None:
            return 0

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算平均亮度
        brightness = gray.mean()
        return float(brightness)

    def _calculate_raw_sharpness(self, image_path: str) -> float:
        """计算图像清晰度原始值（Laplacian方差，不归一化）。

        用于 face-quality-filter 功能，提供绝对清晰度参考值。
        参考阈值：1080p 视频 ~300，720p ~150，4K ~500。

        Args:
            image_path: 图片路径

        Returns:
            Laplacian方差原始值，越大越清晰
        """
        import cv2
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            logger.warning(f"cv2.imread failed for sharpness calculation: {image_path}, error: {e}")
            return 0.0
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    def _calculate_eyes_open_score(self, image_path: str) -> float:
        """评估眼睛是否睁开（0-1）。

        原理：睁眼时眼睛区域有更多纹理（瞳孔、虹膜、睫毛），
        Laplacian 方差高于下半脸；闭眼时眼睛区域更平滑。

        Args:
            image_path: 人脸裁剪图片路径

        Returns:
            0（闭眼）到 1（睁眼）之间的分数
        """
        import cv2
        try:
            img = cv2.imread(image_path)
        except Exception as e:
            logger.warning(f"cv2.imread failed for eyes_open calculation: {image_path}, error: {e}")
            return 0.0
        if img is None:
            return 0.0

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return 0.0

        # 上半部分（眼睛区域）
        eye_region = img[0:h // 2, :, :]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_var = cv2.Laplacian(gray_eye, cv2.CV_64F).var()

        # 下半部分（嘴部区域，作为参照）
        lower_region = img[h // 2:, :, :]
        gray_lower = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
        lower_var = cv2.Laplacian(gray_lower, cv2.CV_64F).var()

        # 比值 > 1 说明眼睛区域纹理比嘴部多（睁眼），< 1 说明眼睛更平滑（闭眼）
        if lower_var < 1:
            return 0.5  # 无法判断，给中间值

        ratio = eye_var / lower_var

        # 使用 sigmoid 映射：ratio=1 → 0.5, ratio=2 → ~0.88, ratio=0.5 → ~0.12
        score = 1.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))
        return float(max(0.0, min(1.0, score)))

    def calculate_face_quality(self, face_dict: Dict) -> float:
        """计算人脸质量分数（0-1之间）

        公式：
        - 姿态分数（40%权重）：1 - (|yaw| + |pitch| + |roll|) / 90
        - 清晰度分数（25%权重）：Laplacian方差归一化到0-1
        - 亮度分数（15%权重）：1 - |brightness - 130| / 130
        - 睁眼分数（20%权重）：眼睛区域纹理评估

        Args:
            face_dict: 人脸信息字典，包含 yaw, pitch, roll, face_path, brightness 等字段

        Returns:
            float: 0-1之间的质量分数
        """
        # 1. 姿态分数
        yaw = abs(face_dict.get('yaw', 0))
        pitch = abs(face_dict.get('pitch', 0))
        roll = abs(face_dict.get('roll', 0))
        pose_score = max(0, 1 - (yaw + pitch + roll) / 90)

        # 2. 清晰度分数（Laplacian方差）
        face_path = face_dict.get('face_path', '')
        clarity_score = self._calculate_clarity(face_path)

        # 3. 亮度分数
        brightness = face_dict.get('brightness', 130)
        brightness_score = max(0, 1 - abs(brightness - 130) / 130)

        # 4. 睁眼分数
        eyes_open_score = self._calculate_eyes_open_score(face_path)

        # 综合分数
        quality = (
            pose_score * 0.40
            + clarity_score * 0.25
            + brightness_score * 0.15
            + eyes_open_score * 0.20
        )
        face_dict['eyes_open_score'] = eyes_open_score
        return round(quality, 4)

    def _calculate_clarity(self, image_path: str) -> float:
        """计算图像清晰度（Laplacian方差）

        Args:
            image_path: 图片路径

        Returns:
            0-1之间的归一化分数
        """
        import cv2
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            logger.warning(f"cv2.imread failed for clarity calculation: {image_path}, error: {e}")
            return 0.0
        if img is None:
            return 0.0

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        # 归一化：假设方差范围0-1000，映射到0-1
        clarity = min(laplacian_var / 1000, 1.0)
        return clarity

    def _select_best_cover(self, faces: List[Dict]) -> Optional[Dict]:
        """选择最佳封面图片

        优先选择：
        1. 睁眼优先（eyes_open_score > 0.5）
        2. 正脸优先（使用姿态评分公式）
        3. 检测置信度高（det_score > 0.9）
        4. 清晰度高（sharpness >= threshold）
        5. 亮度正常（60-200之间，避免过暗或过曝）

        Args:
            faces: 人脸列表

        Returns:
            最佳封面人脸字典，如果没有合适的则返回 None
        """
        if not faces:
            return None

        # 获取清晰度阈值（face-quality-filter 配置）
        fqc = self._get_face_quality_config()
        sharpness_threshold = fqc.get("sharpness_threshold", 300)

        # 第一步：计算每个候选的姿态评分 + 睁眼评分
        # 封面要求竖直，所以只考虑roll角（歪斜程度），忽略yaw和pitch（摄像头角度）
        for face in faces:
            # 只考虑roll角（竖直程度）
            roll_angle = abs(face.get("roll", 0))
            
            # 姿态评分：roll角越小越好
            # roll < 15° 得满分，roll > 45° 得0分
            face["_pose_score"] = max(0, 100 - roll_angle * 2)
            face["_roll_angle"] = roll_angle
            
            # 计算睁眼评分（如果还没有）
            if "eyes_open_score" not in face:
                face["eyes_open_score"] = self._calculate_eyes_open_score(face.get("face_path", ""))

        # 排除极端姿态：yaw/pitch 极端可能是误检
        # 但 roll 极端可能是横屏拍摄导致的，不排除
        non_extreme_faces = [
            f for f in faces
            if abs(f.get("yaw", 0)) <= 70
            and abs(f.get("pitch", 0)) <= 70
        ]
        if non_extreme_faces:
            faces = non_extreme_faces
        else:
            # 所有人脸都有极端 yaw/pitch，使用全部人脸
            logger.warning("Cover selection: all faces have extreme yaw/pitch, using all faces")

        # 睁眼过滤：优先选择睁眼的人脸（eyes_open_score > 0.5）
        open_eyes_faces = [f for f in faces if f.get("eyes_open_score", 0) > 0.5]
        if open_eyes_faces:
            logger.info(
                f"Cover selection: {len(open_eyes_faces)}/{len(faces)} faces with open eyes"
            )
            faces = open_eyes_faces
        else:
            logger.info("Cover selection: no open-eye faces found, using all faces")

        # 第二步：筛选高置信度的正脸（严格标准），且清晰度达标
        # 姿态评分 > 70 (意味着总角度 < 30度，接近正脸)
        high_conf_frontal = [
            f for f in faces
            if (f.get("det_score", 0) > 0.9
                and f.get("_pose_score", 0) > 70
                and f.get('sharpness', 0) >= sharpness_threshold * 0.5)
        ]

        # 计算综合评分：姿态分数(主导) + 睁眼分数 + 检测分数(辅助) + 清晰度奖励
        def _compute_cover_score(f):
            f["_cover_score"] = (
                f.get("_pose_score", 0) +
                f.get("eyes_open_score", 0) * 30 +
                f.get("det_score", 0) * 20 +
                f.get("sharpness", 0) / 100
            )
            return f["_cover_score"]

        if high_conf_frontal:
            # 如果有高质量正脸，按综合评分排序后选择最佳
            for face in high_conf_frontal:
                _compute_cover_score(face)
            high_conf_frontal.sort(key=lambda f: f.get("_cover_score", 0), reverse=True)
            candidates = high_conf_frontal
        else:
            # 没有符合正脸标准的人脸，选择所有清晰度达标的人脸
            all_valid = [
                f for f in faces
                if f.get('sharpness', 0) >= sharpness_threshold * 0.5
            ]

            if not all_valid:
                logger.warning(
                    "No candidates meet sharpness threshold (%.0f) for cover selection",
                    sharpness_threshold * 0.5
                )
                # 容错：退回使用所有人脸
                all_valid = faces

            # 使用姿态评分作为主要排序依据（正脸优先）
            for face in all_valid:
                _compute_cover_score(face)

            # 按综合评分从高到低排序
            all_valid.sort(key=lambda f: f.get("_cover_score", 0), reverse=True)

            # 取前 10 个作为候选
            candidates = all_valid[:10]

        # 计算每个候选的亮度
        for face in candidates:
            face_path = face.get("face_path", "")
            brightness = self._calculate_brightness(face_path)
            face["brightness"] = brightness

        # 优先选择亮度正常的（60-200）
        normal_brightness = [f for f in candidates if 60 <= f.get("brightness", 0) <= 200]

        if normal_brightness:
            # 在亮度正常的候选中，按综合评分排序选择最佳
            normal_brightness.sort(key=lambda f: f.get("_cover_score", 0), reverse=True)
            best_face = normal_brightness[0]
        else:
            # 如果没有亮度正常的，选择亮度最接近正常范围的
            best_face = min(
                candidates,
                key=lambda f: abs(f.get("brightness", 0) - 130)
            )

        logger.info(
            f"Selected cover (Eyes + Frontal Priority): {best_face.get('face_path')}, "
            f"pose_score={best_face.get('_pose_score', 0):.1f}, "
            f"eyes_open={best_face.get('eyes_open_score', 0):.3f}, "
            f"yaw={best_face.get('yaw', 0):.1f}, pitch={best_face.get('pitch', 0):.1f}, "
            f"roll={best_face.get('roll', 0):.1f}"
        )

        return best_face

    def _get_gate_app(self):
        """非人脸闸门专用检测器：小 det_size，适配人脸裁剪小图。

        裁剪图通常只有 60-130px，用主检测器的 640x640 会因过度放大而
        全部漏检（实测 70px 小图在 640 下 0 检出，在 320 下 det_score≈0.7
        正常检出）。因此闸门固定用 320x320。
        """
        if self._gate_app is None:
            from insightface.app import FaceAnalysis
            from src.core.face_utils import (
                get_insightface_model_root, ensure_insightface_models_copied,
                get_face_providers,
            )
            model_root = get_insightface_model_root()
            ensure_insightface_models_copied("auraface")
            # 模型缺失守卫: 不触发 insightface 自动下载(下载源不存在)
            _gdir = Path(model_root) / "models" / "auraface"
            _gneed = ["scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx"]
            if any(not (_gdir / f).exists() for f in _gneed):
                logger.warning("闸门检测器跳过: auraface 模型缺失(%s)", _gdir)
                return None
            # 纯检测 320 小图：scrfd 在 CoreML 下静态 shape 崩溃 → 强制 CPU
            self._gate_app = FaceAnalysis(
                name="auraface",
                root=model_root,
                providers=get_face_providers(full_pipeline=False),
                allowed_modules=["detection"],
            )
            self._gate_app.prepare(
                ctx_id=0, det_size=(320, 320),
                det_thresh=type(self)._gate_det_thresh(),
            )
        return self._gate_app

    def _prescreen_frames(self, frames_dir) -> Dict[str, List[str]]:
        """帧预筛：用 320×320 轻量检测器给帧分类（正脸/侧脸/无人脸）。

        两阶段流水线的第一阶段——先快速识别正脸帧，让正脸数据先交付，
        侧脸帧后续补分析（渐进式呈现：观感上处理速度不同）。

        Args:
            frames_dir: 帧目录

        Returns:
            {
                "frontal": [帧路径...],   # 含正脸（yaw/pitch < 25°，可作封面）
                "side": [帧路径...],      # 含人脸但姿态偏侧（识别辅助）
                "empty": [帧路径...],     # 无人脸（跳过完整分析，省算力）
            }
            **三类互斥**：每帧只归入一类（frontal 优先 > side > empty）。
            历史 bug（BUG-023）：此前"同一帧可能同时在 frontal 和 side"，导致正脸/侧脸
            两个阶段各写一次同一张脸 → 全库 55% 人脸记录冗余。因为正脸阶段对帧做的是
            帧级检测（会检出该帧所有脸，含侧脸），把混帧归入 frontal 不会漏脸。
        """
        import cv2 as _cv2p

        gate = self._get_gate_app()
        if gate is None:
            # 模型缺失时闸门不可用: 全部按"无人脸"处理(上层会因 available=False 跳过)
            logger.warning("闸门检测器不可用，帧预筛全部置 empty")
            return {"frontal": [], "side": [], "empty": sorted(Path(frames_dir).glob("*.jpg"))}
        result = {"frontal": [], "side": [], "empty": []}
        frames = sorted(Path(frames_dir).glob("*.jpg"))
        if not frames:
            return result

        t0 = time.time()
        n_frontal_frames = 0
        n_side_frames = 0
        n_empty_frames = 0
        for fp in frames:
            img = _cv2p.imread(str(fp))
            if img is None:
                continue
            try:
                detected = gate.get(img)
            except Exception:
                detected = []
            del img

            if not detected:
                result["empty"].append(str(fp))
                n_empty_frames += 1
                continue

            has_frontal = False
            has_side = False
            for d in detected:
                kps = getattr(d, "kps", None)
                if kps is None:
                    has_side = True
                    continue
                try:
                    yaw, pitch, roll = estimate_pose(kps, getattr(d, "bbox", None))
                except Exception:
                    has_side = True
                    continue
                if abs(yaw) < 25 and abs(pitch) < 25:
                    has_frontal = True
                else:
                    has_side = True

            if has_frontal:
                result["frontal"].append(str(fp))
                n_frontal_frames += 1
            elif has_side:
                result["side"].append(str(fp))
                n_side_frames += 1
            else:
                result["empty"].append(str(fp))
                n_empty_frames += 1

        logger.info(
            "帧预筛: %d 帧 -> 正脸 %d, 侧脸 %d, 无人脸 %d (%.1fs)",
            len(frames), n_frontal_frames, n_side_frames, n_empty_frames,
            time.time() - t0,
        )
        return result

    def _resolve_entity_id(self, db, task_id) -> Optional[int]:
        """按 task_id 解析文件实体 id。

        actor_faces.entity_id 是"按文件去重/统计作品数"的依据, 必须落库正确。
        历史 bug: create_actor/update_actor 只读 face["entity_id"], 而该键仅在
        rebuild_actor_faces 里被设置 → 74% 的 face 记录 entity_id 为空, 卡片
        作品数系统性少算。这里统一从 tasks 表兜底解析。
        """
        if not task_id:
            return None
        try:
            from src.db.models import Task
            row = db.query(Task.entity_id).filter(Task.id == task_id).first()
            return row[0] if row else None
        except Exception:
            return None

    @staticmethod
    def _dedup_faces(faces: List[Dict]) -> List[Dict]:
        """按 (帧文件名, bbox) 去重，避免同一张脸被重复入库（BUG-023）。

        历史 bug：帧预筛的"正脸/侧脸"分类不互斥，同一帧被正脸阶段与侧脸兜底阶段
        各处理一次，同一张脸写入两条完全相同的 actor_faces 记录
        （全库约 55% 冗余、82% 演员受影响）。
        同键多条时保留 quality_score（缺失则 det_score）最高的一条。

        Args:
            faces: 人脸 dict 列表（需含 frame_path 与 bbox）

        Returns:
            去重后的人脸列表（按首次出现顺序）
        """
        if not faces:
            return faces
        seen = {}
        for face in faces:
            frame_name = Path(face.get("frame_path") or "").name
            key = (frame_name, tuple(face.get("bbox") or ()))
            prev = seen.get(key)
            if prev is None:
                seen[key] = face
                continue
            cur = float(face.get("quality_score") or face.get("det_score") or 0.0)
            prv = float(prev.get("quality_score") or prev.get("det_score") or 0.0)
            if cur > prv:
                seen[key] = face
        deduped = list(seen.values())
        if len(deduped) < len(faces):
            logger.info(
                "人脸去重(BUG-023): %d -> %d 张（移除 %d 张重复）",
                len(faces), len(deduped), len(faces) - len(deduped),
            )
        return deduped

    def create_actor(self, faces: List[Dict], task_id: int, file_md5: str = None, db_session=None, video_median_sharpness: float = None) -> int:
        """Create a new actor with face samples.

        Computes the average embedding from all faces (including various poses like
        lying down) and stores individual face records for later comparison.
        Cover image prioritizes frontal faces with normal brightness.

        Args:
            faces: List of face dicts with keys: face_path, embedding, yaw, pitch, roll.
            task_id: The task that produced these faces.
            file_md5: The MD5 hash of the video file (for tracking source).
            db_session: Optional shared database session.
            video_median_sharpness: 该视频全部人脸的 sharpness 中位数，用于自适应可用性闸门。

        Returns:
            The new actor's ID, or None if creation failed.
        """
        from src.db.models import Actor, ActorFace, get_session
        from src.db.models import get_db_engine
        from sqlalchemy.exc import IntegrityError

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            # 兼容 sqlite3.Connection：转换为 SQLAlchemy session
            import sqlite3
            if isinstance(db_session, sqlite3.Connection):
                engine = get_db_engine()
                db = get_session(engine)
                own_session = True
            else:
                db = db_session

        # 异常清理用: 提前占位, 避免失败发生在建号前时 except 里未绑定
        code = None
        actor_dir = None

        # BUG-023: 去除同一张脸的重复提取（预筛分类重叠等历史原因）
        faces = self._dedup_faces(faces)

        # 文件实体 id: 优先用 face 自带, 否则按 task_id 解析(保证落库, 供按文件统计)
        _fallback_entity_id = self._resolve_entity_id(db, task_id)

        try:
            # 验证所有人脸文件存在且有效
            valid_faces = []
            for face in faces:
                face_path = Path(face["face_path"])
                if not face_path.exists():
                    logger.warning(f"Face file not found: {face_path}")
                    continue
                if face_path.stat().st_size == 0:
                    logger.warning(f"Face file is empty: {face_path}")
                    continue
                valid_faces.append(face)
            
            if not valid_faces:
                logger.error("No valid faces to create actor")
                if own_session:
                    db.close()
                return None

            faces = valid_faces

            # 非人脸 actor 闸门：对 actor 的样本裁剪图用 320x320 检测器重检，
            # 均匀抽样（最多 sample_max 张），通过率低于 min_pass_rate 判为
            # 非人脸 actor（后脑勺/身体局部/纯模糊），跳过创建。
            # 实测区分度（group8 视频 round8 导出）：垃圾簇 0%-5%，
            # 真脸簇最低 35%，20% 阈值余量充足。
            from src.core.face_utils import validate_face_landmarks

            _gate_enabled = True
            _min_pass_rate = 0.2
            _sample_max = 20
            try:
                from src.core.config import ConfigManager
                _cfg = ConfigManager()
                _gate_enabled = bool(_cfg.get("face_recognition.nonface_gate.enabled", True))
                _min_pass_rate = float(_cfg.get("face_recognition.nonface_gate.min_pass_rate", 0.2))
                _sample_max = int(_cfg.get("face_recognition.nonface_gate.sample_max", 20))
            except Exception:
                pass

            if _gate_enabled and faces:
                try:
                    gate_app = self._get_gate_app()
                except Exception as gate_exc:
                    logger.warning("非人脸闸门检测器初始化失败，本次跳过闸门: %s", gate_exc)
                    gate_app = None
                if gate_app is not None:
                    n = len(faces)
                    if n == 1:
                        idxs = [0]
                    else:
                        idxs = np.linspace(0, n - 1, min(_sample_max, n), dtype=int)
                    _geo_checked = 0
                    _geo_pass = 0
                    _sharp_vals = []
                    for _i in idxs:
                        try:
                            _img = cv2.imread(faces[int(_i)]["face_path"])
                        except Exception as e:
                            logger.warning(f"cv2.imread failed for nonface gate: {faces[int(_i)]['face_path']}, error: {e}")
                            continue
                        if _img is None:
                            continue
                        _geo_checked += 1
                        # 可用性质量：Laplacian 方差（清晰度），供整簇模糊判定
                        _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
                        _sharp_vals.append(float(cv2.Laplacian(_gray, cv2.CV_64F).var()))
                        try:
                            _det = gate_app.get(_img)
                        except Exception:
                            _det = []
                        if any(
                            validate_face_landmarks(
                                getattr(_d, "kps", None),
                                getattr(_d, "bbox", None),
                                det_score=float(getattr(_d, "det_score", 0.5)),
                            )
                            for _d in _det
                        ):
                            _geo_pass += 1
                    if _geo_checked > 0 and (_geo_pass / _geo_checked) < _min_pass_rate:
                        logger.info(
                            "非人脸 actor（重检通过率 %d/%d=%.0f%% < %.0f%%），跳过创建",
                            _geo_pass, _geo_checked,
                            100.0 * _geo_pass / _geo_checked,
                            100.0 * _min_pass_rate,
                        )
                        if own_session:
                            db.close()
                        return None
                    # 可用性质量下限：整簇样本中位清晰度过低 → 即使是真脸也无使用价值，
                    # 跳过创建（人工审核结论：模糊脸宁缺勿滥，如 round12 的 ACTOR_213）
                    # 自适应：软画质视频（全片中位清晰度低）按比例放宽阈值，
                    # 避免误杀真脸（task1555: 全片中位~13，固定30把167张脸的真人簇拒了）
                    _min_sharp = 30.0
                    try:
                        _min_sharp = float(_cfg.get("face_recognition.usability_gate.min_median_sharpness", 30.0))
                    except Exception:
                        pass
                    if video_median_sharpness and video_median_sharpness > 0:
                        _min_sharp = min(_min_sharp, 0.5 * video_median_sharpness)
                    if _sharp_vals and _min_sharp > 0:
                        _med_sharp = float(np.median(_sharp_vals))
                        if _med_sharp < _min_sharp:
                            logger.info(
                                "不可用 actor（中位清晰度 %.1f < %.1f，整簇过模糊），跳过创建",
                                _med_sharp, _min_sharp,
                            )
                            if own_session:
                                db.close()
                            return None

            # 静态图片检测：计算聚类内姿态方差
            # 如果所有脸的 yaw/pitch/roll 标准差极低，说明是同一张静态图片被重复检测
            # （如视频中的手机屏幕、电视、海报等），而非真实人脸
            if len(faces) >= 3:
                _yaws = [abs(f.get("yaw", 0)) for f in faces]
                _pitches = [abs(f.get("pitch", 0)) for f in faces]
                _rolls = [abs(f.get("roll", 0)) for f in faces]
                _yaw_std = float(np.std(_yaws))
                _pitch_std = float(np.std(_pitches))
                _roll_std = float(np.std(_rolls))
                # 阈值：角度标准差 < 0.5° 说明几乎完全一致 → 静态图片
                _static_threshold = 0.5
                if _yaw_std < _static_threshold and _pitch_std < _static_threshold and _roll_std < _static_threshold:
                    logger.info(
                        "静态图片 actor（姿态方差过低 yaw=%.1f pitch=%.1f roll=%.1f std），跳过创建",
                        _yaw_std, _pitch_std, _roll_std,
                    )
                    if own_session:
                        db.close()
                    return None

            # 计算清晰度并过滤低清晰度人脸
            for face in faces:
                clarity_score = self._calculate_clarity(face['face_path'])
                face['clarity'] = clarity_score

            clear_faces = [f for f in faces if f.get('clarity', 0) >= self.MIN_CLARITY_FOR_COVER]

            logger.info(
                "Clarity filter: %d/%d faces passed (threshold=%.2f)",
                len(clear_faces), len(faces), self.MIN_CLARITY_FOR_COVER
            )

            if len(clear_faces) < len(faces):
                filtered_out = [f for f in faces if f.get('clarity', 0) < self.MIN_CLARITY_FOR_COVER]
                for f in filtered_out:
                    logger.info(
                        "Filtered low-clarity face: %s (clarity=%.3f)",
                        f['face_path'], f.get('clarity', 0)
                    )

            if not clear_faces:
                logger.warning(
                    "All %d faces are below clarity threshold (%.2f), skipping actor creation",
                    len(faces), self.MIN_CLARITY_FOR_COVER
                )
                if own_session:
                    db.close()
                return None

            faces = clear_faces

            # ── Top-N 代表脸精选 ──
            # 业务模型: 每人物保留少量高质量代表脸(默认 5, config max_samples_per_actor),
            # 不需要把簇内全部脸入库。精选后 avg_embedding 用全部 faces(保留簇信息),
            # 但只移动/登记精选出的代表脸。
            _max_n = self._max_samples_per_actor()
            if len(faces) > _max_n:
                faces = self._select_representative_faces(faces, _max_n)

            # Compute average embedding from ALL faces (including lying/side poses)
            embeddings = [np.array(f["embedding"]) for f in faces]
            avg_embedding = np.mean(embeddings, axis=0)

            # Cover image: prioritize frontal faces with normal brightness
            best_face = self._select_best_cover(faces)

            # ── 编号由 id 派生(单一事实来源) ──
            # 先落库拿到自增 id, 再让 code = ACTOR_{id:03d}。
            # 这样"编号"与"记录号"永久一致(不再有空洞复用导致的脱节),
            # 且并发插入时各拿不同 id → code 天然不同, UNIQUE(actors.code)
            # 冲突从根上消失。
            import uuid as _uuid
            actor = Actor(
                code=f"__pending__{_uuid.uuid4().hex}",
                face_embedding=encode_embedding(avg_embedding),
                sample_count=len(faces),
                # cover_face 由 _apply_actor_cover 统一维护（见登记后）
                cover_face=None,
            )
            db.add(actor)
            try:
                db.flush()
            except IntegrityError as _ie:
                # 极少数情况(placeholder 撞名/并发), 重试一次
                logger.warning("Actor 建号冲突, 重置后重试: %s", _ie)
                db.rollback()
                db.add(actor)
                db.flush()
            actor.code = f"ACTOR_{actor.id:03d}"
            db.flush()

            code = actor.code

            # 创建 Actor 专属目录
            actor_dir = self.faces_dir / code
            # 若目录已存在（历史 actor 删除后目录残留导致的 code 重用），
            # 先整体清空，避免旧残留文件混入新 actor（目录必须与 DB 一一对应）。
            if actor_dir.exists():
                import shutil as _shutil
                logger.warning(
                    "Actor 目录已存在（code 重用或残留），清空后重建: %s", actor_dir,
                )
                _shutil.rmtree(actor_dir)
            actor_dir.mkdir(parents=True, exist_ok=True)
            
            # 将临时人脸文件移动到 Actor 专属目录
            moved_faces = []
            for i, face in enumerate(faces):
                temp_path = Path(face["face_path"])
                if temp_path.exists():
                    # 生成新的持久化文件名
                    new_filename = f"face_{i}.jpg"
                    new_path = actor_dir / new_filename
                    
                    try:
                        # 移动文件（新目录已清空，目标不会存在；双保险判断）
                        if new_path.exists():
                            logger.warning(
                                "目标文件已存在，跳过以避免覆盖: %s", new_path,
                            )
                            continue
                        # 移动文件
                        temp_path.rename(new_path)
                        
                        # 更新 face dict 中的路径
                        face["face_path"] = str(new_path)
                        
                        # 更新 best_face 引用
                        if best_face and best_face["face_path"] == str(temp_path):
                            best_face["face_path"] = str(new_path)
                        
                        moved_faces.append(face)
                    except Exception as e:
                        logger.error(f"Failed to move face file {temp_path} to {new_path}: {e}")
                        # 清理已移动的文件
                        for moved_face in moved_faces:
                            try:
                                Path(moved_face["face_path"]).unlink()
                            except:
                                pass
                        raise
            
            # 验证所有文件都已成功移动
            if len(moved_faces) != len(faces):
                raise Exception(f"Only {len(moved_faces)}/{len(faces)} faces were moved successfully")

            # (Actor 行已在上方创建并落库, 此处仅补写封面由 _apply_actor_cover 统一维护)

            # 创建 ActorFace 记录（is_cover 由 _apply_actor_cover 统一设置）
            first_face_id = None
            for i, face in enumerate(faces):
                # 存储相对路径
                face_path_abs = Path(face["face_path"])
                try:
                    face_path_rel = str(face_path_abs.relative_to(self.faces_dir))
                except ValueError:
                    face_path_rel = face_path_abs.name  # 如果不在 faces_dir 下，只存文件名
                
                actor_face = ActorFace(
                    actor_id=actor.id,
                    task_id=task_id,
                    entity_id=face.get("entity_id") or _fallback_entity_id,
                    file_md5=file_md5,
                    face_path=face_path_rel,
                    yaw=face["yaw"],
                    pitch=face["pitch"],
                    roll=face["roll"],
                    is_cover=0,
                    sharpness=face.get('sharpness', 0),
                    quality_score=face.get('quality_score', 0),
                    eyes_open_score=face.get('eyes_open_score', 0),
                    embedding=encode_embedding(face.get("embedding")),
                    face_type='main' if i == 0 else 'merged',
                    main_face_id=None if i == 0 else first_face_id,
                    # 时空/检测维度
                    frame_id=face.get("frame_id"),
                    timestamp=face.get("timestamp"),
                    bbox=json.dumps(face["bbox"]) if face.get("bbox") else None,
                    det_score=face.get("det_score"),
                    kps=json.dumps(face["kps"]) if face.get("kps") is not None else None,
                    scene_id=face.get("scene_id"),
                    sex=face.get("sex"),
                    age=face.get("age"),
                )
                db.add(actor_face)
                db.flush()  # 获取 ID
                
                if i == 0:
                    first_face_id = actor_face.id  # 记录主脸 ID

            # 统一应用封面（best_face 首选，失败自动回退有效脸）
            self._apply_actor_cover(
                db, actor.id,
                preferred_path=best_face["face_path"] if best_face else None,
            )
            db.commit()
            
            # 验证 ActorFace 记录已创建
            face_count = db.query(ActorFace).filter(ActorFace.actor_id == actor.id).count()
            if face_count == 0:
                logger.error(f"Actor {actor.id} was created but has no face records, rolling back")
                db.rollback()
                # 清理已创建的文件
                for face in faces:
                    try:
                        Path(face["face_path"]).unlink()
                    except:
                        pass
                try:
                    actor_dir.rmdir()
                except:
                    pass
                return None
            
            logger.info("Created actor %s (id=%d) with %d faces", code, actor.id, len(faces))
            return actor.id
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create actor: {e}")
            # 清理已创建的文件
            for face in faces:
                try:
                    Path(face["face_path"]).unlink()
                except:
                    pass
            try:
                if actor_dir is not None and actor_dir.exists():
                    actor_dir.rmdir()
            except:
                pass
            raise
        finally:
            if own_session:
                db.close()

    def update_actor(self, actor_id: int, faces: List[Dict], task_id: int, file_md5: str = None, db_session=None) -> None:
        """Add new face samples to an existing actor.

        Updates the average embedding, sample count, and potentially the cover
        image if a better frontal face is found.

        Args:
            actor_id: The actor to update.
            faces: New face dicts.
            task_id: The task that produced these faces.
            file_md5: The MD5 hash of the video file (for tracking source).
            db_session: Optional shared database session.
        """
        from src.db.models import Actor, ActorFace, get_session
        from src.db.models import get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            import sqlite3
            if isinstance(db_session, sqlite3.Connection):
                engine = get_db_engine()
                db = get_session(engine)
                own_session = True
            else:
                db = db_session

        # 文件实体 id: 优先用 face 自带, 否则按 task_id 解析(保证落库, 供按文件统计)
        _fallback_entity_id = self._resolve_entity_id(db, task_id)

        # BUG-023: 去除同一张脸的重复提取（预筛分类重叠等历史原因）
        faces = self._dedup_faces(faces)

        try:
            actor = db.query(Actor).filter(Actor.id == actor_id).first()
            if not actor:
                logger.warning("Actor %d not found for update", actor_id)
                return

            # 每视频追加 Top-N 代表脸(默认 5)：新簇脸精选后再并入,
            # 避免一次并入把几十上百张脸全塞进 actor(业务模型: 每人物少量代表脸)。
            _max_n = self._max_samples_per_actor()
            if len(faces) > _max_n:
                faces = self._select_representative_faces(faces, _max_n)

            # 获取该人物的主脸 ID
            main_face = db.query(ActorFace).filter(
                ActorFace.actor_id == actor_id,
                ActorFace.face_type == 'main'
            ).first()
            main_face_id = main_face.id if main_face else None

            # Recompute average embedding from ALL faces (including various poses)
            existing_emb = decode_embedding(actor.face_embedding)
            new_embeddings = [f["embedding"] for f in faces]
            all_embeddings = [existing_emb] + new_embeddings
            avg_embedding = np.mean(all_embeddings, axis=0)

            actor.face_embedding = encode_embedding(avg_embedding)

            # ── 封面选择（先于文件搬移：此时 temp 路径可读，brightness 计算有效）──
            best_new = self.select_best_frontal_face(faces)
            should_update_cover = False
            best_path_rel = None
            if best_new:
                current_cover = (
                    db.query(ActorFace)
                    .filter(ActorFace.actor_id == actor_id, ActorFace.is_cover == 1)
                    .first()
                )

                if current_cover:
                    current_angle = (
                        abs(current_cover.yaw or 0)
                        + abs(current_cover.pitch or 0)
                        + abs(current_cover.roll or 0)
                    )
                    new_angle = (
                        abs(best_new["yaw"]) + abs(best_new["pitch"]) + abs(best_new["roll"])
                    )
                    should_update_cover = new_angle < current_angle
                else:
                    should_update_cover = True

                if should_update_cover:
                    # 旧封面 demote 由登记后的 _apply_actor_cover 统一完成（清全部 is_cover）
                    pass

            # 统一分配磁盘编号（封面先搬，编号最小），
            # 避免"封面 count() + 非封面 count()"两次取号同名覆盖：
            # db.add() 未 flush 前 count() 不变，两次取号相同 → 后 rename 覆盖先 rename 的图。
            actor_dir = self.faces_dir / actor.code
            actor_dir.mkdir(parents=True, exist_ok=True)
            # 编号从磁盘 max+1 开始（兼容历史孤儿），全程只递增，绝不复用
            _next_idx = self._next_face_index(actor_dir)

            # 封面文件优先搬移（若需更新封面）。成功归位后加入 persisted_faces
            # （与其余脸统一登记，避免封面脸漏登记）。
            persisted_faces = []
            if best_new and should_update_cover:
                old_path = Path(best_new["face_path"])
                _cover_ok = False
                if old_path.exists():
                    try:
                        inside = old_path.resolve().is_relative_to(actor_dir.resolve())
                    except Exception:
                        inside = False
                    if inside:
                        # 已在 actor 目录：确认相对路径后收进持久化列表
                        try:
                            best_new["face_path"] = str(old_path.relative_to(self.faces_dir))
                        except ValueError:
                            pass
                        _cover_ok = True
                    else:
                        dst = actor_dir / f"face_{_next_idx}.jpg"
                        while dst.exists():
                            _next_idx += 1
                            dst = actor_dir / f"face_{_next_idx}.jpg"
                        try:
                            old_path.rename(dst)
                            _next_idx += 1
                            # best_new 与 faces 中的元素是同一对象，此处更新即全列表生效
                            try:
                                best_new["face_path"] = str(dst.relative_to(self.faces_dir))
                            except ValueError:
                                best_new["face_path"] = dst.name
                            _cover_ok = True
                        except Exception as e:
                            logger.warning(
                                "update_actor: 封面搬移失败（跳过封面更新）: %s -> %s, %s",
                                old_path, dst, e,
                            )
                if _cover_ok:
                    persisted_faces.append(best_new)
                else:
                    # 封面脸未能落盘：本次不更新 cover_face，保留旧封面
                    should_update_cover = False
                if should_update_cover:
                    best_path_rel = best_new["face_path"]

            # 搬移其余新脸文件到 actor 目录；persisted_faces 只收集
            # 成功落盘到 actor 目录的脸（face_path 已是 ACTOR_xxx/face_N.jpg 相对路径）。
            # 历史 bug：文件不存在/搬移失败时 continue，但下方 DB 登记仍遍历原始 faces，
            # 以未更新的 temp 绝对路径入库 → 记录悬空（文件随后被清理，目录与 DB 不一致，
            # 实测 task 59/61/63 重建后多个 actor 磁盘比 DB 少 1 个文件）。
            for face in faces:
                # 仅当封面段已处理 best_new（需更新封面）时才跳过；否则 best_new
                # 作为普通脸走下方搬移逻辑（历史 bug：best_new 被无条件 continue
                # 后既不搬移也不登记 → 漏脸）。
                if best_new and should_update_cover and face is best_new:
                    continue
                old_path = Path(face["face_path"])
                if not old_path.exists():
                    # 文件不存在：跳过入库（历史曾保留原 path 入库 → 悬空记录）
                    logger.warning(
                        "update_actor: 跳过不存在的人脸文件: %s", old_path,
                    )
                    continue
                # 已在 actor 目录内的文件（如二次 update 传入已归位文件）无需再搬
                try:
                    if old_path.resolve().is_relative_to(actor_dir.resolve()):
                        # 已归位：确认 face_path 是相对路径后直接收进持久化列表
                        try:
                            face["face_path"] = str(old_path.relative_to(self.faces_dir))
                        except ValueError:
                            pass
                        persisted_faces.append(face)
                        continue
                except Exception:
                    pass
                dst = actor_dir / f"face_{_next_idx}.jpg"
                while dst.exists():
                    _next_idx += 1
                    dst = actor_dir / f"face_{_next_idx}.jpg"
                try:
                    old_path.rename(dst)
                    _next_idx += 1
                except Exception as e:
                    logger.warning(
                        "update_actor: 移动人脸文件失败（跳过入库）: %s -> %s, %s",
                        old_path, dst, e,
                    )
                    continue
                # 存储相对路径
                try:
                    face["face_path"] = str(dst.relative_to(self.faces_dir))
                except ValueError:
                    face["face_path"] = dst.name
                persisted_faces.append(face)

            # 登记新脸（is_cover 由 _apply_actor_cover 统一维护，此处全部置 0，
            # 避免 cover_face 字段与 is_cover 标记脱节的历史 bug）
            for face in persisted_faces:
                af = ActorFace(
                    actor_id=actor_id,
                    task_id=task_id,
                    entity_id=face.get("entity_id") or _fallback_entity_id,
                    file_md5=file_md5,
                    face_path=face["face_path"],
                    yaw=face["yaw"],
                    pitch=face["pitch"],
                    roll=face["roll"],
                    is_cover=0,
                    sharpness=face.get('sharpness', 0),
                    quality_score=face.get('quality_score', 0),
                    eyes_open_score=face.get('eyes_open_score', 0),
                    embedding=encode_embedding(face.get("embedding")),
                    face_type='merged',
                    main_face_id=main_face_id,
                    frame_id=face.get("frame_id"),
                    timestamp=face.get("timestamp"),
                    bbox=json.dumps(face["bbox"]) if face.get("bbox") else None,
                    det_score=face.get("det_score"),
                    kps=json.dumps(face["kps"]) if face.get("kps") is not None else None,
                    scene_id=face.get("scene_id"),
                    sex=face.get("sex"),
                    age=face.get("age"),
                )
                db.add(af)

            db.flush()
            # 统一应用封面：本次候选角度更优时以它为封面，否则保留现有有效封面
            if best_new and should_update_cover and best_path_rel:
                self._apply_actor_cover(db, actor_id, preferred_path=best_path_rel)
            else:
                self._apply_actor_cover(db, actor_id)
            # sample_count 以最终落库行数为准（含历史 + 本次新增）
            self._refresh_sample_count(db, actor_id)

            db.commit()
            logger.info("Updated actor %d with %d new faces", actor_id, len(faces))
        except Exception:
            db.rollback()
            raise
        finally:
            if own_session:
                db.close()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def recognize_actor(self, frames_path: str, task_id: int, file_md5: str = None) -> Optional[int]:
        """Recognize the actor in a video's frames.

        Extracts faces, computes the average embedding, and either matches
        an existing actor or creates a new one.

        Args:
            frames_path: Path to the frame directory.
            task_id: The task that produced the frames.
            file_md5: The MD5 hash of the video file (for tracking source).

        Returns:
            actor_id if a face was found/matched, None otherwise.
        """
        if not self.available:
            logger.info("Face recognition unavailable, skipping")
            return None

        try:
            faces = self.extract_faces_from_frames(frames_path)
            if not faces:
                logger.info("No faces detected in %s", frames_path)
                return None

            avg_embedding = np.mean([f["embedding"] for f in faces], axis=0)

            actor_id = self.match_actor(avg_embedding)
            if actor_id:
                self.update_actor(actor_id, faces, task_id, file_md5)
            else:
                actor_id = self.create_actor(faces, task_id, file_md5)

            return actor_id
        except Exception as exc:
            logger.error("Face recognition failed for task %d: %s", task_id, exc)
            return None

    # ------------------------------------------------------------------
    # Multi-actor recognition
    # ------------------------------------------------------------------

    def recognize_actors(self, frames_path: str, task_id: int, file_md5: str = None, db_session=None) -> List[int]:
        """识别视频中的所有人物

        提取所有人脸，使用 DBSCAN 聚类分组，为每个聚类匹配或创建人物。
        包含“保底 1 人”策略：如果标准检测失败，自动降级重试。

        Args:
            frames_path: 帧目录路径
            task_id: 任务ID
            file_md5: 视频文件MD5
            db_session: 可选数据库会话，用于共享事务

        Returns:
            识别到的人物ID列表
        """
        logger.info(f"recognize_actors called: frames_path={frames_path}, task_id={task_id}")
        if not self.available:
            logger.info("Face recognition unavailable, skipping")
            return []

        try:
            # 1. 提取所有人脸 (标准参数)
            faces = self.extract_faces_from_frames(frames_path)

            # ── 保底 1 人策略 (Guarantee 1 Person) ──
            # 如果标准检测结果为 0，自动降低门槛重试一次
            if not faces:
                logger.warning("Standard detection found 0 faces, retrying with relaxed thresholds...")
                # 降低检测置信度 (0.15 -> 0.05) 和 清晰度阈值 (0.01 -> 0.005)
                faces = self.extract_faces_from_frames(frames_path, det_thresh=0.05, clarity_threshold=0.005)

                if faces:
                    logger.info("Relaxed retry successful: found %d faces", len(faces))
                else:
                    logger.info("Even relaxed retry found 0 faces. Skipping actor recognition.")

            if not faces:
                return []

            # 2. 聚类（从配置读取参数）
            # 使用层次聚类算法进行人脸聚类
            from src.core.config import ConfigManager
            _config = ConfigManager()
            _distance_threshold = _config.get("face_recognition.distance_threshold", DEFAULT_DISTANCE_THRESHOLD)
            _iterations = _config.get("face_recognition.iterations", 20)
            _use_auto = bool(_config.get("face_recognition.auto_params.enabled", False))
            clusterer = FaceClusterer(
                distance_threshold=_distance_threshold,
                use_auto_params=_use_auto,
                track_config=self._track_grouping_config(),
            )
            clusters = clusterer.cluster_faces(faces)

            if not clusters:
                logger.info("No clusters formed from %d faces", len(faces))
                return []

            # 3. 为每个聚类匹配或创建人物
            actor_ids = []
            _match_existing = bool(_config.get("face_recognition.match_existing_actors", True))
            for cluster_idx, cluster_faces in enumerate(clusters):

                # 尝试匹配已有人物（受 match_existing_actors 开关控制；
                # 关闭时单文件优先，每个聚类独立建新人物）
                actor_id = None
                if _match_existing:
                    actor_id = self.match_actor_v2(cluster_faces, db_session=db_session)
                if actor_id:
                    self.update_actor(actor_id, cluster_faces, task_id, file_md5, db_session=db_session)
                    logger.info(
                        "Cluster %d/%d: matched existing actor %d (%d faces)",
                        cluster_idx + 1, len(clusters), actor_id, len(cluster_faces)
                    )
                else:
                    actor_id = self.create_actor(cluster_faces, task_id, file_md5, db_session=db_session)
                    if actor_id is None:
                        logger.info(
                            "Cluster %d/%d: skipped — all faces below clarity threshold (%d faces)",
                            cluster_idx + 1, len(clusters), len(cluster_faces)
                        )
                    else:
                        logger.info(
                            "Cluster %d/%d: created new actor %d (%d faces)",
                            cluster_idx + 1, len(clusters), actor_id, len(cluster_faces)
                        )

                if actor_id is not None and actor_id not in actor_ids:
                    actor_ids.append(actor_id)

            logger.info(
                "Task %d: recognized %d actors from %d faces in %d clusters",
                task_id, len(actor_ids), len(faces), len(clusters)
            )
            return actor_ids

        except Exception as exc:
            logger.error("Multi-actor recognition failed for task %d: %s", task_id, exc)
            logger.error("Full traceback:\n%s", traceback.format_exc())
            return []

    def precheck_video_quality(self, video_path: str) -> Dict:
        """快速预检视频质量（< 10秒），决定是否值得处理。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            {
                'should_process': bool,  # 是否应该处理
                'quality_score': float,  # 质量评分 (0-1)
                'skip_reason': str,      # 跳过原因（如果should_process=False）
                'details': {             # 详细信息
                    'resolution': [width, height],
                    'duration': float,
                    'avg_clarity': float,
                    'avg_brightness': float,
                    'sampled_frames': int
                }
            }
        """
        from src.core.config import ConfigManager
        cfg = ConfigManager()
        
        # 预检配置
        precheck_cfg = cfg.get("face_recognition.precheck", {})
        if not precheck_cfg.get("enabled", True):
            return {
                'should_process': True,
                'quality_score': 1.0,
                'skip_reason': None,
                'details': {}
            }
        
        result = {
            'should_process': True,
            'quality_score': 0.0,
            'skip_reason': None,
            'details': {}
        }
        
        try:
            # 1. 视频元数据检查
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['should_process'] = False
                result['skip_reason'] = '无法打开视频文件'
                return result
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            result['details']['resolution'] = [width, height]
            result['details']['duration'] = duration
            
            # 分辨率检查：与主流程 face_recognition.min_resolution 一致，按像素面积判断。
            # 历史 bug：按 [640, 480] 宽高逐项 AND 判断，720x406 这类面积达标
            # （29 万 px > 23 万）的单边略低视频被误杀（用户实测 2025_mmexport...8475.mp4
            # 多人脸清晰可识别）。兼容三种配置形态：
            #   - [640, 480]（旧）→ 取其面积作为像素门槛
            #   - 230400（数字）→ 直接作为像素门槛
            #   - 未配置 → 回退主流程 face_recognition.min_resolution
            _min_res_cfg = precheck_cfg.get("min_resolution")
            if _min_res_cfg is None:
                try:
                    _min_res_cfg = int(ConfigManager().get("face_recognition.min_resolution", 230400))
                except Exception:
                    _min_res_cfg = 230400
            try:
                if isinstance(_min_res_cfg, (list, tuple)) and len(_min_res_cfg) == 2:
                    _min_px = int(_min_res_cfg[0]) * int(_min_res_cfg[1])
                else:
                    _min_px = int(_min_res_cfg)
            except Exception:
                _min_px = 230400
            # 兜底：极低短边（<320px，约 240p 以下）即使面积达标也无识别价值
            _min_short_edge = int(precheck_cfg.get("min_short_edge", 320))
            _short_edge = min(width, height)
            if width * height < _min_px or _short_edge < _min_short_edge:
                result['should_process'] = False
                result['skip_reason'] = (
                    f'分辨率过低 ({width}x{height}={width*height}px '
                    f'< {_min_px}px 或短边 {_short_edge}px < {_min_short_edge}px)'
                )
                cap.release()
                return result
            
            # 时长检查
            min_duration = precheck_cfg.get("min_duration", 10)
            if duration < min_duration:
                result['should_process'] = False
                result['skip_reason'] = f'时长过短 ({duration:.1f}s < {min_duration}s)'
                cap.release()
                return result

            # 码率估算（文件大小 / 时长）——无解码的强质量信号。
            # 同分辨率下码率越低压缩越狠，人脸越糊（如 50 分钟仅 86MB ≈ 236kbps）。
            # 作为评分维度之一参与降权，不单独拦截（用户确认）。
            try:
                _file_bytes = os.path.getsize(video_path)
            except Exception:
                _file_bytes = 0
            # 视频轨码率 ≈ 总码率 × 0.9（音频约占 10%），粗略即可
            total_kbps = (_file_bytes * 8 / 1000) / duration if duration > 0 else 0
            video_kbps = total_kbps * 0.9
            result['details']['file_size_bytes'] = _file_bytes
            result['details']['bitrate_kbps'] = round(video_kbps, 1)
            
            # 2. 采样关键帧
            sample_frames = precheck_cfg.get("sample_frames", 5)
            sampled_frames = []
            
            # 均匀采样（跳过前10%和后10%）
            start_frame = int(total_frames * 0.1)
            end_frame = int(total_frames * 0.9)
            step = max(1, (end_frame - start_frame) // sample_frames)
            
            for i in range(sample_frames):
                frame_idx = start_frame + i * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    sampled_frames.append(frame)
            
            cap.release()
            
            if not sampled_frames:
                result['should_process'] = False
                result['skip_reason'] = '无法采样关键帧'
                return result
            
            result['details']['sampled_frames'] = len(sampled_frames)
            
            # 3. 快速质量评估
            clarities = []
            brightnesses = []
            
            for frame in sampled_frames:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 清晰度（Laplacian方差）
                clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                clarities.append(clarity)
                
                # 亮度（平均值）
                brightness = gray.mean()
                brightnesses.append(brightness)
            
            avg_clarity = np.mean(clarities)
            avg_brightness = np.mean(brightnesses)
            
            result['details']['avg_clarity'] = avg_clarity
            result['details']['avg_brightness'] = avg_brightness
            
            # 4. 计算质量评分
            # 维度: resolution / clarity / brightness / contrast / bitrate
            # 旧 config 无 bitrate 权重时，按比例从其他维度让出，保证总和 1.0。
            weights = dict(precheck_cfg.get("weights", {
                "resolution": 0.25,
                "clarity": 0.35,
                "brightness": 0.15,
                "contrast": 0.05,
                "bitrate": 0.20,
            }))
            _br_w = float(weights.get("bitrate", 0.20))
            _others = {k: float(v) for k, v in weights.items() if k != "bitrate"}
            _sum = sum(_others.values()) or 1.0
            # 归一化非码率权重到 (1 - bitrate权重)，bitrate 权重截到 [0.05, 0.4]
            _br_w = max(0.05, min(0.40, _br_w))
            for _k in _others:
                weights[_k] = _others[_k] / _sum * (1.0 - _br_w)
            weights["bitrate"] = _br_w

            # 分辨率评分（0-1）
            res_score = min(1.0, (width * height) / (1920 * 1080))

            # 清晰度评分（0-1）
            min_clarity = precheck_cfg.get("min_clarity", 20.0)
            clarity_score = min(1.0, avg_clarity / (min_clarity * 3))

            # 亮度评分（0-1）- 理想范围 80-180
            if 80 <= avg_brightness <= 180:
                brightness_score = 1.0
            elif avg_brightness < 80:
                brightness_score = avg_brightness / 80
            else:
                brightness_score = max(0, 1 - (avg_brightness - 180) / 75)

            # 对比度评分（简化为清晰度）
            contrast_score = clarity_score

            # 码率评分（0-1）——视频轨码率越高压缩损失越小。
            # 参考量级：240p~360p 低码率 ≈ 200-400kbps；480p ≈ 500-800kbps；
            # 720p ≈ 1-2.5Mbps；1080p ≈ 2.5-8Mbps。
            # 曲线：≤300kbps→0.15；300k~1M 线性 0.15→0.55；1M~2.5M 线性 0.55→0.85；
            # ≥2.5Mbps→1.0。低码率给保底 0.15（不归零，避免与清晰度双重惩罚过狠）。
            video_kbps = result['details'].get('bitrate_kbps', 0) or 0
            if video_kbps <= 300:
                bitrate_score = 0.15
            elif video_kbps <= 1000:
                bitrate_score = 0.15 + 0.40 * (video_kbps - 300) / 700
            elif video_kbps <= 2500:
                bitrate_score = 0.55 + 0.30 * (video_kbps - 1000) / 1500
            else:
                bitrate_score = 1.0

            # 综合评分（权重已归一化，和 = 1.0）
            quality_score = (
                res_score * weights.get("resolution", 0.25) +
                clarity_score * weights.get("clarity", 0.35) +
                brightness_score * weights.get("brightness", 0.15) +
                contrast_score * weights.get("contrast", 0.05) +
                bitrate_score * weights.get("bitrate", 0.20)
            )

            result['quality_score'] = quality_score
            
            # 5. 判断是否处理
            min_quality = precheck_cfg.get("min_quality_score", 0.5)
            if quality_score < min_quality:
                result['should_process'] = False
                result['skip_reason'] = f'质量评分过低 ({quality_score:.2f} < {min_quality})'
            
            return result
            
        except Exception as e:
            logger.error("Precheck failed: %s", e)
            result['should_process'] = False
            result['skip_reason'] = f'预检失败: {str(e)}'
            return result

    def _quick_face_presence(self, video_path: str, n_samples: int = 24) -> Dict:
        """快速人脸存在性检测(两阶段流水线前置快筛)。

        场景切分/抽帧之前直接从视频均匀抽帧, 用 320 gate 检测器(CPU ~50fps)
        判断视频里是否真的有人脸。用于:
          - 无人脸视频快速跳过完整识别(省场景切分+抽帧+预筛的 1-2 分钟)
          - 替代"零脸→判动画跳过"的错误语义(实测 320 gate 召回优于 640/Vision,
            见 face_test_harness/data/results/face_detect_bench.json)

        注意: 零脸 ≠ 动画(真人视频可能人脸小/遮挡/暗光漏检), 返回 no_face 时
        调用方应标记为"未检出"(face_scan_status=none)由用户确认, 而非静默丢弃。

        Args:
            video_path: 视频路径
            n_samples: 采样帧数(默认 24, 10%-90% 区间均匀)

        Returns:
            {"has_face": bool, "faces_total": int, "frames_checked": int,
             "sample_frame_nums": [..] 或 None}
        """
        import cv2 as _cv2q
        cap = _cv2q.VideoCapture(video_path)
        if not cap.isOpened():
            return {"has_face": False, "faces_total": 0, "frames_checked": 0,
                    "sample_frame_nums": None, "error": "无法打开视频"}
        total = int(cap.get(_cv2q.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return {"has_face": False, "faces_total": 0, "frames_checked": 0,
                    "sample_frame_nums": None, "error": "无帧信息"}
        # 10%-90% 均匀采样 n_samples 帧
        start_frac, end_frac = 0.1, 0.9
        span = max(1, total - 1)
        sample_ids = sorted({
            int(span * (start_frac + (end_frac - start_frac) * k / max(1, n_samples - 1)))
            for k in range(n_samples)
        })

        gate = self._get_gate_app()
        if gate is None:
            logger.warning("快速人脸存在性检测跳过: 闸门检测器不可用(模型缺失)")
            return {"has_face": False, "frames_checked": 0, "faces_total": 0,
                    "skip_reason": "face_model_missing"}
        t0 = time.time()
        faces_total = 0
        frames_checked = 0
        for fn in sample_ids:
            cap.set(_cv2q.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames_checked += 1
            try:
                det = gate.get(frame)
            except Exception:
                det = []
            n = len(det)
            faces_total += n
            if n > 0:
                cap.release()
                return {
                    "has_face": True, "faces_total": faces_total,
                    "frames_checked": frames_checked,
                    "sample_frame_nums": sample_ids,
                    "seconds": round(time.time() - t0, 2),
                }
        cap.release()
        logger.info(
            "快速人脸存在性检测: %d 帧均未检出人脸 (%.2fs)",
            frames_checked, time.time() - t0,
        )
        return {
            "has_face": False, "faces_total": 0,
            "frames_checked": frames_checked,
            "sample_frame_nums": sample_ids,
            "seconds": round(time.time() - t0, 2),
        }

    def _is_2d_anime_video(self, video_path: str, n_frames: int = 3) -> Optional[bool]:
        """用 VLM 判断视频是否为 2D 手绘动画(仅拦 2D, 不判 3D)。

        2D 动画(手绘线条+平涂上色)的角色脸会被 AuraFace 当真人脸建人物
        (如《快乐方程式》误建 ACTOR)。真人成人片常带磨皮滤镜/厂商水印,
        VLM 易误判成"3D/CG"——因此判定口径**只认 2D 手绘**,真人/3D/其他
        一律视为非 2D(不拦),宁可多建不误杀真人。

        抽帧避开片头(15%/50%/85%),3 帧一次请求 VLM, 让它输出 yes/no。

        Returns:
            True  → 判定为 2D 动画(应跳过识别)
            False → 非 2D 动画(继续识别)
            None  → 判定失败/超时/模型不可用(按"不拦"处理, 保真人)
        """
        try:
            import base64 as _b64
            import urllib.request as _urlreq

            # 判定开关: face_recognition.skip_anime_video (默认 False —— 保守, 需用户显式开启)
            try:
                from src.core.config import ConfigManager as _Cfg
                if not _Cfg().get("face_recognition.skip_anime_video", False):
                    return False
            except Exception:
                return False

            # 取 VLM 配置(base_url/model)
            try:
                from src.core.config import ConfigManager as _Cfg2
                _vcfg = _Cfg2().get_vlm_config()
                base_url = _vcfg.get("base_url", "http://localhost:11434")
                model = _vcfg.get("model", "minicpm-v4.5")
            except Exception:
                base_url = "http://localhost:11434"
                model = "minicpm-v4.5"

            # 抽 n 帧(15%/50%/85%, 避开片头)
            import cv2 as _cv2a
            cap = _cv2a.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            total = int(cap.get(_cv2a.CAP_PROP_FRAME_COUNT))
            fps = cap.get(_cv2a.CAP_PROP_FPS) or 25.0
            cap.release()
            if total <= 0 or fps <= 0:
                return None
            span = max(1, total - 1)
            # 15% 起, 均匀取 n_frames, 上限 90%
            fracs = [0.15 + 0.75 * k / max(1, n_frames - 1) for k in range(n_frames)]
            sample_ids = sorted({min(int(span * f), total - 1) for f in fracs})

            frame_imgs = []
            cap = _cv2a.VideoCapture(video_path)
            for fn in sample_ids:
                cap.set(_cv2a.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                # 压缩到 ~320 宽, jpg 编码(控制 token 与体积)
                h, w = frame.shape[:2]
                if w > 320:
                    nh = int(h * 320 / w)
                    frame = _cv2a.resize(frame, (320, nh))
                ok, buf = _cv2a.imencode(".jpg", frame, [_cv2a.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    frame_imgs.append(_b64.b64encode(buf.tobytes()).decode())
            cap.release()
            if not frame_imgs:
                return None

            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "判断这些画面是否是【2D手绘动画】(日本动漫/漫画风格: 线条+平涂上色)。\n"
                        "注意: 真人实拍即使磨皮/滤镜/带水印/画质差也不算动画; 3D渲染也不是2D手绘。\n"
                        "只输出一个词: yes 或 no。"
                    ),
                    "images": frame_imgs,
                }],
                "stream": False,
                "options": {"num_predict": 5},
            }
            req = _urlreq.Request(
                base_url.rstrip("/") + "/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with _urlreq.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read())
            verdict = (d.get("message", {}).get("content", "") or "").strip().lower()
            is_anime = verdict.startswith("yes")
            logger.info("VLM 2D动画判定(%s): %s → %s", Path(video_path).name, verdict, is_anime)
            return is_anime
        except Exception as e:
            logger.warning("VLM 2D动画判定失败(按不拦处理): %s", e)
            return None


    def recognize_actors_from_video(self, video_path: str, task_id: int, file_md5: str = None, db_session=None) -> List[int]:
        """从视频文件中识别人物（正确流程：场景检测 → 帧抽取 → 人脸检测）。

        流程：
        1. 快速人脸存在性检测(320 gate 抽帧判有无脸, ~1s; 零脸 → none 跳过)
        2. 场景检测（PySceneDetect，几秒）
        3. 帧抽取（每场景3-5帧，快）
        4. 人脸检测+特征提取（在抽好的帧上）
        5. 聚类+匹配人物

        Args:
            video_path: 视频文件路径
            task_id: 任务ID
            file_md5: 视频文件MD5
            db_session: 可选数据库会话，用于共享事务

        Returns:
            识别到的人物ID列表
        """
        logger.info(f"recognize_actors_from_video called: video_path={video_path}, task_id={task_id}")
        if not self.available:
            logger.info("Face recognition unavailable, skipping")
            return []

        if not video_path or not Path(video_path).exists():
            logger.warning("Video file does not exist: %s", video_path)
            return []

        # 去重检查：如果该视频已被处理过，直接返回已有的人物 ID
        if db_session is not None:
            import sqlite3
            try:
                # 方法1：通过 actor_faces 的 task_id 关联 tasks 表查询
                if isinstance(db_session, sqlite3.Connection):
                    cur = db_session.cursor()
                    # 先检查 tasks 表是否有该视频的记录
                    cur.execute("SELECT id FROM tasks WHERE file_path = ?", (video_path,))
                    task_row = cur.fetchone()
                    if task_row:
                        task_id_existing = task_row[0]
                        # 检查该任务是否已有人脸记录
                        cur.execute("""
                            SELECT DISTINCT actor_id 
                            FROM actor_faces 
                            WHERE task_id = ?
                        """, (task_id_existing,))
                        existing_actor_ids = [r[0] for r in cur.fetchall()]
                        if existing_actor_ids:
                            logger.info("Task %d: 视频已处理过 (task_id=%d)，跳过重复处理 (已有 %d 个人物)", 
                                       task_id, task_id_existing, len(existing_actor_ids))
                            return existing_actor_ids
                else:
                    from src.db.models import Task
                    existing_task = db_session.query(Task).filter(
                        Task.file_path == video_path
                    ).first()
                    if existing_task:
                        from src.db.models import ActorFace
                        existing_faces = db_session.query(ActorFace).filter(
                            ActorFace.task_id == existing_task.id
                        ).all()
                        if existing_faces:
                            existing_actor_ids = list(set(f.actor_id for f in existing_faces))
                            logger.info("Task %d: 视频已处理过 (task_id=%d)，跳过重复处理 (已有 %d 个人物)", 
                                       task_id, existing_task.id, len(existing_actor_ids))
                            return existing_actor_ids
            except Exception as dedup_exc:
                logger.warning("Task %d: 去重检查失败，继续处理: %s", task_id, dedup_exc)

        # 预检：快速评估视频质量（< 10秒）
        precheck_result = self.precheck_video_quality(video_path)
        if not precheck_result['should_process']:
            logger.info(
                "Task %d: 预检跳过 - %s (质量评分: %.2f)",
                task_id, precheck_result['skip_reason'], precheck_result['quality_score']
            )
            return []
        
        logger.info(
            "Task %d: 预检通过 - 质量评分: %.2f (分辨率: %s, 时长: %.1fs, 清晰度: %.1f)",
            task_id,
            precheck_result['quality_score'],
            precheck_result['details'].get('resolution'),
            precheck_result['details'].get('duration', 0),
            precheck_result['details'].get('avg_clarity', 0)
        )

        # 统一将 sqlite3.Connection 转换为 SQLAlchemy session
        import sqlite3
        _own_session = False
        if db_session is not None and isinstance(db_session, sqlite3.Connection):
            from src.db.models import get_session
            from src.db.models import get_db_engine
            engine = get_db_engine()
            db_session = get_session(engine)
            _own_session = True
        elif db_session is None:
            from src.db.models import get_session
            from src.db.models import get_db_engine
            engine = get_db_engine()
            db_session = get_session(engine)
            _own_session = True

        def _finish(result):
            if _own_session:
                try:
                    db_session.commit()
                except Exception:
                    pass
                try:
                    db_session.close()
                except Exception:
                    pass
            return result

        # 分辨率门槛
        try:
            from src.core.config import ConfigManager as _CfgMgr
            _min_res = int(_CfgMgr().get("face_recognition.min_resolution", 230400))
        except Exception:
            _min_res = 230400
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            if _cap.isOpened():
                _w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                _h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                _ret, _frame = _cap.read()
                _cap.release()
                if not _ret or _frame is None or _w <= 0 or _h <= 0:
                    logger.warning("Task %d: 视频文件损坏/无法解码，跳过", task_id)
                    return []
                if _w * _h < _min_res:
                    logger.info("Task %d: 低分辨率视频 (%dx%d)，跳过", task_id, _w, _h)
                    return []
            else:
                logger.warning("Task %d: 视频文件无法打开，跳过", task_id)
                return []
        except Exception as _res_exc:
            logger.warning("Task %d: 文件检查失败: %s", task_id, _res_exc)

        # ── 阶段1: 快速人脸存在性检测(前置快筛, ~1s) ──
        # 320 gate 抽 24 帧判有无脸。零脸不判"动画"(真人视频可能漏检)，
        # 返回空由调用方标记 face_scan_status=none 供用户确认。
        # 待接入动画/真人分类模型后, 可在此细分"动画"与"真人漏检"。
        _presence = self._quick_face_presence(video_path)
        if not _presence.get("has_face", False):
            logger.info(
                "Task %d: 快速存在性检测未检出人脸(%d帧), 跳过完整识别 → none",
                task_id, _presence.get("frames_checked", 0),
            )
            return _finish([])
        logger.info(
            "Task %d: 快速存在性检测检出人脸, 进入完整识别 (%.2fs)",
            task_id, _presence.get("seconds", 0),
        )

        # ── 阶段1.5: 2D 动画判定(可选) ──
        # 检出人脸不代表真人: 2D 动画角色脸也会被 AuraFace 检出并建人物
        # (如《快乐方程式》误建)。用 VLM 判定是否 2D 手绘动画, 是则跳过识别,
        # face_scan_status 留给调用方标记(此处返回空列表)。
        # 开关: config face_recognition.skip_anime_video=true 才启用。
        try:
            _anime = self._is_2d_anime_video(video_path)
        except Exception as _anime_exc:
            _anime = None
            logger.warning("Task %d: 2D动画判定异常(继续识别): %s", task_id, _anime_exc)
        if _anime:
            logger.info(
                "Task %d: VLM 判定为 2D 动画, 跳过人脸识别 (face_scan_status=skipped)",
                task_id,
            )
            # 标记 face_scan_status='skipped'(区别于"完整扫描无脸"的 none,
            # 前端可辨识为"动画/跳过识别")
            try:
                from src.db.models import FileAnalysis as _FAM
                _fa = db_session.query(_FAM).filter(
                    _FAM.task_id == task_id
                ).first()
                if _fa is not None:
                    _fa.face_scan_status = "skipped"
                    db_session.flush()
            except Exception as _fa_exc:
                logger.warning("Task %d: 标记动画跳过状态失败: %s", task_id, _fa_exc)
            return _finish([])

        # ── 阶段2+3: 帧抽取(均匀抽帧替代场景检测, 2026-09 实验定案) ──
        # 历史: 先 PySceneDetect 全片扫描(解码13s+直方图10s, 长视频 20s+)再按场景抽帧。
        # 实测(face_test_harness bench): 均匀抽帧在场景少/长视频覆盖更优(1537: 21/27 vs 15/27),
        # 快切视频略低(20: 704 vs 763 有脸帧)但省 8-24s 全片扫描。海量混合库整体吞吐更优。
        # Media Engine 硬解实测对"顺序解码取帧"无益(回读瓶颈, 9x 更慢), 故仍用软解。
        # 策略: 顺序解码一 pass, 每隔 step 帧存一张, 目标帧数按时长自适应。
        import tempfile
        import cv2
        from src.core.config import ConfigManager
        _config = ConfigManager()

        temp_frames_dir = Path(tempfile.mkdtemp(prefix=f"face_frames_task{task_id}_"))

        t_extract = time.time()
        all_frame_paths = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # 目标帧数: 按"目标采样间隔"推导（见 _uniform_sampling_target）。
        # 间隔的意义：聚类要求同一个人的相邻样本落在"人脸外观变化尺度"之内——
        # 实测约 2 秒间隔时同人距离 0.31 vs 跨人 0.855（可分）；7 秒间隔时 0.807 vs 0.855（不可分）。
        # 配置键 face_recognition.sampling_target_interval_seconds 默认 8.0（= 改动前行为）。
        _sampling_interval = float(
            _config.get("face_recognition.sampling_target_interval_seconds", 8.0)
        )
        # 帧数护栏（成本上限）：长视频会因此达不到目标间隔，故做成可配置。
        # 默认 400 = 改动前行为；取值由标定决定（见 docs/架构/人脸识别-验证样本登记.md）。
        _max_frames = int(_config.get("face_recognition.sampling_max_frames", 400))
        _target = 200
        if total_frames_hint > 0 and fps > 0:
            _target = self._uniform_sampling_target(
                total_frames_hint / fps, _sampling_interval, max_frames=_max_frames,
            )
        step = max(1, (total_frames_hint - 1) // max(1, _target - 1)) if total_frames_hint > 0 else 1
        logger.info(
            "Task %d: 抽帧目标 %d 帧 (目标间隔 %.1fs, 实际间隔 %.1fs)",
            task_id, _target, _sampling_interval,
            ((total_frames_hint / fps) / _target) if (_target and fps > 0) else 0,
        )

        # 顺序解码一 pass: 每隔 step 帧存一张(避免 cap.set 跳帧的 H.264 seek 开销)
        fn = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if fn % step == 0 and saved < _target:
                out_p = Path(str(temp_frames_dir)) / f"frame_{fn:06d}.jpg"
                cv2.imwrite(str(out_p), frame)
                all_frame_paths.append(str(out_p))
                saved += 1
            fn += 1
            # 每 5000 帧提示一次进度(debug 级别)
        cap.release()
        logger.info(
            "Task %d: 均匀抽帧完成: 目标 %d, 实际 %d 帧 (%.1fs, 时长 %.0fs)",
            task_id, _target, len(all_frame_paths), time.time() - t_extract,
            (total_frames_hint / fps) if fps > 0 else 0,
        )

        # 估算场景数(替代已移除的 scenes): 用采样帧数与相邻帧间距粗估。
        # 均匀抽帧下无场景语义, 用采样帧数近似作为动态参数的密度参考。
        _sampled_frames = len(all_frame_paths)
        _est_scenes = max(1, _sampled_frames // 3)

        if not all_frame_paths:
            logger.info("Task %d: 未抽取到帧，跳过", task_id)
            self._cleanup_temp_dir(temp_frames_dir)
            return _finish([])

        # ── 阶段4: 两阶段人脸检测+特征提取 ──
        # 两阶段流水线（业务需求：正脸先交付，侧脸后补全，渐进式呈现）：
        #   1. 预筛：320×320 轻量检测给帧分类（正脸/侧脸/无人脸）
        #   2. 正脸帧先完整分析 → 产出 actor（快，先可用）
        #   3. 侧脸帧补完整分析 → 并入已有 actor（后台补全）
        #   无人脸帧跳过完整分析（省算力，完整分析约 135ms/帧，是预筛的 5 倍）
        t_detect = time.time()

        # 阶段4a: 帧预筛分类
        prescreen = self._prescreen_frames(str(temp_frames_dir))
        frontal_frames = prescreen.get("frontal", [])
        side_frames = prescreen.get("side", [])
        empty_frames = prescreen.get("empty", [])
        logger.info(
            "Task %d: 帧预筛完成: 正脸帧 %d, 侧脸帧 %d, 无人脸帧 %d",
            task_id, len(frontal_frames), len(side_frames), len(empty_frames),
        )

        # 阶段4b: 正脸帧优先完整分析（先交付可用结果）
        faces = []
        if frontal_frames:
            faces = self.extract_faces_from_frames(
                str(temp_frames_dir), det_thresh=self._stage_det_thresh(),
                frame_filter=frontal_frames,
            )

        # 保底：正脸帧未检出人脸时，降低阈值重试
        if not faces and frontal_frames:
            logger.warning("Task %d: 正脸帧标准检测未找到人脸，降低阈值重试...", task_id)
            faces = self.extract_faces_from_frames(
                str(temp_frames_dir), det_thresh=0.05, clarity_threshold=0.005,
                frame_filter=frontal_frames,
            )

        logger.info(
            "Task %d: 正脸帧分析完成，%d 张人脸 (%.1fs)",
            task_id, len(faces), time.time() - t_detect,
        )

        # 注意：不在此处清理临时帧目录——侧脸补分析（阶段6）还需要用帧。
        # 清理移到侧脸补分析完成后。

        if not faces:
            logger.info("Task %d: 未检测到人脸，跳过识别", task_id)
            # 正脸帧无结果时，尝试侧脸帧（避免颗粒无收）
            if side_frames:
                logger.info("Task %d: 正脸帧无人脸，改用侧脸帧分析...", task_id)
                faces = self.extract_faces_from_frames(
                    str(temp_frames_dir), det_thresh=0.05, clarity_threshold=0.005,
                    frame_filter=side_frames,
                )
                if faces:
                    logger.info("Task %d: 侧脸帧检出 %d 张人脸", task_id, len(faces))
            if not faces:
                self._cleanup_temp_dir(temp_frames_dir)
                return _finish([])

        logger.info("Task %d: 检测到 %d 张人脸 (%.1fs)", task_id, len(faces), time.time() - t_detect)

        # 全片人脸中位清晰度：作为可用性闸门的自适应基准
        # （软画质视频全片 sharpness 都低，固定阈值会误杀真人，见 task1555 复盘）
        _video_med_sharp = None
        try:
            _sharp_all = [f.get("sharpness", 0) for f in faces if f.get("sharpness")]
            if _sharp_all:
                _video_med_sharp = float(np.median(_sharp_all))
                logger.info("Task %d: 全片中位清晰度 %.1f", task_id, _video_med_sharp)
        except Exception:
            pass

        # ── 阶段5: 聚类+匹配人物 ──
        from src.core.config import ConfigManager
        _config = ConfigManager()
        _distance_threshold = _config.get("face_recognition.distance_threshold", DEFAULT_DISTANCE_THRESHOLD)
        _iterations = _config.get("face_recognition.iterations", 20)
        _min_cluster_size = _config.get("face_recognition.min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)
        _merge_threshold = _config.get("face_recognition.merge_threshold", DEFAULT_MERGE_THRESHOLD)

        # 动态参数/自监督参数开关（config 驱动，默认开）
        _use_dynamic = bool(_config.get("face_recognition.dynamic_params.enabled", True))
        _use_auto = bool(_config.get("face_recognition.auto_params.enabled", False))

        t_cluster = time.time()
        clusterer = FaceClusterer(
            distance_threshold=_distance_threshold,
            use_dynamic_params=_use_dynamic,
            use_auto_params=_use_auto,
            track_config=self._track_grouping_config(),
        )
        # 收集视频特征喂给动态参数选择（时长/场景数/人脸数/密度）
        if _use_dynamic or _use_auto:
            try:
                _dur = 0.0
                try:
                    import cv2 as _cv2f
                    _capf = _cv2f.VideoCapture(video_path)
                    if _capf.isOpened():
                        _dur = _capf.get(_cv2f.CAP_PROP_FRAME_COUNT) / max(_capf.get(_cv2f.CAP_PROP_FPS) or 30.0, 1.0)
                    _capf.release()
                except Exception:
                    pass
                clusterer.set_video_features({
                    'duration': _dur or _est_scenes * 10.0,
                    'n_faces': len(faces),
                    'n_scenes': _est_scenes,
                    'face_density': len(faces) / _dur if _dur > 0 else 0.0,
                })
            except Exception as _vf_exc:
                logger.warning("Task %d: 视频特征收集失败，使用静态参数: %s", task_id, _vf_exc)
        clusters = clusterer.cluster_faces(faces)

        # 自适应 min_cluster_size：人脸总量少的稀疏视频放宽门槛，
        # 否则真人视频颗粒无收——task1556 复盘：25 张脸全部碎在 <10 的小簇里
        # 修改：使用配置值作为下限，不再强制 max(4, ...)
        _effective_min = min(_min_cluster_size, max(2, len(faces) // 15))

        if not clusters:
            logger.info(
                "Task %d: 从 %d 张人脸中未形成聚类，转入侧脸帧兜底分析",
                task_id, len(faces),
            )
            # 不 return：正脸不足时侧脸帧(阶段6)仍可能产出人物(如 task592 复盘：
            # 正脸23帧仅3张脸各自成簇被过滤，侧脸142帧大量人脸被此 return 浪费)

        # min_cluster_size 过滤（_effective_min 已在聚类阶段按视频自适应）
        valid_clusters = [c for c in clusters if len(c) >= _effective_min]
        dropped = len(clusters) - len(valid_clusters)
        if dropped > 0:
            logger.info(
                "Task %d: 丢弃 %d 个小聚类（<%d张脸，配置%d）",
                task_id, dropped, _effective_min, _min_cluster_size,
            )
        clusters = valid_clusters

        if not clusters:
            logger.info(
                "Task %d: 正脸聚类过滤后无有效聚类，转入侧脸帧兜底分析",
                task_id,
            )
            # 不 return：见上方注释，侧脸帧(阶段6)兜底仍可能产出人物

        logger.info("Task %d: 聚类完成，%d 个有效聚类 (%.1fs)", task_id, len(clusters), time.time() - t_cluster)

        # ── 聚类质量信号：低置信结果标记，供下游人工复核 ──
        # 不静默出错：embedding 质量差（低画质）时聚类结果不可靠，
        # 在 FileAnalysis 上标记 face_scan_status='low_confidence'(与主流程一致,
        # 不再写 tasks 表——历史 bug: tasks 残留脏值, 首页状态以 file_analysis 为准)。
        _cluster_confidence = getattr(clusterer, "confidence", None)
        _cluster_tier = getattr(clusterer, "quality_tier", None)
        _low_confidence = _cluster_confidence is not None and _cluster_confidence < 0.5
        if _low_confidence:
            logger.warning(
                "Task %d: 聚类质量低 (tier=%s, 置信度=%.2f)，结果需人工复核",
                task_id, _cluster_tier, _cluster_confidence,
            )
            try:
                if isinstance(db_session, sqlite3.Connection):
                    db_session.execute(
                        "UPDATE file_analysis SET face_scan_status='low_confidence' "
                        "WHERE task_id=?",
                        (task_id,),
                    )
                    db_session.commit()
                elif db_session is not None:
                    from src.db.models import FileAnalysis as _FAM
                    _fa_row = db_session.query(_FAM).filter(
                        _FAM.task_id == task_id
                    ).first()
                    if _fa_row is not None:
                        _fa_row.face_scan_status = 'low_confidence'
                        db_session.commit()
            except Exception as _lc_exc:
                logger.warning("Task %d: 写入 low_confidence 标记失败: %s", task_id, _lc_exc)

        # 解析 entity_id
        entity_id = None
        try:
            import sqlite3
            if isinstance(db_session, sqlite3.Connection):
                cur = db_session.cursor()
                cur.execute("SELECT entity_id FROM tasks WHERE id=?", (task_id,))
                row = cur.fetchone()
                if row:
                    entity_id = row[0]
            else:
                from src.db.models import Task
                if db_session is not None:
                    task_row = db_session.query(Task).filter(Task.id == task_id).first()
                    if task_row:
                        entity_id = task_row.entity_id  # noqa: F841 — 历史遗留：解析结果未再使用
        except Exception as ent_exc:
            logger.warning("Task %d: 解析 entity_id 失败: %s", task_id, ent_exc)

        # 跨文件归一开关：关闭时跳过 match_actor，每个视频的聚类独立成新人物。
        # （单文件优先策略：跨文件匹配尚未校准，误匹配会污染库人物）
        # 默认 True（与 config.yaml 的 match_existing_actors: true 一致）：
        # 缺失配置时若默认 False，侧脸补全阶段会重复新建同一人物的 actor。
        try:
            _match_existing = bool(_config.get("face_recognition.match_existing_actors", True))
        except Exception:
            _match_existing = False

        # 为每个聚类匹配或创建人物
        actor_ids = []
        cluster_actor_map = {}
        _created_ids = set()  # 本次运行新建的人物id（post-merge 方向判定用）
        for cluster_idx, cluster_faces in enumerate(clusters):

            actor_id = None
            if _match_existing:
                actor_id = self.match_actor_v2(cluster_faces, db_session=db_session)
            if actor_id:
                self.update_actor(actor_id, cluster_faces, task_id, file_md5, db_session=db_session)
                logger.info("Task %d: 聚类%d 匹配已有人物 %d (%d张脸)", task_id, cluster_idx+1, actor_id, len(cluster_faces))
            else:
                actor_id = self.create_actor(
                    cluster_faces, task_id, file_md5,
                    db_session=db_session,
                    video_median_sharpness=_video_med_sharp,
                )
                if actor_id is not None:
                    logger.info("Task %d: 聚类%d 创建新人物 %d (%d张脸)", task_id, cluster_idx+1, actor_id, len(cluster_faces))
                    _created_ids.add(actor_id)
                else:
                    logger.info("Task %d: 聚类%d 跳过（质量不达标）", task_id, cluster_idx+1)

            if actor_id is not None:
                cluster_actor_map[cluster_idx] = actor_id
                if actor_id not in actor_ids:
                    actor_ids.append(actor_id)

            # C：每个聚类（人物）提交一次，避免整段识别共用一个长写事务
            _maybe_commit(db_session, task_id, f"cluster{cluster_idx + 1}")

        # post-clustering merge
        # 改进：使用最大样本相似度（max-cross）而不是质心相似度
        # 这样可以更好地处理同一个人被拆成多个簇的情况
        if len(clusters) > 1 and _merge_threshold > 0:
            merged_indices = set()
            for i in range(len(clusters)):
                if i in merged_indices:
                    continue
                for j in range(i + 1, len(clusters)):
                    if j in merged_indices:
                        continue
                    
                    # 计算最大样本相似度（max-cross）
                    # 对两个簇的所有样本对计算相似度，取最大值
                    max_sim = 0.0
                    for face_i in clusters[i]:
                        emb_i = np.array(face_i['embedding'])
                        norm_i = np.linalg.norm(emb_i)
                        if norm_i == 0:
                            continue
                        emb_i = emb_i / norm_i
                        
                        for face_j in clusters[j]:
                            emb_j = np.array(face_j['embedding'])
                            norm_j = np.linalg.norm(emb_j)
                            if norm_j == 0:
                                continue
                            emb_j = emb_j / norm_j
                            
                            sim = float(np.dot(emb_i, emb_j))
                            if sim > max_sim:
                                max_sim = sim
                    
                    if max_sim > _merge_threshold:
                        actor_i = cluster_actor_map.get(i)
                        actor_j = cluster_actor_map.get(j)
                        if actor_i and actor_j and actor_i != actor_j:
                            # 合并方向：优先保留运行前已存在的库人物（code/封面/历史稳定），
                            # 把本次新建的人物并进去（task1555 复盘：曾把库人物3误并进新人物）
                            source_actor_id, target_actor_id = actor_j, actor_i
                            if actor_i in _created_ids and actor_j not in _created_ids:
                                source_actor_id, target_actor_id = actor_i, actor_j
                            logger.info(
                                "Task %d: post-merge 人物%d→%d (聚类%d/%d, max-cross相似度%.4f)",
                                task_id, source_actor_id, target_actor_id, j, i, max_sim,
                            )
                            try:
                                self._merge_actors(
                                    source_actor_id=source_actor_id,
                                    target_actor_id=target_actor_id,
                                    source_faces=clusters[j] if source_actor_id == actor_j else clusters[i],
                                    task_id=task_id,
                                    file_md5=file_md5,
                                    db_session=db_session,
                                )
                                merged_indices.add(j)
                                if source_actor_id in actor_ids:
                                    actor_ids.remove(source_actor_id)
                                for k, v in list(cluster_actor_map.items()):
                                    if v == source_actor_id:
                                        cluster_actor_map[k] = target_actor_id
                            except Exception as merge_exc:
                                logger.warning("Task %d: post-merge 失败: %s", task_id, merge_exc)
                            _maybe_commit(db_session, task_id, "post-merge")

        # 任务-人物关联（审核页按此展示视频的人物列表）
        try:
            from src.db.models import TaskActor as _TA
            for aid in actor_ids:
                if not db_session.query(_TA).filter(
                    _TA.task_id == task_id, _TA.actor_id == aid
                ).first():
                    db_session.add(_TA(task_id=task_id, actor_id=aid))
        except Exception as ta_exc:
            logger.warning("Task %d: 写入任务-人物关联失败: %s", task_id, ta_exc)
        _maybe_commit(db_session, task_id, "task-actors")

        # ── 阶段6: 侧脸帧分析（兜底/补全，渐进式呈现）──
        # 正脸帧已产出 actor 并交付；侧脸帧数据不丢弃，补分析后并入已有 actor。
        # 正脸无有效聚类时(如 task592：人脸少/侧脸为主)，侧脸帧作为兜底主分析，
        # 独立聚类并新建人物——历史 bug：正脸聚类为空直接 return 浪费侧脸帧。
        if side_frames:
            t_side = time.time()
            side_faces = []
            try:
                side_faces = self.extract_faces_from_frames(
                    str(temp_frames_dir), det_thresh=self._stage_det_thresh(),
                    frame_filter=side_frames,
                )
            except Exception as side_exc:
                logger.warning("Task %d: 侧脸帧分析失败: %s", task_id, side_exc)

            if side_faces:
                logger.info(
                    "Task %d: 侧脸帧分析完成，%d 张人脸 (%.1fs)，开始并入已有人物",
                    task_id, len(side_faces), time.time() - t_side,
                )
                # 侧脸并入：对侧脸簇做聚类，匹配到已有 actor 则并入，否则新建
                try:
                    _side_dur = 0.0
                    try:
                        _side_dur = _dur if _dur and _dur > 0 else _est_scenes * 10.0
                    except Exception:
                        _side_dur = _est_scenes * 10.0
                    _side_clusterer = FaceClusterer(
                        distance_threshold=_distance_threshold,
                        use_dynamic_params=_use_dynamic,
                        use_auto_params=_use_auto,
                        track_config=self._track_grouping_config(),
                    )
                    if _use_dynamic or _use_auto:
                        _side_clusterer.set_video_features({
                            'duration': _side_dur,
                            'n_faces': len(side_faces),
                            'n_scenes': _est_scenes,
                            'face_density': len(side_faces) / _side_dur if _side_dur > 0 else 0.0,
                        })
                    side_clusters = _side_clusterer.cluster_faces(side_faces)
                    # 兜底段门槛与主流程一致(min_cluster_size 配置, 默认3):
                    # 历史 bug: 这里单独算 max(2, n//15) → 81 张脸时门槛=5,
                    # 把 4 张的确认簇滤掉(主流程同场景只要求≥3), 兜底形同虚设。
                    _side_min_size = min(
                        _min_cluster_size, max(2, len(side_faces) // 15)
                    )
                    side_clusters = [c for c in side_clusters if len(c) >= _side_min_size]

                    # 同视频内正脸/侧脸合并：先只在「本任务已建立/已并入的 actor」里匹配
                    # （同一视频同一 session 刚产出，同人侧脸并入零跨文件误并风险）。
                    # 匹配不到再按 match_existing_actors 决定是否扩大到全库已有 actor。
                    # 默认 True（与 config.yaml 一致）：缺失配置时若为 False，
                    # 侧脸簇不会匹配正脸阶段刚建的同一人物 → 重复新建 actor
                    # （如 task 1483: 正脸 ACTOR_319 与侧脸 ACTOR_321 完全同帧同脸）。
                    _side_match = bool(_config.get("face_recognition.match_existing_actors", True))
                    for _sc_idx, _sc in enumerate(side_clusters):
                        _aid = None
                        if actor_ids:
                            _aid = self.match_actor_v2(
                                _sc, db_session=db_session, scope_actor_ids=list(actor_ids),
                            )
                        if not _aid and _side_match:
                            _aid = self.match_actor_v2(_sc, db_session=db_session)
                        if _aid:
                            self.update_actor(_aid, _sc, task_id, file_md5, db_session=db_session)
                            logger.info(
                                "Task %d: 侧脸簇%d 并入人物 %d (%d张脸)",
                                task_id, _sc_idx + 1, _aid, len(_sc),
                            )
                            if _aid not in actor_ids:
                                actor_ids.append(_aid)
                        else:
                            _aid = self.create_actor(
                                _sc, task_id, file_md5,
                                db_session=db_session,
                                video_median_sharpness=_video_med_sharp,
                            )
                            if _aid is not None:
                                logger.info(
                                    "Task %d: 侧脸簇%d 创建新人物 %d (%d张脸)",
                                    task_id, _sc_idx + 1, _aid, len(_sc),
                                )
                                # 新人物也补任务关联
                                try:
                                    from src.db.models import TaskActor as _TA2
                                    if not db_session.query(_TA2).filter(
                                        _TA2.task_id == task_id, _TA2.actor_id == _aid
                                    ).first():
                                        db_session.add(_TA2(task_id=task_id, actor_id=_aid))
                                except Exception:
                                    pass
                                if _aid not in actor_ids:
                                    actor_ids.append(_aid)
                        # C：每个侧脸簇提交一次，缩短写事务
                        _maybe_commit(db_session, task_id, f"side{_sc_idx + 1}")
                except Exception as side_cluster_exc:
                    logger.warning("Task %d: 侧脸聚类并入失败: %s", task_id, side_cluster_exc)

        # 清理临时帧目录（两阶段全部完成）
        try:
            self._cleanup_temp_dir(temp_frames_dir)
        except Exception as cleanup_exc:
            logger.warning("Task %d: 清理临时帧目录失败: %s", task_id, cleanup_exc)

        logger.info("Task %d: 识别完成，%d 个人物", task_id, len(actor_ids))

        # ── 最终 face_scan_status 统一写入 ──────────────────────
        # 历史 bug: 识别过程中途若聚类低置信会写 low_confidence(见上), 但识别
        # 完成后不再更新——结果「明明识别到人物, face_scan_status 却停留
        # low_confidence/空」, 前端人脸列不显示「已识别」。
        # 这里按最终结果覆盖: 有 actor → complete; 无 actor → none。
        try:
            _final_scan = "complete" if actor_ids else "none"
            if isinstance(db_session, sqlite3.Connection):
                db_session.execute(
                    "UPDATE file_analysis SET face_scan_status=? WHERE task_id=?",
                    (_final_scan, task_id),
                )
                db_session.commit()
            elif db_session is not None:
                from src.db.models import FileAnalysis as _FAM2
                _fa2 = db_session.query(_FAM2).filter(_FAM2.task_id == task_id).first()
                if _fa2 is not None:
                    _fa2.face_scan_status = _final_scan
                    db_session.commit()
            logger.info("Task %d: 人脸扫描状态 = %s (%d 人物)", task_id, _final_scan, len(actor_ids))
        except Exception as _scan_exc:
            logger.warning("Task %d: 写 face_scan_status 失败: %s", task_id, _scan_exc)

        return _finish(actor_ids)

    def _merge_actors(
        self,
        source_actor_id: int,
        target_actor_id: int,
        source_faces: List[Dict],
        task_id: int,
        file_md5: str = None,
        db_session=None,
    ) -> None:
        """合并两个人物：将 source 的所有人脸迁移到 target，然后删除 source。

        用于 post-clustering merge：当两个聚类被识别为同一人时，
        将两者合并。

        Args:
            source_actor_id: 被合并的人物ID（将被删除）
            target_actor_id: 目标人物ID（保留）
            source_faces: source 聚类的人脸列表
            task_id: 当前任务ID
            file_md5: 视频文件MD5
            db_session: 可选数据库会话
        """
        from src.db.models import Actor, ActorFace, TaskActor, get_session
        from src.db.models import get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            db = db_session

        try:
            source_actor = db.query(Actor).filter(Actor.id == source_actor_id).first()
            target_actor = db.query(Actor).filter(Actor.id == target_actor_id).first()
            if not source_actor or not target_actor:
                logger.warning(
                    "_merge_actors: 人物不存在 source=%d target=%d",
                    source_actor_id, target_actor_id,
                )
                return

            # 1. 迁移 source 的所有 ActorFace 到 target
            source_faces_db = (
                db.query(ActorFace)
                .filter(ActorFace.actor_id == source_actor_id)
                .all()
            )

            # 1.5 物理搬移文件到 target 目录（否则会留下悬空目录，
            # 且 target 的文件夹视图不完整）
            target_dir = self.faces_dir / target_actor.code
            source_dir = self.faces_dir / source_actor.code
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                # 起始编号取 target 目录现有文件 max+1（避免与残留文件撞名覆盖）
                _next_idx = self._next_face_index(target_dir)
                moved_files = 0
                skipped_files = 0
                for face in source_faces_db:
                    # 相对路径 → 绝对路径（统一经 _resolve_face_path 解析）
                    src = self._resolve_face_path(face.face_path)
                    if src is None or not src.exists():
                        logger.warning(
                            "_merge_actors: source 人脸文件不存在（仅迁移DB记录）: %s",
                            face.face_path,
                        )
                        continue
                    # 来源校验：只搬移确实位于 source actor 目录内的文件，
                    # 防止把其他人物目录的文件误搬进 target。
                    try:
                        src_resolved = src.resolve()
                        src_dir_resolved = source_dir.resolve()
                        belongs_to_source = (
                            src_resolved == src_dir_resolved
                            or src_resolved.is_relative_to(src_dir_resolved)
                        )
                    except Exception:
                        belongs_to_source = False
                    if not belongs_to_source:
                        logger.warning(
                            "_merge_actors: 跳过来源不在 source 目录的人脸: %s",
                            face.face_path,
                        )
                        skipped_files += 1
                        continue
                    # 目标去重：目标目录已存在同名文件（或同源文件已搬过）则跳过
                    if src.parent.resolve() == target_dir.resolve():
                        continue  # 已在 target 目录（理论上不发生）
                    dst = target_dir / f"face_{_next_idx}.jpg"
                    while dst.exists():
                        _next_idx += 1
                        dst = target_dir / f"face_{_next_idx}.jpg"
                    src.rename(dst)
                    # 存储相对路径
                    try:
                        face.face_path = str(dst.relative_to(self.faces_dir))
                    except ValueError:
                        face.face_path = dst.name
                    _next_idx += 1
                    moved_files += 1
                if moved_files or skipped_files:
                    logger.info(
                        "_merge_actors: 搬移 %d 个人脸文件 → %s（跳过 %d）",
                        moved_files, target_dir, skipped_files,
                    )
            except Exception as mv_exc:
                logger.warning(
                    "_merge_actors: 文件搬移失败（仅迁移DB记录）: %s", mv_exc
                )

            for face in source_faces_db:
                face.actor_id = target_actor_id
            migrated_count = len(source_faces_db)
            logger.info(
                "_merge_actors: 迁移 %d 张人脸 actor %d → %d",
                migrated_count, source_actor_id, target_actor_id,
            )

            # 2. 迁移 TaskActor 关联（处理唯一约束冲突）
            source_task_actors = (
                db.query(TaskActor)
                .filter(TaskActor.actor_id == source_actor_id)
                .all()
            )
            for ta in source_task_actors:
                existing = (
                    db.query(TaskActor)
                    .filter(
                        TaskActor.task_id == ta.task_id,
                        TaskActor.actor_id == target_actor_id,
                    )
                    .first()
                )
                if existing:
                    # 目标已有该关联，删除 source 的
                    db.delete(ta)
                else:
                    ta.actor_id = target_actor_id
            logger.info(
                "_merge_actors: 迁移 %d 条任务关联 actor %d → %d",
                len(source_task_actors), source_actor_id, target_actor_id,
            )

            # 3. 更新 target 的 embedding 和 sample_count
            # 用 DB 中 target 现存人脸 embedding 聚合（含刚迁移的），
            # 避免依赖内存里可能不完整的 source_faces 列表。
            try:
                _target_emb_rows = [
                    r[0] for r in db.query(ActorFace.embedding)
                    .filter(ActorFace.actor_id == target_actor_id,
                            ActorFace.embedding.isnot(None))
                    .all()
                ]
                if _target_emb_rows:
                    _emb_arr = np.array([decode_embedding(e) for e in _target_emb_rows])
                    target_actor.face_embedding = encode_embedding(np.mean(_emb_arr, axis=0))
            except Exception as emb_exc:
                logger.warning("_merge_actors: 重算 target embedding 失败: %s", emb_exc)
            self._refresh_sample_count(db, target_actor_id)

            # 4. 删除 source actor，并清理其磁盘目录（含孤儿残留），
            #    保证磁盘目录与 DB 一一对应。
            db.delete(source_actor)
            db.flush()
            self._remove_actor_dir(source_actor.code)
            logger.info(
                "_merge_actors: 删除 source 人物 %d，合并完成",
                source_actor_id,
            )

            if own_session:
                db.commit()
                db.close()

        except Exception as exc:
            logger.error(
                "_merge_actors failed (source=%d → target=%d): %s",
                source_actor_id, target_actor_id, exc,
            )
            if own_session:
                try:
                    db.rollback()
                except Exception:
                    pass
                try:
                    db.close()
                except Exception:
                    pass
            raise

    def _extract_embeddings_from_detection(self, detection_result, task_id: int, video_quality_info=None) -> List[Dict]:
        """从 FaceDetectionResult 中提取人脸 embedding。

        使用全帧检测+识别方式（一步到位），避免 crop+重检测导致的
        embedding 不一致问题。对每张 dense_frame：
        1. 读取全帧图像
        2. 用 InsightFace 全帧检测+识别，直接获取 embedding
        3. 保存人脸裁剪图
        4. 按 embedding norm 过滤低质量结果
        5. 返回标准格式的人脸字典列表

        Args:
            detection_result: FaceDetectionResult 对象
            task_id: 任务ID
            video_quality_info: VideoQualityInfo（可选），用于动态 sharpness 阈值

        Returns:
            List of dicts with keys: frame_path, face_path, embedding, yaw, pitch, roll, det_score, norm
        """
        all_faces: List[Dict] = []
        faces_processed = 0
        faces_failed = 0

        import gc

        # embedding 范数下限**不在此处过滤**：它需要"本片全部人脸的范数"才能自适应，
        # 故改为**收集完再过滤**（与清晰度一致，见 openspec unify-face-thresholds §3）。
        for df_idx, dense_frame in enumerate(detection_result.dense_frames):
            frame_path = dense_frame.frame_path
            if not frame_path or not Path(frame_path).exists():
                continue

            # 读取全帧图像
            try:
                img = cv2.imread(frame_path)
            except Exception as e:
                logger.warning(f"cv2.imread failed for dense frame: {frame_path}, error: {e}")
                continue
            if img is None:
                continue

            # 全帧检测+识别（一步到位，避免 crop+重检测导致 embedding 偏移）
            try:
                detected = self.app.get(img) if self.app else []
            except Exception as e:
                logger.warning("InsightFace 全帧检测失败 df%d: %s", df_idx, e)
                del img
                continue

            if not detected:
                del img
                continue

            for face_idx, face_obj in enumerate(detected):
                embedding = face_obj.embedding
                if embedding is None:
                    faces_failed += 1
                    continue

                # 计算 embedding norm（供收集后的自适应过滤使用）
                embedding_norm = float(np.linalg.norm(embedding))

                # 获取 bbox 和 pose（clamp 到图像边界，防止越界）
                h_img, w_img = img.shape[:2]
                bbox = face_obj.bbox.astype(int).tolist()
                x1, y1, x2, y2 = bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                if x2 <= x1 or y2 <= y1:
                    faces_failed += 1
                    continue

                w, h = x2 - x1, y2 - y1
                if w < 50 or h < 50:
                    logger.debug("跳过小人脸: %dx%d", w, h)
                    faces_failed += 1
                    continue

                # 获取 pose（复用统一方法，基于关键点估算）
                yaw, pitch, roll = self._extract_pose(face_obj)

                # 关键点几何验证：过滤后脑勺/皮肤/脚等非人脸误检。
                # 真脸的关键点布局稳定（双眼水平、鼻尖居中偏下、嘴角在下），
                # 身体部位误检的关键点布局退化，可通过几何检查排除。
                from src.core.face_utils import validate_face_landmarks
                _kps = getattr(face_obj, 'kps', None)
                _det_score = float(getattr(face_obj, 'det_score', 0.5))
                if not validate_face_landmarks(_kps, getattr(face_obj, 'bbox', None), det_score=_det_score):
                    logger.debug(
                        "df%d_f%d: 关键点几何不合法（疑似后脑勺/非人脸），跳过",
                        df_idx, face_idx,
                    )
                    faces_failed += 1
                    continue

                det_score = _det_score

                # 解析帧号（frame_N.jpg / supp_N.jpg → N）
                frame_id = None
                try:
                    _fp = Path(frame_path)
                    _m = re.match(r"^(?:frame|supp)_(\d+)", _fp.stem)
                    if _m:
                        frame_id = int(_m.group(1))
                except Exception:
                    pass

                # 裁剪并保存人脸图
                face_img = img[y1:y2, x1:x2]
                temp_name = f"temp_vdet_task{task_id}_df{df_idx}_f{face_idx}.jpg"
                temp_path = self.faces_dir / temp_name
                cv2.imwrite(str(temp_path), face_img)

                all_faces.append({
                    "frame_path": frame_path,
                    "face_path": str(temp_path),
                    "embedding": embedding,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "det_score": det_score,
                    "norm": embedding_norm,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "frame_id": frame_id,
                    "timestamp": None,
                    "sex": getattr(face_obj, "sex", None),
                    "age": getattr(face_obj, "age", None),
                    "kps": np.asarray(face_obj.kps, dtype=float).tolist() if hasattr(face_obj, 'kps') and face_obj.kps is not None else None,
                })

                faces_processed += 1

            del img

            if df_idx % 10 == 0:
                gc.collect()

        # 计算质量分数（复用现有方法）
        for face in all_faces:
            face['brightness'] = self._calculate_brightness(face['face_path'])
            face['sharpness'] = self._calculate_raw_sharpness(face['face_path'])
            face['quality_score'] = self.calculate_face_quality(face)

        # 质量过滤 — 改用 quality_score（包含姿态、亮度、清晰度综合评分）
        # 而不是原始 sharpness 值，避免大脸（近距离）因为像素多而占优势
        fqc = self._get_face_quality_config()
        if fqc.get("enabled", False):
            from src.core.config import ConfigManager
            from src.core.face_thresholds import build_profile, log_profile

            # 范数规则：上限取配置（低清按比例放宽），**下限**由本片范数分位自适应
            # （分位 ≥ p25，见 face_thresholds._embedding_norm_floor）
            norm_cfg = {
                "configured_high": ConfigManager().get(
                    "face_recognition.embedding_norm_min", 15.0),
                "absolute_low": fqc.get("norm_absolute_low", 8.0),
                "percentile": fqc.get("norm_adaptive_percentile", 25),
                "min_faces": fqc.get("adaptive_min_faces", 5),
                "enabled": fqc.get("adaptive_enabled", True),
                "low_quality_ratio": fqc.get("norm_low_quality_ratio", 0.7),
            }
            profile = build_profile(fqc, all_faces, video_quality_info, norm_config=norm_cfg)
            log_profile(profile, where="detection_embeddings", task=task_id)

            # 第一步：按 embedding 范数下限过滤退化特征（**收集后**才过滤 → 可自适应）
            if profile.min_embedding_norm > 0:
                before_norm = len(all_faces)
                kept = [f for f in all_faces
                        if float(f.get("norm") or 0.0) >= profile.min_embedding_norm]
                if len(kept) < before_norm:
                    logger.info(
                        "embedding 范数过滤: %d/%d 张人脸被丢弃 (下限=%.1f, 规则=%s)",
                        before_norm - len(kept), before_norm,
                        profile.min_embedding_norm, profile.basis.get("norm_rule"),
                    )
                    for f in all_faces:
                        if f not in kept:
                            try:
                                Path(f["face_path"]).unlink()
                            except Exception:
                                pass
                all_faces = kept
                if not all_faces:
                    logger.warning("所有人脸 embedding 范数均低于下限，无人脸保留")
                    return []

            # 第二步：过滤掉 sharpness 绝对值过低的人脸（可能是误检）
            # 阈值按视频质量动态调整（低质量视频用低阈值，避免误杀全部人脸）。
            # 对"眼睛睁开"的人脸（eyes_open_score > 0.5）放宽 sharpness 门槛：
            # 睁眼帧对封面有独特价值（用户期望封面为睁眼表情），
            # 即使因运动模糊 sharpness 略低也应保留进入封面候选。
            min_sharp = profile.sharpness_floor
            open_eyes_threshold = fqc.get("cover_eyes_open_threshold", 0.5)

            def _keep_for_cover(face: Dict) -> bool:
                if face.get("sharpness", 0) >= min_sharp:
                    return True
                # 睁眼但略模糊 → 保留（供封面选择），但需要达到绝对下限的一半
                eyes = face.get("eyes_open_score", 0)
                if eyes > open_eyes_threshold:
                    return face.get("sharpness", 0) >= min_sharp * 0.5
                return False

            valid_faces = [f for f in all_faces if _keep_for_cover(f)]
            if len(valid_faces) < len(all_faces):
                removed = len(all_faces) - len(valid_faces)
                logger.info(
                    f"sharpness 绝对值过滤: {removed}/{len(all_faces)} 张人脸被丢弃 "
                    f"(min_sharpness={min_sharp:.1f})"
                )
                for f in all_faces:
                    if f not in valid_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception:
                            pass
            all_faces = valid_faces

            if not all_faces:
                logger.warning("质量过滤: 所有人脸 sharpness 均低于阈值，无人脸保留")
                return []

            # 第二步：使用 quality_score 过滤
            min_quality = fqc.get("min_quality", 0.3)
            min_keep = fqc.get("min_faces_to_keep", 3)
            good_faces = [f for f in all_faces if f.get("quality_score", 0) >= min_quality]
            if len(good_faces) >= min_keep:
                removed = len(all_faces) - len(good_faces)
                if removed > 0:
                    logger.info(
                        f"质量过滤: {removed}/{len(all_faces)} 张人脸被过滤 "
                        f"(阈值={min_quality:.2f})"
                    )
                all_faces = good_faces
            else:
                # 高质量人脸不足，按 quality_score 排序保留最好的 min_keep 张
                # 但仍然要求 sharpness >= min_sharp 以防止极低质量人脸被保留
                sorted_faces = sorted(all_faces, key=lambda f: f.get("quality_score", 0), reverse=True)
                kept = sorted_faces[:max(min_keep, 1)]
                # 二次 sharpness 检查（同样用自适应下限）
                min_sharp = self._adaptive_sharpness_floor(fqc, all_faces, video_quality_info)
                final_kept = [f for f in kept if f.get("sharpness", 0) >= min_sharp]
                if not final_kept:
                    # 没有符合 sharpness 要求的，保留质量最好的那张
                    final_kept = kept[:1]
                    logger.warning(
                        f"质量过滤（容错模式）: sharpness 均低于 {min_sharp:.1f}，仅保留最佳 1 张"
                    )
                removed = len(all_faces) - len(final_kept)
                logger.info(
                    f"质量过滤（容错模式）: {removed}/{len(all_faces)} 张被过滤，"
                    f"仅 {len(final_kept)} 张人脸保留"
                )
                all_faces = final_kept

            # 第三步：睁眼验证（仅低分辨率视频）——过滤后脑勺等非人脸误检。
            # 低清视频（360p/480p）小人脸常被误检为"人脸"，但后脑勺/皮肤等
            # 误检的 eyes_open_score 偏低（0.2-0.8），真脸通常 >0.9。
            # 高清视频不做此过滤（侧面/闭眼的真脸 eyes 可能偏低）。
            _res = getattr(video_quality_info, "resolution", None) if video_quality_info is not None else None
            _low_res = bool(_res and len(_res) == 2 and (_res[0] * _res[1]) < 230400)
            if _low_res and all_faces:
                min_eyes = fqc.get("low_res_min_eyes_open", 0.9)
                valid_eyes = [f for f in all_faces if f.get("eyes_open_score", 0) >= min_eyes]
                if valid_eyes:
                    removed_eyes = len(all_faces) - len(valid_eyes)
                    if removed_eyes > 0:
                        logger.info(
                            f"睁眼验证过滤（低分辨率）: {removed_eyes}/{len(all_faces)} 张人脸被丢弃 "
                            f"(eyes_open < {min_eyes})——疑似后脑勺/非人脸误检"
                        )
                        for f in all_faces:
                            if f not in valid_eyes:
                                try:
                                    Path(f["face_path"]).unlink()
                                except Exception:
                                    pass
                    all_faces = valid_eyes
                else:
                    # 没有可靠人脸（eyes 全部低于阈值）→ 返回空，不保留误检
                    logger.warning(
                        f"睁眼验证过滤（低分辨率）: 所有人脸 eyes_open_score < {min_eyes}，"
                        f"判定无可靠人脸，丢弃全部"
                    )
                    for f in all_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception:
                            pass
                    return []

        # 正脸过滤（复用现有逻辑）
        if self.FRONTAL_ONLY and all_faces:
            frontal_faces = [
                f for f in all_faces
                if (abs(f["yaw"]) < self.YAW_THRESHOLD
                    and abs(f["pitch"]) < self.PITCH_THRESHOLD
                    and abs(f["roll"]) < self.ROLL_THRESHOLD)
            ]
            if frontal_faces:
                logger.info("Frontal filter: %d/%d faces passed", len(frontal_faces), len(all_faces))
                for f in all_faces:
                    if f not in frontal_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception as e:
                            logger.warning("Failed to delete non-frontal face: %s", e)
                return frontal_faces
            else:
                logger.warning("No frontal faces found, using %d closest", min(3, len(all_faces)))
                sorted_faces = sorted(all_faces, key=lambda f: abs(f["yaw"]) + abs(f["pitch"]) + abs(f["roll"]))
                best_faces = sorted_faces[:min(3, len(sorted_faces))]
                for f in all_faces:
                    if f not in best_faces:
                        try:
                            Path(f["face_path"]).unlink()
                        except Exception as e:
                            logger.warning("Failed to delete non-best face: %s", e)
                return best_faces

        logger.info(
            "Task %d: _extract_embeddings_from_detection 完成 - 成功 %d, 失败 %d",
            task_id, faces_processed, faces_failed
        )
        return all_faces


    @staticmethod
    def _cleanup_temp_dir(dir_path: Path):
        """清理临时目录及其内容。"""
        try:
            import shutil
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.debug("临时目录已清理: %s", dir_path)
        except Exception as e:
            logger.warning("清理临时目录失败: %s, %s", dir_path, e)


    # ------------------------------------------------------------------
    # 文件路径 / 目录一致性辅助
    #
    # actor_faces.face_path 统一存相对 faces_dir 的路径（如 ACTOR_001/face_0.jpg）。
    # 任何"按 DB 记录删文件/查文件"的代码都必须经 _resolve_face_path 解析，
    # 否则相对路径会落在进程 CWD 下找不到文件 → 磁盘孤儿文件累积
    # （历史 bug：rebuild 后目录与 DB 记录严重不一致）。
    # ------------------------------------------------------------------

    def _resolve_face_path(self, face_path: Optional[str]) -> Optional[Path]:
        """把 DB 里的 face_path 解析为磁盘绝对路径。

        兼容三种形态：
        - 相对 faces_dir：'ACTOR_001/face_0.jpg'
        - 相对项目根的旧数据：'data/actor_faces/ACTOR_001/face_0.jpg'
        - 绝对路径

        Args:
            face_path: DB 存储的路径

        Returns:
            解析后的 Path，无法解析/为空时返回 None
        """
        if not face_path:
            return None
        p = Path(face_path).expanduser()
        if p.is_absolute():
            return p
        # 先试 faces_dir 相对（当前存储规范）
        candidate = self.faces_dir / p
        if candidate.exists():
            return candidate
        # 再试项目根相对（兼容历史绝对→相对迁移残留）
        root_candidate = Path(__file__).resolve().parent.parent.parent / p
        if root_candidate.exists():
            return root_candidate
        # 都不存在时按当前规范返回（调用方据此 unlink，unlink 对不存在文件安全）
        return candidate

    @staticmethod
    def _safe_unlink(path: Optional[Path]) -> bool:
        """安全删除文件（不存在不报错）。返回是否实际删除。"""
        if path is None:
            return False
        try:
            if path.exists():
                path.unlink()
                return True
        except Exception as e:
            logger.warning("删除文件失败: %s, %s", path, e)
        return False

    def _next_face_index(self, actor_dir: Path) -> int:
        """返回 actor 目录中下一个可用的 face_N.jpg 编号（磁盘现有最大 +1）。

        用于 update_actor / _merge_actors 追加文件时避免与目录残留文件撞名
        （历史 bug：按 DB count 编号，rebuild 残留的孤儿文件会被覆盖/错位）。

        Args:
            actor_dir: 人物专属目录

        Returns:
            下一个可用编号（>=0）。目录不存在或为空时返回 0。
        """
        max_idx = -1
        if actor_dir.exists():
            for p in actor_dir.glob("face_*.jpg"):
                try:
                    max_idx = max(max_idx, int(p.stem.split("_")[1]) + 1)
                except (IndexError, ValueError):
                    continue
        return max_idx + 1

    def _remove_actor_dir(self, actor_code: str) -> None:
        """删除人物磁盘目录（含残留文件）。

        用于人物被删除/合并后清理磁盘，保证目录与 DB 一一对应。
        只删除 faces_dir 下的 ACTOR_xxx 目录，杜绝误删。
        """
        if not actor_code:
            return
        actor_dir = self.faces_dir / actor_code
        try:
            # 防御：必须位于 faces_dir 下且名字形如 ACTOR_xxx
            if (self.faces_dir / actor_code).resolve().is_relative_to(self.faces_dir.resolve()):
                import shutil
                if actor_dir.exists():
                    shutil.rmtree(actor_dir)
                    logger.info("已删除人物目录: %s", actor_dir)
            else:
                logger.error("拒绝删除 faces_dir 外的目录: %s", actor_dir)
        except Exception as e:
            logger.warning("删除人物目录失败: %s, %s", actor_dir, e)

    def _sync_actor_dir(
        self,
        db,
        actor_code: str,
        keep_paths: Optional[set] = None,
    ) -> int:
        """把人物目录清理到与 DB 记录一一对应（删除未被 DB 引用的 face_*.jpg）。

        用于 rebuild/合并等操作后收敛目录残留。只处理 DB 中仍存在该 actor 的情况；
        对已删除的 actor，调用方应先 _remove_actor_dir。

        Args:
            db: 数据库会话
            actor_code: 人物代码（如 ACTOR_001）
            keep_paths: 可选。额外保留的绝对路径集合（如本次新建尚未落库的文件）

        Returns:
            删除的孤儿文件数量
        """
        actor_dir = self.faces_dir / actor_code
        if not actor_dir.exists():
            return 0

        from src.db.models import Actor, ActorFace
        actor = db.query(Actor).filter(Actor.code == actor_code).first()
        if actor is None:
            return 0

        db_rel_paths = {
            r[0]
            for r in db.query(ActorFace.face_path).filter(ActorFace.actor_id == actor.id).all()
            if r[0]
        }
        deleted = 0
        for p in actor_dir.glob("face_*.jpg"):
            try:
                rel = str(p.relative_to(self.faces_dir))
            except ValueError:
                rel = p.name
            if rel in db_rel_paths:
                continue
            if keep_paths and p.resolve() in keep_paths:
                continue
            if self._safe_unlink(p):
                deleted += 1
        if deleted:
            logger.info(
                "目录一致性清理 %s: 删除 %d 个孤儿文件",
                actor_code, deleted,
            )
        return deleted

    def _sweep_temp_files(self) -> int:
        """清理 faces_dir 根目录下的临时人脸产物。

        两类残留：
        1. temp_*.jpg —— 检测阶段产生、最终未进入任何 actor 的中间裁剪图
           （聚类被过滤/建人物被闸门拦截/中断）
        2. .rebuild_t*_ 目录 —— rebuild_actor_faces 的任务私有裁剪目录，
           正常退出会自清理，仅异常/进程被杀时遗留（兜底）

        Returns:
            删除的文件数量
        """
        if not self.faces_dir.exists():
            return 0
        deleted = 0
        for p in self.faces_dir.glob("temp_*.jpg"):
            if self._safe_unlink(p):
                deleted += 1
        # 兜底清理 rebuild 私有裁剪目录残留
        for d in self.faces_dir.glob(".rebuild_t*"):
            try:
                if d.is_dir():
                    import shutil as _shutil_sweep
                    _shutil_sweep.rmtree(d, ignore_errors=True)
                    deleted += 1
            except Exception:
                pass
        if deleted:
            logger.info("清理临时人脸文件 %d 个: %s", deleted, self.faces_dir)
        return deleted

    def _refresh_sample_count(self, db, actor_id: int) -> None:
        """按 actor_faces 实际行数回写 actor.sample_count。"""
        try:
            from src.db.models import Actor, ActorFace
            actor = db.query(Actor).filter(Actor.id == actor_id).first()
            if actor is None:
                return
            actor.sample_count = db.query(ActorFace).filter(
                ActorFace.actor_id == actor_id
            ).count()
        except Exception as e:
            logger.warning("刷新 sample_count 失败 actor_id=%d: %s", actor_id, e)

    def _max_samples_per_actor(self) -> int:
        """每个人物每次(新建/并入)保留的代表脸数量上限(config 可调, 默认 5)。

        业务模型: 视频识别产出的是「参与的人物 + 每人物少量高质量代表脸」,
        不需要把聚类出的每一张脸都入库(历史: 单个 actor 存几十~几百张)。
        Top-N 张代表脸用于跨文件归集(embedding 匹配), 5 张覆盖多姿态足够。
        """
        try:
            from src.core.config import ConfigManager as _CfgMgr
            return max(1, int(_CfgMgr().get("face_recognition.max_samples_per_actor", 5)))
        except Exception:
            return 5

    def _select_representative_faces(self, faces: List[Dict], n: int) -> List[Dict]:
        """从簇内人脸精选 n 张代表脸(质量优先 + 姿态多样)。

        排序键 quality_score(降序, 缺失按 0); 同分时正脸(|yaw| 小)优先。
        姿态分桶贪心: yaw 分 3 桶(左/中/右), 每桶最多 ceil(n/3),
        按质量从高到低逐张选入, 保证代表脸覆盖不同角度(利于跨文件匹配)。

        Args:
            faces: 簇内人脸 dict 列表(已含 quality_score/yaw 等)
            n: 期望保留数量

        Returns:
            精选后的子列表(保持原引用; 若 len(faces)<=n 原样返回)
        """
        if len(faces) <= n:
            return faces

        def _score_key(f):
            q = f.get("quality_score") or 0
            yaw = abs(f.get("yaw", 0)) or 0
            return (q, -yaw)  # 质量高优先, yaw 小(正脸)优先

        ranked = sorted(faces, key=_score_key, reverse=True)
        per_bucket = max(1, (n + 2) // 3)  # ceil(n/3)
        buckets: Dict[int, List[Dict]] = {-1: [], 0: [], 1: []}

        def _bucket(f) -> int:
            yaw = f.get("yaw", 0) or 0
            if yaw < -15:
                return -1
            if yaw > 15:
                return 1
            return 0

        chosen: List[Dict] = []
        # 第一轮: 每个姿态桶按质量选 per_bucket 张(保证跨姿态覆盖)
        for f in ranked:
            b = _bucket(f)
            if len(buckets[b]) < per_bucket:
                buckets[b].append(f)
        chosen = buckets[-1] + buckets[0] + buckets[1]
        # 第二轮: 候选不足 n 时, 从未入选的高质量脸补齐
        if len(chosen) < n:
            chosen_ids = {id(f) for f in chosen}
            for f in ranked:
                if len(chosen) >= n:
                    break
                if id(f) not in chosen_ids:
                    chosen.append(f)
                    chosen_ids.add(id(f))
        # 超出 n 则按质量截断
        if len(chosen) > n:
            chosen = sorted(chosen, key=_score_key, reverse=True)[:n]
        logger.info(
            "精选代表脸: %d → %d 张 (质量优先+姿态多样, Top-%d)",
            len(faces), len(chosen), n,
        )
        return chosen

    def _trim_actor_to_max(self, db, actor_id: int) -> int:
        """把 actor 的人脸收敛到 Top-N 代表脸(删除多余记录+文件)。

        业务模型: 每人物只保留少量高质量代表脸。同一 actor 在一视频内
        经多次并入(正脸簇/侧脸簇/merge)会累积超过上限(如 764 的 ACTOR_248
        累积 24 张), 视频识别完成时做最终收敛。

        Args:
            db: SQLAlchemy session
            actor_id: 人物 ID

        Returns:
            删除的脸数
        """
        try:
            from src.db.models import ActorFace
            _max_n = self._max_samples_per_actor()
            rows = db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
            if len(rows) <= _max_n:
                return 0
            # 构造临时 dict 列表复用精选逻辑(质量+姿态多样)
            face_dicts = []
            for af in rows:
                try:
                    emb = decode_embedding(af.embedding)
                except Exception:
                    emb = None
                face_dicts.append({
                    "_af": af,
                    "quality_score": af.quality_score or 0,
                    "yaw": af.yaw or 0,
                    "pitch": af.pitch or 0,
                    "roll": af.roll or 0,
                    "embedding": emb,
                })
            keep = self._select_representative_faces(face_dicts, _max_n)
            keep_ids = {id(f["_af"]) for f in keep}
            removed = 0
            for af in rows:
                if id(af) in keep_ids:
                    continue
                self._safe_unlink(self._resolve_face_path(af.face_path))
                db.delete(af)
                removed += 1
            db.flush()
            # 封面与 sample_count 收敛(helper 内部处理封面)
            self._apply_actor_cover(db, actor_id)
            self._refresh_sample_count(db, actor_id)
            if removed:
                logger.info(
                    "收敛人物 %d 人脸: %d → %d (删除 %d)", actor_id,
                    len(rows), len(rows) - removed, removed,
                )
            return removed
        except Exception as e:
            logger.warning("收敛人物 %d 失败: %s", actor_id, e)
            return 0

    def _apply_actor_cover(
        self,
        db,
        actor_id: int,
        preferred_path: Optional[str] = None,
    ) -> Optional[str]:
        """原子地统一 actor.cover_face 与 actor_faces.is_cover。

        实现已抽到 `src/db/actor_cover.apply_actor_cover`（单一事实来源），
        供 API 层（合并/拆分/人工设封面）与识别路径共用，避免各处内联维护脱节。
        """
        from src.db.actor_cover import apply_actor_cover
        return apply_actor_cover(db, actor_id, self.faces_dir, preferred_path)


    # ------------------------------------------------------------------
    # Selective face rebuilding
    # ------------------------------------------------------------------

    def rebuild_task_faces(self, task_id: int, db_session=None) -> Dict:
        """重建指定任务的人脸数据。

        删除该任务旧的人脸记录，从帧文件重新提取人脸并识别人物。
        保留其他任务的人脸数据不受影响。

        Args:
            task_id: 要重建的任务ID。
            db_session: 可选数据库会话，不传则新建。

        Returns:
            字典，包含 task_id, actor_ids, faces_count, status, message
        """
        from src.db.models import Task, Actor, ActorFace, TaskActor, FileAnalysis, get_session, get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            db = db_session

        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                return {
                    "task_id": task_id,
                    "status": "error",
                    "message": f"任务 {task_id} 不存在",
                    "actor_ids": [],
                    "faces_count": 0,
                }

            # 删除旧的人脸记录（同时删除磁盘文件）。
            # 注意：face_path 存的是相对 faces_dir 的路径（如 ACTOR_001/face_0.jpg），
            # 必须经 _resolve_face_path 解析后再删；直接用 Path(face_path) 会落在
            # 进程 CWD 下找不到文件 → 只删 DB 不删文件 → 目录孤儿累积。
            old_faces = db.query(ActorFace).filter(ActorFace.task_id == task_id).all()
            affected_actor_ids = set()
            for old_face in old_faces:
                if old_face.actor_id is not None:
                    affected_actor_ids.add(old_face.actor_id)
                self._safe_unlink(self._resolve_face_path(old_face.face_path))
                db.delete(old_face)

            # 清理受影响人物目录中本次任务可能遗留的孤儿文件
            # （历史重建残留 + 本任务刚删除的记录对应文件）
            for _aid in affected_actor_ids:
                _act = db.query(Actor).filter(Actor.id == _aid).first()
                if _act is not None:
                    self._sync_actor_dir(db, _act.code)

            db.query(TaskActor).filter(TaskActor.task_id == task_id).delete()
            task.actor_id = None
            db.flush()
            # 回写受删除影响的人物 sample_count
            for _aid in affected_actor_ids:
                self._refresh_sample_count(db, _aid)
            # 立即提交删除，释放写锁——避免长时间检测期间其他任务的
            # 写操作被 SQLite 单写锁阻塞（database is locked）。
            # 识别结果会在 recognize_actors_from_video 中用同一 session 写入，
            # 最后由外层统一提交。
            db.commit()

            file_md5 = task.file_md5
            video_path = task.file_path

            # 优先使用两阶段检测（从视频文件）
            actor_ids = []
            if video_path and Path(video_path).exists():
                # 分辨率检查：低于 min_resolution 像素门槛的低清视频跳过识别，
                # 并记录 face_scan_status='skipped'（区别于真实扫描后无人脸）。
                _skip_reason = None
                try:
                    import cv2 as _cv2
                    from src.core.config import ConfigManager as _CfgMgr
                    _min_res = int(_CfgMgr().get("face_recognition.min_resolution", 230400))
                    _cap = _cv2.VideoCapture(video_path)
                    if _cap.isOpened():
                        _w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                        _h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                        _cap.release()
                        if _w > 0 and _h > 0 and _w * _h < _min_res:
                            _skip_reason = f"低分辨率视频 ({_w}x{_h}={_w*_h}px < {_min_res}px)"
                except Exception:
                    pass

                if _skip_reason:
                    # 只写 FileAnalysis（文件维度持久），不写 Task 表
                    if task.entity_id:
                        _fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                        if _fa:
                            _fa.face_scan_status = "skipped"
                    if own_session:
                        db.commit()
                    logger.info("任务 %d: %s，跳过人脸识别", task_id, _skip_reason)
                    return {
                        "task_id": task_id,
                        "status": "skipped",
                        "message": f"{_skip_reason}，跳过人脸识别",
                        "actor_ids": [],
                        "faces_count": 0,
                        "face_scan_status": "skipped",
                    }

                logger.info("任务 %d: 使用两阶段人脸检测（从视频文件）", task_id)
                actor_ids = self.recognize_actors_from_video(video_path, task_id, file_md5, db_session=db)
            else:
                # 回退到旧方法（从帧目录）
                logger.info("任务 %d: 视频文件不存在，回退到帧目录检测", task_id)
                frames_path = None
                if task.entity_id:
                    fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                    if fa:
                        frames_path = fa.frames_path

                if frames_path and Path(frames_path).exists():
                    actor_ids = self.recognize_actors(frames_path, task_id, file_md5, db_session=db)
                else:
                    if task.entity_id:
                        _fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                        if _fa:
                            _fa.face_scan_status = "skipped"
                    if own_session:
                        db.commit()
                    return {
                        "task_id": task_id,
                        "status": "skipped",
                        "message": "视频文件和帧目录都不存在",
                        "actor_ids": [],
                        "faces_count": 0,
                        "face_scan_status": "skipped",
                    }

            actor_ids = [a for a in actor_ids if a is not None]

            # 分块提交：先把识别阶段创建的 Actor/ActorFace 落库，释放写锁。
            # 避免大量人脸记录积累在单个大事务里，导致并发任务 database is locked。
            if own_session or db_session is not None:
                db.commit()

            # ── 识别后目录一致性清理（关键修复）──
            # 识别阶段可能将新人脸文件复制到已有人物目录（update_actor），
            # 但人物目录中可能残留其他任务重建时遗留的旧文件（DB 中有记录、文件未删）。
            # 必须在识别完成后对所有匹配到的人物：
            # 1. 删除其他任务的旧 ActorFace 记录（保留当前任务的）
            # 2. 删除对应的磁盘文件
            # 3. 执行 _sync_actor_dir 确保"磁盘文件 == DB 记录"
            # 否则目录会无限膨胀（如 ACTOR_005 从 20 张膨胀到 879 张）。
            _all_affected = set(affected_actor_ids) | set(actor_ids)
            for _aid in _all_affected:
                _act = db.query(Actor).filter(Actor.id == _aid).first()
                if _act is None:
                    continue
                # 删除其他任务的旧人脸记录（保留当前任务的）
                _old_faces = db.query(ActorFace).filter(
                    ActorFace.actor_id == _aid,
                    ActorFace.task_id != task_id,
                ).all()
                for _of in _old_faces:
                    self._safe_unlink(self._resolve_face_path(_of.face_path))
                    db.delete(_of)
                # 同步人物目录，删除孤儿文件
                self._sync_actor_dir(db, _act.code)
                # 刷新 sample_count
                self._refresh_sample_count(db, _aid)

            if actor_ids:
                task.actor_id = actor_ids[0]
                for aid in actor_ids:
                    existing_ta = db.query(TaskActor).filter(
                        TaskActor.task_id == task_id,
                        TaskActor.actor_id == aid
                    ).first()
                    if not existing_ta:
                        db.add(TaskActor(task_id=task_id, actor_id=aid))

            # 视频级最终收敛: 本次识别涉及的每个 actor 只保留 Top-N 代表脸
            # (同一人物经多次并入/merge 会累积超上限, 识别完成时统一收敛)。
            for aid in actor_ids:
                self._trim_actor_to_max(db, aid)
            db.flush()

            faces_count = db.query(ActorFace).filter(ActorFace.task_id == task_id).count()

            # 记录人脸识别扫描状态——只写入 FileAnalysis（文件维度持久存储，
            # 任务清空后仍保留）。不写入 Task 表。
            scan_status = "complete" if actor_ids else "none"
            if task.entity_id:
                _fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                if _fa:
                    _fa.face_scan_status = scan_status
            logger.info(
                "任务 %d: 人脸扫描状态 = %s (%d 人物, %d 人脸)",
                task_id, scan_status, len(actor_ids), faces_count,
            )

            if own_session:
                db.commit()

            return {
                "task_id": task_id,
                "actor_ids": actor_ids,
                "faces_count": faces_count,
                "status": "success",
                "face_scan_status": scan_status,
                "message": f"重建成功: 识别到 {len(actor_ids)} 个人物, {faces_count} 张人脸",
            }

        except Exception as e:
            if own_session:
                db.rollback()
            logger.error("重建任务 %d 人脸失败: %s", task_id, e)
            logger.error(traceback.format_exc())
            return {
                "task_id": task_id,
                "status": "error",
                "message": str(e),
                "actor_ids": [],
                "faces_count": 0,
            }
        finally:
            if own_session:
                db.close()

    def rebuild_actor_faces(self, actor_id: int, db_session=None) -> Dict:
        """重建指定人物的人脸数据。

        从该人物关联的所有视频中重新提取人脸，找出匹配该人物的人脸，
        然后重新计算特征向量并更新人物记录。

        Args:
            actor_id: 要重建的人物ID。
            db_session: 可选数据库会话，不传则新建。

        Returns:
            字典，包含 actor_id, status, message, faces_count, task_count
        """
        from src.db.models import Task, Actor, ActorFace, TaskActor, FileAnalysis, get_session, get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            db = db_session

        try:
            # 本函数创建的任务私有裁剪目录列表，所有出口（含异常）统一清理
            crop_dirs: List[Path] = []
            actor = db.query(Actor).filter(Actor.id == actor_id).first()
            if not actor:
                return {
                    "actor_id": actor_id,
                    "status": "error",
                    "message": f"人物 {actor_id} 不存在",
                    "faces_count": 0,
                    "task_count": 0,
                }

            task_ids_from_ta = [
                ta.task_id for ta in
                db.query(TaskActor).filter(TaskActor.actor_id == actor_id).all()
            ]
            task_ids_from_faces = [
                af.task_id for af in
                db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
                if af.task_id is not None
            ]
            task_ids_legacy = [
                t.id for t in
                db.query(Task).filter(Task.actor_id == actor_id).all()
            ]
            all_task_ids = list(set(task_ids_from_ta + task_ids_from_faces + task_ids_legacy))

            if not all_task_ids:
                return {
                    "actor_id": actor_id,
                    "status": "skipped",
                    "message": "该人物没有关联的视频",
                    "faces_count": 0,
                    "task_count": 0,
                }

            if not actor.face_embedding:
                return {
                    "actor_id": actor_id,
                    "status": "error",
                    "message": "人物没有人脸特征数据",
                    "faces_count": 0,
                    "task_count": 0,
                }

            actor_embedding = decode_embedding(actor.face_embedding)

            # ── rebuild 匹配参考集 ──
            # 历史实现用 actor.face_embedding（所有人脸均值）做单向量硬阈值(0.75)匹配：
            # 均值向量对单人跨帧/多姿态相似度天然偏低，且若 actor 混入多人，均值与谁都
            # 不像 → 每次 rebuild 只能找回极少数脸（实测 25→1），样本持续缩水。
            # 改用该 actor 现存 per-face embeddings 做最近邻：每张候选脸取与任一已存脸
            # 的最大相似度 ≥ 阈值（默认 0.55，与 match_actor_v2 对齐）即判定属于该人物。
            # 参考集在删除旧 DB 记录之前收集。
            _rebuild_threshold = 0.55
            try:
                from src.core.config import ConfigManager as _CfgMgr2
                _rebuild_threshold = float(
                    _CfgMgr2().get("face_recognition.rebuild_match_threshold", 0.55)
                )
            except Exception:
                pass
            _ref_embs = []
            try:
                from src.db.models import ActorFace as _AF_Model
                _ref_rows = (
                    db.query(_AF_Model.embedding)
                    .filter(
                        _AF_Model.actor_id == actor_id,
                        _AF_Model.embedding.isnot(None),
                    )
                    .all()
                )
                for (_r,) in _ref_rows:
                    _emb = decode_embedding(_r)
                    if _emb is not None:
                        _ref_embs.append(_emb)
            except Exception as _ref_exc:
                logger.warning("重建人物 %d: 加载 per-face 参考 embedding 失败: %s", actor_id, _ref_exc)
            # 无 per-face 参考时回退到 actor 均值 embedding（旧行为）
            if not _ref_embs:
                _ref_embs = [actor_embedding]
            logger.info(
                "重建人物 %d: 参考集 %d 张脸，匹配阈值 %.2f",
                actor_id, len(_ref_embs), _rebuild_threshold,
            )

            all_matching_faces = []
            processed_tasks = 0

            for tid in all_task_ids:
                task = db.query(Task).filter(Task.id == tid).first()
                if not task:
                    continue

                frames_path = None
                if task.entity_id:
                    fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                    if fa:
                        frames_path = fa.frames_path

                if not frames_path or not Path(frames_path).exists():
                    continue

                # 任务私有裁剪目录：位于 faces_dir 内（同文件系统，rename/copy 安全）
                # 且带 task 标识 + 随机后缀，与共享 faces_dir 根、其他任务彻底隔离，
                # 避免 temp_{frame}_{x}_{y}.jpg 撞名覆写（历史 bug：rebuild 反复
                # extract 时旧 temp 被覆写 → 目录与 DB 不一致/脸数缩水）。
                # 注意：这些目录里的 temp 文件会被复制进 actor 目录后再统一清理，
                # 不能在此处提前删除（face dict 的 face_path 仍指向其中的文件）。
                import tempfile as _tf
                crop_dir = Path(
                    _tf.mkdtemp(prefix=f".rebuild_t{task.id}_", dir=str(self.faces_dir))
                )
                crop_dirs.append(crop_dir)
                try:
                    faces = self.extract_faces_from_frames(frames_path, temp_dir=crop_dir)
                    if not faces:
                        continue

                    for face in faces:
                        # per-face 最近邻：候选脸 vs 参考集任一成员的最大相似度
                        _cand_emb = face.get("embedding")
                        if _cand_emb is None:
                            continue
                        _best_sim = 0.0
                        for _ref in _ref_embs:
                            try:
                                _s = self.compute_similarity(_cand_emb, _ref)
                            except Exception:
                                continue
                            if _s > _best_sim:
                                _best_sim = _s
                        if _best_sim >= _rebuild_threshold:
                            face["task_id"] = tid
                            face["file_md5"] = task.file_md5
                            face["entity_id"] = task.entity_id
                            face["match_score"] = _best_sim
                            all_matching_faces.append(face)

                    processed_tasks += 1
                except Exception as e:
                    logger.warning("处理任务 %d 时出错: %s", tid, e)
                    continue

            # 跨任务/重复任务去重：同一人物可能关联多个 task，而这些 task 的帧目录
            # 可能来自同一视频（重复导入），同一帧的同一张脸会被多次提取匹配。
            # 以 (帧文件名, bbox) 为键去重（帧目录名因任务私有裁剪目录而异，不能用绝对路径），
            # 避免同一张脸被重复入库（历史 bug：目录膨胀到数百张但 DB 只应有几十张）。
            _seen_src = {}
            for face in all_matching_faces:
                _frame_name = Path(face.get("frame_path") or "").name
                _key = (_frame_name, tuple(face.get("bbox") or ()))
                _prev = _seen_src.get(_key)
                if _prev is None:
                    _seen_src[_key] = face
                elif face.get("match_score", 0) > _prev.get("match_score", 0):
                    # 同帧同框出现多次时保留相似度更高的那次
                    _seen_src[_key] = face
            _dedup_count = len(all_matching_faces) - len(_seen_src)
            all_matching_faces = list(_seen_src.values())
            if _dedup_count:
                logger.warning(
                    "重建人物 %d: 去重移除 %d 张重复人脸，保留 %d 张",
                    actor_id, _dedup_count, len(all_matching_faces),
                )

            # Top-N 代表脸精选: rebuild 重建后 actor 只保留少量高质量代表脸
            # (业务模型, config max_samples_per_actor, 默认 5), 与 create/update 一致。
            _max_n = self._max_samples_per_actor()
            if len(all_matching_faces) > _max_n:
                all_matching_faces = self._select_representative_faces(
                    all_matching_faces, _max_n
                )

            if not all_matching_faces:
                return {
                    "actor_id": actor_id,
                    "status": "skipped",
                    "message": f"未找到匹配的人脸（扫描了 {processed_tasks} 个视频）",
                    "faces_count": 0,
                    "task_count": processed_tasks,
                }

            # 重建语义：该 actor 的全部人脸被替换。先删除 DB 记录 + 对应文件，
            # 再整体清空 actor 目录（含历史孤儿文件），保证新目录与 DB 一一对应。
            old_faces = db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
            for old_face in old_faces:
                self._safe_unlink(self._resolve_face_path(old_face.face_path))
                db.delete(old_face)
            db.flush()

            # 整体清空 actor 目录（此时目录内只可能剩孤儿/残留文件）
            import shutil as _shutil
            actor_dir = self.faces_dir / actor.code
            if actor_dir.exists():
                _shutil.rmtree(actor_dir)
            actor_dir.mkdir(parents=True, exist_ok=True)

            moved_faces = []
            for i, face in enumerate(all_matching_faces):
                temp_path = Path(face["face_path"])
                new_filename = f"face_{i}.jpg"
                new_path = actor_dir / new_filename

                if temp_path.exists():
                    # 目标不存在才写入，避免覆盖（新目录理论上为空，双保险）
                    if new_path.exists():
                        logger.warning(
                            "重建人物 %d: 目标文件已存在 %s，跳过",
                            actor_id, new_path,
                        )
                        continue
                    try:
                        # copy 而非 rename：源文件位于任务私有裁剪目录，rename 会把
                        # 文件从 crop_dir 移走；crop_dir 随后整体删除，且若同帧被其他
                        # actor 匹配（多人物共享任务）rename 会破坏其引用。copy 后
                        # crop_dir 统一清理即可，互不影响。
                        import shutil as _shutil_copy
                        _shutil_copy.copy2(str(temp_path), str(new_path))
                        # 存储相对 faces_dir 的路径（与 create_actor/update_actor 一致）
                        try:
                            face["face_path"] = str(new_path.relative_to(self.faces_dir))
                        except ValueError:
                            face["face_path"] = str(new_path)
                        moved_faces.append(face)
                    except Exception as e:
                        logger.error("复制人脸文件失败: %s", e)
                        continue

            if not moved_faces:
                if own_session:
                    db.rollback()
                return {
                    "actor_id": actor_id,
                    "status": "error",
                    "message": "移动人脸文件失败",
                    "faces_count": 0,
                    "task_count": processed_tasks,
                }

            all_embeddings = [np.array(f["embedding"]) for f in moved_faces]
            avg_embedding = np.mean(all_embeddings, axis=0)

            best_face = self._select_best_cover(moved_faces)
            _preferred_cover = best_face["face_path"] if best_face else None

            actor.face_embedding = encode_embedding(avg_embedding)
            actor.sample_count = len(moved_faces)

            avg_sharpness = sum(f.get("sharpness", 0) for f in moved_faces) / len(moved_faces)
            actor.avg_face_sharpness = avg_sharpness

            # 登记新脸（is_cover 由 _apply_actor_cover 统一维护，见下）
            for face in moved_faces:
                actor_face = ActorFace(
                    actor_id=actor_id,
                    task_id=face.get("task_id"),
                    entity_id=face.get("entity_id"),
                    file_md5=face.get("file_md5"),
                    face_path=face["face_path"],
                    yaw=face["yaw"],
                    pitch=face["pitch"],
                    roll=face["roll"],
                    is_cover=0,
                    sharpness=face.get("sharpness", 0),
                    quality_score=face.get("quality_score", 0),
                    embedding=encode_embedding(face.get("embedding")),
                    frame_id=face.get("frame_id"),
                    timestamp=face.get("timestamp"),
                    bbox=json.dumps(face["bbox"]) if face.get("bbox") else None,
                    det_score=face.get("det_score"),
                    kps=json.dumps(face["kps"]) if face.get("kps") is not None else None,
                    scene_id=face.get("scene_id"),
                    sex=face.get("sex"),
                    age=face.get("age"),
                )
                db.add(actor_face)

            db.flush()
            # 统一应用封面：优先 best_face，失败自动回退有效脸
            self._apply_actor_cover(db, actor_id, preferred_path=_preferred_cover)

            if own_session:
                db.commit()

            return {
                "actor_id": actor_id,
                "status": "success",
                "message": f"重建成功: {len(moved_faces)} 张人脸, 来自 {processed_tasks} 个视频",
                "faces_count": len(moved_faces),
                "task_count": processed_tasks,
            }

        except Exception as e:
            if own_session:
                db.rollback()
            logger.error("重建人物 %d 人脸失败: %s", actor_id, e)
            logger.error(traceback.format_exc())
            return {
                "actor_id": actor_id,
                "status": "error",
                "message": str(e),
                "faces_count": 0,
                "task_count": 0,
            }
        finally:
            # 清理本次重建产生的任务私有裁剪目录（正常/异常/提前 return 均执行），
            # 避免 .rebuild_t*_ 残留堆积在 faces_dir 内。
            for _cd in crop_dirs:
                try:
                    if _cd.exists():
                        import shutil as _shutil_clean
                        _shutil_clean.rmtree(_cd, ignore_errors=True)
                except Exception:
                    pass
            if own_session:
                db.close()

    def rebuild_actors_batch(self, actor_ids: List[int]) -> Dict:
        """批量重建多个人物的人脸数据。

        依次重建每个人物的人脸，返回汇总统计。

        Args:
            actor_ids: 人物ID列表。

        Returns:
            字典，包含 total, success, failed, skipped, results 列表
        """
        results = []
        success_count = 0
        failed_count = 0
        skipped_count = 0

        for actor_id in actor_ids:
            result = self.rebuild_actor_faces(actor_id)
            results.append(result)
            if result["status"] == "success":
                success_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
            else:
                failed_count += 1

        return {
            "total": len(actor_ids),
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "results": results,
        }

    def maintain_actor_store(self, db_session=None, dry_run: bool = False) -> Dict:
        """全量维护 faces_dir ↔ actor_faces 一致性。

        修复三类不一致（均可 dry_run 预览）：
        1. actor 目录中未被任何 DB 记录引用的 face_*.jpg → 孤儿，删除
        2. DB 有 actor 但磁盘目录不存在 → 跳过（不重建，需 rebuild_actor_faces）
        3. faces_dir 根的 temp_*.jpg 中间产物 + .rebuild_t*_ 残留目录 → 删除

        Args:
            db_session: 可选数据库会话
            dry_run: True 时只统计不删除

        Returns:
            统计字典 {orphan_files, temp_files, dangling_dirs, db_faces, ...}
        """
        from src.db.models import Actor, ActorFace, get_session, get_db_engine

        own_session = db_session is None
        if own_session:
            engine = get_db_engine()
            db = get_session(engine)
        else:
            db = db_session

        stats = {
            "dry_run": dry_run,
            "db_actors": 0,
            "db_faces": 0,
            "orphan_files": 0,
            "temp_files": 0,
            "dangling_dirs": 0,
            "dirs_scanned": 0,
            "sample_count_fixed": 0,
            "details": [],
        }
        try:
            # DB actor -> code / 目录
            actors = db.query(Actor).all()
            stats["db_actors"] = len(actors)
            db_codes = {a.code for a in actors}
            stats["db_faces"] = db.query(ActorFace).count()

            # 1) 每个 DB actor 目录收敛孤儿
            for a in actors:
                actor_dir = self.faces_dir / a.code
                if not actor_dir.exists():
                    stats["dirs_scanned"] += 1
                    continue
                db_paths = {
                    r[0]
                    for r in db.query(ActorFace.face_path)
                    .filter(ActorFace.actor_id == a.id).all()
                    if r[0]
                }
                orphan_in_dir = 0
                for p in actor_dir.glob("face_*.jpg"):
                    try:
                        rel = str(p.relative_to(self.faces_dir))
                    except ValueError:
                        rel = p.name
                    if rel not in db_paths:
                        orphan_in_dir += 1
                        if not dry_run:
                            self._safe_unlink(p)
                if orphan_in_dir:
                    stats["orphan_files"] += orphan_in_dir
                    stats["details"].append(
                        f"{a.code}: 孤儿 {orphan_in_dir} 个"
                    )
                # sample_count 与 DB 行数一致性
                real_count = (
                    db.query(ActorFace)
                    .filter(ActorFace.actor_id == a.id).count()
                )
                if a.sample_count != real_count:
                    stats["sample_count_fixed"] += 1
                    if not dry_run:
                        a.sample_count = real_count
                stats["dirs_scanned"] += 1

            # 2) 悬空 actor 目录（DB 中已无此 actor）
            if self.faces_dir.exists():
                for d in self.faces_dir.glob("ACTOR_*"):
                    if not d.is_dir():
                        continue
                    if d.name not in db_codes:
                        n_files = sum(1 for _ in d.glob("*.jpg"))
                        stats["dangling_dirs"] += 1
                        stats["details"].append(
                            f"{d.name}: 悬空目录（{n_files} 文件，DB 无此人物）"
                        )
                        if not dry_run:
                            import shutil as _shutil
                            _shutil.rmtree(d, ignore_errors=True)

            # 3) 根目录 temp 临时裁剪文件
            stats["temp_files"] = self._sweep_temp_files() if not dry_run else sum(
                1 for _ in self.faces_dir.glob("temp_*.jpg")
            ) if self.faces_dir.exists() else 0

            if not dry_run and own_session:
                db.commit()
            logger.info(
                "maintain_actor_store(dry_run=%s): 孤儿=%d 悬空目录=%d temp=%d sample_count修正=%d",
                dry_run, stats["orphan_files"], stats["dangling_dirs"],
                stats["temp_files"], stats["sample_count_fixed"],
            )
            return stats
        except Exception as e:
            if own_session:
                db.rollback()
            logger.error("maintain_actor_store 失败: %s", e)
            logger.error(traceback.format_exc())
            return {**stats, "error": str(e)}
        finally:
            if own_session:
                db.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    @staticmethod
    def _extract_pose(face) -> tuple:
        """从人脸对象估算 Euler 角（yaw, pitch, roll）。

        优先使用 5 点关键点估算（AuraFace/SCRFD 不直接输出 pose）；
        兼容旧模型：若对象带 ``pose`` 属性则直接读取。
        """
        kps = getattr(face, 'kps', None)
        if kps is None:
            pose = getattr(face, "pose", None)
            if pose is not None and hasattr(pose, "__len__") and len(pose) >= 3:
                return float(pose[0]), float(pose[1]), float(pose[2])
        return estimate_pose(kps, getattr(face, 'bbox', None))
