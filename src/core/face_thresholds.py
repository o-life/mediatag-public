"""视频级人脸阈值解析层。

统一规则形态（见 openspec/changes/unify-face-thresholds）：

    effective = clip( f(本片统计量), 绝对下限, 配置上限 )

两条不变式：

1. **只放宽、不收紧** —— 生效值不会高于配置的绝对上限，
   因此清晰片源的判定结果与引入本层之前完全一致。
2. **取值由标定集决定** —— 本模块不内置任何来自单个文件观测的取值；
   代码固化的是"关系与推导方法"，具体数值来自配置或标定集。

本层是全体人脸阈值的**唯一来源**：调用方不再各自读取配置或硬编码常量。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np

logger = logging.getLogger(__name__)

#: 低清基准像素（约 360p）。低于此像素量的片源，人脸裁剪更小、指标天然偏低。
LOW_RES_PIXELS = 230400

# ── 代码默认值：**单一来源**（见 openspec/changes/unify-face-thresholds §6.1–6.3）──
# 原先同一语义的阈值在不同路径有不同默认值（识别 0.55 vs 聚类 0.6；合并 0.4 vs 配置 0.55；
# 最小簇 10 vs 配置 3），无法解释也无法调优。这里收敛为一份，取值与仓库 `config.yaml` 一致。
# 注意：配置里已有这些键时行为不变；仅在**配置缺失**时默认值才生效（本仓库即前者）。
DEFAULT_DISTANCE_THRESHOLD = 0.6   # 聚类/匹配的余弦距离阈值
DEFAULT_MERGE_THRESHOLD = 0.55     # 簇合并的相似度阈值
DEFAULT_MIN_CLUSTER_SIZE = 3       # 最小簇大小（稀疏视频另有 min(本值, max(2, n//15)) 的放宽）
DEFAULT_MIN_SAMPLES = 3            # DBSCAN min_samples

#: 低质量片源按"视频清晰度阈值"的比例下调人脸清晰度下限。
LOW_QUALITY_RATIO = 0.1


def absolute_sharpness_threshold(fqc: Dict[str, Any], video_quality_info: Any = None) -> float:
    """配置给出的 sharpness 绝对下限（绝对上限），含低质量/低清时的下调。

    对低质量或低清片源放宽到 ``max(绝对下限, 视频清晰度阈值 × LOW_QUALITY_RATIO)``；
    其余情况即为配置值 ``min_sharpness_absolute``。

    Args:
        fqc: ``face_quality`` 配置字典
        video_quality_info: 可选的视频质量信息（含 is_low_quality / resolution / threshold）

    Returns:
        清晰度的绝对阈值
    """
    base = float(fqc.get("min_sharpness_absolute", 50))

    if video_quality_info is None:
        return base

    is_low_quality = bool(getattr(video_quality_info, "is_low_quality", False))
    # 低分辨率（< 360p 基准像素）→ 人脸裁剪图更小、sharpness 天然偏低，
    # 即使全帧 avg_sharpness 达标（is_low_quality=False）也应放宽
    resolution = getattr(video_quality_info, "resolution", None)
    is_low_res = bool(resolution and len(resolution) == 2
                      and (resolution[0] * resolution[1]) < LOW_RES_PIXELS)

    if is_low_quality or is_low_res:
        low = float(fqc.get("min_sharpness_absolute_low", 8))
        threshold = float(getattr(video_quality_info, "threshold", 0) or 0)
        scaled = max(low, threshold * LOW_QUALITY_RATIO)
        logger.info(
            "低质量/低分辨率视频 (%s): sharpness 下限从 %.0f 下调到 %.1f (视频阈值 %.1f)",
            "low_quality" if is_low_quality else "low_res",
            base, scaled, threshold,
        )
        return scaled

    return base


def absolute_embedding_norm_threshold(norm_cfg: Dict[str, Any],
                                      video_quality_info: Any = None) -> float:
    """配置给出的 embedding 范数**上限**（含低质量/低清时的按比例放宽）。

    沿用历史行为：低质量或低清片源的 embedding 范数天然偏低（实测 13–14），
    统一阈值会漏检，故放宽到 ``configured_high × low_quality_ratio``（默认 ×0.7）。
    """
    base = float(norm_cfg.get("configured_high", 15.0))
    if video_quality_info is None:
        return base

    is_low_quality = bool(getattr(video_quality_info, "is_low_quality", False))
    resolution = getattr(video_quality_info, "resolution", None)
    is_low_res = bool(resolution and len(resolution) == 2
                      and (resolution[0] * resolution[1]) < LOW_RES_PIXELS)
    if is_low_quality or is_low_res:
        ratio = float(norm_cfg.get("low_quality_ratio", 0.7))
        return base * ratio
    return base


def _embedding_norm_floor(norm_cfg: Dict[str, Any], faces: Iterable[Dict[str, Any]],
                          video_quality_info: Any = None) -> tuple:
    """范数下限的生效值与其推导依据（分位）。

    形态与清晰度一致：``min(配置上限, max(绝对下限, 本片范数 p<ref>))``，只放宽不收紧。
    ⚠️ 分位必须 **≥ p25**：阈值 15.0 ≈ 库内 p12（p10=14.2 / p25=17.2），
    取 p5/p10 会把**所有**片源都放宽（违反"清晰片源行为不变"）。
    """
    high = absolute_embedding_norm_threshold(norm_cfg, video_quality_info)
    if not norm_cfg.get("enabled", True):
        return high, {"norm_rule": "off"}

    values = [float(f.get("norm") or 0.0) for f in faces]
    values = [v for v in values if v > 0]
    min_faces = int(norm_cfg.get("min_faces", 5))
    if len(values) < min_faces:
        return high, {"norm_rule": "skipped_too_few_faces", "norm_faces": len(values)}

    percentile = float(norm_cfg.get("percentile", 25))
    reference = float(np.percentile(values, percentile))
    junk_floor = float(norm_cfg.get("absolute_low", 8.0))
    floor = min(high, max(junk_floor, reference))
    return floor, {
        "norm_rule": "on",
        "norm_high": round(high, 2),
        "norm_p%g" % percentile: round(reference, 2),
        "norm_faces": len(values),
    }


@dataclass(frozen=True)
class VideoThresholdProfile:
    """一次人脸处理所使用的阈值集合（不可变）。

    每个字段都是**最终生效值**；``basis`` 记录推导依据，供日志追溯
    （对应规格要求"生效取值可追溯"）。
    """

    #: 人脸清晰度下限（Laplacian 方差）
    sharpness_floor: float
    #: embedding 范数下限（0 = 本层未接管，调用方沿用旧逻辑）
    min_embedding_norm: float = 0.0
    #: 推导依据（仅用于日志/排查，不参与判定）
    basis: Dict[str, Any] = field(default_factory=dict)

    def as_log_fields(self) -> Dict[str, Any]:
        """展平为可直接写日志的键值对。"""
        fields = {"sharpness_floor": round(self.sharpness_floor, 2)}
        if self.min_embedding_norm > 0:
            fields["min_embedding_norm"] = round(self.min_embedding_norm, 2)
        return {**fields, **self.basis}


def log_profile(profile: "VideoThresholdProfile", *, where: str, **context: Any) -> None:
    """输出"本片阈值最终取值"汇总日志（规格要求：判定的取值可事后追溯）。

    每条规则一行，含自适应结果与推导依据；``where`` 标明调用阶段，
    ``context`` 追加定位信息（任务号 / 帧目录等）。
    """
    fields = " ".join(f"{k}={v}" for k, v in profile.as_log_fields().items())
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info("人脸阈值生效取值 [%s] %s%s", where, fields, f" {ctx}" if ctx else "")


def build_profile(
    fqc: Dict[str, Any],
    faces: Iterable[Dict[str, Any]],
    video_quality_info: Any = None,
    norm_config: Dict[str, Any] = None,
) -> VideoThresholdProfile:
    """按"本片已检出人脸"的统计量解析出阈值集合。

    Args:
        fqc: ``face_quality`` 配置字典（清晰度规则）
        faces: 当前过滤阶段的人脸全集（清晰度规则需 ``sharpness``；
            范数规则需 ``norm``）
        video_quality_info: 可选的视频质量信息
        norm_config: 可选的 embedding 范数规则配置
            （``configured_high`` / ``absolute_low`` / ``percentile`` /
            ``min_faces`` / ``enabled`` / ``low_quality_ratio``）；
            不传则不解析范数（``min_embedding_norm = 0``）

    Returns:
        VideoThresholdProfile
    """
    faces = list(faces)
    norm_floor, norm_basis = (0.0, {})
    if norm_config:
        norm_floor, norm_basis = _embedding_norm_floor(norm_config, faces, video_quality_info)

    base = absolute_sharpness_threshold(fqc, video_quality_info)

    if not fqc.get("adaptive_enabled", True):
        return VideoThresholdProfile(sharpness_floor=base, min_embedding_norm=norm_floor,
                                     basis={"adaptive": "off", **norm_basis})

    values = [float(f.get("sharpness") or 0.0) for f in faces]
    values = [v for v in values if v > 0]
    min_faces = int(fqc.get("adaptive_min_faces", 5))
    if len(values) < min_faces:
        return VideoThresholdProfile(
            sharpness_floor=base,
            min_embedding_norm=norm_floor,
            basis={"adaptive": "skipped_too_few_faces", "faces": len(values), **norm_basis},
        )

    percentile = float(fqc.get("adaptive_percentile", 30))
    reference = float(np.percentile(values, percentile))
    junk_floor = float(fqc.get("min_sharpness_absolute_low", 8))
    floor = min(base, max(junk_floor, reference))

    return VideoThresholdProfile(
        sharpness_floor=floor,
        min_embedding_norm=norm_floor,
        basis={
            "adaptive": "on",
            "absolute_threshold": round(base, 2),
            "face_p%g" % percentile: round(reference, 2),
            "junk_floor": junk_floor,
            "faces": len(values),
            **norm_basis,
        },
    )
