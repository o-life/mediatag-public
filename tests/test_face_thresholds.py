"""阈值解析层（`src/core/face_thresholds.py`）测试。

迁移自 `FaceRecognizer._dynamic_sharpness_threshold` + `._adaptive_sharpness_floor`。
核心是**等价性**：把迁移前的公式原样抄成本文件的参照实现，在网格输入上逐一比对，
确保"换成解析层"这件事本身不改变任何取值。
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

from src.core.face_thresholds import (
    LOW_QUALITY_RATIO,
    LOW_RES_PIXELS,
    absolute_sharpness_threshold,
    build_profile,
)


# ─────────────────────────────────────────────────────────────
# 参照实现：迁移前的公式（逐字照抄，只把 self 换成无状态函数）
# ─────────────────────────────────────────────────────────────

def reference_absolute(fqc: dict, video_quality_info=None) -> float:
    base = float(fqc.get("min_sharpness_absolute", 50))
    if video_quality_info is None:
        return base
    is_low_quality = getattr(video_quality_info, "is_low_quality", False)
    resolution = getattr(video_quality_info, "resolution", None)
    is_low_res = False
    if resolution and len(resolution) == 2:
        is_low_res = (resolution[0] * resolution[1]) < 230400
    if is_low_quality or is_low_res:
        low = float(fqc.get("min_sharpness_absolute_low", 8))
        threshold = float(getattr(video_quality_info, "threshold", 0) or 0)
        return max(low, threshold * 0.1)
    return base


def reference_floor(fqc: dict, faces: list, video_quality_info=None) -> float:
    base = reference_absolute(fqc, video_quality_info)
    if not fqc.get("adaptive_enabled", True):
        return base
    values = [float(f.get("sharpness") or 0.0) for f in faces]
    values = [v for v in values if v > 0]
    if len(values) < int(fqc.get("adaptive_min_faces", 5)):
        return base
    junk_floor = float(fqc.get("min_sharpness_absolute_low", 8))
    ref = float(np.percentile(values, float(fqc.get("adaptive_percentile", 30))))
    return float(min(base, max(junk_floor, ref)))


@dataclass
class FakeQuality:
    resolution: Optional[Tuple[int, int]] = None
    is_low_quality: bool = False
    threshold: float = 0.0


# 事故片 60 帧检出、通过尺寸过滤的 52 张人脸真实清晰度分布
REAL_SOFT_VIDEO_SHARPNESS = [
    4.9, 5.4, 5.8, 6.6, 7.2, 7.5, 8.1, 9.9, 10.2, 10.8, 10.8, 12.1, 12.5,
    12.9, 13.4, 13.5, 15.0, 16.4, 16.6, 16.7, 20.1, 20.5, 23.5, 23.6, 24.2,
    26.2, 26.3, 27.6, 28.7, 28.8, 28.9, 29.0, 31.7, 31.9, 32.4, 38.6, 39.1,
    39.4, 40.6, 41.9, 42.6, 45.9, 48.3, 49.9, 53.6, 61.0, 80.2, 89.4, 93.0,
    93.3, 120.1, 166.3,
]

BASE_FQC = {
    "enabled": True,
    "min_sharpness_absolute": 50,
    "min_sharpness_absolute_low": 8,
    "adaptive_enabled": True,
    "adaptive_percentile": 30,
    "adaptive_min_faces": 5,
}

FQC_VARIANTS: Sequence[dict] = [
    BASE_FQC,
    {**BASE_FQC, "min_sharpness_absolute": 10},          # 仓库 config 的取值
    {**BASE_FQC, "adaptive_enabled": False},
    {**BASE_FQC, "adaptive_percentile": 50},
    {**BASE_FQC, "adaptive_min_faces": 3},
    {**BASE_FQC, "min_sharpness_absolute_low": 20},
    {},                                                  # 全缺省（走代码默认）
]

QUALITY_VARIANTS = [
    None,
    FakeQuality(),
    FakeQuality(resolution=(1280, 590)),                              # 高清、非低质
    FakeQuality(resolution=(640, 480), threshold=100.0),              # 低清、阈值 100
    FakeQuality(is_low_quality=True, threshold=90.0),                 # 低质、阈值 90
    FakeQuality(is_low_quality=True, resolution=(1920, 1080), threshold=0.0),
]

FACES_VARIANTS = [
    [],
    [{"sharpness": 4.0}] * 4,                                          # 少于门槛
    [{"sharpness": s} for s in REAL_SOFT_VIDEO_SHARPNESS],             # 事故片分布
    [{"sharpness": 200.0}] * 20,                                       # 高质量
    [{"sharpness": 0.0}] * 20,                                         # 全零（无效）
    [{}] * 20,                                                         # 缺字段
    [{"sharpness": None}] * 20,
]


class TestEquivalenceWithLegacyFormula:
    """解析层与迁移前公式必须逐点一致。"""

    @pytest.mark.parametrize("fqc", FQC_VARIANTS)
    @pytest.mark.parametrize("quality", QUALITY_VARIANTS)
    def test_absolute_threshold(self, fqc, quality):
        assert absolute_sharpness_threshold(fqc, quality) == reference_absolute(fqc, quality)

    @pytest.mark.parametrize("fqc", FQC_VARIANTS)
    @pytest.mark.parametrize("quality", QUALITY_VARIANTS)
    @pytest.mark.parametrize("faces", FACES_VARIANTS)
    def test_sharpness_floor(self, fqc, quality, faces):
        got = build_profile(fqc, faces, quality).sharpness_floor
        assert got == reference_floor(fqc, faces, quality)


class TestProfileContract:
    def test_immutable(self):
        profile = build_profile(BASE_FQC, [{"sharpness": 30.0}] * 10)
        with pytest.raises(Exception):
            profile.sharpness_floor = 1.0  # type: ignore[misc]

    def test_only_relaxes_never_tightens(self):
        """生效值不得超过绝对上限（清晰片源行为不变）。"""
        for fqc in FQC_VARIANTS:
            for quality in QUALITY_VARIANTS:
                for faces in FACES_VARIANTS:
                    profile = build_profile(fqc, faces, quality)
                    cap = reference_absolute(fqc, quality)
                    assert profile.sharpness_floor <= cap + 1e-9

    def test_traceable_log_fields(self):
        """生效取值与推导依据必须可追溯（规格要求）。"""
        profile = build_profile(BASE_FQC, [{"sharpness": s} for s in REAL_SOFT_VIDEO_SHARPNESS])
        fields = profile.as_log_fields()
        assert fields["sharpness_floor"] == pytest.approx(profile.sharpness_floor, abs=0.01)
        assert fields["adaptive"] == "on"
        assert fields["faces"] == len(REAL_SOFT_VIDEO_SHARPNESS)
        assert "face_p30" in fields

    def test_records_why_adaptive_was_skipped(self):
        too_few = build_profile(BASE_FQC, [{"sharpness": 4.0}] * 3).as_log_fields()
        assert too_few["adaptive"] == "skipped_too_few_faces"
        off = build_profile({**BASE_FQC, "adaptive_enabled": False}, []).as_log_fields()
        assert off["adaptive"] == "off"

    def test_incident_video_floor_is_relaxed(self):
        """回归事故片：该分布下生效值应明显低于 50（否则整片被判无人脸）。"""
        profile = build_profile(BASE_FQC, [{"sharpness": s} for s in REAL_SOFT_VIDEO_SHARPNESS])
        assert 8 < profile.sharpness_floor < 50
        assert profile.sharpness_floor == reference_floor(
            BASE_FQC, [{"sharpness": s} for s in REAL_SOFT_VIDEO_SHARPNESS]
        )


class TestProfileLog:
    """规格要求"每条阈值的最终取值可事后追溯"——汇总日志必须含生效值与依据。"""

    def test_log_contains_effective_value_and_basis(self, caplog):
        import logging

        from src.core.face_thresholds import log_profile

        profile = build_profile(BASE_FQC, [{"sharpness": s} for s in REAL_SOFT_VIDEO_SHARPNESS])
        with caplog.at_level(logging.INFO):
            log_profile(profile, where="unit", task=42)
        msg = caplog.text
        assert "人脸阈值生效取值" in msg
        assert "unit" in msg
        assert "sharpness_floor=" in msg
        assert "adaptive=on" in msg
        assert "task=42" in msg

    def test_log_without_context_is_still_valid(self, caplog):
        import logging

        from src.core.face_thresholds import log_profile

        profile = build_profile({**BASE_FQC, "adaptive_enabled": False}, [])
        with caplog.at_level(logging.INFO):
            log_profile(profile, where="unit")
        assert "adaptive=off" in caplog.text


class TestConstants:
    def test_low_res_boundary(self):
        """恰好等于基准像素不算低清；低于才算。"""
        assert LOW_RES_PIXELS == 230400
        exact = FakeQuality(resolution=(640, 360))          # 230400
        below = FakeQuality(resolution=(640, 359))          # 229760
        fqc = {**BASE_FQC, "min_sharpness_absolute_low": 8}
        assert absolute_sharpness_threshold(fqc, exact) == 50
        assert absolute_sharpness_threshold(fqc, below) == 8

    def test_low_quality_ratio(self):
        fqc = {**BASE_FQC, "min_sharpness_absolute_low": 8}
        q = FakeQuality(is_low_quality=True, threshold=100.0)
        assert absolute_sharpness_threshold(fqc, q) == 100.0 * LOW_QUALITY_RATIO


class TestEmbeddingNormRule:
    """§3：embedding 范数下限自适应（形态同清晰度：clip(分位, 绝对下限, 配置上限)）。

    库内范数实测：p5=12.8 / p10=14.2 / p25=17.2；阈值 15.0 ≈ p12。
    因此参考分位必须 **≥ p25**，否则会把清晰片源也放宽。
    """

    NORM_CFG = {
        "configured_high": 15.0, "absolute_low": 8.0, "percentile": 25,
        "min_faces": 5, "enabled": True, "low_quality_ratio": 0.7,
    }

    def _norm(self, norms, quality=None, **over):
        cfg = {**self.NORM_CFG, **over}
        faces = [{"sharpness": 100.0, "norm": n} for n in norms]
        return build_profile(BASE_FQC, faces, quality, norm_config=cfg).min_embedding_norm

    def test_clear_video_unchanged(self):
        """清晰片源：本片分位高于配置上限 → 生效值仍为 15.0。"""
        assert self._norm([17.0, 18.5, 19.0, 20.0, 22.0]) == 15.0

    def test_soft_video_relaxes(self):
        """低范数片源：按本片分位放宽，但不低于绝对下限。"""
        got = self._norm([10.0, 11.0, 12.0, 13.0, 14.0])
        assert 8.0 < got < 15.0

    def test_never_below_absolute_low(self):
        assert self._norm([1.0, 2.0, 3.0, 4.0, 5.0]) == 8.0

    def test_p5_would_relax_a_clear_video(self):
        """反例（分位敏感性）：库内分布下取 p5 会把清晰片源也放宽到 12.8。"""
        library_like = [12.8] * 5 + [17.2] * 15
        assert self._norm(library_like, percentile=25) == 15.0      # 正确：不变
        assert self._norm(library_like, percentile=5) == 12.8       # 错误分位：被放宽

    def test_low_quality_relaxes_ceiling(self):
        """低质量/低清片源：上限按比例（0.7）放宽 —— 沿用历史行为。"""
        q = FakeQuality(is_low_quality=True)
        assert self._norm([17.0, 18.0, 19.0, 20.0, 21.0], q) == pytest.approx(10.5)

    def test_disabled_falls_back_to_configured_high(self):
        assert self._norm([10.0, 11.0, 12.0, 13.0, 14.0], enabled=False) == 15.0

    def test_too_few_faces_falls_back(self):
        assert self._norm([10.0, 11.0]) == 15.0

    def test_only_relaxes_never_tightens(self):
        for norms in ([100.0] * 10, [0.1] * 10):
            assert self._norm(norms) <= 15.0

    def test_absent_norm_config_means_not_applied(self):
        """不传 norm_config 时本层不接管范数（min_embedding_norm = 0）。"""
        faces = [{"sharpness": 100.0, "norm": 10.0}] * 10
        assert build_profile(BASE_FQC, faces).min_embedding_norm == 0.0

    def test_log_fields_expose_norm(self):
        faces = [{"sharpness": 100.0, "norm": 12.0}] * 10
        profile = build_profile(BASE_FQC, faces, norm_config=self.NORM_CFG)
        fields = profile.as_log_fields()
        assert fields["min_embedding_norm"] == pytest.approx(profile.min_embedding_norm, abs=0.01)
        assert fields["norm_rule"] == "on"


class TestDefaultConvergence:
    """§6.1–6.3：同一语义的阈值只能有**一个**代码默认值。

    原先存在分歧：距离阈值 识别 0.55 / 聚类 0.6；合并阈值 代码 0.4 / 配置 0.55；
    最小簇 代码 10 / 配置 3 —— 无法解释也无法调优。
    """

    def test_constants_exist_with_repo_config_values(self):
        from src.core.face_thresholds import (
            DEFAULT_DISTANCE_THRESHOLD,
            DEFAULT_MERGE_THRESHOLD,
            DEFAULT_MIN_CLUSTER_SIZE,
            DEFAULT_MIN_SAMPLES,
        )

        assert DEFAULT_DISTANCE_THRESHOLD == 0.6   # 与仓库 config.yaml 一致
        assert DEFAULT_MERGE_THRESHOLD == 0.55
        assert DEFAULT_MIN_CLUSTER_SIZE == 3
        assert DEFAULT_MIN_SAMPLES == 3

    def test_clusterer_defaults_use_constants(self):
        import inspect

        from src.core.face_clustering import FaceClusterer
        from src.core.face_thresholds import (
            DEFAULT_DISTANCE_THRESHOLD,
            DEFAULT_MIN_SAMPLES,
        )

        sig = inspect.signature(FaceClusterer.__init__)
        assert sig.parameters["distance_threshold"].default == DEFAULT_DISTANCE_THRESHOLD
        assert sig.parameters["min_samples"].default == DEFAULT_MIN_SAMPLES

    def test_no_divergent_literals_in_sources(self):
        """源码级护栏：不得再出现分歧的默认值字面量。"""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        offenders = []
        for rel in ("src/core/face_recognition.py", "src/core/face_clustering.py"):
            text = (root / rel).read_text(encoding="utf-8")
            for bad in ('distance_threshold", 0.55', 'distance_threshold: float = 0.50',
                        'min_cluster_size", 10', 'merge_threshold", 0.4'):
                if bad in text:
                    offenders.append(f"{rel}: {bad}")
        assert not offenders, f"仍有分歧的默认值字面量: {offenders}"
