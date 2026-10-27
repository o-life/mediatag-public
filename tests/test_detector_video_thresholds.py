"""检测阈值的"本片生效值"（unify-face-thresholds §4.3）。

历史问题：`FaceDetector.detect_video` 低清时**直接改写** `self.confidence_threshold` /
`self.min_face_area_ratio`，而下一部视频又拿它们当"基准值"读 →
**一次低清会让后续高清视频也按低阈值跑**（取值串味）；共享实例在并发下还会互相干扰。

修法：配置值不再被改写；按视频算出"生效值"单独存，仅在变化时重建缓存的检测器。
"""

import pytest

from src.core.face_detector import FaceDetector

LOW_RES = (640, 360)      # 230400 = 阈值边界之下取更小
LOW = (640, 300)          # 192000 < 230400
HIGH = (1920, 1080)       # 2073600


@pytest.fixture
def detector():
    # __init__ 不加载模型（检测器/预扫描器是惰性属性），可安全构造
    return FaceDetector(config={
        "face_detection": {"confidence_threshold": 0.5, "min_face_area_ratio": 0.005}
    })


class TestEffectiveThresholds:
    def test_defaults_equal_configured(self, detector):
        assert detector.effective_confidence == 0.5
        assert detector.effective_min_face_area_ratio == 0.005

    def test_low_res_relaxes(self, detector):
        detector._apply_video_thresholds(*LOW)
        assert detector.effective_confidence == 0.25
        assert detector.effective_min_face_area_ratio == 0.002

    def test_high_res_keeps_configured(self, detector):
        detector._apply_video_thresholds(*HIGH)
        assert detector.effective_confidence == 0.5
        assert detector.effective_min_face_area_ratio == 0.005

    def test_boundary_is_not_low_res(self, detector):
        """恰好等于 360p 基准像素不算低清（与 face_recognition 口径一致）。"""
        detector._apply_video_thresholds(640, 360)
        assert detector.effective_confidence == 0.5


class TestNoStickyState:
    """核心回归：低清**不得**污染后续高清视频的取值。"""

    def test_low_then_high_restores(self, detector):
        detector._apply_video_thresholds(*LOW)
        assert detector.effective_confidence == 0.25
        detector._apply_video_thresholds(*HIGH)
        assert detector.effective_confidence == 0.5, "低清放宽不得粘住"
        assert detector.effective_min_face_area_ratio == 0.005

    def test_config_values_never_mutated(self, detector):
        detector._apply_video_thresholds(*LOW)
        assert detector.confidence_threshold == 0.5, "配置值（基准）不得被改写"
        assert detector.min_face_area_ratio == 0.005

    def test_repeated_low_res_is_idempotent(self, detector):
        detector._apply_video_thresholds(*LOW)
        detector._apply_video_thresholds(*LOW)
        assert detector.effective_confidence == 0.25
        assert detector.effective_min_face_area_ratio == 0.002

    def test_only_relaxes_never_tightens(self, detector):
        """配置本来就比放宽值更宽时，不得被"收紧"回去。"""
        loose = FaceDetector(config={
            "face_detection": {"confidence_threshold": 0.1, "min_face_area_ratio": 0.001}
        })
        loose._apply_video_thresholds(*LOW)
        assert loose.effective_confidence == pytest.approx(0.1)
        assert loose.effective_min_face_area_ratio == pytest.approx(0.001)


class TestCacheInvalidation:
    def test_effective_change_drops_cached_instances(self, detector, monkeypatch):
        """生效值变化时要丢弃按旧值构建的缓存实例（否则新阈值不生效）。"""
        calls = {}

        class _FakeDetector:
            def close(self):
                calls["closed"] = True

        detector._detector = _FakeDetector()          # type: ignore[assignment]
        detector._apply_video_thresholds(*LOW)
        assert calls.get("closed") is True
        assert detector._detector is None

    def test_same_values_keep_cache(self, detector):
        class _FakeDetector:
            def close(self):  # pragma: no cover - 不应被调用
                raise AssertionError("生效值未变时不应重建缓存")

        detector._detector = _FakeDetector()          # type: ignore[assignment]
        detector._apply_video_thresholds(*HIGH)       # 与初始生效值相同
        assert isinstance(detector._detector, _FakeDetector)
