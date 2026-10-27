"""均匀抽帧目标帧数的推导测试。

背景（见 docs/架构/人脸识别-验证样本登记.md 的"M4 判定"）：
聚类要求**同一个人的相邻样本**落在人脸外观变化尺度之内。实测约 2 秒间隔时
同人距离 0.307 vs 跨人 0.855（可分）；7 秒间隔时 0.807 vs 0.855（不可分）。
原实现写死"约每 8 秒一帧"，对长视频（425s → 60 帧 → 实际间隔 7.1 秒）恰好落在不可分区间。

本文件的核心是**等价性**：默认间隔 8.0 必须与改动前的算法逐点一致。
"""

import pytest

from src.core.face_recognition import FaceRecognizer


def legacy_target(dur: float) -> int:
    """改动前的算法（逐字照抄）。"""
    return max(60, min(400, int(dur / 8)))


class TestEquivalenceWithLegacy:
    @pytest.mark.parametrize("dur", [
        0, 1, 5, 10, 30, 60, 99.9, 100, 206.016, 242.04, 271.9, 278.6, 290.4,
        425.1, 428, 472.3, 486.4, 532.5, 540, 800, 1200, 2400, 3200, 3600, 7200,
    ])
    def test_default_interval_matches_legacy(self, dur):
        got = FaceRecognizer._uniform_sampling_target(dur, 8.0)
        assert got == legacy_target(dur)

    def test_non_positive_duration_uses_floor(self):
        """时长未知/为 0 → 取下限（与改动前一致）。"""
        assert FaceRecognizer._uniform_sampling_target(0, 8.0) == 60
        assert FaceRecognizer._uniform_sampling_target(-1, 8.0) == 60

    def test_invalid_interval_does_not_silently_thin_out(self):
        """间隔配置异常 → 退回上限（宁可多抽），不静默变成稀疏采样。"""
        assert FaceRecognizer._uniform_sampling_target(100, 0) == 400
        assert FaceRecognizer._uniform_sampling_target(100, -1) == 400


class TestIntervalDriven:
    def test_denser_interval_yields_more_frames(self):
        """间隔越小 → 帧数越多（在上下限之间时）。"""
        dur = 425.1
        assert FaceRecognizer._uniform_sampling_target(dur, 8.0) == 60      # 原实现
        assert FaceRecognizer._uniform_sampling_target(dur, 2.0) == 212     # 目标 2 秒
        assert FaceRecognizer._uniform_sampling_target(dur, 1.0) == 400     # 触到上限

    def test_actual_interval_close_to_target(self):
        """上限未触及时，实际间隔应接近目标间隔。"""
        dur = 425.1
        for interval in (2.0, 3.0, 4.0):
            n = FaceRecognizer._uniform_sampling_target(dur, interval)
            assert abs(dur / n - interval) < 0.1

    def test_floor_protects_short_videos(self):
        """短视频不因间隔大而少于下限帧数。"""
        assert FaceRecognizer._uniform_sampling_target(30, 8.0) == 60
        assert FaceRecognizer._uniform_sampling_target(30, 2.0) == 60

    def test_cap_is_a_cost_guardrail(self):
        """长视频会因上限而达不到目标间隔——这是显式的成本取舍，需可观测。"""
        dur = 3600.0
        n = FaceRecognizer._uniform_sampling_target(dur, 2.0)
        assert n == 400
        assert dur / n > 2.0  # 达不到目标间隔（9 秒），说明上限在起作用


class TestIncidentShapedCases:
    """本次排查中真实文件的时长，用于钉住行为。"""

    @pytest.mark.parametrize("dur,expected_at_2s", [
        (100.07, 60),    # 样本 6：短视频，被下限托住
        (206.02, 103),   # 样本 4
        (242.04, 121),   # 样本 3
        (425.10, 212),   # 事故片
        (532.50, 266),   # 样本 1
    ])
    def test_real_durations(self, dur, expected_at_2s):
        assert FaceRecognizer._uniform_sampling_target(dur, 2.0) == expected_at_2s
