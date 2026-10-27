"""正脸/侧脸阶段检测阈值（M2）与 prepare 触发条件的测试。

背景（见 docs/架构/人脸识别-验证样本登记.md 的 M2）：
两处调用 `extract_faces_from_frames` 时**未传** `det_thresh`，落回签名默认值 0.5，
而预筛用 0.15 —— 口径不一致导致"预筛说有正脸、正式检测却检出 0~1 张"。
同时 `prepare()` 只在 `det_size` 变化时才应用阈值，使传参"不生效"。
"""

import pytest

from src.core.face_recognition import FaceRecognizer


class TestStageDetThresh:
    def test_default_matches_legacy_behaviour(self):
        """未配置时必须是 0.5 —— 与改动前（函数签名默认值）一致，零行为变化。"""
        assert FaceRecognizer._stage_det_thresh() == 0.5

    def test_reads_from_config(self, monkeypatch):
        from src.core import config as cfgmod

        def fake_get(self, key, default=None):
            if key == "face_recognition.stage_det_thresh":
                return 0.15
            return default

        monkeypatch.setattr(cfgmod.ConfigManager, "get", fake_get)
        assert FaceRecognizer._stage_det_thresh() == 0.15

    def test_returns_float(self, monkeypatch):
        from src.core import config as cfgmod

        monkeypatch.setattr(cfgmod.ConfigManager, "get",
                            lambda self, k, d=None: "0.2" if k.endswith("stage_det_thresh") else d)
        assert isinstance(FaceRecognizer._stage_det_thresh(), float)


class TestNeedsPrepare:
    def test_size_change_triggers(self):
        assert FaceRecognizer._needs_prepare((640, 640), (960, 960), 0.15, 0.15) is True

    def test_threshold_change_triggers(self):
        """这就是历史漏洞：尺寸没变，但阈值变了 —— 必须重新 prepare，否则传参不生效。"""
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), 0.5, 0.15) is True
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), 0.15, 0.5) is True

    def test_same_size_and_threshold_skips(self):
        """同尺寸同阈值不重复 prepare（保证无逐帧开销）。"""
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), 0.15, 0.15) is False
        assert FaceRecognizer._needs_prepare([320, 320], [320, 320], 0.5, 0.5) is False

    def test_unknown_current_threshold_triggers(self):
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), None, 0.5) is True

    def test_tiny_float_noise_does_not_trigger(self):
        """浮点噪声不应造成反复 prepare。"""
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), 0.15, 0.15 + 1e-9) is False

    def test_unparsable_threshold_triggers(self):
        assert FaceRecognizer._needs_prepare((640, 640), (640, 640), "bad", 0.5) is True


class TestSingleSourceDetThresh:
    """识别器 / 闸门的检测阈值不再硬编码，收为配置单一来源（§4.2 第一步）。"""

    def test_defaults_match_legacy_behaviour(self, monkeypatch):
        """未配置时必须等于改动前的硬编码值 0.15 —— 零行为变化。"""
        from src.core import config as cfgmod

        monkeypatch.setattr(cfgmod.ConfigManager, "get", lambda self, k, d=None: d)
        assert FaceRecognizer._recognition_det_thresh() == 0.15
        assert FaceRecognizer._gate_det_thresh() == 0.15

    def test_configurable(self, monkeypatch):
        from src.core import config as cfgmod

        def fake_get(self, key, default=None):
            return {
                "face_recognition.det_thresh": 0.3,
                "face_recognition.gate_det_thresh": 0.25,
            }.get(key, default)

        monkeypatch.setattr(cfgmod.ConfigManager, "get", fake_get)
        assert FaceRecognizer._recognition_det_thresh() == 0.3
        assert FaceRecognizer._gate_det_thresh() == 0.25

    def test_returns_float(self, monkeypatch):
        from src.core import config as cfgmod

        monkeypatch.setattr(cfgmod.ConfigManager, "get",
                            lambda self, k, d=None: "0.2" if k.endswith("det_thresh") else d)
        assert isinstance(FaceRecognizer._recognition_det_thresh(), float)
        assert isinstance(FaceRecognizer._gate_det_thresh(), float)

    def test_no_hardcoded_call_sites(self):
        """源码级护栏：识别器/闸门里不得再出现写死的 det_thresh=0.15。"""
        from pathlib import Path

        src = Path("src/core/face_recognition.py").read_text(encoding="utf-8")
        assert "det_thresh=0.15" not in src, \
            "仍有硬编码 det_thresh=0.15，应改走 _recognition_det_thresh()/_gate_det_thresh()"


class TestCallSitesPassThreshold:
    """两处调用点必须显式传阈值（源码级回归护栏，防止再次漏传）。"""

    @pytest.mark.parametrize("snippet", [
        "frame_filter=frontal_frames",   # 正脸阶段
        "frame_filter=side_frames",      # 侧脸阶段
    ])
    def test_explicit_det_thresh(self, snippet):
        from pathlib import Path

        src = Path("src/core/face_recognition.py").read_text(encoding="utf-8")
        # 找到该调用点前后一小段，断言其中含显式阈值
        idx = src.find(snippet)
        assert idx > 0, f"未找到调用点 {snippet}"
        window = src[max(0, idx - 220):idx]
        assert "det_thresh=" in window, \
            f"{snippet} 的调用点未显式传 det_thresh（会落回默认 0.5，见 M2）"
