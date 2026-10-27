"""NSFW 检测器后端可配置：默认 `opennsfw`（可商用），`none` 则跳过预筛走 VLM 兜底。

历史后端 NudeNet（AGPL-3.0）已移除，不再随包分发。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.nsfw_detector import (
    OpenNSFWDetector, create_nsfw_detector,
)


class TestNsfwDetectorBackend:
    def test_opennsfw_backend(self):
        d = create_nsfw_detector("opennsfw")
        assert isinstance(d, OpenNSFWDetector)
        assert d.threshold == 0.5
        # 构造不加载模型（惰性），不应触碰 onnxruntime
        assert d._clf is None

    def test_threshold_passed(self):
        assert create_nsfw_detector("opennsfw", threshold=0.7).threshold == 0.7

    def test_none_backend_skips_stage1(self):
        assert create_nsfw_detector("none") is None

    def test_default_reads_config(self):
        # 缺省（None）→ 读 config nsfw.detector（默认 opennsfw）
        assert isinstance(create_nsfw_detector(None), OpenNSFWDetector)

    def test_unknown_backend_falls_back_to_opennsfw(self):
        assert isinstance(create_nsfw_detector("whatever"), OpenNSFWDetector)
