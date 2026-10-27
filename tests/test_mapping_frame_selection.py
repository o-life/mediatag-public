"""Tests for 逐帧标签映射的选帧策略 (src/core/nsfw_detector.py)。

覆盖:
1. select_mapping_frames —— NSFW 充足 / 不足 / 0 检出(男性向漏检盲区)
2. sample_fallback_frames —— 场景内/平铺目录抽样(从 TwoStageNSFWAnalyzer 提取)
3. TwoStageNSFWAnalyzer._sample_fallback_frames 仍委派到模块级实现(无回归)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.nsfw_detector import (
    FrameAnalysis,
    TwoStageNSFWAnalyzer,
    sample_fallback_frames,
    select_mapping_frames,
    select_scene_frames,
)


def _unit_frames(unit=0, count=3):
    return [f"/frames/unit_{unit}/frame_{i:03d}.jpg" for i in range(count)]


def _scan(path, score, is_nsfw):
    return FrameAnalysis(
        frame_path=path, is_nsfw=is_nsfw, nsfw_score=score, detections=[]
    )


# -----------------------------------------------------------------------
# 1. select_mapping_frames
# -----------------------------------------------------------------------

class TestSelectMappingFrames:
    def test_enough_nsfw_selects_nsfw_frames(self):
        """检出足量 NSFW → 只选 NSFW 帧, 按分数降序, 截断到 frames_cap。"""
        frames = _unit_frames(count=4)
        scans = [
            _scan(frames[0], 0.0, False),
            _scan(frames[1], 0.60, True),
            _scan(frames[2], 0.90, True),
            _scan(frames[3], 0.75, True),
        ]
        selected, mode = select_mapping_frames(frames, scans, frames_cap=2, min_nsfw_frames=1)
        assert mode == "nsfw"
        # 分数降序: 0.90(frame_002) > 0.75(frame_003) > 0.60(frame_001)
        assert selected == [frames[2], frames[3]]
        assert frames[1] not in selected  # 被 cap 截掉

    def test_zero_nsfw_falls_back_not_skip(self):
        """0 检出(320n 对男性向/男同内容系统性漏检) → 回退抽样, 不丢弃场景。"""
        frames = _unit_frames(count=3)
        scans = [_scan(f, 0.0, False) for f in frames]
        selected, mode = select_mapping_frames(frames, scans, frames_cap=3, min_nsfw_frames=1)
        assert mode == "fallback"
        assert selected  # 非空: 场景仍会产出标签/封面
        assert set(selected).issubset(set(frames))

    def test_below_min_frames_falls_back(self):
        """NSFW 帧数 < min_nsfw_frames → 回退。"""
        frames = _unit_frames(count=4)
        scans = [
            _scan(frames[0], 0.8, True),  # 只有 1 个 NSFW
            _scan(frames[1], 0.0, False),
            _scan(frames[2], 0.0, False),
            _scan(frames[3], 0.0, False),
        ]
        selected, mode = select_mapping_frames(frames, scans, frames_cap=3, min_nsfw_frames=2)
        assert mode == "fallback"
        assert selected

    def test_min_one_uses_single_nsfw_frame(self):
        """min_nsfw_frames=1 且恰有 1 个 NSFW 帧 → 走 nsfw 分支(不回退)。"""
        frames = _unit_frames(count=3)
        scans = [
            _scan(frames[0], 0.0, False),
            _scan(frames[1], 0.66, True),
            _scan(frames[2], 0.0, False),
        ]
        selected, mode = select_mapping_frames(frames, scans, frames_cap=3, min_nsfw_frames=1)
        assert mode == "nsfw"
        assert selected == [frames[1]]

    def test_fallback_respects_cap(self):
        """回退抽样同样受 frames_cap 约束。"""
        frames = _unit_frames(count=5)
        scans = [_scan(f, 0.0, False) for f in frames]
        selected, mode = select_mapping_frames(frames, scans, frames_cap=2, min_nsfw_frames=1)
        assert mode == "fallback"
        assert len(selected) == 2
        assert set(selected).issubset(set(frames))


# -----------------------------------------------------------------------
# 2. sample_fallback_frames (模块级)
# -----------------------------------------------------------------------

class TestSampleFallbackFrames:
    def test_small_unit_returns_all(self):
        """帧数 <= 上限 → 原样返回全部。"""
        frames = _unit_frames(count=3)
        assert sample_fallback_frames(frames, max_samples=3, per_scene=3) == frames

    def test_large_flat_dir_samples_capped(self):
        """平铺大目录 → 抽样到上限。"""
        frames = [f"/frames/frame_{i:03d}.jpg" for i in range(20)]
        out = sample_fallback_frames(frames, max_samples=5, per_scene=3)
        assert len(out) == 5
        assert set(out).issubset(set(frames))

    def test_multi_scene_covers_each_scene(self):
        """多场景: 每个场景至少覆盖 1 帧。"""
        frames = []
        for u in range(3):
            frames += _unit_frames(unit=u, count=4)
        out = sample_fallback_frames(frames, max_samples=6, per_scene=2)
        assert len(out) <= 6
        covered = {Path(p).parent.name for p in out}
        assert covered == {"unit_0", "unit_1", "unit_2"}


# -----------------------------------------------------------------------
# 3. 无回归: 分析器方法仍可用(委派给模块级实现)
# -----------------------------------------------------------------------

class TestAnalyzerDelegation:
    def test_method_matches_module_function(self):
        frames = [f"/frames/unit_0/frame_{i:03d}.jpg" for i in range(10)]
        analyzer = TwoStageNSFWAnalyzer.__new__(TwoStageNSFWAnalyzer)  # 跳过 __init__(不加载模型)
        got = analyzer._sample_fallback_frames(frames, max_samples=4, per_scene=2)
        exp = sample_fallback_frames(frames, max_samples=4, per_scene=2)
        assert got == exp


# -----------------------------------------------------------------------
# 4. select_scene_frames (合并分析模式: 选帧 + 封面帧)
# -----------------------------------------------------------------------

class TestSelectSceneFrames:
    def test_nsfw_cover_is_highest_score(self):
        """有足量 NSFW → 选中 NSFW 帧, 封面 = 分数最高帧。"""
        frames = _unit_frames(count=4)
        scans = [
            _scan(frames[0], 0.0, False),
            _scan(frames[1], 0.60, True),
            _scan(frames[2], 0.95, True),
            _scan(frames[3], 0.75, True),
        ]
        selected, cover, mode = select_scene_frames(frames, scans, frames_cap=2, min_nsfw_frames=1)
        assert mode == "nsfw"
        assert cover == frames[2]          # 0.95 最高
        assert selected[0] == cover        # 封面即选中列表首项
        assert len(selected) == 2

    def test_fallback_cover_is_median_frame(self):
        """0 检出 → 回退抽样, 封面取场景内中位帧(仍产出封面, 不丢弃场景)。"""
        frames = _unit_frames(count=5)
        scans = [_scan(f, 0.0, False) for f in frames]
        selected, cover, mode = select_scene_frames(frames, scans, frames_cap=3, min_nsfw_frames=1)
        assert mode == "fallback"
        assert cover == frames[2]          # 5 帧的中位
        assert selected and set(selected).issubset(set(frames))

    def test_empty_frames(self):
        selected, cover, mode = select_scene_frames([], [], frames_cap=3, min_nsfw_frames=1)
        assert cover is None


# -----------------------------------------------------------------------
# 5. analyze_frames_once (合并模式: 多帧一次调用)
# -----------------------------------------------------------------------

class TestAnalyzeFramesOnce:
    def _client(self):
        from src.core.vlm import OllamaVLMClient
        with patch("src.core.vlm.OllamaVLMClient._detect_endpoint", return_value="ollama"):
            return OllamaVLMClient(config=None, model="test-model", base_url="http://localhost:11434")

    def test_single_call_with_all_frames(self, tmp_path):
        """多帧只发一次请求, 且 images 含全部帧。"""
        paths = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
            paths.append(str(p))

        client = self._client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "双人,后入,卧室"}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.core.vlm.requests.post", return_value=mock_resp) as mock_post:
            out = client.analyze_frames_once(paths, "测试提示词")

        assert out == "双人,后入,卧室"
        assert mock_post.call_count == 1              # 关键: 只调用一次
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert len(payload["images"]) == 3            # 三帧一次送入
        assert payload["prompt"] == "测试提示词"

    def test_empty_frames_no_call(self):
        client = self._client()
        with patch("src.core.vlm.requests.post") as mock_post:
            assert client.analyze_frames_once([], "x") == ""
        assert mock_post.call_count == 0

    def test_error_returns_empty(self):
        import requests as req
        client = self._client()
        with patch("src.core.vlm.requests.post", side_effect=req.exceptions.ConnectionError("x")):
            assert client.analyze_frames_once(["/nope.jpg"], "x") == ""

