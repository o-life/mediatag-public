"""性能档位表（add-public-sale-readiness 任务 0.3）。

三档（HIGH/MID/LOW）覆盖三个旋钮：
- 内嵌推理并发：slots + ctx（embedded_inference 消费）
- 抽帧密度：frames_per_scene / max_frames_per_video 倍率（api 抽帧消费）
- 场景检测采样步长：sample_rate（scenedetect 消费）

档位探测 = 物理内存（sysctl hw.memsize）。覆盖入口（低配模拟/测试）：
env `MEDIA_FORCE_TIER` 或配置 `perf.tier_override`，取值 HIGH/MID/LOW。
阈值初版依据 design.md D2 定性预期，**任务 0.2 三机型实测后校准**。
"""

import functools
import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

TIERS = ("HIGH", "MID", "LOW")

# 内存阈值（GB）：HIGH ≥ 32；MID ≥ 14（产品宣传线 16GB 落此档）；LOW < 14
_HIGH_RAM_GB = 32.0
_MID_RAM_GB = 14.0

# 每帧分析耗时预估（秒，含排队）。
# 2026-10-07 本机基线（128GB,LOW 档实测 115 帧/177s ≈ 1.5s/帧）:
# 常数按档位保守放大;真机校准待任务 0.2(mini 已关停,待借 16GB 真机)。
_PER_FRAME_SECONDS: Dict[str, float] = {"HIGH": 2.0, "MID": 3.0, "LOW": 4.0}

_PARAMS: Dict[str, Dict] = {
    "HIGH": {"slots": 4, "ctx": 32768, "density_scale": 1.0, "sample_rate": 1},
    "MID": {"slots": 2, "ctx": 16384, "density_scale": 0.6, "sample_rate": 2},
    "LOW": {"slots": 1, "ctx": 8192, "density_scale": 0.4, "sample_rate": 4},
}


@functools.lru_cache(maxsize=1)
def total_ram_gb() -> float:
    """物理内存 GB；探测失败返回 0（按 LOW 处理，宁降不炸）。"""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 2**30
    except Exception as e:  # noqa: BLE001 - 探测失败按最低档
        logger.warning("内存探测失败,按 LOW 档: %s", e)
        return 0.0


def detect_tier(config=None) -> str:
    """返回当前生效档位；覆盖入口优先（env > 配置 > 内存探测）。"""
    forced = os.environ.get("MEDIA_FORCE_TIER", "")
    if not forced and config is not None:
        try:
            perf = config.get("perf", {}) or {}
            forced = str(perf.get("tier_override") or "")
        except Exception:  # noqa: BLE001 - 配置缺失走探测
            forced = ""
    if forced.upper() in TIERS:
        return forced.upper()
    ram = total_ram_gb()
    if ram >= _HIGH_RAM_GB:
        return "HIGH"
    if ram >= _MID_RAM_GB:
        return "MID"
    return "LOW"


def tier_params(tier: str) -> Dict:
    """档位参数表；未知档位按 LOW（宁降不炸）。"""
    return dict(_PARAMS.get(tier, _PARAMS["LOW"]))


def apply_density(frames_per_scene: int, max_frames: int,
                  tier: str) -> Tuple[int, int]:
    """按档位缩放抽帧密度（下限 1；max_frames ≤0 表示不设限）。"""
    scale = tier_params(tier)["density_scale"]
    fps = max(1, round(frames_per_scene * scale)) if frames_per_scene else frames_per_scene
    mf = max(1, round(max_frames * scale)) if max_frames and max_frames > 0 else max_frames
    return fps, mf


def estimated_minutes(frames: int, tier: str) -> float:
    """按档位单帧耗时预估分析分钟数（UI 展示用；非承诺值）。"""
    return round(frames * _PER_FRAME_SECONDS.get(tier, 14.0) / 60.0, 1)


def active(config=None) -> Dict:
    """一次性取当前档位+参数（分析入口用）。"""
    tier = detect_tier(config)
    params = tier_params(tier)
    logger.info("性能档位: %s (ram=%.0fGB) params=%s", tier, total_ram_gb(), params)
    return {"tier": tier, **params}


def override_source(config=None) -> Optional[str]:
    """返回覆盖来源描述（None=自动探测），供诊断与 UI 标注。"""
    if os.environ.get("MEDIA_FORCE_TIER"):
        return f"env:MEDIA_FORCE_TIER={os.environ['MEDIA_FORCE_TIER']}"
    if config is not None:
        try:
            v = (config.get("perf", {}) or {}).get("tier_override")
            if v:
                return f"config:perf.tier_override={v}"
        except Exception:  # noqa: BLE001
            pass
    return None
