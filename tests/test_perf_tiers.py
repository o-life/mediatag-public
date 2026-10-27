"""perf_tiers 档位表单元测试(纯逻辑,不加载真模型)。"""

import pytest

import src.core.perf_tiers as pt
from src.core.perf_tiers import (
    apply_density,
    detect_tier,
    estimated_minutes,
    tier_params,
)


class FakeConfig:
    def __init__(self, perf=None):
        self._perf = perf or {}

    def get(self, key, default=None):
        if key == "perf":
            return self._perf
        return default


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MEDIA_FORCE_TIER", raising=False)


def test_detect_tier_ram_boundaries(monkeypatch):
    for ram, expect in ((64.0, "HIGH"), (32.0, "HIGH"), (31.9, "MID"),
                        (16.0, "MID"), (14.0, "MID"), (13.9, "LOW"), (0.0, "LOW")):
        monkeypatch.setattr(pt, "total_ram_gb", lambda r=ram: r)
        assert detect_tier() == expect, f"ram={ram}"


def test_env_override_beats_ram(monkeypatch):
    monkeypatch.setattr(pt, "total_ram_gb", lambda: 64.0)
    monkeypatch.setenv("MEDIA_FORCE_TIER", "LOW")
    assert detect_tier() == "LOW"


def test_config_override_and_invalid_ignored(monkeypatch):
    monkeypatch.setattr(pt, "total_ram_gb", lambda: 64.0)
    assert detect_tier(FakeConfig({"tier_override": "MID"})) == "MID"
    monkeypatch.delenv("MEDIA_FORCE_TIER", raising=False)
    assert detect_tier(FakeConfig({"tier_override": "ultra"})) == "HIGH"


def test_apply_density_scales_with_floor():
    assert apply_density(5, 60, "HIGH") == (5, 60)
    assert apply_density(5, 60, "LOW") == (2, 24)
    assert apply_density(1, 0, "LOW") == (1, 0)  # 下限 1;不设限保持 0
    assert apply_density(2, 3, "LOW") == (1, 1)  # 缩放后下限钳制


def test_tier_params_unknown_falls_to_low():
    assert tier_params("ultra") == tier_params("LOW")


def test_estimated_minutes():
    assert estimated_minutes(100, "HIGH") == pytest.approx(3.3)  # 2.0s/帧(本机基线)
    assert estimated_minutes(115, "LOW") == pytest.approx(7.7)  # 函数四舍五入到 1 位小数
    assert estimated_minutes(0, "LOW") == 0.0


def test_embedded_server_uses_tier_params(tmp_path):
    from src.core.embedded_inference import EmbeddedLlamaServer

    srv = EmbeddedLlamaServer({"model": "m.gguf", "mmproj": "",
                               "log_path": str(tmp_path / "l.log")}, tier="LOW")
    assert srv.ctx == 8192 and srv.slots == 1
    srv_hi = EmbeddedLlamaServer({"model": "m.gguf", "mmproj": "",
                                  "log_path": str(tmp_path / "l2.log")}, tier="HIGH")
    assert srv_hi.ctx == 32768 and srv_hi.slots == 4
    # 显式配置压过档位
    srv_cfg = EmbeddedLlamaServer({"model": "m.gguf", "mmproj": "", "ctx": 4096,
                                   "slots": 2, "log_path": str(tmp_path / "l3.log")},
                                  tier="LOW")
    assert srv_cfg.ctx == 4096 and srv_cfg.slots == 2
