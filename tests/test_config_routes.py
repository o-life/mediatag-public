"""设置页「上次选择的提示词模板」的读写接口。

修复背景：前端会 GET/PUT ``/config/vlm-prompt/last-template``，但服务端只有单段的
``/config/{key}``，多一段的路径匹配不上 ⇒ 该"记忆"永远失效（读取被静默忽略）。
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M


class _FakeConfig:
    def __init__(self):
        self.store = {}

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value


@pytest.fixture
def cfg(monkeypatch):
    fake = _FakeConfig()
    monkeypatch.setattr(M, "config_manager", fake)
    return fake


class TestLastTemplate:
    def test_round_trip(self, cfg):
        client = TestClient(M.app)
        assert client.get("/config/vlm-prompt/last-template").json()["value"] == ""

        assert client.put("/config/vlm-prompt/last-template",
                          json={"value": "成人内容模板"}).status_code == 200
        assert client.get("/config/vlm-prompt/last-template").json()["value"] == "成人内容模板"
        assert cfg.get("vlm_prompt_last_template") == "成人内容模板"
