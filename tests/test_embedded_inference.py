"""EmbeddedLlamaServer 单元测试——假二进制,不加载真模型(gate 可跑)。"""

import hashlib
import sys
import time

import pytest
import requests

import src.core.embedded_inference as ei
from src.core.embedded_inference import (
    EmbeddedInferenceError,
    EmbeddedLlamaServer,
    get_embedded_server,
    resolve_runtime,
    verify_model_asset,
)

FAKE_SERVER = """#!/usr/bin/env python3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def server_cfg(tmp_path):
    fake = tmp_path / "fake_llama.py"
    fake.write_text(FAKE_SERVER)
    fake.chmod(0o755)
    model = tmp_path / "fake-model.gguf"
    model.write_bytes(b"fake-model-bytes")  # 首启 SHA 校验需要真实存在的文件
    return {
        "binary": str(fake),
        "model": str(model),
        "mmproj": "",
        "log_path": str(tmp_path / "llama.log"),
    }


def _wait_healthy(srv, timeout=35):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if srv.is_healthy():
            return True
        time.sleep(0.5)
    return False


def test_start_healthy_and_idempotent(server_cfg):
    srv = EmbeddedLlamaServer(server_cfg)
    try:
        url1 = srv.ensure_running()
        assert requests.get(f"{url1}/health", timeout=3).status_code == 200
        assert srv.is_healthy()
        url2 = srv.ensure_running()
        assert url2 == url1
    finally:
        srv.stop()


def test_crash_restarts_once_same_port(server_cfg):
    srv = EmbeddedLlamaServer(server_cfg)
    try:
        url1 = srv.ensure_running()
        srv.proc.kill()  # 模拟崩溃(非主动 stop,看护线程应自动重启一次)
        assert _wait_healthy(srv), "kill 后 30s 内未恢复健康"
        assert srv._restarts == 1
        assert srv.base_url == url1, "重启后 base_url 必须不变(客户端零感知)"
        assert srv.proc is not None and srv.proc.poll() is None
    finally:
        srv.stop()


def test_restart_budget_exhausted_marks_failed(server_cfg):
    srv = EmbeddedLlamaServer(server_cfg)
    try:
        srv.ensure_running()
        for _ in range(3):  # 预算 3 次:前 3 杀每次都自愈
            srv.proc.kill()
            deadline = time.time() + 30
            while time.time() < deadline and not srv.is_healthy():
                time.sleep(0.3)
        srv.proc.kill()  # 第 4 杀:预算耗尽 → 置失败态
        deadline = time.time() + 20
        while time.time() < deadline and not srv.failed_reason:
            time.sleep(0.3)
        assert srv.failed_reason, "预算耗尽未置失败态"
        with pytest.raises(RuntimeError):
            srv.ensure_running()
    finally:
        srv.stop()


def test_stop_terminates_process(server_cfg):
    srv = EmbeddedLlamaServer(server_cfg)
    srv.ensure_running()
    srv.stop()
    assert srv.proc is None and srv.port is None and not srv.is_healthy()


def test_singleton_identity():
    cfg = type("C", (), {"get": staticmethod(lambda k, d=None: {})})()
    assert get_embedded_server(cfg) is get_embedded_server(cfg)


# ── 任务 1.1/1.2:资源解析、SHA256 校验、打包态隔离 ──────────────────


def test_resolve_missing_assets_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ei, "_is_frozen", lambda: True)
    monkeypatch.setattr(ei, "_bundle_root", lambda: tmp_path / "bundle")
    monkeypatch.setattr(ei, "_data_root", lambda: tmp_path / "data")
    with pytest.raises(EmbeddedInferenceError) as e:
        resolve_runtime({})
    assert e.value.code == "INF_MISSING_ASSET"
    assert "重新安装" in str(e.value)


def test_resolve_prefers_bundle_when_frozen(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    model_dir = bundle / "models" / "qwen3-vl"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3VL-8B-Instruct-Q4_K_M.gguf").write_bytes(b"m")
    bin_dir = bundle / "bundled_bin" / "llama_cpp" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-server").write_bytes(b"b")
    monkeypatch.setattr(ei, "_is_frozen", lambda: True)
    monkeypatch.setattr(ei, "_bundle_root", lambda: bundle)
    monkeypatch.setattr(ei, "_data_root", lambda: tmp_path / "data")
    rt = resolve_runtime({})
    assert str(bundle) in rt["model"] and str(bundle) in rt["binary"]


def test_verify_sha256_ok_cache_and_tamper(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"hello-model")
    good = hashlib.sha256(b"hello-model").hexdigest()
    log = tmp_path / "logs"
    assert verify_model_asset(str(model), log) == good  # 基线记录
    assert verify_model_asset(str(model), log, good) == good  # 命中缓存
    mtime = model.stat().st_mtime_ns  # 篡改内容
    model.write_bytes(b"evil-content")
    with pytest.raises(EmbeddedInferenceError) as e:
        verify_model_asset(str(model), log, good)
    assert e.value.code == "INF_ASSET_CORRUPT"
    model.write_bytes(b"hello-model2")  # mtime 变化 → 重算,清单不符同样拦截
    with pytest.raises(EmbeddedInferenceError):
        verify_model_asset(str(model), log, good)
    assert mtime != model.stat().st_mtime_ns


def test_error_message_is_user_readable():
    e = EmbeddedInferenceError("INF_CRASHED", "推理引擎反复崩溃", "请重启应用或联系支持")
    assert "推理引擎反复崩溃" in str(e) and "重启应用" in str(e)


def test_frozen_forces_embedded(monkeypatch, tmp_path):
    calls = {}

    class FakeSrv:
        def ensure_running(self):
            calls["ran"] = True
            return "http://127.0.0.1:1"

        model_name = "fake"

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("MEDIA_DEV_ENDPOINTS", raising=False)
    monkeypatch.setattr(ei, "get_embedded_server", lambda cfg: FakeSrv())

    from src.core.vlm import create_vlm_client

    cfg = FakeConfig2({})
    client = create_vlm_client(cfg)
    assert calls.get("ran") is True and client is not None


class FakeConfig2:
    def __init__(self, blobs):
        self._b = blobs

    def get(self, key, default=None):
        return self._b.get(key, default)

    def get_vlm_config(self):
        return {"timeout": 5}
