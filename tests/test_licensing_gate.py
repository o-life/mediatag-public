"""licensing 模块单元测试（纯逻辑、离线，不触网、不依赖系统钥匙串）。

覆盖：
- client：Ed25519 响应验签（正常/篡改/缺头/错钥）、业务错误分类、网络重试策略；
- gate：启动模式判定表（免费额度模式：完整版/免费版 + 离线宽限 + 取消激活清理）；
- quota：全局累计计数器（只增不减、库存快照、上限判定）；
- store：本地文件存储（0600）读写清除 + 损坏文件容忍；
- fingerprint：IOPlatformUUID 提取与稳定性。
"""

import base64
import sqlite3
import stat
import sys

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, "src")

from licensing import client, fingerprint, gate, quota, state, store  # noqa: E402


# ── 测试用 Ed25519 密钥对（与生产公钥无关，仅用于构造签名响应） ──────
def _make_signer():
    sk = Ed25519PrivateKey.generate()
    pub_b64 = base64.b64encode(sk.public_key().public_bytes_raw()).decode()
    return sk, pub_b64


@pytest.fixture()
def signer(monkeypatch):
    sk, pub_b64 = _make_signer()
    monkeypatch.setattr(client, "_RAW_PUBLIC_KEY", base64.b64decode(pub_b64))
    return sk


def _signed_response(sk, status_code, payload: dict, headers: dict | None = None):
    resp = httpx.Response(status_code, json=payload)
    sig = base64.b64encode(sk.sign(resp.content)).decode()
    all_headers = {"x-license-signature": f"ed25519:{sig}"}
    if headers:
        all_headers.update(headers)
    for key, value in all_headers.items():
        resp.headers[key] = value
    return resp


def _install_transport(monkeypatch, handler):
    def _fake_make_client():
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(client, "_make_client", _fake_make_client)
    monkeypatch.setattr(client, "_sleep", lambda _s: None)


# ── client.activate / verify ─────────────────────────────────────────

def test_activate_success_signed(monkeypatch, signer):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        data = {
            "status": "active", "token": "tok-1", "product": "media-tag",
            "plan": "永久版", "plan_type": "perpetual", "features": {"basic": True},
            "expires_at": None, "email": "t@example.com",
            "devices_used": 1, "devices_max": 1,
        }
        return _signed_response(signer, 200, data)

    _install_transport(monkeypatch, handler)
    res = client.activate("ABCD-EFGH", "f" * 64, "mac")
    assert res.token == "tok-1"
    assert res.plan == "永久版" and res.features == {"basic": True}
    assert calls[0].read() == (
        b'{"code":"ABCD-EFGH","fingerprint":"' + b"f" * 64 + b'","device_name":"mac"}'
    )


def test_activate_tampered_body_rejected(monkeypatch, signer):
    """签名与 body 不匹配 → NetworkError，绝不采信内容。"""
    def handler(request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(
            200, json={"status": "active", "token": "evil", "product": "media-tag",
                       "plan": "x", "plan_type": "perpetual", "features": {}}
        )
        # 对"另一份内容"签名
        sig = base64.b64encode(signer.sign(b'{"status":"active"}')).decode()
        resp.headers["x-license-signature"] = f"ed25519:{sig}"
        return resp

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.NetworkError):
        client.activate("ABCD-EFGH", "f" * 64, "")


def test_response_missing_signature_header_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "active", "token": "evil"})

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.NetworkError):
        client.verify("tok", "f" * 64)


def test_unsigned_error_response_is_classified(monkeypatch):
    """生产服务端 4xx 不签名：可分类为业务错误（拒绝性路径，无法伪造授权）。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_token", "message": "no"})

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.LicenseError) as excinfo:
        client.verify("tok", "f" * 64)
    assert excinfo.value.error_code == "invalid_token"


def test_error_response_with_bad_signature_rejected(monkeypatch, signer):
    """4xx 若带了签名头则必须验签通过，篡改即不可信。"""
    other_sk, _ = _make_signer()

    def handler(request: httpx.Request) -> httpx.Response:
        resp = httpx.Response(401, json={"error": "invalid_token", "message": "no"})
        sig = base64.b64encode(other_sk.sign(resp.content)).decode()
        resp.headers["x-license-signature"] = f"ed25519:{sig}"
        return resp

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.NetworkError):
        client.verify("tok", "f" * 64)


def test_response_wrong_key_rejected(monkeypatch):
    """用其他密钥签名的响应 → 验签失败。"""
    other_sk, _ = _make_signer()

    def handler(request: httpx.Request) -> httpx.Response:
        return _signed_response(other_sk, 200, {"status": "active", "token": "evil"})

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.NetworkError):
        client.verify("tok", "f" * 64)


def test_activate_invalid_code_no_retry(monkeypatch, signer):
    """业务错误绝不重试：404 invalid_code 只发一次请求。"""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _signed_response(signer, 404, {"error": "invalid_code", "message": "bad code"})

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.LicenseError) as excinfo:
        client.activate("WRONG", "f" * 64, "")
    assert excinfo.value.error_code == "invalid_code"
    assert len(calls) == 1


def test_activate_device_limit_with_devices(monkeypatch, signer):
    devices = [{"device_name": "旧Mac", "last_seen_at": "2026-09-01", "is_current": False}]

    def handler(request: httpx.Request) -> httpx.Response:
        return _signed_response(signer, 403, {
            "error": "device_limit_reached", "message": "limit",
            "devices": devices, "devices_max": 1, "devices_used": 1,
        })

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.LicenseError) as excinfo:
        client.activate("FULL", "f" * 64, "")
    assert excinfo.value.error_code == "device_limit_reached"
    assert excinfo.value.data["devices"] == devices


def test_verify_active_is_200_business_state(monkeypatch, signer):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok-9"
        return _signed_response(signer, 200, {
            "status": "active", "product": "media-tag", "plan": "永久版",
            "plan_type": "perpetual", "features": {"basic": True},
            "expires_at": None, "days_to_expiry": None,
        })

    _install_transport(monkeypatch, handler)
    active = client.verify("tok-9", "f" * 64)
    assert active.status == "active" and active.features == {"basic": True}


def test_verify_revoked_is_200_business_state(monkeypatch, signer):
    def handler(request: httpx.Request) -> httpx.Response:
        return _signed_response(signer, 200, {
            "status": "revoked", "features": {}, "plan": "永久版",
        })

    _install_transport(monkeypatch, handler)
    revoked = client.verify("tok-9", "f" * 64)
    assert revoked.status == "revoked"


def test_verify_401_and_403_business_errors(monkeypatch, signer):
    def handler_401(request: httpx.Request) -> httpx.Response:
        return _signed_response(signer, 401, {"error": "invalid_token", "message": "no"})

    _install_transport(monkeypatch, handler_401)
    with pytest.raises(client.LicenseError) as excinfo:
        client.verify("tok", "f" * 64)
    assert excinfo.value.error_code == "invalid_token"

    def handler_403(request: httpx.Request) -> httpx.Response:
        return _signed_response(signer, 403, {"error": "device_mismatch", "message": "mm"})

    _install_transport(monkeypatch, handler_403)
    with pytest.raises(client.LicenseError) as excinfo:
        client.verify("tok", "f" * 64)
    assert excinfo.value.error_code == "device_mismatch"


def test_network_error_retries_twice_then_succeeds(monkeypatch, signer):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("down")
        return _signed_response(signer, 200, {
            "status": "active", "product": "media-tag", "plan": "p",
            "plan_type": "perpetual", "features": {}, "expires_at": None,
        })

    _install_transport(monkeypatch, handler)
    res = client.verify("tok", "f" * 64)
    assert res.status == "active"
    assert len(calls) == 3  # 首次 + 2 次重试


def test_network_error_exhausts_retries(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("down")

    _install_transport(monkeypatch, handler)
    with pytest.raises(client.NetworkError):
        client.verify("tok", "f" * 64)
    assert len(calls) == 3


# ── store ────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_DIR_OVERRIDE", tmp_path)
    return tmp_path


def test_store_file_roundtrip(tmp_store):
    store.save_token("tok-abc")
    store.save_license_info({"plan": "永久版", "verified_at": "2026-09-22T10:00:00"})
    assert store.load_token() == "tok-abc"
    assert store.load_license_info()["plan"] == "永久版"
    store.clear()
    assert store.load_token() is None
    assert store.load_license_info() is None


def test_store_file_with_0600(tmp_store):
    """授权文件权限必须为 0600。"""
    store.save_token("tok-fb")
    assert store.load_token() == "tok-fb"
    target = tmp_store / "license.json"
    assert target.exists()
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode & 0o777 == 0o600
    store.clear()
    assert not target.exists()
    assert store.load_token() is None


def test_store_corrupted_file_tolerated(tmp_store):
    """损坏的授权文件按无授权处理（不抛异常）。"""
    target = tmp_store / "license.json"
    target.write_text("{not json", encoding="utf-8")
    assert store.load_token() is None
    assert store.load_license_info() is None


# ── gate：启动模式判定（免费额度模式，07-授权管理.md §1.2）────────────

@pytest.fixture(autouse=True)
def _reset_mode():
    state.set_mode(state.Mode.FREE, "test_reset")
    yield


@pytest.fixture()
def gate_env(monkeypatch):
    """隔离 store / verify / 指纹 / 库存快照的假环境。"""
    saved: dict = {"license_info": None, "cleared": False, "stock_synced": 0}

    def _sync():
        saved["stock_synced"] += 1
        return saved["stock_synced"]

    monkeypatch.setattr(fingerprint, "get_fingerprint", lambda: "a" * 64)
    monkeypatch.setattr(store, "load_token", lambda: saved.get("token"))
    monkeypatch.setattr(
        store, "save_license_info", lambda info: saved.update(license_info=info)
    )
    monkeypatch.setattr(store, "load_license_info", lambda: saved.get("license_info"))
    monkeypatch.setattr(store, "clear", lambda: saved.update(cleared=True))
    monkeypatch.setattr(quota, "sync_from_stock", _sync)
    return saved


def test_gate_no_token_free(gate_env):
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert result.reason == "no_token"
    assert result.notice is None
    assert not gate_env["cleared"]


def test_gate_active_pro_and_snapshot(gate_env, monkeypatch):
    gate_env["token"] = "tok-1"
    monkeypatch.setattr(client, "verify", lambda token, fp: client.VerifyResult(
        status="active", plan="永久版", plan_type="perpetual",
        features={"basic": True}, expires_at=None, days_to_expiry=None,
    ))
    result = gate.resolve_mode()
    assert result.mode is state.Mode.PRO
    assert result.reason == "active"
    assert result.notice is None
    assert gate_env["license_info"]["status"] == "active"
    assert gate_env["license_info"]["verified_at"]
    assert state.is_pro()


def test_gate_revoked_downgrades_to_free_and_clears(gate_env, monkeypatch):
    """被动取消激活（吊销）→ 免费版 + 库存快照 + 清本地激活 + 一次性提示。"""
    gate_env["token"] = "tok-1"
    monkeypatch.setattr(client, "verify", lambda token, fp: client.VerifyResult(
        status="revoked", plan=None, plan_type=None, features={},
        expires_at=None, days_to_expiry=None,
    ))
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert result.notice and "免费版" in result.notice
    assert gate_env["cleared"]
    assert gate_env["stock_synced"] == 1


def test_gate_expired_defensive_free_keeps_token(gate_env, monkeypatch):
    gate_env["token"] = "tok-1"
    monkeypatch.setattr(client, "verify", lambda token, fp: client.VerifyResult(
        status="expired", plan=None, plan_type=None, features={},
        expires_at=None, days_to_expiry=None,
    ))
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert result.notice
    assert not gate_env["cleared"]  # 防御分支：保留激活信息


def test_gate_invalid_token_deactivates(gate_env, monkeypatch):
    gate_env["token"] = "tok-1"

    def _raise(token, fp):
        raise client.LicenseError("invalid_token", "no")

    monkeypatch.setattr(client, "verify", _raise)
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert gate_env["cleared"] and gate_env["stock_synced"] == 1
    assert result.notice


def test_gate_device_mismatch_deactivates(gate_env, monkeypatch):
    gate_env["token"] = "tok-1"

    def _raise(token, fp):
        raise client.LicenseError("device_mismatch", "mm")

    monkeypatch.setattr(client, "verify", _raise)
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert gate_env["cleared"] and gate_env["stock_synced"] == 1


def test_gate_offline_with_active_cache_stays_pro(gate_env, monkeypatch):
    """断网/验签失败：沿用本地缓存授权状态（离线宽限，产品决策）。"""
    gate_env["token"] = "tok-1"
    gate_env["license_info"] = {"status": "active", "verified_at": "2026-09-22T10:00:00"}

    def _raise(token, fp):
        raise client.NetworkError("down")

    monkeypatch.setattr(client, "verify", _raise)
    result = gate.resolve_mode()
    assert result.mode is state.Mode.PRO
    assert result.reason == "offline_cached"
    assert result.notice is None
    assert not gate_env["cleared"]


def test_gate_offline_without_cache_free(gate_env, monkeypatch):
    gate_env["token"] = "tok-1"  # 有 token 但从未成功验证过 → 无缓存 → 免费版

    def _raise(token, fp):
        raise client.NetworkError("down")

    monkeypatch.setattr(client, "verify", _raise)
    result = gate.resolve_mode()
    assert result.mode is state.Mode.FREE
    assert result.reason == "offline_no_cache"
    assert not gate_env["cleared"]


def test_gate_rate_limited_falls_back_to_cache(gate_env, monkeypatch):
    """限流是暂态业务错误：与断网同策略，沿用缓存。"""
    gate_env["token"] = "tok-1"
    gate_env["license_info"] = {"status": "active"}

    def _raise(token, fp):
        raise client.LicenseError("rate_limited", "busy")

    monkeypatch.setattr(client, "verify", _raise)
    result = gate.resolve_mode()
    assert result.mode is state.Mode.PRO


# ── quota：免费额度计数器 ─────────────────────────────────────────────

@pytest.fixture()
def tmp_quota(monkeypatch, tmp_path):
    monkeypatch.setattr(quota, "_DIR_OVERRIDE", tmp_path)
    return tmp_path


def test_quota_register_and_remaining(tmp_quota):
    assert quota.processed() == 0 and quota.remaining() == 100
    quota.register_imported(30)
    assert quota.processed() == 30 and quota.remaining() == 70
    assert not quota.exceeded()
    quota.register_imported(0)  # 无导入不变化
    assert quota.processed() == 30


def test_quota_exceeded_at_limit(tmp_quota):
    quota.register_imported(100)
    assert quota.exceeded() and quota.remaining() == 0


def test_quota_only_grows(tmp_quota):
    """计数器只增不减：删除文件/任务不返还额度。"""
    quota.register_imported(40)
    quota.register_imported(10)
    assert quota.processed() == 50
    # 没有任何 API 能减少它（除取消激活时的库存快照 max）


def test_quota_sync_from_stock(tmp_quota):
    """取消激活库存快照：counter = max(counter, 全库任务总数)。"""
    themes = tmp_quota / "data" / "themes"
    for name, n in (("A", 5), ("B", 7)):
        db = themes / name / "data.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE tasks (id INTEGER)")
        conn.executemany("INSERT INTO tasks VALUES (?)", [(i,) for i in range(n)])
        conn.commit()
        conn.close()

    quota.sync_from_stock()
    assert quota.processed() == 12  # max(0, 12)：存量>原计数

    quota.register_imported(3)  # 15
    quota.sync_from_stock()  # max(15, 12) → 15：库存低于计数不回退
    assert quota.processed() == 15


def test_quota_sync_survives_vault_recreate(tmp_quota):
    """删 vault 重建：计数器存应用级，重建后的空库不会清零额度。"""
    quota.register_imported(80)
    quota.sync_from_stock()  # 无任何主题库 → max(80, 0)
    assert quota.processed() == 80


def test_quota_sync_corrupt_db_skipped(tmp_quota):
    themes = tmp_quota / "data" / "themes"
    bad = themes / "bad" / "data.db"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not a db", encoding="utf-8")
    quota.sync_from_stock()  # 损坏库跳过不抛异常
    assert quota.processed() == 0


# ── fingerprint ──────────────────────────────────────────────────────

def test_fingerprint_from_ioreg(monkeypatch):
    sample = (
        '+-o Root  <class IOPlatformExpertDevice>\n'
        '  |   "IOPlatformUUID" = "12345678-90AB-CDEF-1234-567890ABCDEF"\n'
    )

    class _Proc:
        stdout = sample
        returncode = 0

    monkeypatch.setattr(fingerprint.subprocess, "run", lambda *a, **k: _Proc())
    fp1 = fingerprint.get_fingerprint()
    fp2 = fingerprint.get_fingerprint()
    assert len(fp1) == 64 and int(fp1, 16) >= 0  # 64 位十六进制
    assert fp1 == fp2  # 同机稳定不变


def test_fingerprint_missing_uuid_raises(monkeypatch):
    class _Proc:
        stdout = "no uuid here"
        returncode = 0

    monkeypatch.setattr(fingerprint.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(fingerprint.FingerprintError):
        fingerprint.get_fingerprint()


def test_fingerprint_salt_is_load_bearing(monkeypatch):
    """不同盐值（改常量）会产生不同指纹——提醒运维绝不可换盐。"""
    sample = '"IOPlatformUUID" = "12345678-90AB-CDEF-1234-567890ABCDEF"'

    class _Proc:
        stdout = sample
        returncode = 0

    monkeypatch.setattr(fingerprint.subprocess, "run", lambda *a, **k: _Proc())
    fp_default = fingerprint.get_fingerprint()
    monkeypatch.setattr(fingerprint, "_SALT", "other-salt")
    assert fingerprint.get_fingerprint() != fp_default
