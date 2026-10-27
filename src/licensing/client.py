"""授权 API 客户端：HTTP 调用 + Ed25519 响应验签。

安全红线（集成规范 §2/§5/§9）：
- API_BASE / PRODUCT_ID / ED25519_PUBLIC_KEY 硬编码常量，禁止配置文件、
  环境变量或远程下发；
- 可能授予授权的 2xx 响应必须验签（x-license-signature，被签数据 = 响应 body
  原始字节），缺失/畸形/不匹配一律按不可信处理（NetworkError），绝不采信 body；
  4xx 错误响应在生产服务端不签名（v1 err() 路径），若带签名头仍强验，
  无签名头仅允许分类为拒绝性业务错误（详见 _verify_signature）；
- 网络层只访问 API_BASE 单一域名，无任何备用/自定义服务器入口；
- 业务错误（invalid_code 等）绝不重试；仅网络异常重试，最多 2 次、指数退避。
"""

import base64
import binascii
import time
from dataclasses import dataclass

import httpx

try:
    from ._license_key import ED25519_PUBLIC_KEY_B64
except ImportError as _exc:  # fail-closed：无公钥则无法验签，拒绝运行（无任何绕过）
    raise RuntimeError(
        "缺少授权公钥常量 src/licensing/_license_key.py（该文件不入 git，由构建注入）。"
        "请运行: uv run --active python scripts/inject_license_key.py "
        "<server/.ed25519-public-key.txt>"
    ) from _exc

API_BASE = "https://license.cloudcraft.xin"
PRODUCT_ID = "media-tag"
REQUEST_TIMEOUT = 10.0  # 秒
NETWORK_RETRIES = 2  # 仅网络异常重试，最多 2 次
RETRY_BACKOFF_SECONDS = 1.0  # 指数退避基数：1s, 2s

_RAW_PUBLIC_KEY = base64.b64decode(ED25519_PUBLIC_KEY_B64)
if len(_RAW_PUBLIC_KEY) != 32:
    raise RuntimeError("Ed25519 公钥长度异常（应为 raw 32 字节的 base64）")

_sleep = time.sleep  # 测试可替换


class LicenseError(Exception):
    """业务错误：服务端已应答且响应验签通过。"""

    def __init__(self, error_code: str, message: str = "", data: dict | None = None):
        super().__init__(message or error_code)
        self.error_code = error_code
        self.message = message or error_code
        self.data = data or {}


class NetworkError(Exception):
    """网络失败/超时/响应验签失败——不可达或不可信，绝不采信任何 body 内容。"""


@dataclass
class ActivateResult:
    token: str
    plan: str
    plan_type: str
    features: dict
    expires_at: str | None
    email: str | None
    devices_used: int
    devices_max: int


@dataclass
class VerifyResult:
    status: str  # active | revoked | expired
    plan: str | None
    plan_type: str | None
    features: dict
    expires_at: str | None
    days_to_expiry: int | None


@dataclass
class StatusResult:
    status: str
    plan: str | None
    features: dict
    expires_at: str | None
    days_to_expiry: int | None
    email: str | None
    devices_used: int
    devices_max: int
    devices: list  # [{device_name, last_seen_at, is_current}]


def _verify_signature(response: httpx.Response, required: bool) -> None:
    """校验 x-license-signature（ed25519:<base64签名>，被签数据=body 原始字节）。

    - required=True（2xx 成功响应，可能授予授权）：签名缺失/畸形/不匹配一律
      NetworkError——绝不采信 body；
    - required=False（4xx 错误响应）：生产服务端的错误响应不带签名（v1.ts 的
      err() 路径，与规范 §4 有偏差）。无签名头时允许按 HTTP 状态+错误码分类——
      错误路径只会产生拒绝性结果（锁定/清除 token/重试），伪造错误响应无法获得
      任何授权状态（授权只能来自带签名的 200）；若带了签名头则必须验签通过。
    """
    header = response.headers.get("x-license-signature", "")
    scheme, _, sig_b64 = header.partition(":")
    raw_sig: bytes | None = None
    if scheme.strip().lower() == "ed25519" and sig_b64.strip():
        try:
            raw_sig = base64.b64decode(sig_b64.strip(), validate=True)
        except (binascii.Error, ValueError):
            raw_sig = None
    if raw_sig is None:
        if not header.strip():
            if required:
                raise NetworkError("成功响应缺少签名头（不可信响应）")
            return  # 无签名的错误响应：仅可分类为业务错误（拒绝性路径）
        raise NetworkError("响应签名头畸形（不可信响应）")
    if len(raw_sig) != 64:
        raise NetworkError("响应签名长度异常（不可信响应）")

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(_RAW_PUBLIC_KEY).verify(
            raw_sig, response.content
        )
    except InvalidSignature as exc:
        raise NetworkError("响应验签失败（不可信响应）") from exc


def _make_client() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT)


def _send(
    method: str,
    path: str,
    json_body: dict | None,
    token: str | None = None,
) -> httpx.Response:
    """发送请求并验签。仅网络类失败重试（含验签失败的瞬时不可信），业务错误不重试。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_exc: Exception | None = None
    for attempt in range(NETWORK_RETRIES + 1):
        if attempt:
            _sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        try:
            with _make_client() as http:
                resp = http.request(
                    method, API_BASE + path, json=json_body, headers=headers
                )
            _verify_signature(resp, required=resp.status_code < 400)
            return resp
        except (httpx.HTTPError, NetworkError) as exc:
            last_exc = exc
    raise NetworkError(f"请求失败（已重试 {NETWORK_RETRIES} 次）: {last_exc}")


def _error_from_response(resp: httpx.Response) -> LicenseError:
    """验签已通过的错误响应 → LicenseError。"""
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    code = data.get("error") or f"http_{resp.status_code}"
    return LicenseError(str(code), str(data.get("message") or ""), data)


def _check_product(data: dict) -> None:
    product = data.get("product")
    if product is not None and product != PRODUCT_ID:
        raise LicenseError("product_mismatch", f"响应产品不匹配: {product}", data)


def _parse_iso(value) -> str | None:
    return str(value) if value else None


def activate(code: str, fingerprint: str, device_name: str) -> ActivateResult:
    """POST /v1/activate。业务失败抛 LicenseError；网络/验签失败抛 NetworkError。"""
    resp = _send(
        "POST",
        "/v1/activate",
        {"code": code.strip(), "fingerprint": fingerprint, "device_name": device_name or ""},
    )
    if resp.status_code != 200:
        raise _error_from_response(resp)
    data = resp.json()
    _check_product(data)
    token = data.get("token")
    if not token:
        raise NetworkError("激活响应缺少 token（不可信响应）")
    return ActivateResult(
        token=str(token),
        plan=str(data.get("plan") or ""),
        plan_type=str(data.get("plan_type") or ""),
        features=dict(data.get("features") or {}),
        expires_at=_parse_iso(data.get("expires_at")),
        email=data.get("email"),
        devices_used=int(data.get("devices_used") or 0),
        devices_max=int(data.get("devices_max") or 0),
    )


def verify(token: str, fingerprint: str) -> VerifyResult:
    """POST /v1/verify。revoked/expired 是 200 业务状态，不是错误。"""
    resp = _send("POST", "/v1/verify", {"fingerprint": fingerprint}, token=token)
    if resp.status_code != 200:
        raise _error_from_response(resp)
    data = resp.json()
    _check_product(data)
    status = data.get("status")
    if status not in ("active", "revoked", "expired"):
        raise NetworkError("verify 响应状态异常（不可信响应）")
    return VerifyResult(
        status=str(status),
        plan=data.get("plan"),
        plan_type=data.get("plan_type"),
        features=dict(data.get("features") or {}),
        expires_at=_parse_iso(data.get("expires_at")),
        days_to_expiry=data.get("days_to_expiry"),
    )


def get_status(token: str) -> StatusResult:
    """GET /v1/status：方案/设备列表（设置页设备管理用）。"""
    resp = _send("GET", "/v1/status", None, token=token)
    if resp.status_code != 200:
        raise _error_from_response(resp)
    data = resp.json()
    _check_product(data)
    devices_raw = data.get("devices")
    return StatusResult(
        status=str(data.get("status") or ""),
        plan=data.get("plan"),
        features=dict(data.get("features") or {}),
        expires_at=_parse_iso(data.get("expires_at")),
        days_to_expiry=data.get("days_to_expiry"),
        email=data.get("email"),
        devices_used=int(data.get("devices_used") or 0),
        devices_max=int(data.get("devices_max") or 0),
        devices=list(devices_raw) if isinstance(devices_raw, list) else [],
    )


def deactivate(token: str) -> dict:
    """POST /v1/deactivate：解绑本设备。返回服务端响应 dict（含 success/devices_used）。"""
    resp = _send("POST", "/v1/deactivate", {}, token=token)
    if resp.status_code != 200:
        raise _error_from_response(resp)
    data = resp.json()
    if not data.get("success"):
        raise LicenseError("deactivate_failed", str(data.get("message") or "解绑失败"), data)
    return data
