"""授权模式解析：每次启动后台 verify 一次，结果决定 完整版(PRO)/免费版(FREE)。

产品定案（docs/界面设计/07-授权管理.md §1.2，2026-09-22）：
- 无任何"锁定屏/启动门"——主界面永远直接呈现，模式只影响额度（免费版 100 文件/1 vault）；
- 被动取消激活（吊销/强制解绑/换机）与主动解绑 → 免费版并清除本地激活信息，
  清理前做额度库存快照（quota.sync_from_stock）；
- 网络失败/验签失败/限流 → 沿用本地缓存授权状态（离线宽限，产品决策）；
  无缓存（新装）→ 免费版；
- 无"过期"概念（当前全部永久版）；expired 为防御分支，同样转免费版。
"""

import logging
from dataclasses import dataclass

from . import client, fingerprint, quota, state, store

_log = logging.getLogger("licensing.gate")

SUPPORT_EMAIL = "support@cloudcraft.xin"

MSG_REVOKED_NOTICE = (
    "授权已取消，MediaTag 已切换为免费版（累计可处理 100 个文件、1 个 vault）。\n"
    "已有数据不受影响。如有疑问请联系 " + SUPPORT_EMAIL
)
MSG_INVALID_NOTICE = (
    "本机的激活信息已失效（可能在其他设备重新激活），已切换为免费版。\n"
    "已有数据不受影响。如有疑问请联系 " + SUPPORT_EMAIL
)
MSG_EXPIRED_NOTICE = (
    "授权状态异常（expired），已切换为免费版。\n"
    "请联系 " + SUPPORT_EMAIL + " 核实。"
)


@dataclass
class ModeResolution:
    mode: state.Mode
    reason: str  # active/no_token/revoked/invalid_token/device_mismatch/expired/offline_cached/offline_no_cache/fingerprint_error/error
    notice: str | None = None  # 需要一次性告知用户的内容（主线程弹非阻断提示）


def _apply(mode: state.Mode, reason: str, notice: str | None = None) -> ModeResolution:
    state.set_mode(mode, reason)
    return ModeResolution(mode=mode, reason=reason, notice=notice)


def _from_cache(reason: str) -> ModeResolution:
    """网络不可达/暂时无法核验：沿用本地缓存授权状态（离线宽限）。"""
    info = store.load_license_info() or {}
    if info.get("status") == "active":
        return _apply(state.Mode.PRO, "offline_cached")
    return _apply(state.Mode.FREE, "offline_no_cache")


def _deactivate(reason: str, notice: str) -> ModeResolution:
    """被动取消激活：库存快照 → 清本地激活信息 → 免费版。"""
    try:
        quota.sync_from_stock()
    except Exception as exc:
        _log.warning("额度库存快照失败（忽略）: %s", exc)
    store.clear()
    return _apply(state.Mode.FREE, reason, notice)


def resolve_mode() -> ModeResolution:
    token = store.load_token()
    if not token:
        return _apply(state.Mode.FREE, "no_token")

    try:
        fp = fingerprint.get_fingerprint()
    except fingerprint.FingerprintError as exc:
        _log.error("设备指纹读取失败: %s", exc)
        return _from_cache("fingerprint_error")

    try:
        res = client.verify(token, fp)
    except client.LicenseError as exc:
        if exc.error_code in ("invalid_token", "device_mismatch"):
            return _deactivate(exc.error_code, MSG_INVALID_NOTICE)
        # rate_limited 等暂态业务错误：与断网同策略，沿用缓存
        _log.warning("verify 业务错误 %s: %s（沿用缓存）", exc.error_code, exc.message)
        return _from_cache(f"biz_{exc.error_code}")
    except client.NetworkError:
        return _from_cache("offline")

    # 响应已验签通过（HTTP 200 业务状态）
    if res.status == "active":
        from datetime import datetime

        store.save_license_info(
            {
                "source": "verify",
                "status": res.status,
                "plan": res.plan,
                "plan_type": res.plan_type,
                "features": res.features,
                "expires_at": res.expires_at,
                "days_to_expiry": res.days_to_expiry,
                "verified_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return _apply(state.Mode.PRO, "active")
    if res.status == "revoked":
        return _deactivate("revoked", MSG_REVOKED_NOTICE)
    if res.status == "expired":
        return _apply(state.Mode.FREE, "expired", MSG_EXPIRED_NOTICE)

    # 未知状态（防御）：按缓存处理
    return _from_cache(f"unknown_{res.status}")
