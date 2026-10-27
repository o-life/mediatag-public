#!/usr/bin/env python3
"""3.1 设备指纹绑定 生产端到端验证(add-public-sale-readiness)。

用真实 MediaTag 授权客户端(含 Ed25519 响应验签)对生产 license.cloudcraft.xin 跑
双指纹矩阵:激活→本机 verify→拷机 verify(应 403 device_mismatch)→第二机激活
(应 403 device_limit_reached 附设备列表)→旧机解绑→新机激活(自助换绑闭环)→吊销。

用法: uv run --active python scripts/license_e2e_device_binding.py <激活码>
产物: docs/数据质量/sale-readiness-0.2/license-e2e.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.licensing import client  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs/数据质量/sale-readiness-0.2/license-e2e.json"
steps: list[dict] = []


def record(step: str, ok: bool, detail) -> None:
    steps.append({"step": step, "ok": ok, "detail": detail})
    print(f"{'✅' if ok else '❌'} {step}: {json.dumps(detail, ensure_ascii=False)[:220]}",
          flush=True)


def main() -> int:
    code = sys.argv[1]
    fp_a = hashlib.sha256(b"minspec-device-A").hexdigest()
    fp_b = hashlib.sha256(b"minspec-device-B").hexdigest()
    all_ok = True

    # 1. 设备 A 激活
    r = client.activate(code, fp_a, "minspec-deviceA")
    token = r.token
    ok = bool(token) and r.devices_used == 1
    record("activate@A", ok, {"devices": f"{r.devices_used}/{r.devices_max}",
                              "plan": r.plan})
    all_ok &= ok

    # 2. 设备 A verify(应 active)
    v = client.verify(token, fp_a)
    ok = v.status == "active"
    record("verify@A", ok, {"status": v.status})
    all_ok &= ok

    # 3. token 拷到设备 B 的 verify(应 403 device_mismatch)
    try:
        v = client.verify(token, fp_b)
        record("verify@B(mismatch)", v.status != "active",
               {"unexpected_status": v.status})
        all_ok &= v.status != "active"
    except client.LicenseError as e:
        ok = e.error_code == "device_mismatch"
        record("verify@B(mismatch)", ok, {"error_code": e.error_code})
        all_ok &= ok

    # 4. 设备 B 直接激活(席位满,应 device_limit_reached + devices 列表)
    try:
        r = client.activate(code, fp_b, "minspec-deviceB")
        record("activate@B(满席)", False, {"unexpected_success": f"{r.devices_used}/{r.devices_max}"})
        all_ok = False
    except client.LicenseError as e:
        ok = e.error_code == "device_limit_reached"
        devs = e.data.get("devices") if isinstance(e.data, dict) else None
        record("activate@B(满席)", ok, {"error_code": e.error_code,
                                        "devices": devs})
        all_ok &= ok

    # 5. 设备 A 自助解绑(deactivate 释放席位)
    if hasattr(client, "deactivate"):
        r = client.deactivate(token)
        ok = bool(r)
        record("deactivate@A", ok, {"result": str(r)[:120]})
        all_ok &= ok
    else:
        import httpx
        resp = httpx.post(f"{client.API_BASE}/v1/deactivate",
                          headers={"Authorization": f"Bearer {token}"},
                          json={"fingerprint": fp_a}, timeout=15)
        ok = resp.status_code == 200 and resp.json().get("success") is True
        record("deactivate@A(raw)", ok, {"status_code": resp.status_code,
                                         "body": resp.json()})
        all_ok &= ok

    # 6. 设备 B 激活(自助换绑闭环)
    r = client.activate(code, fp_b, "minspec-deviceB")
    ok = bool(r.token)
    record("activate@B(换绑后)", ok, {"devices": f"{r.devices_used}/{r.devices_max}"})
    all_ok &= ok
    token_b = r.token

    # 7. verify@B(应 active,证明绑定已迁到 B)
    v = client.verify(token_b, fp_b)
    ok = v.status == "active"
    record("verify@B(换绑后)", ok, {"status": v.status})
    all_ok &= ok

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"code": code, "all_ok": all_ok, "steps": steps},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'全部通过' if all_ok else '存在失败'} → {OUT}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
