"""VLM（Ollama / 本地权重）就绪判定。

**为什么单独收口**：2026-09-20 实测了一次事故 —— 配置的 `vlm.model` 在 Ollama 里不存在
（模型被删过），而 `src/api/main.py` 的预检在"模型缺失"时走 **`_mark_task_error()`**：
不是跳过打标，而是把任务**直接判死**。调度器又会在几秒内把队列里**每一条**都派发一遍，
于是 **2196 条排队任务全部变成 ERROR**（"清库存"变成"清库存并全部判死"）。

正确的形态是**缓冲**：模型没就绪 ⇒ **不派发、不消费任务**，等就绪后自动继续。
本模块提供两件事：

- :func:`is_vlm_ready` —— 判定当前生效的 VLM 供应商是否可用（供调度器派发前调用）
- :func:`defer_task_for_vlm` —— 把任务**原样放回队列**（状态回到 QUEUED，清掉错误标记）

判定口径与供应商类型一致（`ConfigManager.get_vlm_config()` 的 `provider`）：
- `folder` / `mlx`：本地权重目录 → 只看目录是否存在，不依赖 Ollama
- 其它（`url`）：查 `{base_url}/api/tags`，确认 `model` 在列表里
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

#: 两次就绪探测之间的最小间隔（秒）——避免每条任务都打一次 HTTP
DEFAULT_RECHECK_INTERVAL = 30.0

#: Ollama 探测超时（秒）
PROBE_TIMEOUT = 5.0

#: 本地权重型供应商
_LOCAL_TYPES = ("folder", "mlx")


def is_vlm_ready(get_config: Optional[Callable[[], dict]] = None) -> Tuple[bool, str]:
    """判定 VLM 是否就绪。返回 ``(是否就绪, 原因)``。

    Args:
        get_config: 可选的配置读取函数 ``() -> vlm_config``；缺省自取
            （`ConfigManager.get_vlm_config()`），便于测试注入。

    任何异常都视为**未就绪**（宁可暂停，也不要放行一批注定失败的任务）。
    """
    # ── 内嵌推理优先（与 vlm.create_vlm_client 的选择条件严格一致）──
    # frozen 恒内嵌；配置显式 enabled 亦内嵌；MEDIA_DEV_ENDPOINTS 开发态解绑。
    # 内嵌模式探测"资源可解析"而非进程存活——llama-server 由首次请求惰性拉起
    # （打包用户机没有 Ollama，此闸若按 URL 探测将永不开闸，任务永不派发）。
    try:
        from src.core.config import get_config_manager as _get_cm

        emb_cfg = _get_cm().get("embedded_inference") or {}
    except Exception:
        emb_cfg = {}
    frozen = getattr(sys, "frozen", False)
    dev_override = bool(os.environ.get("MEDIA_DEV_ENDPOINTS"))
    if (emb_cfg.get("enabled") or frozen) and not (frozen and dev_override):
        try:
            from src.core.embedded_inference import resolve_runtime

            rt = resolve_runtime(emb_cfg)
            if Path(rt["binary"]).exists() and Path(rt["model"]).exists():
                return True, f"内嵌推理资源就绪: {Path(rt['model']).name}"
            return False, "内嵌推理资源缺失（运行时/模型文件）"
        except Exception as exc:  # 资源解析失败=未就绪，原因进调度器暂停理由
            return False, f"内嵌推理资源不可用: {exc}"

    try:
        if get_config is None:
            from src.core.config import get_config_manager

            cfg = get_config_manager().get_vlm_config() or {}
        else:
            cfg = get_config() or {}
    except Exception as exc:  # 配置读不出来也算未就绪
        return False, f"读取 VLM 配置失败: {exc}"

    provider = str(cfg.get("provider") or "url").lower()
    model = str(cfg.get("model") or "").strip()
    base_url = str(cfg.get("base_url") or "").rstrip("/")

    if provider in _LOCAL_TYPES:
        if model and Path(model).expanduser().exists():
            return True, f"本地模型目录就绪: {model}"
        return False, f"本地模型目录不存在: {model or '(未配置 model)'}"

    if not base_url:
        return False, "未配置 VLM 服务地址 (vlm.base_url)"
    if not model:
        return False, "未配置 VLM 模型名 (vlm.model)"

    try:
        import requests

        resp = requests.get(f"{base_url}/api/tags", timeout=PROBE_TIMEOUT)
    except Exception as exc:
        return False, f"无法连接 VLM 服务 {base_url}: {exc}"

    if resp.status_code != 200:
        return False, f"VLM 服务 {base_url} 响应异常: HTTP {resp.status_code}"

    try:
        names = [m.get("name", "") for m in (resp.json() or {}).get("models", [])]
    except Exception as exc:
        return False, f"VLM 服务 {base_url} 返回无法解析: {exc}"

    if any(model in n for n in names):
        return True, f"模型 {model} 就绪"
    return False, f"模型 {model} 未安装。可用: {', '.join(names[:5]) or '(无)'}"


def defer_task_for_vlm(db, task) -> None:
    """把任务**原样放回队列**（不消费）。

    只在"还没开始产出业务数据"的阶段调用（预检在建立内容身份之前）：
    此时任务除了状态标记之外没有任何副作用需要回滚。
    """
    task.status = "QUEUED"
    task.error_message = None
    task.completed_at = None
    db.commit()
