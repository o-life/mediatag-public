"""VLM 就绪性闸门：模型没准备好 ⇒ **暂停派发**，而不是逐条任务报错。

背景（2026-09-20 实测）：配置的 `vlm.model` 被从 Ollama 删掉后，预检走
`_mark_task_error()` 把任务直接判死，而调度器会在几秒内把队列里每一条都派发一遍
→ **2196 条排队任务全变 ERROR**。

本文件守住四条规则：
1. `is_vlm_ready` 的判定口径（含"连不上也算未就绪"）
2. 未就绪时**不消费队列**：任务留在队列、调度器置 PAUSED、不执行 worker
3. 就绪后**自动恢复**并派发（不用人工干预）
4. 任务被"暂缓"时状态回到 QUEUED（不留下 ERROR 标记）
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.scheduler import SchedulerStatus, TaskPriority, TaskScheduler  # noqa: E402
from src.core.vlm_readiness import defer_task_for_vlm, is_vlm_ready  # noqa: E402


# ── is_vlm_ready ────────────────────────────────────────────────────
class _Resp:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {}
        self.status_code = status_code

    def json(self):
        return self._payload


def _cfg(**over):
    base = {"provider": "url", "base_url": "http://localhost:11434", "model": "minicpm-v4.5:8b"}
    base.update(over)
    return base


def test_ready_when_model_listed(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(
        {"models": [{"name": "minicpm-v4.5:8b"}, {"name": "other"}]}))
    ok, reason = is_vlm_ready(lambda: _cfg())
    assert ok and "就绪" in reason


def test_not_ready_when_model_missing(monkeypatch):
    """事故现场：模型列表里没有配置的那个 → 未就绪（而不是"跳过打标"）。"""
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp(
        {"models": [{"name": "minicpm-v4.6:latest"}]}))
    ok, reason = is_vlm_ready(lambda: _cfg())
    assert not ok
    assert "未安装" in reason and "minicpm-v4.5:8b" in reason


def test_not_ready_when_ollama_down(monkeypatch):
    def _boom(*a, **k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("requests.get", _boom)
    ok, reason = is_vlm_ready(lambda: _cfg())
    assert not ok and "无法连接" in reason


def test_not_ready_on_bad_http_status(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp({}, status_code=500))
    ok, _ = is_vlm_ready(lambda: _cfg())
    assert not ok


def test_local_folder_provider_checked_by_path(tmp_path):
    model_dir = tmp_path / "qwen-vl"
    model_dir.mkdir()
    ok, reason = is_vlm_ready(lambda: _cfg(provider="folder", model=str(model_dir)))
    assert ok and "本地模型目录" in reason

    ok2, reason2 = is_vlm_ready(lambda: _cfg(provider="folder", model=str(tmp_path / "nope")))
    assert not ok2 and "不存在" in reason2


def test_config_read_failure_is_not_ready():
    def _boom():
        raise RuntimeError("no config")

    ok, reason = is_vlm_ready(_boom)
    assert not ok and "读取 VLM 配置失败" in reason


# ── 任务"暂缓"语义 ──────────────────────────────────────────────────
class _Task:
    def __init__(self):
        self.status = "ERROR"
        self.error_message = "VLM 模型 minicpm-v4.5:8b 未在 Ollama 中安装"
        self.completed_at = "2026-09-20"


class _DB:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def test_defer_task_returns_it_to_queue():
    """被"暂缓"的任务必须回到 QUEUED 且不留错误标记（对比：旧行为是 ERROR）。"""
    task, db = _Task(), _DB()
    defer_task_for_vlm(db, task)
    assert task.status == "QUEUED"
    assert task.error_message is None
    assert task.completed_at is None
    assert db.commits == 1


# ── 调度器闸门 ──────────────────────────────────────────────────────
def test_gate_pauses_without_consuming_queue_and_resumes():
    async def scenario():
        sched = TaskScheduler(max_concurrent=1)
        executed = []
        sched._run_task_worker = lambda tid: executed.append(tid)
        sched._loop = asyncio.get_running_loop()

        state = {"ready": False}
        sched.set_readiness_check(
            lambda: (state["ready"], "模型 minicpm-v4.5:8b 未安装"), interval_seconds=0.05)
        sched.status = SchedulerStatus.RUNNING

        await sched.submit_task(7, TaskPriority.NORMAL)
        await asyncio.sleep(0.15)

        # 未就绪：暂停、任务仍在队列、worker 一次都没跑
        assert sched.status == SchedulerStatus.PAUSED
        assert executed == []
        assert sched.task_queue.qsize() == 1
        assert "未安装" in sched.get_status()["paused_reason"]

        # 就绪后：哨兵自动恢复并派发（无需人工 resume）
        state["ready"] = True
        for _ in range(80):
            await asyncio.sleep(0.02)
            if executed:
                break

        assert executed == [7], "就绪后应自动恢复并执行"
        assert sched.status == SchedulerStatus.RUNNING
        assert sched.get_status()["paused_reason"] == ""
        await sched.stop()

    asyncio.run(scenario())


def test_manual_pause_wins_over_readiness_watch():
    """用户手动 pause 后，哨兵不得把它自动恢复（用户意图优先）。"""
    async def scenario():
        sched = TaskScheduler(max_concurrent=1)
        sched._loop = asyncio.get_running_loop()
        sched.set_readiness_check(lambda: (False, "未就绪"), interval_seconds=0.05)
        sched.status = SchedulerStatus.RUNNING
        await sched.submit_task(1)
        await asyncio.sleep(0.1)
        assert sched.status == SchedulerStatus.PAUSED

        await sched.pause()          # 用户接管
        assert sched._readiness_paused is False
        await asyncio.sleep(0.15)    # 哨兵若还活着会在此恢复 RUNNING
        assert sched.status == SchedulerStatus.PAUSED
        await sched.stop()

    asyncio.run(scenario())


def test_request_readiness_pause_is_thread_safe_without_loop():
    """工作线程里直接请求暂停（无事件循环）也必须立即生效。"""
    sched = TaskScheduler(max_concurrent=1)
    sched.status = SchedulerStatus.RUNNING
    sched.request_readiness_pause("VLM 未就绪: 模型未安装")
    assert sched.status == SchedulerStatus.PAUSED
    assert "VLM 未就绪" in sched.get_status()["paused_reason"]


def test_idle_scheduler_is_not_paused_by_gate():
    """队列为空时不该因为模型缺失就把调度器标成"暂停"。"""
    async def scenario():
        sched = TaskScheduler(max_concurrent=1)
        sched._loop = asyncio.get_running_loop()
        sched.set_readiness_check(lambda: (False, "未安装"), interval_seconds=0.05)
        sched.status = SchedulerStatus.RUNNING
        await sched._schedule_next()
        assert sched.status == SchedulerStatus.RUNNING
        await sched.stop()

    asyncio.run(scenario())


def test_no_gate_installed_keeps_old_behaviour():
    """没装闸门时行为不变（不影响既有部署/测试）。"""
    async def scenario():
        sched = TaskScheduler(max_concurrent=1)
        executed = []
        sched._run_task_worker = lambda tid: executed.append(tid)
        sched._loop = asyncio.get_running_loop()
        sched.status = SchedulerStatus.RUNNING
        await sched.submit_task(3)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if executed:
                break
        assert executed == [3]
        await sched.stop()

    asyncio.run(scenario())
