"""长耗时 / 重 IO 的接口必须是**同步 `def`**，不能是 `async def`。

为什么（真实事故）：`POST /task/import` 原先写成 `async def`，而它要对上百个文件
逐个算 MD5 并写库（耗时数分钟）。**`async def` 会跑在事件循环里**，于是整个后端
被独占 —— 前端所有轮询超时、显示"后端断开"，界面卡在"导入中"（实测 7 分钟无响应，
采样显示事件循环线程在跑 Python、而 Qt 主线程阻塞在同步 HTTP 上）。

写成同步 `def` 时，FastAPI 会把它**丢到线程池**执行，事件循环保持空闲。

注意：不是"越同步越好"——纯 CPU 极短、且需要 `await`（如调度器控制的接口）
仍应保持 `async def`。这里只钉住**确认过做阻塞 IO 且无 await** 的那批。
"""

import inspect

import pytest

import src.api.main as M
import src.api.routes.actors as A
import src.api.routes.videos as V

BLOCKING_ENDPOINTS = [
    (M, "import_videos"),        # 逐文件 MD5 + 写库，可耗时数分钟（本次事故主角）
    (M, "get_videos"),           # 封面解析读帧图片（判黑/择优）
    (M, "get_actor"),            # 读人脸图
    (M, "import_dictionary"),    # 批量写字典
    (M, "test_connection"),      # 外部 HTTP
    (M, "env_check"),            # 外部 HTTP
    (M, "model_installed"),      # 外部 HTTP
    (V, "get_videos_by_tag"),    # 读封面图
    (A, "get_actor_videos"),     # 读封面图
]


@pytest.mark.parametrize(
    "module, name",
    BLOCKING_ENDPOINTS,
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n in BLOCKING_ENDPOINTS],
)
def test_blocking_endpoint_is_sync_def(module, name):
    fn = getattr(module, name, None)
    assert fn is not None, f"找不到 {name}"
    assert not inspect.iscoroutinefunction(fn), (
        f"{name} 不能是 async def：它做阻塞 IO（算 MD5 / 写库 / 读图 / 外部请求），"
        "写成 async def 会跑在事件循环里把整个后端堵死，"
        "表现为前端轮询超时、界面卡在\"导入中\"。请保持同步 def 交给线程池。"
    )
