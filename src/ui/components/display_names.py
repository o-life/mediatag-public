"""展示层术语转换：用户可见文案不出现 actor 术语（底层要求）。

术语决议（2026-09-14）：用户可见文案「演员」→「人物」，但**代码标识符与
DB 表名/值不动**（`actors` / `actor_faces` / `actor.code = ACTOR_{id:03d}`）。
因此界面上直接展示 code 时仍会露出 `ACTOR_122` 这类字样——本模块只在
**展示层**做转换（`ACTOR_122` → 「人物 122」），存储与接口返回一律不变。
"""

import re

_ACTOR_CODE_RE = re.compile(r"^ACTOR_(\d+)$")


def actor_code_display(code) -> str:
    """把人物编号转成用户可见文案：``ACTOR_122`` → ``人物 122``。

    - 数字部分原样保留（含前导零），保证与库里的 code 一一对应、可回溯日志；
    - 非 ``ACTOR_`` 前缀的编号（测试/历史数据如 ``F1``）原样返回；
    - ``None`` / 空串返回空串。
    """
    if not code:
        return ""
    text = str(code).strip()
    m = _ACTOR_CODE_RE.match(text)
    if m:
        return f"人物 {m.group(1)}"
    return text
