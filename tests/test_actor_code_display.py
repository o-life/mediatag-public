"""人物编号展示层术语转换：ACTOR_122 → 人物 122（界面不出 actor 术语）。

术语决议（2026-09-14）：用户可见文案「演员」→「人物」，但 DB 值不动——
`actor.code` 仍是 `ACTOR_{id:03d}`。因此卡片/详情/合并弹窗在展示 code 时
必须经 `actor_code_display()` 转换；本测试锁定该纯函数的行为。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ui.components.display_names import actor_code_display


class TestActorCodeDisplay:
    def test_actor_prefix_converted(self):
        assert actor_code_display("ACTOR_122") == "人物 122"

    def test_leading_zeros_kept(self):
        # 数字部分原样保留，保证与库里的 code 一一对应
        assert actor_code_display("ACTOR_001") == "人物 001"

    def test_non_actor_code_unchanged(self):
        assert actor_code_display("F1") == "F1"
        assert actor_code_display("P-33") == "P-33"

    def test_empty_and_none(self):
        assert actor_code_display("") == ""
        assert actor_code_display(None) == ""

    def test_whitespace_stripped(self):
        assert actor_code_display("  ACTOR_9  ") == "人物 9"
