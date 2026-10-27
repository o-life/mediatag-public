"""状态机并轨测试（任务 2.4）。

核心保证:
1. 词表覆盖 api/main.py 与 scheduler 里**全部大写状态字面量**（静态扫描,
   新增状态字面量而未在 VALID_TRANSITIONS 立案 → 测试失败）;
2. 管线真实转移链全部合法;
3. 非法转移被拒。
"""

import re
from pathlib import Path


from src.core.state_machine import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    all_statuses,
    can_transition,
    validate_transition,
)

ROOT = Path(__file__).resolve().parent.parent

# 管线真实转移链（api/main.py + scheduler 实测流）
HAPPY_PATH = ["PENDING", "QUEUED", "CHECKING", "SIFTING", "ANALYZING",
              "CLUSTERING", "WRITING", "COMPLETE"]


def test_happy_path_chain_legal():
    for a, b in zip(HAPPY_PATH, HAPPY_PATH[1:]):
        assert can_transition(a, b), f"{a} → {b} 应合法"


def test_reentry_paths_legal():
    assert can_transition("ERROR", "QUEUED")       # 重试
    assert can_transition("PAUSED", "QUEUED")      # 恢复
    assert can_transition("COMPLETE", "QUEUED")    # force 重跑
    assert can_transition("QUEUED", "PENDING")     # 取消排队
    assert can_transition("SIFTING", "SIFTED")     # 断点保存


def test_fail_path_from_every_active_state():
    for st in ACTIVE_STATUSES:
        assert can_transition(st, "ERROR"), f"{st} 应可进入 ERROR"
        assert can_transition(st, "PAUSED"), f"{st} 应可暂停"


def test_illegal_transitions_rejected():
    assert not can_transition("PENDING", "ANALYZING")   # 必须经 QUEUED/CHECKING/SIFTING
    assert not can_transition("COMPLETE", "ANALYZING")  # 必须经 QUEUED
    assert not can_transition("DUPLICATE", "QUEUED")    # 终态
    assert validate_transition("DUPLICATE", "QUEUED") is not None
    assert validate_transition("不存在的状态", "QUEUED") is not None


def test_wordlist_covers_canonical_statuses():
    canonical = set(HAPPY_PATH) | {"ERROR", "PAUSED", "FAILED", "DUPLICATE",
                                   "SIFTED", "PROCESSING"}
    missing = canonical - set(VALID_TRANSITIONS)
    assert not missing, f"词表缺状态: {missing}"


def test_static_scan_api_status_literals_within_wordlist():
    """静态扫描:api/main.py 与 scheduler.py 中 task.status = 'X' 的直写
    字面量必须全部在词表内（新增转移先立案后使用）。"""
    pat = re.compile(r"(?:task|t)\.status\s*=\s*'([A-Z_]+)'")
    offenders = {}
    for rel in ("src/api/main.py", "src/core/scheduler.py"):
        f = ROOT / rel
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for status in pat.findall(line):
                if status not in VALID_TRANSITIONS:
                    offenders.setdefault(rel, []).append(f"{i}:{status}")
    assert not offenders, f"状态字面量超出词表: {offenders}"


def test_terminal_and_active_sets_consistent():
    assert set(TERMINAL_STATUSES) <= set(VALID_TRANSITIONS)
    for st in ACTIVE_STATUSES:
        assert st in VALID_TRANSITIONS and VALID_TRANSITIONS[st], f"{st} 无出边"
    assert all_statuses() == list(VALID_TRANSITIONS.keys())
