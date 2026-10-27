"""任务状态机——**生产管线状态词表的单一事实源**（任务 2.4 并轨）。

历史教训：旧词表（scheduled→scanning→…小写）与生产库实际状态（大写
QUEUED/CHECKING/…）完全脱节，且本模块此前**零消费者**——管线直写
task.status，这里从未被调用。

并轨后约定：
- 本模块即词表唯一权威。管线出于性能与事务边界考虑仍直写 task.status，
  但**新增转移必须先在 VALID_TRANSITIONS 立案**——
  tests/test_state_machine.py 对 api 层做静态扫描，状态字面量超出词表
  即测试失败；
- 新代码写状态请走 update_tasks_status / transition（带校验）；
- 旧小写词表已随零消费者一并移除，勿再引用。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 活动态（可暂停/可失败）
ACTIVE_STATUSES: List[str] = [
    "CHECKING", "SIFTING", "SIFTED", "ANALYZING", "CLUSTERING",
    "WRITING", "PROCESSING",
]

VALID_TRANSITIONS: Dict[str, List[str]] = {
    "PENDING":    ["QUEUED"],
    "QUEUED":     ["CHECKING", "PENDING"],   # 调度器拾起 / 取消排队退回
    "CHECKING":   ["SIFTING", "ERROR", "PAUSED", "QUEUED"],
    "SIFTING":    ["SIFTED", "ANALYZING", "ERROR", "PAUSED", "QUEUED"],
    "SIFTED":     ["ANALYZING", "ERROR", "PAUSED", "QUEUED"],
    "ANALYZING":  ["CLUSTERING", "ERROR", "PAUSED", "QUEUED"],
    "CLUSTERING": ["WRITING", "ERROR", "PAUSED", "QUEUED"],
    "WRITING":    ["COMPLETE", "ERROR", "PAUSED", "QUEUED"],
    "PROCESSING": ["COMPLETE", "ERROR", "PAUSED", "QUEUED"],  # 泛活动态
    "PAUSED":     ["QUEUED"],                # 恢复 = 重新入队
    "ERROR":      ["QUEUED"],                # 重试 = 重新入队
    "FAILED":     ["QUEUED"],
    "COMPLETE":   ["QUEUED"],                # 仅 force 重跑
    "DUPLICATE":  [],                        # 终态
}

TERMINAL_STATUSES: List[str] = ["COMPLETE", "DUPLICATE"]


def all_statuses() -> List[str]:
    """词表全量状态。"""
    return list(VALID_TRANSITIONS.keys())


def can_transition(from_status: str, to_status: str) -> bool:
    """Check whether a transition is valid (unknown source → False)."""
    return to_status in VALID_TRANSITIONS.get(from_status, [])


def validate_transition(from_status: str, to_status: str) -> Optional[str]:
    """Validate a transition and return an error message or None."""
    if from_status not in VALID_TRANSITIONS:
        return f"Invalid status: {from_status}"
    allowed = VALID_TRANSITIONS[from_status]
    if not allowed:
        return f"'{from_status}' is terminal (no transitions allowed)"
    if to_status not in allowed:
        return (f"Cannot transition from '{from_status}' to '{to_status}'. "
                f"Allowed: {', '.join(allowed)}")
    return None


def transition(db, task_id: int, from_status: str, to_status: str) -> Optional[Dict[str, Any]]:
    """Perform a validated state transition. Returns the updated task dict."""
    from src.db.schema import get_connection, init_db
    from src.db import repository as repo

    init_db()
    db = get_connection()

    error = validate_transition(from_status, to_status)
    if error:
        db.close()
        raise ValueError(error)

    # 进入终态 COMPLETE 时回填完成时间
    if to_status == "COMPLETE":
        repo.complete_task(db, task_id)
    else:
        repo.update_task_status(db, task_id, to_status)

    task = repo.get_task(db, task_id)
    db.close()
    return task


def update_tasks_status(db, task_id: int, status: str) -> None:
    """Update a task's status. Validates transition from current status."""
    from src.db import repository as repo

    task = repo.get_task(db, task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    error = validate_transition(task["status"], status)
    if error:
        raise ValueError(error)

    repo.update_task_status(db, task_id, status)


def get_unfinished_tasks(db) -> List[Dict[str, Any]]:
    """Return all tasks NOT in a terminal/writing status."""
    from src.db import repository as repo
    all_tasks = repo.list_tasks(db)
    unfinished = [t for t in all_tasks if t["status"] not in ("COMPLETE", "WRITING")]
    return unfinished


def delete_task(db, task_id: int) -> None:
    """Delete a task and all its associated montage units."""
    from src.db import repository as repo

    task = repo.get_task(db, task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")

    # Delete associated units first (FK will cascade, but be explicit)
    db.execute("DELETE FROM face_embeddings WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_detections WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_dense_frames WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM face_segments WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM montage_units WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM nfo_writes WHERE task_id=?", (task_id,))
    db.execute("DELETE FROM audit_queue WHERE source_task_id=?", (task_id,))
    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    db.commit()
