"""SQLite 库文件的复制与只读打开。

为什么单独收口：跨库复制在本仓库有两处使用者（`scripts/split_workspaces_by_theme.py`
的按主题拆分、`scripts/consolidate_workspaces_into_theme.py` 的并库迁移），
而这里有一个**必须记住的坑**：

    只删目标主文件会留下指向**旧库**的 `-wal`/`-shm`，
    之后 SQLite 打开目标库时按残留 WAL 恢复 → `disk I/O error`。
    （拆分迁移时踩过一次，见 `docs/架构/主题工作区-实现与运维说明.md` §6）

复制用 SQLite 的 **backup API** 而不是 `cp`：它能把尚未 checkpoint 进主文件的
WAL 内容一并写入目标库，且是崩溃一致的快照。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

#: 与主库文件同名的伴随文件（WAL 模式）
SIDECAR_SUFFIXES = ("-wal", "-shm")


def remove_sidecars(db_path: PathLike) -> None:
    """删除 `-wal` / `-shm`（不删主文件）。"""
    for suffix in SIDECAR_SUFFIXES:
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()


def copy_db(src: PathLike, dst: PathLike) -> None:
    """把 `src` 完整复制为 `dst`（含 WAL 内容，覆盖已存在的 `dst`）。

    `src` 以**只读**打开，自身不会被改动。
    """
    src_path, dst_path = Path(src), Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    for suffix in ("", *SIDECAR_SUFFIXES):
        p = Path(str(dst_path) + suffix)
        if p.exists():
            p.unlink()

    src_con = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst_con = sqlite3.connect(str(dst_path))
    try:
        src_con.backup(dst_con)
    finally:
        dst_con.close()
        src_con.close()


def open_ro(db_path: PathLike) -> sqlite3.Connection:
    """只读打开（核对/审计用）：先试 `mode=ro`，失败再退化为 `query_only`。

    ⚠️ 退化分支**不是逻辑写操作**（`query_only=1` 拒绝一切写语句），但它会以读写方式
    打开库：SQLite 因此在目录里创建/删除 `-wal`/`-shm`（关库时还会 checkpoint）。
    实测踩到过：只读核对把 5 个主题库的目录 mtime 改了。数据没有被改动，但
    "绝不触碰 WAL" 的说法要打个折——真正的数据保证是 `query_only`，不是文件系统。
    """
    p = Path(db_path)
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        con.execute("PRAGMA query_only=1")
        return con
    except sqlite3.OperationalError:
        con = sqlite3.connect(str(p))
        con.execute("PRAGMA query_only=1")
        return con


def table_names(con: sqlite3.Connection) -> list:
    """库内所有用户表名（不含 `sqlite_` 内部表），按名排序。"""
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def columns(con: sqlite3.Connection, table: str) -> list:
    """表的列名列表（表不存在则返回空列表）。"""
    try:
        return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    except sqlite3.Error:
        return []
