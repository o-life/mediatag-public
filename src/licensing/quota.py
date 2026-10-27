"""免费版额度：全局累计处理文件数计数器。

业务规则（docs/界面设计/07-授权管理.md §1.3）：
- 上限 100，全局累计（跨 vault），卡在导入入口；
- 完整版期间不计数；
- 取消激活那一刻做库存快照：counter = max(counter, 全库任务总数)——
  存量 ≤100 可继续导入到 100；存量 >100 则锁定新增导入；
- 计数器存应用级全局文件（不在任何 vault 数据库内），只增不减：
  删文件/删任务/删 vault/重建 vault 都不能返还或清零。
"""

import glob
import json
import logging
import sqlite3
from pathlib import Path

LIMIT = 100

_log = logging.getLogger("licensing.quota")

# 测试可覆盖；None = 使用应用数据目录
_DIR_OVERRIDE: Path | None = None


def _app_base() -> Path:
    if _DIR_OVERRIDE is not None:
        return Path(_DIR_OVERRIDE)
    try:
        from ..core.path_manager import app_support_base
    except ImportError:
        from core.path_manager import app_support_base
    return app_support_base()


def _file() -> Path:
    # 命名不直白（防篡改不做强要求，仅提高随意改动的门槛——规范 §9.6）
    return _app_base() / "usage_state.json"


def _load() -> int:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
        value = int(data.get("processed", 0))
        return max(0, value)
    except (OSError, ValueError, AttributeError):
        return 0


def _save(value: int) -> None:
    path = _file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"processed": int(value)}), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except OSError as exc:
        _log.error("额度计数器写入失败: %s", exc)


def processed() -> int:
    """当前累计已处理文件数。"""
    return _load()


def remaining() -> int:
    """免费版还可导入的数量（完整版不调用）。"""
    return max(0, LIMIT - _load())


def exceeded() -> bool:
    return _load() >= LIMIT


def register_imported(count: int) -> None:
    """导入成功后累加（免费版专用；完整版不调用也不影响正确性）。"""
    if count <= 0:
        return
    _save(_load() + int(count))


def _stock_total() -> int:
    """统计全部主题工作区（data/themes/*/data.db）的 tasks 总数。"""
    total = 0
    pattern = str(_app_base() / "data" / "themes" / "*" / "data.db")
    for db_path in glob.glob(pattern):
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
                total += int(row[0]) if row else 0
        except (sqlite3.Error, OSError) as exc:
            _log.warning("统计库存跳过 %s: %s", db_path, exc)
    return total


def sync_from_stock() -> int:
    """取消激活时的库存快照：counter = max(counter, 全库任务总数)。

    在每一条「清除本地激活信息」的路径上调用（主动解绑/被动吊销/强制解绑/换机）。
    """
    stock = _stock_total()
    new_value = max(_load(), stock)
    _save(new_value)
    _log.info("额度库存快照: stock=%s → counter=%s", stock, new_value)
    return new_value
