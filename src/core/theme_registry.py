"""主题注册表：**数据文件之外**的主题清单（解开自举问题）。

为什么必须在库外：`themes` / `theme_prompts` 现在存在数据库里，而
"有哪些主题、各自的库与 config 在哪、当前是哪个" **必须先于"打开某个库"可知**——
否则每个主题一个数据文件之后，无法决定该打开哪个文件。

存放位置：`<base>/themes/registry.json`（`base` 取 `PathManager.base_dir`：
打包态 = `~/Library/Application Support/MediaTag`，开发态 = `<repo>/data`）。

    {
      "version": 1,
      "current": "日常",
      "themes": [
        {"name": "日常", "is_default": true,
         "label_mapping_path": "...", "prompt": "..."}
      ]
    }

写入一律**原子**（先写临时文件再 `os.replace`），避免中断留下半写文件。
本模块只负责这份清单；`themes` / `theme_prompts` 两张表保留不删（迁移导出后不再读写）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_THEME = "日常"
REGISTRY_VERSION = 1

#: 显式覆盖注册表路径（测试用：让注册表落在临时目录，不污染仓库 data/）
REGISTRY_PATH_ENV = "VIDEO_TAG_THEME_REGISTRY_PATH"


def default_registry_path() -> Path:
    override = os.environ.get(REGISTRY_PATH_ENV)
    if override:
        return Path(override)
    try:
        from src.core.path_manager import get_path_manager
    except ImportError:
        from core.path_manager import get_path_manager

    return Path(get_path_manager().base_dir) / "themes" / "registry.json"


def _empty() -> Dict[str, Any]:
    return {"version": REGISTRY_VERSION, "current": "", "themes": []}


class ThemeRegistry:
    """`registry.json` 的读写与查询（无缓存：文件很小，每次读盘最简单）。"""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path is not None else default_registry_path()

    @property
    def path(self) -> Path:
        return self._path

    # ── 读写 ────────────────────────────────────────────────────
    def load(self) -> Dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("registry 根节点不是对象")
        except FileNotFoundError:
            return _empty()
        except Exception as exc:
            logger.warning("读取主题注册表失败（按空处理）%s: %s", self._path, exc)
            return _empty()
        data.setdefault("version", REGISTRY_VERSION)
        data.setdefault("current", "")
        data.setdefault("themes", [])
        if not isinstance(data["themes"], list):
            data["themes"] = []
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self._path)

    # ── 查询 ────────────────────────────────────────────────────
    def list_themes(self) -> List[Dict[str, Any]]:
        return list(self.load()["themes"])

    def names(self) -> List[str]:
        return [t.get("name") for t in self.list_themes() if t.get("name")]

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        for t in self.list_themes():
            if t.get("name") == name:
                return t
        return None

    def current(self) -> str:
        return self.load().get("current") or ""

    def default_theme(self) -> str:
        for t in self.list_themes():
            if t.get("is_default"):
                return t.get("name") or ""
        return ""

    def effective_current(self) -> str:
        """当前主题；未指定时回退默认主题，再回退第一个。"""
        cur = self.current()
        if cur and self.get(cur):
            return cur
        return self.default_theme() or (self.names()[0] if self.names() else "")

    # ── 变更 ────────────────────────────────────────────────────
    def set_current(self, name: str) -> None:
        data = self.load()
        if not any(t.get("name") == name for t in data["themes"]):
            raise ValueError(f"主题不存在: {name}")
        data["current"] = name
        self.save(data)

    def upsert(self, name: str, **fields: Any) -> Dict[str, Any]:
        """新增或更新一个主题（按 name 匹配）。"""
        name = (name or "").strip()
        if not name:
            raise ValueError("主题名不能为空")
        data = self.load()
        existing = next((t for t in data["themes"] if t.get("name") == name), None)
        if existing is None:
            existing = {"name": name}
            data["themes"].append(existing)
        existing.update({k: v for k, v in fields.items() if v is not None})
        if fields.get("is_default"):
            for t in data["themes"]:
                if t is not existing:
                    t["is_default"] = False
            existing["is_default"] = True
        if not data.get("current"):
            data["current"] = name
        self.save(data)
        return existing

    def delete(self, name: str) -> None:
        data = self.load()
        target = next((t for t in data["themes"] if t.get("name") == name), None)
        if target is None:
            raise ValueError(f"主题不存在: {name}")
        if target.get("is_default"):
            raise ValueError("默认主题不可删除")
        data["themes"] = [t for t in data["themes"] if t.get("name") != name]
        if data.get("current") == name:
            data["current"] = self.default_theme() or (data["themes"][0]["name"] if data["themes"] else "")
        self.save(data)

    # ── 播种与迁移 ──────────────────────────────────────────────
    def seed(self, presets: List[Dict[str, Any]]) -> int:
        """按名称幂等播种预置主题；返回新建数量。已存在的**不覆盖**用户改动。"""
        data = self.load()
        existing = {t.get("name") for t in data["themes"] if t.get("name")}
        has_default = any(t.get("is_default") for t in data["themes"])
        created = 0
        for preset in presets:
            name = preset.get("name")
            if not name or name in existing:
                continue
            rec: Dict[str, Any] = {
                "name": name,
                "is_default": bool(preset.get("default")) and not has_default,
            }
            for key in ("description", "label_mapping_path", "prompt"):
                if preset.get(key):
                    rec[key] = preset[key]
            data["themes"].append(rec)
            if rec["is_default"]:
                has_default = True
            created += 1
        if not data.get("current") and data["themes"]:
            data["current"] = self.default_theme() or data["themes"][0]["name"]
        if created or not self._path.exists():
            self.save(data)
        return created

    def import_from_db(self, session) -> int:
        """一次性迁移：把 `themes` / `theme_prompts` 两表导出到注册表。

        已有同名主题则**不覆盖**（以注册表为准）。返回导入数量。
        """
        from src.db.models import Theme, ThemePrompt  # 延迟导入，避免循环依赖

        records = []
        for theme in session.query(Theme).order_by(Theme.id).all():
            prompt_row = (
                session.query(ThemePrompt).filter(ThemePrompt.theme_id == theme.id).first()
            )
            rec: Dict[str, Any] = {
                "name": theme.name,
                "is_default": bool(theme.is_default),
                # 迁移而来：在 Phase 5 拆分之前暂时沿用既有的单一数据库
                # （否则用户数据会立刻"消失"）。Phase 5 拆分后清除此标记。
                "legacy_db": True,
            }
            if theme.description:
                rec["description"] = theme.description
            if theme.label_mapping_path:
                rec["label_mapping_path"] = theme.label_mapping_path
            if prompt_row and prompt_row.prompt:
                rec["prompt"] = prompt_row.prompt
            records.append(rec)

        data = self.load()
        existing = {t.get("name") for t in data["themes"] if t.get("name")}
        imported = 0
        for rec in records:
            if rec["name"] in existing:
                continue
            data["themes"].append(rec)
            imported += 1
        if not data.get("current"):
            # 优先沿用**旧的运行配置**里的当前主题（`analysis.active_theme`），
            # 否则"用户正在用「男男」"会在迁移瞬间被悄悄改成默认主题（行为变更）。
            preferred = ""
            try:
                from src.core.theme_workspace import ThemeWorkspace

                legacy = ThemeWorkspace().legacy_config()
                preferred = ((legacy.get("analysis") or {}).get("active_theme") or "").strip()
            except Exception:
                preferred = ""
            names = {r["name"] for r in data["themes"]}
            if preferred and preferred in names:
                data["current"] = preferred
            else:
                cur = next((r["name"] for r in records if r.get("is_default")), "")
                data["current"] = cur or (records[0]["name"] if records else "")
        if imported or not self._path.exists():
            self.save(data)
        return imported


_registry: Optional[ThemeRegistry] = None


def get_theme_registry() -> ThemeRegistry:
    """进程内共享的注册表实例（路径固定，无需每次 new）。"""
    global _registry
    if _registry is None:
        _registry = ThemeRegistry()
    return _registry
