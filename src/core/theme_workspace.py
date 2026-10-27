"""主题工作区：每个分析主题一个自包含目录。

```
<base>/themes/
├── registry.json          # 主题清单 + 当前主题（见 theme_registry）
└── <主题名>/
    ├── config.yaml        # 该主题的业务配置（数据库 / 人脸目录 / 当前主题）
    ├── data.db            # 该主题的数据文件
    └── faces/             # 该主题的人脸资源库
```

**当前主题由注册表决定**（`theme_registry.current`）——本模块不再维护第二份指针，
避免"两处真相"。

Phase 1 的定位是**布局与解析**，并且要求**行为等价**：
既有主题在这阶段生成的 `config.yaml` 仍指向**现有的单一数据库与人脸根目录**
（否则用户的数据会立刻"消失"）。真正把每个主题切到自己的
`<主题>/data.db` 与 `<主题>/faces/`，由 Phase 5 的拆分迁移统一改写。

因此本模块负责：
- 目录布局与路径解析（主题名 → 目录/配置/库/人脸目录）
- 懒创建：首次进入某主题时建目录、生成 config、建空库、建人脸目录
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

DB_FILENAME = "data.db"
CONFIG_FILENAME = "config.yaml"
FACES_DIRNAME = "faces"

#: 关闭"自动建档"的环境变量（测试用：置 0 时不建任何主题目录）
AUTOCREATE_ENV = "VIDEO_TAG_THEME_AUTOCREATE"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class ThemeWorkspace:
    """主题目录布局与懒创建。base_dir 可显式传入（测试用）。"""

    def __init__(self, base_dir: Optional[Path] = None):
        self._explicit_base = Path(base_dir) if base_dir is not None else None
        self._cached_base: Optional[Path] = None

    # ── 基础路径 ────────────────────────────────────────────────
    @property
    def base_dir(self) -> Path:
        if self._explicit_base is not None:
            return self._explicit_base
        if self._cached_base is None:
            try:
                from src.core.path_manager import get_path_manager
            except ImportError:
                from core.path_manager import get_path_manager

            self._cached_base = Path(get_path_manager().base_dir)
        return self._cached_base

    @property
    def themes_root(self) -> Path:
        return self.base_dir / "themes"

    def theme_dir(self, name: str) -> Path:
        return self.themes_root / name

    def config_path(self, name: str) -> Path:
        return self.theme_dir(name) / CONFIG_FILENAME

    def db_path(self, name: str) -> Path:
        return self.theme_dir(name) / DB_FILENAME

    def faces_dir(self, name: str) -> Path:
        return self.theme_dir(name) / FACES_DIRNAME

    def exists(self, name: str) -> bool:
        return bool(name) and self.config_path(name).exists()

    # ── 当前主题（真相在注册表） ────────────────────────────────
    def current_name(self) -> str:
        try:
            from src.core.theme_registry import get_theme_registry

            return get_theme_registry().effective_current()
        except Exception:
            return ""

    def current_config_path(self) -> Optional[Path]:
        name = self.current_name()
        if name and self.config_path(name).exists():
            return self.config_path(name)
        return None

    # ── 既有（单库）配置：Phase 1 让各主题先指向它，保证行为等价 ──
    def _legacy_config_candidates(self) -> list:
        import sys

        cands = []
        frozen = bool(getattr(sys, "frozen", False))
        if not frozen:
            cands.append(_project_root() / "config.local.yaml")
            cands.append(_project_root() / "config.yaml")
        cands.append(self.base_dir / "config.yaml")
        return cands

    def legacy_config(self) -> Dict[str, Any]:
        """既有的单一运行配置（用于生成主题 config 的基底）。"""
        for path in self._legacy_config_candidates():
            try:
                if path.is_file():
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    if isinstance(data, dict) and data:
                        return data
            except Exception as exc:  # pragma: no cover
                logger.warning("读取既有配置失败 %s: %s", path, exc)
        return {}

    def _legacy_database_path(self, legacy: Dict[str, Any]) -> str:
        raw = ((legacy.get("database") or {}).get("path")) or str(self.db_path(""))
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = _project_root() / p
        return str(p)

    def _legacy_faces_dir(self, legacy: Dict[str, Any]) -> str:
        raw = ((legacy.get("face_recognition") or {}).get("faces_dir")) or "data/actor_faces"
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = _project_root() / p
        return str(p) + ("/" if str(raw).endswith("/") else "")

    # ── 懒创建 ──────────────────────────────────────────────────
    def _is_legacy_theme(self, name: str) -> bool:
        """是否为"迁移而来、暂时沿用老库"的主题（见 registry.import_from_db）。"""
        try:
            from src.core.theme_registry import get_theme_registry

            rec = get_theme_registry().get(name) or {}
            return bool(rec.get("legacy_db"))
        except Exception:
            return False

    def build_config(self, name: str, legacy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成某主题的**覆盖配置**——只写与全局不同的键。

        **配置分层**（见 `ConfigManager.OVERLAY_EXACT/OVERLAY_PREFIXES`）::

            有效配置 = 全局基底(config.local.yaml) ← 主题覆盖(<主题>/config.yaml)

        为什么不再整份拷贝：主题 config 若冻结一份完整快照，改全局设置
        （VLM 供应商、抽帧数、NFO 输出）对已存在的主题**不生效**——这正是分层要修的。

        - **既有主题**（`legacy_db` 未清除）：路径仍指向既有单库/人脸根目录（行为等价）；
          拆分迁移后应被改写到档内 `data.db` / `faces/`。
        - **新建主题**：档内 `<主题>/data.db` 与 `<主题>/faces/`（天然隔离）。
        - 片源策略（`analysis.mapping_*`）**不写死**——缺省即继承全局；
          只有用户显式按主题改了才落进覆盖文件。
        """
        legacy = legacy if legacy is not None else self.legacy_config()
        inherits = self._is_legacy_theme(name)
        if inherits:
            db_path = self._legacy_database_path(legacy)
            faces_dir = self._legacy_faces_dir(legacy)
        else:
            db_path = str(self.db_path(name))
            faces_dir = str(self.faces_dir(name)) + "/"
        return {
            "theme": {
                "name": name,
                "managed_by": "theme_workspace",
                "inherits_legacy_content": inherits,
            },
            "database": {"path": db_path},
            "face_recognition": {"faces_dir": faces_dir},
            "analysis": {"active_theme": name},
        }

    def ensure_theme(self, name: str, legacy: Optional[Dict[str, Any]] = None) -> Path:
        """确保主题目录/配置/人脸目录存在；返回主题目录。

        **不建库**——建库（`init_db`）由启动链路显式做，避免在只读场景误建文件。
        """
        if not name:
            raise ValueError("主题名不能为空")
        tdir = self.theme_dir(name)
        (tdir).mkdir(parents=True, exist_ok=True)
        self.faces_dir(name).mkdir(parents=True, exist_ok=True)
        cfg_path = self.config_path(name)
        if not cfg_path.exists():
            cfg_path.write_text(
                yaml.dump(self.build_config(name, legacy), allow_unicode=True,
                          default_flow_style=False, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("已生成主题配置: %s", cfg_path)
        return tdir

    def rename_theme(self, old_name: str, new_name: str) -> Path:
        """重命名主题：搬移目录（同卷 rename 保留 inode，运行中的库句柄
        不受影响）并同步 config 内的主题名；新连接按新路径解析（建议重启）。"""
        if not new_name:
            raise ValueError("主题名不能为空")
        old_dir = self.theme_dir(old_name)
        if not old_dir.exists():
            raise FileNotFoundError(f"主题目录不存在: {old_dir}")
        new_dir = self.theme_dir(new_name)
        if new_dir.exists():
            raise FileExistsError(f"目标主题目录已存在: {new_dir}")
        old_dir.rename(new_dir)
        cfg_path = self.config_path(new_name)
        cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                logger.warning("重命名后读取主题配置失败: %s", exc)
        theme_sec = cfg.get("theme") or {}
        if theme_sec.get("name") == old_name:
            theme_sec["name"] = new_name
        # 非继承主题：库/人脸路径随目录改写（继承主题路径指向外部既有位置，保持不动）
        if not theme_sec.get("inherits_legacy_content"):
            cfg.setdefault("database", {})["path"] = str(self.db_path(new_name))
            cfg.setdefault("face_recognition", {})["faces_dir"] = str(self.faces_dir(new_name)) + "/"
        if theme_sec:
            cfg["theme"] = theme_sec
        cfg_path.write_text(
            yaml.dump(cfg, allow_unicode=True,
                      default_flow_style=False, sort_keys=True),
            encoding="utf-8",
        )
        return new_dir

    def effective_db_path(self, name: str) -> Path:
        """该主题**实际生效**的数据库路径（读它的 config，缺省回退档内 data.db）。"""
        try:
            cfg = yaml.safe_load(self.config_path(name).read_text(encoding="utf-8")) or {}
            raw = (cfg.get("database") or {}).get("path")
            if raw:
                p = Path(str(raw)).expanduser()
                if not p.is_absolute():
                    p = _project_root() / p
                return p
        except Exception:
            pass
        return self.db_path(name)

    def effective_faces_dir(self, name: str) -> Path:
        """该主题**实际生效**的人脸目录（读它的 config）。"""
        try:
            cfg = yaml.safe_load(self.config_path(name).read_text(encoding="utf-8")) or {}
            raw = (cfg.get("face_recognition") or {}).get("faces_dir")
            if raw:
                p = Path(str(raw)).expanduser()
                if not p.is_absolute():
                    p = self.theme_dir(name) / p
                return p
        except Exception:
            pass
        return self.faces_dir(name)


_workspace: Optional[ThemeWorkspace] = None


def get_theme_workspace() -> ThemeWorkspace:
    global _workspace
    if _workspace is None:
        _workspace = ThemeWorkspace()
    return _workspace


def bootstrap_current_theme() -> Optional[str]:
    """启动链路调用：为**当前主题**建档（主题目录 + config.yaml + faces 目录）。

    注册表为空时**什么都不做**——此时回退旧布局，启动链路稍后会把库内主题导出到
    注册表，下一次启动即可用上主题目录（无鸡生蛋问题）。
    测试可用 `VIDEO_TAG_THEME_AUTOCREATE=0` 关闭。
    """
    if os.environ.get(AUTOCREATE_ENV, "1") != "1":
        return None
    try:
        ws = get_theme_workspace()
        name = ws.current_name()
        if not name:
            return None
        ws.ensure_theme(name)
        return name
    except Exception as exc:
        logger.warning("主题工作区初始化失败（回退旧布局）: %s", exc)
        return None
