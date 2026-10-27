"""Configuration management for AI Video Factory."""

import copy
import logging
import yaml
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """深合并：`overlay` 覆盖 `base`。

    只对"两边都是 dict"递归；其余（含 None、list、标量）一律以 overlay 为准——
    `None` 是有效值（如 `mapping_nsfw_filter: None` 表示"按主题自动"），不能被当成缺省。
    例外：覆盖层对**整块 dict** 写 None 视为"未覆盖"（主题文件曾误写
    `video: null` 炸掉基底全部视频配置，任务 2.3）——块级 None 不落地。
    """
    out = dict(base)
    for k, v in (overlay or {}).items():
        if v is None and isinstance(out.get(k), dict):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigManager:
    """Manages application configuration from YAML file."""

    # Default VLM prompt for adult content classification (男男/Gay template)
    DEFAULT_VLM_PROMPT = (
        "你是一个成人内容分类专家。请分析这些视频帧，输出与性活动相关的标签。\n\n"
        "【优先使用的标准标签】\n\n"
        "场景：卧室, 客厅, 浴室, 酒店, 办公室, 更衣室, 健身房, 按摩店, 公园, 街道, 车内, 野外\n\n"
        "体位/动作：口交, 深喉, 69, 后入, 传教士式, 骑乘, 双龙, 侧卧, 站立抱起, 抚摸, 自慰, 互撸, 颜射, 内射\n\n"
        "人数：单人, 双人, 三人, 多人\n\n"
        "年龄/性向：青年, 少年感, 熟男, 中年, Gay, 双性恋\n\n"
        "身材：纤细, 肌肉, 微胖, 熊系\n\n"
        "颜值：精致, 清纯, 刚毅, 粗犷\n\n"
        "视角：器官特写, 第一视角\n\n"
        "肤色：白皙, 古铜, 黝黑\n\n"
        "服装：西装, 制服, 全裸, 半裸\n\n"
        "【允许扩展的标签类型】\n"
        "如果视频中有标准标签未覆盖的有意义内容，可以输出以下类型的标签：\n"
        "- 内容类型：有情节, 无情节, 剧情向, 纯动作, 前戏为主, 正片为主\n"
        "- 具体性活动：如舔弄、亲吻、拥抱等\n"
        "- 特殊场景：如阳台、楼梯、厨房等\n"
        "- 道具/物品：如玩具、手铐、丝袜等\n"
        "- 身体部位：如胸部、臀部、腿部等\n"
        "- 其他有意义的分类标签\n\n"
        "【严格禁止输出的内容】\n"
        "- 通用描述词：明显、直接、处于、聚焦、所有、可以、能够、正在\n"
        "- 抽象概念：场景、画面、内容、情况、状态、问题、结果\n"
        "- 连接词和副词：和、与、或、很、非常、也、都\n"
        "- 无意义的动词：看到、出现、展示、包含、呈现\n\n"
        "【输出规则】\n"
        "1. 优先使用标准标签，但允许输出其他有意义的标签\n"
        "2. 每个标签2-4个汉字\n"
        "3. 用逗号分隔，不要输出其他任何文字\n\n"
        "【输出示例】\n"
        "双人,后入,卧室,肌肉,白皙,青年,亲吻,丝袜\n\n"
        "现在请分析这些帧："
    )

    DEFAULT_CONFIG = {
        "vlm": {
            "provider": "url",
            "base_url": "http://localhost:11434",
            "model": "minicpm-v4.5",
            "timeout": 60
        },
        "video": {
            "frame_count": 15,
            "frame_format": "jpg",
            "output_dir": "./data/frames"
        },
        "analysis": {
            "merge_vlm_passes": True,  # 合并"主分析+逐帧映射"为每场景一次调用; False 回退旧的两次遍历
            "mapping_nsfw_filter": None,  # None → 按主题自动(成人内容主题启用); False 强制关闭; True 强制开启
            "mapping_frames_per_unit": None,  # None → 沿用设置页「抽帧数」(video.frames_per_scene); 数字则为其上限
            "mapping_nsfw_min_frames": 1,  # 场景内 NSFW 帧数低于此值 → 回退抽样交 VLM 判定(不丢弃场景)
            # ── 标签质量门槛(2026-09-30 批3,依据 docs/数据质量/审计-2026-09-30) ──
            "strict_vocabulary": True,  # 闸2:非词典标签不入 tag_frame_mappings
            "frame_quality_filter": True,  # 闸3:黑帧/糊帧/废帧不送 VLM 不入库
            "frame_quality_black_mean": 16,  # 闸3:灰度均值(0-255)低于此判黑帧
            "frame_quality_min_sharpness": 12,  # 闸3:640px 归一后 Laplacian 方差低于此判糊帧
            "min_tag_scene_support": 2,  # 闸4:标签须出现的最少场景/单元数(受比例上限,见下)
            "tag_scene_support_ratio": 0.25,  # 闸4:还须 ≥ 场景数×此比例;实际取两者较小值
            "max_tags_per_file": 30,  # 闸4:每文件标签数上限,超出按场景支持数截断
        },
        "nfo": {
            "output_dir": "./output",
            "format": "xml"
        }
    }

    #: **写路由**：这些键由程序写入时落进本主题覆盖文件，其余键落全局。
    #:
    #: 原则（2026-09-19 定）：
    #: - **只有内容域与主题身份随主题** —— 该主题的数据文件、人脸目录、自描述、身份。
    #: - **分析行为类一律全局**：`analysis.merge_vlm_passes`、`analysis.mapping_*`
    #:   （NSFW 筛选/每场景帧数/回退阈值）、`video.*`、`vlm.*`、`nfo.*`。
    #:   用户意见：它们"影响分析行为，但更像**处理策略**而非**内容**"；
    #:   若将来确有按主题差异再下放（见下"读取"说明）。
    #:
    #: **读取**与"写路由"无关：合并读对**任意键**生效——主题 config 里出现什么键，
    #: 该键就按主题覆盖。所以"将来下放"只需把键写进主题 config（或加入本清单），
    #: 不需要改读取路径。
    OVERLAY_EXACT = (
        "database.path",                 # 该主题的数据文件
        "face_recognition.faces_dir",    # 该主题的人脸目录
        "analysis.active_theme",         # 该主题即当前主题（自描述）
    )
    OVERLAY_PREFIXES = (
        "theme.",                        # 主题身份：name / label_mapping_path / inherits_legacy_content
    )

    def _is_overlay_key(self, key: str) -> bool:
        return key in self.OVERLAY_EXACT or any(
            key.startswith(p) for p in self.OVERLAY_PREFIXES)

    def __init__(self, config_path: Optional[str] = None):
        # 显式传入 config_path = 单文件模式（测试/兼容）：读写都走它，不做分层。
        self._single_file = config_path is not None
        self.overlay_path: Optional[Path] = None
        if config_path is None:
            # 1) 当前**主题**的覆盖配置（主题由启动链路 ensure；不存在则回退旧布局）
            try:
                from src.core.theme_workspace import get_theme_workspace

                _candidate = get_theme_workspace().current_config_path()
                if _candidate is not None:
                    self.overlay_path = Path(_candidate)
                    config_path = str(_candidate)
            except Exception as _theme_exc:
                logger.warning("解析主题配置失败，回退旧布局: %s", _theme_exc)

        if config_path is None:
            # 根据运行环境选择配置文件路径
            import sys
            if getattr(sys, 'frozen', False):
                # 打包应用：使用应用数据目录
                from .path_manager import app_support_base
                app_data = app_support_base()
                config_path = str(app_data / "config.yaml")
            else:
                # 开发环境：读写 config.local.yaml（已在 .gitignore 中），
                # 使仓库里的 config.yaml 只作为中性模板、不被运行时写脏。
                # 首次运行时从模板复制一份出来。
                local_path = Path('config.local.yaml')
                template_path = Path('config.yaml')
                if not local_path.exists() and template_path.exists():
                    try:
                        import shutil
                        shutil.copy2(template_path, local_path)
                        logger.info("已从模板创建本地配置: %s", local_path)
                    except Exception as e:
                        logger.warning("创建本地配置失败, 回退模板: %s", e)
                config_path = str(local_path) if local_path.exists() else str(template_path)
        self.config_path = Path(config_path)
        if self._single_file:
            self.base_path = self.config_path
            self.overlay_path = None
        else:
            self.base_path = self._resolve_base_path()
        self.config = self._load_config()

    @staticmethod
    def _resolve_base_path() -> Path:
        """全局基底配置路径（各主题共享的那一份）。"""
        import sys

        if getattr(sys, 'frozen', False):
            # 打包应用：使用应用数据目录
            from .path_manager import app_support_base
            app_data = app_support_base()
            return app_data / "config.yaml"

        # 开发环境：读写 config.local.yaml（已在 .gitignore 中），
        # 使仓库里的 config.yaml 只作为中性模板、不被运行时写脏。
        # 首次运行时从模板复制一份出来。
        local_path = Path('config.local.yaml')
        template_path = Path('config.yaml')
        if not local_path.exists() and template_path.exists():
            try:
                import shutil
                shutil.copy2(template_path, local_path)
                logger.info("已从模板创建本地配置: %s", local_path)
            except Exception as e:
                logger.warning("创建本地配置失败, 回退模板: %s", e)
        return local_path if local_path.exists() else template_path

    @staticmethod
    def _read_yaml(path: Path) -> Optional[dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading config: {e}")
            return None

    def _load_config(self) -> dict:
        """**合并读**：全局基底 ← 主题覆盖（覆盖层只需写与全局不同的键）。"""
        base = self._read_yaml(self.base_path)
        if base is None:
            base = copy.deepcopy(self.DEFAULT_CONFIG)
            self._save_config(base, self.base_path)

        if self.overlay_path and self.overlay_path != self.base_path:
            overlay = self._read_yaml(self.overlay_path)
            if overlay:
                return _deep_merge(base, overlay)
        return base

    def _save_config(self, config: dict, path: Optional[Path] = None):
        """Save config to YAML file."""
        target = Path(path) if path is not None else self.config_path
        try:
            with open(target, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation support (e.g., 'vlm.model')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @staticmethod
    def _set_on_dict(config: dict, key: str, value: Any) -> None:
        keys = key.split('.')
        for k in keys[:-1]:
            if not isinstance(config.get(k), dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def set(self, key: str, value: Any):
        """Set config value with dot notation support and save to file.

        **按键路由写**：随主题的键写主题覆盖文件，其余写全局。
        写盘时从**目标文件自身**的内容出发（不把合并视图灌进去）——
        否则会把另一层的内容整份抄过来，两层从此互相污染。
        """
        self._set_on_dict(self.config, key, value)

        if self._single_file or self.overlay_path is None or self.overlay_path == self.base_path:
            target = self.base_path
        elif self._is_overlay_key(key):
            target = self.overlay_path
        else:
            target = self.base_path

        data = self._read_yaml(Path(target))
        if data is None:
            data = {}
        self._set_on_dict(data, key, value)
        self._save_config(data, Path(target))

    def get_vlm_config(self) -> dict:
        """Get VLM configuration（含多供应商支持）。

        结构:
            vlm:
              active: 供应商名
              timeout: 120
              vendors: {名称: {type, vendor, base_url, model, ...}}
        兼容旧结构: vlm.provider/base_url/model（迁移到 vendors 的单条记录）
        """
        vlm = self.get("vlm", {}) or {}

        # 多供应商：vendors dict
        vendors = vlm.get("vendors") or {}

        # 兼容旧结构：若没有 vendors，从旧的 provider/base_url/model 构造一条
        if not vendors:
            old_provider = vlm.get("provider", "url")
            old_model = vlm.get("model", "minicpm-v4.5")
            old_base = vlm.get("base_url", "http://localhost:11434")
            old_vendor_name = vlm.get("vendor") or "默认供应商"
            vendors = {
                old_vendor_name: {
                    "type": "folder" if old_provider in ("mlx", "folder") else "url",
                    "vendor": old_vendor_name,
                    "base_url": old_base,
                    "model": old_model,
                }
            }
            # 回写迁移后的结构（幂等，下次直接读 vendors）
            try:
                self.config["vlm"]["vendors"] = vendors
                self.config["vlm"]["active"] = old_vendor_name
                self._save_config(self.config)
            except Exception:
                pass

        active = vlm.get("active") or (list(vendors.keys())[0] if vendors else "default")
        active_cfg = vendors.get(active) or {}

        return {
            "provider": active_cfg.get("type", "url"),
            "vendor": active_cfg.get("vendor", active),
            "vendor_name": active,
            "base_url": active_cfg.get("base_url", "http://localhost:11434"),
            "model": active_cfg.get("model", "minicpm-v4.5"),
            "timeout": vlm.get("timeout", 60),
            "vendors": vendors,
            "active": active,
        }

    # ── 供应商管理（多供应商共存 + 切换） ──────────────────────────

    def get_vlm_vendors(self) -> dict:
        """返回全部供应商（含 active）。"""
        vlm = self.get("vlm", {}) or {}
        vendors = vlm.get("vendors") or {}
        if not vendors:
            # 触发一次迁移
            self.get_vlm_config()
            vendors = (self.get("vlm", {}) or {}).get("vendors") or {}
        return {
            "active": vlm.get("active") or (list(vendors.keys())[0] if vendors else ""),
            "vendors": vendors,
        }

    def save_vlm_vendor(self, name: str, cfg: dict) -> dict:
        """新增或更新一个供应商。返回 (成功, 消息)。"""
        name = (name or "").strip()
        if not name:
            return False, "供应商名称不能为空"
        vlm = dict(self.get("vlm", {}) or {})
        vendors = dict(vlm.get("vendors") or {})
        vendors[name] = {
            "type": cfg.get("type", "url"),
            "vendor": cfg.get("vendor") or name,
            "base_url": (cfg.get("base_url") or "").strip(),
            "model": (cfg.get("model") or "").strip(),
            # S13 ⑪：云端三要素之一——API Key 随供应商持久化（文件存储 0600，
            # 沿用 licensing 模块的存储定论，不走 Keychain）
            "api_key": (cfg.get("api_key") or "").strip(),
        }
        vlm["vendors"] = vendors
        if not vlm.get("active"):
            vlm["active"] = name
        self.set("vlm", vlm)
        return True, f"供应商「{name}」已保存"

    def delete_vlm_vendor(self, name: str) -> dict:
        """删除一个供应商。返回 (成功, 消息)。"""
        vlm = dict(self.get("vlm", {}) or {})
        vendors = dict(vlm.get("vendors") or {})
        if name not in vendors:
            return False, f"供应商「{name}」不存在"
        del vendors[name]
        vlm["vendors"] = vendors
        # 若删除的是 active，切换到一个剩余供应商
        if vlm.get("active") == name:
            vlm["active"] = next(iter(vendors), "")
        self.set("vlm", vlm)
        return True, f"供应商「{name}」已删除"

    def set_active_vlm_vendor(self, name: str) -> dict:
        """切换当前激活的供应商。返回 (成功, 消息)。"""
        vlm = dict(self.get("vlm", {}) or {})
        vendors = vlm.get("vendors") or {}
        if name not in vendors:
            return False, f"供应商「{name}」不存在"
        vlm["active"] = name
        self.set("vlm", vlm)
        return True, f"已切换到供应商「{name}」"

    def get_video_config(self) -> dict:
        """Get video processing configuration.

        返回**合并后的完整 video 块**——历史实现是手工投影(仅 2 键),
        max_frames_per_video/frame_density 等键被系统性丢弃,帧数上限
        从未生效(add-public-sale-readiness 任务 2.3 根因之一)。
        """
        video = self.get("video")
        out = dict(video) if isinstance(video, dict) else {}
        out.setdefault("frame_format", "jpg")
        out.setdefault("frames_per_scene", 5)
        return out

    def get_nfo_config(self) -> dict:
        """Get NFO configuration."""
        return {
            "format": self.get("nfo.format", "xml")
        }

    def get_database_config(self) -> dict:
        """Get database configuration."""
        return {
            "path": self.get("database.path", "./data/audit_cache.db")
        }

    @staticmethod
    def _registry_prompt(theme_name: Optional[str]) -> Optional[str]:
        """从**主题注册表**取该主题的提示词（主题不存在/无专属提示词 → None）。

        为什么走注册表：`themes`/`theme_prompts` 原本在数据库里，而"主题清单"
        必须先于"打开某个主题的数据文件"可知（见 src/core/theme_registry.py）。
        """
        if not theme_name:
            return None
        try:
            from src.core.theme_registry import get_theme_registry

            rec = get_theme_registry().get(theme_name)
            prompt = (rec or {}).get("prompt") or ""
            prompt = prompt.strip()
            if prompt and "（默认提示词" not in prompt:
                return prompt
        except Exception:
            pass
        return None

    def get_vlm_prompt(self) -> str:
        """VLM 提示词：优先**默认主题**的 prompt（注册表），回退全局配置。"""
        try:
            from src.core.theme_registry import get_theme_registry

            default_name = get_theme_registry().default_theme()
            prompt = self._registry_prompt(default_name)
            if prompt:
                return prompt
        except Exception:
            pass
        return self.get("vlm_prompt", self.DEFAULT_VLM_PROMPT)

    def get_vlm_prompt_for_theme(self, theme_name: Optional[str] = None) -> str:
        """按主题名获取 VLM 提示词（主题↔提示词联动）。

        - theme_name 指定时：查该主题的 prompt，有则用
        - 无主题或该主题无 prompt：回退 get_vlm_prompt()（默认主题 → 全局）
        """
        return self._registry_prompt(theme_name) or self.get_vlm_prompt()

    def _theme_prompt_by_name(self, theme_name: Optional[str]) -> Optional[str]:
        """按主题名取提示词; 主题不存在/无专属提示词时返回 None。"""
        return self._registry_prompt(theme_name)

    def get_active_analysis_theme(self) -> str:
        """当前分析主题（注册表的 current；未指定时回退默认主题）。"""
        try:
            from src.core.theme_registry import get_theme_registry

            return get_theme_registry().effective_current()
        except Exception:
            return self.get("analysis.active_theme", "") or ""

    def set_active_analysis_theme(self, theme_name: str) -> None:
        """设置当前分析主题（写注册表；空字符串表示不指定）。"""
        from src.core.theme_registry import get_theme_registry

        reg = get_theme_registry()
        if theme_name:
            reg.set_current(theme_name)
        else:
            data = reg.load()
            data["current"] = ""
            reg.save(data)

    def get_analysis_prompt(self, theme_name: Optional[str] = None) -> str:
        """动态解析分析提示词。

        优先级: 指定主题 → 当前分析主题(前端选择) → 默认主题 → 静态兜底。
        每次调用实时查库, 前端改主题/提示词后下一次分析即生效。
        """
        for name in (theme_name, self.get_active_analysis_theme()):
            prompt = self._theme_prompt_by_name(name)
            if prompt:
                return prompt
        return self.get_vlm_prompt() or self.DEFAULT_VLM_PROMPT

    def set_vlm_prompt(self, prompt: str):
        """Save the VLM prompt to config."""
        self.set("vlm_prompt", prompt)

    def get_all(self) -> dict:
        """Get all configuration."""
        return self.config.copy()

    def update(self, updates: dict):
        """Update multiple config values and save."""
        for key, value in updates.items():
            self.set(key, value)


# ── 进程内共享实例 ──────────────────────────────────────────────────
# 为什么需要：``ConfigManager()`` 每次都新建实例并各自缓存一份 dict。
# 若「写」与「读」落在不同实例上（例如路由里 ``ConfigManager().set()`` 只写盘，
# 而 ``main`` 里长持的实例仍读旧值），切换分析主题后 ``/config`` 与分析链路
# 都看不到新值 —— 这正是"切主题后提示词不跟随"的根因。
# 约定：**写配置一律走 get_config_manager()**；只读处可自行 new（构造时读盘）。
_shared_config_manager: Optional["ConfigManager"] = None


def get_config_manager() -> "ConfigManager":
    """返回进程内共享的 ConfigManager 实例（单例）。"""
    global _shared_config_manager
    if _shared_config_manager is None:
        _shared_config_manager = ConfigManager()
    return _shared_config_manager
