"""预置分析主题：开箱即用的主题配置（标签映射文件 + 提示词）。

首次启动、或用户在「分析主题管理」里点「载入预置主题」时按名称幂等播种。
预置只是初始值——用户可自由增删改。

术语区分：本模块的「分析主题」= 业务主题（标签库 + 提示词 + 人脸资源库），
区别于「外观」（界面深/浅色皮肤）。
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 名称与现有库的既有主题对齐，避免播种后出现「日常」+「日常生活」这类重复概念
# 公开版仅附带通用主题；成人内容/女女/男女/男男四主题的词典与提示词不入公开仓
# （内部完整版 6 主题，官方安装包按内部版分发）。
PRESET_THEMES: List[Dict] = [
    {"name": "日常", "description": "日常生活/通用内容",
     "mapping": "日常生活标签映射表.md", "default": True},
    {"name": "AI 视频", "description": "AI 生成/短剧内容质控",
     "mapping": "AI生成审计标签映射表.md"},
]


def presets_dir() -> Path:
    """预置素材目录：开发态 = 项目 resources/themes；打包态 = bundle 内 Resources/resources/themes。"""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent.parent / "Resources"
    else:
        base = Path(__file__).resolve().parent.parent.parent
    return base / "resources" / "themes"


def build_preset_records(themes_dir: Optional[Path] = None) -> List[Dict]:
    """把预置主题整理成注册表记录（映射表缺失的主题跳过并告警）。"""
    themes_dir = Path(themes_dir) if themes_dir else presets_dir()
    records: List[Dict] = []
    for preset in PRESET_THEMES:
        name = preset["name"]
        mapping_file = themes_dir / preset["mapping"]
        if not mapping_file.exists():
            logger.warning("预置主题「%s」的映射表缺失，跳过: %s", name, mapping_file)
            continue
        rec: Dict = {
            "name": name,
            "default": bool(preset.get("default")),
            "label_mapping_path": str(mapping_file),
        }
        if preset.get("description"):
            rec["description"] = preset["description"]
        prompt_name = preset.get("prompt")
        if prompt_name:
            prompt_file = themes_dir / prompt_name
            if prompt_file.exists():
                rec["prompt"] = prompt_file.read_text(encoding="utf-8")
            else:
                logger.warning("预置主题「%s」的提示词缺失，仅建主题: %s", name, prompt_file)
        records.append(rec)
    return records


def seed_preset_themes(registry, themes_dir: Optional[Path] = None) -> int:
    """按名称幂等播种预置分析主题到**主题注册表**（数据文件之外）。

    - 同名已存在：不覆盖用户改动（由 `ThemeRegistry.seed` 保证）
    - 映射表缺失：跳过该主题并告警（不阻断启动）
    - 提示词文件缺失：仅告警，主题照建

    Args:
        registry: `src.core.theme_registry.ThemeRegistry`
        themes_dir: 预置素材目录，默认 presets_dir()

    Returns:
        新建的主题数
    """
    created = registry.seed(build_preset_records(themes_dir))
    if created:
        logger.info("已播种 %d 个预置分析主题", created)
    return created
