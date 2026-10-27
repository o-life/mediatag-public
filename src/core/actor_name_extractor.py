"""Person-name extractor for video filenames.

Extracts person names from video filenames using regex pattern matching and a
configurable ignore-keyword list (platform names, release tags, device names,
production-company words, ...). The list lives in config.yaml under
``filename_parsing.ignore_keywords``.
"""

import os
import re
from typing import List

# Fallback when config is unavailable: common platform / release / device /
# production-company markers. Kept identical to the shipped config default.
_DEFAULT_IGNORE_KEYWORDS = (
    "bilibili", "youtube", "vimeo", "netflix", "dailymotion", "twitter",
    "instagram", "tiktok", "抖音", "微博", "微信",
    "web-dl", "webrip", "bluray", "brrip", "hdtv", "dvdrip", "remux",
    "x264", "x265", "hevc", "hdr", "1080p", "720p", "2160p", "4k", "8k",
    "iphone", "ipad", "android", "gopro", "dji",
    "screen recording", "录屏", "截屏",
    "studio", "production", "entertainment", "media", "film", "video",
)


class ActorNameExtractor:
    """Extract person names from video file paths.

    Pipeline: strip path/extension → remove dates & technical markers →
    split on connectors → match capitalised name patterns → drop ignore-keywords.
    """

    # Regex: year (4-digit) or compact date (8-digit)
    _YEAR_DATE_RE = re.compile(r"\b\d{4}\b|\b\d{8}\b")

    # Regex: video code patterns (e.g., ABC-123, ABCD-456, XYZ-1234)
    _VIDEO_CODE_RE = re.compile(r"\b[A-Z]{2,6}-\d{2,5}\b", re.IGNORECASE)

    # Regex: technical / quality markers
    _TECH_MARKER_RE = re.compile(
        r"\b(HD|4K|1080p|720p|FHD|"
        r"[A-Z]{1,3}\d+|\d+[A-Z]{1,3}|"
        r"no\s*\d+|"
        r"uncensored|remastered|full)\b",
        re.IGNORECASE,
    )

    # Regex: person-name pattern — 1 to 3 capitalised tokens (including dot-separated)
    # 不使用 \b 在末尾，避免在点号处断开
    # 每个词都以大写字母开头，后面跟任意字母
    _NAME_PATTERN_RE = re.compile(
        r"\b[A-Z][a-zA-Z]*(?:[.\s]+[A-Z][a-zA-Z]*){0,2}"
    )

    # Config key holding the user-editable ignore-keyword list.
    CONFIG_KEY = "filename_parsing.ignore_keywords"

    def __init__(self, ignore_keywords=None) -> None:
        if ignore_keywords is None:
            ignore_keywords = self._load_ignore_keywords()
        # Drop empty entries: an empty keyword would match every candidate.
        self._ignore_lower = frozenset(
            k for k in (str(x).strip().lower() for x in ignore_keywords) if k
        )

    @classmethod
    def _load_ignore_keywords(cls) -> List[str]:
        """Read the ignore-keyword list from config, falling back to defaults."""
        try:
            from src.core.config import ConfigManager
            configured = ConfigManager().get(cls.CONFIG_KEY, None)
        except Exception:
            configured = None
        if configured is None:
            return list(_DEFAULT_IGNORE_KEYWORDS)
        return [str(k) for k in configured if str(k).strip()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_filename(self, file_path: str) -> List[str]:
        """Extract actor names from a video file path.

        Args:
            file_path: Absolute or relative path to the video file.

        Returns:
            Ordered list of unique actor names found in the filename.
            Returns an empty list when no valid names are detected.
        """
        filename = os.path.splitext(os.path.basename(file_path))[0]

        cleaned = self._strip_dates(filename)
        cleaned = self._strip_video_codes(cleaned)
        cleaned = self._strip_tech_markers(cleaned)

        parts = self._split_on_connectors(cleaned)

        names: List[str] = []
        seen: set = set()

        for part in parts:
            matches = self._NAME_PATTERN_RE.findall(part)
            for name in matches:
                # Normalize dots to spaces for consistent storage
                normalized = re.sub(r"\.", " ", name).strip()
                if self._is_valid_name(normalized) and normalized not in seen:
                    names.append(normalized)
                    seen.add(normalized)

        return names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_dates(self, text: str) -> str:
        return self._YEAR_DATE_RE.sub("", text)

    def _strip_video_codes(self, text: str) -> str:
        """Remove video code patterns like ABC-123, XYZ-1234."""
        return self._VIDEO_CODE_RE.sub("", text)

    def _strip_tech_markers(self, text: str) -> str:
        return self._TECH_MARKER_RE.sub("", text)

    @staticmethod
    def _split_on_connectors(text: str) -> List[str]:
        """Split on hyphens, underscores, or spaces (but NOT dots)."""
        # 只按连字符、下划线、空格分割，不按点号分割
        # 因为点号在名字中是分隔符（如 "Ryo.McCall"）
        return [p.strip() for p in re.split(r"[-_\s]+", text) if p.strip()]

    def _is_valid_name(self, name: str) -> bool:
        """Return True when *name* is a plausible person name."""
        tokens = name.split()
        if len(tokens) < 1 or len(tokens) > 3:
            return False
        name_lower = name.lower()
        # Drop candidates hitting any configured ignore keyword (substring match).
        if any(keyword in name_lower for keyword in self._ignore_lower):
            return False
        
        # First token must start with an uppercase letter
        if not tokens[0][0].isupper():
            return False
        return True

    def extract_from_watermark(self, frames_path: str, vlm_client) -> List[str]:
        """Extract actor names from video frame watermarks using VLM.

        Args:
            frames_path: Path to frames directory
            vlm_client: OllamaVLMClient instance

        Returns:
            List of extracted actor names
        """
        from pathlib import Path
        import logging
        logger = logging.getLogger(__name__)

        frames_dir = Path(frames_path)
        if not frames_dir.exists():
            return []

        # 选择中间帧作为代表性样本
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            return []

        # 选择中间帧
        mid_idx = len(frame_files) // 2
        representative_frame = frame_files[mid_idx]

        # VLM prompt for watermark detection
        prompt = (
            "请仔细检查这张图片，识别画面中的水印文字。\n\n"
            "【要求】\n"
            "1. 只关注画面角落或边缘的半透明文字\n"
            "2. 水印通常是人物名字或工作室名称\n"
            "3. 如果看到人名，请完整输出\n"
            "4. 如果没有看到水印，回复：无水印\n\n"
            "【输出格式】\n"
            "- 如果检测到人物名：直接输出名字（如：John Smith）\n"
            "- 如果检测到工作室名：输出工作室名\n"
            "- 如果没有水印：回复：无水印\n\n"
            "请只输出识别到的文字内容，不要其他解释。"
        )

        try:
            # 调用 VLM 分析单帧
            result = vlm_client.analyze_frames(str(representative_frame.parent), prompt)
            result = result.strip()

            # 解析结果
            if not result or result == "无水印" or "无水印" in result:
                return []

            # 提取可能的人物名（首字母大写的词组）
            names = self._NAME_PATTERN_RE.findall(result)
            valid_names = [name for name in names if self._is_valid_name(name)]

            logger.info(f"水印识别结果: {result} -> {valid_names}")
            return valid_names

        except Exception as e:
            logger.warning(f"VLM 水印识别失败: {e}")
            return []

    def confirm_actor_name(self, actor_id: int, candidate_names: List[str], db) -> str:
        """Confirm the primary actor name from candidates using frequency analysis.

        Args:
            actor_id: Actor ID in database
            candidate_names: List of candidate names from various sources
            db: Database session

        Returns:
            Confirmed primary name, or empty string if no valid candidates
        """
        import json
        import logging
        from src.db.models import Actor
        logger = logging.getLogger(__name__)

        if not candidate_names:
            return ""

        # 统计每个候选名称的出现频率
        name_counts = {}
        for name in candidate_names:
            name = name.strip()
            if not name:
                continue
            name_counts[name] = name_counts.get(name, 0) + 1

        if not name_counts:
            return ""

        # 选择出现次数最多的作为主名
        primary_name = max(name_counts, key=name_counts.get)
        primary_count = name_counts[primary_name]

        # 其他名称作为别名
        aliases = [name for name, count in name_counts.items() if name != primary_name]

        # 更新数据库
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if actor:
            actor.primary_name = primary_name
            # aliases 字段是 TEXT 类型，需要序列化为 JSON 字符串
            actor.aliases = json.dumps(aliases, ensure_ascii=False) if aliases else None
            db.commit()

            # 记录到 task_actor_names 表（用于追溯）
            # 注意：这里不关联具体 task，只记录确认结果
            logger.info(
                f"人物 {actor_id} 确认名称: {primary_name} (出现 {primary_count} 次), "
                f"别名: {aliases}"
            )

        return primary_name
