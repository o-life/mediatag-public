import math
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sqlalchemy.orm import Session
from src.db.models import DictionaryEntry, AuditQueueItem
import logging

logger = logging.getLogger(__name__)

# 同义词扩展表：将常见变体映射到标准表达
SYNONYMS: Dict[str, List[str]] = {
    # 场景
    "卧室": ["卧房", "寝室", "房间", "睡房"],
    "浴室": ["卫生间", "厕所", "洗手间", "澡堂"],
    "厨房": ["灶房", "灶台"],
    "客厅": ["起居室", "大厅"],
    "户外": ["室外", "野外", "露天"],
    "办公室": ["写字楼", "工作间"],
    "学校": ["教室", "校园"],
    "医院": ["诊所", "病房"],
    # 人数
    "单人": ["一个人", "独自", "单独"],
    "双人": ["两个人", "两人", "一对"],
    "多人": ["几个人", "群体", "一群"],
    # 体型
    "苗条": ["瘦", "纤细", "修长"],
    "丰满": ["胖", "微胖", "圆润", "丰腴"],
    "肌肉": ["健壮", "结实", "壮硕"],
    # 年龄（"岁"曾在此表中,子串匹配致"18岁"→青年,2026-09-30 审计移除）
    "年轻": ["青年", "年轻人"],
    "中年": ["中年人", "成熟"],
    "老年": ["老人", "年迈"],
    # 服装
    "制服": ["工作服", "职业装"],
    "泳装": ["泳衣", "比基尼"],
    "睡衣": ["睡袍", "家居服"],
    # 动作/体位
    "站立": ["站着", "站立姿势"],
    "坐姿": ["坐着", "坐下"],
    "躺卧": ["躺着", "卧姿"],
}

# 反向同义词映射：从变体查找标准词
_REVERSE_SYNONYMS: Dict[str, str] = {}
for _standard, _variants in SYNONYMS.items():
    for _variant in _variants:
        _REVERSE_SYNONYMS[_variant] = _standard
    _REVERSE_SYNONYMS[_standard] = _standard


# ── 标签质量门槛纯函数(2026-09-30 批3,依据 docs/数据质量/审计-2026-09-30) ──

# 帧级互斥对:一帧画面不能同时呈现(与 scripts/tag_quality_audit.py 同源,
# 剔除了白天×雪天这类可共存的伪互斥)
CONTRADICTION_PAIRS_FRAME: List[Tuple[str, str]] = [
    # 人数
    ("单人", "双人"), ("单人", "三人"), ("单人", "多人"),
    ("双人", "三人"), ("双人", "多人"), ("三人", "多人"),
    # 时间天气
    ("白天", "夜晚"), ("晴天", "雨天"), ("晴天", "阴天"),
    ("日出", "日落"),
    # 场景(强互斥)
    ("卧室", "健身房"), ("卧室", "办公室"), ("卧室", "超市"),
    ("卧室", "商场"), ("卧室", "餐厅"), ("卧室", "更衣室"),
    ("卧室", "车内"), ("卧室", "博物馆"),
    ("浴室", "办公室"), ("浴室", "健身房"), ("车内", "健身房"),
    # 服装
    ("全裸", "西装"), ("全裸", "制服"), ("半裸", "西装"),
    # 属性
    ("Gay", "Lesbian"), ("异性恋", "Gay"), ("异性恋", "Lesbian"),
    ("少女感", "熟女"), ("少年感", "熟男"), ("中年", "老年"),
    # 内容类型 / 视觉风格
    ("古装", "科幻"), ("写实", "二次元"), ("写实", "超现实"),
    # 身材 / 肤色
    ("肌肉", "纤细"), ("肌肉", "骨感"), ("丰满", "骨感"),
    ("白皙", "黝黑"), ("白皙", "古铜"),
]

# 文件级互斥对:整个文件不成立的组合(场景/时间/服装在多场景视频中
# 可先后出现,故只保留跨维度硬冲突;人数维度沿用既有"保留最大值"逻辑)
CONTRADICTION_PAIRS_FILE: List[Tuple[str, str]] = [
    ("写实", "二次元"), ("写实", "超现实"), ("古装", "科幻"),
    ("Gay", "Lesbian"), ("异性恋", "Gay"), ("异性恋", "Lesbian"),
]


def enforce_vocabulary(tags: List[str], std_tags: Set[str]) -> Tuple[List[str], List[str]]:
    """闸 2:解析边界词典强制。返回 (词典内标签, 被剔除的非词典标签)。

    VLM 原文直写 tag_frame_mappings 曾产生 31,978 种标签(词典仅 210);
    非词典标签从此不再入库。剔除项由调用方记日志/计数。
    """
    kept, dropped = [], []
    for t in tags or []:
        t = (t or "").strip()
        if not t:
            continue
        (kept if t in std_tags else dropped).append(t)
    seen: Set[str] = set()
    kept = [t for t in kept if not (t in seen or seen.add(t))]
    return kept, dropped


def apply_frequency_gate(
    tag_support: Dict[str, int],
    n_units: int,
    min_support: int = 2,
    ratio: float = 0.25,
    max_tags: int = 30,
) -> Set[str]:
    """闸 4:证据频率门槛 + 数量上限。

    标签须在 ≥ min(min_support, max(1, ceil(n_units×ratio))) 个场景/单元
    出现才保留(短文件不误杀,长文件须多场景佐证);超出 max_tags 按
    支持数降序截断。tag_support = 标签 → 出现过的场景数/单元数。
    """
    if not tag_support:
        return set()
    if n_units <= 0:
        return set(tag_support)
    required = min(min_support, max(1, math.ceil(n_units * ratio)))
    kept = {t for t, c in tag_support.items() if c >= required}
    if len(kept) > max_tags:
        kept = set(sorted(kept, key=lambda t: (-tag_support[t], t))[:max_tags])
    return kept


def filter_contradictions(
    tags: List[str],
    counts: Optional[Dict[str, int]] = None,
    file_level: bool = False,
) -> List[str]:
    """闸 5:互斥共现过滤。两互斥标签同现时保留证据多的一方(平局保先)。

    file_level=False 用帧级全量互斥对;True 只用文件级硬冲突子集
    (场景/时间/服装在多场景视频中可先后出现,不能按互斥删除)。
    """
    pairs = CONTRADICTION_PAIRS_FILE if file_level else CONTRADICTION_PAIRS_FRAME
    counts = counts or {}
    tags = list(dict.fromkeys(t for t in tags if t))
    removed: Set[str] = set()
    for a, b in pairs:
        if a in removed or b in removed:
            continue
        if a in tags and b in tags:
            ca, cb = counts.get(a, 1), counts.get(b, 1)
            loser = b if ca >= cb else a
            removed.add(loser)
            logger.info("互斥冲突: %s×%s 共现, 剔除 %s (证据 %s≤%s)",
                        a, b, loser, min(ca, cb), max(ca, cb))
    return [t for t in tags if t not in removed]


def sanitize_alias(alias: str) -> Optional[str]:
    """别名入库守卫:拒绝 <2 字与纯数字,防"岁"类过泛别名再次污染。"""
    a = (alias or "").strip()
    if len(a) < 2 or a.isdigit():
        return None
    return a


class DictionaryManager:
    def __init__(self, md_path: str, db_session: Session):
        self.md_path = Path(md_path)
        self.db = db_session

    def parse_md_to_dict(self) -> Dict[str, List[str]]:
        """
        Parses the standard tag mapping markdown file into a dictionary of {standard_tag: [aliases]}.
        """
        mapping = {}
        try:
            content = self.md_path.read_text(encoding='utf-8')
            # Split by sections starting with '###' (the dimensions)
            sections = re.split(r'\n### ', content)
            
            for section in sections[1:]: # Skip the header/intro
                # Extract tables within each section
                tables = re.findall(r'\|\|\s*([^|]*)\|.*?\|\n(?:\| :--- \|.*?\|\n)*(\|.*\|)', section, re.S)
                for header_row, body in tables:
                    # The table rows are separated by newlines and start/end with |
                    # Wait, a better way to parse the markdown table
                    pass

            # Revised parsing logic for the specific format used in standard_tags.md
            # Look for rows like | **标签** | 别名1, 别名2 | 说明 | or | **标签** | 说明 |
            # Pattern 1: 3 columns (tag, aliases, description)
            pattern1 = r'\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]*?)\s*\|\s*([^|]*)\s*\|'
            matches1 = re.findall(pattern1, content)
            for match in matches1:
                standard_tag = match[0].strip()
                aliases_str = match[1].strip()
                # Split aliases by comma or Chinese comma, and strip whitespace
                aliases = [a.strip() for a in re.split(r'[,，]', aliases_str) if a.strip()]
                mapping[standard_tag] = aliases
            
            # Pattern 2: 2 columns (tag, description) - for tables without aliases
            pattern2 = r'\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]*)\s*\|'
            matches2 = re.findall(pattern2, content)
            for match in matches2:
                standard_tag = match[0].strip()
                # Skip if already matched by pattern1
                if standard_tag not in mapping:
                    # No aliases, just the tag itself
                    mapping[standard_tag] = []

            return mapping
        except Exception as e:
            print(f"Error parsing MD dictionary: {e}")
            return {}

    def sync_to_db(self):
        """
        Syncs the parsed MD mappings to the SQLite dictionary table.
        """
        mapping = self.parse_md_to_dict()
        for tag, aliases in mapping.items():
            # Update or insert
            entry = self.db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == tag).first()
            if entry:
                entry.aliases = ",".join(aliases)
            else:
                new_entry = DictionaryEntry(standard_tag=tag, aliases=",".join(aliases))
                self.db.add(new_entry)
        self.db.commit()

    def map_raw_text_to_tags(self, raw_text: str) -> Tuple[List[str], List[str]]:
        """
        Maps VLM output text to standard tags based on the dictionary aliases.
        Uses jieba segmentation and synonym expansion for improved matching.
        Returns: (mapped_tags, unmatched_labels)
        """
        # Fetch all entries from DB for fast lookup
        entries = self.db.query(DictionaryEntry).all()
        mapped_tags = set()
        unmatched_labels = []

        # Extract keywords from raw text using jieba
        keywords = self._extract_keywords(raw_text)

        # Expand keywords with synonyms
        expanded_keywords = self._expand_synonyms(keywords)

        # Build a combined set for matching: original + expanded
        all_keywords = set(keywords) | expanded_keywords

        for entry in entries:
            aliases = [a.strip() for a in entry.aliases.split(',')] if entry.aliases else []
            tag = entry.standard_tag
            # 标准标签: ≥3 字才允许原文子串命中(长词误中率低); ≤2 字必须分词精确命中
            if (len(tag) >= 3 and tag in raw_text) or tag in all_keywords:
                mapped_tags.add(tag)
                continue
            # 别名: 一律分词精确命中。子串包含曾使"岁"命中任何含"岁"的文本
            # (青年由此覆盖 4,185 文件),2026-09-30 审计后废除
            if any(alias and alias in all_keywords for alias in aliases):
                mapped_tags.add(tag)

        # Find unmatched keywords
        for keyword in keywords:
            if keyword not in mapped_tags and len(keyword) > 1:
                # Check if it matches any alias
                matched = False
                for entry in entries:
                    aliases = [a.strip() for a in entry.aliases.split(',')] if entry.aliases else []
                    if keyword == entry.standard_tag or keyword in aliases:
                        matched = True
                        break

                if not matched:
                    unmatched_labels.append(keyword)

        # Normalize synonyms before adding to audit queue
        from src.core.synonyms import normalize_tags
        unmatched_labels = normalize_tags(unmatched_labels)

        # Add unmatched labels to audit queue
        for label in unmatched_labels:
            self._add_to_audit_queue(label)

        # 处理互斥标签冲突
        mapped_tags = self._resolve_tag_conflicts(list(mapped_tags), raw_text)

        return sorted(mapped_tags), unmatched_labels

    def _expand_synonyms(self, keywords: List[str]) -> Set[str]:
        """Expand keywords using synonym table (精确映射)。

        仅接受完全相等的词条:关键词在 SYNONYMS 表中 → 加其变体与标准词;
        关键词是某变体 → 加标准词。原双向部分包含(variant in keyword or
        keyword in variant)会把所有互为子串的词条联动扩展,是过泛映射来源
        之一,2026-09-30 审计后废除(人数类此前已豁免,现全维度一律精确)。
        """
        expanded = set()
        for keyword in keywords:
            # Check if keyword is a standard term with variants
            if keyword in SYNONYMS:
                expanded.update(SYNONYMS[keyword])
                expanded.add(keyword)
            # Check if keyword is a variant of a standard term
            if keyword in _REVERSE_SYNONYMS:
                expanded.add(_REVERSE_SYNONYMS[keyword])
        return expanded

    def _resolve_tag_conflicts(self, tags: List[str], raw_text: str) -> List[str]:
        """处理互斥标签冲突。
        
        规则：
        1. 人数维度：保留"最大值"（多人 > 三人 > 双人 > 单人）
        2. 场景维度：保留"主场景"（出现次数最多的场景）
        3. 其他维度：保留所有标签
        
        Args:
            tags: 映射后的标签列表
            raw_text: VLM 原始输出文本（用于统计场景出现次数）
        
        Returns:
            处理冲突后的标签列表
        """
        # 人数维度：保留最大值
        person_count_priority = {"多人": 4, "三人": 3, "双人": 2, "单人": 1}
        person_count_tags = [tag for tag in tags if tag in person_count_priority]
        
        if len(person_count_tags) > 1:
            # 保留优先级最高的
            best_tag = max(person_count_tags, key=lambda t: person_count_priority[t])
            tags = [tag for tag in tags if tag not in person_count_tags or tag == best_tag]
            logger.info(f"人数维度冲突处理: {person_count_tags} -> {best_tag}")
        
        # 场景维度：保留出现次数最多的场景
        scene_tags = ["卧室", "浴室", "客厅", "厨房", "户外", "办公室", "学校", "医院", "酒店", "酒店房间"]
        scene_tags_found = [tag for tag in tags if tag in scene_tags]
        
        if len(scene_tags_found) > 1:
            # 统计每个场景在原始文本中的出现次数
            scene_counts = {}
            for scene in scene_tags_found:
                count = raw_text.count(scene)
                scene_counts[scene] = count
            
            # 保留出现次数最多的场景
            if scene_counts:
                main_scene = max(scene_counts.items(), key=lambda x: x[1])[0]
                tags = [tag for tag in tags if tag not in scene_tags_found or tag == main_scene]
                logger.info(f"场景维度冲突处理: {scene_counts} -> {main_scene}")

        # 闸 5:文件级互斥对(写实×二次元等跨维度硬冲突;场景/时间/服装
        # 在多场景视频中可先后出现,不在此层删)
        tags = filter_contradictions(tags, file_level=True)

        return tags

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (improved implementation)."""
        # Try to extract JSON array if present
        json_match = re.search(r'\[([^\]]+)\]', text)
        if json_match:
            try:
                import json
                tags_str = '[' + json_match.group(1) + ']'
                tags = json.loads(tags_str)
                if isinstance(tags, list):
                    return [t.strip() for t in tags if isinstance(t, str) and t.strip()]
            except:
                pass
        
        # Extract comma-separated tags (only if parts look like actual tags, not long phrases)
        if ',' in text or '，' in text:
            parts = re.split(r'[,，]', text)
            tags = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
            # Only treat as tag list if all parts are short (likely tags, not sentences)
            if tags and all(len(t) <= 6 for t in tags):
                return tags
        
        # For natural language, use jieba for Chinese word segmentation
        keywords = set()
        
        # Get all standard tags and aliases for reverse matching
        entries = self.db.query(DictionaryEntry).all()
        all_known_tags = set()
        for entry in entries:
            all_known_tags.add(entry.standard_tag)
            if entry.aliases:
                aliases = [a.strip() for a in entry.aliases.split(',') if a.strip()]
                all_known_tags.update(aliases)
        
        # Check if any known tags appear in the text
        for tag in all_known_tags:
            if tag in text:
                keywords.add(tag)
        
        # Use jieba for Chinese word segmentation
        try:
            import jieba
            words = jieba.cut(text)
            for word in words:
                word = word.strip()
                # Filter: keep words with 2+ characters
                if len(word) >= 2:
                    keywords.add(word)
        except ImportError:
            # Fallback: extract 2-4 character Chinese words
            chinese_patterns = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
            # Filter out common phrases
            common_phrases = {'这个', '视频', '展示', '包含', '只有', '参与', '可以', '看到', '出现'}
            chinese_patterns = [p for p in chinese_patterns if p not in common_phrases]
            keywords.update(chinese_patterns)
        
        # Extract English words (3+ characters)
        english_words = re.findall(r'[a-zA-Z]{3,}', text)
        keywords.update([w.lower() for w in english_words])
        
        # Filter out common stop words
        stop_words = {
            '这个', '那个', '什么', '怎么', '为什么', '可以', '能够', '应该',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
            'to', 'for', 'of', 'and', 'or', 'this', 'that', 'with', 'from', 
            'have', 'has', 'it', 'its', 'as', 'by', 'be', 'been', 'being',
            'shows', 'scene', 'action', 'performer'
        }
        keywords = {k for k in keywords if k.lower() not in stop_words and len(k) > 1}
        
        return list(keywords)

    def _add_to_audit_queue(self, raw_label: str):
        """Add unmatched label to audit queue."""
        # Filter out meaningless and irrelevant labels
        meaningless_labels = {
            # Uncertainty words
            '未见', '疑似', '可能', '或许', '大概', '也许', '不确定', '似乎', '像是',
            # Descriptive verbs (not tags)
            '看到', '出现', '展示', '包含', '呈现', '显示',
            # Pronouns and question words
            '这个', '那个', '什么', '怎么', '为什么',
            # Vague/general terms
            '关键', '主要', '不同', '同时', '部分', '多处', '轻微', '平均', '程度', '比较',
            '人均', '偏浅', '偏深',
            # Generic nouns (meaningless in any context)
            '同样', '图片', '视频', '画面', '场景', '内容', '信息', '数据', '文件',
            '格式', '类型', '方式', '方法', '情况', '状态', '问题', '结果', '效果',
            '质量', '清晰度', '分辨率', '环境', '角度', '光线', '颜色',
            # Spatial/positional words
            '背景', '前景', '中间', '旁边', '附近', '周围', '上下', '左右', '前后',
            '里外', '内外', '远近', '高低', '大小', '多少', '长短', '宽窄',
            # English common words (not adult content)
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for',
            'of', 'and', 'or', 'but', 'with', 'from', 'by', 'as', 'it', 'its',
            'not', 'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would', 'can',
            'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those',
            'shows', 'scene', 'action', 'performer', 'video', 'image', 'picture',
            'person', 'people', 'man', 'woman', 'room', 'place', 'area', 'view',
            'same', 'different', 'similar', 'like', 'also', 'only', 'very', 'much',
            # Video technical descriptions (not meaningful for classification)
            '慢动作', '快进', '回放', '特写镜头', '远景', '近景',
            # Meaningless classification words
            '人群', '为主', '居多', '偏多', '偏少',
        }
        if raw_label in meaningless_labels or len(raw_label.strip()) < 2:
            return

        # Filter out pure English words less than 4 characters (likely noise)
        if raw_label.isascii() and len(raw_label) < 4:
            return

        # ── 成人内容特别标签推荐保留 ──
        # 即使不在标准库中，这些描述性标签也有价值，推荐添加到标签库
        valuable_descriptive_tags = {
            # 肤色
            '白皙', '古铜', '黝黑', '健康色', '小麦色',
            # 体型特征
            '纤细', '丰满', '精壮', '肌肉', '微胖', '熊系', '虎背熊腰',
            # 面部特征
            '清纯', '精致', '熟男', '少年感', '刚毅', '粗犷',
            # 场景氛围
            '酒店浴室', '泳池', '性爱场景', '有情节',
            # 镜头语言
            '稳定特写', '动态运镜', '全景扫描', '近距离拍摄',
            # 成人内容特有
            '器官特写', '身体线条', '全身像', '第一视角',
        }
        if raw_label in valuable_descriptive_tags:
            logger.info(f"特别标签推荐保留: '{raw_label}'（有价值描述性标签）")
        
        # Check if already exists
        existing = self.db.query(AuditQueueItem).filter(
            AuditQueueItem.raw_label == raw_label,
            AuditQueueItem.status == 'pending'
        ).first()
        
        if existing:
            existing.count += 1
        else:
            # Try to find AI recommendation
            ai_rec = self._find_ai_recommendation(raw_label)
            new_item = AuditQueueItem(
                raw_label=raw_label,
                count=1,
                ai_recommendation=ai_rec,
                status='pending'
            )
            self.db.add(new_item)
        
        self.db.commit()

    def _find_ai_recommendation(self, raw_label: str) -> Optional[str]:
        """Find AI recommendation for unmatched label using TF-IDF + multi-strategy matching.

        Matching strategy (in priority order):
        1. Exact match in standard tags or aliases
        2. TF-IDF similarity with context weighting (opt-5)
        3. Category name match (if raw_label is a category name, return it directly)
        4. Synonym expansion match
        5. Jieba-segmented token match
        6. Edit distance similarity (threshold 0.5)
        7. Substring containment match
        """
        entries = self.db.query(DictionaryEntry).all()

        if not entries:
            return None

        # Build category set from entries
        categories = set()
        for entry in entries:
            if entry.category:
                categories.add(entry.category)

        # Build TF-IDF index from all tags + aliases (opt-5)
        from collections import Counter
        import math

        # Build document frequency map
        all_tokens = []
        doc_freq = Counter()
        for entry in entries:
            tokens = [entry.standard_tag.lower()]
            if entry.aliases:
                tokens.extend([a.strip().lower() for a in entry.aliases.split(',') if a.strip()])
            all_tokens.append(tokens)
            for t in set(tokens):
                doc_freq[t] += 1

        # TF-IDF for raw_label tokens
        try:
            import jieba
            raw_tokens = [w.strip() for w in jieba.cut(raw_label) if len(w.strip()) >= 1]
        except ImportError:
            raw_tokens = [raw_label.lower()]

        raw_token_set = set(raw_tokens)
        idf = {t: math.log(len(all_tokens) / max(doc_freq[t], 1)) for t in raw_token_set}

        # Score each entry using TF-IDF + synonym expansion + edit distance
        best_match = None
        best_score = 0.0
        threshold = 0.3  # Lower threshold for TF-IDF matching

        for idx, entry in enumerate(all_tokens):
            std_tag = entries[idx].standard_tag
            aliases = [a.strip() for a in entries[idx].aliases.split(',')] if entries[idx].aliases else []

            # TF-IDF score
            tfidf_score = 0.0
            for t in raw_token_set:
                if t in entry or any(t in a for a in aliases):
                    tfidf_score += idf.get(t, 1.0)
            if entry:
                tfidf_score /= len(entry)

            # Synonym expansion bonus (opt-5)
            synonym_bonus = 0.0
            for keyword in raw_tokens:
                if keyword in SYNONYMS:
                    for variant in SYNONYMS[keyword]:
                        if variant == std_tag or variant in aliases:
                            synonym_bonus = 0.2
                            break
                if keyword in _REVERSE_SYNONYMS and _REVERSE_SYNONYMS[keyword] == std_tag:
                    synonym_bonus = max(synonym_bonus, 0.2)

            # Edit distance fallback
            edit_score = 0.0
            for candidate in [std_tag] + aliases:
                dist = self._levenshtein_distance(raw_label.lower(), candidate.lower())
                max_len = max(len(raw_label), len(candidate))
                if max_len > 0:
                    sim = 1.0 - dist / max_len
                    edit_score = max(edit_score, sim)

            # Context weight: category match bonus (opt-5)
            category_bonus = 0.1 if entries[idx].category in categories and raw_label.lower() == entries[idx].category.lower() else 0.0

            # Combined score
            combined = tfidf_score * 0.4 + edit_score * 0.3 + synonym_bonus + category_bonus
            if combined > best_score and combined >= threshold:
                best_score = combined
                best_match = std_tag

        if best_match:
            logger.info(f"AI recommendation for '{raw_label}': '{best_match}' (score={best_score:.3f})")

        return best_match

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return DictionaryManager._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def update_md_from_db(self):
        """
        TBD: Sync DB changes back to the MD file in a readable format.
        Implementation depends on how heavily we want to maintain MD structure.
        """
        pass

    def save_to_md(self):
        """Save current dictionary to markdown file."""
        # TODO: Implement MD export
        pass


# ── 标签映射导入（分析主题绑定）────────────────────────────────
# 支持两种 md 格式：
#   1) 映射表：'### 1. [场景]' + 三列表格 '| **标准标签** | 别名 | 说明 |'
#   2) 标签库：'## 场景' + '- 标签' 列表（无别名，标准标签本身即关键词）
_TABLE_ROW_RE = re.compile(r'\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]*?)\s*\|\s*([^|]*)\s*\|')
_SECTION_SPLIT_RE = re.compile(r'\n###\s+')
_CATEGORY_RE = re.compile(r'\d+\.\s*\[([^\]]+)\]')
_HEADING_RE = re.compile(r'^##+\s+(.+)$')


def _upsert_dict_entry(session: Session, standard_tag: str, category: str,
                       aliases: List[str], theme_id: Optional[int]) -> None:
    """幂等写入一条字典项。

    注意：dictionary.standard_tag 是主键（全局唯一），同一标签只能存一条。
    因此若一个标签被多个分析主题引用，就把它视为「通用标签」（theme_id=NULL）——
    所有主题都能看到，避免切换主题后标签库残缺。
    """
    existing = session.query(DictionaryEntry).filter(
        DictionaryEntry.standard_tag == standard_tag
    ).first()
    aliases_str = ",".join(aliases)
    if existing:
        if aliases_str:
            existing.aliases = aliases_str
        existing.category = category
        if theme_id is not None:
            if existing.theme_id is None:
                pass                          # 已是通用标签，保持
            elif existing.theme_id != theme_id:
                existing.theme_id = None      # 被多个主题引用 → 降级为通用
            # 同主题重复导入：保持原归属
    else:
        session.add(DictionaryEntry(
            standard_tag=standard_tag,
            category=category,
            aliases=aliases_str,
            file_count=0,
            theme_id=theme_id,
        ))


def count_mapping_md_tags(md_path: str,
                          skip_categories: Tuple[str, ...] = ('一致性', '生成质量')) -> int:
    """只解析、不写库：统计某映射表能产出多少标签。

    用于"每主题一个库"下展示**非当前主题**的标签数（其库未打开时无法直接查）。
    """
    path = Path(md_path)
    if not path.exists():
        return 0
    try:
        content = path.read_text(encoding='utf-8')
    except Exception:
        return 0

    count = 0
    for section in _SECTION_SPLIT_RE.split(content)[1:]:
        category_match = _CATEGORY_RE.match(section)
        if not category_match:
            continue
        if category_match.group(1).strip() in skip_categories:
            continue
        count += sum(1 for tag, _a, _d in _TABLE_ROW_RE.findall(section) if tag.strip())
    if count:
        return count

    current_category: Optional[str] = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            name = heading.group(1).strip()
            current_category = None if name in skip_categories else name
            continue
        if current_category and line.startswith('- ') and line[2:].strip():
            count += 1
    return count


def import_mapping_md(session: Session, md_path: str,
                      theme_id: Optional[int] = None,
                      skip_categories: Tuple[str, ...] = ('一致性', '生成质量')) -> int:
    """从标签映射 md 导入标签到 dictionary，写入 theme_id 实现「标签按分析主题归属」。

    先按映射表格式解析，无结果时回退标签库格式。幂等（按标准标签 upsert）。

    Args:
        session: SQLAlchemy Session
        md_path: 映射表 / 标签库 md 路径
        theme_id: 归属的分析主题 id；None 表示不归属主题（通用）
        skip_categories: 跳过的分类（AI 生成质控等，不属于内容标签）

    Returns:
        处理的标签条数（含更新）
    """
    path = Path(md_path)
    if not path.exists():
        logger.warning("标签映射文件不存在，跳过: %s", md_path)
        return 0

    try:
        content = path.read_text(encoding='utf-8')
    except Exception as exc:
        logger.warning("读取标签映射文件失败 %s: %s", md_path, exc)
        return 0

    count = 0

    # 格式1：映射表（### N. [分类] + 三列表格）
    for section in _SECTION_SPLIT_RE.split(content)[1:]:
        category_match = _CATEGORY_RE.match(section)
        if not category_match:
            continue
        category = category_match.group(1).strip()
        if category in skip_categories:
            continue
        for standard_tag, aliases_str, _desc in _TABLE_ROW_RE.findall(section):
            standard_tag = standard_tag.strip()
            if not standard_tag:
                continue
            aliases = [a.strip() for a in re.split(r'[,，]+', aliases_str) if a.strip()]
            _upsert_dict_entry(session, standard_tag, category, aliases, theme_id)
            count += 1

    if count:
        session.commit()
        logger.info("导入标签映射(表格格式) %s: %d 条, theme_id=%s", md_path, count, theme_id)
        return count

    # 格式2：标签库（## 分类 + '- 标签' 列表）
    current_category: Optional[str] = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            name = heading.group(1).strip()
            current_category = None if name in skip_categories else name
            continue
        if current_category and line.startswith('- '):
            standard_tag = line[2:].strip()
            if standard_tag:
                _upsert_dict_entry(session, standard_tag, current_category, [], theme_id)
                count += 1

    if count:
        session.commit()
        logger.info("导入标签库(列表格式) %s: %d 条, theme_id=%s", md_path, count, theme_id)
    else:
        logger.warning("未能从 %s 解析出任何标签（格式不支持？）", md_path)
    return count
