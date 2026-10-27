"""按文件名厂牌自动分组（影楼 S4.5）—— 纯逻辑层，可单测。

产品语义：解析文件名中的厂牌（出品方）→ 分组 → **预览确认后**才建项目归集。
识别不了的（相机文件 IMG_、录屏 Video/VID、UUID/哈希、纯时间戳）保持未归集。

规则（2026-09-30 按 31395 个真实文件名分布定稿）：
1. 排除设备/工具前缀（IMG/VID/Video/录屏/RPReplay…）与 UUID/纯哈希/纯数字开头；
2. 首段切分优先级：" - " > "-" > "_" > "."（MEN.-.Title → MEN）；
3. 首段去年份前缀（2019_Drill → Drill），纯数字/空段不可判；
4. 规范键 = 小写去符号（Bel Ami Online ≡ BelAmiOnline），相同键归同组；
5. 前缀聚合：规范键长度 ≥5 且互为前缀 → 并入高频组（belami 吞 belamionline；
   阈值防误吞：men(3) 不吞 menatplay(9)）；
6. 组名取组内出现最多的原始写法（保留用户熟悉的拼写）。

分组只产出**建议**，写库（建客户/项目+清单挂接）由 UI 确认后走 studio API。
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

__all__ = ["parse_brand", "group_by_brand", "TOOL_PREFIXES"]

# 设备/工具/录屏类前缀（规范键）：不是厂牌，永远不建组
TOOL_PREFIXES = {
    "img", "vid", "video", "svid", "videorecorder",
    "screenrecorder", "screenrecording", "screen",
    "rpreplay", "mmexport", "chat", "wxexport", "qqexport",
    "录屏", "一同看", "微信图片", "screenshot",
    # 下载/转发工具名（来源不是厂牌）
    "tweeload", "savetwitter", "telegram",
}

_UUID_LIKE = re.compile(r"^[0-9a-fA-F]{8}[-_0-9a-fA-F]{12,}$")
_HEXISH = re.compile(r"^[0-9a-fA-F]{12,}$")
_YEAR_PREFIX = re.compile(r"^\d{4}[-_ ]?")
_MIN_KEY_LEN_TO_PREFIX_MERGE = 5


def _norm_key(brand: str) -> str:
    """规范键：小写 + 去全部符号/空格（中文保留）。Bel Ami Online ≡ BelAmiOnline。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", brand.lower())


def parse_brand(file_name: str) -> Optional[str]:
    """从文件名解析厂牌原始写法；识别不了返回 None。"""
    stem = Path(file_name).stem.strip()
    if not stem or _UUID_LIKE.match(stem) or _HEXISH.match(stem):
        return None

    # 分隔符优先级：" - "（最常见厂牌分隔）> "-" > "_" > "."
    first = None
    for sep in (" - ", "-", "_", "."):
        if sep in stem:
            first = stem.split(sep, 1)[0].strip()
            break
    if not first:
        return None

    first = _YEAR_PREFIX.sub("", first).strip()
    if not first or first[0].isdigit():
        return None
    key = _norm_key(first)
    # 工具/设备前缀按前缀匹配（"录屏2026" 这类带尾随数字的同样排除）
    if not key or any(key == t or key.startswith(t) for t in TOOL_PREFIXES):
        return None
    return first


def group_by_brand(items: list) -> dict:
    """按厂牌分组。items 为 /videos 字典列表（需 file_name / task_id）。

    返回 {组名: {"items": [...], "count": n}}（按计数降序）+
    "unclassified": {"items": [...], "count": n}（不含在组字典顶部，见 keys）。
    组名已做规范键合并与前缀聚合；单例组（仅 1 个文件）单独归入 "singletons"
    便于预览默认不建组。"""
    buckets = defaultdict(list)          # norm_key → [item]
    orig_names = defaultdict(Counter)    # norm_key → {原始写法: 计数}（组名取众数）

    for item in items:
        brand = parse_brand(item.get("file_name") or "")
        if not brand:
            continue
        key = _norm_key(brand)
        buckets[key].append(item)
        orig_names[key][brand] += 1

    groups = {}
    for key, items_ in buckets.items():
        groups[key] = {"items": items_, "count": len(items_)}

    # 前缀聚合：互为前缀且短键 ≥5 字符 → 低频组并入**高频**组（保留大组组名；
    # 短键阈值防误吞：men(3) 不得吞 menatplay(9)）
    keys_by_count = sorted(groups, key=lambda k: -groups[k]["count"])
    for k_main in keys_by_count:
        if k_main not in groups:
            continue
        for k_other in list(groups):
            if k_other == k_main or k_other not in groups:
                continue
            _shorter = k_main if len(k_main) <= len(k_other) else k_other
            if len(_shorter) < _MIN_KEY_LEN_TO_PREFIX_MERGE:
                continue
            if k_other.startswith(k_main) or k_main.startswith(k_other):
                groups[k_main]["items"].extend(groups.pop(k_other)["items"])
                orig_names[k_main].update(orig_names.pop(k_other))

    # 组名：先排除"是其他写法前缀"的截断名（Bel Am ⊂ Bel Ami），
    # 再取剩余写法中最高频者，去首尾装饰符；单例拆出
    result = {}
    singletons = {"items": [], "count": 0}
    for key, grp in groups.items():
        grp["count"] = len(grp["items"])
        orig_counter = orig_names.get(key)
        if orig_counter:
            spellings = list(orig_counter)
            _non_trunc = [s for s in spellings
                          if not any(o != s and o.startswith(s)
                                     for o in spellings)]
            _pool = _non_trunc or spellings
            group_name = max(_pool, key=lambda s: orig_counter[s])
        else:
            group_name = key
        group_name = group_name.strip().strip(".-_ ").strip() or key
        grp["name"] = group_name
        if grp["count"] <= 1:
            singletons["items"].extend(grp["items"])
            singletons["count"] += grp["count"]
        else:
            result[group_name] = grp

    result = dict(sorted(result.items(),
                         key=lambda kv: -kv[1]["count"]))
    result["__singletons__"] = singletons
    return result
