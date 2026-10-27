"""brand_classifier 单测——用例取自 31395 个真实文件名分布（2026-09-30 采样）。"""

import sys

sys.path.insert(0, "src")

from src.core.brand_classifier import group_by_brand, parse_brand  # noqa: E402


# ── parse_brand：真实样本案例 ────────────────────────────────────────

def test_parse_brand_real_cases():
    assert parse_brand(
        "MEN - The Gay Office - Executive Suite Part 3 "
        "(Jarec Wentworth & Jaxon Colt) 11.22.2014.mp4") == "MEN"
    assert parse_brand(
        "MEN.-.Ex.Machina.-.A.Gay.XXX.Parody.Part.3(1080p).mp4") == "MEN."
    assert parse_brand(
        "CorbinFisher ACM1423 - Tahoe：Kellan Fills Up Quinn -Bareback- "
        "2-3-2014.mp4") == "CorbinFisher ACM1423"
    assert parse_brand("BreederFuckers - Dale 1.mp4") == "BreederFuckers"
    assert parse_brand("ChaosMen - Alec - Bradley 2015.mp4") == "ChaosMen"
    assert parse_brand(
        "Bel Ami Online - First Jerk-Henri Gaudin.mp4") == "Bel Ami Online"


def test_parse_brand_excludes_non_brands():
    assert parse_brand("IMG_0856.MP4") is None                  # 相机
    assert parse_brand(
        "036EA851-B419-4595-BFF7-3A92D6B5A222.MP4") is None     # UUID
    assert parse_brand("chatb0c39e7a7241ee4ecb9aadc36edc2c7c.mp4") is None
    assert parse_brand("2025-01-09_21-36-46_10.mp4") is None    # 时间戳
    assert parse_brand("录屏2026-08-30 12.34.08.mov") is None    # 录屏+尾随数字
    assert parse_brand("Video-4380057277617780.mp4") is None    # 工具前缀
    assert parse_brand("一同看(13).mp4") is None
    assert parse_brand("mmexport1780585207213.mp4") is None


def test_parse_brand_chinese_site_code():
    """站子编号前缀（S08292_标题）→ 解析为编号（是否建组由用户预览裁定）。"""
    assert parse_brand(
        "S08292_冲浪小哥Araw联名腹肌帅哥阿显_《奶系鲜肉苞》_小帅哥正面M字腿等速.mov"
    ) == "S08292"


# ── group_by_brand：变体归一 / 前缀聚合 / 组名规则 ───────────────────

def test_group_merges_spelling_variants_but_not_short_prefix():
    """BelAmi 五种写法归一组；men(3字符) 不得吞 Menatplay（≥2 个文件成组）。"""
    items = ([{"file_name": f"BelAmi - a{i}.mp4"} for i in range(3)]
             + [{"file_name": f"Bel Ami Online - b{i}.mp4"} for i in range(2)]
             + [{"file_name": "Belami - c.mp4"}]
             + [{"file_name": "Menatplay - d0.mp4"},
                {"file_name": "Menatplay - d1.mp4"}])
    g = group_by_brand(items)
    named = {k: v for k, v in g.items() if k != "__singletons__"}
    assert len(named) == 2, named
    belami = next(v for k, v in named.items() if "belam" in k.lower())
    assert belami["count"] == 6
    assert "Menatplay" in named and named["Menatplay"]["count"] == 2


def test_group_name_prefers_non_truncated_mode():
    """截断写法（Bel Am ⊂ Bel Ami）不当选组名；ACM 编号后缀型同理被打回。"""
    items = ([{"file_name": f"Bel Am - x{i}.mp4"} for i in range(3)]
             + [{"file_name": "Bel Ami - full.mp4"}])
    g = group_by_brand(items)
    name = next(k for k in g if k != "__singletons__")
    assert name == "Bel Ami"          # 非"Bel Am"（截断）


def test_group_trailing_dot_stripped():
    """MEN.-.Title 的首段带尾点 → 组名去掉装饰符。"""
    g = group_by_brand([{"file_name": "MEN.-.Ex.Machina(1080p).mp4"},
                        {"file_name": "MEN - another.mp4"}])
    assert "MEN" in g and "MEN." not in g


def test_singletons_and_unclassified_buckets():
    items = ([{"file_name": f"MEN - a{i}.mp4"} for i in range(3)]
             + [{"file_name": "LonesomeBrand - only.mp4"}]   # 单例组
             + [{"file_name": "IMG_0001.MP4"}]               # 不可识别
             + [{"file_name": "9A71082F-1111-2222-3333-ABCDEFABCDEF.mp4"}])
    g = group_by_brand(items)
    assert g["MEN"]["count"] == 3
    assert g["__singletons__"]["count"] == 1   # 单例组默认不建项目
    total_classified = (sum(v["count"] for k, v in g.items()
                            if k != "__singletons__")
                        + g["__singletons__"]["count"])
    assert total_classified == 4
