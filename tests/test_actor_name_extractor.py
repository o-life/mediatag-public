"""ActorNameExtractor：忽略词改为可配置后的行为。

背景：原先类里硬编码了一张成人厂牌排除表；现改为读取 config 的
``filename_parsing.ignore_keywords``，默认值只含中性的平台/压制/设备/制作方词。
"""
from src.core.actor_name_extractor import _DEFAULT_IGNORE_KEYWORDS, ActorNameExtractor

# 曾在硬编码表里出现、现不应出现在任何默认值中的词
_LEGACY_STUDIO_TERMS = (
    "brazzers", "moodyz", "heyzo", "tokyohot", "caribbeancom", "prestige",
    "1pondo", "heydouga", "pacopaco", "naturalhigh", "attackers", "madonna",
    "bang bros", "team skeet", "mofos", "naughty america", "reality kings",
)


class TestDefaultsAreNeutral:
    def test_defaults_contain_no_legacy_studio_terms(self):
        joined = " ".join(_DEFAULT_IGNORE_KEYWORDS).lower()
        for term in _LEGACY_STUDIO_TERMS:
            assert term not in joined, f"默认忽略词不应包含 {term!r}"

    def test_defaults_keep_common_neutral_markers(self):
        joined = " ".join(_DEFAULT_IGNORE_KEYWORDS).lower()
        for term in ("bilibili", "1080p", "iphone", "studio"):
            assert term in joined


class TestIgnoreKeywordsConfigurable:
    def test_default_instance_drops_platform_and_quality_terms(self):
        ex = ActorNameExtractor()
        names = ex.extract_from_filename("John Smith bilibili 1080p.mp4")
        lowered = [n.lower() for n in names]
        assert not any("bilibili" in n for n in lowered)
        assert not any("1080p" in n for n in lowered)

    def test_user_supplied_keyword_is_dropped(self):
        ex = ActorNameExtractor(ignore_keywords=["xyzcorp"])
        names = ex.extract_from_filename("Xyzcorp John Smith.mp4")
        assert not any("xyzcorp" in n.lower() for n in names)
        assert "John" in names

    def test_empty_list_disables_filtering(self):
        ex = ActorNameExtractor(ignore_keywords=[])
        names = ex.extract_from_filename("Studio John.mp4")
        assert "Studio" in names

    def test_blank_entries_do_not_match_everything(self):
        ex = ActorNameExtractor(ignore_keywords=["", "   "])
        names = ex.extract_from_filename("John Smith.mp4")
        assert "John" in names


class TestExtractionStillWorks:
    def test_plain_name_extracted(self):
        ex = ActorNameExtractor(ignore_keywords=[])
        names = ex.extract_from_filename("Ryo Yamashta - Something.mp4")
        assert "Ryo" in names

    def test_dates_and_quality_markers_stripped(self):
        ex = ActorNameExtractor(ignore_keywords=[])
        names = ex.extract_from_filename("2019_John Smith_1080p.mp4")
        assert "John" in names
        assert not any(n.isdigit() for n in names)
