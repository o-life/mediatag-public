"""主题注册表（`<base>/themes/registry.json`）。

存在理由：`themes`/`theme_prompts` 原本在数据库里，而"有哪些主题、各自的库与
config 在哪"必须**先于**打开某个数据文件可知——每主题一个库之后这就成了自举问题。
"""

import json

import pytest

from src.core.theme_registry import ThemeRegistry


@pytest.fixture
def reg(tmp_path):
    return ThemeRegistry(tmp_path / "themes" / "registry.json")


@pytest.fixture
def session(tmp_path):
    """自建会话（不用 conftest 的 db_session：它对 :memory: 会因池参数报错）。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'reg.db'}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


PRESETS = [
    {"name": "日常", "default": True, "mapping": "日常生活标签映射表.md"},
    {"name": "成人内容", "mapping": "专业成人内容标签映射表.md"},
    {"name": "男男", "mapping": "男男标签库.md", "prompt": "男男提示词"},
]


class TestBasics:
    def test_empty_when_missing(self, reg):
        assert reg.list_themes() == []
        assert reg.current() == ""

    def test_upsert_and_get(self, reg):
        reg.upsert("日常", is_default=True, description="默认")
        assert reg.names() == ["日常"]
        assert reg.get("日常")["description"] == "默认"
        assert reg.current() == "日常"          # 首个主题自动成为当前

    def test_upsert_updates_in_place(self, reg):
        reg.upsert("成人内容", label_mapping_path="/a.md")
        reg.upsert("成人内容", label_mapping_path="/b.md")
        assert len(reg.list_themes()) == 1
        assert reg.get("成人内容")["label_mapping_path"] == "/b.md"

    def test_only_one_default(self, reg):
        reg.upsert("a", is_default=True)
        reg.upsert("b", is_default=True)
        defaults = [t["name"] for t in reg.list_themes() if t.get("is_default")]
        assert defaults == ["b"]

    def test_blank_name_rejected(self, reg):
        with pytest.raises(ValueError):
            reg.upsert("   ")


class TestCurrent:
    def test_set_current_requires_existing(self, reg):
        with pytest.raises(ValueError):
            reg.set_current("不存在")

    def test_effective_current_falls_back_to_default(self, reg):
        reg.upsert("a", is_default=True)
        reg.upsert("b")
        reg.save({**reg.load(), "current": ""})
        assert reg.effective_current() == "a"

    def test_effective_current_ignores_dangling_pointer(self, reg):
        reg.upsert("a", is_default=True)
        reg.save({**reg.load(), "current": "已删除的"})
        assert reg.effective_current() == "a"


class TestDelete:
    def test_delete_non_default(self, reg):
        reg.upsert("a", is_default=True)
        reg.upsert("b")
        reg.delete("b")
        assert reg.names() == ["a"]

    def test_default_cannot_be_deleted(self, reg):
        reg.upsert("a", is_default=True)
        with pytest.raises(ValueError):
            reg.delete("a")

    def test_deleting_current_moves_pointer(self, reg):
        reg.upsert("a", is_default=True)
        reg.upsert("b")
        reg.set_current("b")
        reg.delete("b")
        assert reg.current() == "a"


class TestSeed:
    def test_seeds_all(self, reg):
        assert reg.seed(PRESETS) == 3
        assert reg.names() == ["日常", "成人内容", "男男"]
        assert reg.get("日常")["is_default"] is True
        assert reg.get("男男")["prompt"] == "男男提示词"
        assert reg.current() == "日常"

    def test_idempotent(self, reg):
        reg.seed(PRESETS)
        assert reg.seed(PRESETS) == 0
        assert len(reg.list_themes()) == 3

    def test_does_not_override_user_edits(self, reg):
        reg.seed(PRESETS)
        reg.upsert("日常", description="用户改过的")
        reg.seed(PRESETS)
        assert reg.get("日常")["description"] == "用户改过的"

    def test_does_not_steal_existing_default(self, reg):
        reg.upsert("我的默认", is_default=True)
        reg.seed(PRESETS)
        defaults = [t["name"] for t in reg.list_themes() if t.get("is_default")]
        assert defaults == ["我的默认"]


class TestImportFromDb:
    def test_exports_themes_and_prompts(self, reg, session):
        from src.db.models import Theme, ThemePrompt

        t = Theme(name="成人内容", is_default=True, label_mapping_path="/x.md",
                  description="成人")
        session.add(t)
        session.flush()
        session.add(ThemePrompt(theme_id=t.id, prompt="提示词内容"))
        session.commit()

        assert reg.import_from_db(session) == 1
        rec = reg.get("成人内容")
        assert rec["is_default"] is True
        assert rec["label_mapping_path"] == "/x.md"
        assert rec["prompt"] == "提示词内容"
        assert reg.current() == "成人内容"

    def test_does_not_override_registry(self, reg, session):
        from src.db.models import Theme

        reg.upsert("成人内容", description="注册表里的")
        session.add(Theme(name="成人内容", description="库里的"))
        session.commit()

        assert reg.import_from_db(session) == 0
        assert reg.get("成人内容")["description"] == "注册表里的"

    def test_idempotent(self, reg, session):
        from src.db.models import Theme

        session.add(Theme(name="日常", is_default=True))
        session.commit()
        assert reg.import_from_db(session) == 1
        assert reg.import_from_db(session) == 0

    def test_marks_exported_themes_as_legacy_db(self, reg, session):
        """迁移而来的主题带 `legacy_db` 标记：Phase 1/3 阶段仍指向既有单库。"""
        from src.db.models import Theme

        session.add(Theme(name="日常", is_default=True))
        session.commit()
        reg.import_from_db(session)
        assert reg.get("日常")["legacy_db"] is True

    def test_current_prefers_legacy_active_theme(self, reg, session, tmp_path, monkeypatch):
        """导出时优先沿用旧配置的 `analysis.active_theme`。

        否则"用户正在用「男男」"会在迁移瞬间被悄悄改成默认主题（行为变更）。
        """
        import src.core.theme_workspace as wsmod
        from src.db.models import Theme

        ws = wsmod.ThemeWorkspace(base_dir=tmp_path)
        monkeypatch.setattr(ws, "legacy_config", lambda: {"analysis": {"active_theme": "男男"}})
        monkeypatch.setattr(wsmod, "ThemeWorkspace", lambda *a, **k: ws)

        session.add(Theme(name="日常", is_default=True))
        session.add(Theme(name="男男"))
        session.commit()
        reg.import_from_db(session)
        assert reg.current() == "男男"

    def test_current_falls_back_to_default_when_legacy_unknown(self, reg, session, tmp_path, monkeypatch):
        import src.core.theme_workspace as wsmod
        from src.db.models import Theme

        ws = wsmod.ThemeWorkspace(base_dir=tmp_path)
        monkeypatch.setattr(ws, "legacy_config",
                            lambda: {"analysis": {"active_theme": "已不存在的主题"}})
        monkeypatch.setattr(wsmod, "ThemeWorkspace", lambda *a, **k: ws)

        session.add(Theme(name="日常", is_default=True))
        session.commit()
        reg.import_from_db(session)
        assert reg.current() == "日常"


class TestFileSafety:
    def test_write_is_atomic_no_tmp_left(self, reg):
        reg.upsert("a")
        assert reg.path.exists()
        assert not reg.path.with_name(reg.path.name + ".tmp").exists()

    def test_unicode_preserved(self, reg):
        reg.upsert("日常")
        raw = reg.path.read_text(encoding="utf-8")
        assert "日常" in raw
        assert json.loads(raw)["themes"][0]["name"] == "日常"

    def test_corrupt_file_is_tolerated(self, reg):
        reg.path.parent.mkdir(parents=True, exist_ok=True)
        reg.path.write_text("{ 这不是合法 JSON", encoding="utf-8")
        assert reg.list_themes() == []          # 不抛错，按空处理
        reg.upsert("a")                          # 且仍可写入
        assert reg.names() == ["a"]
