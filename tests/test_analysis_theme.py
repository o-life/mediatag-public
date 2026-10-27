"""分析主题（业务主题）：预置播种 / 标签映射导入 / 端点。

元数据来源已改为**主题注册表**（`<base>/themes/registry.json`，数据文件之外）——
因为"有哪些主题、各自的库与 config 在哪"必须先于打开某个数据文件可知。
过渡期 `themes` 表仍按**主题名**保留一行，用于 `dictionary.theme_id` 的标签归属统计。

术语：「分析主题」= 业务主题（标签库 + 提示词 + 人脸资源库），区别于「外观」（深/浅色皮肤）。
Qt 交互测试见 tests/test_tag_edit_dialog.py（`VT_UI_TESTS=1`）。
"""

import asyncio
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.api.routes.themes as T
from src.core.dictionary import import_mapping_md
from src.core.preset_themes import PRESET_THEMES, build_preset_records, presets_dir, seed_preset_themes
from src.core.theme_registry import ThemeRegistry
from src.db.models import Base, DictionaryEntry, Theme

# ── 素材 ────────────────────────────────────────────────────────────

TABLE_MD = """# 测试映射表

### 1. [场景]
| 标准标签 | 同义词 | 说明 |
| --- | --- | --- |
| **室内** | Indoor, 屋内 | 室内环境 |
| **户外** | Outdoor | 室外环境 |

### 2. [一致性]
| 标准标签 | 同义词 | 说明 |
| --- | --- | --- |
| **服装跳变** | | 应被默认跳过 |
"""

LIST_MD = """# 标准标签库

## 场景
- 卧室
- 客厅

## 生成质量
- 形变
"""


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture
def registry(tmp_path):
    return ThemeRegistry(tmp_path / "themes" / "registry.json")


@pytest.fixture
def env(session, registry, tmp_path, monkeypatch):
    """把端点里的库会话、注册表与主题工作区都指到测试对象。"""
    engine = session.get_bind()
    monkeypatch.setattr(T, "get_db_engine", lambda *a, **k: engine)
    monkeypatch.setattr(T, "get_session", lambda *a, **k: sessionmaker(bind=engine)())
    monkeypatch.setattr(T, "get_theme_registry", lambda: registry)

    import src.core.theme_workspace as wsmod

    monkeypatch.setattr(wsmod, "get_theme_workspace",
                        lambda: wsmod.ThemeWorkspace(base_dir=tmp_path))
    return registry


def _mk(session, registry, name, mapping=None, is_default=False, prompt=None):
    """建一个主题：注册表持有元数据，DB 有一行桥接（承载旧 theme_id）。"""
    fields = {}
    if mapping:
        fields["label_mapping_path"] = mapping
    if prompt:
        fields["prompt"] = prompt
    registry.upsert(name, **fields)
    if is_default:
        registry.upsert(name, is_default=True)
    row = Theme(name=name)
    session.add(row)
    session.flush()
    session.commit()
    return row


# ── 预置主题播种（注册表） ──────────────────────────────────────────


class TestPresetSeeding:
    def test_shipped_presets_all_have_mapping_files(self):
        """随包分发的预置素材必须齐全——否则打包版开箱无预置主题。"""
        themes_dir = presets_dir()
        missing = [p["mapping"] for p in PRESET_THEMES
                   if not (themes_dir / p["mapping"]).exists()]
        assert not missing, f"resources/themes 缺少预置映射表: {missing}"

    def test_presets_dir_is_project_resources_in_dev(self):
        assert presets_dir().name == "themes"
        assert presets_dir().parent.name == "resources"

    def test_records_only_include_existing_mappings(self, registry, tmp_path):
        assert build_preset_records(tmp_path) == []          # 空目录 → 无记录
        records = build_preset_records(presets_dir())
        assert [r["name"] for r in records] == [p["name"] for p in PRESET_THEMES]
        assert all(Path(r["label_mapping_path"]).exists() for r in records)

    def test_seed_creates_and_binds(self, registry):
        created = seed_preset_themes(registry, themes_dir=presets_dir())
        assert created == len(PRESET_THEMES)
        assert sorted(registry.names()) == sorted(p["name"] for p in PRESET_THEMES)
        assert all(r.get("label_mapping_path") for r in registry.list_themes())
        defaults = [r["name"] for r in registry.list_themes() if r.get("is_default")]
        assert len(defaults) == 1, "有且仅有一个默认分析主题"

    def test_seed_is_idempotent(self, registry):
        seed_preset_themes(registry, themes_dir=presets_dir())
        assert seed_preset_themes(registry, themes_dir=presets_dir()) == 0
        assert len(registry.names()) == len(PRESET_THEMES)

    def test_seed_does_not_override_user_edits(self, registry):
        seed_preset_themes(registry, themes_dir=presets_dir())
        registry.upsert(PRESET_THEMES[0]["name"], description="用户改过的说明")
        seed_preset_themes(registry, themes_dir=presets_dir())
        assert registry.get(PRESET_THEMES[0]["name"])["description"] == "用户改过的说明"

    def test_missing_mapping_file_is_skipped_not_fatal(self, registry, tmp_path):
        assert seed_preset_themes(registry, themes_dir=tmp_path) == 0
        assert registry.names() == []

    def test_existing_default_is_not_overridden(self, registry):
        registry.upsert("用户自设默认", is_default=True)
        seed_preset_themes(registry, themes_dir=presets_dir())
        defaults = [r["name"] for r in registry.list_themes() if r.get("is_default")]
        assert defaults == ["用户自设默认"]

    def test_prompt_text_is_loaded_from_file(self, registry, tmp_path, monkeypatch):
        """机制测试:预置主题带 prompt 文件名时,播种从文件读入提示词内容。

        公开版预置主题不含专属提示词文件(成人主题词典不入公开仓),
        故以临时目录 + monkeypatch 的自定义预置条目验证加载机制。
        """
        (tmp_path / "自定义映射.md").write_text("# 标准标签库", encoding="utf-8")
        (tmp_path / "自定义提示词.md").write_text("## 任务\n分析内容", encoding="utf-8")
        monkeypatch.setattr("src.core.preset_themes.PRESET_THEMES",
                            [{"name": "自定义", "mapping": "自定义映射.md",
                              "prompt": "自定义提示词.md"}])
        created = seed_preset_themes(registry, themes_dir=tmp_path)
        assert created == 1
        row = registry.get("自定义")
        assert row["prompt"] == "## 任务\n分析内容"


# ── 标签映射导入（仍在库内，按 theme_id 归属） ──────────────────────


class TestImportMappingMd:
    def test_table_format_writes_theme_id(self, session, tmp_path):
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        count = import_mapping_md(session, str(md), theme_id=7)
        assert count == 2
        rows = {d.standard_tag: d for d in session.query(DictionaryEntry).all()}
        assert rows["室内"].theme_id == 7
        assert rows["室内"].category == "场景"
        assert "Indoor" in rows["室内"].aliases

    def test_skip_categories(self, session, tmp_path):
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        import_mapping_md(session, str(md), theme_id=1)
        assert "服装跳变" not in [d.standard_tag for d in session.query(DictionaryEntry).all()]

    def test_skip_categories_is_configurable(self, session, tmp_path):
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        import_mapping_md(session, str(md), theme_id=1, skip_categories=())
        assert "服装跳变" in [d.standard_tag for d in session.query(DictionaryEntry).all()]

    def test_list_format_fallback(self, session, tmp_path):
        md = tmp_path / "lib.md"
        md.write_text(LIST_MD, encoding="utf-8")
        assert import_mapping_md(session, str(md), theme_id=3) == 2
        rows = {d.standard_tag: d for d in session.query(DictionaryEntry).all()}
        assert rows["卧室"].theme_id == 3
        assert "形变" not in rows

    def test_idempotent(self, session, tmp_path):
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        import_mapping_md(session, str(md), theme_id=1)
        n = session.query(DictionaryEntry).count()
        import_mapping_md(session, str(md), theme_id=1)
        assert session.query(DictionaryEntry).count() == n

    def test_multi_theme_tag_downgrades_to_common(self, session, tmp_path):
        a, b = tmp_path / "a.md", tmp_path / "b.md"
        a.write_text(TABLE_MD, encoding="utf-8")
        b.write_text(TABLE_MD, encoding="utf-8")
        import_mapping_md(session, str(a), theme_id=1)
        import_mapping_md(session, str(b), theme_id=2)
        row = session.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == "室内").one()
        assert row.theme_id is None

    def test_common_tag_stays_common(self, session, tmp_path):
        session.add(DictionaryEntry(standard_tag="室内", category="场景", theme_id=None))
        session.commit()
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        import_mapping_md(session, str(md), theme_id=5)
        row = session.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == "室内").one()
        assert row.theme_id is None

    def test_missing_file_returns_zero(self, session, tmp_path):
        assert import_mapping_md(session, str(tmp_path / "nope.md"), theme_id=1) == 0


# ── 端点 ────────────────────────────────────────────────────────────


class TestActivateEndpoint:
    def test_activate_switches_and_requires_restart(self, session, registry, env, tmp_path):
        """每主题一个数据文件后：激活=**冷切**（写 current + 预建目录），**不导标签**。

        标签随主题自己的库走（启动链路按当前主题导入）；把目标主题的映射导进
        当前库会污染当前主题——这正是这里要防的。
        """
        md = tmp_path / "t.md"
        md.write_text(TABLE_MD, encoding="utf-8")
        row = _mk(session, registry, "成人内容", mapping=str(md), prompt="p")

        resp = asyncio.run(T.activate_theme(row.id))
        assert resp["status"] == "success"
        assert registry.current() == "成人内容"
        assert resp["restart_required"] is True
        assert resp["imported_tags"] == 0
        assert resp["theme"]["has_prompt"] is True
        # 不得把目标主题的标签写进当前库
        assert session.query(DictionaryEntry).count() == 0

    def test_activate_without_mapping_still_switches(self, session, registry, env):
        row = _mk(session, registry, "空主题")
        resp = asyncio.run(T.activate_theme(row.id))
        assert registry.current() == "空主题"
        assert resp["imported_tags"] == 0

    def test_activate_unknown_theme_404(self, session, registry, env):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            asyncio.run(T.activate_theme(9999))
        assert exc.value.status_code == 404


class TestSeedPresetsEndpoint:
    def test_endpoint_seeds_registry_and_bridges_db(self, session, registry, env):
        resp = asyncio.run(T.seed_presets())
        assert resp["status"] == "success"
        assert resp["created"] == len(PRESET_THEMES)
        assert sorted(registry.names()) == sorted(p["name"] for p in PRESET_THEMES)
        # 桥接：DB 里有同名行（供旧 theme_id 归属）
        assert session.query(Theme).count() == len(PRESET_THEMES)


class TestDeleteEndpoint:
    def test_default_theme_cannot_be_deleted(self, session, registry, env):
        from fastapi import HTTPException

        row = _mk(session, registry, "默认", is_default=True)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(T.delete_theme(row.id))
        assert exc.value.status_code == 400
        assert registry.get("默认") is not None

    def test_delete_removes_from_registry_and_reassigns_tags(self, session, registry, env):
        row = _mk(session, registry, "临时")
        session.add(DictionaryEntry(standard_tag="室内", category="场景", theme_id=row.id))
        session.commit()

        asyncio.run(T.delete_theme(row.id))
        assert registry.get("临时") is None
        tag = session.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == "室内").one()
        assert tag.theme_id is None, "删除主题后标签应归入通用，而不是变成孤儿"


class TestListAndUpdate:
    def test_list_exposes_mapping_and_prompt(self, session, registry, env):
        _mk(session, registry, "带路径", mapping="/tmp/x.md", prompt="提示词")
        out = asyncio.run(T.list_themes())
        item = next(t for t in out["themes"] if t["name"] == "带路径")
        assert item["label_mapping_path"] == "/tmp/x.md"
        assert item["prompt"] == "提示词"
        assert "faces_dir" in item and "tag_count" in item

    def test_list_marks_current(self, session, registry, env):
        _mk(session, registry, "A")
        _mk(session, registry, "B")
        registry.set_current("B")
        out = asyncio.run(T.list_themes())
        assert out["current"] == "B"
        assert [t["name"] for t in out["themes"] if t["is_current"]] == ["B"]

    def test_update_sets_default_exclusively(self, session, registry, env):
        _mk(session, registry, "A", is_default=True)
        row_b = _mk(session, registry, "B")

        asyncio.run(T.update_theme(row_b.id, T.ThemeUpdateRequest(is_default=1)))
        defaults = [r["name"] for r in registry.list_themes() if r.get("is_default")]
        assert defaults == ["B"]

    def test_prompt_roundtrip(self, session, registry, env):
        row = _mk(session, registry, "P")
        asyncio.run(T.set_theme_prompt(row.id, T.ThemePromptRequest(prompt="新提示词")))
        got = asyncio.run(T.get_theme_prompt(row.id))
        assert got["prompt"] == "新提示词"


# ── 提示词与当前主题走注册表（不再读库） ────────────────────────────


class TestConfigPromptSource:
    def test_prompt_from_registry(self, registry, monkeypatch):
        from src.core import config as cfgmod

        registry.upsert("男男", prompt="男男提示词")
        monkeypatch.setattr(cfgmod, "get_theme_registry", lambda: registry, raising=False)
        import src.core.theme_registry as trmod
        monkeypatch.setattr(trmod, "get_theme_registry", lambda: registry)

        cfg = cfgmod.ConfigManager()
        assert cfg.get_vlm_prompt_for_theme("男男") == "男男提示词"
        assert cfg._theme_prompt_by_name("不存在") is None

    def test_active_theme_from_registry(self, registry, monkeypatch):
        import src.core.theme_registry as trmod
        from src.core import config as cfgmod

        registry.upsert("日常", is_default=True)
        registry.upsert("成人内容")
        registry.set_current("成人内容")
        monkeypatch.setattr(trmod, "get_theme_registry", lambda: registry)

        assert cfgmod.ConfigManager().get_active_analysis_theme() == "成人内容"


class TestSharedConfigManager:
    def test_accessor_is_singleton(self):
        from src.core.config import get_config_manager

        assert get_config_manager() is get_config_manager()

    def test_api_uses_shared_instance(self):
        import src.api.main as M
        from src.core.config import get_config_manager

        assert M.config_manager is get_config_manager(), \
            "main 的 config_manager 必须是共享实例，否则路由写入对它不可见"
