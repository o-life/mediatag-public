"""主题工作区：目录布局、配置生成、懒创建、以及对解析层的接管。

Phase 1 的定位是**布局与解析**且要求**行为等价**：
- 迁移而来的既有主题（`legacy_db`）→ config 仍指向既有单库/人脸根目录；
- 新建主题 → 直接用档内 `<主题>/data.db` 与 `<主题>/faces/`。
真正把既有主题切到档内路径由 Phase 5 的拆分迁移完成。
"""

from pathlib import Path

import pytest
import yaml

from src.core.theme_registry import ThemeRegistry
from src.core.theme_workspace import ThemeWorkspace, bootstrap_current_theme

LEGACY = {
    "database": {"path": "~/Library/Application Support/MediaTag/db/audit_cache.db"},
    "face_recognition": {"faces_dir": "/Volumes/MediaData/actor_faces/"},
    "vlm": {"provider": "url"},
}


@pytest.fixture
def reg(tmp_path, monkeypatch):
    r = ThemeRegistry(tmp_path / "themes" / "registry.json")
    import src.core.theme_registry as trmod

    monkeypatch.setattr(trmod, "get_theme_registry", lambda: r)
    return r


@pytest.fixture
def ws(tmp_path, reg):
    return ThemeWorkspace(base_dir=tmp_path)


class TestLayout:
    def test_paths(self, ws, tmp_path):
        assert ws.themes_root == tmp_path / "themes"
        assert ws.theme_dir("日常") == tmp_path / "themes" / "日常"
        assert ws.config_path("日常").name == "config.yaml"
        assert ws.db_path("日常").name == "data.db"
        assert ws.faces_dir("日常").name == "faces"

    def test_exists_only_after_creation(self, ws):
        assert ws.exists("日常") is False
        ws.ensure_theme("日常")
        assert ws.exists("日常") is True


class TestEnsureTheme:
    def test_creates_dir_config_and_faces(self, ws):
        tdir = ws.ensure_theme("新建的")
        assert (tdir / "config.yaml").is_file()
        assert (tdir / "faces").is_dir()
        cfg = yaml.safe_load((tdir / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["theme"]["name"] == "新建的"

    def test_does_not_overwrite_existing_config(self, ws):
        tdir = ws.ensure_theme("日常")
        (tdir / "config.yaml").write_text("database:\n  path: /custom.db\n", encoding="utf-8")
        ws.ensure_theme("日常")
        cfg = yaml.safe_load((tdir / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["database"]["path"] == "/custom.db"

    def test_blank_name_rejected(self, ws):
        with pytest.raises(ValueError):
            ws.ensure_theme("")


class TestBuildConfig:
    def test_new_theme_uses_local_paths(self, ws, reg, monkeypatch):
        monkeypatch.setattr(ws, "legacy_config", lambda: dict(LEGACY))
        cfg = ws.build_config("全新的")          # 注册表里没有 → 非 legacy
        assert cfg["database"]["path"] == str(ws.db_path("全新的"))
        assert cfg["face_recognition"]["faces_dir"] == str(ws.faces_dir("全新的")) + "/"
        assert cfg["theme"]["inherits_legacy_content"] is False
        assert cfg["analysis"]["active_theme"] == "全新的"
        # 覆盖配置：**只写与全局不同的键**（VLM/抽帧数等由全局基底提供，不再整份抄进来）
        assert "vlm" not in cfg and "video" not in cfg and "nfo" not in cfg
        assert set(cfg) == {"theme", "database", "face_recognition", "analysis"}

    def test_legacy_theme_keeps_existing_paths(self, ws, reg, monkeypatch):
        """行为等价的关键：迁移而来的主题暂时仍指向既有单库与人脸根目录。"""
        reg.upsert("成人内容", legacy_db=True)
        monkeypatch.setattr(ws, "legacy_config", lambda: dict(LEGACY))
        cfg = ws.build_config("成人内容")
        assert cfg["database"]["path"] == str(Path(LEGACY["database"]["path"]).expanduser())
        assert cfg["face_recognition"]["faces_dir"] == LEGACY["face_recognition"]["faces_dir"]
        assert cfg["theme"]["inherits_legacy_content"] is True


class TestEffectivePaths:
    def test_reads_from_theme_config(self, ws, reg, monkeypatch):
        monkeypatch.setattr(ws, "legacy_config", lambda: dict(LEGACY))
        ws.ensure_theme("全的")
        assert ws.effective_db_path("全的") == ws.db_path("全的")
        assert ws.effective_faces_dir("全的") == ws.faces_dir("全的")

    def test_missing_config_falls_back_to_local(self, ws):
        assert ws.effective_db_path("不存在") == ws.db_path("不存在")

    def test_relative_faces_dir_resolves_against_theme_dir(self, ws, tmp_path):
        tdir = ws.ensure_theme("相对")
        (tdir / "config.yaml").write_text(
            "face_recognition:\n  faces_dir: faces/\n", encoding="utf-8")
        assert ws.effective_faces_dir("相对") == ws.faces_dir("相对")


class TestCurrentTheme:
    def test_current_name_follows_registry(self, ws, reg):
        reg.upsert("日常", is_default=True)
        reg.upsert("成人内容")
        reg.set_current("成人内容")
        assert ws.current_name() == "成人内容"

    def test_current_config_none_when_not_bootstrapped(self, ws, reg):
        reg.upsert("日常", is_default=True)          # 有主题但没建档
        assert ws.current_config_path() is None
        ws.ensure_theme("日常")
        assert ws.current_config_path() == ws.config_path("日常")


class TestBootstrap:
    def test_no_registry_is_noop(self, ws, reg, monkeypatch):
        monkeypatch.delenv("VIDEO_TAG_THEME_AUTOCREATE", raising=False)
        import src.core.theme_workspace as wsmod
        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        assert bootstrap_current_theme() is None      # 无主题 → 回退旧布局
        assert not ws.themes_root.exists()

    def test_creates_current_theme(self, ws, reg, monkeypatch):
        monkeypatch.delenv("VIDEO_TAG_THEME_AUTOCREATE", raising=False)
        import src.core.theme_workspace as wsmod
        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        reg.upsert("日常", is_default=True)
        assert bootstrap_current_theme() == "日常"
        assert ws.config_path("日常").is_file()

    def test_disabled_by_env(self, ws, reg, monkeypatch):
        monkeypatch.setenv("VIDEO_TAG_THEME_AUTOCREATE", "0")
        import src.core.theme_workspace as wsmod
        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        reg.upsert("日常", is_default=True)
        assert bootstrap_current_theme() is None
        assert not ws.config_path("日常").exists()


class TestPathManagerIntegration:
    def test_config_and_db_follow_current_theme(self, ws, reg, tmp_path, monkeypatch):
        from src.core import path_manager as pm
        import src.core.theme_workspace as wsmod

        monkeypatch.setattr(ws, "legacy_config", lambda: dict(LEGACY))
        reg.upsert("全的")
        ws.ensure_theme("全的")
        reg.set_current("全的")
        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)

        m = pm.PathManager()
        monkeypatch.setattr(m, "_base_dir", tmp_path)
        assert m.config_file == ws.config_path("全的")
        assert m.db_file == ws.db_path("全的")

    def test_falls_back_to_legacy_without_theme(self, ws, reg, tmp_path, monkeypatch):
        from src.core import path_manager as pm
        import src.core.theme_workspace as wsmod

        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        m = pm.PathManager()
        monkeypatch.setattr(m, "_base_dir", tmp_path)
        assert m.config_file == tmp_path / "config.yaml"


class TestFacePathsFollowTheme:
    def test_relative_resolves_against_theme_dir(self, ws, reg, monkeypatch):
        import src.core.theme_workspace as wsmod
        from src.core.face_paths import resolve_faces_dir

        monkeypatch.setattr(ws, "legacy_config", lambda: dict(LEGACY))
        reg.upsert("全的")
        ws.ensure_theme("全的")
        reg.set_current("全的")
        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)

        got = resolve_faces_dir("faces/")
        assert got == ws.faces_dir("全的").resolve()

    def test_absolute_ignores_theme(self, ws, reg, monkeypatch):
        import src.core.theme_workspace as wsmod
        from src.core.face_paths import resolve_faces_dir

        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        assert resolve_faces_dir("/tmp/faces_root") == Path("/tmp/faces_root").resolve()

    def test_no_theme_falls_back_to_project_root(self, ws, monkeypatch):
        import src.core.theme_workspace as wsmod
        from src.core.face_paths import project_root, resolve_faces_dir

        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: ws)
        assert resolve_faces_dir("data/actor_faces") == (project_root() / "data/actor_faces").resolve()
