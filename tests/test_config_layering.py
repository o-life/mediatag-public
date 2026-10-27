"""配置分层：全局基底 ← 主题覆盖（合并读 + 按键路由写）。

背景：每主题一个数据文件后，主题 config 不该是整份拷贝——否则全局设置
（VLM 供应商、抽帧数、NFO 输出）改了不生效。分层后主题 config **只写与全局不同的键**。

原则（见 openspec/changes/per-theme-vault）：
**回退/工程开关全局**（`analysis.merge_vlm_passes`）；
**片源处理策略与主题身份随主题**（`analysis.mapping_*`、`theme.*`、`database.path`、`faces_dir`）。
"""

import yaml
import pytest

from src.core.config import ConfigManager, _deep_merge


@pytest.fixture
def layered(tmp_path, monkeypatch):
    """构造"全局 + 主题覆盖"双文件环境，并让 ConfigManager 走分层路径。"""
    base = tmp_path / "config.local.yaml"
    overlay = tmp_path / "themes" / "男男" / "config.yaml"
    overlay.parent.mkdir(parents=True)

    base.write_text(yaml.dump({
        "video": {"frames_per_scene": 5},
        "vlm": {"active": "本地"},
        "analysis": {"merge_vlm_passes": True},
        "database": {"path": "global.db"},
    }, allow_unicode=True), encoding="utf-8")
    overlay.write_text(yaml.dump({
        "theme": {"name": "男男", "inherits_legacy_content": False},
        "database": {"path": "themes/男男/data.db"},
        "face_recognition": {"faces_dir": "themes/男男/faces"},
        "analysis": {"mapping_nsfw_filter": True},
    }, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(ConfigManager, "_resolve_base_path", staticmethod(lambda: base))
    monkeypatch.setattr(ConfigManager, "current_overlay", None, raising=False)

    import src.core.theme_workspace as wsmod

    class _WS:
        def current_config_path(self):
            return overlay

    monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: _WS())
    cm = ConfigManager()
    cm._test_base, cm._test_overlay = base, overlay
    return cm


class TestMergedRead:
    def test_overlay_wins_over_base(self, layered):
        assert layered.get("database.path") == "themes/男男/data.db"

    def test_base_fills_keys_absent_from_overlay(self, layered):
        """全局设置对主题生效——这正是分层要修的老问题（整份拷贝时改了不生效）。"""
        assert layered.get("video.frames_per_scene") == 5
        assert layered.get("vlm.active") == "本地"
        assert layered.get("analysis.merge_vlm_passes") is True

    def test_overlay_own_keys(self, layered):
        assert layered.get("theme.name") == "男男"
        assert layered.get("analysis.mapping_nsfw_filter") is True

    def test_missing_key_returns_default(self, layered):
        assert layered.get("nothing.here", "fallback") == "fallback"


class TestWriteRouting:
    def test_global_key_writes_base_only(self, layered):
        layered.set("video.frames_per_scene", 9)
        base = yaml.safe_load(layered._test_base.read_text(encoding="utf-8"))
        overlay = yaml.safe_load(layered._test_overlay.read_text(encoding="utf-8"))
        assert base["video"]["frames_per_scene"] == 9
        # 关键：写全局**不能**把覆盖层的内容抄进来，也不能改覆盖层
        assert "theme" not in base
        assert base["database"]["path"] == "global.db"      # 未被覆盖层的值污染
        assert overlay["analysis"]["mapping_nsfw_filter"] is True

    def test_overlay_key_writes_overlay_only(self, layered):
        layered.set("analysis.active_theme", "日常")          # 覆盖键（内容域自描述）
        base = yaml.safe_load(layered._test_base.read_text(encoding="utf-8"))
        overlay = yaml.safe_load(layered._test_overlay.read_text(encoding="utf-8"))
        assert overlay["analysis"]["active_theme"] == "日常"
        assert overlay["theme"]["name"] == "男男"             # 覆盖层其余内容不动
        # 全局文件不得被写脏（不含 overlay 专属键）
        assert "theme" not in base
        assert base["analysis"] == {"merge_vlm_passes": True}

    def test_paths_are_overlay_keys(self, layered):
        layered.set("database.path", "themes/日常/data.db")
        base = yaml.safe_load(layered._test_base.read_text(encoding="utf-8"))
        assert base["database"]["path"] == "global.db"       # 全局那份不动
        overlay = yaml.safe_load(layered._test_overlay.read_text(encoding="utf-8"))
        assert overlay["database"]["path"] == "themes/日常/data.db"

    def test_read_back_is_immediate(self, layered):
        layered.set("video.frames_per_scene", 12)
        assert layered.get("video.frames_per_scene") == 12

    def test_regression_switch_does_not_leak_other_layer(self, layered):
        """连续跨层写：两层内容不互相渗透。"""
        layered.set("video.frames_per_scene", 7)              # 全局
        layered.set("theme.note", "按主题备注")                 # 覆盖
        layered.set("vlm.active", "远程")                       # 全局
        layered.set("analysis.active_theme", "日常")            # 覆盖
        base = yaml.safe_load(layered._test_base.read_text(encoding="utf-8"))
        overlay = yaml.safe_load(layered._test_overlay.read_text(encoding="utf-8"))
        assert base["video"]["frames_per_scene"] == 7
        assert base["vlm"]["active"] == "远程"
        assert "note" not in base.get("theme", {})
        assert overlay["theme"]["note"] == "按主题备注"
        assert overlay["analysis"]["active_theme"] == "日常"
        assert "video" not in overlay and "vlm" not in overlay


class TestKeyOwnership:
    @pytest.mark.parametrize("key", [
        "database.path",
        "face_recognition.faces_dir",
        "analysis.active_theme",
        "theme.name",
        "theme.label_mapping_path",
    ])
    def test_overlay_keys(self, layered, key):
        """随主题的只有**内容域与主题身份**。"""
        assert layered._is_overlay_key(key) is True

    @pytest.mark.parametrize("key", [
        "analysis.merge_vlm_passes",           # 走哪条代码路径的回退开关
        "analysis.mapping_nsfw_filter",        # 分析行为/处理策略 → 全局
        "analysis.mapping_frames_per_unit",
        "analysis.mapping_nsfw_min_frames",
        "video.frames_per_scene",
        "vlm.active",
        "nfo.output_dir",
        "database.sqlite.timeout",
    ])
    def test_global_keys(self, layered, key):
        assert layered._is_overlay_key(key) is False


class TestReadSideStillSupportsPerTheme:
    """**读取**与写路由无关：主题 config 里出现什么键，该键就按主题覆盖。

    这是"将来若确有按主题差异再下放"的通道——不需要改读取路径。
    """

    def test_hand_written_mapping_key_in_overlay_wins(self, tmp_path, monkeypatch):
        import yaml as _yaml

        base = tmp_path / "config.local.yaml"
        overlay = tmp_path / "themes" / "男男" / "config.yaml"
        overlay.parent.mkdir(parents=True)
        base.write_text(_yaml.dump({"analysis": {"mapping_nsfw_filter": None}}), encoding="utf-8")
        overlay.write_text(_yaml.dump({"analysis": {"mapping_nsfw_filter": True}}), encoding="utf-8")
        monkeypatch.setattr(ConfigManager, "_resolve_base_path", staticmethod(lambda: base))

        import src.core.theme_workspace as wsmod

        class _WS:
            def current_config_path(self):
                return overlay

        monkeypatch.setattr(wsmod, "get_theme_workspace", lambda: _WS())
        assert ConfigManager().get("analysis.mapping_nsfw_filter") is True


class TestSingleFileMode:
    def test_explicit_path_disables_layering(self, tmp_path):
        """显式传入路径 = 单文件模式（测试/兼容）：读写都走它。"""
        p = tmp_path / "only.yaml"
        p.write_text(yaml.dump({"video": {"frames_per_scene": 4}}), encoding="utf-8")
        cm = ConfigManager(str(p))
        assert cm.base_path == p and cm.overlay_path is None
        cm.set("video.frames_per_scene", 6)
        cm.set("database.path", "x.db")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert data["video"]["frames_per_scene"] == 6
        assert data["database"]["path"] == "x.db"


class TestDeepMerge:
    def test_nested_merge_keeps_siblings(self):
        out = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
        assert out == {"a": {"x": 1, "y": 9}}

    def test_none_is_a_value_not_a_gap(self):
        """None 是有效值（mapping_nsfw_filter: None = 按主题自动），不能被当缺省跳过。"""
        out = _deep_merge({"analysis": {"mapping_nsfw_filter": True}},
                          {"analysis": {"mapping_nsfw_filter": None}})
        assert out["analysis"]["mapping_nsfw_filter"] is None

    def test_lists_replaced_not_merged(self):
        out = _deep_merge({"a": [1, 2, 3]}, {"a": [9]})
        assert out["a"] == [9]

    def test_base_untouched(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestVideoBlock:
    def test_get_video_config_returns_full_block(self, layered):
        """手工投影缺键的历史缺陷:完整 video 块必须透出(任务 2.3)。"""
        import yaml
        base = yaml.safe_load(open(layered._test_base))
        base["video"]["max_frames_per_video"] = 60
        base["video"]["frame_density"] = 4
        open(layered._test_base, "w").write(yaml.dump(base, allow_unicode=True))
        cm = ConfigManager()
        vc = cm.get_video_config()
        assert vc["max_frames_per_video"] == 60
        assert vc["frame_density"] == 4
        assert vc["frames_per_scene"] == 5

    def test_overlay_none_block_does_not_nuke_base(self, layered, monkeypatch):
        """覆盖层整块 None 视为未覆盖——曾误写 video: null 炸掉基底全部视频配置。"""
        import yaml
        overlay = yaml.safe_load(open(layered._test_overlay))
        overlay["video"] = None
        open(layered._test_overlay, "w").write(yaml.dump(overlay, allow_unicode=True))
        cm = ConfigManager()
        assert cm.get_video_config()["frames_per_scene"] == 5
