"""``faces_dir`` 解析口径统一（Vault 计划 Phase 0）。

历史：同一配置项散在 5 处解析、用了 3 种口径——
`FaceRecognizer.__init__` / `_api_faces_dir` 相对路径回退**项目根**；
`routes/actors._faces_dir` 与 `list_actors` / `get_actor` 两处内联按**进程 CWD**；
是否 `expanduser` 也不一致。于是同一个相对路径可能被解析到不同目录。

现收口到 `src/core/face_paths.py::resolve_faces_dir`，本文件锁定该口径
并用源码级护栏防止再分叉。
"""

from pathlib import Path

from src.core.face_paths import DEFAULT_FACES_DIR, project_root, resolve_faces_dir


class TestResolveFacesDir:
    def test_absolute_stays_absolute(self, tmp_path):
        target = tmp_path / "faces"
        assert resolve_faces_dir(str(target)) == target.resolve()

    def test_tilde_is_expanded(self):
        got = resolve_faces_dir("~/actor_faces")
        assert got == (Path.home() / "actor_faces").resolve()
        assert not str(got).startswith("~")

    def test_relative_resolves_against_project_root(self):
        assert resolve_faces_dir("data/actor_faces") == (project_root() / "data/actor_faces").resolve()

    def test_relative_is_not_cwd_relative(self, tmp_path, monkeypatch):
        """关键回归：相对路径必须相对项目根，不得跟随进程 CWD。"""
        monkeypatch.chdir(tmp_path)
        got = resolve_faces_dir("data/actor_faces")
        assert got == (project_root() / "data/actor_faces").resolve()
        assert tmp_path not in got.parents

    def test_reads_from_injected_config_getter(self):
        got = resolve_faces_dir(get_config=lambda key, default=None: "/tmp/faces_root")
        assert got == Path("/tmp/faces_root").resolve()

    def test_falls_back_to_default_when_config_missing(self):
        got = resolve_faces_dir(get_config=lambda key, default=None: default)
        assert got == (project_root() / DEFAULT_FACES_DIR).resolve()

    def test_trailing_separator_is_normalized(self, tmp_path):
        """生产配置带尾斜杠（/Volumes/MediaData/actor_faces/）——解析结果应无尾斜杠差异。"""
        assert resolve_faces_dir(str(tmp_path) + "/") == tmp_path.resolve()


class TestSingleSourceOfTruth:
    """源码级护栏：运行时不得再各写一份 faces_dir 解析。"""

    def _read(self, rel: str) -> str:
        return (project_root() / rel).read_text(encoding="utf-8")

    def test_no_inline_config_key_resolution_left(self):
        offenders = []
        for rel in ("src/api/main.py", "src/api/routes/actors.py",
                    "src/core/face_recognition.py"):
            text = self._read(rel)
            if 'get("face_recognition.faces_dir"' in text:
                offenders.append(rel)
        assert not offenders, f"仍有内联 faces_dir 解析，应改走 resolve_faces_dir(): {offenders}"

    def test_runtime_call_sites_use_helper(self):
        assert "resolve_faces_dir" in self._read("src/api/main.py")
        assert "resolve_faces_dir" in self._read("src/api/routes/actors.py")
        assert "resolve_faces_dir" in self._read("src/core/face_recognition.py")
