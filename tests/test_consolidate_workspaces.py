"""并库迁移（scripts/consolidate_workspaces_into_theme.py）的规则测试。

用**合成的小库**覆盖四条容易出错的规则（都是实测踩到过的）：

1. **并集账目**：目标库 = 基底 ∪（原库中仍存在于某主题库的行）——每个 id 只出现一次
2. **影子行重分配**：同 id 但内容是**另一个业务实体**（rowid 复用）→ 分配新 id 并入，
   且引用它的 `entity_id` 同步改写
3. **已删除的数据不复活**：原库有、但所有主题库都没有的行 → 不并入
4. **在线守卫**：端口占用 / 桌面应用在跑时，`--apply` 必须拒绝执行

跑法::

    uv run --active python -m pytest tests/test_consolidate_workspaces.py -q
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import consolidate_workspaces_into_theme as cwt  # noqa: E402
from src.db.models import init_db  # noqa: E402


# ── 造库 ────────────────────────────────────────────────────────────
def _new_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    init_db(str(path))
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys=OFF")
    return con


def _entity(con, eid, md5, theme):
    con.execute("INSERT INTO file_entities (id, file_md5, file_size, analysis_theme) "
                "VALUES (?,?,?,?)", (eid, md5, 100, theme))


def _location(con, lid, eid, path):
    con.execute("INSERT INTO file_locations (id, entity_id, file_path, file_name, is_primary) "
                "VALUES (?,?,?,?,1)", (lid, eid, path, Path(path).name))


def _analysis(con, aid, eid, status="READY"):
    con.execute("INSERT INTO file_analysis (id, entity_id, status, progress) VALUES (?,?,?,0)",
                (aid, eid, status))


def _actor(con, aid, code, sample=1):
    con.execute("INSERT INTO actors (id, code, sample_count) VALUES (?,?,?)", (aid, code, sample))


def _face(con, fid, actor_id, entity_id, path, md5="m"):
    con.execute("INSERT INTO actor_faces (id, actor_id, entity_id, face_path, file_md5) "
                "VALUES (?,?,?,?,?)", (fid, actor_id, entity_id, path, md5))


def _tfm(con, tid, entity_id, tag, path):
    con.execute("INSERT INTO tag_frame_mappings (id, entity_id, tag_name, frame_path) "
                "VALUES (?,?,?,?)", (tid, entity_id, tag, path))


def _scenario(tmp: Path):
    """原库 + 三个主题库（id 保留的子集 + 一个影子行 + 一条已删数据）。"""
    original = tmp / "original.db"
    con = _new_db(original)
    # 实体 1..6（各主题归属），其中 6 只在「日常」
    for eid, theme in [(1, "男男"), (2, "男男"), (3, "成人内容"), (4, "成人内容"),
                       (5, "成人内容"), (6, "日常")]:
        _entity(con, eid, f"md5{eid}", theme)
    for eid in range(1, 7):
        _location(con, 10 + eid, eid, f"/v/{eid}.mp4")
        _analysis(con, 20 + eid, eid)
    _actor(con, 31, "ACTOR_031")
    _actor(con, 32, "ACTOR_032")
    _face(con, 41, 31, 1, "ACTOR_031/face_0.jpg")
    _face(con, 42, 31, 2, "ACTOR_031/face_1.jpg")
    _face(con, 43, 32, 3, "ACTOR_032/face_0.jpg")
    _face(con, 44, 32, 4, "ACTOR_032/face_1.jpg")     # ← 只在原库（各主题库都没有 → 不复活）
    con.execute("INSERT INTO tasks (id, entity_id, status) VALUES (51,1,'QUEUED')")
    _tfm(con, 61, 1, "卧室", "/f/1.jpg")
    con.commit()
    con.close()

    themes = tmp / "data" / "themes"
    for name in ["男男", "成人内容", "日常", "男女", "女女", "AI 视频"]:
        (themes / name).mkdir(parents=True)
    # 基底：男男 = {1,2} + 拆分后新增（tfm 62）
    con = _new_db(themes / "男男" / "data.db")
    for eid in (1, 2):
        _entity(con, eid, f"md5{eid}", "男男")
        _location(con, 10 + eid, eid, f"/v/{eid}.mp4")
        _analysis(con, 20 + eid, eid)
    _actor(con, 31, "ACTOR_031")
    _face(con, 41, 31, 1, "ACTOR_031/face_0.jpg")
    _face(con, 42, 31, 2, "ACTOR_031/face_1.jpg")
    con.execute("INSERT INTO tasks (id, entity_id, status) VALUES (51,1,'QUEUED')")
    _tfm(con, 61, 1, "卧室", "/f/1.jpg")
    _tfm(con, 62, 1, "浴室", "/f/2.jpg")          # ← 拆分后新增
    con.commit()
    con.close()

    # 成人内容 = {3,4} + **影子行**：id 5 内容是另一个实体（rowid 复用）
    con = _new_db(themes / "成人内容" / "data.db")
    for eid in (3, 4):
        _entity(con, eid, f"md5{eid}", "成人内容")
        _location(con, 10 + eid, eid, f"/v/{eid}.mp4")
        _analysis(con, 20 + eid, eid)
    _actor(con, 32, "ACTOR_032")
    _entity(con, 5, "shadow-md5", "成人内容")      # ← 与基底/原库的 5 撞号但内容不同
    _location(con, 15, 5, "/v/shadow.mp4")
    _analysis(con, 25, 5, status="PENDING")        # ← 与基底/原库的 25 撞号但状态不同
    _face(con, 43, 32, 3, "ACTOR_032/face_0.jpg")
    _face(con, 45, 32, 5, "ACTOR_032/shadow.jpg")
    con.execute("INSERT INTO tasks (id, entity_id, status) VALUES (52,5,'CHECKING')")
    con.commit()
    con.close()

    # 日常 = {6}（内容与原库一致 → 不产生漂移）
    con = _new_db(themes / "日常" / "data.db")
    _entity(con, 6, "md56", "日常")
    _location(con, 16, 6, "/v/6.mp4")
    _analysis(con, 26, 6)
    con.commit()
    con.close()

    faces_root = tmp / "legacy_faces"
    # 拆分复制落点：<人脸根>/<主题>/<相对路径>
    for rel in ["ACTOR_032/face_0.jpg", "ACTOR_032/shadow.jpg"]:
        p = faces_root / "成人内容" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    # 拆分前的原位置：<人脸根>/<相对路径>
    for rel in ["ACTOR_031/face_0.jpg", "ACTOR_031/face_1.jpg", "ACTOR_032/face_0.jpg"]:
        p = faces_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")

    return SimpleNamespace(
        base=tmp / "data", source_db=original, target_theme="男男",
        tag_library=ROOT / "resources/themes/男男标签库.md",
        legacy_faces_root=faces_root, apply=False, skip_faces=False,
    )


def _ids(db: Path, table: str):
    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute(f"SELECT id FROM {table} ORDER BY id")]
    finally:
        con.close()


def _row(db: Path, table: str, rid: int):
    con = sqlite3.connect(str(db))
    try:
        con.row_factory = sqlite3.Row
        return con.execute(f"SELECT * FROM {table} WHERE id=?", (rid,)).fetchone()
    finally:
        con.close()


# ── 测试 ────────────────────────────────────────────────────────────
@pytest.fixture
def scenario(tmp_path):
    return _scenario(tmp_path)


def test_rehearsal_produces_full_union(scenario, tmp_path):
    """并集账目：6 个实体全在；基底新增的 tfm 62 保住；已删的 face 44 不复活。"""
    out = tmp_path / "rehearsal.db"
    log, stats = cwt.build_plan(scenario, keep=out)

    assert stats["verify"]["ok"], log
    assert _ids(out, "file_entities") == [1, 2, 3, 4, 5, 6, 7]      # 6 个 + 影子重分配出的 7
    assert 62 in _ids(out, "tag_frame_mappings")
    assert 44 not in _ids(out, "actor_faces")                        # 各处都已删除 → 不复活
    assert stats["fill"]["file_entities"] == 4                       # 3 / 4 / 6 / 5(原库那行)
    assert "跳过] actor_faces: 1 行" in "\n".join(log)


def test_shadow_row_gets_new_id_and_references_remapped(scenario, tmp_path):
    """影子行：实体 5（成人内容那一版）→ 新 id，且引用它的 location/analysis/task 同步改写。

    新 id 取"补齐之后的 max+1"，因此一定落在所有原库 id 之后（不会撞上待插入的行）。
    """
    out = tmp_path / "rehearsal.db"
    log, stats = cwt.build_plan(scenario, keep=out)

    new_id = stats["entity_map"][5]
    assert new_id == 7                                   # 补齐后实体 id 到 6 → 新 id 7
    assert _row(out, "file_entities", new_id)["file_md5"] == "shadow-md5"
    assert _row(out, "file_entities", 5)["file_md5"] == "md55"     # 原库那行没被顶掉

    con = sqlite3.connect(str(out))
    con.row_factory = sqlite3.Row
    try:
        moved_loc = con.execute(
            "SELECT id, entity_id FROM file_locations WHERE file_path='/v/shadow.mp4'").fetchone()
        moved_ana = con.execute(
            "SELECT id, entity_id, status FROM file_analysis WHERE status='PENDING'").fetchone()
        task = con.execute("SELECT entity_id, status FROM tasks WHERE id=52").fetchone()
        orig_ana = con.execute("SELECT entity_id, status FROM file_analysis WHERE id=25").fetchone()
    finally:
        con.close()

    assert moved_loc["entity_id"] == new_id and moved_loc["id"] > 16
    assert moved_ana["entity_id"] == new_id and moved_ana["id"] > 26
    assert task["entity_id"] == new_id                   # 同一个任务 → 更新其链接
    assert orig_ana["entity_id"] == 5 and orig_ana["status"] == "READY"   # 原库那行仍是 READY


def test_no_new_orphans_and_theme_normalized(scenario, tmp_path):
    """归属归一 + 孤儿闸门：不得引入新孤儿（历史遗留不算）。"""
    out = tmp_path / "rehearsal.db"
    log, stats = cwt.build_plan(scenario, keep=out)
    con = sqlite3.connect(str(out))
    try:
        themes = [r[0] for r in con.execute(
            "SELECT DISTINCT analysis_theme FROM file_entities").fetchall()]
        orphans = cwt.collect_orphans(con)
    finally:
        con.close()
    assert themes == ["男男"]
    assert orphans == {}
    assert stats["verify"]["new_orphans"] == {}


def test_faces_only_referenced_copied(scenario, tmp_path):
    """人脸只复制被引用到的 face_path，且源目录不被删改。"""
    log, stats = cwt.build_plan(scenario)
    faces = stats["faces"]
    assert faces["copied"] + faces["already"] == faces["referenced"]
    assert faces["missing"] == 0
    # 源仍在（复制式）
    assert (scenario.legacy_faces_root / "成人内容" / "ACTOR_032" / "face_0.jpg").exists()


def test_apply_refused_when_api_online(scenario, monkeypatch, capsys):
    """在线守卫：端口被占用时 --apply 必须拒绝。"""
    monkeypatch.setattr(cwt, "port_busy", lambda *a, **k: True)
    monkeypatch.setattr(cwt, "ui_running", lambda: False)
    monkeypatch.setattr(sys, "argv", ["consolidate", "--apply",
                                      "--base", str(scenario.base),
                                      "--source-db", str(scenario.source_db),
                                      "--legacy-faces-root", str(scenario.legacy_faces_root)])
    rc = cwt.main()
    assert rc == 2
    assert "必须离线" in capsys.readouterr().out


def test_apply_refused_when_desktop_running(scenario, monkeypatch, capsys):
    """在线守卫：桌面应用在跑时也要拒绝。"""
    monkeypatch.setattr(cwt, "port_busy", lambda *a, **k: False)
    monkeypatch.setattr(cwt, "ui_running", lambda: True)
    monkeypatch.setattr(sys, "argv", ["consolidate", "--apply",
                                      "--base", str(scenario.base),
                                      "--source-db", str(scenario.source_db),
                                      "--legacy-faces-root", str(scenario.legacy_faces_root)])
    assert cwt.main() == 2
    assert "桌面应用仍在运行" in capsys.readouterr().out


def test_dry_run_does_not_touch_real_db(scenario):
    """彩排绝不改真实库：基底库的行数在彩排前后一致。"""
    base_db = scenario.base / "themes" / "男男" / "data.db"
    before = _ids(base_db, "file_entities")
    cwt.build_plan(scenario)
    assert _ids(base_db, "file_entities") == before
    assert not list((scenario.base / "themes").glob("_backup_*"))
