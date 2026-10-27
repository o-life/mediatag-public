"""Tests for "编号 = 记录号"（actors.code 恒等于 ACTOR_{id:03d}）。

覆盖:
1. create_actor 落库后 code 由 id 派生（不再有独立编号策略导致的脱节）
2. 连续建号保持对齐（曾因"找空洞复用"逐号漂移）
3. 迁移脚本的纯函数: 映射计算 / 路径前缀改写 / 两阶段改名规划 / 校验
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.models import Base, Actor, ActorFace
from src.core.face_recognition import FaceRecognizer
from scripts.migrate_actor_code_to_id import (
    actor_code_for_id,
    compute_mapping,
    mapping_csv_rows,
    plan_two_phase,
    rewrite_path_prefix,
    verify_alignment,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


@pytest.fixture
def rec(tmp_path):
    """不加载模型的 FaceRecognizer（只测建号逻辑）。"""
    r = FaceRecognizer.__new__(FaceRecognizer)
    r.faces_dir = tmp_path / "actor_faces"
    r.faces_dir.mkdir(parents=True)
    r.MIN_CLARITY_FOR_COVER = 0.01
    r._max_samples_per_actor = lambda: 5
    r._select_representative_faces = lambda faces, n: faces[:n]
    r._select_best_cover = lambda faces: faces[0]
    # create_actor 会用 cv2 重算清晰度; 测试用假图 → 直接给高清晰度
    r._calculate_clarity = lambda p: 1.0
    r._get_gate_app = lambda: None   # 跳过"非人脸闸门"（需真实模型）
    return r


def _faces(tmp_path, n=3):
    """构造 n 个"人脸字典"（含真实存在的图片文件）。

    姿态需有差异: create_actor 有"静态图片检测"，yaw/pitch/roll 标准差
    过低会判定为手机屏幕/海报等而跳过建号。
    """
    out = []
    for i in range(n):
        p = tmp_path / f"temp_face_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
        out.append({
            "face_path": str(p),
            "embedding": np.zeros(512, dtype=np.float32) + i,
            "yaw": float(i * 7.0), "pitch": float(i * 3.0), "roll": float(i * 2.0),
            "clarity": 1.0, "sharpness": 100.0, "quality_score": 0.9,
        })
    return out


# ---------------------------------------------------------------------------
# 1/2. create_actor 建号
# ---------------------------------------------------------------------------

class TestCreateActorCode:
    def test_code_derives_from_id(self, db, rec, tmp_path):
        aid = rec.create_actor(_faces(tmp_path), task_id=None, file_md5="m1", db_session=db)
        assert aid is not None
        a = db.query(Actor).filter(Actor.id == aid).first()
        assert a.code == f"ACTOR_{aid:03d}", f"code={a.code} 应为 ACTOR_{aid:03d}"
        assert (rec.faces_dir / a.code).is_dir(), "目录名应与编号一致"

    def test_sequential_creates_stay_aligned(self, db, rec, tmp_path):
        """连建 3 个, 每个的 code 都应等于自己的 id（历史: 会逐号漂移）。"""
        ids = []
        for k in range(3):
            d = tmp_path / f"s{k}"
            d.mkdir()
            ids.append(rec.create_actor(_faces(d, 2), task_id=None,
                                        file_md5=f"m{k}", db_session=db))
        rows = [(a.id, a.code) for a in db.query(Actor).all()]
        assert verify_alignment(rows) == [], f"存在脱节: {verify_alignment(rows)}"
        assert len(set(ids)) == 3

    def test_face_paths_point_into_own_dir(self, db, rec, tmp_path):
        aid = rec.create_actor(_faces(tmp_path), task_id=None, file_md5="m", db_session=db)
        code = db.query(Actor.code).filter(Actor.id == aid).scalar()
        paths = [r[0] for r in db.query(ActorFace.face_path)
                 .filter(ActorFace.actor_id == aid).all()]
        assert paths, "应登记人脸记录"
        assert all(p.startswith(code + "/") for p in paths), f"face_path 前缀应为 {code}: {paths}"


# ---------------------------------------------------------------------------
# 3. 迁移脚本纯函数
# ---------------------------------------------------------------------------

class TestMigrationHelpers:
    def test_actor_code_for_id(self):
        assert actor_code_for_id(1) == "ACTOR_001"
        assert actor_code_for_id(341) == "ACTOR_341"
        assert actor_code_for_id(1234) == "ACTOR_1234"   # 超 3 位仍正确

    def test_compute_mapping_only_mismatched(self):
        actors = [(1, "ACTOR_001"), (341, "ACTOR_004"), (786, "ACTOR_787")]
        m = compute_mapping(actors)
        assert 1 not in m, "已一致的不应进映射"
        assert m[341] == ("ACTOR_004", "ACTOR_341")
        assert m[786] == ("ACTOR_787", "ACTOR_786")

    def test_compute_mapping_new_codes_unique(self):
        """id 唯一 → 新编号必唯一（这是能安全改名的前提）。"""
        actors = [(i, f"ACTOR_{i + 1:03d}") for i in range(1, 50)]
        m = compute_mapping(actors)
        news = [new for _, new in m.values()]
        assert len(news) == len(set(news))

    def test_rewrite_path_prefix_relative(self):
        assert rewrite_path_prefix("ACTOR_004/face_0.jpg", "ACTOR_004", "ACTOR_341") \
            == "ACTOR_341/face_0.jpg"

    def test_rewrite_path_prefix_legacy_form(self):
        assert rewrite_path_prefix("data/actor_faces/ACTOR_004/face_0.jpg",
                                   "ACTOR_004", "ACTOR_341") \
            == "data/actor_faces/ACTOR_341/face_0.jpg"

    def test_rewrite_does_not_touch_other_actor(self):
        """尾斜杠防止 ACTOR_1 误伤 ACTOR_10。"""
        assert rewrite_path_prefix("ACTOR_10/face_0.jpg", "ACTOR_1", "ACTOR_9") \
            == "ACTOR_10/face_0.jpg"

    def test_rewrite_none_and_empty(self):
        assert rewrite_path_prefix(None, "ACTOR_1", "ACTOR_2") is None
        assert rewrite_path_prefix("", "ACTOR_1", "ACTOR_2") == ""

    def test_two_phase_plan_covers_every_actor(self):
        m = {341: ("ACTOR_004", "ACTOR_341"), 342: ("ACTOR_005", "ACTOR_342")}
        to_temp, to_final = plan_two_phase(m)
        assert {s for s, _ in to_temp} == {"ACTOR_004", "ACTOR_005"}
        assert {d for _, d in to_final} == {"ACTOR_341", "ACTOR_342"}
        # 临时名唯一（避免彼此覆盖）
        tmps = [d for _, d in to_temp]
        assert len(tmps) == len(set(tmps))
        # 每个临时名都对应一个 final 起点（闭环）
        assert {s for s, _ in to_final} == set(tmps)

    def test_two_phase_handles_swap_cycle(self):
        """环形占用: A 的新名正是 B 的旧名 —— 两阶段必须能安全处理。"""
        m = {2: ("ACTOR_001", "ACTOR_002"), 1: ("ACTOR_002", "ACTOR_001")}
        to_temp, to_final = plan_two_phase(m)
        # 先全部搬到临时名（此时旧名全空出），再搬到目标
        assert all(s in {"ACTOR_001", "ACTOR_002"} for s, _ in to_temp)
        assert all(d in {"ACTOR_001", "ACTOR_002"} for _, d in to_final)

    def test_verify_alignment(self):
        assert verify_alignment([(1, "ACTOR_001"), (341, "ACTOR_341")]) == []
        assert verify_alignment([(341, "ACTOR_004")]) == [(341, "ACTOR_004")]

    def test_mapping_csv_rows_sorted(self):
        m = {342: ("ACTOR_005", "ACTOR_342"), 341: ("ACTOR_004", "ACTOR_341")}
        rows = mapping_csv_rows(m)
        assert [r["id"] for r in rows] == [341, 342]
        assert rows[0] == {"id": 341, "old_code": "ACTOR_004", "new_code": "ACTOR_341"}


# ---------------------------------------------------------------------------
# 4. 端到端: 在合成数据上真实执行迁移（含环形占用）
# ---------------------------------------------------------------------------

def _mk_db(path):
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE actors (id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL, cover_face TEXT);
        CREATE TABLE actor_faces (id INTEGER PRIMARY KEY, actor_id INTEGER, face_path TEXT);
    """)
    return con


def _mk_actor_dir(faces_dir, code, n=2):
    d = faces_dir / code
    d.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        f = d / f"face_{i}.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        names.append(f"{code}/face_{i}.jpg")
    return names


class TestMigrationEndToEnd:
    def _seed(self, tmp_path, actors):
        """actors: [(id, code)] → 建库 + 目录 + face 记录。"""
        faces_dir = tmp_path / "actor_faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        con = _mk_db(tmp_path / "t.db")
        for aid, code in actors:
            con.execute("INSERT INTO actors(id, code, cover_face) VALUES (?,?,?)",
                        (aid, code, f"{code}/face_0.jpg"))
            for j, rel in enumerate(_mk_actor_dir(faces_dir, code)):
                con.execute("INSERT INTO actor_faces(actor_id, face_path) VALUES (?,?)",
                            (aid, rel))
        con.commit()
        return con, faces_dir

    def _migrate(self, con, faces_dir):
        from scripts.migrate_actor_code_to_id import compute_mapping, do_migrate, load_actors
        mapping = compute_mapping(load_actors(con))
        do_migrate(con, faces_dir, mapping, apply=True)
        return mapping

    def test_simple_rename_updates_everything(self, tmp_path):
        con, faces_dir = self._seed(tmp_path, [(5, "ACTOR_003")])
        self._migrate(con, faces_dir)

        assert con.execute("SELECT code FROM actors WHERE id=5").fetchone()[0] == "ACTOR_005"
        assert con.execute("SELECT cover_face FROM actors WHERE id=5").fetchone()[0] == "ACTOR_005/face_0.jpg"
        for (fp,) in con.execute("SELECT face_path FROM actor_faces WHERE actor_id=5"):
            assert fp.startswith("ACTOR_005/"), fp
            assert (faces_dir / fp).exists(), f"文件应仍在: {fp}"
        assert not (faces_dir / "ACTOR_003").exists(), "旧目录应已消失"
        assert (faces_dir / "ACTOR_005").is_dir()

    def test_swap_cycle_no_data_loss(self, tmp_path):
        """环形占用: 1↔2 互换 —— 两阶段改名必须不丢文件、不互相覆盖。"""
        con, faces_dir = self._seed(tmp_path, [(1, "ACTOR_002"), (2, "ACTOR_001")])
        self._migrate(con, faces_dir)

        assert con.execute("SELECT code FROM actors WHERE id=1").fetchone()[0] == "ACTOR_001"
        assert con.execute("SELECT code FROM actors WHERE id=2").fetchone()[0] == "ACTOR_002"
        # 两个目录都在，且各自含 2 个文件（无覆盖/丢失）
        for code in ("ACTOR_001", "ACTOR_002"):
            n = len(list((faces_dir / code).glob("face_*.jpg")))
            assert n == 2, f"{code} 应有 2 个文件, 实际 {n}"
        # 无临时目录残留
        assert not [p for p in faces_dir.iterdir() if p.name.startswith(".__codefix_")]
        # face_path 全部可解析
        for (fp,) in con.execute("SELECT face_path FROM actor_faces"):
            assert (faces_dir / fp).exists(), fp

    def test_alignment_after_migration(self, tmp_path):
        con, faces_dir = self._seed(tmp_path, [(1, "ACTOR_005"), (7, "ACTOR_002"), (9, "ACTOR_009")])
        self._migrate(con, faces_dir)
        rows = con.execute("SELECT id, code FROM actors").fetchall()
        assert verify_alignment(rows) == [], verify_alignment(rows)

    def test_dry_run_changes_nothing(self, tmp_path):
        con, faces_dir = self._seed(tmp_path, [(5, "ACTOR_003")])
        from scripts.migrate_actor_code_to_id import compute_mapping, do_migrate, load_actors
        mapping = compute_mapping(load_actors(con))
        do_migrate(con, faces_dir, mapping, apply=False)
        assert con.execute("SELECT code FROM actors WHERE id=5").fetchone()[0] == "ACTOR_003"
        assert (faces_dir / "ACTOR_003").is_dir()

    def test_legacy_prefix_in_face_path_rewritten(self, tmp_path):
        """兼容历史形态 'data/actor_faces/ACTOR_x/...'。"""
        con, faces_dir = self._seed(tmp_path, [(5, "ACTOR_003")])
        con.execute("UPDATE actor_faces SET face_path='data/actor_faces/ACTOR_003/face_0.jpg' "
                    "WHERE actor_id=5")
        con.commit()
        self._migrate(con, faces_dir)
        got = con.execute("SELECT face_path FROM actor_faces WHERE actor_id=5").fetchone()[0]
        assert got == "data/actor_faces/ACTOR_005/face_0.jpg", got
