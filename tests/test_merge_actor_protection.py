"""Tests for 合并演员的两项保护:
1. 身份信息(人工重命名/别名)不因合并丢失
2. 人脸文件搬移失败/缺失时的保护(不产生指向缺失文件的坏记录)

背景:
- "重命名丢失": 合并后源记录被删除, 其 name/primary_name/aliases 一并消失;
  且 target 原为查询返回的第一个(通常 id 最小), 前端又用 set 传参(顺序随机)
  → 想保留的那个未必是 target。
- "搬移失败": 文件留在源目录、源目录却被删除 → 记录指向不存在的文件。
"""

import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.api.main as M
from src.api.main import _parse_alias_list, merge_actor_identity
from src.db.models import Actor, ActorFace, Base


# ---------------------------------------------------------------------------
# 纯函数: 身份信息合并
# ---------------------------------------------------------------------------

class _A:
    """轻量替身(只带身份字段)。"""
    def __init__(self, name=None, primary_name=None, aliases=None):
        self.name = name
        self.primary_name = primary_name
        self.aliases = aliases


class TestParseAliasList:
    def test_json_array(self):
        assert _parse_alias_list('["a", "b"]') == ["a", "b"]

    def test_comma_separated(self):
        assert _parse_alias_list("a, b ,c") == ["a", "b", "c"]

    def test_empty_and_none(self):
        assert _parse_alias_list(None) == []
        assert _parse_alias_list("") == []
        assert _parse_alias_list("  ") == []

    def test_malformed_json_falls_back(self):
        assert _parse_alias_list('["a",') == ['["a"']

    def test_non_string_entries_skipped(self):
        assert _parse_alias_list('["a", 1, null, "b"]') == ["a", "b"]


class TestMergeActorIdentity:
    def test_target_keeps_own_name(self):
        r = merge_actor_identity(_A(name="张三"), [_A(name="李四")])
        assert r["name"] == "张三"
        assert "李四" in r["aliases"], "源的名字应作为别名保留"

    def test_target_inherits_source_name_when_empty(self):
        """核心: 重命名的那个不是 target 时, 名字不能丢。"""
        r = merge_actor_identity(_A(name=None), [_A(name="张三")])
        assert r["name"] == "张三"
        assert "张三" not in r["aliases"], "已作为主名则不再重复计入别名"

    def test_inherits_primary_name(self):
        r = merge_actor_identity(_A(), [_A(primary_name="张老师")])
        assert r["primary_name"] == "张老师"

    def test_first_non_empty_wins(self):
        r = merge_actor_identity(_A(), [_A(), _A(name="第一个"), _A(name="第二个")])
        assert r["name"] == "第一个"
        assert "第二个" in r["aliases"]

    def test_aliases_merged_and_deduped(self):
        r = merge_actor_identity(
            _A(aliases='["旧别名"]'),
            [_A(aliases='["旧别名", "新别名"]')],
        )
        assert r["aliases"].count("旧别名") == 1
        assert "新别名" in r["aliases"]

    def test_no_name_anywhere(self):
        r = merge_actor_identity(_A(), [_A()])
        assert r["name"] is None
        assert r["primary_name"] is None
        assert r["aliases"] == []

    def test_replaced_target_name_preserved_as_alias(self):
        """target 原名字被覆盖时, 原名字也要保留为别名(信息只增不减)。"""
        r = merge_actor_identity(_A(primary_name="原名"), [_A(primary_name="新主名")])
        assert r["primary_name"] == "原名", "target 已有主名则保留自己的"
        assert "新主名" in r["aliases"]


# ---------------------------------------------------------------------------
# 端到端: 真实合并(临时库 + 临时 faces_dir)
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    """临时 DB + 临时 faces_dir, 并把模块级依赖指过去。"""
    engine = create_engine(f"sqlite:///{tmp_path/'m.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    faces = tmp_path / "actor_faces"
    faces.mkdir()

    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "_api_faces_dir", lambda: faces)
    return Session(), faces


def _add_actor(db, code, name=None, primary_name=None, aliases=None, n_faces=2, faces_dir=None,
               missing=0, outside=False):
    a = Actor(code=code, name=name, primary_name=primary_name, aliases=aliases)
    db.add(a); db.flush()
    d = faces_dir / code
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n_faces):
        f = d / f"face_{i}.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        path = f"{code}/face_{i}.jpg"
        if i < missing:
            f.unlink()                      # 文件缺失但记录仍在
        if outside and i == n_faces - 1:
            other = faces_dir / "_elsewhere"
            other.mkdir(exist_ok=True)
            alt = other / f"x{i}.jpg"
            alt.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
            path = str(alt)                 # 指向源目录之外(绝对路径)
        db.add(ActorFace(actor_id=a.id, face_path=path, file_md5="m1", is_cover=0))
    db.commit()
    return a


def _client():
    from fastapi.testclient import TestClient
    return TestClient(M.app)


class TestMergeEndToEnd:
    def test_target_follows_request_order(self, env):
        """actor_ids[0] 必须是 target —— 不再由查询顺序决定。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", name="A", faces_dir=faces)
        b = _add_actor(db, "ACTOR_002", name="B", faces_dir=faces)
        r = _client().post("/actors/merge", json={"actor_ids": [b.id, a.id]})
        assert r.status_code == 200, r.text
        assert r.json()["merged_actor_id"] == b.id, "第二个参数在前 → 应作为主记录"

    def test_renamed_actor_name_survives_merge(self, env):
        """人工重命名的演员作为源时, 名字必须保留(继承或入别名)。"""
        db, faces = env
        plain = _add_actor(db, "ACTOR_001", faces_dir=faces)             # 无名字(主记录)
        named = _add_actor(db, "ACTOR_002", name="张三", faces_dir=faces)  # 被重命名
        r = _client().post("/actors/merge", json={"actor_ids": [plain.id, named.id]})
        assert r.status_code == 200, r.text
        db.expire_all()
        t = db.query(Actor).filter(Actor.id == plain.id).first()
        assert t.name == "张三", f"名字应被继承, 实际={t.name!r}"

    def test_source_name_kept_in_aliases_when_target_named(self, env):
        """两边都有名字: target 的名字为主, 源的名字进别名(不丢)。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", name="甲", faces_dir=faces)
        b = _add_actor(db, "ACTOR_002", name="乙", faces_dir=faces)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 200, r.text
        db.expire_all()
        t = db.query(Actor).filter(Actor.id == a.id).first()
        assert t.name == "甲"
        assert "乙" in _parse_alias_list(t.aliases)

    def test_aliases_merged(self, env):
        db, faces = env
        a = _add_actor(db, "ACTOR_001", aliases=json.dumps(["老别名"]), faces_dir=faces)
        b = _add_actor(db, "ACTOR_002", aliases=json.dumps(["新别名"]), faces_dir=faces)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 200, r.text
        db.expire_all()
        t = db.query(Actor).filter(Actor.id == a.id).first()
        got = _parse_alias_list(t.aliases)
        assert set(got) >= {"老别名", "新别名"}

    def test_faces_and_files_land_in_target(self, env):
        db, faces = env
        a = _add_actor(db, "ACTOR_001", faces_dir=faces, n_faces=3)
        b = _add_actor(db, "ACTOR_002", faces_dir=faces, n_faces=2)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 200, r.text
        db.expire_all()
        recs = db.query(ActorFace).filter(ActorFace.actor_id == a.id).all()
        assert len(recs) == 5
        for f in recs:
            assert f.face_path.startswith("ACTOR_001/")
            assert (faces / f.face_path).exists(), f"文件应在目标目录: {f.face_path}"
        assert not (faces / "ACTOR_002").exists(), "源目录应已删除(文件已全部搬走)"

    def test_missing_file_records_kept_and_reported(self, env):
        """文件缺失的记录保留(不擅自杀数据), 并在响应里报告。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", faces_dir=faces, n_faces=2)
        b = _add_actor(db, "ACTOR_002", faces_dir=faces, n_faces=3, missing=2)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 200, r.text
        assert r.json()["missing_files"] == 2
        db.expire_all()
        assert db.query(ActorFace).filter(ActorFace.actor_id == a.id).count() == 5

    def test_file_outside_source_dir_not_moved_but_reassigned(self, env):
        """指向源目录之外的文件不搬移(搬了才是错的), 但记录归属照常迁移。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", faces_dir=faces, n_faces=1)
        b = _add_actor(db, "ACTOR_002", faces_dir=faces, n_faces=2, outside=True)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 200, r.text
        assert r.json()["outside_source_dir"] == 1
        db.expire_all()
        assert db.query(ActorFace).filter(ActorFace.actor_id == a.id).count() == 3

    def test_rename_then_merge_back_and_forth(self, env):
        """重命名 → 合并 → 名字仍在(用户报告的场景)。"""
        db, faces = env
        x = _add_actor(db, "ACTOR_001", faces_dir=faces)
        y = _add_actor(db, "ACTOR_002", faces_dir=faces)
        # 人工重命名 x
        db.query(Actor).filter(Actor.id == x.id).update({"name": "我起的名字"})
        db.commit()
        # 合并成 y 为主记录(x 作为源)
        r = _client().post("/actors/merge", json={"actor_ids": [y.id, x.id]})
        assert r.status_code == 200, r.text
        db.expire_all()
        t = db.query(Actor).filter(Actor.id == y.id).first()
        assert t.name == "我起的名字" or "我起的名字" in _parse_alias_list(t.aliases)

    def test_move_failure_aborts_and_rolls_back_disk(self, env, monkeypatch):
        """搬移中途失败: 中止 + DB 回滚 + 已搬走的文件也搬回(不留新错位)。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", faces_dir=faces, n_faces=1)
        b = _add_actor(db, "ACTOR_002", faces_dir=faces, n_faces=2)

        orig_rename = Path.rename
        state = {"n": 0}

        def flaky_rename(self, target):
            if "ACTOR_001" in str(target):       # 目标是主记录目录 → 搬入
                state["n"] += 1
                if state["n"] >= 2:              # 第二次搬移失败
                    raise OSError("disk full")
            return orig_rename(self, target)

        monkeypatch.setattr(Path, "rename", flaky_rename)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, b.id]})
        assert r.status_code == 500, r.text

        db.expire_all()
        # DB 未变: b 仍存在且记录仍属 b
        assert db.query(Actor).filter(Actor.id == b.id).count() == 1
        assert db.query(ActorFace).filter(ActorFace.actor_id == b.id).count() == 2
        # 磁盘: 第一个已搬走的文件被搬回源目录
        assert len(list((faces / "ACTOR_002").glob("*.jpg"))) == 2
        # 主记录目录只剩它自己的 1 张脸(源的脸没有残留进来)
        own = sorted(p.name for p in (faces / "ACTOR_001").glob("*.jpg"))
        assert own == ["face_0.jpg"], f"目标目录不应残留源的脸: {own}"

    def test_partial_group_not_found(self, env):
        """含不存在的 actor → 404(不做任何改动)。"""
        db, faces = env
        a = _add_actor(db, "ACTOR_001", faces_dir=faces)
        r = _client().post("/actors/merge", json={"actor_ids": [a.id, 999999]})
        assert r.status_code == 404
