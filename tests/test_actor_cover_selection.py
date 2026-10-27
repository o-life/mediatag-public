"""Tests for 「人工指定演员封面」：从详情页的"全部人脸"里选一张设为封面。

覆盖:
- POST /actors/{id}/cover 正常设置：cover_face 与 is_cover 同步且唯一
- 切换封面会清掉旧标记（不出现两个封面）
- 非法 face_id / 不属于该演员 / 文件缺失 → 400，且不改动数据
- 详情接口 faces.all 与 faces.similar 都带 id（供前端点击引用）
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
from src.db.models import Actor, ActorFace, Base


@pytest.fixture
def env(tmp_path, monkeypatch):
    """临时 DB + 临时 faces_dir，并把模块级依赖指过去。"""
    engine = create_engine(f"sqlite:///{tmp_path/'c.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    faces = tmp_path / "actor_faces"
    faces.mkdir()

    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "_api_faces_dir", lambda: faces)
    return Session(), faces


def _actor_with_faces(db, faces_dir, code="ACTOR_001", n=3):
    a = Actor(code=code, cover_face=None)
    db.add(a)
    db.flush()
    d = faces_dir / code
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        f = d / f"face_{i}.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        af = ActorFace(actor_id=a.id, face_path=f"{code}/face_{i}.jpg",
                       file_md5="m1", is_cover=0, quality_score=0.5)
        db.add(af)
        rows.append(af)
    db.commit()
    return a, rows


def _client():
    from fastapi.testclient import TestClient
    return TestClient(M.app)


class TestSetActorCoverEndpoint:
    def test_sets_cover_and_syncs_flag(self, env):
        db, faces = env
        a, rows = _actor_with_faces(db, faces)
        r = _client().post(f"/actors/{a.id}/cover", json={"face_id": rows[2].id})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cover_face"] == "ACTOR_001/face_2.jpg"

        db.expire_all()
        actor = db.query(Actor).filter(Actor.id == a.id).first()
        assert actor.cover_face == "ACTOR_001/face_2.jpg"
        covers = db.query(ActorFace).filter(
            ActorFace.actor_id == a.id, ActorFace.is_cover == 1).all()
        assert len(covers) == 1, "封面必须唯一"
        assert covers[0].id == rows[2].id

    def test_switching_cover_clears_previous(self, env):
        db, faces = env
        a, rows = _actor_with_faces(db, faces)
        c = _client()
        assert c.post(f"/actors/{a.id}/cover", json={"face_id": rows[0].id}).status_code == 200
        assert c.post(f"/actors/{a.id}/cover", json={"face_id": rows[1].id}).status_code == 200

        db.expire_all()
        covers = db.query(ActorFace).filter(
            ActorFace.actor_id == a.id, ActorFace.is_cover == 1).all()
        assert len(covers) == 1
        assert covers[0].id == rows[1].id
        assert db.query(Actor).filter(Actor.id == a.id).first().cover_face == "ACTOR_001/face_1.jpg"

    def test_face_not_belonging_to_actor_rejected(self, env):
        db, faces = env
        a1, rows1 = _actor_with_faces(db, faces, code="ACTOR_001", n=2)
        a2, rows2 = _actor_with_faces(db, faces, code="ACTOR_002", n=2)
        r = _client().post(f"/actors/{a1.id}/cover", json={"face_id": rows2[0].id})
        assert r.status_code == 400
        db.expire_all()
        assert db.query(Actor).filter(Actor.id == a1.id).first().cover_face is None

    def test_unknown_face_id_rejected(self, env):
        db, faces = env
        a, _ = _actor_with_faces(db, faces)
        r = _client().post(f"/actors/{a.id}/cover", json={"face_id": 999999})
        assert r.status_code == 400

    def test_unknown_actor_rejected(self, env):
        db, faces = env
        r = _client().post("/actors/999999/cover", json={"face_id": 1})
        assert r.status_code == 400

    def test_missing_image_file_rejected(self, env):
        db, faces = env
        a, rows = _actor_with_faces(db, faces)
        (faces / "ACTOR_001" / "face_1.jpg").unlink()
        r = _client().post(f"/actors/{a.id}/cover", json={"face_id": rows[1].id})
        assert r.status_code == 400
        assert "不存在" in r.json()["detail"]

    def test_missing_field_is_422(self, env):
        db, faces = env
        a, _ = _actor_with_faces(db, faces)
        assert _client().post(f"/actors/{a.id}/cover", json={}).status_code == 422


class TestDetailFacesCarryId:
    def test_all_and_similar_have_id(self, env):
        """前端点击需要 id，两个区块都必须带。"""
        db, faces = env
        a, rows = _actor_with_faces(db, faces)
        # 先设一个封面，确保 is_cover 标记能回读
        _client().post(f"/actors/{a.id}/cover", json={"face_id": rows[0].id})

        r = _client().get(f"/actors/{a.id}")
        assert r.status_code == 200, r.text
        f = r.json()["faces"]

        assert f["all"], "应有全部人脸"
        for item in f["all"]:
            assert isinstance(item.get("id"), int), f"缺 id: {item}"
            assert "is_cover" in item
        assert f["all"][0]["is_cover"] is True, "封面应排在最前"

        similar = f["similar"]
        if similar:                      # 依赖 file_md5 关联任务，可能为空
            for item in similar:
                assert isinstance(item.get("id"), int), f"similar 缺 id: {item}"
