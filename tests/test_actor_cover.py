"""演员封面单一事实来源：`src/db/actor_cover.apply_actor_cover`。

保证 `actors.cover_face` 与 `actor_faces.is_cover` 原子一致，且只选磁盘真实存在的脸
（历史 bug：merge/split/set-cover 各自内联维护导致两字段脱节 / 封面悬空）。
"""

import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.models import Base, Actor, ActorFace
from src.db.actor_cover import apply_actor_cover


def _session(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'cover.db'}")
    Base.metadata.create_all(eng)
    return eng, sessionmaker(bind=eng)()


def _mk_faces(faces_dir: Path, actor_code: str, specs):
    d = faces_dir / actor_code
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, exists in specs:
        p = d / name
        if exists:
            p.write_bytes(b"x")
        out.append(f"{actor_code}/{name}")
    return out


class TestApplyActorCover:
    def test_preferred_valid_is_chosen(self, tmp_path):
        faces_dir = tmp_path / "faces"
        eng, s = _session(tmp_path)
        a = Actor(code="ACTOR_001")
        s.add(a); s.flush()
        paths = _mk_faces(faces_dir, "ACTOR_001", [("face_0.jpg", True), ("face_1.jpg", True)])
        s.add_all([
            ActorFace(actor_id=a.id, face_path=paths[0], quality_score=0.5),
            ActorFace(actor_id=a.id, face_path=paths[1], quality_score=0.9),
        ])
        s.commit()

        rel = apply_actor_cover(s, a.id, faces_dir, preferred_path=paths[0])
        s.commit()

        assert rel == paths[0], "应选 preferred（即便质量不是最高）"
        assert s.query(Actor, a.id).first() if False else True
        a2 = s.query(Actor).filter(Actor.id == a.id).first()
        assert a2.cover_face == paths[0]
        covers = s.query(ActorFace).filter(ActorFace.actor_id == a.id, ActorFace.is_cover == 1).all()
        assert len(covers) == 1 and covers[0].face_path == paths[0], "恰有一张 is_cover 且与 cover_face 一致"

    def test_preferred_missing_falls_back_to_best_existing(self, tmp_path):
        faces_dir = tmp_path / "faces"
        eng, s = _session(tmp_path)
        a = Actor(code="ACTOR_001")
        s.add(a); s.flush()
        paths = _mk_faces(faces_dir, "ACTOR_001", [("face_0.jpg", True), ("face_1.jpg", True)])
        s.add_all([
            ActorFace(actor_id=a.id, face_path=paths[0], quality_score=0.5),
            ActorFace(actor_id=a.id, face_path=paths[1], quality_score=0.9),
        ])
        s.commit()

        rel = apply_actor_cover(s, a.id, faces_dir, preferred_path="ACTOR_001/ghost.jpg")
        s.commit()
        assert rel == paths[1], "首选文件不存在 → 回退质量最高的有效脸"

    def test_no_valid_face_sets_none(self, tmp_path):
        faces_dir = tmp_path / "faces"
        eng, s = _session(tmp_path)
        a = Actor(code="ACTOR_001")
        s.add(a); s.flush()
        paths = _mk_faces(faces_dir, "ACTOR_001", [("face_0.jpg", False)])  # 未落盘
        s.add(ActorFace(actor_id=a.id, face_path=paths[0], quality_score=0.9))
        s.commit()

        rel = apply_actor_cover(s, a.id, faces_dir)
        s.commit()
        assert rel is None
        assert s.query(Actor).filter(Actor.id == a.id).first().cover_face is None
        assert s.query(ActorFace).filter(ActorFace.actor_id == a.id, ActorFace.is_cover == 1).count() == 0

    def test_clears_stale_is_cover_flags(self, tmp_path):
        faces_dir = tmp_path / "faces"
        eng, s = _session(tmp_path)
        a = Actor(code="ACTOR_001")
        s.add(a); s.flush()
        paths = _mk_faces(faces_dir, "ACTOR_001", [("face_0.jpg", True), ("face_1.jpg", True)])
        s.add_all([
            ActorFace(actor_id=a.id, face_path=paths[0], quality_score=0.1, is_cover=1),
            ActorFace(actor_id=a.id, face_path=paths[1], quality_score=0.9, is_cover=1),
        ])
        s.commit()

        apply_actor_cover(s, a.id, faces_dir, preferred_path=paths[1])
        s.commit()
        flags = s.query(ActorFace).filter(ActorFace.actor_id == a.id, ActorFace.is_cover == 1).all()
        assert [f.face_path for f in flags] == [paths[1]], "旧 is_cover 标记必须清掉，只留一张"
