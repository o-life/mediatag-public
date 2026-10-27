"""Tests for /videos 返回的 actor cover_face 必须是可渲染的**绝对**路径。

背景: 库里 actors.cover_face 存的是相对 faces_dir 的路径（ACTOR_001/face_0.jpg）。
前端拿到后直接 os.path.exists() 判定 → 永远失败 → 演员头像只显示占位图标，
悬浮预览也没有图可显示。

覆盖:
- 正常封面 → 解析为绝对路径
- 封面文件缺失（重建/合并后常见）→ 回退到该演员任一存在的人脸
- 该演员一张可用人脸都没有 → None（前端不显示）
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
from src.db.models import (
    Actor, ActorFace, Base, FileAnalysis, FileEntity, FileLocation,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'c.db'}")
    Base.metadata.create_all(engine)
    faces = tmp_path / "actor_faces"
    faces.mkdir()
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "_api_faces_dir", lambda: faces)
    return sessionmaker(bind=engine)(), faces


def _face_file(faces_dir: Path, rel: str) -> Path:
    p = faces_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    return p


def _video_with_actor(db, faces_dir, cover_face, face_files, create_files=True):
    """建 1 个视频 + 1 个演员（faces 为 (rel_path, is_cover, quality) 列表）。

    注意: 演员必须至少有一条 actor_faces 才会出现在 /videos（该端点按
    ActorFace.entity_id 反查演员），所以即使测"无可用封面"也要给一行记录。
    create_files=False 时只建 DB 记录、不落盘（模拟文件已被删除）。
    """
    ent = FileEntity(file_md5="md5-1", file_size=1024, media_type="video")
    db.add(ent)
    db.flush()
    db.add(FileLocation(entity_id=ent.id, file_path="/Volumes/MediaData/a.mp4",
                        file_name="a.mp4", is_primary=1))
    db.add(FileAnalysis(entity_id=ent.id, mapped_tags='["标签A","标签B"]', status="READY"))
    actor = Actor(code="ACTOR_001", name="某演员", cover_face=cover_face)
    db.add(actor)
    db.flush()
    for rel, is_cover, quality in face_files:
        if create_files:
            _face_file(faces_dir, rel)
        db.add(ActorFace(actor_id=actor.id, entity_id=ent.id, file_md5="md5-1",
                         face_path=rel, is_cover=is_cover, quality_score=quality))
    db.commit()
    return actor


def _videos():
    from fastapi.testclient import TestClient
    r = TestClient(M.app).get("/videos")
    assert r.status_code == 200, r.text
    return r.json()


class TestCoverFaceIsAbsolute:
    def test_relative_cover_becomes_absolute(self, env):
        db, faces = env
        _video_with_actor(db, faces, "ACTOR_001/face_0.jpg",
                          [("ACTOR_001/face_0.jpg", 1, 0.9)])
        v = _videos()[0]
        cover = v["actors"][0]["cover_face"]
        assert cover is not None
        assert Path(cover).is_absolute(), f"必须是绝对路径，实际 {cover}"
        assert Path(cover).exists()
        assert cover.endswith("ACTOR_001/face_0.jpg")

    def test_missing_cover_falls_back_to_existing_face(self, env):
        """cover_face 指向已删除文件 → 回退到该演员仍存在的人脸。"""
        db, faces = env
        # 封面指向 face_9（不创建），实际存在 face_1
        _video_with_actor(db, faces, "ACTOR_001/face_9.jpg",
                          [("ACTOR_001/face_1.jpg", 0, 0.5)])
        cover = _videos()[0]["actors"][0]["cover_face"]
        assert cover is not None, "应回退到存在的人脸，而不是返回死路径"
        assert Path(cover).exists()
        assert cover.endswith("face_1.jpg")

    def test_no_usable_face_returns_none(self, env):
        db, faces = env
        # 封面与唯一的人脸记录都指向已不存在的文件
        _video_with_actor(db, faces, "ACTOR_001/gone.jpg",
                          [("ACTOR_001/also_gone.jpg", 0, 0.5)], create_files=False)
        assert _videos()[0]["actors"][0]["cover_face"] is None

    def test_absolute_cover_passthrough(self, env, tmp_path):
        db, faces = env
        outside = tmp_path / "elsewhere" / "f.jpg"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"\xff\xd8\xff\xe0")
        # 需要一条 actor_faces 记录，演员才会出现在 /videos
        _video_with_actor(db, faces, str(outside),
                          [("ACTOR_001/face_0.jpg", 1, 0.9)])
        cover = _videos()[0]["actors"][0]["cover_face"]
        assert cover == str(outside)


class TestVideosPayloadForListView:
    def test_mapped_tags_present_for_tags_column(self, env):
        """列表视图的「标签」列读 mapped_tags —— 确认该键存在且有值。"""
        db, faces = env
        _video_with_actor(db, faces, None, [("ACTOR_001/face_0.jpg", 1, 0.9)])
        v = _videos()[0]
        assert v["mapped_tags"] == ["标签A", "标签B"]
        # "tags" 从来不是 /videos 的字段，列表视图读它必然空白
        assert "tags" not in v

    def test_analysis_id_and_task_id_present(self, env):
        """勾选联动依赖 id(FileAnalysis.id) 与 task_id。"""
        db, faces = env
        _video_with_actor(db, faces, None, [])
        v = _videos()[0]
        assert isinstance(v["id"], int)
        assert "task_id" in v
