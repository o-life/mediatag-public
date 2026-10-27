"""标签树计数口径：按**内容(entity_id)**去重，与「按标签取作品」保持一致。

修复背景：原按 ``COUNT(DISTINCT task_id)`` 统计，而"删除作品"只把映射行的归属
（entity_id）置空、**不删行** ⇒ 删除后计数不变、列表却少一条（计数与列表不一致）。
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
import src.api.routes.tags as T
from src.db.models import Base, DictionaryEntry, FileEntity, TagFrameMapping


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'tags.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(T, "get_db_engine", lambda: engine)
    monkeypatch.setattr(T, "get_session", lambda *a, **k: Session())
    return Session


def _client():
    return TestClient(M.app)  # 标签路由挂载在主应用上（/api/tags/tree）


def _add_mappings(Session, entity_md5, tag="室内", frames=3):
    db = Session()
    entity = FileEntity(file_md5=entity_md5, media_type="video")
    db.add(entity)
    db.flush()
    for i in range(frames):
        db.add(TagFrameMapping(
            entity_id=entity.id, tag_name=tag,
            frame_path=f"/tmp/{entity_md5}_{i}.jpg", video_frame_number=i,
        ))
    db.commit()
    eid = entity.id
    db.close()
    return eid


def _count(Session, tag="室内"):
    db = Session()
    if db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == tag).first() is None:
        db.add(DictionaryEntry(standard_tag=tag, aliases="房间内", category="场景", file_count=0))
        db.commit()
    db.close()
    tree = _client().get("/api/tags/tree").json()
    for cat in tree["categories"]:
        for t in cat["tags"]:
            if t["name"] == tag:
                return t["file_count"]
    return None


class TestTagTreeCount:
    def test_counts_distinct_entities(self, env):
        _add_mappings(env, "a" * 32)
        _add_mappings(env, "b" * 32)
        assert _count(env) == 2

    def test_multiple_mappings_same_entity_count_once(self, env):
        _add_mappings(env, "c" * 32, frames=5)
        assert _count(env) == 1

    def test_count_drops_after_work_deleted(self, env):
        """删除作品 = 映射行 entity_id 置空（不清行）⇒ 计数应随之下降。"""
        keep = _add_mappings(env, "d" * 32)
        gone = _add_mappings(env, "e" * 32)
        assert _count(env) == 2

        db = env()
        for m in db.query(TagFrameMapping).filter(TagFrameMapping.entity_id == gone).all():
            m.entity_id = None
        db.commit()
        db.close()

        assert _count(env) == 1, "置空归属后不应再计入（否则与列表不一致）"
        assert keep is not None
