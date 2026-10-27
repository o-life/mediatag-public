"""标签库列表的展示口径：文件数用**实时统计**、分类包含**已创建的空分类**。

修复背景：
- 标准库的"文件数"列取自字典表里一个从不回写的字段 ⇒ **恒为 0**
- 分类列表只从"已有标签的分类"推导 ⇒ **新建的空分类在界面不可见**，无法把标签移进去
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
from src.db.models import Base, DictionaryEntry, FileEntity, TagCategory, TagFrameMapping


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'dictlist.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    return Session


def _client():
    return TestClient(M.app)


def _seed(Session):
    db = Session()
    # 字典里的 file_count 故意写成 0（历史遗留、从不回写）
    db.add(DictionaryEntry(standard_tag="室内", aliases="房间内", category="场景", file_count=0))
    for md5 in ("a" * 32, "b" * 32):
        e = FileEntity(file_md5=md5, media_type="video")
        db.add(e)
        db.flush()
        for i in range(2):  # 同一内容多帧只应算一次
            db.add(TagFrameMapping(entity_id=e.id, tag_name="室内",
                                   frame_path=f"/tmp/{md5}_{i}.jpg", video_frame_number=i))
    db.commit()
    db.close()


class TestDictionaryListing:
    def test_file_count_is_live_not_stored(self, env):
        _seed(env)
        tags = {t["name"]: t for t in _client().get("/dictionary").json()["tags"]}
        assert tags["室内"]["file_count"] == 2, "应按内容实时统计，而不是读恒为 0 的历史字段"

    def test_categories_include_declared_empty_category(self, env):
        db = env()
        db.add(TagCategory(name="全新维度"))
        db.commit()
        db.close()

        cats = _client().get("/dictionary/categories").json()["categories"]
        assert "全新维度" in cats, "新建的空分类必须可见，否则无法把标签移入"

    def test_categories_include_used_categories(self, env):
        _seed(env)
        cats = _client().get("/dictionary/categories").json()["categories"]
        assert "场景" in cats
