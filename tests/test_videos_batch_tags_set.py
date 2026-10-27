"""Tests for /api/videos/batch-tags —— 整表设置标签（标签编辑的唯一入口）。

该接口只有"整表覆盖"一种语义：
- 不做"添加"：标签需定位到对应帧（tag_frame_mappings），单独追加文字没有意义；
- "替换"同理；**删除 = 传一个不含该标签的列表**。
因此请求体不设 action 等判别字段（历史上 add/remove/replace 已删除）。
"""

import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.api.main as M
import src.api.routes.videos as V
from src.db.models import Base, FileAnalysis, FileEntity, FileLocation


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(V, "get_db_engine", lambda: engine)
    return Session()


_seq = iter(range(1, 10_000))


def _analysis(db, tags):
    n = next(_seq)  # 不能用 id()：对象被回收后可能复用同一 id → md5 撞唯一约束
    ent = FileEntity(file_md5=f"md5-{n}", file_size=1024, media_type="video")
    db.add(ent)
    db.flush()
    db.add(FileLocation(entity_id=ent.id, file_path=f"/x/a{n}.mp4",
                        file_name=f"a{n}.mp4", is_primary=1))
    fa = FileAnalysis(entity_id=ent.id,
                      mapped_tags=json.dumps(tags, ensure_ascii=False) if tags is not None else None,
                      status="READY")
    db.add(fa)
    db.commit()
    return fa


def _client():
    from fastapi.testclient import TestClient
    return TestClient(M.app)


def _tags_of(db, aid):
    db.expire_all()
    fa = db.query(FileAnalysis).filter(FileAnalysis.id == aid).first()
    return json.loads(fa.mapped_tags) if fa.mapped_tags else []


class TestSetTags:
    def test_overwrites_whole_list(self, env):
        db = env
        fa = _analysis(db, ["旧甲", "旧乙", "旧丙"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["全新"],
        })
        assert r.status_code == 200, r.text
        assert r.json()["updated_count"] == 1
        assert _tags_of(db, fa.id) == ["全新"], "应整体覆盖，而非增量"

    def test_preserves_given_order(self, env):
        db = env
        fa = _analysis(db, ["a", "b"])
        _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["丙", "甲", "乙"],
        })
        assert _tags_of(db, fa.id) == ["丙", "甲", "乙"]

    def test_dedupes_and_drops_blank(self, env):
        db = env
        fa = _analysis(db, ["x"])
        _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["甲", "甲", "  ", "", "甲 "],
        })
        assert _tags_of(db, fa.id) == ["甲"]

    def test_can_clear_all_tags(self, env):
        """删除标签 = 传不含它的列表；传空即清空。"""
        db = env
        fa = _analysis(db, ["甲", "乙"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": [],
        })
        assert r.status_code == 200
        assert _tags_of(db, fa.id) == [], "应能清空"

    def test_removing_one_tag_is_just_omitting_it(self, env):
        """删单个标签：传其余标签即可（无需专门的 remove 动作）。"""
        db = env
        fa = _analysis(db, ["甲", "乙", "丙"])
        _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["甲", "丙"],
        })
        assert _tags_of(db, fa.id) == ["甲", "丙"]

    def test_tags_defaults_to_empty(self, env):
        """省略 tags 视为空列表（等价于清空），不应 422。"""
        db = env
        fa = _analysis(db, ["甲"])
        r = _client().post("/api/videos/batch-tags", json={"analysis_ids": [fa.id]})
        assert r.status_code == 200, r.text
        assert _tags_of(db, fa.id) == []

    def test_unchanged_list_reports_zero_updates(self, env):
        db = env
        fa = _analysis(db, ["甲", "乙"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["甲", "乙"],
        })
        assert r.json()["updated_count"] == 0, "内容未变不应计为更新"

    def test_applies_to_all_given_ids(self, env):
        db = env
        a = _analysis(db, ["1"])
        b = _analysis(db, ["2"])
        _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [a.id, b.id], "tags": ["同一套"],
        })
        assert _tags_of(db, a.id) == ["同一套"]
        assert _tags_of(db, b.id) == ["同一套"]

    def test_null_mapped_tags_treated_as_empty(self, env):
        db = env
        fa = _analysis(db, None)
        _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "tags": ["甲"],
        })
        assert _tags_of(db, fa.id) == ["甲"]

    def test_unknown_id_reported_not_fatal(self, env):
        db = env
        fa = _analysis(db, ["甲"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id, 999999], "tags": ["新"],
        })
        body = r.json()
        assert body["failed_ids"] == [999999]
        assert body["success"] is False
        assert _tags_of(db, fa.id) == ["新"], "有效的那个仍应写入"


class TestRemovedLegacyFields:
    """历史回归: add/remove/replace 动作已删除，接口只做整表覆盖。

    陈旧调用方若仍传 action/old_tag/new_tag，必须**明确报错**而不是被静默忽略 ——
    否则「想 add」会变成「整表覆盖」，把原标签清掉（破坏性）。
    """

    def test_action_field_rejected(self, env):
        db = env
        fa = _analysis(db, ["甲"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "action": "add", "tags": ["乙"],
        })
        assert r.status_code == 422, f"应拒绝而非静默覆盖: {r.text}"
        assert _tags_of(db, fa.id) == ["甲"], "数据不得被改动"

    def test_old_and_new_tag_rejected(self, env):
        db = env
        fa = _analysis(db, ["甲"])
        r = _client().post("/api/videos/batch-tags", json={
            "analysis_ids": [fa.id], "old_tag": "甲", "new_tag": "乙",
        })
        assert r.status_code == 422
        assert _tags_of(db, fa.id) == ["甲"]

    def test_handler_has_no_legacy_actions(self):
        """源码层确认三个旧分支已删除（防回退）。"""
        import inspect
        src = inspect.getsource(V.set_video_tags)
        for legacy in ('"add"', '"remove"', '"replace"'):
            assert legacy not in src, f"仍残留旧动作: {legacy}"
