"""审核队列的批量/单条接口（修复：前端调用的路径与服务端不一致）。

前端批量进度弹窗会调 ``POST /audit-queue/batch``（期望逐项结果），单条"删除"会调
``DELETE /audit-queue/{id}``——此前服务端分别是 ``/audit-queue/bulk``（且不返回逐项结果）
与"无该路由"，导致这些按钮全部报错。本用例锁住修好后的行为。
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
from src.db.models import AuditQueueItem, Base, DictionaryEntry


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'audit.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    return Session


def _client():
    return TestClient(M.app)


def _seed(Session):
    db = Session()
    db.add(DictionaryEntry(standard_tag="室内", aliases="房间内,indoors", category="场景", file_count=0))
    rec = AuditQueueItem(raw_label="室内场景", count=3, ai_recommendation="室内", status="pending")
    norec = AuditQueueItem(raw_label="奇怪新词", count=1, ai_recommendation=None, status="pending")
    db.add_all([rec, norec])
    db.commit()
    ids = (rec.id, norec.id)
    db.close()
    return ids


class TestBatchAudit:
    def test_confirm_adds_alias_and_skips_without_recommendation(self, env):
        rec_id, norec_id = _seed(env)

        r = _client().post("/audit-queue/batch", json={"item_ids": [rec_id, norec_id], "action": "confirm"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["processed"] == 1

        by_id = {x["item_id"]: x for x in body["results"]}
        assert by_id[rec_id]["status"] == "confirmed"
        assert by_id[norec_id]["status"] == "skipped"

        db = env()
        tag = db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == "室内").first()
        assert "室内场景" in [a.strip() for a in tag.aliases.split(",")]
        assert db.query(AuditQueueItem).filter(AuditQueueItem.id == rec_id).first().status == "confirmed"
        db.close()

    def test_reject(self, env):
        rec_id, norec_id = _seed(env)
        r = _client().post("/audit-queue/batch", json={"item_ids": [rec_id], "action": "reject"})
        assert r.json()["processed"] == 1
        db = env()
        assert db.query(AuditQueueItem).filter(AuditQueueItem.id == rec_id).first().status == "rejected"
        db.close()

    def test_delete_batch(self, env):
        rec_id, norec_id = _seed(env)
        r = _client().post("/audit-queue/batch", json={"item_ids": [rec_id, norec_id], "action": "delete"})
        results = r.json()["results"]
        assert [x["status"] for x in results] == ["deleted", "deleted"]
        db = env()
        assert db.query(AuditQueueItem).count() == 0
        db.close()

    def test_unknown_action_reports_error(self, env):
        rec_id, _ = _seed(env)
        r = _client().post("/audit-queue/batch", json={"item_ids": [rec_id], "action": "wat"})
        assert r.json()["results"][0]["status"] == "error"

    def test_empty_item_ids_processes_nothing(self, env):
        """显式空列表不得退化成"全部待审核"（否则一次误传会改动所有条目）。"""
        rec_id, norec_id = _seed(env)
        r = _client().post("/audit-queue/batch", json={"item_ids": [], "action": "reject"})
        assert r.json()["processed"] == 0
        assert r.json()["results"] == []
        db = env()
        assert db.query(AuditQueueItem).filter(AuditQueueItem.status == "pending").count() == 2
        db.close()

    def test_none_item_ids_means_all_pending(self, env):
        rec_id, norec_id = _seed(env)
        r = _client().post("/audit-queue/batch", json={"action": "reject"})
        assert r.json()["processed"] == 2


class TestDeleteSingleAuditItem:
    def test_delete_one(self, env):
        rec_id, _ = _seed(env)
        r = _client().delete(f"/audit-queue/{rec_id}")
        assert r.status_code == 200 and r.json()["deleted"] == 1
        db = env()
        assert db.query(AuditQueueItem).filter(AuditQueueItem.id == rec_id).first() is None
        db.close()

    def test_delete_missing_is_404(self, env):
        assert _client().delete("/audit-queue/99999").status_code == 404
