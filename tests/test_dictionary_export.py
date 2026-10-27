"""标准标签库导出接口：格式必须与导入解析规则一致，可"导出→编辑→导入"往返。

修复背景：设置页与标签管理页的「导出标准库」原先调用的接口在服务端不存在
（标签管理页还会把 JSON 写进 .md），导出功能整体不可用。
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
from src.db.models import Base, DictionaryEntry


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'dict.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(M, "engine", engine)
    monkeypatch.setattr(M, "get_session", lambda *a, **k: Session())
    return Session


def _client():
    return TestClient(M.app)


def _seed(Session):
    db = Session()
    db.add_all([
        DictionaryEntry(standard_tag="室内", aliases="房间内,indoors", category="场景", file_count=0),
        DictionaryEntry(standard_tag="室外", aliases="outdoors", category="场景", file_count=0),
        DictionaryEntry(standard_tag="单人", aliases="", category="人数", file_count=0),
    ])
    db.commit()
    db.close()


class TestDictionaryExport:
    def test_export_shape_matches_import_parser(self, env):
        _seed(env)
        r = _client().get("/dictionary/export")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 3
        content = body["content"]

        # 分类段头（导入用 ^\d+\.\s*\[分类\] 解析）；分类顺序取决于排序，不写死序号
        assert "[场景]" in content and "### " in content
        assert "[人数]" in content
        assert "| 标签 | 别名 | 说明 |" in content
        # 标签行（导入用 | **标签** | 别名 | 说明 | 解析）
        assert "| **室内** | 房间内, indoors | |" in content
        assert "| **单人** |  | |" in content

    def test_round_trip_restores_tags_aliases_categories(self, env):
        _seed(env)
        content = _client().get("/dictionary/export").json()["content"]

        # 清空后按导出的内容重新导入
        db = env()
        db.query(DictionaryEntry).delete()
        db.commit()
        db.close()

        r = _client().post("/dictionary/import", json={"content": content})
        assert r.status_code == 200, r.text
        assert r.json()["imported"] == 3

        db = env()
        rows = {e.standard_tag: e for e in db.query(DictionaryEntry).all()}
        db.close()
        assert set(rows) == {"室内", "室外", "单人"}
        assert rows["室内"].category == "场景"
        assert [a.strip() for a in rows["室内"].aliases.split(",")] == ["房间内", "indoors"]
        assert rows["单人"].category == "人数"
