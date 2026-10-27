"""演员列表：搜索走**服务端**（覆盖全部演员）；重命名必须**界面可见**。

修复背景：
- 搜索原先只过滤"当前页已加载的数据"，且只能越搜越窄 → 跨页的人搜不到
- 重命名只写 name，而界面显示优先 primary_name → 已有主名时"改了看不到变化"
"""

import json
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
from src.db.models import Actor, ActorFace, Base


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path/'actors.db'}")
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
        Actor(code="ACTOR_001", name="小王", primary_name="王大明", aliases=json.dumps(["Wang"])),
        Actor(code="ACTOR_002", name="小李", primary_name=None, aliases=json.dumps(["Li"])),
        Actor(code="ACTOR_003", name="老王", primary_name="王二", aliases=None),
    ])
    db.commit()
    db.close()


class TestActorSearch:
    def test_search_matches_name_or_alias(self, env):
        _seed(env)
        names = lambda q: {a["code"] for a in _client().get(f"/actors?search={q}").json()["actors"]}
        assert names("王") == {"ACTOR_001", "ACTOR_003"}, "按名称/主名模糊匹配"
        assert names("Li") == {"ACTOR_002"}, "别名也要能命中"
        assert names("王大明") == {"ACTOR_001"}

    def test_search_total_reflects_all_matches_not_current_page(self, env):
        _seed(env)
        body = _client().get("/actors?search=王&page=1&page_size=1").json()
        assert body["total"] >= 2, "total 应是全部匹配数，而不是当前页条数"
        assert len(body["actors"]) == 1

    def test_empty_search_returns_all(self, env):
        _seed(env)
        assert _client().get("/actors").json()["total"] == 3


class TestActorFuzzySearch:
    """模糊搜索：子串 / 顺序子序列 / 近似拼写（错别字、同音字）都要能命中。"""

    def _seed(self, Session):
        db = Session()
        db.add_all([
            Actor(code="F1", name="五十岚", primary_name="五十岚", aliases=None),
            Actor(code="F2", name="廖承宇", primary_name="廖承宇", aliases=None),
            Actor(code="F3", name="Cydney", primary_name=None, aliases=None),
        ])
        db.commit()
        db.close()

    def _codes(self, q):
        body = _client().get("/actors", params={"search": q}).json()
        return [a["code"] for a in body["actors"]]

    def test_substring_match(self, env):
        self._seed(env)
        assert self._codes("五十") == ["F1"], "部分字（子串）即可命中"

    def test_subsequence_match(self, env):
        self._seed(env)
        assert self._codes("五岚") == ["F1"], "顺序子序列（中间夹字）也应命中"

    def test_typo_tolerance(self, env):
        self._seed(env)
        assert self._codes("廖成宇") == ["F2"], "近似拼写（错别字/同音字）应能容错"

    def test_case_insensitive(self, env):
        self._seed(env)
        assert self._codes("cyd") == ["F3"]

    def test_total_is_all_matches(self, env):
        self._seed(env)
        assert _client().get("/actors?search=五").json()["total"] == 1


class TestActorEnglishSearch:
    """英文名：大小写不敏感、词序无关、分隔符/重音归一、拼写容错（不做拼音）。"""

    def _seed(self, Session):
        db = Session()
        db.add_all([
            Actor(code="E1", name="Cydney Jane", primary_name="Cydney Jane", aliases=None),
            Actor(code="E2", name="Taylor Swift", primary_name="Taylor Swift", aliases=None),
            Actor(code="E3", name="Jean-Luc Picard", primary_name="Jean-Luc Picard", aliases=None),
            Actor(code="E4", name="José Álvarez", primary_name="José Álvarez", aliases=None),
            Actor(code="E5", name="O'Neil", primary_name="O'Neil", aliases=None),
        ])
        db.commit()
        db.close()

    def _codes(self, q):
        return [a["code"] for a in _client().get("/actors", params={"search": q}).json()["actors"]]

    def test_case_insensitive(self, env):
        self._seed(env)
        assert "E1" in self._codes("cydn"), "大小写不敏感"

    def test_token_order_insensitive(self, env):
        self._seed(env)
        assert self._codes("jane cyd") == ["E1"], "两个词按任意顺序都要命中"
        assert "E2" in self._codes("swi tay")

    def test_separators_and_accents_normalized(self, env):
        self._seed(env)
        assert "E3" in self._codes("jean luc"), "连字符当空格"
        assert "E5" in self._codes("oneil"), "撇号归一"
        assert "E4" in self._codes("jose"), "去重音 é→e"

    def test_typo_tolerance_english(self, env):
        self._seed(env)
        assert "E1" in self._codes("cidney"), "英文拼写容错（difflib）"


class TestActorRename:
    def test_rename_updates_display_field_and_keeps_old_as_alias(self, env):
        _seed(env)
        db = env()
        aid = db.query(Actor).filter(Actor.code == "ACTOR_001").first().id
        db.close()

        r = _client().put(f"/actors/{aid}/rename", json={"name": "新名字"})
        assert r.status_code == 200, r.text

        db = env()
        a = db.query(Actor).filter(Actor.id == aid).first()
        assert a.name == "新名字"
        assert a.primary_name == "新名字", "显示优先主名，必须同步，否则界面看不到变化"
        aliases = json.loads(a.aliases) if a.aliases else []
        assert "王大明" in aliases, "旧显示名应保留为别名（信息只增不减）"
        assert "Wang" in aliases
        db.close()

    def test_rename_rejects_empty_name(self, env):
        _seed(env)
        db = env()
        aid = db.query(Actor).filter(Actor.code == "ACTOR_002").first().id
        db.close()
        assert _client().put(f"/actors/{aid}/rename", json={"name": "   "}).status_code == 400


class TestActorCoverFilter:
    """only_with_cover：total 与分页都按"过滤后可展示"的集合算（total == 实际可见数）。

    人脸管理页只展示有封面的卡片；若服务端 total 仍按全量算，就会出现
    "共 N 项"与可见卡片数/页数对不上。
    """

    def _seed(self, Session, tmp_path):
        cover_ok = tmp_path / "cover_ok.jpg"
        cover_ok.write_bytes(b"x")
        face_ok = tmp_path / "face_ok.jpg"
        face_ok.write_bytes(b"x")

        db = Session()
        a1 = Actor(code="C1", name="有封面", cover_face=str(cover_ok))
        a2 = Actor(code="C2", name="靠回退", cover_face=str(tmp_path / "missing.jpg"))
        a3 = Actor(code="C3", name="无封面", cover_face=None)
        db.add_all([a1, a2, a3])
        db.flush()
        # a2 的封面文件不存在 → 应回退到质量最高的一张真实存在的脸
        db.add(ActorFace(actor_id=a2.id, face_path=str(face_ok), quality_score=0.9, is_cover=0))
        db.commit()
        db.close()

    def test_only_with_cover_filters_total_and_pagination(self, env, tmp_path):
        self._seed(env, tmp_path)
        body = _client().get("/actors", params={"only_with_cover": "true"}).json()
        assert body["total"] == 2, "total 应只数有封面（含可回退）的演员"
        assert {a["code"] for a in body["actors"]} == {"C1", "C2"}

        page = _client().get(
            "/actors", params={"only_with_cover": "true", "page": 1, "page_size": 1}
        ).json()
        assert page["total"] == 2, "total 必须是过滤后的总数，而不是全量"
        assert len(page["actors"]) == 1

    def test_without_flag_total_unchanged(self, env, tmp_path):
        self._seed(env, tmp_path)
        assert _client().get("/actors").json()["total"] == 3

    def test_search_combined_with_cover_filter(self, env, tmp_path):
        self._seed(env, tmp_path)
        body = _client().get(
            "/actors", params={"search": "有", "only_with_cover": "true"}
        ).json()
        assert {a["code"] for a in body["actors"]} == {"C1"}


class TestActorIdsFilter:
    """ids 参数（人脸管理页「只看已勾选」）：勾选跨页，前端按 id 集合向服务端取数。

    关键：total 与分页都必须按**过滤后**的集合算（与 only_with_cover 同口径），
    否则"共 N 项"/页数与可见卡片对不上。
    """

    def _all_ids(self, Session):
        db = Session()
        ids = [a.id for a in db.query(Actor).order_by(Actor.id).all()]
        db.close()
        return ids

    def test_ids_returns_only_requested(self, env):
        _seed(env)
        ids = self._all_ids(env)
        body = _client().get("/actors", params={"ids": f"{ids[0]},{ids[2]}"}).json()
        assert {a["code"] for a in body["actors"]} == {"ACTOR_001", "ACTOR_003"}
        assert body["total"] == 2, "total 应按过滤后的集合算"

    def test_ids_pagination_over_filtered_set(self, env):
        _seed(env)
        ids = ",".join(str(i) for i in self._all_ids(env))
        page = _client().get(
            "/actors", params={"ids": ids, "page": 1, "page_size": 2}
        ).json()
        assert page["total"] == 3
        assert len(page["actors"]) == 2, "分页作用于过滤后的集合"

    def test_non_digit_chunks_ignored(self, env):
        _seed(env)
        first = self._all_ids(env)[0]
        dirty = "abc," + str(first) + ",, "
        body = _client().get("/actors", params={"ids": dirty}).json()
        assert body["total"] == 1, "脏片段应被忽略而不是 500"

    def test_empty_ids_returns_all(self, env):
        _seed(env)
        body = _client().get("/actors", params={"ids": ""}).json()
        assert body["total"] == 3, "空串等同未传，不做过滤"


class TestActorIdsEndpoint:
    """GET /actors/ids —— 供前端剔除已删除演员的"幽灵勾选"。

    关键：该静态路由必须排在 /actors/{actor_id} 之前，否则 "ids" 会被当成
    actor_id（422）；且演员被删除后其 id 应从有效列表中消失。
    """

    def test_returns_all_actor_ids(self, env):
        _seed(env)
        resp = _client().get("/actors/ids")
        assert resp.status_code == 200, "不能被 /actors/{actor_id} 参数化路由吞掉"
        assert len(resp.json()["ids"]) == 3

    def test_deleted_actor_id_disappears(self, env):
        _seed(env)
        db = env()
        victim = db.query(Actor).filter(Actor.code == "ACTOR_002").first()
        vid = victim.id
        db.delete(victim)
        db.commit()
        db.close()

        ids = _client().get("/actors/ids").json()["ids"]
        assert vid not in ids, "被删除的演员 id 必须从有效列表消失（前端据此清勾选）"
