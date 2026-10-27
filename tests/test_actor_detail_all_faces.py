"""Tests for /actors/{id} 返回的"全部人脸"（合并后核对用）。

背景: 合并两个演员后，若他们的脸都来自同一个视频文件，详情页原本只有
"主要特征 + 相似特征来源（每个源文件 1 张代表脸）"两块 → 合并后界面看起来
"一张都没多"，用户会误以为合并丢了脸。因此接口额外返回:
  faces.all        —— 该演员的全部人脸（封面优先、其余按质量分降序）
  faces.total      —— 总张数
  faces.file_count —— 来源视频数

这些用例直接验证真实运行库（只读 GET，不写数据）。
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/auraface/scrfd_10g_bnkps.onnx"),
    reason="需要 AuraFace 人脸模型(应用内「设置 → 环境初始化」下载)")


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB = os.path.expanduser("~/Library/Application Support/MediaTag/db/audit_cache.db")


def _db():
    if not os.path.exists(DB):
        pytest.skip("正式库不存在")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _pick_actor_with_faces():
    """挑一个脸数最多的演员（合并后往往脸很多）。"""
    con = _db()
    try:
        row = con.execute(
            "SELECT actor_id, COUNT(*) c FROM actor_faces GROUP BY actor_id "
            "ORDER BY c DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


class TestAllFacesInDetail:
    def test_all_faces_matches_db(self, client):
        """faces.total 必须等于 DB 里该演员的真实脸数（不多不少）。"""
        aid = _pick_actor_with_faces()
        if aid is None:
            pytest.skip("库中没有带人脸的演员")

        con = _db()
        try:
            db_n = con.execute(
                "SELECT COUNT(*) FROM actor_faces WHERE actor_id=?", (aid,)
            ).fetchone()[0]
            db_files = con.execute(
                "SELECT COUNT(DISTINCT file_md5) FROM actor_faces "
                "WHERE actor_id=? AND file_md5 IS NOT NULL", (aid,)
            ).fetchone()[0]
        finally:
            con.close()

        r = client.get(f"/actors/{aid}")
        assert r.status_code == 200
        faces = r.json()["faces"]

        assert faces["total"] == db_n, f"total={faces['total']} 应为 {db_n}"
        assert len(faces["all"]) == db_n
        assert faces["file_count"] == db_files

    def test_cover_is_first_and_flagged(self, client):
        """封面应排在第一张并被标记（便于核对最佳样本）。"""
        aid = _pick_actor_with_faces()
        if aid is None:
            pytest.skip("库中没有带人脸的演员")
        faces = client.get(f"/actors/{aid}").json()["faces"]
        allf = faces["all"]
        if not allf:
            pytest.skip("该演员没有人脸记录")
        assert allf[0]["is_cover"] is True, "第一张应为封面"
        assert sum(1 for f in allf if f["is_cover"]) == 1, "封面应恰好 1 张"

    def test_sorted_by_quality_desc_after_cover(self, client):
        aid = _pick_actor_with_faces()
        if aid is None:
            pytest.skip("库中没有带人脸的演员")
        allf = client.get(f"/actors/{aid}").json()["faces"]["all"]
        if len(allf) < 3:
            pytest.skip("脸数太少，排序无意义")
        rest = [f["quality_score"] or 0 for f in allf[1:]]
        assert rest == sorted(rest, reverse=True), "非封面人脸应按质量分降序"

    def test_all_faces_have_paths(self, client):
        """每条记录都带 face_path（UI 需要用它加载图片）。"""
        aid = _pick_actor_with_faces()
        if aid is None:
            pytest.skip("库中没有带人脸的演员")
        for f in client.get(f"/actors/{aid}").json()["faces"]["all"]:
            assert f.get("face_path"), "缺 face_path"

    def test_merged_actor_shows_growth(self, client):
        """合并场景: 脸数应远多于"相似特征"条数（这正是原先看不到的部分）。"""
        aid = _pick_actor_with_faces()
        if aid is None:
            pytest.skip("库中没有带人脸的演员")
        faces = client.get(f"/actors/{aid}").json()["faces"]
        if faces["total"] <= len(faces["similar"]):
            pytest.skip("该演员的脸未集中在单一来源，无法体现该差异")
        assert faces["total"] > len(faces["similar"]), (
            f"全部人脸({faces['total']}) 应多于相似特征({len(faces['similar'])})"
        )
