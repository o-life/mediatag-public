"""Tests for 状态筛选的多值解析 (src/api/main.py::resolve_status_filter)。

背景: 首页筛选原为单选, 「待处理(QUEUED)」与「处理中」被拆成两项, 无法一屏
看全"在跑管线"; 改为可多选勾选框后, 前端以逗号连接多值传给 /task?status=。
这里验证解析与过滤语义(OR)。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import STATUS_FILTER_MAP, resolve_status_filter
from src.db.models import Task


# -----------------------------------------------------------------------
# 1. resolve_status_filter: 纯函数语义
# -----------------------------------------------------------------------

class TestResolveStatusFilter:
    @pytest.mark.parametrize("value", [None, "", "all", "  "])
    def test_empty_or_all_means_no_filter(self, value):
        assert resolve_status_filter(value) == []

    def test_single_pending(self):
        assert resolve_status_filter("pending") == ["QUEUED"]

    def test_legacy_queued_alias(self):
        """兼容旧客户端口径「排队中」。"""
        assert resolve_status_filter("queued") == ["QUEUED"]

    def test_not_started_merges_pending_and_paused(self):
        assert resolve_status_filter("not_started") == ["PAUSED", "PENDING"]

    def test_processing_covers_all_running_states(self):
        got = resolve_status_filter("processing")
        assert set(got) == {
            "CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "WRITING", "PROCESSING",
        }

    def test_error_merges_error_and_failed(self):
        assert resolve_status_filter("error") == ["ERROR", "FAILED"]

    def test_multi_value_is_union(self):
        """核心用例: 待处理 + 处理中 = 整个在跑管线。"""
        got = resolve_status_filter("pending,processing")
        assert "QUEUED" in got
        assert "ANALYZING" in got
        assert "SIFTING" in got
        assert "COMPLETE" not in got
        assert "ERROR" not in got

    def test_multi_value_tolerates_whitespace(self):
        assert resolve_status_filter("pending, processing") == resolve_status_filter(
            "pending,processing"
        )

    def test_duplicate_keys_collapse(self):
        assert resolve_status_filter("pending,pending") == ["QUEUED"]

    def test_all_wins_over_others(self):
        """混入 all → 视为不过滤(避免"全部 + 某状态"这种自相矛盾的筛选)。"""
        assert resolve_status_filter("pending,all") == []

    def test_unknown_key_ignored(self):
        assert resolve_status_filter("bogus") == []

    def test_unknown_key_does_not_break_known_ones(self):
        assert resolve_status_filter("bogus,pending") == ["QUEUED"]

    def test_map_covers_frontend_options(self):
        """前端下拉的每个 value 都应能被解析(除 all)。"""
        for key in ("pending", "not_started", "processing", "complete", "error", "duplicate"):
            assert resolve_status_filter(key), f"{key} 未映射到任何数据库状态"
        assert set(STATUS_FILTER_MAP) >= {
            "pending", "not_started", "processing", "complete", "error", "duplicate",
        }


# -----------------------------------------------------------------------
# 2. 与真实查询结合: 多选确实得到"管线并集"
# -----------------------------------------------------------------------

class TestStatusFilterQuery:
    @pytest.fixture
    def session(self, tmp_path):
        """自建临时库(不用 conftest 的 db_session: 其 :memory: 与 pool_size=20 冲突)。"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.db.models import Base

        engine = create_engine(f"sqlite:///{tmp_path/'filter.db'}")
        Base.metadata.create_all(engine)
        s = sessionmaker(bind=engine)()
        yield s
        s.close()

    def _seed(self, session):
        statuses = ["QUEUED", "QUEUED", "ANALYZING", "SIFTING", "COMPLETE", "ERROR"]
        for i, st in enumerate(statuses):
            session.add(Task(file_path=f"/v/{i}.mp4", status=st))
        session.commit()

    def _count(self, session, status_value):
        q = session.query(Task)
        statuses = resolve_status_filter(status_value)
        if statuses:
            q = q.filter(Task.status.in_(statuses))
        return q.count()

    def test_pipeline_union_matches_sum_of_parts(self, session):
        self._seed(session)
        pending = self._count(session, "pending")
        processing = self._count(session, "processing")
        both = self._count(session, "pending,processing")
        assert pending == 2
        assert processing == 2
        assert both == pending + processing == 4

    def test_all_returns_everything(self, session):
        self._seed(session)
        assert self._count(session, "all") == 6

    def test_complete_excludes_pipeline(self, session):
        self._seed(session)
        assert self._count(session, "complete") == 1
