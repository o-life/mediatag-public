"""Tests for 演员"作品数"统计 (_actor_source_stats)。

约定(用户明确): **按文件去重**; 关联口径与详情页一致
(任务关联表 ∪ 人脸表 ∪ tasks.actor_id), 悬空 task_id 不计。

覆盖历史 bug: ACTOR_122 实际 7 部、卡片曾显示 3 ——
因为旧口径只数 actor_faces.entity_id(多为 NULL)且不查 task_actors。
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import _actor_source_stats
from src.db.models import (
    Base, Actor, ActorFace, FileEntity, FileLocation, Task, TaskActor,
)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'a.db'}")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _file(db, md5, path):
    e = FileEntity(file_md5=md5, file_size=1000, duration=60.0)
    db.add(e); db.flush()
    db.add(FileLocation(entity_id=e.id, file_path=path, file_name=Path(path).name, is_primary=1))
    db.flush()
    return e


def _task(db, entity, status="COMPLETE"):
    t = Task(entity_id=entity.id, file_path=f"/v/{entity.file_md5}.mp4",
             file_md5=entity.file_md5, status=status)
    db.add(t); db.flush()
    return t


def _actor(db, code="ACTOR_T"):
    a = Actor(code=code)
    db.add(a); db.flush()
    return a


class TestActorSourceStats:
    def test_counts_files_from_task_actors(self, db):
        """task_actors 关联的作品要计数(旧口径完全忽略它 → 少算的根因)。"""
        a = _actor(db)
        ents = [_file(db, f"m{i}", f"/v/f{i}.mp4") for i in range(7)]
        for e in ents:
            t = _task(db, e)
            db.add(TaskActor(task_id=t.id, actor_id=a.id))
        db.commit()
        # 无任何 actor_faces 记录(旧口径会算出 0)
        count, files = _actor_source_stats(db, a.id)
        assert count == 7
        assert len(files) == 7

    def test_dedupes_by_file(self, db):
        """同一文件被多个任务引用 → 只算 1 部(按文件去重)。"""
        a = _actor(db)
        e = _file(db, "same_md5", "/v/same.mp4")
        for _ in range(3):
            t = _task(db, e)
            db.add(TaskActor(task_id=t.id, actor_id=a.id))
        db.commit()
        count, _ = _actor_source_stats(db, a.id)
        assert count == 1

    def test_dangling_task_ids_ignored(self, db):
        """悬空 task_id(任务已删)不计入。"""
        a = _actor(db)
        e = _file(db, "real", "/v/real.mp4")
        t = _task(db, e)
        db.add(TaskActor(task_id=t.id, actor_id=a.id))
        db.add(TaskActor(task_id=999999, actor_id=a.id))   # 不存在
        db.commit()
        count, _ = _actor_source_stats(db, a.id)
        assert count == 1

    def test_union_of_three_sources(self, db):
        """三处关联取并集且不重复计数: task_actors / actor_faces / tasks.actor_id。"""
        a = _actor(db)
        e1, e2, e3 = _file(db, "m1", "/v/1.mp4"), _file(db, "m2", "/v/2.mp4"), _file(db, "m3", "/v/3.mp4")
        t1 = _task(db, e1); t2 = _task(db, e2); t3 = _task(db, e3)
        db.add(TaskActor(task_id=t1.id, actor_id=a.id))          # 来源①
        db.add(ActorFace(actor_id=a.id, task_id=t2.id, file_md5=e2.file_md5,
                             face_path="f.jpg"))  # 来源②
        t3.actor_id = a.id                                        # 来源③
        db.commit()
        count, files = _actor_source_stats(db, a.id)
        assert count == 3
        assert set(files) == {"1.mp4", "2.mp4", "3.mp4"}

    def test_falls_back_to_md5_when_entity_missing(self, db):
        """任务无 entity_id 时按 file_md5 去重(不丢数)。"""
        a = _actor(db)
        t = Task(entity_id=None, file_path="/v/x.mp4", file_md5="only_md5", status="COMPLETE")
        db.add(t); db.flush()
        db.add(TaskActor(task_id=t.id, actor_id=a.id))
        db.commit()
        count, _ = _actor_source_stats(db, a.id)
        assert count == 1

    def test_no_relations_returns_zero(self, db):
        a = _actor(db)
        db.commit()
        count, files = _actor_source_stats(db, a.id)
        assert count == 0
        assert files == []

    def test_actor_122_scenario(self, db):
        """复刻 ACTOR_122: 3 个人脸任务 + 4 个仅 task_actors 的作品 → 7 部。"""
        a = _actor(db, "ACTOR_122")
        # 3 个有人脸记录的任务(entity_id 为空 —— 历史数据形态)
        for i in range(3):
            e = _file(db, f"face{i}", f"/v/face{i}.mp4")
            t = _task(db, e)
            db.add(ActorFace(actor_id=a.id, task_id=t.id, entity_id=None,
                             file_md5=e.file_md5, face_path=f"f{i}.jpg"))
        # 4 个仅通过 task_actors 关联、无人脸记录
        for i in range(4):
            e = _file(db, f"ta{i}", f"/v/ta{i}.mp4")
            t = _task(db, e)
            db.add(TaskActor(task_id=t.id, actor_id=a.id))
        db.commit()
        count, files = _actor_source_stats(db, a.id)
        assert count == 7, f"应 7 部, 实际 {count}"
        assert len(files) == 7
