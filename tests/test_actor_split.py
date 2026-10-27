"""拆分簇的数据一致性（冒烟）：人脸被全部拆走的任务，必须从原人物的作品列表移除。

用户实测 bug：视频里某人物的人脸全部拆出后，作品仍残留在原人物的作品列表——
根因 = 拆分端点迁移人脸时只清理了 task_actors 与 actor_faces 两个来源，
漏了 Task.actor_id 直连列（作品列表查询三来源取并集）。
修复对齐合并流程先例：直连列在 remaining==0 时移交给新人物。
"""

import asyncio
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, "src")

from src.db.models import Actor, ActorFace, Task, TaskActor, Base  # noqa: E402
from src.api.routes import actors as actors_routes  # noqa: E402


@pytest.fixture()
def db_env(monkeypatch, tmp_path):
    """临时 SQLite + 打桩的 engine/session/faces_dir。"""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(actors_routes, "get_db_engine", lambda: engine)
    monkeypatch.setattr(actors_routes, "get_session", lambda engine: TestSession())
    monkeypatch.setattr(actors_routes, "_faces_dir", lambda: tmp_path / "faces")
    (tmp_path / "faces").mkdir()
    return TestSession


def _seed(db):
    """人物 A + 两个任务：T1（A 的全部 2 张脸）、T2（A 的 3 张脸中的 1 张）。"""
    a = Actor(code="ACTOR_001")
    db.add(a)
    db.flush()
    t1 = Task(file_path="/v1.mp4", status="COMPLETE", actor_id=a.id)
    t2 = Task(file_path="/v2.mp4", status="COMPLETE", actor_id=a.id)
    db.add_all([t1, t2])
    db.flush()
    faces = []
    for task, n in ((t1, 2), (t2, 1)):
        for i in range(n):
            f = ActorFace(
                actor_id=a.id,
                task_id=task.id,
                face_path=f"ACTOR_001/face_{task.id}_{i}.jpg",
            )
            db.add(f)
            faces.append(f)
    db.add_all([
        TaskActor(task_id=t1.id, actor_id=a.id),
        TaskActor(task_id=t2.id, actor_id=a.id),
    ])
    db.commit()
    return a, t1, t2, faces


def test_split_hands_over_direct_actor_id(db_env):
    """全部人脸被拆走的任务：Task.actor_id 移交给新人物，作品列表随之剔除。"""
    Session = db_env
    db = Session()
    a, t1, t2, faces = _seed(db)
    a_id, t1_id, t2_id = a.id, t1.id, t2.id
    face_ids = [f.id for f in faces if f.task_id == t1_id]
    db.close()

    result = asyncio.run(
        actors_routes.split_actor_faces(a_id, {"face_ids": face_ids})
    )
    new_id = result["new_actor_id"]

    db = Session()
    # 直连列：全拆走的 T1 移交给新人物；部分保留的 T2 不动
    assert db.get(Task, t1_id).actor_id == new_id
    assert db.get(Task, t2_id).actor_id == a_id
    # task_actors：原人物解除 T1 关联，新人物建立
    pairs = {(ta.task_id, ta.actor_id) for ta in db.query(TaskActor).all()}
    assert (t1_id, a_id) not in pairs
    assert (t1_id, new_id) in pairs
    assert (t2_id, a_id) in pairs
    db.close()

    # 作品列表查询：原人物只剩 T2；新人物得到 T1
    va = actors_routes.get_actor_videos(a_id, page=1, size=20)
    vb = actors_routes.get_actor_videos(new_id, page=1, size=20)
    assert {v["id"] for v in va["videos"]} == {t2_id}
    assert {v["id"] for v in vb["videos"]} == {t1_id}


def test_split_partial_faces_keep_video(db_env):
    """部分人脸拆走（视频里仍有原人物人脸）：作品保留在原人物列表。"""
    Session = db_env
    db = Session()
    a, t1, t2, faces = _seed(db)
    a_id, t1_id = a.id, t1.id
    face_ids = [f.id for f in faces if f.task_id == t1_id][:1]
    db.close()

    result = asyncio.run(
        actors_routes.split_actor_faces(a_id, {"face_ids": face_ids})
    )
    new_id = result["new_actor_id"]

    db = Session()
    # 视频里仍有原人物人脸 → 直连列不动
    assert db.get(Task, t1_id).actor_id == a_id
    db.close()

    va = actors_routes.get_actor_videos(a_id, page=1, size=20)
    vb = actors_routes.get_actor_videos(new_id, page=1, size=20)
    assert t1_id in {v["id"] for v in va["videos"]}
    assert t1_id in {v["id"] for v in vb["videos"]}
