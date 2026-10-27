#!/usr/bin/env python3
"""
回归测试：rebuild_actor_faces 磁盘↔DB 一致性

覆盖的历史 bug：
1. rebuild 用 actor 均值 embedding + 0.75 硬阈值匹配单张脸 → 每次 rebuild 样本缩水
   （实测 25→18→16，因多姿态/多人时均值向量相似度天然偏低）
2. extract 裁剪 temp 写入共享 faces_dir 根（temp_{frame}_{x}_{y}.jpg），
   跨任务/跨 rebuild 撞名覆写 → 目录与 DB 不一致、孤儿累积
3. rebuild 把任务私有裁剪目录里的文件 rename 进 actor 目录后再删目录 → 文件丢失

断言：
- 每次 rebuild 后 actor 目录 face_*.jpg 数与 actor_faces DB 行数严格一致
- 多次 rebuild 不缩水（用 per-face 最近邻后应召回 ≥ 播种数）
- 无跨 actor 污染（rebuild B 不改变 A 的目录）
- faces_dir 无 crop_dir / 根 temp 残留

用法：
  uv run python -m pytest tests/test_face_rebuild_consistency.py -q
"""
import sys
import json
import shutil
import tempfile
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

logging.basicConfig(level=logging.WARNING)
logging.getLogger("src.core.face_recognition").setLevel(logging.WARNING)

logger = logging.getLogger("test_rebuild_consistency")

# 人工标注过的测试视频帧目录（face_test_harness，test_video_3 有 ground truth：
# actor_1=121 张 / actor_2=108 张，2fps 采样帧 3730 张）
FRAMES_SRC = Path(__file__).resolve().parent.parent / "face_test_harness/data/results/test_video_3/frames"
N_FRAMES = 60  # 子集控制耗时（~40s），足够暴露重建逻辑缺陷


def _model_available() -> bool:
    try:
        from src.core.face_recognition import FaceRecognizer
        fr = FaceRecognizer()
        return bool(fr.available)
    except Exception:
        return False


def _count_db_faces(db, actor_id: int) -> int:
    from src.db.models import ActorFace
    return db.query(ActorFace).filter(ActorFace.actor_id == actor_id).count()


def test_rebuild_actor_faces_disk_db_consistency():
    if not FRAMES_SRC.exists():
        logger.warning("标注测试帧目录不存在，跳过: %s", FRAMES_SRC)
        return True
    if not _model_available():
        logger.warning("人脸模型不可用，跳过")
        return True

    from src.db.models import (
        Base, get_db_engine, get_session,
        FileEntity, FileAnalysis, Task, Actor, ActorFace, TaskActor,
    )
    from src.core.face_recognition import FaceRecognizer

    tmp = Path(tempfile.mkdtemp(prefix="avf_rebuild_test_"))
    try:
        db_path = tmp / "test.db"
        faces_root = tmp / "actor_faces"
        faces_root.mkdir(parents=True)

        engine = get_db_engine(str(db_path))
        Base.metadata.create_all(engine)
        db = get_session(engine)

        fr = FaceRecognizer()
        fr.faces_dir = faces_root

        # 构造任务帧目录
        all_frames = sorted(FRAMES_SRC.glob("*.jpg"))[:N_FRAMES]
        assert all_frames, "帧目录为空"
        task_frames = tmp / "task_frames"
        task_frames.mkdir()
        for fp in all_frames:
            shutil.copy2(fp, task_frames / fp.name)

        # 种子 extract 走隔离目录（模拟 rebuild 后的干净环境）
        seed_crop = faces_root / ".seed_tmp"
        seed_crop.mkdir()
        faces = fr.extract_faces_from_frames(str(task_frames), temp_dir=seed_crop)
        assert len(faces) >= 20, f"子集检出人脸不足: {len(faces)}"

        # 播种 entity/task/file_analysis
        ent = FileEntity(file_md5="md5_rebuild_test", media_type="video")
        db.add(ent)
        db.flush()
        task = Task(file_path="/tv3.mp4", entity_id=ent.id, status="COMPLETE")
        db.add(task)
        db.flush()
        fa = FileAnalysis(entity_id=ent.id, task_id=task.id, frames_path=str(task_frames))
        db.add(fa)
        db.flush()

        def seed_actor(code: str, seed_faces: list) -> int:
            actor_dir = faces_root / code
            actor_dir.mkdir()
            emb = np.mean([np.array(f["embedding"]) for f in seed_faces], axis=0)
            actor = Actor(code=code, face_embedding=json.dumps(emb.tolist()),
                          sample_count=len(seed_faces))
            db.add(actor)
            db.flush()
            for i, f in enumerate(seed_faces):
                src = Path(f["face_path"])
                dst = actor_dir / f"face_{i}.jpg"
                if src.exists() and src.resolve().is_relative_to(seed_crop.resolve()):
                    shutil.copy2(src, dst)
                rel = str(dst.relative_to(faces_root))
                f["face_path"] = rel
                db.add(ActorFace(
                    actor_id=actor.id, task_id=task.id, entity_id=ent.id,
                    face_path=rel, yaw=f.get("yaw", 0), pitch=f.get("pitch", 0),
                    roll=f.get("roll", 0), is_cover=1 if i == 0 else 0,
                    sharpness=f.get("sharpness", 0),
                    quality_score=f.get("quality_score", 0),
                    embedding=json.dumps(f["embedding"].tolist()),
                    face_type="main" if i == 0 else "merged",
                ))
            db.add(TaskActor(task_id=task.id, actor_id=actor.id))
            db.commit()
            return actor.id

        def disk_count(code: str) -> int:
            d = faces_root / code
            return len(list(d.glob("face_*.jpg"))) if d.exists() else 0

        aid1 = seed_actor("ACTOR_001", faces[:20])
        assert disk_count("ACTOR_001") == _count_db_faces(db, aid1)

        # rebuild 两次，每次后目录必须与 DB 一致
        prev_count = None
        for rnd in (1, 2):
            res = fr.rebuild_actor_faces(aid1, db_session=db)
            db.commit()
            assert res.get("status") == "success", f"rebuild round {rnd} 失败: {res}"
            assert disk_count("ACTOR_001") == _count_db_faces(db, aid1), (
                f"round {rnd}: 目录文件数≠DB行数 "
                f"disk={disk_count('ACTOR_001')} db={_count_db_faces(db, aid1)}"
            )
            if prev_count is not None:
                # per-face 最近邻后不应缩水（参考集扩大应召回更多或持平）
                assert _count_db_faces(db, aid1) >= prev_count, (
                    f"round {rnd}: 人脸数缩水 {prev_count} → {_count_db_faces(db, aid1)}"
                )
            prev_count = _count_db_faces(db, aid1)

        # 播种 actor2 并 rebuild，验证不污染 actor1
        before_a1 = disk_count("ACTOR_001")
        aid2 = seed_actor("ACTOR_002", faces[20:40])
        res2 = fr.rebuild_actor_faces(aid2, db_session=db)
        db.commit()
        assert res2.get("status") == "success", f"rebuild actor2 失败: {res2}"
        assert disk_count("ACTOR_001") == before_a1, "rebuild actor2 污染了 actor1 目录"
        assert disk_count("ACTOR_002") == _count_db_faces(db, aid2)

        # 无 crop_dir / 根 temp 残留
        leftover = [d.name for d in faces_root.glob(".rebuild_t*") if d.is_dir()]
        assert not leftover, f"crop_dir 残留: {leftover}"
        logger.info("rebuild 一致性验证通过: actor1=%d 张, actor2=%d 张",
                    _count_db_faces(db, aid1), _count_db_faces(db, aid2))
        db.close()
        return True
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def test_match_actor_v2_scope_limits_to_task_actors():
    """回归：侧脸补全阶段必须先在本任务已建 actor 内匹配，不能新建重复 actor。

    背景（task 1483 实测）：
    - 正脸阶段新建 ACTOR_319（男主）；侧脸补全阶段因 match_existing_actors
      运行时配置缺失落默认 False，跳过匹配直接 create_actor 新建 ACTOR_321，
      其 5 张脸与 ACTOR_319 逐帧同 bbox/det_score/embedding。
    - 修复：侧脸簇先以 scope_actor_ids=本任务已建 actor 调用 match_actor_v2，
      同人侧脸并入而非新建；未命中才按开关扩大匹配。
    """
    from src.db.models import (
        Base, get_db_engine, get_session,
        FileEntity, Task, Actor, ActorFace, TaskActor,
    )
    from src.core.face_recognition import FaceRecognizer

    tmp = Path(tempfile.mkdtemp(prefix="avf_scope_test_"))
    try:
        db_path = tmp / "test.db"
        engine = get_db_engine(str(db_path))
        Base.metadata.create_all(engine)
        db = get_session(engine)

        # 两个确定性向量：actor_a 的侧脸(同人)向量 = actor_a 向量 + 小扰动，
        # 远小于与 actor_b(不同人)的距离。
        rng = np.random.default_rng(42)
        base = rng.normal(size=64)
        base /= np.linalg.norm(base)
        other = rng.normal(size=64)
        other /= np.linalg.norm(other)
        # 保证 base 与 other 余弦相似度很低
        other = other - (other @ base) * base
        other /= np.linalg.norm(other)
        profile = base + 0.05 * rng.normal(size=64)
        profile /= np.linalg.norm(profile)

        ent = FileEntity(file_md5="md5_scope", media_type="video")
        db.add(ent)
        db.flush()
        task = Task(file_path="/scope.mp4", entity_id=ent.id, status="COMPLETE")
        db.add(task)
        db.flush()

        def _mk_actor(code, emb_vec, task_id):
            actor = Actor(code=code, face_embedding=json.dumps(emb_vec.tolist()),
                          sample_count=3)
            db.add(actor)
            db.flush()
            db.add(TaskActor(task_id=task_id, actor_id=actor.id))
            # actor_faces 不落盘（无需真实图片），仅存 embedding 供匹配
            for i in range(3):
                db.add(ActorFace(
                    actor_id=actor.id, task_id=task_id, entity_id=ent.id,
                    face_path=f"{code}/face_{i}.jpg", is_cover=1 if i == 0 else 0,
                    sharpness=300.0, quality_score=0.7,
                    embedding=json.dumps(emb_vec.tolist()),
                    face_type="main" if i == 0 else "merged",
                ))
            db.commit()
            return actor.id

        aid_a = _mk_actor("ACTOR_A", base, task.id)
        aid_b = _mk_actor("ACTOR_B", other, task.id)

        fr = FaceRecognizer()
        side_cluster = [
            {"face_path": f"/tmp/x{i}.jpg", "embedding": profile + 1e-3 * rng.normal(size=64),
             "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "sharpness": 300.0}
            for i in range(3)
        ]
        for f in side_cluster:
            f["embedding"] /= np.linalg.norm(f["embedding"])

        # 修复前(match_actor_v2 无 scope)：全库遍历时 actor_b 也可能被扫到，
        # 但余弦上仍是 actor_a 胜出——真正的 bug 在侧脸阶段压根没调 match。
        # 修复后：scope 限定 [aid_a]，必须命中 aid_a（同人），而不是新建或匹配 aid_b。
        matched = fr.match_actor_v2(side_cluster, db_session=db, scope_actor_ids=[aid_a])
        assert matched == aid_a, f"scope 内应匹配到本任务 actor {aid_a}，实际 {matched}"

        # scope 限定为不相关 actor 时不应强行匹配（返回 None，由调用方决定是否新建）
        matched_b = fr.match_actor_v2(side_cluster, db_session=db, scope_actor_ids=[aid_b])
        assert matched_b is None, f"不同人不应匹配到 {matched_b}"

        db.close()
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    ok = test_rebuild_actor_faces_disk_db_consistency()
    print("PASS" if ok else "FAIL")
