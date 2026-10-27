import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from sqlalchemy import or_
from src.db.models import Task, TaskActor, Actor, TagFrameMapping, ActorFace, FileAnalysis, get_session, get_db_engine


def _faces_dir() -> Path:
    """返回 faces_dir 绝对路径（唯一解析入口，见 src/core/face_paths.py）。

    此前这里按进程 CWD 解析相对路径，与 API 侧/识别器的"回退项目根"口径不一致。
    """
    from src.core.face_paths import resolve_faces_dir

    return resolve_faces_dir()


router = APIRouter(prefix="/api/actors", tags=["actors"])


@router.post("/test-new-face-detection/{task_id}")
async def test_new_face_detection(task_id: int):
    """测试新的人脸检测模块（两阶段检测）
    
    Args:
        task_id: 任务ID
    
    Returns:
        检测结果，包括人脸片段、密集帧、检测到的人脸数量等
    """
    engine = get_db_engine()
    db = get_session(engine)
    
    try:
        # 获取任务信息
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 获取视频路径
        video_path = task.file_path
        if not video_path:
            raise HTTPException(status_code=400, detail="任务没有关联的视频文件")
        
        # 导入新的人脸检测模块
        from src.core.face_detector import FaceDetector
        
        # 初始化检测器
        detector = FaceDetector()
        
        # 执行两阶段检测
        result = detector.detect_video(
            video_path=video_path,
            task_id=task_id,
            output_dir=None,  # 不保存密集帧到磁盘（仅测试）
        )
        
        # 返回结果
        return {
            "success": True,
            "task_id": task_id,
            "video_path": video_path,
            "face_segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "face_count": seg.face_count,
                }
                for seg in result.face_segments
            ],
            "dense_frame_count": result.dense_frame_count,
            "total_faces_detected": result.total_faces_detected,
            "prescan_frame_count": result.prescan_frame_count,
            "message": f"检测到 {len(result.face_segments)} 个人脸片段，共 {result.total_faces_detected} 张人脸"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人脸检测失败: {str(e)}")
    finally:
        db.close()


@router.post("/test-new-face-recognition/{task_id}")
async def test_new_face_recognition(task_id: int):
    """测试新的人脸识别模块（特征提取）
    
    Args:
        task_id: 任务ID
    
    Returns:
        识别结果，包括提取的特征向量数量等
    """
    
    engine = get_db_engine()
    db = get_session(engine)
    
    try:
        # 获取任务信息
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 获取视频路径
        video_path = task.file_path
        if not video_path:
            raise HTTPException(status_code=400, detail="任务没有关联的视频文件")
        
        # 先执行人脸检测（保存帧到临时目录以便后续识别）
        import tempfile
        import shutil
        from src.core.face_detector import FaceDetector
        
        temp_dir = tempfile.mkdtemp(prefix="face_test_")
        try:
            detector = FaceDetector()
            detection_result = detector.detect_video(
                video_path=video_path,
                task_id=task_id,
                output_dir=temp_dir,
            )
            
            if detection_result.total_faces_detected == 0:
                return {
                    "success": False,
                    "message": "未检测到人脸，无法进行识别"
                }
            
            # 导入新的人脸识别模块
            from src.core.face_recognition import FaceRecognizer
            
            # 初始化识别器
            recognizer = FaceRecognizer()
            
            # 使用内置方法提取特征向量
            face_list = recognizer._extract_embeddings_from_detection(
                detection_result, task_id
            )
            
            # 构建返回结果
            recognition_results = []
            for face in face_list:
                embedding = face.get("embedding")
                recognition_results.append({
                    "frame_path": face.get("frame_path", ""),
                    "face_path": face.get("face_path", ""),
                    "embedding_dim": len(embedding) if embedding is not None else 0,
                    "yaw": face.get("yaw", 0),
                    "pitch": face.get("pitch", 0),
                    "roll": face.get("roll", 0),
                    "det_score": face.get("det_score", 0),
                    "quality_score": face.get("quality_score", 0),
                })
            
            # 测试封面选择
            best_cover = None
            cover_selection = None
            if face_list:
                best_cover = recognizer._select_best_cover(face_list)
                if best_cover:
                    cover_selection = {
                        "face_path": best_cover.get("face_path", ""),
                        "pose_score": best_cover.get("_pose_score", 0),
                        "cover_score": best_cover.get("_cover_score", 0),
                        "yaw": best_cover.get("yaw", 0),
                        "pitch": best_cover.get("pitch", 0),
                        "roll": best_cover.get("roll", 0),
                        "det_score": best_cover.get("det_score", 0),
                        "sharpness": best_cover.get("sharpness", 0),
                        "brightness": best_cover.get("brightness", 0),
                        "quality_score": best_cover.get("quality_score", 0),
                    }

            return {
                "success": True,
                "task_id": task_id,
                "faces_detected": detection_result.total_faces_detected,
                "embeddings_extracted": len(face_list),
                "recognition_results": recognition_results[:10],
                "best_cover": cover_selection,
                "message": f"成功提取 {len(face_list)} 个特征向量"
            }
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人脸识别失败: {str(e)}")
    finally:
        db.close()


@router.get("/{actor_id}/videos")
def get_actor_videos(
    actor_id: int,
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, Any]:
    """获取指定人物的视频列表（分页）"""
    engine = get_db_engine()
    db = get_session(engine)
    
    try:
        # 验证人物是否存在
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="人物不存在")
        
        # 获取人物关联的任务（通过 task_actors 关联表、legacy actor_id 或 actor_faces 表）
        task_ids_from_ta = [
            ta.task_id for ta in
            db.query(TaskActor).filter(TaskActor.actor_id == actor_id).all()
        ]
        
        # 从 actor_faces 表获取关联的任务
        task_ids_from_faces = [
            af.task_id for af in
            db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
            if af.task_id is not None
        ]
        
        # 合并所有关联的任务 ID
        all_task_ids = list(set(task_ids_from_ta + task_ids_from_faces))
        
        # 查询任务
        query = db.query(Task).filter(
            or_(
                Task.id.in_(all_task_ids) if all_task_ids else False,
                Task.actor_id == actor_id
            )
        )
        
        total = query.count()
        
        # 分页
        tasks = query.order_by(Task.created_at.desc()).offset(
            (page - 1) * size
        ).limit(size).all()
        
        # 构建返回数据
        videos = []
        for task in tasks:
            # 获取主路径
            primary_path = None
            if task.entity:
                for loc in task.entity.locations:
                    if loc.is_primary == 1:
                        primary_path = loc.file_path
                        break
                if not primary_path and task.entity.locations:
                    primary_path = task.entity.locations[0].file_path
            
            # 如果没有 entity，使用 task 的 file_path
            if not primary_path:
                primary_path = task.file_path
            
            # 从 FileAnalysis 读取标签（业务数据持久化核心）
            tags_list = []
            if task.entity_id:
                fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                if fa and fa.mapped_tags:
                    try:
                        tags_list = json.loads(fa.mapped_tags)
                    except (json.JSONDecodeError, TypeError):
                        tags_list = []
            
            # 获取封面路径（优先使用标签封面，其次使用人物人脸）
            cover_path = None
            
            # 1. 尝试从 tag_frame_mappings 获取封面
            first_mapping = db.query(TagFrameMapping).filter(
                TagFrameMapping.task_id == task.id
            ).first()
            if first_mapping and first_mapping.frame_path:
                fp = first_mapping.frame_path
                if Path(fp).exists():
                    try:
                        import cv2 as _cv2
                        _img = _cv2.imread(fp)
                        if _img is not None:
                            _gray = _cv2.cvtColor(_img, _cv2.COLOR_BGR2GRAY)
                            if _gray.mean() >= 10:
                                cover_path = fp
                    except Exception:
                        cover_path = fp
            
            # 2. 如果没有标签封面，使用人物在该文件中的人脸图片
            if not cover_path and task.entity_id:
                actor_face = db.query(ActorFace).filter(
                    ActorFace.actor_id == actor_id,
                    ActorFace.entity_id == task.entity_id
                ).first()
                if actor_face:
                    cover_path = actor_face.face_path
                    if cover_path and not Path(cover_path).is_absolute():
                        cover_path = str(_faces_dir() / cover_path)
            
            # 媒体类型（照片/视频）
            media_type = "photo" if (task.entity and task.entity.media_type == "photo") else "video"

            # 照片兜底：无封面时直接用照片自身作为封面
            if media_type == "photo" and not cover_path and primary_path:
                cover_path = primary_path

            videos.append({
                "id": task.id,
                "file_path": primary_path,
                "title": task.file_path.split("/")[-1] if task.file_path else None,
                "tags": tags_list,
                "status": task.status,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "cover_path": cover_path,
                "media_type": media_type,
            })
        
        return {
            "total": total,
            "page": page,
            "size": size,
            "videos": videos
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取人物视频列表失败: {str(e)}")
    finally:
        db.close()


def _quality_level(sharpness: Optional[float]) -> str:
    """把清晰度分数映射为质量等级（high/medium/low）。"""
    if sharpness is None or sharpness <= 0:
        return "unknown"
    if sharpness >= 300:
        return "high"
    if sharpness >= 150:
        return "medium"
    return "low"


@router.get("/{actor_id}/quality")
async def get_actor_quality(actor_id: int) -> Dict[str, Any]:
    """获取人物的人脸质量指标（face-quality-filter 阶段 5.1）。

    返回 avg_face_sharpness（平均人脸清晰度）、质量等级、人脸样本数。
    """
    engine = get_db_engine()
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="人物不存在")

        faces = db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
        sharpness_values = [f.sharpness for f in faces if f.sharpness and f.sharpness > 0]

        avg_sharpness = actor.avg_face_sharpness
        if avg_sharpness is None and sharpness_values:
            avg_sharpness = sum(sharpness_values) / len(sharpness_values)

        return {
            "actor_id": actor_id,
            "name": actor.name,
            "code": actor.code,
            "face_count": len(faces),
            "sharpness_samples": len(sharpness_values),
            "avg_face_sharpness": round(avg_sharpness, 2) if avg_sharpness else None,
            "quality_level": _quality_level(avg_sharpness),
        }
    finally:
        db.close()


@router.get("/{actor_id}/faces")
async def get_actor_faces(actor_id: int) -> Dict[str, Any]:
    """获取人物的全部人脸（含来源文件信息）。

    只返回关联到文件（照片/视频）的人脸——无来源文件的孤儿人脸不展示。
    """
    import os as _os
    engine = get_db_engine()
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="人物不存在")

        # 有效文件 MD5 集合（用于过滤无来源人脸）
        from sqlalchemy import text as _sa_text
        valid_md5s = set(
            r[0] for r in db.execute(
                _sa_text("SELECT file_md5 FROM file_entities WHERE file_md5 IS NOT NULL")
            ).all()
        )
        # 有实体关联的任务（task → entity → 文件），用于 task_id 来源判断
        valid_task_ids = set(
            r[0] for r in db.execute(
                _sa_text("SELECT id FROM tasks WHERE entity_id IS NOT NULL")
            ).all()
        )

        faces = db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()
        face_list = []
        skipped = 0
        for f in faces:
            # 过滤无来源人脸：entity_id / file_md5 / task_id 任一能关联到文件即有效
            has_source = (
                f.entity_id is not None
                or (f.file_md5 is not None and f.file_md5 in valid_md5s)
                or (f.task_id is not None and f.task_id in valid_task_ids)
            )
            if not has_source:
                skipped += 1
                continue

            face_path = f.face_path
            if face_path and not Path(face_path).is_absolute():
                face_path = str(_faces_dir() / face_path)
            exists = _os.path.exists(face_path) if face_path else False
            face_list.append({
                "id": f.id,
                "face_path": face_path,
                "exists": exists,
                "yaw": f.yaw,
                "pitch": f.pitch,
                "roll": f.roll,
                "sharpness": f.sharpness,
                "quality_score": f.quality_score,
                "is_cover": f.is_cover,
                "file_md5": f.file_md5,
                "entity_id": f.entity_id,
                "task_id": f.task_id,
            })

        return {
            "actor_id": actor_id,
            "name": actor.name,
            "code": actor.code,
            "total_faces": len(faces),
            "displayed_faces": len(face_list),
            "skipped_orphan": skipped,
            "faces": face_list,
        }
    finally:
        db.close()


@router.post("/{actor_id}/split")
async def split_actor_faces(actor_id: int, request: dict) -> Dict[str, Any]:
    """把选中的一些人脸拆分为新的人物（手动拆分簇）。

    body: {"face_ids": [1, 2, 3]}
    - 校验所有 face_ids 属于该 actor
    - 创建新 Actor（代码自动生成 ACTOR_XXX）
    - 把选中人脸迁移到新 actor
    - 更新 task_actors 关联（如果该 actor 关联的任务现在只剩新 actor 的人脸）
    """
    face_ids = request.get("face_ids") or []
    if not face_ids or not isinstance(face_ids, list):
        raise HTTPException(status_code=400, detail="face_ids 不能为空")

    engine = get_db_engine()
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="人物不存在")

        faces = db.query(ActorFace).filter(
            ActorFace.id.in_(face_ids), ActorFace.actor_id == actor_id
        ).all()
        if not faces:
            raise HTTPException(status_code=400, detail="没有属于该人物的有效人脸")

        # 新建人物: 先落库拿自增 id, 再由 id 派生编号(code = ACTOR_{id:03d})。
        # 与 FaceRecognizer.create_actor 一致 —— 编号从属记录号, 永久一致且并发无冲突。
        import uuid as _uuid
        new_actor = Actor(code=f"__pending__{_uuid.uuid4().hex}")
        db.add(new_actor)
        db.flush()
        new_code = f"ACTOR_{new_actor.id:03d}"
        new_actor.code = new_code
        db.flush()

        # 物理搬移选中人脸的图片文件到新人物目录（否则新人物目录为空、
        # 文件仍留在原人物目录 → 目录与 DB 记录不一致）
        faces_root = _faces_dir()
        new_dir = faces_root / new_code
        new_dir.mkdir(parents=True, exist_ok=True)
        next_idx = 0
        for p in new_dir.glob("face_*.jpg"):
            try:
                next_idx = max(next_idx, int(p.stem.split("_")[1]) + 1)
            except (IndexError, ValueError):
                continue

        moved_count = 0
        for f in faces:
            src = Path(f.face_path)
            if not src.is_absolute():
                src = faces_root / src
            # 只搬移确实位于原人物目录内的文件（来源校验）
            old_dir = faces_root / actor.code
            try:
                belongs = src.resolve().is_relative_to(old_dir.resolve())
            except Exception:
                belongs = False
            if src.exists() and belongs:
                dst = new_dir / f"face_{next_idx}.jpg"
                while dst.exists():
                    next_idx += 1
                    dst = new_dir / f"face_{next_idx}.jpg"
                src.rename(dst)
                try:
                    f.face_path = str(dst.relative_to(faces_root))
                except ValueError:
                    f.face_path = dst.name
                next_idx += 1
                moved_count += 1
            f.actor_id = new_actor.id

        # 更新 task_actors：被拆走的任务的人物关联
        affected_task_ids = set(f.task_id for f in faces if f.task_id)
        for tid in affected_task_ids:
            # 如果该任务仍有原人物的人脸，保留；否则移除 task_actors 中原人物关联
            remaining = db.query(ActorFace).filter(
                ActorFace.actor_id == actor_id, ActorFace.task_id == tid
            ).count()
            if remaining == 0:
                db.query(TaskActor).filter(
                    TaskActor.task_id == tid, TaskActor.actor_id == actor_id
                ).delete()
                # 直连列移交：该任务的原人物人脸已全部拆走时，Task.actor_id 若仍指向
                # 原人物，作品列表查询（Task.actor_id == actor_id 直连分支）会残留该
                # 作品。与合并流程（main.py 合并分支改指存续人物）一致，移交给新人物。
                task_row = db.query(Task).filter(Task.id == tid).first()
                if task_row and task_row.actor_id == actor_id:
                    task_row.actor_id = new_actor.id
            # 确保新人物关联该任务
            if not db.query(TaskActor).filter(
                TaskActor.task_id == tid, TaskActor.actor_id == new_actor.id
            ).first():
                db.add(TaskActor(task_id=tid, actor_id=new_actor.id))

        # 清理原人物目录中已搬走/不再属于它的孤儿文件，
        # 并刷新两个人物的 sample_count / 封面
        from sqlalchemy import func as _sa_func
        actor.sample_count = db.query(_sa_func.count(ActorFace.id)).filter(
            ActorFace.actor_id == actor.id
        ).scalar() or 0
        new_actor.sample_count = db.query(_sa_func.count(ActorFace.id)).filter(
            ActorFace.actor_id == new_actor.id
        ).scalar() or 0

        # 新人物封面统一由共享 helper 维护（只选磁盘真实存在的脸）
        from src.db.actor_cover import apply_actor_cover
        apply_actor_cover(db, new_actor.id, _faces_dir())

        db.commit()
        # 旧人物目录收敛（删除已搬走文件留下的孤儿/空目录）。
        # 注意: db.commit() 已在上方执行——此后属"可选收尾", 任何异常都不得让接口报错。
        # 历史 bug: 此处引用了未在本函数导入的 _sa_text → NameError → 前端收到 500,
        # 用户误以为拆分失败, 但数据其实已提交(2026-09-15 actor 1842→1963 实测)。
        try:
            from sqlalchemy import text as _sa_text
            old_dir = faces_root / actor.code
            if old_dir.exists():
                _db_paths = {
                    r[0] for r in db.execute(
                        _sa_text("SELECT face_path FROM actor_faces WHERE actor_id=:aid"),
                        {"aid": actor.id},
                    ).fetchall() if r[0]
                }
                for _p in old_dir.glob("face_*.jpg"):
                    _rel = str(_p.relative_to(faces_root))
                    if _rel not in _db_paths:
                        _p.unlink()
                if not list(old_dir.glob("*.jpg")):
                    old_dir.rmdir()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "拆分后旧目录收敛失败(不影响已提交的拆分结果): %s", e
            )

        return {
            "status": "success",
            "new_actor_id": new_actor.id,
            "new_actor_code": new_code,
            "moved_faces": moved_count,
            "selected_faces": len(faces),
        }
    finally:
        db.close()
