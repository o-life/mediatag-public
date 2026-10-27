import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Dict, Any
from src.db.models import Task, FileEntity, FileAnalysis, ActorFace, Actor, TagFrameMapping, get_session, get_db_engine

router = APIRouter(prefix="/api/videos", tags=["videos"])


class BatchUpdateTagsRequest(BaseModel):
    video_ids: List[int]
    tag_name: str


class BatchDeleteTagsRequest(BaseModel):
    video_ids: List[int]
    tag_name: str


class BatchDeleteRequest(BaseModel):
    analysis_ids: List[int]


class BatchTagsRequest(BaseModel):
    """整表设置若干文件的标签列表（标签编辑的唯一入口）。"""

    # 标签编辑只有"整表覆盖"一种语义，故不设 action。若有陈旧调用方仍传
    # action/old_tag 等字段，这里直接 422 报错 —— 绝不静默忽略后把标签覆盖掉
    # （本接口是破坏性的：传什么就以什么为准）。
    model_config = ConfigDict(extra="forbid")

    analysis_ids: List[int]
    tags: List[str] = Field(default_factory=list)


@router.get("/by-tag")
def get_videos_by_tag(
    tag: str = Query(..., description="标签名称"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, Any]:
    """获取指定标签下的视频列表（分页），封面使用该标签对应的关键帧。

    性能：批量查询（N+1 已消——原逐行懒加载 entity/locations/actor，
    数据量大时按 tag 检索会缓冲分钟级，2026-09-29 实测）。"""
    from src.db.models import TagFrameMapping, FileAnalysis, FileEntity
    from sqlalchemy.orm import joinedload

    engine = get_db_engine()
    db = get_session(engine)

    try:
        # 标签 → 实体 ID 集（单查询）
        tagged_entity_ids = [
            row[0] for row in db.query(TagFrameMapping.entity_id).filter(
                TagFrameMapping.tag_name == tag,
                TagFrameMapping.entity_id.isnot(None)
            ).distinct().all()
        ]
        if not tagged_entity_ids:
            return {"total": 0, "page": page, "size": size, "videos": []}

        # 实体 → 文件分析（批量预载 entity.locations）
        analyses = (db.query(FileAnalysis)
                    .options(joinedload(FileAnalysis.entity)
                             .selectinload(FileEntity.locations))
                    .filter(FileAnalysis.entity_id.in_(tagged_entity_ids))
                    .all())
        total = len(analyses)

        # N+1 批量预取：帧映射 / 人脸脸面 / 人物（各一次查询，循环内零查询）
        mappings_by_entity = {}

        for m in db.query(TagFrameMapping).filter(
                TagFrameMapping.entity_id.in_(tagged_entity_ids),
                TagFrameMapping.tag_name == tag).all():
            mappings_by_entity.setdefault(m.entity_id, []).append(m)
        faces_by_entity = {}
        for af in db.query(ActorFace).filter(
                ActorFace.entity_id.in_(tagged_entity_ids)).all():
            faces_by_entity.setdefault(af.entity_id, []).append(af)
        actor_id_set = {af.actor_id for rows in faces_by_entity.values()
                        for af in rows if af.actor_id}
        actors_by_id = db.query(Actor).filter(
            Actor.id.in_(actor_id_set or {0})).all()
        paginated = analyses[(page - 1) * size: (page - 1) * size + size]

        videos = []
        # 懒导入防循环（main → routes；此函数内已有同款先例）
        from src.api.main import _actor_cover_abs
        _cover_abs_cache = {}  # 人物封面解析缓存（同人物跨行复用）
        for fa in paginated:
            entity = fa.entity
            if not entity:
                continue
            primary_path = None
            if entity.locations:
                loc = next((l for l in entity.locations if l.is_primary == 1),
                           entity.locations[0])
                primary_path = loc.file_path

            # 解析标签
        # 解析标签
            tags_list = []
            if fa.mapped_tags:
                try:
                    tags_list = json.loads(fa.mapped_tags)
                except (json.JSONDecodeError, TypeError):
                    tags_list = []

            # 该标签对应的关键帧（帧集已批量预取，避免 N+1）
            cover_path = None
            video_frame_number = None

            tag_mappings = mappings_by_entity.get(entity.id, [])

            # 选中间帧：管线写帧时已做过黑帧/损坏过滤，此处零读盘——
            # 原 cv2 逐帧验黑是 23s/页 的瓶颈（36 视频 × 全部帧 imread），
            # 文件缺失时向两侧顺延找存在的帧即可
            ordered = sorted(tag_mappings,
                             key=lambda m: m.video_frame_number or 0)
            for idx in range(len(ordered)):
                mid = len(ordered) // 2 + (1 if idx % 2 else -1) * ((idx + 1) // 2)
                if 0 <= mid < len(ordered):
                    cand = ordered[mid]
                    if cand.frame_path and os.path.exists(cand.frame_path):
                        cover_path = cand.frame_path
                        video_frame_number = cand.video_frame_number or 0
                        break

            if not cover_path and primary_path:
                # 不在 macOS 受保护目录(桌面/文稿/下载)下探测 —— 仅"猜"帧目录位置而已,
                # 却会触发系统授权弹窗("想访问桌面文件夹")。以 DB 的 frames_path 为准。
                from src.api.main import _is_tcc_protected_path
                if not _is_tcc_protected_path(Path(primary_path).parent):
                    video_path = Path(primary_path)
                    frames_dir = video_path.parent / f"{video_path.stem}_frames"
                    if frames_dir.exists():
                        for unit_dir in sorted(frames_dir.glob("unit_*")):
                            if unit_dir.is_dir():
                                frame_files = sorted(unit_dir.glob("frame_*.jpg"))
                                if frame_files:
                                    cover_path = str(frame_files[0])
                                    break
                        if not cover_path:
                            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                            if frame_files:
                                cover_path = str(frame_files[0])

            # 人物列表（actor_faces / actors 均已批量预取，避免 N+1）
            actor_ids = {af.actor_id for af in faces_by_entity.get(entity.id, [])
                         if af.actor_id}
            actors_list = []
            for a in actors_by_id:
                if a.id not in actor_ids:
                    continue
                ck = (a.id, a.cover_face)
                if ck not in _cover_abs_cache:
                    _cover_abs_cache[ck] = _actor_cover_abs(db, a)
                actors_list.append({
                    "id": a.id, "code": a.code, "name": a.name,
                    "cover_face": _cover_abs_cache[ck],
                })

            videos.append({
                "id": fa.id,
                "entity_id": entity.id,
                "file_path": primary_path,
                "title": Path(primary_path).name if primary_path else None,
                "tags": tags_list,
                "status": fa.status,
                "created_at": fa.created_at.isoformat() if fa.created_at else None,
                "cover_path": cover_path,
                "video_frame_number": video_frame_number,
                "actors": actors_list,
            })

        return {
            "total": total,
            "page": page,
            "size": size,
            "videos": videos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取视频列表失败: {str(e)}")
    finally:
        db.close()


@router.get("/all")
async def get_all_videos(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页数量")
) -> Dict[str, Any]:
    """获取所有视频列表（分页）"""
    from sqlalchemy.orm import joinedload
    engine = get_db_engine()
    db = get_session(engine)

    try:
        # 查询所有视频（批量预载 entity+locations，消 N+1）
        query = db.query(Task).options(
            joinedload(Task.entity).selectinload(FileEntity.locations))
        total = query.count()

        # 分页
        tasks = query.order_by(Task.updated_at.desc()).offset(
            (page - 1) * size
        ).limit(size).all()

        # 批量预取 FileAnalysis（原逐行查询，50 行=50 查询）
        page_entity_ids = [t.entity_id for t in tasks if t.entity_id]
        fa_by_entity = {}
        if page_entity_ids:
            for fa in db.query(FileAnalysis).filter(
                    FileAnalysis.entity_id.in_(page_entity_ids)).all():
                fa_by_entity.setdefault(fa.entity_id, fa)

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

            if not primary_path:
                primary_path = task.file_path

            # 从 FileAnalysis 读取标签（已批量预取）
            tags_list = []
            fa = fa_by_entity.get(task.entity_id)
            if fa and fa.mapped_tags:
                try:
                    tags_list = json.loads(fa.mapped_tags)
                except (json.JSONDecodeError, TypeError):
                    tags_list = []

            # 封面路径：照片=文件本身；视频前端按 frames_path 兜底选首帧
            media_type = task.entity.media_type if task.entity else "video"
            cover_path = primary_path if media_type == "photo" else None

            videos.append({
                "id": task.id,
                "file_path": primary_path,
                "title": task.file_path.split("/")[-1] if task.file_path else None,
                "tags": tags_list,
                "status": task.status,
                "media_type": media_type,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "cover_path": cover_path
            })

        return {
            "total": total,
            "page": page,
            "size": size,
            "videos": videos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取视频列表失败: {str(e)}")
    finally:
        db.close()


@router.post("/batch-update-tags")
async def batch_update_tags(request: BatchUpdateTagsRequest) -> Dict[str, Any]:
    """批量给视频添加标签 — 写入 FileAnalysis（业务数据持久化核心）"""
    engine = get_db_engine()
    db = get_session(engine)

    try:
        updated_count = 0

        for video_id in request.video_ids:
            task = db.query(Task).filter(Task.id == video_id).first()
            if not task or not task.entity_id:
                continue

            # 获取或创建 FileAnalysis
            fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
            if not fa:
                fa = FileAnalysis(
                    entity_id=task.entity_id,
                    task_id=task.id,
                    status=task.status or 'PENDING',
                    progress=task.progress or 0
                )
                db.add(fa)
                db.flush()

            # 解析现有标签（从 FileAnalysis 读取）
            tags_list = []
            if fa.mapped_tags:
                try:
                    tags_list = json.loads(fa.mapped_tags)
                except (json.JSONDecodeError, TypeError):
                    tags_list = []

            # 添加新标签（如果不存在）
            if request.tag_name not in tags_list:
                tags_list.append(request.tag_name)
                fa.mapped_tags = json.dumps(tags_list, ensure_ascii=False)
                updated_count += 1

        db.commit()

        return {
            "success": True,
            "updated_count": updated_count,
            "message": f"成功为 {updated_count} 个视频添加标签"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量更新标签失败: {str(e)}")
    finally:
        db.close()


@router.post("/batch-delete-tags")
async def batch_delete_tags(request: BatchDeleteTagsRequest) -> Dict[str, Any]:
    """批量删除视频标签 — 从 FileAnalysis 读写（业务数据持久化核心）"""
    engine = get_db_engine()
    db = get_session(engine)

    try:
        updated_count = 0

        for video_id in request.video_ids:
            task = db.query(Task).filter(Task.id == video_id).first()
            if not task or not task.entity_id:
                continue

            # 获取 FileAnalysis
            fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
            if not fa:
                continue

            # 解析现有标签
            tags_list = []
            if fa.mapped_tags:
                try:
                    tags_list = json.loads(fa.mapped_tags)
                except (json.JSONDecodeError, TypeError):
                    tags_list = []

            # 删除指定标签
            if request.tag_name in tags_list:
                tags_list.remove(request.tag_name)
                fa.mapped_tags = json.dumps(tags_list, ensure_ascii=False)
                updated_count += 1

        db.commit()

        return {
            "success": True,
            "updated_count": updated_count,
            "message": f"成功从 {updated_count} 个视频删除标签"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量删除标签失败: {str(e)}")
    finally:
        db.close()


@router.delete("/batch")
async def batch_delete_videos(request: BatchDeleteRequest) -> Dict[str, Any]:
    """批量删除分析记录（含关联的文件实体、人脸数据等）。

    请求体：{"analysis_ids": [int]}
    返回：{"success": true, "deleted_count": int, "failed_ids": [int]}
    """
    engine = get_db_engine()
    db = get_session(engine)

    deleted_count = 0
    failed_ids = []

    try:
        for analysis_id in request.analysis_ids:
            try:
                fa = db.query(FileAnalysis).filter(FileAnalysis.id == analysis_id).first()
                if not fa:
                    failed_ids.append(analysis_id)
                    continue

                entity_id = fa.entity_id

                # 删除 FileAnalysis
                db.delete(fa)

                # 删除关联的 FileEntity（级联删除 locations, actor_faces, task_actor_names 等）
                if entity_id:
                    entity = db.query(FileEntity).filter(FileEntity.id == entity_id).first()
                    if entity:
                        # 解除 tasks 关联（避免外键约束）
                        tasks = db.query(Task).filter(Task.entity_id == entity_id).all()
                        for task in tasks:
                            task.entity_id = None
                        # 解除 tag_frame_mappings 关联
                        mappings = db.query(TagFrameMapping).filter(TagFrameMapping.entity_id == entity_id).all()
                        for m in mappings:
                            m.entity_id = None
                        db.delete(entity)

                deleted_count += 1

            except Exception:
                failed_ids.append(analysis_id)
                db.rollback()
                continue

        # 只在有成功删除时提交
        if deleted_count > 0:
            db.commit()

        return {
            "success": len(failed_ids) == 0,
            "deleted_count": deleted_count,
            "failed_ids": failed_ids
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")
    finally:
        db.close()


@router.post("/batch-tags")
async def set_video_tags(request: BatchTagsRequest) -> Dict[str, Any]:
    """整表设置文件的标签列表 —— 标签编辑的唯一入口。

    为什么只有"设置"这一种操作：标签需要能定位到对应帧
    （tag_frame_mappings），单独追加一段文字没有意义，故不提供"添加"；
    "替换"同理。**删除 = 传一个不含该标签的列表**。

    请求体：{"analysis_ids": [int], "tags": [str]}
    返回：{"success": true, "updated_count": int, "failed_ids": [int]}

    传入的 tags 整体覆盖 mapped_tags（顺序变化也算一次更新）。
    """
    engine = get_db_engine()
    db = get_session(engine)

    updated_count = 0
    failed_ids = []

    # 传入标签先规整：去空、去首尾空格、去重（保序）；对所有文件同一份
    new_tags = []
    _seen = set()
    for _t in request.tags:
        _t = (_t or "").strip()
        if _t and _t not in _seen:
            _seen.add(_t)
            new_tags.append(_t)

    try:
        for analysis_id in request.analysis_ids:
            try:
                fa = db.query(FileAnalysis).filter(FileAnalysis.id == analysis_id).first()
                if not fa:
                    failed_ids.append(analysis_id)
                    continue

                # 解析现有标签（用于判断是否真的有变化）
                tags_list = []
                if fa.mapped_tags:
                    try:
                        tags_list = json.loads(fa.mapped_tags)
                    except (json.JSONDecodeError, TypeError):
                        tags_list = []

                if new_tags != tags_list:
                    fa.mapped_tags = json.dumps(new_tags, ensure_ascii=False)
                    updated_count += 1

            except Exception:
                failed_ids.append(analysis_id)
                db.rollback()
                continue

        if updated_count > 0:
            db.commit()

        return {
            "success": len(failed_ids) == 0,
            "updated_count": updated_count,
            "failed_ids": failed_ids
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量编辑标签失败: {str(e)}")
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


@router.get("/{analysis_id}/quality")
async def get_video_quality(analysis_id: int) -> Dict[str, Any]:
    """获取视频的分析质量指标（face-quality-filter 阶段 5.1）。

    从 FileAnalysis.analysis_details 的 video_quality 字段读取
    （分析时由 VideoQualityAnalyzer 写入）。
    """
    engine = get_db_engine()
    db = get_session(engine)
    try:
        fa = db.query(FileAnalysis).filter(FileAnalysis.id == analysis_id).first()
        if not fa:
            raise HTTPException(status_code=404, detail="分析记录不存在")

        vq = {}
        if fa.analysis_details:
            try:
                details = json.loads(fa.analysis_details)
                vq = details.get("video_quality", {}) or {}
            except (json.JSONDecodeError, TypeError):
                vq = {}

        avg_sharpness = vq.get("avg_sharpness")
        return {
            "analysis_id": analysis_id,
            "entity_id": fa.entity_id,
            "resolution": vq.get("resolution"),
            "fps": vq.get("fps"),
            "avg_sharpness": avg_sharpness,
            "threshold": vq.get("threshold"),
            "is_low_quality": vq.get("is_low_quality"),
            "quality_level": _quality_level(avg_sharpness),
        }
    finally:
        db.close()
