"""影楼基础包 S3.2 —— 客单/活动/类型词表 API（prefix=/api/studio）。

数据模型见 src/db/models.py（Client/EventType/Project/ProjectAsset）。
归属模型（project-assets-roster 2026-09-30）：项目自持清单——成员关系存
project_assets（锚 file_md5），多归属，与任务生命周期完全解耦；
tasks.project_id 为影子列（停写）。

约定：
- 所有写操作走 get_session 会话并显式 commit（与 actors.py 同构）；
- 删除保护：客户有活动时拒删（409）；预置类型不可删（只能停用）；
  项目删除仅移除该子树清单行（内容本体/分析结果/其他项目成员资格不动）；
- 活动名未提供时自动生成"{客户名}·{类型} {日期}"。
"""

import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import or_
from src.db.models import (
    Client,
    EventType,
    Project,
    ProjectAsset,
    get_db_engine,
    get_session,
)

router = APIRouter(prefix="/api/studio", tags=["studio"])


def _task_file_md5s(db, task_ids) -> list:
    """工单 id 列表 → 内容哈希列表（entity 关联优先，回退影子 file_md5）。

    未知 task_id 忽略；解析不出 md5 的工单无法入清单（归属锚内容哈希）。"""
    if not task_ids:
        return []
    from sqlalchemy import bindparam
    from sqlalchemy import text as sa_text
    stmt = sa_text("""
            SELECT DISTINCT COALESCE(e.file_md5, t.file_md5)
            FROM tasks t LEFT JOIN file_entities e ON t.entity_id = e.id
            WHERE t.id IN :ids AND COALESCE(e.file_md5, t.file_md5) IS NOT NULL
        """).bindparams(bindparam("ids", expanding=True))
    rows = db.execute(
        stmt, {"ids": [int(i) for i in task_ids]},
    ).fetchall()
    return [r[0] for r in rows]


def _client_name(db, client_id: int) -> str:
    c = db.get(Client, client_id)
    return c.name if c else f"客户{client_id}"


def _type_name(db, event_type_id: Optional[int]) -> str:
    if not event_type_id:
        return ""
    et = db.get(EventType, event_type_id)
    return et.name if et else ""


def _load_custom_fields(raw: Optional[str]) -> Dict[str, str]:
    """custom_fields TEXT → dict；坏数据容错为空。"""
    import json as _json
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dump_custom_fields(value) -> str:
    """任意输入 → 合法 JSON 文本（dict 之外的类型丢弃为空对象）。"""
    import json as _json
    if isinstance(value, dict):
        clean = {str(k): str(v) for k, v in value.items() if str(k).strip()}
        return _json.dumps(clean, ensure_ascii=False)
    return "{}"


def _project_dict(db, p: Project) -> Dict[str, Any]:
    # asset_count = **直接成员数**（不含子级，单项目自持清单口径）。
    # 父级子树汇总由消费方承担：详情页范围树前端 rollup、/videos?project_id
    # 服务端子树过滤（_subtree_roster）——勿在此改成子树口径（会造成前端
    # 双重汇总，2026-09-30 验收实测确认）。
    asset_count = (
        db.query(ProjectAsset).filter(ProjectAsset.project_id == p.id).count()
    )
    et_name = _type_name(db, p.event_type_id)
    return {
        "id": p.id,
        "client_id": p.client_id,
        "parent_id": p.parent_id,
        "event_type_id": p.event_type_id,
        "event_type": et_name,
        "name": p.name,
        "event_date": p.event_date.isoformat() if p.event_date else None,
        "package": p.package,
        "status": p.status,
        "notes": p.notes,
        "custom_fields": _load_custom_fields(p.custom_fields),
        "asset_count": asset_count,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _client_dict(db, c: Client) -> Dict[str, Any]:
    projects = db.query(Project).filter(Project.client_id == c.id).all()
    last_date = max(
        (p.event_date for p in projects if p.event_date), default=None
    )
    return {
        "id": c.id,
        "name": c.name,
        "contact": c.contact,
        "notes": c.notes,
        "project_count": len(projects),
        "last_event_date": last_date.isoformat() if last_date else None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


# ── 客户 ─────────────────────────────────────────────────────────────


@router.get("/clients")
def list_clients(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        query = db.query(Client)
        if search:
            like = f"%{search}%"
            query = query.filter(
                or_(Client.name.like(like), Client.contact.like(like))
            )
        total = query.count()
        rows = (
            query.order_by(Client.created_at.desc(), Client.id.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        return {
            "total": total,
            "page": page,
            "size": size,
            "clients": [_client_dict(db, c) for c in rows],
        }
    finally:
        db.close()


@router.post("/clients")
def create_client(request: dict):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        name = (request.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="客户称呼不能为空")
        c = Client(
            name=name,
            contact=request.get("contact"),
            notes=request.get("notes"),
        )
        db.add(c)
        db.commit()
        return {"status": "success", "id": c.id}
    finally:
        db.close()


@router.patch("/clients/{client_id}")
def update_client(client_id: int, request: dict):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        c = db.get(Client, client_id)
        if not c:
            raise HTTPException(status_code=404, detail="客户不存在")
        if "name" in request:
            name = (request.get("name") or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="客户称呼不能为空")
            c.name = name
        for field in ("contact", "notes"):
            if field in request:
                setattr(c, field, request[field])
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


@router.delete("/clients/{client_id}")
def delete_client(client_id: int):
    """删除客户。有活动时拒删（409）——先删除/移交活动。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        c = db.get(Client, client_id)
        if not c:
            raise HTTPException(status_code=404, detail="客户不存在")
        projects = db.query(Project).filter(Project.client_id == client_id).all()
        if projects:
            raise HTTPException(
                status_code=409,
                detail=f"该客户名下还有 {len(projects)} 个活动，请先删除或移交活动",
            )
        db.delete(c)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


# ── 活动类型词表 ─────────────────────────────────────────────────────


@router.get("/event-types")
def list_event_types(include_inactive: bool = False):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        query = db.query(EventType)
        if not include_inactive:
            query = query.filter(EventType.active == 1)
        rows = (
            query.order_by(EventType.sort_order, EventType.id).all()
        )
        return {
            "event_types": [
                {
                    "id": et.id,
                    "name": et.name,
                    "is_builtin": bool(et.is_builtin),
                    "sort_order": et.sort_order,
                    "active": bool(et.active),
                }
                for et in rows
            ]
        }
    finally:
        db.close()


@router.post("/event-types")
def create_event_type(request: dict):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        name = (request.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="类型名称不能为空")
        exists = db.query(EventType).filter(EventType.name == name).first()
        if exists:
            raise HTTPException(status_code=409, detail="类型已存在")
        max_order = (
            db.query(EventType).order_by(EventType.sort_order.desc()).first()
        )
        et = EventType(
            name=name,
            is_builtin=0,
            sort_order=(max_order.sort_order + 1) if max_order else 1,
            active=1,
        )
        db.add(et)
        db.commit()
        return {"status": "success", "id": et.id}
    finally:
        db.close()


@router.patch("/event-types/{type_id}")
def update_event_type(type_id: int, request: dict):
    """改名/停用/启用。预置类型可停用不可删除（历史活动引用保留）。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        et = db.get(EventType, type_id)
        if not et:
            raise HTTPException(status_code=404, detail="类型不存在")
        if "name" in request:
            name = (request.get("name") or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="类型名称不能为空")
            if et.is_builtin:
                raise HTTPException(status_code=400, detail="预置类型不可改名")
            et.name = name
        if "active" in request:
            et.active = 1 if request["active"] else 0
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


@router.delete("/event-types/{type_id}")
def delete_event_type(type_id: int):
    """删除自定义类型。预置类型与被活动引用的类型拒删（提示停用）。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        et = db.get(EventType, type_id)
        if not et:
            raise HTTPException(status_code=404, detail="类型不存在")
        if et.is_builtin:
            raise HTTPException(status_code=400, detail="预置类型不可删除，可停用")
        used = db.query(Project).filter(Project.event_type_id == type_id).count()
        if used:
            raise HTTPException(
                status_code=409, detail=f"仍有 {used} 个活动使用该类型，请先停用"
            )
        db.delete(et)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


# ── 活动（项目）──────────────────────────────────────────────────────


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.date.fromisoformat(s)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail=f"日期格式错误: {s}")


@router.get("/projects")
def list_projects(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    client_id: Optional[int] = None,
    event_type_id: Optional[int] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    search: Optional[str] = None,
):
    """活动列表（分页 + 多维筛选：客户/类型/状态/日期范围/搜索）。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        query = db.query(Project)
        if client_id:
            query = query.filter(Project.client_id == client_id)
        if event_type_id:
            query = query.filter(Project.event_type_id == event_type_id)
        if status:
            query = query.filter(Project.status == status)
        if date_from:
            query = query.filter(Project.event_date >= _parse_date(date_from))
        if date_to:
            query = query.filter(Project.event_date <= _parse_date(date_to))
        if search:
            like = f"%{search}%"
            query = query.filter(or_(Project.name.like(like), Project.package.like(like)))
        total = query.count()
        rows = (
            query.order_by(Project.event_date.desc().nullslast(), Project.id.desc())
            .offset((page - 1) * size)
            .limit(size)
            .all()
        )
        out = []
        for p in rows:
            d = _project_dict(db, p)
            d["client_name"] = _client_name(db, p.client_id)
            out.append(d)
        return {"total": total, "page": page, "size": size, "projects": out}
    finally:
        db.close()


@router.post("/projects")
def create_project(request: dict):
    """新建项目/文件夹。

    带 parent_id = 在某文件夹下新增子文件夹（client_id 继承根活动，可缺省）；
    不带 = 顶层活动（兼容原流程，仍需 client_id）。
    """
    engine = get_db_engine()
    db = get_session(engine)
    try:
        parent_id = request.get("parent_id")
        parent = None
        if parent_id:
            parent = db.get(Project, parent_id)
            if not parent:
                raise HTTPException(status_code=404, detail="父文件夹不存在")
        client_id = request.get("client_id")
        if not client_id:
            if parent is not None:
                client_id = parent.client_id  # 子文件夹继承根活动的客户
            else:
                raise HTTPException(status_code=400, detail="client_id 不能为空")
        client = db.get(Client, client_id)
        if not client:
            raise HTTPException(status_code=404, detail="客户不存在")
        event_type_id = request.get("event_type_id")
        event_date = _parse_date(request.get("event_date"))
        name = (request.get("name") or "").strip()
        if not name and parent is not None:
            name = "新建文件夹"
        if not name:
            type_name = _type_name(db, event_type_id)
            date_part = f" {event_date.strftime('%m-%d')}" if event_date else ""
            name = f"{client.name}·{type_name}{date_part}".strip(" ·") or f"{client.name} 活动"
        p = Project(
            client_id=client_id,
            parent_id=parent.id if parent is not None else None,
            event_type_id=event_type_id,
            name=name,
            event_date=event_date,
            package=request.get("package"),
            status=request.get("status") or "拍摄中",
            notes=request.get("notes"),
            custom_fields=_dump_custom_fields(request.get("custom_fields")),
        )
        db.add(p)
        db.commit()
        return {"status": "success", "id": p.id, "name": p.name}
    finally:
        db.close()


@router.get("/projects/{project_id}")
def get_project(project_id: int):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(status_code=404, detail="活动不存在")
        d = _project_dict(db, p)
        d["client_name"] = _client_name(db, p.client_id)
        return d
    finally:
        db.close()


@router.patch("/projects/{project_id}")
def update_project(project_id: int, request: dict):
    engine = get_db_engine()
    db = get_session(engine)
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(status_code=404, detail="活动不存在")
        for field in ("name", "package", "status", "notes"):
            if field in request:
                setattr(p, field, request[field])
        if "custom_fields" in request:
            p.custom_fields = _dump_custom_fields(request["custom_fields"])
        if "event_type_id" in request:
            p.event_type_id = request["event_type_id"]
        if "event_date" in request:
            p.event_date = _parse_date(request["event_date"])
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


@router.delete("/projects/{project_id}")
def delete_project(project_id: int):
    """删除项目（含全部子级）：仅摘除该子树的清单行。

    内容本体、分析结果、磁盘文件、以及内容在其他项目中的成员资格全部保留；
    失去所有清单行的内容自然回到"未归集"口径。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(status_code=404, detail="活动不存在")
        # 收集整棵子树（含自身），逐层向下找
        ids = [project_id]
        frontier = [project_id]
        while frontier:
            rows = (
                db.query(Project.id)
                .filter(Project.parent_id.in_(frontier))
                .all()
            )
            frontier = [r[0] for r in rows if r[0] not in ids]
            ids.extend(frontier)
        db.query(ProjectAsset).filter(
            ProjectAsset.project_id.in_(ids)
        ).delete(synchronize_session=False)
        db.query(Project).filter(Project.id.in_(ids)).delete(
            synchronize_session=False
        )
        db.commit()
        return {"status": "success", "deleted": len(ids)}
    finally:
        db.close()


# ── 素材挂接（项目自持清单，锚 file_md5）────────────────────────────


@router.post("/projects/{project_id}/assign")
def assign_tasks(project_id: int, request: dict):
    """把内容加入项目清单（多归属：已在其他项目不受影响）。

    入参二选一：``task_ids``（服务端解析内容哈希）或 ``file_md5s``。
    重复加入幂等（唯一约束兜底），返回 added=新增行数、duplicates=命中已存在数。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(status_code=404, detail="活动不存在")
        md5s = list(request.get("file_md5s") or [])
        if not md5s:
            md5s = _task_file_md5s(db, request.get("task_ids") or [])
        if not md5s:
            raise HTTPException(status_code=400, detail="未提供可归集的内容")
        existing = {
            r[0] for r in db.query(ProjectAsset.file_md5)
            .filter(ProjectAsset.project_id == project_id,
                    ProjectAsset.file_md5.in_(md5s))
            .all()
        }
        for m in md5s:
            if m not in existing:
                db.add(ProjectAsset(project_id=project_id, file_md5=m))
        db.commit()
        added = len(md5s) - len(existing)
        return {"status": "success", "assigned": added,
                "duplicates": len(existing)}
    finally:
        db.close()


@router.post("/projects/{project_id}/unassign")
def unassign_from_project(project_id: int, request: dict):
    """把内容移出指定项目（仅删该项目的清单行；其他项目成员资格不动）。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        p = db.get(Project, project_id)
        if not p:
            raise HTTPException(status_code=404, detail="活动不存在")
        md5s = list(request.get("file_md5s") or [])
        if not md5s:
            md5s = _task_file_md5s(db, request.get("task_ids") or [])
        if not md5s:
            raise HTTPException(status_code=400, detail="未提供可移出的内容")
        removed = (
            db.query(ProjectAsset)
            .filter(ProjectAsset.project_id == project_id,
                    ProjectAsset.file_md5.in_(md5s))
            .delete(synchronize_session=False)
        )
        db.commit()
        return {"status": "success", "unassigned": removed}
    finally:
        db.close()


@router.post("/unassign")
def unassign_tasks(request: dict):
    """把内容移出**全部**项目（清单行全删 → 回未归集口径）。"""
    engine = get_db_engine()
    db = get_session(engine)
    try:
        md5s = list(request.get("file_md5s") or [])
        if not md5s:
            md5s = _task_file_md5s(db, request.get("task_ids") or [])
        if not md5s:
            raise HTTPException(status_code=400, detail="未提供可移出的内容")
        updated = (
            db.query(ProjectAsset)
            .filter(ProjectAsset.file_md5.in_(md5s))
            .delete(synchronize_session=False)
        )
        db.commit()
        return {"status": "success", "unassigned": updated}
    finally:
        db.close()
