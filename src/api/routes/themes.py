"""分析主题 API。

**元数据来源已改为「主题注册表」**（`<base>/themes/registry.json`，见
`src/core/theme_registry.py`）——因为"有哪些主题、各自的库与 config 在哪"
必须先于"打开某个主题的数据文件"可知，不能放在它自己管理的库里。

过渡期桥接（Phase 4 移除）：`themes` 表仍按**主题名**保留一行，用于现有
`dictionary.theme_id` 的标签归属统计；注册表是元数据的唯一真相，DB 那行只是
"名字 → 旧 id"的映射。
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.theme_registry import get_theme_registry
from src.db.models import (
    Theme, DictionaryEntry,
    get_session, get_db_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/themes", tags=["themes"])


class ThemeCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    label_mapping_path: Optional[str] = None
    faces_dir: Optional[str] = None


class ThemeUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_default: Optional[int] = None
    label_mapping_path: Optional[str] = None
    faces_dir: Optional[str] = None


class ThemePromptRequest(BaseModel):
    prompt: str


# ── 过渡期桥接：注册表主题名 ↔ DB 旧 id ─────────────────────────────

def _db_theme_id(db, name: str) -> Optional[int]:
    row = db.query(Theme).filter(Theme.name == name).first()
    return row.id if row else None


def _ensure_db_row(db, name: str) -> Optional[int]:
    """确保 DB 里存在同名主题行（仅用于旧 theme_id 归属），返回其 id。"""
    row = db.query(Theme).filter(Theme.name == name).first()
    if row is None:
        row = Theme(name=name)
        db.add(row)
        db.flush()
    return row.id


def _tag_count(db, name: str, rec: dict, current: str) -> int:
    """标签数。

    **每主题一个数据文件**之后：当前主题的库就是本库 → 直接数（每库只有本主题标签）；
    其它主题的库没打开 → 用它的映射表**只解析**得出条数（不写库）。
    """
    if name == current:
        return db.query(DictionaryEntry).count()
    md = (rec.get("label_mapping_path") or "").strip()
    if md:
        from src.core.dictionary import count_mapping_md_tags

        return count_mapping_md_tags(md)
    return 0


def _serialize(db, rec: dict, current: str = "") -> dict:
    return {
        "id": _db_theme_id(db, rec.get("name", "")),
        "name": rec.get("name"),
        "description": rec.get("description"),
        "is_default": 1 if rec.get("is_default") else 0,
        "label_mapping_path": rec.get("label_mapping_path"),
        "faces_dir": rec.get("faces_dir"),
        "tag_count": _tag_count(db, rec.get("name", ""), rec, current),
        "prompt": rec.get("prompt") or "",
    }


@router.get("")
async def list_themes():
    """列出所有主题（元数据取注册表；tag_count 取当前库）。"""
    db = get_session(get_db_engine())
    try:
        reg = get_theme_registry()
        records = reg.list_themes()
        current = reg.effective_current()
        result = []
        for rec in sorted(records, key=lambda r: (not r.get("is_default"), r.get("name") or "")):
            item = _serialize(db, rec, current)
            item["is_current"] = rec.get("name") == current
            result.append(item)
        return {"status": "success", "current": current, "themes": result}
    finally:
        db.close()


@router.post("")
async def create_theme(request: ThemeCreateRequest):
    """创建主题（写注册表；DB 同步一行用于旧 tags 归属）。"""
    reg = get_theme_registry()
    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="主题名不能为空")
    if reg.get(name):
        raise HTTPException(status_code=400, detail=f"主题「{name}」已存在")

    db = get_session(get_db_engine())
    try:
        reg.upsert(
            name,
            description=request.description,
            label_mapping_path=(request.label_mapping_path or None),
            faces_dir=(request.faces_dir or None),
        )
        _ensure_db_row(db, name)
        db.commit()
        return {"status": "success", "name": name}
    finally:
        db.close()


@router.put("/{theme_id}")
async def update_theme(theme_id: int, request: ThemeUpdateRequest):
    """更新主题。``theme_id`` 为**过渡期的 DB id**，主键实际用 ``request.name``。"""
    reg = get_theme_registry()
    db = get_session(get_db_engine())
    try:
        row = db.query(Theme).filter(Theme.id == theme_id).first()
        current_name = row.name if row else None
        if current_name is None:
            # 兼容按名字调用（注册表里存在即可）
            current_name = request.name if request.name and reg.get(request.name) else None
        if current_name is None or not reg.get(current_name):
            raise HTTPException(status_code=404, detail="主题不存在")

        new_name = (request.name or "").strip() or current_name
        if new_name != current_name and reg.get(new_name):
            raise HTTPException(status_code=400, detail=f"主题「{new_name}」已存在")

        fields = {}
        if request.description is not None:
            fields["description"] = request.description
        if request.label_mapping_path is not None:
            fields["label_mapping_path"] = request.label_mapping_path or None
        if request.faces_dir is not None:
            fields["faces_dir"] = request.faces_dir or None
        if new_name != current_name:
            fields["name"] = new_name
        if fields:
            reg.delete(current_name) if new_name != current_name else None
            reg.upsert(new_name, **fields)
        if request.is_default is not None:
            if request.is_default == 1:
                reg.upsert(new_name, is_default=True)
            else:
                data = reg.load()
                for t in data["themes"]:
                    if t.get("name") == new_name:
                        t["is_default"] = False
                reg.save(data)

        # 桥接：DB 行改名 / 保证存在
        if row is not None and new_name != current_name:
            row.name = new_name
        _ensure_db_row(db, new_name)
        db.commit()
        return {"status": "success", "message": f"主题「{new_name}」已更新"}
    finally:
        db.close()


@router.delete("/{theme_id}")
async def delete_theme(theme_id: int):
    """删除主题（默认主题不可删）。"""
    reg = get_theme_registry()
    db = get_session(get_db_engine())
    try:
        row = db.query(Theme).filter(Theme.id == theme_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="主题不存在")
        name = row.name
        if not reg.get(name):
            # 注册表里没有（迁移遗留）→ 只清库
            pass
        else:
            try:
                reg.delete(name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        db.query(DictionaryEntry).filter(DictionaryEntry.theme_id == theme_id).update({"theme_id": None})
        db.delete(row)  # cascade 删除 theme_prompts（遗留）
        db.commit()
        return {"status": "success", "message": f"主题「{name}」已删除"}
    finally:
        db.close()


@router.get("/{theme_id}/prompt")
async def get_theme_prompt(theme_id: int):
    """获取主题的提示词（注册表）。"""
    db = get_session(get_db_engine())
    try:
        row = db.query(Theme).filter(Theme.id == theme_id).first()
    finally:
        db.close()
    rec = get_theme_registry().get(row.name) if row else None
    return {"status": "success", "theme_id": theme_id, "prompt": (rec or {}).get("prompt") or ""}


@router.put("/{theme_id}/prompt")
async def set_theme_prompt(theme_id: int, request: ThemePromptRequest):
    """设置主题的提示词（注册表）。"""
    db = get_session(get_db_engine())
    try:
        row = db.query(Theme).filter(Theme.id == theme_id).first()
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="主题不存在")
    reg = get_theme_registry()
    if not reg.get(row.name):
        raise HTTPException(status_code=404, detail="主题不存在于注册表")
    reg.upsert(row.name, prompt=request.prompt)
    return {"status": "success", "message": f"主题「{row.name}」prompt 已保存"}


@router.post("/seed-presets")
async def seed_presets():
    """载入/补建预置分析主题（幂等）。供「分析主题管理」对话框使用。"""
    from src.core.preset_themes import seed_preset_themes

    db = get_session(get_db_engine())
    try:
        created = seed_preset_themes(get_theme_registry())
        for name in get_theme_registry().names():
            _ensure_db_row(db, name)
        db.commit()
        return {"status": "success", "created": created}
    finally:
        db.close()


@router.post("/{theme_id}/activate")
async def activate_theme(theme_id: int):
    """激活分析主题：写注册表 current，**不导标签**（每主题一库，标签随库走）。

    ⚠️ 冷切语义：真正生效要**重启进程**——重启后按新主题的 `data.db` 与 `faces/` 打开；
    该主题若还没有库，启动链路会建库并导入它自己的映射表。
    因此这里**不能**把目标主题的映射导进当前库（那会污染当前主题）。
    """
    db = get_session(get_db_engine())
    try:
        row = db.query(Theme).filter(Theme.id == theme_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="主题不存在")
        name = row.name
        reg = get_theme_registry()
        rec = reg.get(name)
        if rec is None:
            raise HTTPException(status_code=404, detail="主题不存在于注册表")

        reg.set_current(name)
        try:                                    # 预建目录/config/faces（不含建库）
            from src.core.theme_workspace import get_theme_workspace

            ws = get_theme_workspace()
            ws.ensure_theme(name)
        except Exception as exc:
            logger.warning("切换主题时预建目录失败（不阻断）: %s", exc)

        logger.info("已切换当前分析主题为「%s」（需重启生效）", name)
        return {
            "status": "success",
            "name": name,
            "restart_required": True,
            "theme": {
                "id": row.id,
                "name": name,
                "label_mapping_path": rec.get("label_mapping_path"),
                "faces_dir": rec.get("faces_dir"),
                "has_prompt": bool((rec.get("prompt") or "").strip()),
            },
            "imported_tags": 0,
        }
    finally:
        db.close()
