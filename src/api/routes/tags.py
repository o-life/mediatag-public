from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from typing import Dict, Any, Optional
from src.db.models import DictionaryEntry, get_session, get_db_engine

router = APIRouter(prefix="/api/tags", tags=["tags"])


@router.get("/tree")
async def get_tag_tree(theme_id: Optional[int] = None) -> Dict[str, Any]:
    """获取标签分类树结构，包含"全部"选项。file_count 从 tag_frame_mappings 实时统计。

    Args:
        theme_id: 可选，按主题过滤标签（Phase 3）。None=全部，0=未分类，>0=指定主题。
    """
    engine = get_db_engine()
    db = get_session(engine)
    
    try:
        # 统计"多少**作品**带此标签"：按内容(entity_id)去重。
        # - 与「按标签取作品」口径一致（那里是 tag_frame_mappings → entity_id → 分析档案）
        # - 删除作品时映射行只被"解除归属"（entity_id 置空）而不清行，
        #   按 entity_id 统计（COUNT DISTINCT 忽略 NULL）可自动排除，避免"计数不变、列表少一条"
        tag_video_counts = dict(
            db.execute(
                text("SELECT tag_name, COUNT(DISTINCT entity_id) FROM tag_frame_mappings GROUP BY tag_name")
            ).all()
        )
        
        # 查询标签（支持主题过滤 Phase 3）
        query = db.query(DictionaryEntry)
        if theme_id is not None and theme_id > 0:
            # 精确匹配主题：切换主题应看到不同标签集
            query = query.filter(DictionaryEntry.theme_id == theme_id)
        elif theme_id == 0:
            query = query.filter(DictionaryEntry.theme_id.is_(None))
        entries = query.order_by(
            DictionaryEntry.category, 
            DictionaryEntry.standard_tag
        ).all()
        
        # 构建分类树
        categories = {}
        for entry in entries:
            category = entry.category or "其他"
            if category not in categories:
                categories[category] = []
            
            tag_name = entry.standard_tag
            # 优先使用 tag_frame_mappings 的实际统计，fallback 到 dictionary 的 file_count
            file_count = tag_video_counts.get(tag_name, entry.file_count or 0)
            
            categories[category].append({
                "name": tag_name,
                "aliases": entry.aliases.split(",") if entry.aliases else [],
                "file_count": file_count,
                "theme_id": entry.theme_id,
            })
        
        # 构建返回结构
        tree = {
            "categories": []
        }
        
        for category_name, tags in sorted(categories.items()):
            tree["categories"].append({
                "name": category_name,
                "tags": tags
            })
        
        return tree
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标签树失败: {str(e)}")
    finally:
        db.close()
