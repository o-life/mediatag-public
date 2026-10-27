# -*- coding: utf-8 -*-
"""
同义词归一化表
将相似的标签统一映射到标准标签
"""

# 同义词映射：key是变体词，value是标准词
SYNONYM_MAP = {
    # 身体状态
    "裸身": "裸体",
    "全裸": "裸体",
    "赤身": "裸体",
    "赤裸": "裸体",
    
    # 人数相关
    "单人": "一人",
    "两个人": "双人",
    "两人": "双人",
    "三个人": "三人",
    "三人": "三人",
    
    # 场景相关
    "床上": "卧室",
    "浴室": "浴室",
    "洗手间": "浴室",
    "厕所": "浴室",
    
    # 动作相关
    "口交": "口交",
    "吹箫": "口交",
    "吞吐": "口交",
    
    # 体位相关
    "传教士": "传教士体位",
    "正常位": "传教士体位",
    "后入": "后入式",
    "狗爬式": "后入式",
    "蹲趴": "后入式",
}

def normalize_tag(tag: str) -> str:
    """
    将标签归一化为标准形式
    
    Args:
        tag: 原始标签
        
    Returns:
        归一化后的标准标签
    """
    return SYNONYM_MAP.get(tag, tag)

def normalize_tags(tags: list) -> list:
    """
    批量归一化标签列表
    
    Args:
        tags: 原始标签列表
        
    Returns:
        归一化后的标签列表（去重）
    """
    normalized = set()
    for tag in tags:
        normalized.add(normalize_tag(tag))
    return list(normalized)
