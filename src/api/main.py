import os
import re
import sys
import json
import shutil
import datetime
import hashlib
import logging
import time
import random
import difflib
import unicodedata
import threading
import asyncio
import subprocess
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from sqlalchemy import or_, text
from sqlalchemy.exc import OperationalError
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from src.db.models import (
    init_db, get_db_engine, get_session, session_factory,
    Task, DictionaryEntry, AuditQueueItem, Actor, ActorFace, TagCategory, TaskActor,
    FileEntity, FileLocation, FileAnalysis, TaskActorName, TagFrameMapping,
    AnalysisLog, Project, ProjectAsset,
)
from src.db.write_gate import is_locked_error, note_lock_retry, note_lock_failure
from src.core.dictionary import DictionaryManager
from src.core.embedding_codec import decode_embedding, encode_embedding
from src.core.config import ConfigManager, get_config_manager
from src.core.scenedetect import detect_scenes
from src.core.frame_sampler import sample_frames_per_unit
from src.core.error_taxonomy import format_error
from src.core.smart_frame_sampler import SmartFrameSampler
from src.core.scheduler import get_scheduler
from src.core.cache_manager import is_frames_dir, is_inside_frames_dir

logger = logging.getLogger(__name__)


class _SkipSifting(Exception):
    """内部信号：断点续传时跳过"场景检测/抽帧"阶段。

    `goto_vlm_analysis=True`（saved_progress>=30，帧已抽好）时应直接进入
    识别/VLM，不重跑场景检测/抽帧、也不碰 sifting_semaphore 与 progress。
    用异常做"整段跳过"可保持原代码缩进不变，改动面最小。
    """


# ── 视频格式常量 ──────────────────────────────────────────────────
# 已统一至 src/core/media_types.py（影楼基础包 S2：含 RAW 判定与扫描面定义）
from src.core.media_types import ALL_MEDIA_EXTENSIONS, PHOTO_EXTENSIONS  # noqa: E402

# ── Emoji sanitizer ──────────────────────────────────────────────────
# macOS allows emoji in file names, but they cause issues in databases
# and UI rendering. Strip all emoji/unicode symbols from filenames.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # supplemental symbols
    "☀-➿"           # misc symbols
    "⌀-⏿"           # misc technical
    "︀-️"           # variation selectors
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F000-\U0001F02F"  # mahjong/domino
    "\U0001F0A0-\U0001F0FF"  # playing cards
    "\U0001F100-\U0001F64F"  # enclosed alphanumerics + emoji
    "\U0001F680-\U0001F6FF"  # transport + map symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # extended symbols
    "\U0000200D"             # zero-width joiner
    "\U0000FE0F"             # variation selector-16
    "‌-‍"          # ZWJ/ZWNJ
    "\U0001F1E6-\U0001F1FF"  # regional indicators
    "\U0001F201-\U0001F2FF"  # CJK symbols
    "]",
)

def sanitize_filename(name: str) -> str:
    """Remove all emoji/special unicode from filename."""
    return _EMOJI_RE.sub("", name)


def _parse_tags_from_vlm(tags_text: str) -> list:
    """Parse tags from VLM response, handling both JSON array and comma-separated formats.

    VLM may return tags in various formats:
    - JSON array: '["互撸", "后入"]' or '{"体位": ["互撸"]}'
    - Markdown code block wrapped JSON: '```json\n{"体位": ["后入"]}\n```'
    - Comma-separated: '互撸,后入'
    - Mixed/malformed: '"互撸", "后入"]'

    Returns a list of cleaned tag names (no quotes, brackets, or JSON fragments).
    """
    if not tags_text or not isinstance(tags_text, str):
        return []

    text = tags_text.strip()
    if not text:
        return []

    # Pre-process: strip markdown code block wrappers (```json ... ``` or ``` ... ```)
    import re
    code_block_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    tags = []

    # 1) Try parsing as a JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    # VLM 有时把整串标签塞进单个 JSON 元素(中文逗号/顿号分隔)——
                    # 实测 2026-09-30(任务 6336)。再切一刀,交给词典闸把关。
                    for part in re.split(r"[，、;；]", item):
                        cleaned = part.strip()
                        if cleaned:
                            tags.append(cleaned)
            if tags:
                return tags
        elif isinstance(parsed, dict):
            # Handle dict like {"体位": ["互撸"]} - extract all string values
            for v in parsed.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            for part in re.split(r"[，、;；]", item):
                                cleaned = part.strip()
                                if cleaned:
                                    tags.append(cleaned)
                elif isinstance(v, str):
                    for part in re.split(r"[，、;；]", v):
                        cleaned = part.strip()
                        if cleaned:
                            tags.append(cleaned)
            if tags:
                return tags
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) Fall back to comma-separated, but strip JSON artifacts from each token
    raw_tokens = text.split(",")
    for token in raw_tokens:
        # Strip surrounding quotes, brackets, braces, and key fragments
        # e.g. '"互撸"' -> '互撸', '"互撸"]' -> '互撸', '"体位": ["互撸"' -> '互撸'
        # First, if token contains a JSON key-value fragment like '"key": ["value"',
        # extract just the value part.
        cleaned = token
        # Handle '"key": ["value"' pattern — keep only after the last ':'
        if ":" in cleaned and ("[" in cleaned or "{" in cleaned):
            cleaned = cleaned.split(":", 1)[1]
        # Remove JSON structural characters
        cleaned = cleaned.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
        # Remove surrounding quotes (single or double)
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        # Skip if it looks like a JSON key (ends with quote before colon was removed)
        # or if it contains a colon (likely a key fragment)
        if ":" in cleaned:
            continue
        if cleaned:
            tags.append(cleaned)

    return tags


def _get_primary_path(task: Task) -> Optional[str]:
    """任务的主文件路径。

    优先 entity → FileLocation(is_primary=1)（内容身份建立后）；
    没有 entity 时回退到 ``task.file_path`` —— 导入只登记作业（entity_id=NULL），
    「未启动」的任务只有这一份路径；不回退就取不到文件名。
    """
    if task.entity:
        loc = next(
            (loc for loc in task.entity.locations if loc.is_primary == 1),
            task.entity.locations[0] if task.entity.locations else None
        )
        if loc:
            return loc.file_path
    return task.file_path


def _rebuild_vlm_prompt(db=None):
    """根据当前标签库动态重建 VLM prompt 并保存到配置"""
    from src.core.config import ConfigManager
    
    # 如果没传入 db，创建新的 session
    close_db = False
    if db is None:
        db = get_session(engine)
        close_db = True
    
    try:
        # 直接从 DictionaryEntry 表按 category 分组查询
        # 这样不依赖 TagCategory 表，确保修改标签分类时能正确反映
        category_names = db.query(DictionaryEntry.category).distinct().all()
        category_names = [c[0] for c in category_names if c[0]]  # 过滤空值
        
        if not category_names:
            return
        
        # 构建分类到标签的映射
        category_tags = {}
        for category_name in category_names:
            tags = db.query(DictionaryEntry).filter(
                DictionaryEntry.category == category_name
            ).all()
            category_tags[category_name] = [t.standard_tag for t in tags]
        
        if not category_tags:
            return
        
        # 生成有意义的 prompt
        prompt_parts = [
            "## 任务",
            "你是一个专业的视频内容分析助手。请分析提供的视频帧，识别并提取关键信息。",
            "",
            "## 输出要求",
            "请按以下 JSON 格式输出分析结果：",
            "",
            "```json",
            "{"
        ]
        
        # 动态生成 JSON 结构
        for category_name, tags in category_tags.items():
            # 根据分类名决定是数组还是字符串
            if category_name in ["人数"]:
                prompt_parts.append(f'  "{category_name}": "{category_name}标签",')
            else:
                prompt_parts.append(f'  "{category_name}": ["{category_name}标签"],')
        
        # 移除最后一个逗号
        if prompt_parts[-1].endswith(','):
            prompt_parts[-1] = prompt_parts[-1][:-1]
        
        prompt_parts.extend([
            "}",
            "```",
            "",
            "## 分析规则"
        ])
        
        # 动态生成分析规则
        for idx, (category_name, tags) in enumerate(category_tags.items(), 1):
            tag_examples = "、".join(tags[:6])  # 最多显示6个示例
            if len(tags) > 6:
                tag_examples += "等"
            
            # 根据分类名生成描述
            if category_name == "场景":
                desc = "识别拍摄环境"
            elif category_name in ["体位", "动作"]:
                desc = "识别主要体位和动作"
            elif category_name == "人数":
                desc = "统计参与人数"
            elif category_name == "属性":
                desc = "识别年龄特征和性向"
            elif category_name == "身材":
                desc = "识别体型特征"
            elif category_name == "颜值":
                desc = "识别面部特征"
            elif category_name == "视角":
                desc = "识别拍摄角度"
            elif category_name == "肤色":
                desc = "识别肤色"
            elif category_name == "服装":
                desc = "识别服装类型"
            else:
                desc = f"识别{category_name}相关内容"
            
            prompt_parts.append(f"{idx}. **{category_name}**：{desc}（{tag_examples}）")
        
        prompt_parts.extend([
            "",
            "## 注意事项",
            "- 如果某个维度无法识别，返回空数组或空字符串",
            "- 标签必须从标准标签库中选择，不要自创标签",
            "- 优先识别最明显的内容，避免过度解读",
            "- 输出必须是严格的 JSON 格式，不要添加额外说明"
        ])
        
        # 保存 prompt 到配置
        config = ConfigManager()
        config.set_vlm_prompt("\n".join(prompt_parts))
        logger.info(f"已自动重建 VLM prompt，包含 {len(category_tags)} 个分类")
        
    finally:
        if close_db:
            db.close()


app = FastAPI(title="AI Video Metadata Audit API")

@app.on_event("startup")
async def bind_worker_to_scheduler():
    """在应用启动时将 worker 函数绑定到调度器并自动启动"""
    # 启动时执行 schema 迁移（幂等）：确保 actor_faces 等表包含最新列，
    # 避免 ORM 查询遇到 no such column 导致 500（如 /task/batch-delete）
    try:
        from src.db import schema as _schema
        from src.core.path_manager import get_path_manager as _get_pm
        _db_path = str(_get_pm().db_file)
        _schema.init_db(_db_path)
        logger.info("启动时 schema 迁移完成: %s", _db_path)
    except Exception as _mig_exc:
        logger.warning("启动时 schema 迁移失败: %s", _mig_exc)
    _log_config_snapshot()

    scheduler = get_scheduler()
    scheduler._run_task_worker = run_audit_worker_sync

    # 派发前置条件闸门：VLM 未就绪（模型没装/Ollama 没起）⇒ 暂停派发，
    # 任务留在队列等就绪后自动继续；而不是让每条任务各自失败
    try:
        from src.core.config import get_config_manager as _get_cfg
        from src.core.vlm_readiness import DEFAULT_RECHECK_INTERVAL, is_vlm_ready

        _cm = _get_cfg()
        scheduler.set_readiness_check(
            lambda: is_vlm_ready(_cm.get_vlm_config),
            interval_seconds=DEFAULT_RECHECK_INTERVAL,
        )
        logger.info("已安装 VLM 就绪性闸门：未就绪时暂停派发，就绪后自动恢复")
    except Exception as _gate_exc:
        logger.warning("安装 VLM 就绪性闸门失败（分析照旧）: %s", _gate_exc)

    # 自动启动调度器
    await scheduler.start()
    logger.info("调度器已自动启动")

    # 预加载共享人脸识别模型：所有 rebuild worker 经 FaceRecognizer.__init__
    # peek() 复用同一份模型(避免并发 N 时各自初始化整套 AuraFace 导致内存
    # 暴涨/初始化风暴)。首次加载约 6s(CoreML)，可接受。
    try:
        from src.core.model_manager import ModelManager as _MM
        _mm = _MM.get_instance()
        if not _mm.is_available("face_recognition"):
            t0 = __import__("time").time()
            _w = _mm.get("face_recognition")
            if _w is not None and _w.is_available():
                logger.info(
                    "预加载共享人脸识别模型完成 (%.1fs, CoreML)",
                    __import__("time").time() - t0,
                )
            else:
                logger.warning("预加载人脸识别模型失败，worker 将各自初始化")
    except Exception as _preload_exc:
        logger.warning("预加载人脸识别模型异常(不影响后续按需加载): %s", _preload_exc)
    
    # 主题**元数据**以注册表为准（数据文件之外，解自举）；标签仍按主题归属写到库里。
    # 顺序：① 把库里既有主题一次性导出到注册表（迁移遗留）→ ② 播种预置 →
    #       ③ 按注册表的映射表路径导入标签（写 theme_id）
    try:
        from src.core.preset_themes import seed_preset_themes
        from src.core.dictionary import import_mapping_md
        from src.core.theme_registry import get_theme_registry
        from src.db.models import Theme

        reg = get_theme_registry()
        db = get_session(engine)
        try:
            exported = reg.import_from_db(db)          # 幂等：已有同名不覆盖
            if exported:
                logger.info("已把库内 %d 个主题导出到注册表", exported)
            created = seed_preset_themes(reg)
            if created:
                logger.info("预置分析主题播种完成，新建 %d 个", created)

            # **每主题一个数据文件**：只导入**当前主题自己**的映射表。
            # （旧行为是把所有主题的映射都导进同一个库并写 theme_id，现在那样会污染本库）
            current = reg.effective_current()
            rec = reg.get(current) or {}
            md_path = rec.get("label_mapping_path")
            imported = 0
            if md_path and Path(md_path).exists():
                imported = import_mapping_md(db, md_path)   # 不传 theme_id：本库即本主题
                # 兼容：仍保留 themes 桥接行（Phase 4 清理）
                row = db.query(Theme).filter(Theme.name == current).first()
                if row is None:
                    db.add(Theme(name=current))
                db.commit()
            logger.info("当前主题「%s」标签导入完成，共 %d 个标签", current, imported)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"自动导入标签库失败: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Include API Routers ────────────────────────────────────────────
from src.api.routes import tags, videos, actors
app.include_router(tags.router)
app.include_router(videos.router)
app.include_router(actors.router)

# 影楼基础包 S3.2：客单/活动/类型词表 API
from src.api.routes.studio import router as studio_router
app.include_router(studio_router)

# 标签主题系统（Phase 3）
from src.api.routes import themes as themes_router
app.include_router(themes_router.router)

# ── Database initialization ────────────────────────────────────────
class DatabaseInitRequest(BaseModel):
    path: str

@app.post("/database/init")
async def init_database(request: DatabaseInitRequest):
    """Initialize database at specified path."""
    try:
        db_path = Path(request.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database with all tables
        init_db(str(db_path))  # 初始化建表；engine 由各请求路径自建
        
        # Update config to use new database path
        config_manager.set("database.path", str(db_path))
        
        return {
            "status": "success",
            "message": f"数据库已初始化: {db_path}",
            "path": str(db_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库初始化失败: {str(e)}")

# Global state/config
from src.core.path_manager import get_path_manager
path_manager = get_path_manager()

# 主题工作区初始化：为"当前主题"建档（目录/config.yaml/faces）。
# 必须在解析 DB / 配置路径之前完成；注册表为空时回退旧布局（见 theme_workspace）。
# 测试通过 VIDEO_TAG_THEME_AUTOCREATE=0 关闭。
from src.core.theme_workspace import bootstrap_current_theme
bootstrap_current_theme()

DB_PATH = str(path_manager.db_file)
engine = init_db(DB_PATH)

# Initialize ConfigManager（共享实例：写配置必须与读配置落在同一对象上，
# 否则切分析主题后 /config 与分析链路仍读到旧值）
config_manager = get_config_manager()

# 词典别名映射 MD（Obsidian 维护）:配置驱动,不再硬编码个人路径;
# 未配置时 DictionaryManager 读不到文件 → 回退 DB 内别名,行为安全
MD_MAPPING_PATH = config_manager.get("dictionary.mapping_md_path", "")

def _log_config_snapshot() -> None:
    """启动配置快照（任务 2.3：关键键位一目了然,配置漂移当场可见）。

    在 FastAPI startup 事件调用——模块导入期执行会早于日志 handler 挂载,
    打包态该行会丢失(实测)。
    """
    try:
        _vc = config_manager.get_video_config()
        from src.core.perf_tiers import detect_tier
        logger.info(
            "[配置快照] db=%s | video=%s | tier=%s | embedded=%s",
            DB_PATH,
            {k: _vc.get(k) for k in ("frames_per_scene", "max_frames_per_video", "frame_density")},
            detect_tier(config_manager),
            bool((config_manager.get("embedded_inference", {}) or {}).get("enabled")) or getattr(sys, "frozen", False),
        )
    except Exception as _snap_exc:  # noqa: BLE001 - 快照失败不阻塞启动
        logger.warning(f"[配置快照] 生成失败(忽略): {_snap_exc}")


def _api_faces_dir() -> Path:
    """返回 faces_dir 绝对路径（唯一解析入口，见 src/core/face_paths.py）。

    收口理由：此前 API 侧、路由侧与 FaceRecognizer 各有一份解析，且相对路径
    有"回退项目根"与"按 CWD"两种口径，同一个值可能解析到不同目录。
    """
    from src.core.face_paths import resolve_faces_dir

    return resolve_faces_dir(get_config=config_manager.get)


def _resolve_face_file(face_path: Optional[str]) -> Optional[Path]:
    """把 actor_faces.face_path 解析为磁盘绝对路径（兼容相对/绝对/项目根相对）。"""
    if not face_path:
        return None
    p = Path(str(face_path)).expanduser()
    if p.is_absolute():
        return p
    faces_dir = _api_faces_dir()
    candidate = faces_dir / p
    if candidate.exists():
        return candidate
    root_candidate = Path(__file__).resolve().parent.parent.parent / p
    if root_candidate.exists():
        return root_candidate
    return candidate


def _actor_cover_abs(db, actor) -> Optional[str]:
    """人物封面的**绝对**路径（供 UI 直接渲染）。

    actors.cover_face 在库里存的是相对 faces_dir 的路径（如 ACTOR_001/face_0.jpg），
    前端拿到后 os.path.exists 判定必然失败 → 只显示占位图标。这里统一解析为绝对路径。

    封面文件缺失时（重建/合并后常见）回退到该人物任意一张存在的人脸，
    避免返回死路径。
    """
    if actor is None:
        return None
    _p = _resolve_face_file(actor.cover_face)
    if _p is not None and _p.exists():
        return str(_p)
    _faces = (
        db.query(ActorFace)
        .filter(ActorFace.actor_id == actor.id)
        .order_by(ActorFace.is_cover.desc(), ActorFace.quality_score.desc())
        .all()
    )
    for _f in _faces:
        _fp = _resolve_face_file(_f.face_path)
        if _fp is not None and _fp.exists():
            return str(_fp)
    return None


def _delete_actor_dir_on_disk(actor_code: Optional[str]) -> None:
    """删除人物磁盘目录（删除 actor 后调用，保证目录与 DB 一一对应）。"""
    if not actor_code:
        return
    try:
        faces_dir = _api_faces_dir()
        target = (faces_dir / actor_code).resolve()
        if target.is_relative_to(faces_dir.resolve()) and target.exists():
            shutil.rmtree(target)
            logger = logging.getLogger(__name__)
            logger.info("已删除人物磁盘目录: %s", target)
    except Exception as e:
        logging.getLogger(__name__).warning("删除人物磁盘目录失败 %s: %s", actor_code, e)


# ── Pydantic models ──────────────────────────────────────────────

class ImportRequest(BaseModel):
    directory: Optional[str] = None
    files: Optional[List[str]] = None

class ConfirmRequest(BaseModel):
    task_id: int
    tags: List[str]

class BatchStartRequest(BaseModel):
    task_ids: List[int]
    force: bool = False

class BatchDeleteRequest(BaseModel):
    task_ids: List[int]

class DictionaryCreateRequest(BaseModel):
    name: str
    category: str = "其他"
    aliases: List[str] = []
    source: Optional[str] = None
    theme_id: Optional[int] = None

class DictionaryUpdateRequest(BaseModel):
    aliases: Optional[List[str]] = None
    category: Optional[str] = None
    theme_id: Optional[int] = None

class DictionaryImportRequest(BaseModel):
    content: str

class CategoryCreateRequest(BaseModel):
    name: str

class AuditBulkRequest(BaseModel):
    action: str  # "confirm" | "reject" | "delete"
    item_ids: Optional[List[int]] = None  # Specific items to process, None means all pending
    target_tag: Optional[str] = None  # Manual target tag for confirm action

class ConfigUpdateRequest(BaseModel):
    value: Any

class MergeActorsRequest(BaseModel):
    actor_ids: List[int]

class SetCoverRequest(BaseModel):
    """人工指定人物封面: face_id 为 actor_faces.id。"""
    face_id: int

class RenameActorRequest(BaseModel):
    name: str

class BatchDeleteActorsRequest(BaseModel):
    actor_ids: List[int]

# --- File Explorer API ---
@app.get("/browser")
async def browse_directory(path: str = "/"):
    try:
        p = Path(path)
        if not p.exists():
            return {"error": "Path does not exist", "status": 404}
        if not p.is_dir():
             return {"error": "Path is not a directory", "status": 400}
        
        items = []
        for entry in os.scandir(path):
            items.append({
                "name": entry.name,
                "path": entry.path,
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if not entry.is_dir() else 0
            })
        
        # Sort results: directories first, then files
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        return {"items": items, "current_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 导入进度（供前端轮询 /task/import/status）──────────────────────────
# 导入是同步 def（跑在 FastAPI 线程池），这里用进程内共享记录暴露实时进度，
# 让前端把静态的「导入中...」换成「N/M」。后端单进程（uvicorn --workers 1），
# 导入线程写、接口线程读，用锁保护以取得一致快照。
_import_progress_lock = threading.Lock()
_import_progress: Dict[str, Any] = {
    "active": False,        # 是否有导入正在进行
    "total": 0,             # 本次待处理文件数
    "processed": 0,         # 已遍历文件数
    "imported": 0,          # 新增作业数
    "duplicate": 0,         # 跳过数（路径已有未结束作业）
    "failed": 0,            # 失败数
    "started_at": None,     # epoch 秒
    "finished_at": None,
}


def _update_import_progress(**fields) -> None:
    with _import_progress_lock:
        _import_progress.update(fields)


def _snapshot_import_progress() -> Dict[str, Any]:
    with _import_progress_lock:
        return dict(_import_progress)


def _commit_batch_with_lock_retry(db, batch, attempts: int = 5, base_delay: float = 0.5) -> None:
    """提交一批作业；遇 SQLite「database is locked」退避重试。

    SQLite 的写-写是串行的：并发的分析 worker / 人脸重建会与导入短暂互斥。
    `busy_timeout` 已设 30s，但长事务仍可能超时。一次瞬时锁**不该**丢掉整批
    （最多 `IMPORT_COMMIT_BATCH` 条）已登记的作业 —— 失败即回滚重来。

    注意：`db.commit()` 抛错后 session 需 `rollback()` 才能复用，而 rollback 会丢弃
    本批尚未提交的对象，所以重试前要用 `batch` 重新 `add()`。
    """
    for attempt in range(attempts):
        try:
            db.commit()
            return
        except OperationalError as e:
            if not is_locked_error(e):
                db.rollback()
                raise
            db.rollback()
            if attempt == attempts - 1:
                note_lock_failure()
                logger.error("导入提交连续 %d 次遇 database is locked，放弃本批", attempts)
                raise
            note_lock_retry()
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            logger.warning(
                "导入提交遇 database is locked，%.2fs 后重试（第 %d/%d 次）",
                delay, attempt + 1, attempts,
            )
            time.sleep(delay)
            for task in batch:
                db.add(task)


@app.post("/task/import")
def import_videos(request: ImportRequest, background: BackgroundTasks):
    """Import videos using file_entity-based logic.

    **导入是轻量操作**：只登记路径与轻量元数据（大小/时长/EXIF），
    **不读文件内容**。实测 5365 个文件 / 81.7 GB 的路径登记仅需 0.18 秒；
    而逐个算 MD5 需要读完整内容（同批约 16.7 分钟，相差约 5500 倍）。
    因此真实 MD5 **后置到分析首步**计算并回填，重复内容的判定与「跳过业务」
    也一并发生在分析阶段（见 openspec/changes/separate-import-from-analysis/design.md）。

    **刻意声明为同步 def（不是 async def）**：本接口要遍历成百上千个文件并写库，
    即便已轻量化，仍可能耗时数秒到数十秒。async def 会**跑在事件循环里**，
    把整个后端堵死 —— 表现为前端所有轮询超时、"后端断开"、界面卡在"导入中"。
    用 def 时 FastAPI 会把它丢到线程池执行，事件循环保持空闲。
    """
    import_start = time.time()
    db = get_session(engine)
    try:
        # ── 1. Collect target files ──────────────────────────────────
        target_files: List[str] = []
        if request.directory:
            video_dir = Path(request.directory)
            if not video_dir.exists() or not video_dir.is_dir():
                raise HTTPException(status_code=400, detail="Invalid directory path")
            # 遍历时**剪枝**：不进入抽帧目录（<视频名>_frames/），否则每一帧（.jpg）
            # 都会被当成照片逐张登记（2026-09 用户报告）。判定见 cache_manager.is_frames_dir。
            for root, dirs, files in os.walk(video_dir):
                dirs[:] = [d for d in dirs if not is_frames_dir(os.path.join(root, d))]
                for name in files:
                    if Path(name).suffix.lower() in ALL_MEDIA_EXTENSIONS:
                        target_files.append(str(Path(root, name).absolute()))
        elif request.files:
            # 防御：即便前端直接传 files（UI 走的就是这条路），也剔除抽帧目录里的文件
            target_files = [fp for fp in request.files if not is_inside_frames_dir(fp)]

        # ── 免费版额度守卫（免费版累计 100 个文件；完整版不受限）──────────
        # 规则见 docs/界面设计/07-授权管理.md §1.3：剩余不足时部分导入；额度为 0 整批拒绝。
        try:
            from src.licensing import quota as _license_quota
            from src.licensing import state as _license_state
        except ImportError:
            from licensing import quota as _license_quota
            from licensing import state as _license_state
        _quota_limited = False
        if not _license_state.is_pro():
            _remaining = _license_quota.remaining()
            if _remaining <= 0:
                raise HTTPException(status_code=403, detail={
                    "error": "quota_exceeded",
                    "message": "免费额度已用完（100/100）。已有数据不受影响；激活后可无限制处理。",
                })
            if len(target_files) > _remaining:
                target_files = target_files[:_remaining]
                _quota_limited = True

        # 初始化导入进度（供前端轮询展示 N/M）
        _update_import_progress(
            active=True, total=len(target_files), processed=0,
            imported=0, duplicate=0, failed=0,
            started_at=time.time(), finished_at=None,
        )

        imported_count = 0
        duplicate_count = 0
        failed_files: List[str] = []
        pending_batch: List[Task] = []  # 本批尚未提交的作业（锁重试时需重新 add）
        imported_ids: List[int] = []    # 已提交的新作业 id（供后台并行补探时长）

        # 已有「未结束作业」的路径集合：一次查出，避免逐文件查库
        busy_paths = {
            row[0]
            for row in db.query(Task.file_path).filter(
                Task.file_path.isnot(None),
                Task.status.in_(NON_TERMINAL_TASK_STATUSES),
            ).all()
        }

        for idx, fp in enumerate(target_files, 1):
            _update_import_progress(processed=idx)
            try:
                p = Path(fp)
                if not p.exists() or not p.is_file():
                    continue
                suffix = p.suffix.lower()
                if suffix not in ALL_MEDIA_EXTENSIONS:
                    continue

                # 使用原始路径进行所有文件操作
                original_fp = str(p)

                # 同一路径已有「未结束的作业」⇒ 跳过（允许重复内容，不重复排队同一路径）
                if original_fp in busy_paths:
                    duplicate_count += 1
                    _update_import_progress(duplicate=duplicate_count)
                    continue

                # ── 2. 轻量元数据（刻意不读文件内容！）───────────────
                # 只取文件大小。**时长惰性获取**：导入不跑 ffprobe（实测约 48ms/文件，
                # 661 个文件 ≈ 32s），改在「激活」时补探（见 batch_start_tasks）与
                # 分析首步兜底（resolve_content_identity）。内容身份（MD5）同理后置。
                file_size = os.path.getsize(original_fp)
                duration = None

                # ── 3. 只登记作业 ────────────────────────────────────
                # 不建 file_entity / file_location：entity 表示「内容」，按 MD5 唯一，
                # 必须在内容身份已知（分析首步算出 MD5）时才创建。
                # 见 openspec/changes/separate-import-from-analysis/design.md §3
                task = Task(
                    entity_id=None,
                    file_path=original_fp,
                    file_md5=None,
                    file_size=file_size,
                    duration=duration,
                    status='PENDING',
                )
                db.add(task)
                pending_batch.append(task)
                imported_count += 1
                busy_paths.add(original_fp)
                _update_import_progress(imported=imported_count)

                # 分批提交：进度可感知、可中断，且不长时间持有 SQLite 写锁。
                # 遇锁退避重试（回滚后重新 add 本批），避免整批被一次瞬时锁丢掉。
                if imported_count % IMPORT_COMMIT_BATCH == 0:
                    _commit_batch_with_lock_retry(db, pending_batch)
                    imported_ids.extend(t.id for t in pending_batch if t.id is not None)
                    pending_batch = []

            except Exception as e:
                logger.error(f"导入文件失败 {fp}: {e}")
                db.rollback()  # 单条失败不得污染已提交的批次与后续文件
                pending_batch = []  # 回滚后本批对象已失效
                failed_files.append(fp)
                _update_import_progress(failed=len(failed_files))
                continue

        _commit_batch_with_lock_retry(db, pending_batch)
        imported_ids.extend(t.id for t in pending_batch if t.id is not None)
        # 导入响应先返回（秒级）；时长在后台并行补探，让「未启动」列表仍有「时长」。
        background.add_task(_backfill_durations, imported_ids)
        logger.info(
            f"导入完成: 新增 {imported_count}, 跳过(路径已登记) {duplicate_count}, "
            f"失败 {len(failed_files)}, 耗时 {round(time.time() - import_start, 1)}s "
            f"（只登记路径、不读文件内容；真实 MD5 留待分析首步计算）"
        )
        if not _license_state.is_pro():
            _license_quota.register_imported(imported_count)
        response = {
            "status": "success",
            "imported": imported_count,
            "duplicate": duplicate_count,
            "failed": failed_files,
        }
        if not _license_state.is_pro():
            response["quota_limited"] = _quota_limited
            response["quota_remaining"] = _license_quota.remaining()
        return response
    finally:
        # 无论成功/异常，都要让前端停止「导入中」轮询
        _update_import_progress(active=False, finished_at=time.time())
        db.close()


@app.get("/task/folders")
def list_task_folders():
    """任务来源树：按所在目录聚合计数（目录尾名排序由前端做）。"""
    import os
    db = get_session(engine)
    try:
        rows = db.query(Task.file_path).all()
        counts = {}
        for (fp,) in rows:
            d = os.path.dirname(fp or "")
            counts[d] = counts.get(d, 0) + 1
        out = [{"dir": d, "count": n} for d, n in counts.items() if d]
        out.sort(key=lambda x: -x["count"])
        return {"folders": out}
    finally:
        db.close()


@app.get("/task/ids")
def list_task_ids(folder: str = None):
    """按目录取任务 id 列表（来源树右键「新建项目归集」用）。"""
    db = get_session(engine)
    try:
        q = db.query(Task.id)
        if folder:
            q = q.filter(Task.file_path.like(f"{folder}/%"))
        return {"ids": [r[0] for r in q.all()]}
    finally:
        db.close()


@app.get("/task/import/status")
def import_status():
    """导入进度快照（供前端把「导入中...」显示成 N/M）。

    无导入进行时 `active=false`；前端据此停表。
    """
    return _snapshot_import_progress()


def _extract_photo_metadata(file_path: str) -> dict:
    """提取照片元数据（EXIF + 分辨率）。

    Returns:
        {"exif_data": dict 或 None, "capture_time": str 或 None, "resolution": "WxH" 或 None}
    """
    result = {"exif_data": None, "capture_time": None, "resolution": None}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()  # 支持 HEIC 照片
        except ImportError:
            pass

        img = Image.open(file_path)
        w, h = img.size
        result["resolution"] = f"{w}x{h}"

        exif_raw = img._getexif()
        if exif_raw:
            exif = {}
            for tag_id, value in exif_raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime", "Make", "Model", "Software"):
                    if isinstance(value, (str, int, float)):
                        exif[tag] = str(value)
            if exif:
                result["exif_data"] = exif
                result["capture_time"] = exif.get("DateTimeOriginal") or exif.get("DateTime")
        img.close()
    except Exception as e:
        logger.warning(f"照片元数据提取失败 {file_path}: {e}")
    return result


def _probe_video_duration(file_path: str) -> Optional[float]:
    """Use ffprobe to get video duration in seconds. Returns None on failure."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration_str = data.get('format', {}).get('duration')
            if duration_str:
                return float(duration_str)
    except Exception:
        pass
    return None


def _probe_durations(paths: List[str], max_workers: int = 8) -> Dict[str, Optional[float]]:
    """并行探测一批文件的时长。

    ffprobe 是子进程 I/O（实测约 48ms/文件），串行探测 661 个文件要 ~32s；并行能
    显著缩短。返回 {path: 时长或 None}。
    """
    result: Dict[str, Optional[float]] = {}
    if not paths:
        return result
    with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as ex:
        futures = {ex.submit(_probe_video_duration, p): p for p in paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                result[path] = fut.result()
            except Exception:
                result[path] = None
    return result


async def _fill_missing_durations(db, tasks) -> None:
    """时长惰性获取：为「缺时长」的视频任务并行补探并落库。

    导入不再探时长（ffprobe ~48ms/文件）；激活时补一次以保留「短任务优先」排序。
    只把路径字符串跨线程，工作线程不接触 SQLAlchemy 对象。
    """
    need = [
        t for t in tasks
        if t.duration is None and t.file_path and _media_type_of(t.file_path) == "video"
    ]
    if not need:
        return
    paths = [t.file_path for t in need]
    probed = await asyncio.get_event_loop().run_in_executor(None, _probe_durations, paths)
    for t in need:
        d = probed.get(t.file_path)
        if d is not None:
            t.duration = d
    db.commit()


def _backfill_durations(task_ids: List[int]) -> None:
    """导入响应返回后，后台并行补探这批新作业的时长。

    目的：导入本身不探时长（保持秒级返回），但「未启动」列表仍应显示时长。
    探测失败/超时都只记日志，不影响任何状态。
    """
    if not task_ids:
        return
    db = get_session(engine)
    try:
        tasks = db.query(Task).filter(
            Task.id.in_(task_ids), Task.duration.is_(None)
        ).all()
        need = [t for t in tasks if t.file_path and _media_type_of(t.file_path) == "video"]
        if not need:
            return
        probed = _probe_durations([t.file_path for t in need])
        filled = 0
        for t in need:
            d = probed.get(t.file_path)
            if d is not None:
                t.duration = d
                filled += 1
        db.commit()
        logger.info(f"后台补探时长完成: {filled}/{len(need)} 个文件")
    except Exception as e:
        logger.warning(f"后台补探时长失败（不影响导入）: {e}")
    finally:
        db.close()


def _auto_clean_frames_if_enabled(cleanup_config: dict, frames_path: Optional[str], task_id: int) -> None:
    """分析完成后按配置清理该作品的抽帧目录。

    14 号定稿（引用保留模型）：**标签证据帧与封面帧永不删除**——
    只删除未被 ``tag_frame_mappings`` 引用的中间帧；目录清空后移除。
    配置键：``frame_cleanup.enabled`` / ``frame_cleanup.keep_days``
    - 未启用 ⇒ 不动
    - ``keep_days == 0`` ⇒ 立即清理（保留引用帧）
    - ``keep_days > 0`` ⇒ 仅当抽帧目录**已超过该天数**才清理
    """
    if not cleanup_config.get("enabled", False):
        return
    if not frames_path or not os.path.exists(frames_path):
        return
    try:
        keep_days = int(cleanup_config.get("keep_days", 0) or 0)
    except (TypeError, ValueError):
        keep_days = 0
    try:
        if keep_days > 0:
            age_days = (time.time() - os.path.getmtime(frames_path)) / 86400
            if age_days <= keep_days:
                return

        # 引用保留：收集该任务所有标签映射引用的帧路径，绝不删除
        from src.db.models import TagFrameMapping
        referenced = set()
        db = get_session(engine)
        try:
            mappings = db.query(TagFrameMapping).filter(
                TagFrameMapping.task_id == task_id).all()
            for m in mappings:
                if m.frame_path:
                    referenced.add(os.path.normpath(m.frame_path))
        finally:
            db.close()

        removed = kept = 0
        for root, _dirs, files in os.walk(frames_path):
            for fn in files:
                fp = os.path.normpath(os.path.join(root, fn))
                if fp in referenced:
                    kept += 1
                    continue
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    pass
        # 目录空了才移除；仍有引用帧则保留目录
        if not any(os.scandir(frames_path)):
            os.rmdir(frames_path)
        logger.info(
            f"任务 {task_id} 抽帧缓存清理：删除 {removed} 张中间帧，"
            f"保留 {kept} 张引用帧（标签证据/封面）: {frames_path}")
    except Exception as e:
        logger.warning(f"任务 {task_id} 抽帧缓存清理失败: {e}")

@app.get("/status")
async def get_status():
    db = get_session(engine)
    try:
        tasks = db.query(Task).all()
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == 'COMPLETE')
        processing = sum(1 for t in tasks if t.status in (
            'CHECKING', 'SIFTING', 'ANALYZING', 'CLUSTERING', 'PROCESSING',
            'WRITING'))
        # 待处理 = QUEUED(已勾选并点击「分析」、进入调度队列的任务);
        # PENDING(新导入未激活)/ PAUSED(暂停)不计入, 单独在筛选「未启动/已暂停」查看。
        pending = sum(1 for t in tasks if t.status == 'QUEUED')
        # 未排队 = 兜底桶（用户 2026-09-29 裁定）：总数减 已完成/处理中/待处理，
        # 剩余全部落入（PENDING/ERROR/DUPLICATE/PAUSED 及未来新状态），
        # 保证四张卡片之和恒等于总任务
        unqueued = max(0, total - completed - processing - pending)

        overall_progress = int((completed / total * 100) if total > 0 else 0)

        return {
            "total_tasks": total,
            "completed": completed,
            "processing": processing,
            "pending": pending,
            "unqueued": unqueued,
            "overall_progress": overall_progress
        }
    finally:
        db.close()

def _project_subtree_ids(db, project_id: int) -> list:
    """项目节点 → 含全部子级的项目 id 集合（子树口径：点父级含子级）。

    project-assets-roster：任务页/详情页的项目过滤统一走此口径（嵌套为
    分组视图，父级视角=子树成员汇总）。"""
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
    return ids


def _subtree_roster(db, project_id: int):
    """项目节点子树的成员哈希集合（空集=该项目无任何成员）。"""
    proj_ids = _project_subtree_ids(db, project_id)
    if not proj_ids:
        return set()
    return {r[0] for r in db.query(ProjectAsset.file_md5).filter(
        ProjectAsset.project_id.in_(proj_ids)).all()}


@app.get("/videos")
def get_videos(project_id: int = None):
    """获取所有已分析的视频文件列表（基于 FileAnalysis）。

    **刻意用同步 def**：封面解析会读帧图片（判黑/择优），属阻塞 IO；
    async def 会把事件循环堵住（1500+ 条时可达数秒），导致其它请求超时。
    交给 FastAPI 线程池执行。
    用于"分析结果"页面，数据持久化，不随任务删除而消失。

    project-assets-roster：可选 project_id 筛选（经归属清单，子树口径）。
    """
    db = get_session(engine)
    try:
        # 查询所有有分析记录的文件实体（批量预载 entity+locations，
        # 并批量取 faces/actors/tasks——原逐行懒加载 6179 行 × 6 查询 = 39.6s）
        from sqlalchemy.orm import joinedload
        analyses = (db.query(FileAnalysis)
                    .options(joinedload(FileAnalysis.entity)
                             .selectinload(FileEntity.locations))
                    .order_by(FileAnalysis.updated_at.desc())
                    .all())

        all_entity_ids = [fa.entity_id for fa in analyses if fa.entity_id]
        faces_by_entity = {}
        actor_id_set = set()
        if all_entity_ids:
            for af in db.query(ActorFace).filter(
                    ActorFace.entity_id.in_(all_entity_ids)).all():
                faces_by_entity.setdefault(af.entity_id, []).append(af)
                if af.actor_id:
                    actor_id_set.add(af.actor_id)
        actors_by_id = {}
        if actor_id_set:
            for a in db.query(Actor).filter(Actor.id.in_(actor_id_set)).all():
                actors_by_id[a.id] = a
        # project-assets-roster：项目过滤经归属清单（子树口径，锚内容 md5）。
        # 旧 task.project_id 口径废弃（影子列停写）。
        if project_id:
            _md5s = _subtree_roster(db, project_id)
            _roster_entity_ids = (
                {e.id for e in db.query(FileEntity.id).filter(
                    FileEntity.file_md5.in_(_md5s)).all()} if _md5s else set())
            analyses = [fa for fa in analyses if fa.entity_id in _roster_entity_ids]

        # 归属聚合：md5 → [project_id]（多归属；project_id 字段取首项兼容旧 UI）
        _proj_by_md5 = {}
        for pa in db.query(ProjectAsset).all():
            _proj_by_md5.setdefault(pa.file_md5, []).append(pa.project_id)

        # 标签证据帧（FM 要求：封面帧必须包含标签真实内容）——
        # frame_path 即打标签时的证据画面，按实体取一帧（分块 IN 防参数超限）
        all_entity_ids = [fa.entity_id for fa in analyses if fa.entity_id]
        tag_frame_by_entity = {}
        for i in range(0, len(all_entity_ids), 900):
            chunk = all_entity_ids[i:i + 900]
            for m in db.query(TagFrameMapping.entity_id, TagFrameMapping.frame_path).filter(
                    TagFrameMapping.entity_id.in_(chunk),
                    TagFrameMapping.frame_path.isnot(None)).all():
                if m.entity_id not in tag_frame_by_entity:
                    tag_frame_by_entity[m.entity_id] = m.frame_path

        result = []
        _cover_abs_cache = {}  # (actor_id, cover_face) → 绝对路径；同人物跨文件复用
        for fa in analyses:
            entity = fa.entity
            if not entity:
                continue

            # 获取文件路径
            location = next((loc for loc in entity.locations if loc.is_primary == 1), entity.locations[0] if entity.locations else None)
            file_path = location.file_path if location else None
            file_name = Path(file_path).name if file_path else None
            folder_path = str(Path(file_path).parent) if file_path else None

            # 人物列表（faces/actors 已批量预取；封面绝对路径用缓存解析器）
            actor_faces = faces_by_entity.get(entity.id, [])
            actor_ids = set([af.actor_id for af in actor_faces if af.actor_id])
            actors_list = []
            for actor_id in actor_ids:
                actor = actors_by_id.get(actor_id)
                if actor:
                    ck = (actor_id, actor.cover_face)
                    if ck not in _cover_abs_cache:
                        _cover_abs_cache[ck] = _actor_cover_abs(db, actor)
                    actors_list.append({
                        "id": actor.id,
                        "code": actor.code,
                        "name": actor.name,
                        "cover_face": _cover_abs_cache[ck],
                    })

            # 解析标签
            mapped_tags = []
            if fa.mapped_tags:
                try:
                    mapped_tags = json.loads(fa.mapped_tags)
                except:
                    mapped_tags = []

            # 封面：视频优先标签证据帧（与标签内容天然一致，FM 要求）；
            # 无证据帧回退 frames_path（前端卡片自选首帧）
            media_type = entity.media_type or "video"
            cover_path = None
            if media_type == "photo":
                cover_path = file_path
            else:
                tag_frame = tag_frame_by_entity.get(entity.id)
                if tag_frame and os.path.exists(tag_frame):
                    cover_path = tag_frame

            # 影楼归属（project-assets-roster）：内容所在项目清单（多归属）
            _v_projects = _proj_by_md5.get(entity.file_md5, [])

            result.append({
                "id": fa.id,
                "entity_id": entity.id,
                "task_id": fa.task_id,  # 用于重建人脸等需要 Task.id 的操作
                "project_ids": _v_projects,
                "project_id": _v_projects[0] if _v_projects else None,  # 兼容旧 UI
                "file_path": file_path,
                "file_name": file_name,
                "folder_path": folder_path,
                "file_size": entity.file_size,
                "duration": entity.duration,
                "media_type": media_type,
                "resolution": entity.resolution,
                "status": fa.status,  # READY, CONFIRMED, ERROR
                "progress": fa.progress,
                "mapped_tags": mapped_tags,
                "frames_path": fa.frames_path,
                "nfo_path": fa.nfo_path,
                "cover_path": cover_path,
                "face_scan_status": fa.face_scan_status,  # 文件维度人脸扫描状态（持久）
                "actors": actors_list,
                "created_at": fa.created_at.isoformat() if fa.created_at else None,
                "updated_at": fa.updated_at.isoformat() if fa.updated_at else None,
            })
        return result
    finally:
        db.close()


@app.delete("/videos/{analysis_id}")
async def delete_video(analysis_id: int):
    """删除分析结果及关联的文件实体。
    会级联删除 FileEntity、FileLocation、ActorFace 等关联数据。
    """
    from src.db.models import TagFrameMapping
    db = get_session(engine)
    try:
        fa = db.query(FileAnalysis).filter(FileAnalysis.id == analysis_id).first()
        if not fa:
            raise HTTPException(status_code=404, detail="分析记录不存在")

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

        db.commit()
        return {"success": True, "message": "分析记录已删除"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
    finally:
        db.close()


@app.post("/videos/batch-delete")
async def batch_delete_videos(request: dict):
    """批量删除分析记录。
    请求体: {"ids": [1, 2, 3]}
    """
    from src.db.models import TagFrameMapping
    ids = request.get("ids", [])
    if not ids:
        return {"success": False, "message": "未提供要删除的记录ID"}
    
    db = get_session(engine)
    deleted_count = 0
    failed_count = 0
    
    try:
        for analysis_id in ids:
            try:
                fa = db.query(FileAnalysis).filter(FileAnalysis.id == analysis_id).first()
                if not fa:
                    failed_count += 1
                    continue

                entity_id = fa.entity_id

                # 删除 FileAnalysis
                db.delete(fa)

                # 删除关联的 FileEntity
                if entity_id:
                    entity = db.query(FileEntity).filter(FileEntity.id == entity_id).first()
                    if entity:
                        # 解除 tasks 关联
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
                failed_count += 1
                continue
        
        db.commit()
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "message": f"成功删除 {deleted_count} 条记录"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")
    finally:
        db.close()


@app.post("/videos/delete")
async def delete_videos_with_options(request: dict):
    """移除工单（project-assets-roster 2026-09-30：删除=仅移除工单）。

    请求体: {"task_ids": [1,2,3]}
    删除 = Task 行 + 工单级关联（analysis_logs / task_actors / tag_frame_mappings
    的该工单维度）。内容实体、分析结果、磁盘文件**全部保留**——同内容重导
    自动继承既有分析结果与项目归属（归属锚 file_md5，见 project_assets）。
    档案级清理不再由本入口承担（如需保留，另行立项为独立显式动作）。
    """
    task_ids = request.get("task_ids", [])
    if not task_ids:
        return {"success": False, "message": "未提供要移除的任务ID"}

    db = get_session(engine)
    deleted_count = 0
    try:
        for task_id in task_ids:
            db.execute(
                text("DELETE FROM analysis_logs WHERE task_id = :tid"),
                {"tid": task_id},
            )
            db.execute(
                text("DELETE FROM task_actors WHERE task_id = :tid"),
                {"tid": task_id},
            )
            db.execute(
                text("DELETE FROM tag_frame_mappings WHERE task_id = :tid"),
                {"tid": task_id},
            )
            removed = (
                db.query(Task)
                .filter(Task.id == task_id)
                .delete(synchronize_session=False)
            )
            deleted_count += 1 if removed else 0

        db.commit()
        return {
            "success": True,
            "deleted_count": deleted_count,
            "failed_count": 0,
            "file_errors": [],
            "message": f"已移除 {deleted_count} 个工单（内容与分析结果已保留）",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"移除失败: {str(e)}")
    finally:
        db.close()


@app.post("/videos/cleanup-stale")
async def cleanup_stale_videos():
    """清理所有文件已不存在的失效分析记录。
    遍历所有 FileAnalysis，检查其关联的 file_path 是否仍存在于磁盘。
    """
    from src.db.models import TagFrameMapping
    db = get_session(engine)
    try:
        analyses = db.query(FileAnalysis).all()
        deleted_count = 0

        for fa in analyses:
            entity = fa.entity
            if not entity:
                db.delete(fa)
                deleted_count += 1
                continue

            # 检查文件是否存在
            location = next((loc for loc in entity.locations if loc.is_primary == 1), entity.locations[0] if entity.locations else None)
            file_path = location.file_path if location else None

            if file_path and os.path.exists(file_path):
                continue  # 文件存在，跳过

            # 文件不存在，清理
            entity_id = fa.entity_id
            db.delete(fa)

            if entity_id:
                entity_to_delete = db.query(FileEntity).filter(FileEntity.id == entity_id).first()
                if entity_to_delete:
                    # 解除 tasks 关联
                    tasks = db.query(Task).filter(Task.entity_id == entity_id).all()
                    for task in tasks:
                        task.entity_id = None
                    # 解除 tag_frame_mappings 关联
                    mappings = db.query(TagFrameMapping).filter(TagFrameMapping.entity_id == entity_id).all()
                    for m in mappings:
                        m.entity_id = None
                    db.delete(entity_to_delete)

            deleted_count += 1

        db.commit()
        return {"success": True, "deleted_count": deleted_count, "message": f"已清理 {deleted_count} 条失效记录"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")
    finally:
        db.close()


# macOS TCC 保护目录（家目录下的一级目录）: 对其做 stat/glob 会触发系统授权弹窗
_TCC_PROTECTED_TOP_DIRS = {"Desktop", "Documents", "Downloads", "桌面", "文稿", "下载"}


def _is_tcc_protected_path(path) -> bool:
    """路径是否位于 macOS 受保护目录（桌面 / 文稿 / 下载）。

    为什么需要它：macOS 对这几个目录有 TCC 保护，任何 stat/glob 都会弹出
    「MediaTag 想访问桌面文件夹」之类的系统授权框。本应用取封面时**只是"猜"**
    帧目录位置（`{视频名}_frames` 与视频同级），不该为此去探测用户的受保护目录
    —— 真正的帧目录以 DB 里的 frames_path 为准（调用方的第一优先分支）。
    """
    try:
        p = Path(str(path)).expanduser()
        rel = p.relative_to(Path.home())
    except Exception:
        return False
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in _TCC_PROTECTED_TOP_DIRS:
        return True
    # iCloud Drive: ~/Library/Mobile Documents/... (含 com~apple~CloudDocs/Desktop|Documents)
    if len(parts) >= 2 and parts[0] == "Library" and parts[1] == "Mobile Documents":
        return True
    return False


# 「全黑帧」判定阈值，与前端 results_page.py 的显示判定保持一致（< 10 视为全黑）
_BLACK_FRAME_MEAN = 10

# 帧目录 → 选中的封面帧 缓存。key 含目录 mtime，重新分析生成新帧后会自动失效。
# 没有它的话，每次 /videos（1500+ 条）都要重新扫帧判黑，代价约 2.5s。
_COVER_FRAME_CACHE: Dict[str, Optional[str]] = {}


def _frame_is_too_dark(path: str) -> bool:
    """帧是否近似全黑。

    用 1/8 降采样灰度解码：实测比全分辨率快约 2 倍，且"是否全黑"的判定
    完全一致（400 张对照 0 不一致）。
    """
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            return True
        return float(img.mean()) < _BLACK_FRAME_MEAN
    except Exception:
        # 检测本身失败时不要因此丢封面，按可用处理
        return False


def _pick_cover_frame(frames_dir: Path) -> Optional[str]:
    """在帧目录里挑一张封面帧：**优先第一张非全黑的**。

    只取每个场景的第一帧（原实现）会大量命中视频开头的黑场/淡入帧 ——
    实测 1563 条里有 124 条因此显示「No Cover (全黑)」，而其中 124/124 都能
    靠"往后找一张非全黑的"救回。查找顺序：按场景顺序逐帧试，最后才退化到
    第一帧（整段全黑时至少给个路径）。
    """
    try:
        key = f"{frames_dir}|{int(frames_dir.stat().st_mtime)}"
    except OSError:
        key = str(frames_dir)
    if key in _COVER_FRAME_CACHE:
        return _COVER_FRAME_CACHE[key]

    candidates: List[Path] = []
    for unit_dir in sorted(frames_dir.glob("unit_*")):
        if unit_dir.is_dir():
            candidates.extend(sorted(unit_dir.glob("frame_*.jpg")))
    if not candidates:
        candidates = sorted(frames_dir.glob("frame_*.jpg"))

    picked: Optional[str] = None
    for f in candidates:
        if not _frame_is_too_dark(str(f)):
            picked = str(f)
            break
    if picked is None and candidates:
        picked = str(candidates[0])  # 整段都黑 → 保持原行为

    _COVER_FRAME_CACHE[key] = picked
    return picked


def _resolve_cover_path(file_path: str, frames_path: str = None, task_id: int = None) -> Optional[str]:
    """从文件路径或 frames_path 解析封面图片路径（挑非全黑帧，见 _pick_cover_frame）。"""
    # 优先使用 frames_path
    if frames_path and os.path.isdir(frames_path):
        picked = _pick_cover_frame(Path(frames_path))
        if picked:
            return picked

    # 回退：从 task_id 推导 output/frames/task_{task_id}/ 目录
    if task_id:
        output_frames_dir = Path(f"./output/frames/task_{task_id}")
        if output_frames_dir.exists():
            picked = _pick_cover_frame(output_frames_dir)
            if picked:
                return picked

    # 回退：从 file_path 推导 frames 目录
    # 但**不在** macOS 受保护目录（桌面/文稿/下载）下探测 —— 那会触发系统授权弹窗，
    # 而这里只是"猜"帧目录位置；真正的位置以 DB 的 frames_path 为准（见上方分支）。
    if file_path and not _is_tcc_protected_path(Path(file_path).parent):
        video_path = Path(file_path)
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        if frames_dir.exists():
            picked = _pick_cover_frame(frames_dir)
            if picked:
                return picked

    return None


# 状态筛选: 前端收敛值 → 数据库状态集合。
# 支持逗号分隔多值(如 "pending,processing" = 整个在跑管线), OR 语义。
STATUS_FILTER_MAP = {
    "pending": ["QUEUED"],
    "queued": ["QUEUED"],          # 兼容旧客户端(旧口径「排队中」)
    "unqueued": ["PENDING", "ERROR", "FAILED", "DUPLICATE", "PAUSED"],
    # 未排队=兜底桶（2026-09-29 裁定）：非 完成/处理中/排队 的全部落这里
    "not_started": ["PENDING", "PAUSED"],
    "processing": ["CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "WRITING", "PROCESSING"],
    "complete": ["COMPLETE"],
    "error": ["ERROR", "FAILED"],
    "duplicate": ["DUPLICATE"],
}


def resolve_status_filter(status: Optional[str]) -> List[str]:
    """把前端状态筛选值解析为数据库状态列表(空列表 = 不额外过滤)。

    支持逗号分隔多值, 例如 "pending,processing" 表示"待处理或处理中"。
    "all" / 空 / 全部无法识别 → 返回空(不过滤)。
    """
    if not status or status == "all":
        return []
    keys = [s.strip() for s in str(status).split(",") if s.strip()]
    if not keys or "all" in keys:
        return []
    resolved = set()
    for k in keys:
        resolved.update(STATUS_FILTER_MAP.get(k, []))
    return sorted(resolved)


@app.get("/task")
async def get_tasks(page: int = 1, page_size: int = 50, status: str = None, face_scan_status: str = None, project_id: int = None, folder: str = None):
    """获取任务作业列表（用于 Dashboard 监控）。
    仅返回作业状态，不包含业务数据。
    支持分页和状态过滤。
    folder: 目录前缀过滤（任务来源树；精确匹配所在目录）。
    """
    db = get_session(engine)
    try:
        # 检查调度器状态
        scheduler = get_scheduler()
        scheduler_running = scheduler.status.value == "running"

        # 构建查询
        query = db.query(Task)

        # project-assets-roster：项目过滤经归属清单（子树口径，锚内容 md5）
        if project_id:
            _md5s = _subtree_roster(db, project_id)
            if not _md5s:
                query = query.filter(text("1 = 0"))
            else:
                _roster_entity_ids = {e.id for e in db.query(FileEntity.id).filter(
                    FileEntity.file_md5.in_(_md5s)).all()}
                query = query.filter(or_(
                    Task.entity_id.in_(_roster_entity_ids),
                    Task.file_md5.in_(_md5s)))

        # 任务来源树：按所在目录过滤（dir + '/%' 前缀匹配）
        if folder:
            query = query.filter(Task.file_path.like(f"{folder}/%"))

        # 状态过滤(支持多选: "pending,processing" = 待处理或处理中)
        _statuses = resolve_status_filter(status)
        if _statuses:
            query = query.filter(Task.status.in_(_statuses))

        # 人脸扫描状态过滤(文件维度, 存于 file_analysis; tasks.face_scan_status 是过渡字段全 None)
        # 用 in_ 子查询避免 join 行膨胀; unscanned = fa 记录缺失或 fa.face_scan_status 为空
        if face_scan_status and face_scan_status != "all":
            if face_scan_status == "unscanned":
                _scanned_tids = db.query(FileAnalysis.task_id).filter(
                    FileAnalysis.face_scan_status.in_(["complete", "none", "skipped", "low_confidence"])
                )
                query = query.filter(~Task.id.in_(_scanned_tids))
            else:
                _matching_tids = db.query(FileAnalysis.task_id).filter(
                    FileAnalysis.face_scan_status == face_scan_status
                )
                query = query.filter(Task.id.in_(_matching_tids))
        
        # 获取总数
        total = query.count()
        
        # 分页
        offset = (page - 1) * page_size
        # 稳定排序: 按 id 降序(最新导入在前)。历史用 updated_at.desc() 导致
        # 任务状态/人脸识别完成后 updated_at 变化 → 任务跨页移动 → 行漂移,
        # 用户无法稳定观察(2026-09)。id 序不随状态变化, 配合前端表头排序。
        # 置顶任务恒排最前（14 号方案：优先按钮），组内按 id 降序
        tasks = query.order_by(Task.pinned.desc(), Task.id.desc()).offset(offset).limit(page_size).all()
        
        result = []
        for t in tasks:
            # 如果调度器未运行，将运行中和排队状态改为已暂停
            status = t.status
            if not scheduler_running and status in ['CHECKING', 'SIFTING', 'SIFTED', 'ANALYZING', 'CLUSTERING', 'WRITING', 'PROCESSING', 'QUEUED', 'PENDING']:
                status = 'PAUSED'

            _file_path = _get_primary_path(t)
            _file_md5 = t.entity.file_md5 if t.entity else t.file_md5
            _file_size = t.entity.file_size if t.entity else t.file_size
            _duration = t.entity.duration if t.entity else t.duration

            # 获取多人物列表
            task_actors_list = []
            task_actor_recs = db.query(TaskActor).filter(TaskActor.task_id == t.id).all()
            if task_actor_recs:
                for ta in task_actor_recs:
                    actor = db.query(Actor).filter(Actor.id == ta.actor_id).first()
                    if actor:
                        task_actors_list.append({
                            "id": actor.id,
                            "code": actor.code,
                            "name": actor.name,
                        })
            elif t.actor_id:
                # 向后兼容：fallback 到 actor_id
                actor = db.query(Actor).filter(Actor.id == t.actor_id).first()
                if actor:
                    task_actors_list.append({
                        "id": actor.id,
                        "code": actor.code,
                        "name": actor.name,
                    })

            # 人脸扫描状态：优先读 FileAnalysis（文件维度持久，任务清空后仍保留），
            # 没有则回退 Task 上的状态（过渡字段）。
            face_scan_status = t.face_scan_status
            if t.entity_id:
                _fa_for_scan = db.query(FileAnalysis).filter(FileAnalysis.entity_id == t.entity_id).first()
                if _fa_for_scan and _fa_for_scan.face_scan_status:
                    face_scan_status = _fa_for_scan.face_scan_status

            result.append({
                "id": t.id,
                "pinned": bool(getattr(t, "pinned", 0)),
                "file_path": _file_path,
                "file_name": Path(_file_path).name if _file_path else None,
                "folder_path": str(Path(_file_path).parent) if _file_path else None,
                "file_size": _file_size,
                "duration": _duration,
                "status": status,
                "progress": t.progress,
                "error_message": t.error_message,
                "face_scan_status": face_scan_status,
                "actors": task_actors_list,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            })
        return {
            "tasks": result,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    finally:
        db.close()

@app.get("/queue")
async def get_queue():
    db = get_session(engine)
    try:
        tasks = db.query(Task).filter(Task.status != 'COMPLETE').all()
        return [{"id": t.id, "file_path": _get_primary_path(t), "status": t.status} for t in tasks]
    finally:
        db.close()

@app.get("/task/{task_id}")
async def get_task(task_id: int):
    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # 获取 FileAnalysis 记录（业务数据在 FileAnalysis 中）
        file_analysis = None
        if task.entity_id:
            file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()

        # 获取多人物列表
        task_actors_list = []
        task_actor_recs = db.query(TaskActor).filter(TaskActor.task_id == task.id).all()
        if task_actor_recs:
            for ta in task_actor_recs:
                actor = db.query(Actor).filter(Actor.id == ta.actor_id).first()
                if actor:
                    task_actors_list.append({
                        "id": actor.id,
                        "code": actor.code,
                        "name": actor.name,
                    })
        elif task.actor_id:
            actor = db.query(Actor).filter(Actor.id == task.actor_id).first()
            if actor:
                task_actors_list.append({
                    "id": actor.id,
                    "code": actor.code,
                    "name": actor.name,
                })

        return {
            "id": task.id,
            "file_path": _get_primary_path(task),
            "status": task.status,
            "mapped_tags": file_analysis.mapped_tags if file_analysis else None,
            "frames_path": file_analysis.frames_path if file_analysis else None,
            "analysis_details": file_analysis.analysis_details if file_analysis else None,
            "actors": task_actors_list,
        }
    finally:
        db.close()

@app.get("/task/{task_id}/analysis-log")
async def get_task_analysis_log(task_id: int):
    """获取任务的分析日志（VLM 原始输出）"""
    db = get_session(engine)
    try:
        # 查询 analysis_logs 表
        logs = db.query(AnalysisLog).filter(AnalysisLog.task_id == task_id).order_by(AnalysisLog.unit_index).all()
        if not logs:
            raise HTTPException(status_code=404, detail="No analysis log found")
        
        # 读取所有日志文件内容
        result = []
        for log in logs:
            try:
                with open(log.logfile, 'r', encoding='utf-8') as f:
                    content = f.read()
                result.append({
                    'unit_index': log.unit_index,
                    'content': content,
                    'created_at': log.created_at.isoformat()
                })
            except FileNotFoundError:
                continue
        
        return {'logs': result}
    finally:
        db.close()

@app.post("/confirm")
async def confirm_task(request: ConfirmRequest):
    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == request.task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        primary_path = _get_primary_path(task)
        if not primary_path:
            raise HTTPException(status_code=404, detail="File path not found for task")
        video_path = Path(primary_path)
        nfo_path = video_path.with_suffix('.nfo')
        
        xml_content = "<movie>\n"
        for tag in request.tags:
            xml_content += f"  <genre>{tag}</genre>\n"
        xml_content += "</movie>"
        
        with open(nfo_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        task.status = 'COMPLETE'

        # 同步业务数据到 FileAnalysis（持久化核心）
        if task.entity_id:
            file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
            if not file_analysis:
                file_analysis = FileAnalysis(
                    entity_id=task.entity_id,
                    task_id=task.id,
                    status='PENDING',
                    progress=0
                )
                db.add(file_analysis)
                db.flush()
            file_analysis.nfo_path = str(nfo_path)
            # 不覆盖已有 mapped_tags（保留 VLM 分析结果）
            # 只在 mapped_tags 为空时才设置
            if not file_analysis.mapped_tags:
                file_analysis.mapped_tags = json.dumps(request.tags)
            file_analysis.status = 'CONFIRMED'
            file_analysis.progress = 100

        db.commit()
        return {"status": "success", "file_written": str(nfo_path)}
    finally:
        db.close()

# ── Dictionary API (new endpoints) ─────────────────────────────────

@app.get("/dictionary")
async def get_dictionary(category: Optional[str] = None, theme_id: Optional[int] = None):
    """Get all dictionary entries with category, aliases, and file count.

    theme_id: 可选，按主题过滤。None=全部，0=未分类，>0=指定主题（精确匹配）。
    """
    db = get_session(engine)
    try:
        query = db.query(DictionaryEntry)
        if category:
            query = query.filter(DictionaryEntry.category == category)
        if theme_id is not None and theme_id > 0:
            # 指定主题：专属标签 + 通用标签
            # （被多个主题引用的标签在导入时降级为 theme_id=NULL，避免切换主题后标签库残缺）
            query = query.filter(
                or_(
                    DictionaryEntry.theme_id == theme_id,
                    DictionaryEntry.theme_id.is_(None),
                )
            )
        elif theme_id == 0:
            query = query.filter(DictionaryEntry.theme_id.is_(None))
        entries = query.all()

        # 覆盖量为**实时统计**（按内容去重，与标签树/标签浏览口径一致）；
        # 字典表里的 file_count 是历史遗留字段、从不回写，不能作为展示值。
        counts = dict(
            db.execute(
                text("SELECT tag_name, COUNT(DISTINCT entity_id) FROM tag_frame_mappings GROUP BY tag_name")
            ).all()
        )

        tags = []
        for e in entries:
            aliases = [a.strip() for a in e.aliases.split(',')] if e.aliases else []
            tags.append({
                "name": e.standard_tag,
                "category": e.category or "其他",
                "aliases": aliases,
                "file_count": counts.get(e.standard_tag, 0),
                "theme_id": e.theme_id,
            })
        return {"tags": tags}
    finally:
        db.close()

@app.get("/dictionary/categories")
async def get_dictionary_categories():
    """Get all categories.

    取"字典里已使用的分类" ∪ "分类表里已创建的分类"——否则**新建的空分类不会出现**，
    用户就无法把标签移入刚建的分类。
    """
    db = get_session(engine)
    try:
        used = {c[0] for c in db.query(DictionaryEntry.category).distinct().all() if c[0]}
        declared = {c[0] for c in db.query(TagCategory.name).all() if c[0]}
        return {"categories": sorted(used | declared)}
    finally:
        db.close()

@app.get("/dictionary/stats")
async def get_dictionary_stats():
    """Get dictionary statistics."""
    db = get_session(engine)
    try:
        total_tags = db.query(DictionaryEntry).count()
        categories = db.query(DictionaryEntry.category).distinct().all()
        category_count = len([c for c in categories if c[0]])  # Exclude None/empty categories
        return {
            "total_tags": total_tags,
            "category_count": category_count
        }
    finally:
        db.close()

@app.get("/dictionary/templates")
async def get_dictionary_templates():
    """Get available dictionary templates from resources/templates directory."""
    templates_dir = Path("resources/templates")
    templates = []
    
    if templates_dir.exists():
        for file in templates_dir.glob("*.md"):
            # Extract template name from filename (e.g., "男女标签库.md" -> "男女标签库")
            name = file.stem
            templates.append({
                "name": name,
                "file": str(file)
            })
    
    return {"templates": templates}

@app.post("/dictionary/categories")
async def create_dictionary_category(request: CategoryCreateRequest):
    """Create a new dictionary category."""
    db = get_session(engine)
    try:
        # 检查分类是否已存在
        existing = db.query(TagCategory).filter(TagCategory.name == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"分类 '{request.name}' 已存在")
        
        # 创建新分类
        category = TagCategory(name=request.name)
        db.add(category)
        db.commit()
        
        # 自动重建 VLM prompt
        _rebuild_vlm_prompt(db)
        
        return {"status": "success", "category": request.name}
    finally:
        db.close()

@app.delete("/dictionary/categories/{category_name}")
async def delete_dictionary_category(category_name: str):
    """Delete a dictionary category."""
    db = get_session(engine)
    try:
        # 检查分类是否存在
        category = db.query(TagCategory).filter(TagCategory.name == category_name).first()
        if not category:
            raise HTTPException(status_code=404, detail=f"分类 '{category_name}' 不存在")
        
        # 检查分类下是否有标签
        tags_in_category = db.query(DictionaryEntry).filter(DictionaryEntry.category == category_name).count()
        if tags_in_category > 0:
            raise HTTPException(status_code=400, detail=f"分类 '{category_name}' 下还有 {tags_in_category} 个标签，无法删除")
        
        # 删除分类
        db.delete(category)
        db.commit()
        
        # 自动重建 VLM prompt
        _rebuild_vlm_prompt(db)
        
        return {"status": "success"}
    finally:
        db.close()

@app.post("/dictionary")
async def create_dictionary_entry(request: DictionaryCreateRequest):
    """Create a new dictionary entry."""
    db = get_session(engine)
    try:
        existing = db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == request.name).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"标签 '{request.name}' 已存在")
        
        entry = DictionaryEntry(
            standard_tag=request.name,
            category=request.category,
            aliases=",".join(request.aliases),
            file_count=0,
            theme_id=request.theme_id,
        )
        db.add(entry)
        db.commit()
        return {"status": "success", "tag": request.name}
    finally:
        db.close()

@app.patch("/dictionary/{tag_name}")
async def update_dictionary_entry(tag_name: str, request: DictionaryUpdateRequest):
    """Update a dictionary entry's aliases or category."""
    db = get_session(engine)
    try:
        entry = db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == tag_name).first()
        if not entry:
            raise HTTPException(status_code=404, detail=f"标签 '{tag_name}' 不存在")
        
        # 更新别名（如果提供了）
        if request.aliases is not None:
            entry.aliases = ",".join(request.aliases)
        
        # 更新分类（如果提供了）
        if request.category is not None:
            entry.category = request.category
        
        # 更新主题关联（如果提供了）
        if request.theme_id is not None:
            entry.theme_id = request.theme_id
        
        db.commit()
        
        # 分类结构可能已改变，重建 VLM prompt
        _rebuild_vlm_prompt(db)
        
        return {"status": "success"}
    finally:
        db.close()

@app.delete("/dictionary/{tag}")
async def delete_dictionary_entry(tag: str):
    """Delete a dictionary entry."""
    db = get_session(engine)
    try:
        db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == tag).delete()
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.post("/dictionary/import")
def import_dictionary(request: DictionaryImportRequest):
    """Import dictionary from Markdown content."""
    db = get_session(engine)
    try:
        # Parse markdown content
        content = request.content
        
        # Split by sections (### headers)
        sections = re.split(r'\n###\s+', content)
        
        imported = 0
        for section in sections[1:]:  # Skip first part (before first ###)
            # Extract category from section header like "1. [场景] -- Scene"
            category_match = re.match(r'\d+\.\s*\[([^\]]+)\]', section)
            if not category_match:
                continue
            category = category_match.group(1)
            
            # Find all tag rows in this section: | **标签** | 别名 | 说明 |
            pattern = r'\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]*?)\s*\|\s*([^|]*)\s*\|'
            matches = re.findall(pattern, section)
            
            for match in matches:
                standard_tag = match[0].strip()
                aliases_str = match[1].strip()
                aliases = [a.strip() for a in re.split(r'[,，]+', aliases_str) if a.strip()]
                
                existing = db.query(DictionaryEntry).filter(DictionaryEntry.standard_tag == standard_tag).first()
                if existing:
                    existing.aliases = ",".join(aliases)
                    existing.category = category
                else:
                    entry = DictionaryEntry(
                        standard_tag=standard_tag,
                        category=category,
                        aliases=",".join(aliases),
                        file_count=0
                    )
                    db.add(entry)
                imported += 1
        
        db.commit()
        return {"status": "success", "imported": imported}
    finally:
        db.close()


@app.get("/dictionary/export")
async def export_dictionary():
    """导出标准标签库为 Markdown。

    输出格式与 ``/dictionary/import`` 的解析规则**一一对应**，因此可"导出→编辑→导入"往返：
    - 每个分类一段 ``### <序号>. [<分类>] -- <分类>``
    - 标签行 ``| **<标准标签>** | 别名1, 别名2 | |``
    """
    db = get_session(engine)
    try:
        entries = db.query(DictionaryEntry).order_by(
            DictionaryEntry.category, DictionaryEntry.standard_tag
        ).all()

        by_category = {}  # 保持插入顺序（Python 3.7+ dict 有序）
        for e in entries:
            by_category.setdefault(e.category or "其他", []).append(e)

        lines = ["# 标准标签库", ""]
        for idx, (category, items) in enumerate(by_category.items(), 1):
            lines.append(f"### {idx}. [{category}] -- {category}")
            lines.append("| 标签 | 别名 | 说明 |")
            lines.append("|---|---|---|")
            for e in items:
                aliases = ", ".join(
                    a.strip() for a in (e.aliases or "").split(",") if a.strip()
                )
                lines.append(f"| **{e.standard_tag}** | {aliases} | |")
            lines.append("")

        return {"status": "success", "content": "\n".join(lines), "count": len(entries)}
    finally:
        db.close()

# ── Audit Queue API ────────────────────────────────────────────────

@app.get("/audit-queue")
async def get_audit_queue(status: str = "pending"):
    """Get audit queue items."""
    db = get_session(engine)
    try:
        query = db.query(AuditQueueItem)
        if status:
            query = query.filter(AuditQueueItem.status == status)
        items = query.order_by(AuditQueueItem.count.desc()).all()
        
        return {
            "items": [{
                "id": item.id,
                "raw_label": item.raw_label,
                "count": item.count,
                "ai_recommendation": item.ai_recommendation or "无推荐",
                "status": item.status
            } for item in items]
        }
    finally:
        db.close()

@app.post("/audit-queue/recalculate-recommendations")
async def recalculate_recommendations():
    """重新计算所有未审核标签的推荐"""
    from src.core.dictionary import DictionaryManager
    
    db = get_session(engine)
    try:
        dict_manager = DictionaryManager(db_session=db, md_path="data/dictionary.md")
        
        # 获取所有未审核标签
        pending_items = db.query(AuditQueueItem).filter(
            AuditQueueItem.status == 'pending'
        ).all()
        
        updated_count = 0
        for item in pending_items:
            # 重新计算推荐
            new_recommendation = dict_manager._find_ai_recommendation(item.raw_label)
            
            if new_recommendation != item.ai_recommendation:
                item.ai_recommendation = new_recommendation
                updated_count += 1
        
        db.commit()
        return {
            "status": "success",
            "message": f"已更新 {updated_count} 个未审核标签的推荐",
            "updated_count": updated_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新计算推荐失败: {str(e)}")
    finally:
        db.close()

@app.post("/audit-queue/{item_id}/confirm")
async def confirm_audit_item(item_id: int, target_tag: str = None):
    """Confirm an audit item - add raw_label as alias to recommended or specified tag."""
    db = get_session(engine)
    try:
        item = db.query(AuditQueueItem).filter(AuditQueueItem.id == item_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="审核项不存在")
        
        # Determine target tag: use provided target_tag or fall back to ai_recommendation
        target = target_tag or item.ai_recommendation
        
        if not target:
            raise HTTPException(
                status_code=400, 
                detail="未指定目标标签，且系统无法自动推荐。请手动选择目标标准标签。"
            )
        
        # Find the target tag entry
        tag_entry = db.query(DictionaryEntry).filter(
            DictionaryEntry.standard_tag == target
        ).first()
        
        if not tag_entry:
            raise HTTPException(
                status_code=404, 
                detail=f"目标标签 '{target}' 不存在"
            )
        
        # Add raw_label as alias if not already present
        existing_aliases = [a.strip() for a in tag_entry.aliases.split(',')] if tag_entry.aliases else []
        if item.raw_label not in existing_aliases:
            existing_aliases.append(item.raw_label)
            tag_entry.aliases = ",".join(existing_aliases)
        
        item.status = 'confirmed'
        db.commit()
        
        return {
            "status": "success", 
            "message": f"已将 '{item.raw_label}' 添加为 '{target}' 的别名",
            "target_tag": target
        }
    finally:
        db.close()

@app.post("/audit-queue/{item_id}/reject")
async def reject_audit_item(item_id: int):
    """Reject an audit item."""
    db = get_session(engine)
    try:
        item = db.query(AuditQueueItem).filter(AuditQueueItem.id == item_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="审核项不存在")
        
        item.status = 'rejected'
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.post("/audit-queue/batch")
async def batch_audit_action(request: AuditBulkRequest):
    """批量处理审核项：确认 / 拒绝 / 删除。

    返回**逐项结果**，供前端批量进度弹窗展示：
    ``results[i] = {item_id, raw_label, status, message}``，
    status ∈ ``confirmed / rejected / deleted / skipped / error``。
    "确认"若既无手动目标标签、又无 AI 推荐 ⇒ 该项 **skipped**（需人工映射）。
    """
    db = get_session(engine)
    try:
        # item_ids=None ⇒ 处理全部待审核；显式空列表 ⇒ **不处理任何条目**
        # （不能把空列表退化成"全部"，否则一次误传就会改动所有待审核项）
        if request.item_ids is None:
            items = db.query(AuditQueueItem).filter(AuditQueueItem.status == 'pending').all()
        elif request.item_ids:
            items = db.query(AuditQueueItem).filter(AuditQueueItem.id.in_(request.item_ids)).all()
        else:
            items = []

        results = []
        processed = 0
        for item in items:
            entry = {"item_id": item.id, "raw_label": item.raw_label, "status": "error", "message": ""}
            try:
                if request.action == 'confirm':
                    target_tag = request.target_tag or item.ai_recommendation
                    tag_entry = None
                    if target_tag:
                        tag_entry = db.query(DictionaryEntry).filter(
                            DictionaryEntry.standard_tag == target_tag
                        ).first()
                    if not tag_entry:
                        entry["status"] = "skipped"
                        entry["message"] = "无 AI 推荐或无匹配的目标标签，需手动映射"
                    else:
                        existing = [a.strip() for a in tag_entry.aliases.split(',')] if tag_entry.aliases else []
                        if item.raw_label not in existing:
                            existing.append(item.raw_label)
                            tag_entry.aliases = ",".join(existing)
                        item.status = 'confirmed'
                        entry["status"] = "confirmed"
                        entry["message"] = f"已添加为 '{target_tag}' 的别名"
                        processed += 1
                elif request.action == 'reject':
                    item.status = 'rejected'
                    entry["status"] = "rejected"
                    entry["message"] = "已拒绝"
                    processed += 1
                elif request.action == 'delete':
                    db.delete(item)
                    entry["status"] = "deleted"
                    entry["message"] = "已删除"
                    processed += 1
                else:
                    entry["message"] = f"未知操作: {request.action}"
            except Exception as exc:
                entry["status"] = "error"
                entry["message"] = str(exc)
            results.append(entry)

        db.commit()
        return {"status": "success", "processed": processed, "results": results}
    finally:
        db.close()


@app.delete("/audit-queue/{item_id}")
async def delete_audit_item(item_id: int):
    """删除单条审核项（不删除字典、不改动作品标签）。"""
    db = get_session(engine)
    try:
        item = db.query(AuditQueueItem).filter(AuditQueueItem.id == item_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="审核项不存在")
        db.delete(item)
        db.commit()
        return {"status": "success", "deleted": 1}
    finally:
        db.close()

# ── Config API ─────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """Get all configuration."""
    return config_manager.get_all()

@app.put("/config/{key}")
async def update_config(key: str, request: ConfigUpdateRequest):
    """Update a configuration value."""
    try:
        config_manager.set(key, request.value)
        return {"status": "success", "key": key, "value": request.value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── 供应商管理 API（多供应商共存 + 切换） ─────────────────────────

@app.get("/vlm/vendors")
async def get_vlm_vendors():
    """获取全部 VLM 供应商（含当前激活）。"""
    data = config_manager.get_vlm_vendors()
    return {"status": "success", **data}


@app.post("/vlm/vendors")
async def add_vlm_vendor(request: dict):
    """新增或更新供应商。请求体: {name, type, vendor, base_url, model}"""
    name = (request.get("name") or "").strip()
    ok, msg = config_manager.save_vlm_vendor(name, request)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@app.put("/vlm/vendors/{name}")
async def update_vlm_vendor(name: str, request: dict):
    """更新供应商。请求体: {type, vendor, base_url, model}"""
    ok, msg = config_manager.save_vlm_vendor(name, request)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@app.delete("/vlm/vendors/{name}")
async def delete_vlm_vendor(name: str):
    """删除供应商。"""
    ok, msg = config_manager.delete_vlm_vendor(name)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}


@app.post("/vlm/vendors/{name}/activate")
async def activate_vlm_vendor(name: str):
    """切换当前激活的供应商。"""
    ok, msg = config_manager.set_active_vlm_vendor(name)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

@app.post("/config/test-connection")
def test_connection(request: dict = None):
    """Test VLM connection based on provider mode.

    provider:
      - url（默认）: URL API，自动识别 Ollama / OpenAI 兼容云端
      - folder: 测本地模型路径是否存在
    """
    request = request or {}
    provider = (request.get("provider") or config_manager.get_vlm_config().get("provider", "url")).lower()
    base_url = request.get("base_url") or config_manager.get_vlm_config().get("base_url", "http://localhost:11434")
    model = request.get("model") or config_manager.get_vlm_config().get("model", "")

    # ── folder：测本地路径 ──
    if provider in ("mlx", "folder"):
        from pathlib import Path
        p = Path(model or "")
        if not model:
            return {"status": "error", "error": "未指定本地模型路径"}
        if p.exists():
            return {"status": "success", "model_available": True, "path_exists": True, "path": str(p)}
        return {"status": "error", "model_available": False, "path_exists": False,
                "error": f"路径不存在: {model}（请将模型文件放入该路径）"}

    # ── url（Ollama / 云端通用）：自动识别协议 ──
    # 1) 试 Ollama 原生 /api/tags（能列出模型 = Ollama）
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            model_available = any(model in m for m in models) if model else False
            return {
                "status": "success",
                "mode": "ollama",
                "ollama_running": True,
                "model_available": model_available,
                "available_models": models,
            }
    except Exception:
        pass

    # 2) 试 OpenAI 兼容 /models（云端 API）
    try:
        models_resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=8)
        if models_resp.status_code == 200:
            try:
                data = models_resp.json()
                model_ids = [m.get("id", "") for m in data.get("data", [])]
                available = (not model) or any(model in mid for mid in model_ids)
                return {"status": "success", "mode": "openai", "api_available": True,
                        "model_available": available, "available_models": model_ids}
            except Exception:
                pass
        # 能连通但未暴露模型列表
        return {"status": "success", "mode": "openai", "api_available": True,
                "model_available": True, "note": "API 可达（未暴露模型列表，以实际调用为准）"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "api_available": False, "model_available": False,
                "error": f"无法连接 API: {base_url}"}
    except Exception as e:
        return {"status": "error", "api_available": False, "model_available": False, "error": str(e)}

@app.get("/config/vlm-prompt/last-template")
async def get_last_template():
    """读取"上次选择的提示词模板"（仅用于记住选择，失败可忽略）。"""
    return {"value": config_manager.get("vlm_prompt_last_template", "") or ""}


@app.put("/config/vlm-prompt/last-template")
async def set_last_template(request: ConfigUpdateRequest):
    """记录"上次选择的提示词模板"。"""
    config_manager.set("vlm_prompt_last_template", request.value or "")
    return {"status": "success"}


@app.get("/config/vlm-prompt/templates")
async def get_vlm_prompt_templates():
    """Get available VLM prompt templates from resources/templates directory."""
    templates_dir = Path("resources/templates")
    templates = []
    
    if templates_dir.exists():
        for file in templates_dir.glob("*.md"):
            name = file.stem
            content = file.read_text(encoding="utf-8")
            # Extract prompt content (skip the first line "# 标准标签库")
            lines = content.split("\n")
            prompt_lines = []
            for line in lines:
                if line.startswith("# "):
                    continue
                prompt_lines.append(line)
            prompt = "\n".join(prompt_lines).strip()
            
            templates.append({
                "name": name,
                "prompt": prompt
            })
    
    return {"templates": templates}

@app.get("/config/vlm-prompt")
async def get_vlm_prompt():
    """Get VLM prompt."""
    prompt = config_manager.get_vlm_prompt()
    return {"prompt": prompt}

@app.post("/config/vlm-prompt")
async def set_vlm_prompt(request: dict):
    """Set VLM prompt."""
    prompt = request.get("prompt", "")
    config_manager.set_vlm_prompt(prompt)
    return {"status": "success"}

# ── Environment Check API ──────────────────────────────────────────

@app.get("/env-check")
def env_check():
    """Check system environment status."""
    checks = {}
    
    # 1. VLM 活跃供应商检测（支持 folder / url 类型）
    vlm_config = config_manager.get_vlm_config()
    vlm_type = vlm_config.get("provider", "url")
    vlm_vendor = vlm_config.get("vendor_name", vlm_config.get("vendor", "未知"))
    vlm_model = vlm_config.get("model", "")

    if vlm_type in ("folder", "mlx"):
        model_path = Path(vlm_model).expanduser()
        if model_path.exists() and model_path.is_dir():
            checks["vlm"] = {
                "status": "ok",
                "message": f"{vlm_vendor} 已就绪",
                "type": vlm_type,
                "vendor_name": vlm_vendor,
            }
        else:
            checks["vlm"] = {
                "status": "error",
                "message": f"{vlm_vendor} 模型路径不存在",
                "type": vlm_type,
                "vendor_name": vlm_vendor,
            }
    else:
        base_url = vlm_config.get("base_url", "http://localhost:11434")
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_available = any(vlm_model in m for m in models) if vlm_model else True
            checks["vlm"] = {
                "status": "ok" if model_available else "warning",
                "message": f"{vlm_vendor} 运行中" if model_available else f"{vlm_vendor} 模型 {vlm_model} 未安装",
                "type": vlm_type,
                "vendor_name": vlm_vendor,
            }
        except Exception:
            checks["vlm"] = {
                "status": "error",
                "message": f"{vlm_vendor} 无法连接",
                "type": vlm_type,
                "vendor_name": vlm_vendor,
            }
    
    # 2. FFmpeg - 优先检查内置版本
    from src.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()
    bundled_ffmpeg = checker._get_bundled_ffmpeg_path()
    if bundled_ffmpeg and bundled_ffmpeg.exists():
        checks["ffmpeg"] = {"status": "ok", "message": "已就绪"}
    elif shutil.which("ffmpeg"):
        checks["ffmpeg"] = {"status": "ok", "message": "已就绪"}
    else:
        checks["ffmpeg"] = {
            "status": "error",
            "message": "未安装（应用不再内置，请安装 ffmpeg，如 brew install ffmpeg）",
        }

    # 3. Face recognition model (AuraFace)
    try:
        aura = _check_auraface()
        if aura["status"] == "ok":
            checks["face_model"] = {"status": "ok", "message": "AuraFace 已就位"}
        else:
            checks["face_model"] = {
                "status": "warning",
                "message": f"AuraFace 模型缺失（{len(aura['missing'])} 个文件），可在设置页环境初始化下载",
            }
    except Exception:
        checks["face_model"] = {"status": "error", "message": "AuraFace 检查失败"}

    # 4. Output directories
    output_dir = Path(config_manager.get("output_dir", "./output"))
    output_ok = output_dir.exists() and os.access(output_dir, os.W_OK)
    checks["output_dirs"] = {
        "status": "ok" if output_ok else "error",
        "message": "可写" if output_ok else "目录不存在或不可写",
    }
    
    return checks


# ── 环境初始化 API ─────────────────────────────────────────────────
# 新环境一键初始化：检测并补齐 FFmpeg（bundled）、人脸模型（AuraFace，Apache 2.0
# 可商用，从 hf-mirror 镜像下载）、输出目录等。

AURAFACE_FILES = [
    "glintr100.onnx",       # 识别模型 (260M)
    "scrfd_10g_bnkps.onnx", # 检测模型 (17M)
    "1k3d68.onnx",          # 关键点 (144M)
    "2d106det.onnx",        # 关键点 (5M)
    "genderage.onnx",       # 性别年龄 (1.3M)
]

def _get_auraface_dir() -> Path:
    """AuraFace 模型目录。

    - 打包(frozen)环境: ``~/.insightface/models/auraface`` —— 与 face_recognition
      的加载路径(get_insightface_model_root)一致, 也避免写入只读的 app 内 Resources
    - 开发环境: config ``face_recognition.model_path``(默认 ``models/auraface``)
    """
    if getattr(sys, "frozen", False):
        return Path.home() / ".insightface" / "models" / "auraface"
    cfg_path = config_manager.get("face_recognition.model_path", "models/auraface")
    return Path(cfg_path).expanduser().resolve()


def _check_auraface() -> Dict[str, Any]:
    """检查 AuraFace 模型是否就位。"""
    model_dir = _get_auraface_dir()
    missing = [f for f in AURAFACE_FILES if not (model_dir / f).exists()]
    if not missing:
        return {"status": "ok", "model_dir": str(model_dir), "missing": []}
    return {"status": "missing", "model_dir": str(model_dir), "missing": missing}


@app.get("/env/status")
async def env_status():
    """环境状态检查（初始化前置）：FFmpeg / 人脸模型 / 输出目录。"""
    from src.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()

    # FFmpeg（bundled 优先）
    bundled_ffmpeg = checker._get_bundled_ffmpeg_path()
    ffmpeg_ok = bool(bundled_ffmpeg and bundled_ffmpeg.exists())
    if not ffmpeg_ok:
        ffmpeg_ok = bool(shutil.which("ffmpeg"))

    # 人脸模型
    aura = _check_auraface()

    # 输出目录
    output_dir = Path(config_manager.get("output_dir", "./output"))
    output_ok = output_dir.exists() and os.access(output_dir, os.W_OK)

    return {
        "ffmpeg": {"ok": ffmpeg_ok},
        "face_model": aura,
        "output_dir": {"ok": output_ok, "path": str(output_dir)},
    }


@app.post("/env/init")
async def env_init(request: dict = None):
    """环境初始化：下载缺失的人脸模型、创建输出目录、确认 FFmpeg。

    body 可选: {"force": true} 强制重新下载模型
    """
    request = request or {}
    force = bool(request.get("force", False))
    steps = []

    # 1. FFmpeg（bundled，只确认）
    from src.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()
    bundled_ffmpeg = checker._get_bundled_ffmpeg_path()
    if bundled_ffmpeg and bundled_ffmpeg.exists():
        steps.append({"step": "ffmpeg", "status": "ok", "message": f"已就绪 ({bundled_ffmpeg})"})
    else:
        steps.append({
            "step": "ffmpeg",
            "status": "missing",
            "message": "未检测到 FFmpeg（应用不再内置），请安装后重试（brew install ffmpeg）",
        })

    # 2. 输出目录
    output_dir = Path(config_manager.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    steps.append({"step": "output_dir", "status": "ok", "message": str(output_dir)})

    # 3. 人脸模型（AuraFace，Apache 2.0 可商用）
    aura = _check_auraface()
    if aura["status"] == "ok" and not force:
        steps.append({"step": "face_model", "status": "ok", "message": f"已就位 ({aura['model_dir']})"})
    else:
        try:
            result = _download_auraface(force=force)
            steps.append(result)
        except Exception as e:
            steps.append({"step": "face_model", "status": "error", "message": f"下载失败: {e}"})

    all_ok = all(s["status"] == "ok" for s in steps)
    return {"status": "success" if all_ok else "partial", "steps": steps}


def _download_auraface(force: bool = False) -> Dict[str, Any]:
    """从 hf-mirror 下载 AuraFace 模型（Apache 2.0）。

    repo: fal/AuraFace-v1 → 镜像 hf-mirror.com
    """
    model_dir = _get_auraface_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    # 使用 huggingface_hub（支持 HF_ENDPOINT 环境变量指向镜像）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="fal/AuraFace-v1",
        local_dir=str(model_dir),
        allow_patterns=["*.onnx", "*.md"],
    )
    # 验证关键文件
    missing = [f for f in AURAFACE_FILES if not (model_dir / f).exists()]
    if missing:
        return {
            "step": "face_model", "status": "error",
            "message": f"下载后仍缺失: {missing}",
        }
    return {
        "step": "face_model", "status": "ok",
        "message": f"已下载到 {model_dir} ({len(AURAFACE_FILES)} 个文件)",
    }


# ── 模型检测 API（版权合规：不提供任何下载接口）────────────────────
# 软件不内置、不下载、不指引下载任何模型。
# 用户自行下载/拷贝模型到指定路径，软件只检测已就位的模型并提供配置入口。
# 注：AuraFace 为 Apache 2.0（可商用），已通过 /env/init 提供一键下载。

# ── 帧缓存清理 API（设置页「帧缓存 → 统计 / 立即清理」） ────────────

@app.get("/frames/cleanup/stats")
async def frames_cleanup_stats(days: int = 30):
    """统计超过保留期（`days` 天）的帧缓存目录数与占用。"""
    from src.core.cache_manager import get_frame_cache_cleanup_stats
    return {"status": "success", **get_frame_cache_cleanup_stats(max_age_days=days)}


@app.post("/frames/cleanup")
async def frames_cleanup(request: dict = None):
    """清理超过保留期（`days` 天）的帧缓存目录。

    请求体: {"days": 30}。只删磁盘上的抽帧中间产物，不影响作品、标签与封面。
    """
    from src.core.cache_manager import clean_frame_cache
    request = request or {}
    days = int(request.get("days", 30))
    result = clean_frame_cache(max_age_days=days)
    return {
        "status": "success",
        "cleaned_count": len(result["removed"]),
        "cleaned_size_mb": result["freed_mb"],
    }

LOCAL_MODELS_DIR = Path("models")


@app.get("/model/installed")
def model_installed():
    """返回已就位的模型列表（Ollama 已装 + 本地 models 目录）。

    版权合规：软件只检测用户已自行下载/拷贝的模型，不提供下载。

    Ollama 模型拆成两组：
    - vision_models: 支持视觉分析的模型（VLM 用途，下拉默认显示这些）
    - models: 全部模型（含纯文本/嵌入模型，供参考）
    """
    result = {
        "ollama": {"available": False, "base_url": "", "models": [], "vision_models": [], "error": None},
        "local_dir": str(LOCAL_MODELS_DIR.resolve()),
        "local": [],
        "current_vlm_model": None,
    }

    # 当前配置的 VLM 模型
    result["current_vlm_model"] = config_manager.get_vlm_config().get("model")

    # 1. Ollama 已安装模型
    vlm_config = config_manager.get_vlm_config()
    base_url = vlm_config.get("base_url", "http://localhost:11434")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]

        # 过滤视觉模型（VLM 用途）
        _VISION_KEYWORDS = [
            "minicpm", "qwen2.5-vl", "qwen2.5vl", "qwen-vl", "llava",
            "gemma3", "phi-3-vision", "bakllava", "llama3.2-vision",
            "moondream", "internvl", "yi-vl", "cogvlm", "deepseek-vl",
        ]
        vision_models = [
            m for m in models
            if any(k in m.lower() for k in _VISION_KEYWORDS)
            and "embed" not in m.lower()
            and "nomic" not in m.lower()
        ]

        result["ollama"] = {
            "available": True,
            "base_url": base_url,
            "models": models,
            "vision_models": vision_models,
            "error": None,
        }
    except Exception as e:
        result["ollama"] = {
            "available": False,
            "base_url": base_url,
            "models": [],
            "vision_models": [],
            "error": str(e),
        }

    # 2. 本地 models 目录内容（用户自行拷贝的模型文件）
    try:
        if LOCAL_MODELS_DIR.exists():
            entries = []
            for p in sorted(LOCAL_MODELS_DIR.iterdir()):
                if p.is_dir():
                    # 子目录：列出里面的模型文件
                    files = [f.name for f in sorted(p.iterdir()) if f.suffix.lower() in (".onnx", ".gguf", ".bin", ".safetensors", ".pt", ".pth")]
                    entries.append({"name": p.name, "type": "dir", "files": files})
                else:
                    entries.append({"name": p.name, "type": "file"})
            result["local"] = entries
        else:
            LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        result["local_error"] = str(e)

    return result


# ── Legacy Dictionary API (kept for backward compat) ───────────────

@app.get("/dictionary/entries")
async def get_dictionary_entries():
    db = get_session(engine)
    try:
        entries = db.query(DictionaryEntry).all()
        return [{"tag": e.standard_tag, "aliases": e.aliases} for e in entries]
    finally:
        db.close()

@app.post("/dictionary/update")
async def update_dictionary_entry_legacy(request: dict):
    db = get_session(engine)
    try:
        tag = request.get("tag")
        aliases = request.get("aliases", "")
        if not tag:
            raise HTTPException(status_code=400, detail="Tag is required")
        
        entry = DictionaryEntry(standard_tag=tag, aliases=aliases)
        db.merge(entry)
        db.commit()
        
        dict_mgr = DictionaryManager(MD_MAPPING_PATH, db)
        dict_mgr.save_to_md() 
        return {"status": "success"}
    finally:
        db.close()

@app.get("/dictionary/tags")
async def get_all_tags():
    db = get_session(engine)
    try:
        entries = db.query(DictionaryEntry).all()
        return [e.standard_tag for e in entries]
    finally:
        db.close()

# ── Actor Management API ──────────────────────────────────────────

def _actor_source_stats(db, actor_id: int):
    """人物的"作品"统计: 按**文件去重**。

    口径 = 详情页同一套关联(任务关联表 ∪ 人脸表 ∪ tasks.actor_id),
    只保留真实存在的任务(dangling 的 task_id 会被 tasks 过滤掉),
    再按文件去重: entity_id 优先, 缺失时回退 file_md5。

    Returns:
        (source_count, source_files)  —— 文件数 与 文件名列表
    """
    rows = db.execute(text("""
        SELECT t.entity_id, t.file_md5
        FROM tasks t
        WHERE t.id IN (
            SELECT task_id FROM task_actors WHERE actor_id = :aid
            UNION SELECT task_id FROM actor_faces
                  WHERE actor_id = :aid AND task_id IS NOT NULL
            UNION SELECT id FROM tasks WHERE actor_id = :aid
        )
    """), {"aid": actor_id}).fetchall()

    seen_keys = set()
    entity_ids = []
    for _eid, _md5 in rows:
        key = _eid if _eid is not None else _md5
        if key is None or key in seen_keys:
            continue
        seen_keys.add(key)
        if _eid is not None:
            entity_ids.append(_eid)

    source_files = []
    if entity_ids:
        loc_rows = (
            db.query(FileLocation.entity_id, FileLocation.file_path)
            .filter(
                FileLocation.entity_id.in_(entity_ids),
                FileLocation.file_path.isnot(None),
            )
            .distinct()
            .all()
        )
        seen_paths = set()
        for _eid2, _fp in loc_rows:
            if _fp and _fp not in seen_paths:
                seen_paths.add(_fp)
                source_files.append(Path(_fp).name)

    return len(seen_keys), source_files


def _normalize_name(s: str) -> str:
    """姓名归一化：中文与英文共用同一套匹配。

    - 去重音：José → jose
    - 大小写不敏感：Cydney → cydney
    - 连字符/撇号/点/下划线等统一当空格：Jean-Luc → jean luc，O'Neil → o neil
    - 合并多余空白
    中文不受影响（无空格、无大小写、无重音），仍是原串。
    """
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _all_tokens_prefix(query_tokens, cand_tokens) -> bool:
    """查询的每个词是否命中目标某个词的**词首**（顺序无关）。"""
    return all(any(ct.startswith(qt) for ct in cand_tokens) for qt in query_tokens)


def _all_tokens_within(query_tokens, cand_tokens) -> bool:
    """查询的每个词是否**出现在**目标某个词中（顺序无关）。"""
    return all(any(qt in ct for ct in cand_tokens) for qt in query_tokens)


def _actor_search_score(query: str, actor: Actor) -> float:
    """人物模糊匹配打分（0 = 不匹配）。越大越相关。

    先归一化（去重音 / 大小写不敏感 / 标点当空格），再按由严到宽取最高分
    （名称 / 主名 / 别名都参与）：
    - 完全相同 100 / 前缀 90
    - **词元前缀 88**——各词按**任意顺序**命中目标的词首（"jane cyd"→"Cydney Jane"）
    - **子串 80**（"五十"→"五十岚"，"yde"→"Cydney"）
    - **词元包含 75**——各词以任意顺序出现在目标词中
    - **顺序子序列 60**（"五岚"→"五十岚"）
    - **近似拼写** ≤50（difflib 相似度 ≥0.6，容忍错别字，如 "cidney"→"Cydney"）

    说明：英文的拼写容错靠 difflib；**不做拼音**（中文同音字不在此处理）。
    """
    q = _normalize_name(query)
    if not q:
        return 0.0
    q_tokens = q.split(" ")

    candidates = []
    for v in (actor.name, actor.primary_name):
        if v:
            candidates.append(str(v))
    if actor.aliases:
        try:
            for a in json.loads(actor.aliases):
                if a:
                    candidates.append(str(a))
        except Exception:
            pass

    best = 0.0
    for raw in candidates:
        c = _normalize_name(raw)
        if not c:
            continue
        c_tokens = c.split(" ")
        if c == q:
            best = max(best, 100.0)
        elif c.startswith(q):
            best = max(best, 90.0)
        elif _all_tokens_prefix(q_tokens, c_tokens):
            best = max(best, 88.0)
        elif q in c:
            best = max(best, 80.0)
        elif _all_tokens_within(q_tokens, c_tokens):
            best = max(best, 75.0)
        elif len(q) >= 2 and _is_subsequence(q, c):
            best = max(best, 60.0)
        else:
            # 近似拼写：整串比较；查询只有单个词时再逐词比较
            # （"cidney"→"Cydney Jane"：整串相似度会被多词稀释，逐词才稳）
            ratio = difflib.SequenceMatcher(None, q, c).ratio()
            if len(q_tokens) == 1:
                for ct in c_tokens:
                    ratio = max(ratio, difflib.SequenceMatcher(None, q, ct).ratio())
            if ratio >= 0.6:
                best = max(best, ratio * 50.0)
    return best


def _is_subsequence(needle: str, haystack: str) -> bool:
    """needle 的字符是否按顺序出现在 haystack 中（不要求连续）。"""
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _resolve_actor_cover(db, actor: Actor, faces_dir: Path) -> Optional[str]:
    """解析人物封面的**绝对路径**；无可用封面时返回 None。

    单一事实来源——列表/搜索/详情共用同一口径，避免"某处按存在性回退、
    某处不回退"导致计数与展示不一致。规则：
    1. cover_face 为相对路径时拼到 faces_dir 下；
    2. 指向的文件不存在时，回退到该人物 quality_score 最高且真实存在的一张脸
       （历史 bug：rebuild/合并后 cover_face 指向已删除的 face_0.jpg）；
    3. 都没有 → None（前端据此判定"无封面，不展示"）。
    """
    cover_face = actor.cover_face
    if cover_face and not Path(cover_face).is_absolute():
        cover_face = str(faces_dir / cover_face)
    if cover_face and os.path.exists(cover_face):
        return cover_face
    if not cover_face:
        return None

    _fallback = (
        db.query(ActorFace.face_path)
        .filter(
            ActorFace.actor_id == actor.id,
            ActorFace.quality_score.isnot(None),
        )
        .order_by(ActorFace.quality_score.desc())
        .first()
    )
    if not (_fallback and _fallback[0]):
        return None
    _fb = _fallback[0]
    _fb_full = _fb if Path(_fb).is_absolute() else str(faces_dir / _fb)
    return _fb_full if os.path.exists(_fb_full) else None


@app.get("/actors")
async def list_actors(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    only_with_cover: bool = Query(False),
    ids: Optional[str] = Query(
        None,
        description="逗号分隔的人物 id：只返回这些人物（人脸管理页「只看已勾选」用；"
                    "勾选是跨页的，故按 id 取数而不是筛当前页）",
    ),
):
    """Get actors with pagination, source file count and file list.

    search: 可选，**模糊搜索**（名称/主名/别名）：大小写不敏感、标点与重音归一，
    支持子串、**任意词序**（"jane cyd"→"Cydney Jane"）、顺序子序列、近似拼写；
    结果按匹配度排序。作用于**全部人物**，不是只过滤当前页。

    only_with_cover: 只要"有封面（可展示）"的人物。人脸管理页只展示有封面的卡片，
        开启后 total 与分页都按**过滤后的集合**计算，保证 total == 实际可见条数。

    ids: 只返回这些人物（total 同样按过滤后计算，分页可用）。
    """
    faces_dir = _api_faces_dir()

    # ids 过滤：解析成集合；给了 ids 就走"内存过滤 + 再分页"分支（人物表是千级）
    id_filter: Optional[set] = None
    if ids:
        id_filter = set()
        for chunk in ids.split(","):
            chunk = chunk.strip()
            if chunk.isdigit():
                id_filter.add(int(chunk))

    db = get_session(engine)
    # 一次请求内缓存封面解析结果，避免过滤/组装阶段重复解析同一人物
    cover_cache: Dict[int, Optional[str]] = {}

    def _cover(actor: Actor) -> Optional[str]:
        if actor.id not in cover_cache:
            cover_cache[actor.id] = _resolve_actor_cover(db, actor, faces_dir)
        return cover_cache[actor.id]

    try:
        search_term = (search or "").strip()
        if search_term or only_with_cover or id_filter is not None:
            # 人物表较小（千级），在内存里做模糊匹配 / 封面过滤与排序，再分页
            if search_term:
                scored = []
                for a in db.query(Actor).all():
                    s = _actor_search_score(search_term, a)
                    if s > 0:
                        scored.append((s, a))
                scored.sort(key=lambda x: (
                    -x[0],
                    -(x[1].created_at.timestamp() if x[1].created_at else 0.0),
                ))
                candidates = [a for _, a in scored]
            else:
                candidates = (
                    db.query(Actor)
                    .order_by(Actor.created_at.desc())
                    .all()
                )
            if id_filter is not None:
                candidates = [a for a in candidates if a.id in id_filter]
            if only_with_cover:
                candidates = [a for a in candidates if _cover(a)]
            total = len(candidates)
            actors = candidates[(page - 1) * page_size: (page - 1) * page_size + page_size]
        else:
            total = db.query(Actor).count()
            actors = (
                db.query(Actor)
                .order_by(Actor.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

        result = []
        for a in actors:
            # 作品数(源文件数): 按**文件去重**(见 _actor_source_stats)。
            # 历史 bug: 原先只数 actor_faces 的 entity_id(为 0 才回退 md5)——
            # 而 74% 的 face 记录 entity_id 为空、且"人物↔任务"关系主要落在
            # task_actors/tasks.actor_id 上, 导致卡片系统性少算
            # (如 ACTOR_122 实际 7 部, 卡片显示 3)。
            source_count, source_files = _actor_source_stats(db, a.id)

            # 封面（绝对路径；不存在时回退到质量最高的一张脸）——统一走 _resolve_actor_cover
            cover_face = _cover(a)

            # Parse aliases from JSON string
            aliases_list = []
            if a.aliases:
                try:
                    aliases_list = json.loads(a.aliases) if a.aliases.startswith('[') else [x.strip() for x in a.aliases.split(',') if x.strip()]
                except:
                    aliases_list = [x.strip() for x in a.aliases.split(',') if x.strip()]

            result.append({
                "id": a.id,
                "code": a.code,
                "name": a.name,
                "primary_name": a.primary_name,
                "aliases": aliases_list,
                "cover_face": cover_face,
                "source_count": source_count,
                "source_files": source_files,
                "face_type": a.face_type,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "actors": result,
        }
    finally:
        db.close()


@app.get("/actors/ids")
def get_actor_ids():
    """全部人物 id 列表（供前端剔除已不存在的"幽灵勾选"）。

    刻意轻量：只回 id，用于前端把跨页持久的勾选集合与服务端做交集，
    清除已被删除（本页删除 / 后端识别自动合并 / 重建）的人物残留勾选。

    必须注册在 ``/actors/{actor_id}`` 之前，否则会被参数化路由吞掉
    （"ids" 被当成 actor_id）。
    """
    db = get_session(engine)
    try:
        return {"ids": [r[0] for r in db.query(Actor.id).all()]}
    finally:
        db.close()


@app.get("/actors/{actor_id}")
def get_actor(actor_id: int):
    """Get actor details including collaborators and face data."""
    faces_dir = _api_faces_dir()

    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")

        # Helper: convert relative path to absolute
        def _abs(p: Optional[str]) -> Optional[str]:
            if p and not Path(p).is_absolute():
                return str(faces_dir / p)
            return p

        # ── Collect actor's file_md5s ──
        actor_md5_rows = (
            db.query(ActorFace.file_md5)
            .filter(ActorFace.actor_id == actor_id, ActorFace.file_md5.isnot(None))
            .distinct()
            .all()
        )
        actor_md5s = set(r[0] for r in actor_md5_rows)

        # ── Batch-query tasks for these md5s ──
        task_by_md5: Dict[str, Task] = {}
        if actor_md5s:
            tasks = (
                db.query(Task)
                .join(FileEntity, Task.entity_id == FileEntity.id)
                .filter(FileEntity.file_md5.in_(list(actor_md5s)))
                .all()
            )
            task_by_md5 = {t.entity.file_md5: t for t in tasks if t.entity}

        # ── Collaborators ──
        collaborators = []
        if actor_md5s:
            # All actor_faces in the same files
            all_faces_in_files = (
                db.query(ActorFace)
                .filter(ActorFace.file_md5.in_(list(actor_md5s)))
                .all()
            )

            # Group by actor_id
            faces_by_actor: Dict[int, list] = {}
            for f in all_faces_in_files:
                faces_by_actor.setdefault(f.actor_id, []).append(f)

            # Batch-load collaborator Actor records
            collab_actor_ids = [aid for aid in faces_by_actor if aid != actor_id]
            collab_actors: Dict[int, Actor] = {}
            if collab_actor_ids:
                collab_actor_rows = db.query(Actor).filter(Actor.id.in_(collab_actor_ids)).all()
                collab_actors = {a.id: a for a in collab_actor_rows}

            for other_id, other_faces in faces_by_actor.items():
                if other_id == actor_id:
                    continue
                collab_actor = collab_actors.get(other_id)
                if not collab_actor:
                    continue

                other_md5s = set(f.file_md5 for f in other_faces if f.file_md5)
                collab_md5s = actor_md5s & other_md5s
                if not collab_md5s:
                    continue

                collab_file_list = []
                for md5 in sorted(collab_md5s):
                    task = task_by_md5.get(md5)
                    collab_file_list.append({
                        "file_md5": md5,
                        "file_path": _get_primary_path(task) if task else None,
                    })

                collaborators.append({
                    "id": collab_actor.id,
                    "code": collab_actor.code,
                    "name": collab_actor.name,
                    "cover_face": _abs(collab_actor.cover_face),
                    "collaboration_count": len(collab_md5s),
                    "collaboration_files": collab_file_list,
                })

        # ── Primary face (is_cover = 1) ──
        primary_face_row = (
            db.query(ActorFace)
            .filter(ActorFace.actor_id == actor_id, ActorFace.is_cover == 1)
            .first()
        )
        primary = None
        if primary_face_row:
            primary = {
                "face_path": _abs(primary_face_row.face_path),
                "yaw": primary_face_row.yaw,
                "pitch": primary_face_row.pitch,
                "roll": primary_face_row.roll,
                "quality_score": primary_face_row.quality_score,
                "eyes_open_score": primary_face_row.eyes_open_score,
            }

        # ── Similar faces: per file_md5, pick face with smallest total angle + normal brightness ──
        similar = []
        if actor_md5s:
            all_actor_faces = (
                db.query(ActorFace)
                .filter(
                    ActorFace.actor_id == actor_id,
                    ActorFace.file_md5.in_(list(actor_md5s)),
                )
                .all()
            )
            faces_by_md5: Dict[str, list] = {}
            for f in all_actor_faces:
                faces_by_md5.setdefault(f.file_md5, []).append(f)

            for md5 in sorted(actor_md5s):
                faces = faces_by_md5.get(md5, [])
                if not faces:
                    continue
                
                # Calculate brightness for each face
                # 注意: face_path 存的是相对 faces_dir 的路径，必须经
                # _resolve_face_file 解析成绝对路径再读；此前直接 cv2.imread
                # 相对路径 → 按进程 CWD 解析 → 必然失败 → brightness 恒为 0，
                # 导致下面"正常亮度优先"从未生效（一直是死代码）。
                import cv2
                for face in faces:
                    try:
                        _fp = _resolve_face_file(face.face_path)
                        img = cv2.imread(str(_fp)) if _fp else None
                        if img is not None:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            face.brightness = float(gray.mean())
                        else:
                            face.brightness = 0
                    except Exception:
                        face.brightness = 0
                
                # Prefer faces with normal brightness (60-200)
                normal_brightness_faces = [f for f in faces if 60 <= f.brightness <= 200]
                candidates = normal_brightness_faces if normal_brightness_faces else faces
                
                best = min(
                    candidates,
                    key=lambda f: abs(f.yaw or 0) + abs(f.pitch or 0) + abs(f.roll or 0),
                )
                task = task_by_md5.get(md5)
                primary_path = _get_primary_path(task) if task else None
                file_name = Path(primary_path).name if primary_path else md5
                similar.append({
                    "id": best.id,               # 供"设为封面"接口引用
                    "is_cover": bool(best.is_cover),
                    "file_md5": md5,
                    "file_name": file_name,
                    "face_path": _abs(best.face_path),
                    "yaw": best.yaw,
                    "pitch": best.pitch,
                    "roll": best.roll,
                    "quality_score": best.quality_score,
                    "eyes_open_score": best.eyes_open_score,
                })

        # ── All faces（该人物的全部人脸，供详情页完整核对） ──
        # 背景: 合并/重建后 actor_faces 可能有几十张脸，但"相似特征来源"是
        # **每个源文件只取 1 张**代表脸——若这些脸都来自同一个视频（合并的两个
        # actor 常同源），界面看起来"一张都没多"，容易被误认为合并丢了脸。
        # 因此额外返回全量人脸列表 + 计数。
        _all_faces_rows = db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all()

        def _file_name_of(md5: Optional[str]) -> str:
            _task = task_by_md5.get(md5) if md5 else None
            _primary = _get_primary_path(_task) if _task else None
            return Path(_primary).name if _primary else (md5 or "未知来源")

        all_faces = []
        for _f in _all_faces_rows:
            all_faces.append({
                "id": _f.id,          # 供"设为封面"接口引用
                "is_cover": bool(_f.is_cover),
                "file_md5": _f.file_md5,
                "file_name": _file_name_of(_f.file_md5),
                "face_path": _abs(_f.face_path),
                "yaw": _f.yaw,
                "pitch": _f.pitch,
                "roll": _f.roll,
                "quality_score": _f.quality_score,
                "eyes_open_score": _f.eyes_open_score,
            })
        # 封面优先，其余按质量分降序（便于一眼看到最佳样本）
        all_faces.sort(key=lambda x: (not x["is_cover"], -(x["quality_score"] or 0)))

        # Parse aliases from JSON string
        aliases_list = []
        if actor.aliases:
            try:
                aliases_list = json.loads(actor.aliases) if actor.aliases.startswith('[') else [x.strip() for x in actor.aliases.split(',') if x.strip()]
            except:
                aliases_list = [x.strip() for x in actor.aliases.split(',') if x.strip()]

        return {
            "id": actor.id,
            "code": actor.code,
            "name": actor.name,
            "primary_name": actor.primary_name,
            "aliases": aliases_list,
            "cover_face": _abs(actor.cover_face),
            "created_at": actor.created_at.isoformat() if actor.created_at else None,
            "avg_face_sharpness": actor.avg_face_sharpness,
            "collaborators": collaborators,
            "faces": {
                "primary": primary,
                "similar": similar,
                "all": all_faces,
                "total": len(all_faces),
                "file_count": len({f["file_md5"] for f in all_faces if f["file_md5"]}),
            },
        }
    finally:
        db.close()


def set_actor_cover(db, actor_id: int, face_id: int):
    """把指定人脸设为该人物的封面（人工指定）。

    单一事实来源：同步 actor_faces.is_cover 与 actors.cover_face —— 先清空该人物
    全部 is_cover，再置选中项为 1，并把 cover_face 写成相对 faces_dir 的路径
    （与 create_actor / 合并 保持一致，避免两处脱节）。

    Returns:
        (cover_rel_path, error_message)；error_message 非空表示未改动任何数据。
    """
    actor = db.query(Actor).filter(Actor.id == actor_id).first()
    if not actor:
        return None, f"人物 {actor_id} 不存在"

    face = (
        db.query(ActorFace)
        .filter(ActorFace.id == face_id, ActorFace.actor_id == actor_id)
        .first()
    )
    if not face:
        return None, f"人脸 {face_id} 不属于人物 {actor_id}"

    _p = _resolve_face_file(face.face_path)
    if _p is None or not _p.exists():
        return None, "该人脸的图片文件不存在，无法设为封面"

    # 单一事实来源：交由共享 helper 统一 is_cover 与 cover_face
    from src.db.actor_cover import apply_actor_cover
    rel = apply_actor_cover(db, actor_id, _api_faces_dir(), preferred_path=face.face_path)
    db.commit()
    return rel, None


@app.post("/actors/{actor_id}/cover")
async def set_actor_cover_endpoint(actor_id: int, request: SetCoverRequest):
    """人工指定人物封面：从该人物的"全部人脸"里选一张。

    body: {"face_id": <actor_faces.id>}
    """
    db = get_session(engine)
    try:
        rel, err = set_actor_cover(db, actor_id, request.face_id)
        if err:
            db.rollback()
            raise HTTPException(status_code=400, detail=err)
        return {
            "status": "success",
            "cover_face": rel,
            "cover_face_abs": str(_api_faces_dir() / rel),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"设置封面失败: {e}")
    finally:
        db.close()


@app.post("/actors/{actor_id}/face-type")
async def set_actor_face_type(actor_id: int, request: dict):
    """标记人物的人脸完整度（训练数据积累）。

    body: {"face_type": "complete" | "partial" | "unknown"}
    - complete: 完整人脸（正样本）
    - partial: 局部/误检（负样本）
    - unknown: 清除标注
    """
    face_type = request.get("face_type")
    if face_type not in ("complete", "partial", "unknown", None):
        raise HTTPException(status_code=400, detail="face_type 必须是 complete/partial/unknown")

    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")
        actor.face_type = face_type if face_type != "unknown" else None
        db.commit()
        return {"success": True, "actor_id": actor_id, "face_type": actor.face_type}
    finally:
        db.close()


def _parse_alias_list(raw: Optional[str]) -> List[str]:
    """把 Actor.aliases（JSON 数组或逗号分隔）解析为字符串列表。"""
    if not raw:
        return []
    try:
        val = json.loads(raw) if raw.strip().startswith("[") else None
        if isinstance(val, list):
            return [x.strip() for x in val if isinstance(x, str) and x.strip()]
    except Exception:
        pass
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def merge_actor_identity(target, sources) -> dict:
    """合并人物"身份信息"，保证人工命名/别名不因合并而丢失。

    历史 bug：合并后源人物被 `db.delete(src)`，其 name/primary_name/aliases
    一并消失；若被重命名的那个恰好不是 target，用户刚起的名字就丢了。

    规则(信息只增不减):
    1. target 的 name / primary_name 为空 → 依次从 sources 里取第一个非空值
    2. 所有 source 的 name / primary_name / aliases 并入 target.aliases（去重、
       保序），但等于 target 最终 name / primary_name 的不重复计入
    3. target 原有的 aliases 保留在最前

    Returns:
        {"name": str|None, "primary_name": str|None, "aliases": list[str]}
    """
    name = getattr(target, "name", None)
    primary = getattr(target, "primary_name", None)

    for src in sources:
        if not primary and getattr(src, "primary_name", None):
            primary = src.primary_name
        if not name and getattr(src, "name", None):
            name = src.name

    merged: List[str] = list(_parse_alias_list(getattr(target, "aliases", None)))
    for src in sources:
        for attr in ("name", "primary_name"):
            val = getattr(src, attr, None)
            if val and val != name and val != primary and val not in merged:
                merged.append(val)
        for al in _parse_alias_list(getattr(src, "aliases", None)):
            if al and al != name and al != primary and al not in merged:
                merged.append(al)

    # target 自身原来的 name/primary_name 若已被覆盖, 也保留为别名(不丢信息)
    for attr in ("name", "primary_name"):
        old = getattr(target, attr, None)
        if old and old != name and old != primary and old not in merged:
            merged.insert(0, old)

    return {"name": name, "primary_name": primary, "aliases": merged}


@app.post("/actors/merge")
async def merge_actors(request: MergeActorsRequest):
    """Merge multiple actors into one (first ID in request becomes the target).

    合并时把 source 的人脸文件物理搬移到 target 目录，删除 source 的磁盘目录，
    保证磁盘与 DB 一一对应（历史 bug：只迁移 DB 记录，文件留在 source 目录，
    source 删除后目录悬空 / target 文件夹视图不完整）。

    信息保护:
    - **target 由请求顺序决定**（actor_ids[0]），不再用查询返回顺序 ——
      以前用 `actors[0]`（通常是 id 最小者），用户想保留的那个未必是它，
      且前端用 set 传参顺序不确定 → 合并结果不可预期。
    - 人工命名/别名**不因合并丢失**（见 merge_actor_identity）。
    - 人脸文件缺失不中止，但会如实报告（不擅自杀数据）。
    """
    if len(request.actor_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 actors required")

    db = get_session(engine)
    try:
        actors = db.query(Actor).filter(Actor.id.in_(request.actor_ids)).all()
        if len(actors) != len(request.actor_ids):
            # 诊断: 点名哪些 id 不存在。合并失败常因前端勾选集合残留了已被
            # 删除的 actor id（_ACTOR_SELECTION 跨页持久、不自动清理）——只回
            # "Some actors not found" 无法定位是哪个 id，故同时写日志并放进
            # detail，前端直接把缺失 id 展示出来。
            _found = {a.id for a in actors}
            _missing = [aid for aid in request.actor_ids if aid not in _found]
            logger.warning(
                "合并人物失败: 请求 id=%s, 其中不存在=%s", request.actor_ids, _missing
            )
            raise HTTPException(
                status_code=404,
                detail=f"Some actors not found: {_missing}",
            )

        # 尊重请求顺序: actor_ids[0] 即目标
        by_id = {a.id: a for a in actors}
        ordered = [by_id[aid] for aid in request.actor_ids]
        target = ordered[0]
        sources = ordered[1:]
        target_code = target.code
        source_codes = [a.code for a in sources]

        # ── 身份信息合并(重命名/别名不丢) ──
        identity = merge_actor_identity(target, sources)

        # 先把 source 的人脸文件物理搬进 target 目录（按 DB face_path 解析）
        target_dir = _api_faces_dir() / target_code
        target_dir.mkdir(parents=True, exist_ok=True)
        next_idx = 0
        if target_dir.exists():
            for p in target_dir.glob("face_*.jpg"):
                try:
                    next_idx = max(next_idx, int(p.stem.split("_")[1]) + 1)
                except (IndexError, ValueError):
                    continue

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(_api_faces_dir()))
            except ValueError:
                return p.name

        moved = 0
        missing_files = 0     # 记录仍在, 但磁盘文件不存在(保留数据, 仅报告)
        outside_src = 0       # 记录指向源目录之外(不搬文件, 仅改归属)
        moved_files = []      # [(原路径, 新路径)] —— 中止时用于回滚磁盘
        for src in sources:
            src_dir = _api_faces_dir() / src.code
            for af in db.query(ActorFace).filter(ActorFace.actor_id == src.id).all():
                _p = _resolve_face_file(af.face_path)
                if _p is None or not _p.exists():
                    # 文件缺失: 保留记录(不擅自杀数据), 归属照常迁移
                    missing_files += 1
                    continue
                # 只搬移位于 source 目录内的文件（来源校验，防误搬）
                try:
                    belongs = _p.resolve().is_relative_to(src_dir.resolve())
                except Exception:
                    belongs = False
                if not belongs:
                    # 指向源目录之外: 搬移会是错的, 仅改归属(记录里的路径仍可解析)
                    outside_src += 1
                    continue
                dst = target_dir / f"face_{next_idx}.jpg"
                while dst.exists():
                    next_idx += 1
                    dst = target_dir / f"face_{next_idx}.jpg"
                try:
                    _p.rename(dst)
                except OSError as _mv_exc:
                    # 搬移失败: 回滚已搬走的文件 + DB, 然后中止。
                    # 否则"文件已移到目标目录、DB 却回滚"会造成新的不一致。
                    for _src_p, _dst_p in moved_files:
                        try:
                            if _dst_p.exists():
                                _dst_p.rename(_src_p)
                        except Exception:
                            pass
                    db.rollback()
                    raise HTTPException(
                        status_code=500,
                        detail=f"搬移人脸文件失败, 已中止合并并回滚: {_mv_exc}",
                    )
                moved_files.append((_p, dst))
                af.face_path = _rel(dst)
                next_idx += 1
                moved += 1

        # 先 flush 搬移时改动的 face_path, 再执行按 actor_id 的 bulk update——
        # 否则 bulk update 的会话同步可能丢掉尚未 flush 的路径修改。
        db.flush()

        # Merge embeddings
        all_embeddings = []
        for a in actors:
            if a.face_embedding:
                all_embeddings.append(decode_embedding(a.face_embedding))
        if all_embeddings:
            avg = np.mean(all_embeddings, axis=0)
            target.face_embedding = encode_embedding(avg)

        # 写回合并后的身份信息(人工命名/别名不丢)
        target.name = identity["name"]
        target.primary_name = identity["primary_name"]
        target.aliases = json.dumps(identity["aliases"], ensure_ascii=False) if identity["aliases"] else None

        # Reassign face records and tasks from sources → target
        for src in sources:
            db.query(ActorFace).filter(ActorFace.actor_id == src.id).update(
                {"actor_id": target.id}
            )
            # 更新 task_actors 关联（先删除冲突记录）
            # 获取 target 已关联的 task_ids
            target_task_ids = [
                ta.task_id for ta in db.query(TaskActor.task_id)
                .filter(TaskActor.actor_id == target.id).all()
            ]
            # 删除 source 中与 target 重复的 task_actors 记录
            if target_task_ids:
                db.query(TaskActor).filter(
                    TaskActor.actor_id == src.id,
                    TaskActor.task_id.in_(target_task_ids)
                ).delete(synchronize_session=False)
            # 更新剩余的 task_actors 关联
            db.query(TaskActor).filter(TaskActor.actor_id == src.id).update(
                {"actor_id": target.id}
            )
            # 向后兼容：更新 legacy actor_id
            db.query(Task).filter(Task.actor_id == src.id).update(
                {"actor_id": target.id}
            )
            db.delete(src)

        # Refresh sample count
        target.sample_count = (
            db.query(ActorFace).filter(ActorFace.actor_id == target.id).count()
        )

        # 封面统一由共享 helper 维护（只选磁盘真实存在的脸，避免 cover_face 悬空）
        from src.db.actor_cover import apply_actor_cover
        apply_actor_cover(db, target.id, _api_faces_dir())

        db.commit()
        # 提交后删除 source 磁盘目录（含孤儿残留文件）
        for code in source_codes:
            _delete_actor_dir_on_disk(code)
        if moved:
            logging.getLogger(__name__).info("合并人物: 搬移 %d 个人脸文件 → %s", moved, target_dir)
        if missing_files or outside_src:
            logging.getLogger(__name__).warning(
                "合并人物: %d 条人脸文件缺失(记录已保留)、%d 条指向源目录之外(仅改归属)",
                missing_files, outside_src,
            )
        return {
            "status": "success",
            "merged_actor_id": target.id,
            "moved": moved,
            "name": target.name,
            "aliases": identity["aliases"],
            "missing_files": missing_files,
            "outside_source_dir": outside_src,
        }
    finally:
        db.close()


@app.delete("/actors/batch")
async def batch_delete_actors(request: BatchDeleteActorsRequest):
    """Delete multiple actors and their associated face records.

    同时删除磁盘上的人脸文件与人物目录，保证磁盘与 DB 一一对应
    （历史 bug：只删 DB 记录不删文件 → 目录孤儿累积 / code 重用混入残留）。
    """
    if not request.actor_ids:
        raise HTTPException(status_code=400, detail="actor_ids is required")

    db = get_session(engine)
    try:
        # 收集将删除的 actor（含 code），先删磁盘文件再删 DB 记录
        actors_to_delete = db.query(Actor).filter(
            Actor.id.in_(request.actor_ids)
        ).all()
        codes = [a.code for a in actors_to_delete]

        # 删除磁盘人脸文件（按 DB face_path 解析到 faces_dir 后删除）
        faces_to_delete = db.query(ActorFace).filter(
            ActorFace.actor_id.in_(request.actor_ids)
        ).all()
        for af in faces_to_delete:
            _p = _resolve_face_file(af.face_path)
            if _p is not None:
                try:
                    if _p.exists():
                        _p.unlink()
                except Exception as e:
                    logging.getLogger(__name__).warning("删除人脸文件失败 %s: %s", _p, e)
        # 删除 actor_faces（外键约束）
        db.query(ActorFace).filter(
            ActorFace.actor_id.in_(request.actor_ids)
        ).delete(synchronize_session=False)
        # 删除 task_actors 关联
        db.query(TaskActor).filter(
            TaskActor.actor_id.in_(request.actor_ids)
        ).delete(synchronize_session=False)
        # 清除 legacy actor_id
        db.query(Task).filter(
            Task.actor_id.in_(request.actor_ids)
        ).update({"actor_id": None}, synchronize_session=False)
        # 再删除 actors
        actors_deleted = db.query(Actor).filter(
            Actor.id.in_(request.actor_ids)
        ).delete(synchronize_session=False)
        db.commit()
        # 提交后删除人物磁盘目录（含任何残留孤儿文件）
        for code in codes:
            _delete_actor_dir_on_disk(code)
        return {"success": True, "deleted_count": actors_deleted}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.put("/actors/{actor_id}/rename")
async def rename_actor(actor_id: int, request: RenameActorRequest):
    """Rename an actor.

    界面显示名优先取"主名"，所以重命名必须同时写入主名，否则用户会"改了看不到变化"。
    旧显示名会保留进别名（信息只增不减）。
    """
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")

        new_name = (request.name or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="名称不能为空")

        old_display = (actor.primary_name or actor.name or "").strip()
        if old_display and old_display != new_name:
            try:
                aliases = json.loads(actor.aliases) if actor.aliases else []
            except Exception:
                aliases = []
            if old_display not in aliases:
                aliases.append(old_display)
            actor.aliases = json.dumps(aliases, ensure_ascii=False)

        actor.name = new_name
        actor.primary_name = new_name  # 显示优先主名
        db.commit()
        return {"status": "success", "name": new_name}
    finally:
        db.close()


@app.delete("/actors/{actor_id}")
async def delete_actor(actor_id: int):
    """Delete an actor (clears task associations, does not delete video files).

    同时删除磁盘人脸文件与人物目录，保证磁盘与 DB 一一对应。
    """
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")
        code = actor.code

        # 删除磁盘人脸文件
        for af in db.query(ActorFace).filter(ActorFace.actor_id == actor_id).all():
            _p = _resolve_face_file(af.face_path)
            if _p is not None:
                try:
                    if _p.exists():
                        _p.unlink()
                except Exception as e:
                    logging.getLogger(__name__).warning("删除人脸文件失败 %s: %s", _p, e)

        # Clear task associations (legacy actor_id)
        db.query(Task).filter(Task.actor_id == actor_id).update({"actor_id": None})
        # Remove task_actors associations
        db.query(TaskActor).filter(TaskActor.actor_id == actor_id).delete(synchronize_session=False)
        # Remove face records
        db.query(ActorFace).filter(ActorFace.actor_id == actor_id).delete()
        db.delete(actor)
        db.commit()
        # 提交后删除人物磁盘目录（含任何残留孤儿文件）
        _delete_actor_dir_on_disk(code)
        return {"status": "success"}
    finally:
        db.close()


@app.get("/actors/{actor_id}/tasks")
async def get_actor_tasks(actor_id: int):
    """Get all tasks (videos) associated with an actor via task_actors table."""
    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail="Actor not found")

        # 优先使用 task_actors 关联表
        task_ids_from_ta = [
            ta.task_id for ta in
            db.query(TaskActor).filter(TaskActor.actor_id == actor_id).all()
        ]

        # 向后兼容：也包含 actor_id 匹配的任务
        tasks = db.query(Task).filter(
            (Task.id.in_(task_ids_from_ta)) | (Task.actor_id == actor_id)
        ).all()

        return [
            {
                "id": t.id,
                "file_path": _get_primary_path(t),
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ]
    finally:
        db.close()


@app.get("/tasks/{task_id}/actors")
async def get_task_actors(task_id: int):
    """获取任务的所有人物（通过 task_actors 关联表）"""
    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # 从 task_actors 获取
        task_actor_recs = db.query(TaskActor).filter(TaskActor.task_id == task_id).all()
        actors = []
        if task_actor_recs:
            for ta in task_actor_recs:
                actor = db.query(Actor).filter(Actor.id == ta.actor_id).first()
                if actor:
                    actors.append({
                        "id": actor.id,
                        "code": actor.code,
                        "name": actor.name,
                        "cover_face": actor.cover_face,
                    })
        elif task.actor_id:
            # 向后兼容
            actor = db.query(Actor).filter(Actor.id == task.actor_id).first()
            if actor:
                actors.append({
                    "id": actor.id,
                    "code": actor.code,
                    "name": actor.name,
                    "cover_face": actor.cover_face,
                })

        return {"task_id": task_id, "actors": actors}
    finally:
        db.close()


# ── Audit Worker ───────────────────────────────────────────────────

@app.post("/audit/{task_id}/start")
async def trigger_audit(task_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_audit_worker, task_id)
    return {"status": "queued"}


@app.post("/task/{task_id}/start")
async def start_single_task(task_id: int):
    """启动单个任务分析 — 通过调度器入队。"""
    db = get_session(engine)
    scheduler = get_scheduler()

    # 确保调度器已启动
    if scheduler.status.value != "running":
        await scheduler.start()

    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # 如果是 ERROR/PAUSED/FAILED/COMPLETE 状态，重置
        if task.status in {"ERROR", "PAUSED", "FAILED", "COMPLETE"}:
            task.error_message = None
            task.progress = 0
            if task.entity_id:
                fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                if fa:
                    fa.mapped_tags = None

        task.status = 'QUEUED'
        task.queued_at = datetime.datetime.utcnow()
        db.commit()

        await scheduler.submit_task(task_id, duration=task.duration)
        return {"status": "success", "message": f"任务 {task_id} 已加入队列"}
    finally:
        db.close()


@app.delete("/task/batch")
async def delete_task_batch(request: BatchDeleteRequest):
    """批量删除任务 — 与 POST /task/batch-delete 等效，统一清理逻辑。"""
    from sqlalchemy import text as sa_text

    db = get_session(engine)
    try:
        deleted = 0
        for task_id in request.task_ids:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                continue

            # 收集帧文件目录
            analyses = db.query(FileAnalysis).filter(FileAnalysis.task_id == task_id).all()
            frames_dirs = [
                a.frames_path for a in analyses
                if a.frames_path and os.path.exists(a.frames_path)
            ]
            legacy_frames = getattr(task, "frames_path", None)
            if legacy_frames and os.path.exists(legacy_frames):
                frames_dirs.append(legacy_frames)

            # 用原始 SQL 清理所有关联记录，避免 ORM cascade 冲突
            db.execute(sa_text("DELETE FROM analysis_logs WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM task_actor_names WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM task_actors WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM tag_frame_mappings WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM tasks WHERE id = :tid"), {"tid": task_id})
            deleted += 1

            for frames_dir in frames_dirs:
                shutil.rmtree(frames_dir, ignore_errors=True)

        db.commit()
        return {"status": "success", "deleted": deleted}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
    finally:
        db.close()

@app.patch("/task/{task_id}/tags")
async def update_task_tags(task_id: int, request: dict):
    """更新单个任务的标签 — 写入 FileAnalysis（业务数据持久化核心）。"""
    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.entity_id:
            raise HTTPException(status_code=400, detail="Task has no entity_id")

        tags = request.get("tags", [])

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

        fa.mapped_tags = json.dumps(tags, ensure_ascii=False)
        db.commit()
        return {"status": "success", "task_id": task_id, "tags": tags}
    finally:
        db.close()


@app.post("/tasks/batch-replace-tag")
async def batch_replace_tag(request: dict):
    """批量替换标签 — 在所有任务的 FileAnalysis 中将 old_tag 替换为 new_tag。"""
    db = get_session(engine)
    try:
        task_ids = request.get("task_ids", [])
        old_tag = request.get("old_tag")
        new_tag = request.get("new_tag")

        if not old_tag or not new_tag:
            raise HTTPException(status_code=400, detail="old_tag and new_tag are required")

        updated_count = 0

        for tid in task_ids:
            task = db.query(Task).filter(Task.id == tid).first()
            if not task or not task.entity_id:
                continue

            fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
            if not fa or not fa.mapped_tags:
                continue

            try:
                tags_list = json.loads(fa.mapped_tags)
            except (json.JSONDecodeError, TypeError):
                continue

            if old_tag in tags_list:
                tags_list.remove(old_tag)
                if new_tag not in tags_list:
                    tags_list.append(new_tag)
                fa.mapped_tags = json.dumps(tags_list, ensure_ascii=False)
                updated_count += 1

        db.commit()
        return {"status": "success", "updated_count": updated_count}
    finally:
        db.close()


@app.post("/actors/rebuild-faces")
async def rebuild_faces_from_frames(
    background_tasks: BackgroundTasks,
    force: bool = False
):
    """从已有帧文件重建人脸特征（异步执行）
    
    扫描所有有帧文件但没有人物记录的任务，重新提取人脸并创建人物记录。
    结果写入日志，前端通过轮询 /actors/rebuild-faces/status 获取进度。
    
    Args:
        force: 是否强制重建（清除旧数据并重新提取）
    """
    import json
    status_file = Path("/tmp/ai_video_factory_rebuild_status.json")
    
    # 检查是否已有任务在运行
    if _rebuild_status_is_active(status_file):
        try:
            status = json.loads(status_file.read_text())
            if status.get("status") == "running":
                return {"message": "重建任务正在运行中", "status": status}
        except Exception:
            pass
    
    # 初始化状态文件
    initial_status = {
        "status": "running",
        "total": 0,
        "processed": 0,
        "success": [],
        "failed": [],
        "skipped": [],
        "force": force
    }
    status_file.write_text(json.dumps(initial_status))
    
    # 启动后台任务
    background_tasks.add_task(_rebuild_faces_worker, status_file)
    
    return {"message": "重建任务已启动，请在后台等待完成", "status": "running", "force": force}


def _rebuild_faces_worker(status_file: Path):
    """后台执行人脸重建"""
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    db = get_session(engine)
    status = {}
    
    try:
        # 读取 force 参数
        status = json.loads(status_file.read_text())
        force = status.get("force", False)
        
        if force:
            # 强制重建：处理所有有帧文件的 FileAnalysis
            analyses = db.query(FileAnalysis).filter(
                FileAnalysis.frames_path.isnot(None)
            ).all()
            # 获取对应的 Task（通过 task_id 关联）
            task_ids = [a.task_id for a in analyses if a.task_id]
            tasks = db.query(Task).filter(Task.id.in_(task_ids)).all() if task_ids else []
            logger.info(f"强制重建模式：将重新处理 {len(tasks)} 个任务")
        else:
            # 普通模式：只处理缺少人脸记录的任务
            # 有帧文件但没有 actor_faces 记录的 FileAnalysis
            analyses_with_no_faces = db.query(FileAnalysis).filter(
                FileAnalysis.frames_path.isnot(None),
                FileAnalysis.entity_id.isnot(None),
                ~FileAnalysis.entity_id.in_(
                    db.query(ActorFace.entity_id).filter(ActorFace.entity_id.isnot(None))
                )
            ).all()

            entity_ids = [a.entity_id for a in analyses_with_no_faces if a.entity_id]
            tasks = db.query(Task).filter(Task.entity_id.in_(entity_ids)).all() if entity_ids else []
            logger.info(f"普通重建模式：将处理 {len(tasks)} 个缺少人脸的任务")
        
        # 更新总数
        status["total"] = len(tasks)
        status_file.write_text(json.dumps(status))
        
        if not tasks:
            status["status"] = "completed"
            status["message"] = "没有需要重建的任务"
            status_file.write_text(json.dumps(status))
            return
        
        logger.info("正在初始化 FaceRecognizer...")
        from src.core.face_recognition import FaceRecognizer
        recognizer = FaceRecognizer()
        logger.info("FaceRecognizer 初始化完成")
        
        logger.info(f"开始处理 {len(tasks)} 个任务")
        for idx, task in enumerate(tasks):
            logger.info(f"处理任务 {idx+1}/{len(tasks)}: task_id={task.id}, entity_id={task.entity_id}")
            file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
            frames_path = file_analysis.frames_path if file_analysis else None
            logger.info(f"  frames_path: {frames_path}")
            if not frames_path:
                logger.warning(f"  Task {task.id}: frames_path is None, skipping")
                status["skipped"].append({
                    "task_id": task.id,
                    "reason": "frames_path is None"
                })
                status["processed"] += 1
                status_file.write_text(json.dumps(status))
                continue
            if not Path(frames_path).exists():
                status["skipped"].append({
                    "task_id": task.id,
                    "reason": f"帧目录不存在: {frames_path}"
                })
                status["processed"] += 1
                status_file.write_text(json.dumps(status))
                continue
            
            try:
                # 强制模式下，先清除旧的人脸数据
                if force:
                    # 删除旧的 actor_faces 记录（face_path 相对 faces_dir，
                    # 必须解析后删除，否则文件残留为孤儿）
                    affected_actor_ids = set()
                    old_faces = db.query(ActorFace).filter(ActorFace.task_id == task.id).all()
                    for old_face in old_faces:
                        if old_face.actor_id is not None:
                            affected_actor_ids.add(old_face.actor_id)
                        _p = _resolve_face_file(old_face.face_path)
                        if _p is not None:
                            try:
                                if _p.exists():
                                    _p.unlink()
                            except Exception as e:
                                logger.warning(f"删除人脸文件失败: {_p}, {e}")
                        db.delete(old_face)
                    # 回写受影响 actor 的 sample_count（ORM 批量 delete 需先逐条删）
                    db.flush()
                    for _aid in affected_actor_ids:
                        _act = db.query(Actor).filter(Actor.id == _aid).first()
                        if _act is not None:
                            _act.sample_count = (
                                db.query(ActorFace)
                                .filter(ActorFace.actor_id == _aid).count()
                            )
                    
                    # 删除 task_actors 关联
                    db.query(TaskActor).filter(TaskActor.task_id == task.id).delete()
                    
                    # 重置 actor_id
                    task.actor_id = None
                    
                    db.commit()
                    logger.info(f"Task {task.id}: 已清除旧的人脸数据")
                
                # 使用多人物识别
                file_md5 = task.file_md5  # 直接从 task 表获取，不依赖 entity

                # 优先使用两阶段视频人脸检测，失败则回退到帧目录方式
                video_path = task.file_path
                if not video_path and task.entity:
                    video_path = _get_primary_path(task)

                actor_ids = None

                if video_path and Path(video_path).exists():
                    try:
                        logger.info(f"Task {task.id}: 使用两阶段视频人脸检测: {video_path}")
                        actor_ids = recognizer.recognize_actors_from_video(video_path, task.id, file_md5)
                        actor_ids = [a for a in actor_ids if a is not None]
                        logger.info(f"Task {task.id}: 两阶段检测识别到 {len(actor_ids)} 个人物")
                    except Exception as video_e:
                        logger.warning(f"Task {task.id}: 两阶段视频检测失败，回退到帧目录方式: {video_e}")
                        actor_ids = None

                if actor_ids is None:
                    logger.info(f"Task {task.id}: 使用帧目录方式识别人脸: {frames_path}")
                    actor_ids = recognizer.recognize_actors(frames_path, task.id, file_md5)
                    actor_ids = [a for a in actor_ids if a is not None]

                if actor_ids:
                    # 设置 actor_id 为第一个人物（向后兼容）
                    task.actor_id = actor_ids[0]
                    # 创建 TaskActor 关联记录
                    for aid in actor_ids:
                        existing_ta = db.query(TaskActor).filter(
                            TaskActor.task_id == task.id,
                            TaskActor.actor_id == aid
                        ).first()
                        if not existing_ta:
                            db.add(TaskActor(task_id=task.id, actor_id=aid))

                    # 统计人脸数量
                    faces_count = db.query(ActorFace).filter(ActorFace.task_id == task.id).count()
                    
                    status["success"].append({
                        "task_id": task.id,
                        "action": "rebuilt" if force else "created",
                        "actor_ids": actor_ids,
                        "faces_count": faces_count
                    })
                else:
                    status["skipped"].append({
                        "task_id": task.id,
                        "reason": "未检测到人脸"
                    })
                
                db.commit()
                logger.info(f"Task {task.id}: 重建成功")
                
            except Exception as e:
                db.rollback()
                status["failed"].append({
                    "task_id": task.id,
                    "error": str(e)
                })
                logger.error(f"Task {task.id}: 重建失败 - {e}")
            
            status["processed"] += 1
            status_file.write_text(json.dumps(status))
        
        status["status"] = "completed"
        status["message"] = f"重建完成: {len(status['success'])} 成功, {len(status['failed'])} 失败, {len(status['skipped'])} 跳过"
        status_file.write_text(json.dumps(status))
        
    except Exception as e:
        status["status"] = "failed"
        status["message"] = f"重建失败: {str(e)}"
        status_file.write_text(json.dumps(status))
    finally:
        db.close()


@app.get("/actors/rebuild-faces/status")
async def get_rebuild_faces_status():
    """获取人脸重建进度"""
    status_file = Path("/tmp/ai_video_factory_rebuild_status.json")
    
    if not status_file.exists():
        return {"status": "idle", "message": "没有正在运行的重建任务"}
    
    try:
        status = json.loads(status_file.read_text())
        return status
    except Exception as e:
        return {"status": "error", "message": f"读取状态失败: {str(e)}"}


# ── 选择性人脸重建 API ────────────────────────────────────────────

class TaskRebuildRequest(BaseModel):
    pass


class ActorRebuildRequest(BaseModel):
    pass


class BatchActorRebuildRequest(BaseModel):
    actor_ids: List[int]


def _rebuild_status_is_active(status_file: Path, stale_seconds: int = 600) -> bool:
    """判断 rebuild 状态文件是否代表「仍在运行」。

    历史 bug(2026-09): 任务崩溃/中断后状态文件残留 running, 之后再次
    重建会被误判「正在运行中」而拒绝启动, 前端轮询永远卡在旧状态。
    因此 running 状态还需文件足够新鲜: mtime 距今超过 stale_seconds
    视为残留, 允许重新启动。
    """
    try:
        if not status_file.exists():
            return False
        status = json.loads(status_file.read_text())
        if status.get("status") != "running":
            return False
        import time as _t
        age = _t.time() - status_file.stat().st_mtime
        return age < stale_seconds
    except Exception:
        return False


@app.post("/tasks/{task_id}/rebuild-faces")
async def rebuild_task_faces_endpoint(
    task_id: int,
    background_tasks: BackgroundTasks,
):
    """按作品重建人脸：只重建指定视频中的人脸

    从该视频的所有帧中重新检测人脸，保留其他视频的人脸数据。
    """
    status_file = Path(f"/tmp/ai_video_factory_task_rebuild_{task_id}.json")

    if _rebuild_status_is_active(status_file):
        try:
            status = json.loads(status_file.read_text())
            return {"message": "重建任务正在运行中", "status": status}
        except Exception:
            pass

    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    finally:
        db.close()

    initial_status = {
        "status": "running",
        "task_id": task_id,
        "total": 1,
        "processed": 0,
        "result": None,
    }
    status_file.write_text(json.dumps(initial_status))

    background_tasks.add_task(_rebuild_task_faces_worker, task_id, status_file)

    return {"message": "作品人脸重建任务已启动", "task_id": task_id, "status": "running"}


# 人脸识别并发信号量：限制同时最多 N 个重建任务并行写库，
# 避免 SQLite 单写锁在高并发下频繁 database is locked。
# N 来自 config face_recognition.max_concurrent：
#   - 0 或未配置：自动 = min(max(2, CPU核数/2), 8)，根据系统资源动态计算
#   - 正整数：按配置值
# 超出的任务在信号量处排队，自动逐个执行。
import threading

def _get_face_rebuild_concurrency() -> int:
    """计算人脸识别并发数（基于 CPU 核数，可被 config 覆盖）。"""
    try:
        from src.core.config import ConfigManager
        configured = int(ConfigManager().get("face_recognition.max_concurrent", 0))
        if configured > 0:
            return configured
    except Exception:
        pass
    try:
        import multiprocessing
        cpus = multiprocessing.cpu_count()
    except Exception:
        cpus = 4
    # 每个 worker 主要用 1 核做模型推理；留一半核给系统和其他任务
    return min(max(2, cpus // 2), 8)

_face_rebuild_semaphore = threading.Semaphore(_get_face_rebuild_concurrency())


def _rebuild_task_faces_worker(task_id: int, status_file: Path):
    """后台执行单任务人脸重建（含并发信号量 + database is locked 重试）"""

    # 获取并发许可，排队等待（最多 2 个同时执行）
    _face_rebuild_semaphore.acquire()
    try:
        _rebuild_task_faces_worker_inner(task_id, status_file)
    finally:
        _face_rebuild_semaphore.release()


def _rebuild_task_faces_worker_inner(task_id: int, status_file: Path):
    """单任务人脸重建实际逻辑（在信号量保护下执行）"""
    import logging
    import time as _time
    logger_local = logging.getLogger(__name__)

    status = {}
    try:
        status = json.loads(status_file.read_text())

        from src.core.face_recognition import FaceRecognizer
        recognizer = FaceRecognizer()

        # SQLite 单写锁：并发重建时可能 database is locked，等待后重试（指数退避）
        max_retries = 5
        for attempt in range(max_retries + 1):
            db = get_session(engine)
            try:
                result = recognizer.rebuild_task_faces(task_id, db_session=db)
                db.commit()  # 提交事务
                status["result"] = result
                break
            except Exception as e:
                db.rollback()
                if is_locked_error(e) and attempt < max_retries:
                    note_lock_retry()
                    logger_local.warning(
                        "任务 %d 数据库锁冲突（第 %d 次），%.1f 秒后重试...",
                        task_id, attempt + 1, 2 ** attempt,
                    )
                    _time.sleep(2 ** attempt)
                    continue
                if is_locked_error(e):
                    note_lock_failure()
                raise
            finally:
                db.close()

        status["processed"] = 1
        status["status"] = "completed"
        status_file.write_text(json.dumps(status))

    except Exception as e:
        logger_local.error("任务 %d 人脸重建失败: %s", task_id, e)
        status["status"] = "failed"
        status["message"] = str(e)
        status_file.write_text(json.dumps(status))


@app.get("/tasks/{task_id}/rebuild-faces/status")
async def get_task_rebuild_status(task_id: int):
    """获取单任务人脸重建进度"""
    status_file = Path(f"/tmp/ai_video_factory_task_rebuild_{task_id}.json")

    if not status_file.exists():
        return {"status": "idle", "message": "没有重建任务记录"}

    try:
        status = json.loads(status_file.read_text())
        return status
    except Exception as e:
        return {"status": "error", "message": f"读取状态失败: {str(e)}"}


@app.post("/actors/{actor_id}/rebuild-faces")
async def rebuild_actor_faces_endpoint(
    actor_id: int,
    background_tasks: BackgroundTasks,
):
    """按人物重建人脸：只重建指定人物的人脸

    从该人物关联的所有视频中重新提取该人物的人脸，
    重新计算特征向量并更新人物记录。
    """
    status_file = Path(f"/tmp/ai_video_factory_actor_rebuild_{actor_id}.json")

    if _rebuild_status_is_active(status_file):
        try:
            status = json.loads(status_file.read_text())
            if status.get("status") == "running":
                return {"message": "重建任务正在运行中", "status": status}
        except Exception:
            pass

    db = get_session(engine)
    try:
        actor = db.query(Actor).filter(Actor.id == actor_id).first()
        if not actor:
            raise HTTPException(status_code=404, detail=f"人物 {actor_id} 不存在")
    finally:
        db.close()

    initial_status = {
        "status": "running",
        "actor_id": actor_id,
        "total": 1,
        "processed": 0,
        "result": None,
    }
    status_file.write_text(json.dumps(initial_status))

    background_tasks.add_task(_rebuild_actor_faces_worker, actor_id, status_file)

    return {"message": "人物人脸重建任务已启动", "actor_id": actor_id, "status": "running"}


def _rebuild_actor_faces_worker(actor_id: int, status_file: Path):
    """后台执行单人物人脸重建"""
    import logging
    logger_local = logging.getLogger(__name__)

    status = {}
    try:
        status = json.loads(status_file.read_text())

        from src.core.face_recognition import FaceRecognizer
        recognizer = FaceRecognizer()

        db = get_session(engine)
        try:
            result = recognizer.rebuild_actor_faces(actor_id, db_session=db)
            # 显式提交: rebuild_actor_faces 收到外部 session 时不自行 commit
            # (own_session=False), 缺 commit 会导致 DB 改动回滚而磁盘已变 → 不一致
            # (历史 bug: 人物重建"成功"但 DB 记录从未更新)。
            try:
                db.commit()
            except Exception as _ce:
                db.rollback()
                logger_local.warning("人物 %d 重建后 commit 失败: %s", actor_id, _ce)
            status["result"] = result
        finally:
            db.close()

        status["processed"] = 1
        status["status"] = "completed"
        status_file.write_text(json.dumps(status))

    except Exception as e:
        logger_local.error("人物 %d 人脸重建失败: %s", actor_id, e)
        status["status"] = "failed"
        status["message"] = str(e)
        status_file.write_text(json.dumps(status))


@app.get("/actors/{actor_id}/rebuild-faces/status")
async def get_actor_rebuild_status(actor_id: int):
    """获取单人物人脸重建进度"""
    status_file = Path(f"/tmp/ai_video_factory_actor_rebuild_{actor_id}.json")

    if not status_file.exists():
        return {"status": "idle", "message": "没有重建任务记录"}

    try:
        status = json.loads(status_file.read_text())
        return status
    except Exception as e:
        return {"status": "error", "message": f"读取状态失败: {str(e)}"}


@app.post("/actors/batch-rebuild-faces")
async def batch_rebuild_actor_faces_endpoint(
    request: BatchActorRebuildRequest,
    background_tasks: BackgroundTasks,
):
    """批量选择性重建：批量重建指定人物的人脸

    请求体中指定人物ID列表，后台异步执行并返回统计结果。
    """
    actor_ids = request.actor_ids
    if not actor_ids:
        raise HTTPException(status_code=400, detail="人物ID列表不能为空")

    status_file = Path("/tmp/ai_video_factory_batch_actor_rebuild_status.json")

    if _rebuild_status_is_active(status_file):
        try:
            status = json.loads(status_file.read_text())
            if status.get("status") == "running":
                return {"message": "批量重建任务正在运行中", "status": status}
        except Exception:
            pass

    db = get_session(engine)
    try:
        valid_ids = []
        for aid in actor_ids:
            actor = db.query(Actor).filter(Actor.id == aid).first()
            if actor:
                valid_ids.append(aid)
        if not valid_ids:
            raise HTTPException(status_code=404, detail="指定的人物都不存在")
    finally:
        db.close()

    initial_status = {
        "status": "running",
        "total": len(valid_ids),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "results": [],
        "actor_ids": valid_ids,
    }
    status_file.write_text(json.dumps(initial_status))

    background_tasks.add_task(_batch_rebuild_actors_worker, valid_ids, status_file)

    return {
        "message": "批量人物人脸重建任务已启动",
        "total": len(valid_ids),
        "status": "running",
    }


def _batch_rebuild_actors_worker(actor_ids: List[int], status_file: Path):
    """后台执行批量人物人脸重建"""
    import logging
    logger_local = logging.getLogger(__name__)

    status = {}
    try:
        status = json.loads(status_file.read_text())

        from src.core.face_recognition import FaceRecognizer
        recognizer = FaceRecognizer()

        db = get_session(engine)
        try:
            for idx, actor_id in enumerate(actor_ids):
                logger_local.info("批量重建进度: %d/%d (人物 %d)", idx + 1, len(actor_ids), actor_id)
                result = recognizer.rebuild_actor_faces(actor_id, db_session=db)
                # 显式提交每个人物的 DB 改动(历史 bug: 缺 commit → DB 回滚而磁盘已变)
                try:
                    db.commit()
                except Exception as _ce:
                    db.rollback()
                    logger_local.warning("批量重建 actor %d 后 commit 失败: %s", actor_id, _ce)
                status["results"].append(result)

                if result["status"] == "success":
                    status["success"] += 1
                elif result["status"] == "skipped":
                    status["skipped"] += 1
                else:
                    status["failed"] += 1

                status["processed"] = idx + 1
                status_file.write_text(json.dumps(status))
        finally:
            db.close()

        status["status"] = "completed"
        status["message"] = (
            f"批量重建完成: {status['success']} 成功, "
            f"{status['failed']} 失败, {status['skipped']} 跳过"
        )
        status_file.write_text(json.dumps(status))

    except Exception as e:
        logger_local.error("批量人物人脸重建失败: %s", e)
        status["status"] = "failed"
        status["message"] = str(e)
        status_file.write_text(json.dumps(status))


@app.get("/actors/batch-rebuild-faces/status")
async def get_batch_rebuild_status():
    """获取批量人物人脸重建进度"""
    status_file = Path("/tmp/ai_video_factory_batch_actor_rebuild_status.json")

    if not status_file.exists():
        return {"status": "idle", "message": "没有正在运行的批量重建任务"}

    try:
        status = json.loads(status_file.read_text())
        return status
    except Exception as e:
        return {"status": "error", "message": f"读取状态失败: {str(e)}"}


@app.post("/actors/confirm-names")
async def confirm_actor_names(background_tasks: BackgroundTasks):
    """
    确认人物名：从文件名和水印中提取并交叉验证
    """
    status_file = Path("/tmp/ai_video_factory_actor_names_status.json")
    
    # 检查是否已有任务在运行
    if _rebuild_status_is_active(status_file):
        try:
            status = json.loads(status_file.read_text())
            if status.get("status") == "running":
                return {"message": "人物名确认任务正在运行中", "status": status}
        except Exception:
            pass
    
    # 初始化状态文件
    initial_status = {
        "status": "running",
        "total": 0,
        "processed": 0,
        "success": [],
        "failed": [],
        "skipped": [],
        "message": "正在初始化..."
    }
    status_file.write_text(json.dumps(initial_status))
    
    # 启动后台任务
    background_tasks.add_task(_confirm_actor_names_worker, status_file)
    
    return {"message": "人物名确认任务已启动，请在后台等待完成", "status": "running"}


def _confirm_actor_names_worker(status_file: Path):
    """后台执行人物名确认"""
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    db = get_session(engine)
    status = {}
    
    try:
        from src.core.actor_name_extractor import ActorNameExtractor
        from src.core.vlm import create_vlm_client
        
        # 获取所有人物
        actors = db.query(Actor).all()
        
        status = json.loads(status_file.read_text())
        status["total"] = len(actors)
        status_file.write_text(json.dumps(status))
        
        if not actors:
            status["status"] = "completed"
            status["message"] = "没有人物需要确认"
            status_file.write_text(json.dumps(status))
            return
        
        extractor = ActorNameExtractor()
        vlm_client = create_vlm_client()
        
        for actor in actors:
            # 获取该人物关联的所有任务（通过 task_actors 或 legacy actor_id）
            task_ids_from_ta = [
                ta.task_id for ta in
                db.query(TaskActor).filter(TaskActor.actor_id == actor.id).all()
            ]
            tasks = db.query(Task).filter(
                (Task.id.in_(task_ids_from_ta)) | (Task.actor_id == actor.id)
            ).all()
            
            if not tasks:
                status["skipped"].append({
                    "actor_id": actor.id,
                    "reason": "没有关联的任务"
                })
                status["processed"] += 1
                status_file.write_text(json.dumps(status))
                continue
            
            # 从文件名提取人物名
            filename_names = []
            for task in tasks:
                primary_path = _get_primary_path(task)
                if primary_path:
                    names = extractor.extract_from_filename(primary_path)
                    filename_names.extend(names)
            
            # 从水印提取人物名
            watermark_names = []
            for task in tasks:
                # 从 FileAnalysis 获取 frames_path
                file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                frames_path = file_analysis.frames_path if file_analysis else None
                if frames_path:
                    try:
                        names = extractor.extract_from_watermark(frames_path, vlm_client)
                        watermark_names.extend(names)
                    except Exception as e:
                        logger.warning(f"水印识别失败 Task {task.id}: {e}")
            
            # 合并所有候选名称
            all_names = filename_names + watermark_names
            
            if not all_names:
                status["skipped"].append({
                    "actor_id": actor.id,
                    "reason": "未提取到任何候选名称"
                })
                status["processed"] += 1
                status_file.write_text(json.dumps(status))
                continue
            
            # 确认人物名
            confirmed_name = extractor.confirm_actor_name(actor.id, all_names, db)
            
            if confirmed_name:
                actor.name = confirmed_name
                db.commit()
                status["success"].append({
                    "actor_id": actor.id,
                    "name": confirmed_name,
                    "candidates": list(set(all_names))
                })
                logger.info(f"人物 {actor.id} 确认名称: {confirmed_name}")
            else:
                status["failed"].append({
                    "actor_id": actor.id,
                    "candidates": list(set(all_names)),
                    "reason": "无法确认名称"
                })
                logger.warning(f"人物 {actor.id} 无法确认名称，候选: {all_names}")
            
            status["processed"] += 1
            status_file.write_text(json.dumps(status))
        
        status["status"] = "completed"
        status["message"] = f"确认完成: {len(status['success'])} 成功, {len(status['failed'])} 失败, {len(status['skipped'])} 跳过"
        status_file.write_text(json.dumps(status))
        
    except Exception as e:
        status["status"] = "failed"
        status["message"] = f"确认失败: {str(e)}"
        status_file.write_text(json.dumps(status))
        logger.error(f"人物名确认失败: {e}")
    finally:
        db.close()


@app.get("/actors/confirm-names/status")
async def get_confirm_names_status():
    """获取人物名确认进度"""
    status_file = Path("/tmp/ai_video_factory_actor_names_status.json")
    
    if not status_file.exists():
        return {"status": "idle", "message": "没有正在运行的确认任务"}
    
    try:
        status = json.loads(status_file.read_text())
        return status
    except Exception as e:
        return {"status": "error", "message": f"读取状态失败: {str(e)}"}


@app.post("/task/batch-start")
async def batch_start_tasks(request: BatchStartRequest):
    """批量启动任务 - 使用调度器管理并发"""
    db = get_session(engine)
    scheduler = get_scheduler()
    
    # 确保调度器已启动
    if scheduler.status.value != "running":
        await scheduler.start()
    
    try:
        queued = 0
        # 根据 force 参数决定处理的状态集合
        if request.force:
            # force=True: 处理所有状态（包括正在处理中的任务）
            processable_statuses = {
                "PENDING", "PAUSED", "ERROR", "FAILED", "QUEUED", "COMPLETE",
                "CHECKING", "SIFTING", "ANALYZING", "CLUSTERING", "WRITING"
            }
        else:
            # force=False: 只处理非处理中状态(待处理/暂停/出错/排队)
            processable_statuses = {"PENDING", "PAUSED", "ERROR", "FAILED", "QUEUED"}
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Batch start: task_ids={request.task_ids}, force={request.force}, processable={processable_statuses}")
        
        tasks_to_start: List[Task] = []
        for task_id in request.task_ids:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.warning(f"Task {task_id} not found")
                continue
            logger.info(f"Task {task_id} status: {task.status}, in processable: {task.status in processable_statuses}")
            if task.status not in processable_statuses:
                logger.warning(f"Task {task_id} status {task.status} not in processable statuses")
                continue
            tasks_to_start.append(task)

        # 时长惰性获取：导入不再探时长，激活时并行补探一次（保留「短任务优先」排序）。
        await _fill_missing_durations(db, tasks_to_start)

        for task in tasks_to_start:
            # 如果是 ERROR/FAILED 状态，清除错误信息并重置进度
            if task.status in {"ERROR", "FAILED"}:
                task.error_message = None
                task.progress = 0
                # 清除 FileAnalysis 中的标签数据（业务数据持久化核心）
                if task.entity_id:
                    fa = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
                    if fa:
                        fa.mapped_tags = None
            # PAUSED 状态保留进度，不重置
            
            # 更新任务状态为 QUEUED
            task.status = 'QUEUED'
            task.queued_at = datetime.datetime.utcnow()
            db.commit()
            
            # 提交到调度器队列（无论是否已在队列中，都重新提交）; 带时长以支持短任务优先
            await scheduler.submit_task(task.id, duration=task.duration)
            queued += 1
        
        return {
            "status": "success",
            "queued": queued,
            "message": f"已将 {queued} 个任务加入调度队列"
        }
    finally:
        db.close()

@app.post("/task/batch-delete")
async def batch_delete_tasks(request: BatchDeleteRequest):
    """批量删除任务 — 基于 file_entity 的删除逻辑。

    核心规则：删除 task 不影响文件实体和分析结果。
    参考：openspec/changes/actor-name-confirmation/design.md §2.2
    """
    from sqlalchemy import text as sa_text

    db = get_session(engine)
    try:
        deleted = 0
        for task_id in request.task_ids:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                continue

            # ── 1. 收集帧文件目录 ──────────────────────────────────
            analyses = db.query(FileAnalysis).filter(
                FileAnalysis.task_id == task_id
            ).all()
            frames_dirs = [
                a.frames_path for a in analyses
                if a.frames_path and os.path.exists(a.frames_path)
            ]
            legacy_frames = getattr(task, "frames_path", None)
            if legacy_frames and os.path.exists(legacy_frames):
                frames_dirs.append(legacy_frames)

            # ── 2. 用原始 SQL 清理关联记录，避免 ORM cascade 冲突 ─
            # 注意：不删 task_actor_names（该表关联 entity_id 而非 task_id，
            # 且删除 task 不影响文件实体；误用 task_id 会 no such column）
            db.execute(sa_text("DELETE FROM analysis_logs WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM task_actors WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM tag_frame_mappings WHERE task_id = :tid"), {"tid": task_id})
            db.execute(sa_text("DELETE FROM tasks WHERE id = :tid"), {"tid": task_id})
            deleted += 1

            # ── 3. 删除帧文件目录 ────────────────────────────────
            for frames_dir in frames_dirs:
                shutil.rmtree(frames_dir, ignore_errors=True)

        db.commit()
        return {"status": "success", "deleted": deleted}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
    finally:
        db.close()


@app.delete("/task/{task_id}")
async def delete_task(task_id: int):
    """删除单个任务 — 基于 file_entity 的删除逻辑。

    核心规则：删除 task 不影响文件实体和分析结果。
    """
    from sqlalchemy import text as sa_text

    db = get_session(engine)
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # ── 1. 收集帧文件目录 ──────────────────────────────────
        analyses = db.query(FileAnalysis).filter(
            FileAnalysis.task_id == task_id
        ).all()
        frames_dirs = [
            a.frames_path for a in analyses
            if a.frames_path and os.path.exists(a.frames_path)
        ]
        legacy_frames = getattr(task, "frames_path", None)
        if legacy_frames and os.path.exists(legacy_frames):
            frames_dirs.append(legacy_frames)

        # ── 2. 用原始 SQL 清理关联记录，避免 ORM cascade 冲突 ─
        db.execute(sa_text("DELETE FROM task_actors WHERE task_id = :tid"), {"tid": task_id})
        db.execute(sa_text("DELETE FROM tag_frame_mappings WHERE task_id = :tid"), {"tid": task_id})
        db.execute(sa_text("DELETE FROM tasks WHERE id = :tid"), {"tid": task_id})

        # ── 3. 删除帧文件目录 ──────────────────────────────────
        for frames_dir in frames_dirs:
            shutil.rmtree(frames_dir, ignore_errors=True)

        db.commit()
        return {"status": "success", "deleted": 1}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
    finally:
        db.close()


# ── File Entity API ────────────────────────────────────────────────

@app.get("/file/{entity_id}")
async def get_file_entity(entity_id: int):
    """获取文件实体详情（包含所有路径、分析历史、人物名、关联任务）。

    参考：openspec/changes/actor-name-confirmation/design.md §2.3
    """
    db = get_session(engine)
    try:
        entity = (
            db.query(FileEntity)
            .filter(FileEntity.id == entity_id)
            .first()
        )
        if not entity:
            raise HTTPException(status_code=404, detail="File entity not found")

        # 使用已加载的 relationship 数据（query 时已自动 lazy-load）
        locations = [
            {
                "id": loc.id,
                "file_path": loc.file_path,
                "file_name": loc.file_name,
                "is_primary": bool(loc.is_primary),
                "created_at": loc.created_at.isoformat() if loc.created_at else None,
            }
            for loc in entity.locations
        ]

        analyses = [
            {
                "id": a.id,
                "task_id": a.task_id,
                "mapped_tags": a.mapped_tags,
                "nfo_path": a.nfo_path,
                "frames_path": a.frames_path,
                "status": a.status,
                "progress": a.progress,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "updated_at": a.updated_at.isoformat() if a.updated_at else None,
            }
            for a in entity.analyses
        ]

        actor_names = [
            {
                "id": an.id,
                "actor_name": an.actor_name,
                "source": an.source,
                "confidence": an.confidence,
                "verified": an.verified,
                "created_at": an.created_at.isoformat() if an.created_at else None,
            }
            for an in entity.actor_names
        ]

        tasks = [
            {
                "id": t.id,
                "status": t.status,
                "progress": t.progress,
                "analysis_details": t.analysis_details,
                "queued_at": t.queued_at.isoformat() if t.queued_at else None,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in entity.tasks
        ]

        return {
            "entity": {
                "id": entity.id,
                "file_md5": entity.file_md5,
                "file_size": entity.file_size,
                "duration": entity.duration,
                "created_at": entity.created_at.isoformat() if entity.created_at else None,
                "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
            },
            "locations": locations,
            "analyses": analyses,
            "actor_names": actor_names,
            "tasks": tasks,
        }
    finally:
        db.close()


# ── Scheduler APIs ─────────────────────────────────────────────────

@app.get("/scheduler/status")
async def get_scheduler_status():
    """获取调度器状态"""
    scheduler = get_scheduler()
    return scheduler.get_status()

@app.post("/scheduler/start")
async def start_scheduler():
    """启动调度器"""
    scheduler = get_scheduler()
    await scheduler.start()
    return {"status": "success", "message": "调度器已启动"}

@app.post("/scheduler/pause")
async def pause_scheduler():
    """暂停调度器"""
    scheduler = get_scheduler()
    await scheduler.pause()
    return {"status": "success", "message": "调度器已暂停"}

@app.post("/scheduler/resume")
async def resume_scheduler():
    """恢复调度器"""
    scheduler = get_scheduler()
    await scheduler.resume()
    return {"status": "success", "message": "调度器已恢复"}

@app.post("/scheduler/reload")
async def reload_scheduler():
    """重新加载待处理任务到调度器队列"""
    scheduler = get_scheduler()
    await scheduler._load_pending_tasks()
    return {"status": "success", "message": "已重新加载待处理任务"}

@app.post("/scheduler/restart")
async def restart_scheduler():
    """重启调度器以应用新配置"""
    # 停止当前调度器
    scheduler = get_scheduler()
    await scheduler.stop()

    # 重新创建调度器实例并替换全局单例(否则 get_scheduler() 仍返回旧实例,
    # 新配置不生效且旧实例已被 stop——历史 bug)
    from src.core.scheduler import TaskScheduler, set_scheduler
    new_scheduler = TaskScheduler()
    set_scheduler(new_scheduler)
    await new_scheduler.start()

    return {"status": "success", "message": "调度器已重启"}

# ── Single Task Control APIs ───────────────────────────────────────

class PinToggleRequest(BaseModel):
    task_ids: List[int]


@app.post("/tasks/pin/toggle")
async def pin_toggle_tasks(req: PinToggleRequest):
    """切换所选任务的置顶状态（逐任务翻转；置顶任务派发恒最前）。

    14 号方案：优先按钮 → 本接口；排序键 = pinned 优先 + 派发策略；
    队列为 PriorityQueue 不支持原地改序 → 变更后整体重排。
    """
    from src.core.scheduler import get_scheduler
    from src.db.models import get_db_engine, session_factory

    engine = get_db_engine()
    SessionLocal = session_factory(engine)
    db = SessionLocal()
    changed = []
    try:
        for tid in req.task_ids:
            task = db.query(Task).filter(Task.id == int(tid)).first()
            if task is not None:
                task.pinned = 0 if (task.pinned or 0) else 1
                changed.append({"id": task.id, "pinned": bool(task.pinned)})
        db.commit()
    finally:
        db.close()
    scheduler = get_scheduler()
    if scheduler is not None and changed:
        await scheduler.resort_queue()
    return {"status": "success", "changed": changed, "count": len(changed)}


@app.post("/task/{task_id}/pause")
async def pause_task(task_id: int):
    """暂停单个任务"""
    scheduler = get_scheduler()
    success = await scheduler.pause_task(task_id)
    if success:
        # 更新数据库状态为 PAUSED
        from src.db.models import Task, get_db_engine
        engine = get_db_engine()
        SessionLocal = session_factory(engine)
        session = SessionLocal()
        try:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = 'PAUSED'
                session.commit()
        finally:
            session.close()
        return {"status": "success", "message": f"任务 {task_id} 已暂停"}
    else:
        raise HTTPException(status_code=400, detail=f"无法暂停任务 {task_id}")

@app.post("/task/batch-pause")
async def batch_pause_tasks(request: dict):
    """批量暂停任务（跳过已完成的任务）"""
    task_ids = request.get("task_ids", [])
    scheduler = get_scheduler()
    
    # 查询任务当前状态，只暂停未完成的任务
    from src.db.models import Task
    engine = get_db_engine()
    SessionLocal = session_factory(engine)
    session = SessionLocal()
    
    # 可暂停的状态：运行中、排队中、分析中等
    pausable_statuses = {"PROCESSING", "SIFTING", "ANALYZING", "CLUSTERING", "QUEUED", "PENDING", "CHECKING", "SIFTED"}
    
    paused = 0
    try:
        tasks = session.query(Task).filter(Task.id.in_(task_ids)).all()
        for task in tasks:
            if task.status in pausable_statuses:
                success = await scheduler.pause_task(task.id)
                if success:
                    task.status = "PAUSED"
                    paused += 1
        session.commit()
    finally:
        session.close()
    return {"success": True, "paused": paused}

@app.post("/task/batch-resume")
async def batch_resume_tasks(request: dict):
    """批量恢复任务（恢复所有未完成的任务）"""
    task_ids = request.get("task_ids", [])
    scheduler = get_scheduler()
    
    # 确保调度器已启动
    if scheduler.status.value != "running":
        await scheduler.start()
    
    # 查询任务当前状态，恢复所有未完成的任务
    from src.db.models import Task
    engine = get_db_engine()
    SessionLocal = session_factory(engine)
    session = SessionLocal()
    
    resumed = 0
    try:
        tasks = session.query(Task).filter(Task.id.in_(task_ids)).all()
        for task in tasks:
            # 恢复所有未完成的任务（包括 PAUSED 和运行中的状态）
            if task.status not in ['COMPLETE', 'FAILED']:
                # 更新数据库状态为 QUEUED
                task.status = "QUEUED"
                # 真正提交任务到调度器队列(带时长以支持短任务优先)
                await scheduler.submit_task(task.id, duration=task.duration)
                resumed += 1
        session.commit()
    finally:
        session.close()
    return {"success": True, "resumed": resumed}

@app.post("/task/batch-unqueue")
async def batch_unqueue_tasks(request: dict):
    """取消排队：把 QUEUED 任务退回 PENDING（未激活），并拦截调度器内存队列。

    - ``{"task_ids": [...]}`` 只退所选；``{"all_queued": true}`` 退全库排队。
    - 只动 status == 'QUEUED' 的任务；已派发运行中的（scheduler.running_tasks）
      不动——运行中任务的中止是「暂停」的职责。
    """
    task_ids = request.get("task_ids") or []
    all_queued = bool(request.get("all_queued"))
    scheduler = get_scheduler()

    from src.db.models import Task, get_db_engine
    engine = get_db_engine()
    SessionLocal = session_factory(engine)
    session = SessionLocal()

    unqueued = 0
    try:
        query = session.query(Task)
        tasks = query.all() if all_queued else query.filter(Task.id.in_(task_ids)).all()
        for task in tasks:
            if task.status == 'QUEUED' and task.id not in scheduler.running_tasks:
                task.status = 'PENDING'
                # 内存队列拦截：worker 派发前不复查库状态，必须记入取消集
                await scheduler.cancel_task(task.id)
                unqueued += 1
        session.commit()
    finally:
        session.close()
    return {"success": True, "unqueued": unqueued}

@app.post("/task/{task_id}/resume")
async def resume_task(task_id: int):
    """恢复单个任务"""
    scheduler = get_scheduler()
    success = await scheduler.resume_task(task_id)
    if success:
        # 更新数据库状态为 SIFTING（恢复执行）
        from src.db.models import Task, get_db_engine
        engine = get_db_engine()
        SessionLocal = session_factory(engine)
        session = SessionLocal()
        try:
            task = session.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = 'SIFTING'
                session.commit()
        finally:
            session.close()
        return {"status": "success", "message": f"任务 {task_id} 已恢复"}
    else:
        raise HTTPException(status_code=400, detail=f"无法恢复任务 {task_id}")


@app.post("/task/{task_id}/force-reanalyze")
async def force_reanalyze_task(task_id: int):
    """强制重新分析单个任务"""
    db = get_session(engine)
    scheduler = get_scheduler()
    
    # 确保调度器已启动
    if scheduler.status.value != "running":
        await scheduler.start()
    
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

        # 获取 FileAnalysis 记录
        file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()

        # 清除所有分析结果
        task.status = 'QUEUED'
        task.progress = 0
        task.error_message = None

        if file_analysis:
            file_analysis.mapped_tags = None
            file_analysis.analysis_details = None
            file_analysis.nfo_path = None
            file_analysis.status = 'PENDING'
            file_analysis.progress = 0

        task.queued_at = datetime.datetime.utcnow()
        db.commit()
        
        # 提交到调度器队列(带时长以支持短任务优先)
        await scheduler.submit_task(task_id, duration=task.duration)
        
        return {
            "status": "success",
            "message": f"任务 {task_id} 已加入强制重新分析队列"
        }
    finally:
        db.close()

class SchedulerConfigRequest(BaseModel):
    max_concurrent: Optional[int] = None
    max_md5_concurrent: Optional[int] = None
    max_sifting_concurrent: Optional[int] = None
    max_vlm_concurrent: Optional[int] = None
    task_timeout: Optional[int] = None
    retry_on_failure: Optional[bool] = None
    max_retries: Optional[int] = None

@app.put("/scheduler/config")
async def update_scheduler_config(request: SchedulerConfigRequest):
    """更新调度器配置"""
    scheduler = get_scheduler()
    scheduler.update_config(
        max_concurrent=request.max_concurrent,
        max_md5_concurrent=request.max_md5_concurrent,
        max_sifting_concurrent=request.max_sifting_concurrent,
        max_vlm_concurrent=request.max_vlm_concurrent,
        task_timeout=request.task_timeout,
        retry_on_failure=request.retry_on_failure,
        max_retries=request.max_retries
    )
    return {"status": "success", "message": "调度器配置已更新"}

@app.get("/scheduler/config")
async def get_scheduler_config():
    """获取调度器配置"""
    scheduler = get_scheduler()
    return {
        "max_concurrent": scheduler.max_concurrent,
        "max_md5_concurrent": scheduler.max_md5_concurrent,
        "max_sifting_concurrent": scheduler.max_sifting_concurrent,
        "max_vlm_concurrent": scheduler.max_vlm_concurrent,
        "task_timeout": scheduler.task_timeout,
        "retry_on_failure": scheduler.retry_on_failure,
        "max_retries": scheduler.max_retries
    }

@app.get("/hardware/info")
async def get_hardware_info():
    """获取硬件信息和并发推荐配置"""
    from src.core.hardware_detector import HardwareDetector
    return HardwareDetector.get_recommendations()

def calculate_file_md5(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件MD5值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


# ── 作业状态分组 ────────────────────────────────────────────────────
# 「未结束」：同一路径已有这类作业时，导入跳过、不重复排队同一路径
NON_TERMINAL_TASK_STATUSES = (
    'PENDING', 'QUEUED', 'CHECKING', 'SIFTING',
    'ANALYZING', 'CLUSTERING', 'WRITING', 'PAUSED',
)
# 「同内容已被处理或正在处理」：重复文件据此跳过业务。
# 不含 ERROR / FAILED —— 那两类是可重试的，不该算「已被处理」。
CONTENT_TAKEN_TASK_STATUSES = ('COMPLETE', 'COMPLETED', 'DUPLICATE') + NON_TERMINAL_TASK_STATUSES

# 导入时每登记这么多个文件提交一次（进度可感知、可中断、不长时间持写锁）
IMPORT_COMMIT_BATCH = 50


def _resolve_task_file_path(task: Task) -> Optional[str]:
    """取任务对应的文件路径：优先实体主路径，其次 task.file_path。

    导入只写 tasks（不建实体），所以新建的作业多半走 task.file_path 分支。
    """
    if task.entity:
        primary = next(
            (loc for loc in task.entity.locations if loc.is_primary == 1),
            task.entity.locations[0] if task.entity.locations else None,
        )
        if primary is not None:
            return primary.file_path
    return task.file_path


def _active_analysis_theme(db) -> Optional[str]:
    """当前分析主题：优先前端「标签管理」页选的动态主题，否则用默认主题名。"""
    theme = config_manager.get_active_analysis_theme() or None
    if not theme:
        try:
            from src.db.models import Theme as _ThemeModel
            default_theme = db.query(_ThemeModel).filter(_ThemeModel.is_default == 1).first()
            if default_theme:
                theme = default_theme.name
        except Exception:
            pass
    return theme


def resolve_content_identity(db, task: Task, file_path: str):
    """为任务建立「内容身份」：算出 MD5 → 定位/创建 file_entity 与 file_location。

    这是本系统数据模型的枢纽（见
    ``openspec/changes/separate-import-from-analysis/design.md`` §3）：

    - ``file_entities`` 表示**内容**，按 ``file_md5`` 唯一 ⇒ 只能在**内容身份已知**
      （即此处算出 MD5）时创建。导入阶段只写 ``tasks``，不建实体、不写假 md5。
    - ``file_locations`` 表示**路径**，按 ``file_path`` 唯一 ⇒ 同一内容挂多条路径，
      「一个文件对应多路径」天然成立，而不必往 ``file_entities`` 里塞多路径。

    返回 ``(状态, 实体, 先前作业)``：

    - ``('ok', entity, None)`` —— 身份就绪，可继续分析
    - ``('duplicate', entity, prior_task)`` —— 同内容已被/正被处理，
      调用方应标 ``DUPLICATE`` 并**跳过后续全部业务**（路径归属已记录）
    - ``('error', None, None)`` —— 算 MD5 失败（调用方标 ERROR）
    """
    computed_md5 = task.file_md5
    if not computed_md5 and task.entity_id:
        own = db.query(FileEntity).filter(FileEntity.id == task.entity_id).first()
        computed_md5 = own.file_md5 if own else None

    try:
        if computed_md5:
            logger.info(f"任务 {task.id} 复用已存 MD5: {computed_md5}（不重算）")
        else:
            logger.info(f"任务 {task.id} 开始计算MD5，文件路径: {file_path}")
            computed_md5 = calculate_file_md5(file_path)
            logger.info(f"任务 {task.id} MD5计算完成: {computed_md5}")
    except Exception as exc:
        logger.error(f"任务 {task.id} 计算 MD5 失败: {exc}")
        return ('error', None, None)

    entity = db.query(FileEntity).filter(FileEntity.file_md5 == computed_md5).first()
    if entity is None:
        # 时长惰性获取：导入不再探时长；内容首次出现、建实体前补探一次，
        # 保证 FileEntity.duration 有值（否则 /videos 会显示 Unknown）。
        if task.duration is None and _media_type_of(file_path) == "video":
            probed_duration = _probe_video_duration(file_path)
            if probed_duration is not None:
                task.duration = probed_duration
        entity = FileEntity(
            file_md5=computed_md5,
            file_size=task.file_size,
            duration=task.duration,
            media_type=_media_type_of(file_path),
            analysis_theme=_active_analysis_theme(db),
        )
        if entity.media_type == "photo":
            meta = _extract_photo_metadata(file_path)
            entity.resolution = meta.get("resolution")
            entity.exif_data = (
                json.dumps(meta["exif_data"], ensure_ascii=False) if meta.get("exif_data") else None
            )
            entity.capture_time = meta.get("capture_time")
        db.add(entity)
        db.flush()
        logger.info(f"任务 {task.id} 新建内容实体 {entity.id}（内容首次出现）")

    # 路径 → 内容：file_path 唯一，已存在则复用（内容变了则改归属）
    location = db.query(FileLocation).filter(FileLocation.file_path == file_path).first()
    if location is None:
        has_primary = db.query(FileLocation).filter(
            FileLocation.entity_id == entity.id
        ).count() > 0
        db.add(FileLocation(
            entity_id=entity.id,
            file_path=file_path,
            file_name=sanitize_filename(Path(file_path).name),
            is_primary=0 if has_primary else 1,
        ))
    elif location.entity_id != entity.id:
        location.entity_id = entity.id

    task.entity_id = entity.id
    task.file_md5 = computed_md5
    db.commit()

    prior = db.query(Task).filter(
        Task.entity_id == entity.id,
        Task.id != task.id,
        Task.status.in_(CONTENT_TAKEN_TASK_STATUSES),
    ).first()
    if prior is not None:
        return ('duplicate', entity, prior)
    return ('ok', entity, None)


def _media_type_of(file_path: str) -> str:
    return "photo" if Path(file_path).suffix.lower() in PHOTO_EXTENSIONS else "video"


def _mark_task_error(db, task: Task, message: str, analysis_details: Optional[dict] = None) -> None:
    """标记任务失败。

    分析记录（``FileAnalysis``）依附于内容实体，而预检阶段可能**还没有实体**
    （导入只写 tasks，实体要等身份已知才创建）—— 此时只标任务，不碰分析记录。

    用户文案经错误分类目录（任务 2.2）归一为"编码+人话+建议";原始细节进日志。
    """
    task.status = 'ERROR'
    task.error_message = format_error(message)
    logger.warning(f"任务 {task.id} 失败[{message[:200]}] → {task.error_message}")
    task.completed_at = datetime.datetime.utcnow()
    if task.entity_id:
        analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
        if analysis is not None:
            analysis.status = 'ERROR'
            analysis.analysis_details = json.dumps(analysis_details or {}, ensure_ascii=False)
    db.commit()


def _get_task_file_info(task: Task) -> Dict[str, Any]:
    """Helper to get file information for a task.

    优先取 task.entity（内容身份，权威值）；没有 entity 时回退到 task 上的冗余字段
    —— 导入只登记作业（entity_id=NULL），此时路径/大小/时长只在 task 上。

    Returns dict with file_path, file_md5, file_size, duration.
    """
    if not task.entity:
        return {
            "file_path": task.file_path,
            "file_md5": task.file_md5,
            "file_size": task.file_size,
            "duration": task.duration,
        }

    # Get primary location (is_primary=1)
    primary_location = next(
        (loc for loc in task.entity.locations if loc.is_primary == 1),
        task.entity.locations[0] if task.entity.locations else None
    )

    return {
        "file_path": (primary_location.file_path if primary_location else None) or task.file_path,
        "file_md5": task.entity.file_md5 or task.file_md5,
        "file_size": task.entity.file_size if task.entity.file_size is not None else task.file_size,
        "duration": task.entity.duration if task.entity.duration is not None else task.duration,
    }


# ── Multi-process parallel preprocessing (opt-2) ────────────────────

def _detect_scenes_process(file_path: str, sample_rate: int = 1) -> list:
    """Detect scenes in a separate process."""
    from src.core.scenedetect import detect_scenes
    return detect_scenes(file_path, sample_rate=sample_rate)


def _get_frame_sampler():
    """Return the frame sampler based on the face_quality config.

    Uses the smart sampler (blur detection + retry) when
    ``face_quality.enabled`` / ``face_quality.retry_on_blur`` are on,
    otherwise falls back to the legacy equidistant sampler.
    """
    try:
        fq = ConfigManager().get("face_quality", {})
        if fq.get("enabled", False) and fq.get("retry_on_blur", False):
            logger.info("Using SmartFrameSampler (blur detection + retry)")
            return SmartFrameSampler(
                sharpness_threshold=fq.get("sharpness_threshold"),
                min_kept_frames=fq.get("min_faces_to_keep", 3),
            )
    except Exception as e:
        logger.warning("Failed to build SmartFrameSampler, using legacy: %s", e)
    return sample_frames_per_unit


def _extract_frames_process(args: tuple) -> dict:
    """Extract frames in a separate process."""
    import os
    import subprocess
    import time

    file_path, scenes, frames_per_scene, video_output_dir, max_frames, *rest = args
    frame_density = rest[0] if rest else None  # 帧/分钟（14 号密度档；缺省走 frames_per_scene）
    os.makedirs(video_output_dir, exist_ok=True)

    # Apply max_frames cap
    if max_frames > 0 and len(scenes) * frames_per_scene > max_frames:
        frames_per_scene = max(1, max_frames // len(scenes))

    # ── 成人内容优化：分镜数 > 50 时改用时间均匀采样 ──
    # 成人视频几乎无叙事性，大量分镜只是同一活动的不同角度切换
    # 直接均匀采样更高效，跳过无意义的场景检测
    adult_content_threshold = 50
    if len(scenes) > adult_content_threshold and max_frames > 0:
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0

            # 均匀分布采样：总帧数 = 时长(分) × 密度（14 号密度档真实消费）
            if frame_density:
                total_frames = min(max_frames, max(1, round(duration / 60 * frame_density)))
            else:
                total_frames = min(max_frames, len(scenes) * frames_per_scene)
            interval = duration / total_frames if duration > 0 and total_frames > 0 else 0

            frame_extract_start = time.time()
            for i in range(total_frames):
                timestamp = i * interval
                frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', file_path,
                       '-vframes', '1', '-q:v', '2', frame_path]
                subprocess.run(cmd, capture_output=True, check=False)
            frame_extract_time = round(time.time() - frame_extract_start, 2)

            return {
                "scenes": [{"start": 0, "end": 0, "type": "uniform", "duration": duration, "frames": total_frames}],
                "frames": [{"scene_index": 0, "count": total_frames}],
                "frame_extract_time": frame_extract_time,
                "frames_path": video_output_dir,
            }
        except Exception:
            pass  # Fall through to normal scene-based sampling

    # ── 成人内容优化：分镜数 > 50 时改用时间均匀采样 ──
    # 成人视频几乎无叙事性，大量分镜只是同一活动的不同角度切换
    # 直接均匀采样更高效，跳过无意义的场景检测
    adult_content_threshold = 50
    if len(scenes) > adult_content_threshold and max_frames > 0:
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0

            # 均匀分布采样：每 N 秒采 1 帧
            total_frames = min(max_frames, len(scenes) * frames_per_scene)
            interval = duration / total_frames if duration > 0 and total_frames > 0 else 0

            frame_extract_start = time.time()
            for i in range(total_frames):
                timestamp = i * interval
                frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', file_path,
                       '-vframes', '1', '-q:v', '2', frame_path]
                subprocess.run(cmd, capture_output=True, check=False)
            frame_extract_time = round(time.time() - frame_extract_start, 2)

            return {
                "scenes": [{"start": 0, "end": 0, "type": "uniform", "duration": duration, "frames": total_frames}],
                "frames": [{"scene_index": 0, "count": total_frames}],
                "frame_extract_time": frame_extract_time,
                "frames_path": video_output_dir,
            }
        except Exception:
            pass  # Fall through to normal scene-based sampling

    # ── 成人内容优化：分镜数 > 50 时改用时间均匀采样 ──
    # 成人视频几乎无叙事性，大量分镜只是同一活动的不同角度切换
    # 直接均匀采样更高效，跳过无意义的场景检测
    adult_content_threshold = 50
    if len(scenes) > adult_content_threshold and max_frames > 0:
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0

            # 均匀分布采样：每 N 秒采 1 帧
            total_frames = min(max_frames, len(scenes) * frames_per_scene)
            interval = duration / total_frames if duration > 0 and total_frames > 0 else 0

            for i in range(total_frames):
                timestamp = i * interval
                frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', file_path,
                       '-vframes', '1', '-q:v', '2', frame_path]
                subprocess.run(cmd, capture_output=True, check=False)

            frame_extract_time = round(__import__('time').time() - __import__('time').time(), 2)
            # 重新计时
            frame_extract_start = __import__('time').time()
            for i in range(total_frames):
                timestamp = i * interval
                frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', file_path,
                       '-vframes', '1', '-q:v', '2', frame_path]
                subprocess.run(cmd, capture_output=True, check=False)
            frame_extract_time = round(__import__('time').time() - frame_extract_start, 2)

            return {
                "scenes": [{"start": 0, "end": 0, "type": "uniform", "duration": duration, "frames": total_frames}],
                "frames": [{"scene_index": 0, "count": total_frames}],
                "frame_extract_time": frame_extract_time,
                "frames_path": video_output_dir,
            }
        except Exception:
            pass  # Fall through to normal scene-based sampling

    if not scenes or len(scenes) == 0:
        # Fallback: uniform sampling
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0
        except:
            duration = 0

        if duration > 0:
            num_frames = max(15, min(60, int(duration / 60 * 2)))
        else:
            num_frames = 15

        # Apply max_frames cap for fallback
        if max_frames > 0:
            num_frames = min(num_frames, max_frames)

        interval = duration / num_frames if duration > 0 else 0
        frame_extract_start = __import__('time').time()

        for i in range(num_frames):
            timestamp = i * interval
            frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
            cmd = ['ffmpeg', '-y', '-ss', str(timestamp), '-i', file_path,
                   '-vframes', '1', '-q:v', '2', frame_path]
            subprocess.run(cmd, capture_output=True, check=False)

        frame_extract_time = round(__import__('time').time() - frame_extract_start, 2)
        return {
            "scenes": [{"start": 0, "end": 0, "type": "fallback", "duration": duration, "frames": num_frames}],
            "frames": [{"scene_index": 0, "count": num_frames}],
            "frame_extract_time": frame_extract_time,
            "frames_path": video_output_dir,
        }
    else:
        # Sample frames per scene
        # 注意：scenes 格式为 {"start","end"}，而 sample_frames_per_unit 期望
        # {"unit_index","start_frame","end_frame"}——必须转换，否则 KeyError。
        # SmartFrameSampler 是实例不是函数，也要统一走 callable 包装。
        frame_extract_start = __import__('time').time()
        frame_sampler = _get_frame_sampler()

        # 转换 scenes → units 格式
        units = []
        for idx, s in enumerate(scenes):
            units.append({
                "unit_index": idx,
                "start_frame": s.get("start_frame", s.get("start", 0)),
                "end_frame": s.get("end_frame", s.get("end", 0)),
            })

        if hasattr(frame_sampler, 'sample_frames_per_unit'):
            # SmartFrameSampler 实例
            frame_results = frame_sampler.sample_frames_per_unit(
                file_path, units, n_frames=frames_per_scene,
                output_base_dir=video_output_dir
            )
        else:
            # legacy 函数
            frame_results = frame_sampler(
                file_path, units, n_frames=frames_per_scene,
                output_base_dir=video_output_dir
            )
        frame_extract_time = round(__import__('time').time() - frame_extract_start, 2)

        frames_info = []
        for scene_idx, scene_frames in frame_results.items():
            frames_info.append({"scene_index": scene_idx, "count": len(scene_frames)})

        return {
            "scenes": [{"start": s.get("start_frame", 0), "end": s.get("end_frame", 0), "type": "scene"} for s in scenes],
            "frames": frames_info,
            "frame_extract_time": frame_extract_time,
            "frames_path": video_output_dir,
        }


def _get_parallel_config(config_manager) -> dict:
    """Get parallel processing configuration."""
    import sys
    perf = config_manager.get("performance", {})
    cpu_count = multiprocessing.cpu_count()

    # Default to CPU count for preprocessing parallelism
    max_preprocess_workers = perf.get("max_preprocess_workers", max(2, cpu_count - 2))

    # Disable parallel preprocessing in PyInstaller bundled environment
    # ProcessPoolExecutor cannot spawn child processes in frozen apps
    is_frozen = getattr(sys, 'frozen', False)
    parallel_enabled = perf.get("parallel_preprocessing", True) and not is_frozen

    return {
        "max_preprocess_workers": max_preprocess_workers,
        "cpu_count": cpu_count,
        "parallel_enabled": parallel_enabled,
    }


def run_audit_worker_sync(task_id: int):
    from src.core.error_taxonomy import format_error  # 超长函数作用域显式绑定,规避静态分析歧义
    """
    同步版本的任务执行 Worker - 在线程池中运行
    
    这个函数是同步的，用于在线程池中执行 CPU 密集型任务，
    避免阻塞 asyncio 事件循环。
    """
    import os  # 在函数开头导入，确保后续使用
    import time
    from src.core.config import get_config_manager
    from src.core.nsfw_detector import TwoStageNSFWAnalyzer
    from src.core.actor_name_extractor import ActorNameExtractor
    
    db = get_session(engine)
    task = None
    # 共享实例：分析开始时读到的是"此刻生效"的配置（含刚切换的分析主题）
    config_manager = get_config_manager()
    # 用于收集分析细节
    analysis_details = {
        "md5": {},
        "scenes": [],
        "frames": [],
        "perf": {},  # 档位与预估（perf_tiers，UI 如实展示用）
        "timing": {
            "scene_detection": 0,
            "frame_extraction": 0,
            "vlm_analysis": 0,
            "tag_mapping": 0,
            "nfo_writing": 0
        },
        "vlm_raw": "",
        "mapped_tags": [],
        "unmatched": []
    }
    
    def check_pause():
        """检查任务是否被暂停，如果是则等待"""
        try:
            from src.core.scheduler import get_scheduler
            
            scheduler = get_scheduler()
            if scheduler.is_task_paused(task_id):
                # 更新任务状态为 PAUSED
                task.status = 'PAUSED'
                db.commit()
                
                # 使用简单的轮询等待，而不是 asyncio
                import time
                while scheduler.is_task_paused(task_id):
                    time.sleep(1)
                
                # 恢复后更新状态
                task.status = 'SIFTING'
                db.commit()
        except Exception as e:
            logger.error(f"检查暂停状态失败: {e}")
    
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            logger.error(f"任务 {task_id} 不存在")
            return

        # ── 步骤 0：确定文件路径（导入只写 tasks，此时可能还没有实体）──
        file_path = _resolve_task_file_path(task)
        if not file_path:
            logger.error(f"任务 {task_id} 没有关联的文件路径")
            _mark_task_error(db, task, "任务没有关联的文件路径或文件实体")
            return

        # ── 预检: 验证文件存在 ──
        if not os.path.exists(file_path):
            logger.error(f"任务 {task_id} 文件不存在: {file_path}")
            _mark_task_error(db, task, f"视频文件不存在: {file_path}", analysis_details)
            return

        # ── 预检: VLM 就绪性 ──
        # 模型/服务没就绪时**不消费任务**（状态回到 QUEUED）并让调度器暂停派发，
        # 而不是把任务判死：调度器会在几秒内把队列里每一条都派发一遍，逐条判死
        # 等于"清库存并全部误伤"（2026-09-20 实测 2196 条排队任务全变 ERROR）。
        try:
            from src.core.vlm_readiness import is_vlm_ready, defer_task_for_vlm

            vlm_ok, vlm_reason = is_vlm_ready(config_manager.get_vlm_config)
        except Exception as preflight_err:
            # 探测本身出错时不阻断：沿用旧行为（非致命），让后续阶段自己报错
            logger.warning(f"任务 {task_id} VLM 预检失败 (非致命): {preflight_err}")
            vlm_ok, vlm_reason = True, ""

        if not vlm_ok:
            logger.warning(f"任务 {task_id} 暂缓（VLM 未就绪）: {vlm_reason}")
            defer_task_for_vlm(db, task)
            try:
                from src.core.scheduler import get_scheduler

                get_scheduler().request_readiness_pause(f"VLM 未就绪: {vlm_reason}")
            except Exception as sched_exc:
                logger.warning(f"请求调度器暂停失败（不致命）: {sched_exc}")
            return

        # ── 步骤 1：建立/复用「内容身份」（算 MD5 → file_entity / file_location）──
        # 导入阶段只登记作业；内容实体在此刻（身份已知）才创建，**不写任何假值**。
        # 同内容已被/正被处理 ⇒ 判为重复，跳过后续全部业务。
        identity, entity, prior_task = resolve_content_identity(db, task, file_path)
        if identity == 'error':
            _mark_task_error(db, task, f"计算文件 MD5 失败: {file_path}", analysis_details)
            return
        if identity == 'duplicate':
            # project-assets-roster/D4：重复工单不再一律悬空——
            # 同内容已有完整分析结果 → 新工单直接继承完成态（秒回，不重跑）；
            # 既有工单未完成 → 维持 DUPLICATE 标注退场（既有工单继续处理，
            # 避免双工单并发同内容），任务页按重复标注展示。
            prior_analysis = db.query(FileAnalysis).filter(
                FileAnalysis.entity_id == entity.id).first()
            if prior_analysis is not None and prior_analysis.status in ('READY', 'CONFIRMED'):
                task.status = 'COMPLETE'
                task.progress = 100
                task.error_message = None
                task.completed_at = datetime.datetime.utcnow()
                db.commit()
                logger.info(
                    f"任务 {task_id} 内容与任务 {prior_task.id} 重复（实体 {entity.id}），"
                    "复用既有分析结果，直接置完成态"
                )
                return
            logger.info(
                f"任务 {task_id} 内容与任务 {prior_task.id} 重复（实体 {entity.id}），跳过后续业务"
            )
            task.status = 'DUPLICATE'
            task.progress = 0
            task.error_message = f"重复文件，同内容已由任务 {prior_task.id} 处理"
            task.completed_at = datetime.datetime.utcnow()
            db.commit()
            return

        # 获取或创建 FileAnalysis 记录 (业务数据持久化核心)
        file_analysis = db.query(FileAnalysis).filter(FileAnalysis.entity_id == task.entity_id).first()
        if not file_analysis:
            file_analysis = FileAnalysis(
                entity_id=task.entity_id,
                task_id=task.id,
                status='PENDING',
                progress=0
            )
            db.add(file_analysis)
            db.flush()
        else:
            # 更新关联的最新任务 ID
            file_analysis.task_id = task.id

        # 内容身份已就绪 ⇒ 路径/大小/时长均从实体取得
        file_info = _get_task_file_info(task)
        file_path = file_info["file_path"] or file_path
        file_md5 = file_info["file_md5"]

        # 更新开始时间
        task.started_at = datetime.datetime.utcnow()
        
        # ── 断点恢复检查 ──
        # 检查是否已有进度，从断点恢复
        saved_progress = file_analysis.progress or 0
        saved_frames_path = file_analysis.frames_path
        saved_analysis_details = file_analysis.analysis_details
        
        if saved_progress > 0 and saved_frames_path:
            logger.info(f"任务 {task_id} 从断点恢复: 进度={saved_progress}%, frames_path={saved_frames_path}")
            # 恢复已有的分析细节
            if saved_analysis_details:
                try:
                    analysis_details = json.loads(saved_analysis_details)
                except:
                    analysis_details = {
                        "md5": {},
                        "scenes": [],
                        "frames": [],
                        "timing": {
                            "scene_detection": 0,
                            "frame_extraction": 0,
                            "vlm_analysis": 0,
                            "tag_mapping": 0,
                            "nfo_writing": 0
                        },
                        "vlm_raw": "",
                        "mapped_tags": [],
                        "unmatched": []
                    }
            frames_path = saved_frames_path
            
            # 根据进度跳转到对应阶段
            if saved_progress >= 70:
                # 已完成 VLM 分析，从聚类阶段开始
                task.status = 'CLUSTERING'
                task.progress = saved_progress
                db.commit()
                goto_vlm_analysis = False
                goto_clustering = True
                logger.info(f"任务 {task_id} 从聚类阶段恢复")
            elif saved_progress >= 30:
                # 已完成抽帧，从 VLM 分析阶段开始
                task.status = 'ANALYZING'
                task.progress = saved_progress
                db.commit()
                goto_vlm_analysis = True
                goto_clustering = False
                logger.info(f"任务 {task_id} 从 VLM 分析阶段恢复")
            else:
                # 进度不足 30%，从头开始
                saved_progress = 0
                goto_vlm_analysis = False
                goto_clustering = False
        else:
            # 没有保存的进度，从头开始
            goto_vlm_analysis = False
            goto_clustering = False

        # 阶段0: MD5检查和去重 (0-5%)
        if saved_progress == 0:
            task.status = 'CHECKING'
            task.progress = 0
            db.commit()

            # ── 文件解码健康体检（add-public-sale-readiness 任务 2.1）──
            # 容器损坏/全坏在进重阶段前拦截（201 坏文件教训:ffprobe 查表
            # 查不出尾部截断,必须实解帧数）;部分损坏记旗放行。
            if Path(file_path).suffix.lower() in {".mp4", ".mov", ".mkv", ".avi",
                                                  ".webm", ".wmv", ".flv", ".ts",
                                                  ".m4v", ".mpg", ".mpeg"}:
                try:
                    from src.core.file_health import HealthChecker
                    _hr = HealthChecker().check(file_path)
                    analysis_details["health"] = _hr
                except Exception as _he:  # noqa: BLE001 - 体检失败不阻塞分析
                    logger.warning(f"任务 {task_id} 文件体检异常(放行): {_he}")
                    _hr = {"verdict": "ok", "detail": f"体检异常:{_he}"}
                if _hr["verdict"] in ("容器损坏", "全坏", "文件缺失"):
                    task.status = 'ERROR'
                    task.progress = 0
                    task.error_message = f"文件损坏-{_hr['verdict']}（体检实解判定,建议重新获取或修复文件）"
                    task.completed_at = datetime.datetime.utcnow()
                    file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                    db.commit()
                    logger.warning(f"任务 {task_id} 文件体检拦截: {_hr['verdict']} {file_path}")
                    return
                if _hr["verdict"] == "部分损坏":
                    logger.warning(f"任务 {task_id} 文件部分损坏,继续分析: {_hr['detail'][:80]}")

        # ── opt-2: 多进程并行预处理 ──
        # 使用 ProcessPoolExecutor 并行计算 MD5 + 场景检测 + 抽帧
        parallel_config = _get_parallel_config(config_manager)
        parallel_enabled = parallel_config["parallel_enabled"]

        # 获取 MD5 信号量（限制磁盘 I/O 密集阶段的并发）
        from src.core.scheduler import get_scheduler
        scheduler = get_scheduler()
        md5_semaphore = scheduler.md5_semaphore
        logger.info(f"任务 {task_id} 等待 MD5 信号量 (可用: {md5_semaphore._value})")
        md5_semaphore.acquire()
        logger.info(f"任务 {task_id} 获得 MD5 信号量，开始计算")

        try:
            if parallel_enabled:
                # 只做场景检测 + 抽帧（MD5 已在「步骤 1」算出，此处不再重算）
                logger.info(f"任务 {task_id} 开始并行预处理，文件路径: {file_path}")
                preprocess_start = time.time()
                video_path = Path(file_path)
                video_output_dir = str(video_path.parent / f"{video_path.stem}_frames")
                frames_per_scene = config_manager.get_video_config().get('frames_per_scene', 5)
                max_frames = int(config_manager.get_video_config().get('max_frames_per_video') or 0)
                # 抽帧密度（14 号定稿：帧/分钟三档；缺省 4=标准）
                frame_density = int(config_manager.get_video_config().get('frame_density', 4) or 4)

                # 档位（perf_tiers，0.3）：并发档在 embedded_inference,此处管密度与采样步长
                from src.core.perf_tiers import active as tier_active, apply_density, override_source
                _tier_info = tier_active(config_manager)
                frames_per_scene, max_frames = apply_density(
                    frames_per_scene, max_frames, _tier_info["tier"])
                # 成人内容优化分支按 frame_density 直算,需同步缩放
                frame_density = max(1, round(frame_density * _tier_info["density_scale"]))
                analysis_details["perf"] = {
                    "tier": _tier_info["tier"],
                    "override": override_source(config_manager),
                    "sample_rate": _tier_info["sample_rate"],
                }

                with ProcessPoolExecutor(max_workers=parallel_config["max_preprocess_workers"]) as executor:
                    scene_future = executor.submit(_detect_scenes_process, file_path,
                                                   _tier_info["sample_rate"])
                    scenes = scene_future.result(timeout=600)
                    scene_time = round(time.time() - preprocess_start, 2)

                    # Submit frame extraction after scene detection completes
                    frame_future = executor.submit(_extract_frames_process, (file_path, scenes, frames_per_scene, video_output_dir, max_frames, frame_density))
                    frame_result = frame_future.result(timeout=1800)

                    # Apply results
                    analysis_details["md5"] = {"hash": file_md5, "time": 0.0}
                    analysis_details["scenes"] = frame_result["scenes"]
                    analysis_details["frames"] = frame_result["frames"]
                    analysis_details["timing"]["scene_detection"] = scene_time
                    analysis_details["timing"]["frame_extraction"] = frame_result["frame_extract_time"]
                    frames_path = frame_result["frames_path"]

                    logger.info(
                        f"任务 {task_id} 并行预处理完成: 场景={len(scenes)}个, "
                        f"抽帧={frame_result['frame_extract_time']}s, "
                        f"总耗时={round(time.time() - preprocess_start, 2)}s"
                    )

                task.status = 'SIFTING'
                task.progress = 30
                file_analysis.frames_path = frames_path
                file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                file_analysis.status = 'SIFTING'
                file_analysis.progress = 30
                db.commit()

                # Skip the original sequential scene-detection + frame-extraction sections
                goto_vlm_analysis = True
            else:
                # 原有串行流程（向后兼容）：MD5 已在「步骤 1」得到，这里只记录细节
                goto_vlm_analysis = False
                analysis_details["md5"] = {"hash": file_md5, "time": 0.0}
                db.commit()
        finally:
            md5_semaphore.release()
            logger.info(f"任务 {task_id} 释放 MD5 信号量")

        task.progress = 5
        db.commit()
        
        # 检查暂停
        check_pause()

        # ── 照片分支（Phase 2：照片直接分析，跳过场景检测/抽帧） ──
        # 照片 = 单帧：人脸识别复用 recognize_actors(帧目录)，VLM 直接分析照片
        _media_type = None
        _analysis_theme = None
        try:
            if task.entity_id:
                _ent = db.query(FileEntity).filter(FileEntity.id == task.entity_id).first()
                if _ent:
                    _media_type = _ent.media_type
                    _analysis_theme = _ent.analysis_theme
        except Exception:
            _media_type = None

        # 主题缺省时跟随前端「标签管理」页选择的当前分析主题(动态)
        if not _analysis_theme:
            try:
                _analysis_theme = config_manager.get_active_analysis_theme() or None
            except Exception:
                pass

        if _media_type == "photo":
            logger.info(f"任务 {task_id}: 照片分析模式（跳过场景检测/抽帧）")
            try:
                import shutil as _shutil
                import tempfile as _tempfile
                # 1. 把照片放入临时"帧目录"（单张；人脸图是中间产物，用完清理）
                #    HEIC/HEIF 先转 JPEG（人脸识别/OpenCV 依赖真实 JPEG 像素）
                _photo_frames = Path(_tempfile.mkdtemp(prefix=f"photo_frames_task{task_id}_"))
                _photo_dst = _photo_frames / "frame_000.jpg"
                _suffix_l = Path(file_path).suffix.lower()
                if _suffix_l in (".heic", ".heif"):
                    try:
                        from PIL import Image as _PILImage
                        try:
                            import pillow_heif
                            pillow_heif.register_heif_opener()
                        except ImportError:
                            pass
                        with _PILImage.open(file_path) as _pil:
                            if _pil.mode != "RGB":
                                _pil = _pil.convert("RGB")
                            _pil.save(str(_photo_dst), format="JPEG", quality=92)
                    except Exception as _conv_e:
                        logger.warning(f"任务 {task_id}: HEIC 转 JPEG 失败，直接拷贝: {_conv_e}")
                        _shutil.copy2(file_path, str(_photo_dst))
                else:
                    _shutil.copy2(file_path, str(_photo_dst))
                analysis_details["scenes"] = [{"start": 0, "end": 0, "type": "photo", "frames": 1}]
                analysis_details["frames"] = [{"scene_index": 0, "count": 1}]
                frames_path = str(_photo_frames)

                # 2. 人脸识别（复用 recognize_actors，单帧；人脸图片保存到人物目录持久化）
                task.status = 'PROCESSING'
                task.progress = 40
                db.commit()
                from src.core.face_recognition import FaceRecognizer
                _recognizer = FaceRecognizer()
                _actor_ids = _recognizer.recognize_actors(frames_path, task_id, file_md5)
                _actor_ids = [a for a in _actor_ids if a is not None]
                logger.info(f"任务 {task_id}: 照片人脸识别 {len(_actor_ids)} 个人物")
                if _actor_ids:
                    task.actor_id = _actor_ids[0]
                    for _aid in _actor_ids:
                        from src.db.models import TaskActor as _TA
                        if not db.query(_TA).filter(_TA.task_id == task.id, _TA.actor_id == _aid).first():
                            db.add(_TA(task_id=task.id, actor_id=_aid))
                    db.commit()

                # 3. VLM 分析照片（直接分析照片文件）
                task.status = 'ANALYZING'
                task.progress = 70
                db.commit()
                # 本函数后段（人名提取）有局部 `from src.core.vlm import create_vlm_client`，
                # 会把该名字变成函数级局部变量 → 此处若直接引用必 UnboundLocalError。
                # 故用别名局部导入，提前绑定，避免照片 VLM 分析静默失败。
                from src.core.vlm import create_vlm_client as _create_vlm_client
                _vlm_client = _create_vlm_client(config=config_manager)
                _img_b64 = ""
                try:
                    import base64 as _b64
                    import io as _io
                    suffix = Path(file_path).suffix.lower()
                    if suffix in (".heic", ".heif"):
                        # HEIC 转 JPEG 再发 VLM（多数 VLM 不支持 HEIC）
                        from PIL import Image as _PILImage
                        try:
                            import pillow_heif
                            pillow_heif.register_heif_opener()
                        except ImportError:
                            pass
                        with _PILImage.open(file_path) as _pil:
                            if _pil.mode != "RGB":
                                _pil = _pil.convert("RGB")
                            _buf = _io.BytesIO()
                            _pil.save(_buf, format="JPEG", quality=92)
                            _img_b64 = _b64.b64encode(_buf.getvalue()).decode()
                    else:
                        with open(file_path, 'rb') as _f:
                            _img_b64 = _b64.b64encode(_f.read()).decode()
                except Exception:
                    pass
                if _img_b64:
                    # 主题↔提示词联动：文件主题 → 当前分析主题 → 默认主题(动态解析)
                    # 闸 1: 主题提示词只含示例词,追加完整词典闭集约束
                    from src.core.vlm import OllamaVLMClient as _OVC
                    _prompt = _OVC.append_vocabulary_constraint(
                        config_manager.get_analysis_prompt(_analysis_theme) or "分析这张图片，输出标签")
                    _vlm_raw = _vlm_client._send_request(_prompt, [_img_b64]).get("content", "")
                    analysis_details["vlm_raw"] = _vlm_raw
                    analysis_details["timing"]["vlm_analysis"] = 1
                    logger.info(f"任务 {task_id}: 照片 VLM 分析完成，长度 {len(_vlm_raw)}")

                    # 3.4 标签映射（与视频分支一致：VLM 原始文本 → 标准标签字典）
                    _tag_mapping_start = time.time()
                    try:
                        _dict_mgr = DictionaryManager(MD_MAPPING_PATH, db)
                        _mapped_tags, _unmatched = _dict_mgr.map_raw_text_to_tags(_vlm_raw)
                        analysis_details["timing"]["tag_mapping"] = round(time.time() - _tag_mapping_start, 2)
                        analysis_details["mapped_tags"] = _mapped_tags
                        analysis_details["unmatched"] = _unmatched
                        # 未匹配标签写入审核队列
                        for _raw_label in _unmatched:
                            _dict_mgr._add_to_audit_queue(_raw_label)
                        file_analysis.mapped_tags = json.dumps(_mapped_tags, ensure_ascii=False)
                        logger.info(f"任务 {task_id}: 照片标签映射完成，匹配 {len(_mapped_tags)} 个，未匹配 {len(_unmatched)} 个")
                    except Exception as _map_e:
                        logger.warning(f"任务 {task_id}: 照片标签映射失败（非阻塞）: {_map_e}")
                        analysis_details["timing"]["tag_mapping"] = 0
                        analysis_details["mapped_tags"] = []
                        analysis_details["unmatched"] = []
                        file_analysis.mapped_tags = json.dumps([])

                # 3.5 写入 XMP 人脸标签（mwg-rs 标准，兼容 Lightroom/Synology）
                # 人脸裁剪图是中间产物不持久化；照片用 XMP 记录人脸区域 + 人物名
                try:
                    from src.core.photo_xmp import write_photo_face_regions
                    _face_regions = []
                    if _actor_ids:
                        from src.db.models import Actor as _ActorModel
                        _actor_name_map = {}
                        for _aid in _actor_ids:
                            _act = db.query(_ActorModel).filter(_ActorModel.id == _aid).first()
                            if _act:
                                _actor_name_map[_aid] = _act.name or _act.code or ""
                        # 查询本次照片识别的人脸 bbox（按 task 关联）
                        _face_rows = db.query(ActorFace).filter(
                            ActorFace.task_id == task.id,
                            ActorFace.bbox.isnot(None),
                        ).all()
                        import json as _json
                        for _fr in _face_rows:
                            try:
                                _box = _json.loads(_fr.bbox)
                                if len(_box) == 4:
                                    _face_regions.append({
                                        "box": _box,
                                        "actor_id": _fr.actor_id,
                                    })
                            except Exception:
                                continue
                    if _face_regions:
                        _xmp_ok = write_photo_face_regions(file_path, _face_regions, _actor_name_map)
                        logger.info(f"任务 {task_id}: 照片 XMP 人脸标签写入 {'成功' if _xmp_ok else '失败'} ({len(_face_regions)} 个区域)")
                except Exception as _xmp_e:
                    logger.warning(f"任务 {task_id}: XMP 写入失败（非阻塞）: {_xmp_e}")

                # 4. 完成
                task.status = 'COMPLETE'
                task.progress = 100
                task.completed_at = datetime.datetime.utcnow()
                file_analysis.status = 'COMPLETE'
                file_analysis.progress = 100
                file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                db.commit()
                logger.info(f"任务 {task_id}: 照片分析完成")
                try:
                    _shutil.rmtree(str(_photo_frames), ignore_errors=True)
                except Exception:
                    pass
                return
            except Exception as _photo_e:
                logger.error(f"任务 {task_id}: 照片分析失败: {_photo_e}")
                task.status = 'ERROR'
                logger.warning(f"照片分析异常(原始细节): {_photo_e}")
                task.error_message = format_error(f"照片分析失败: {str(_photo_e)}")
                task.completed_at = datetime.datetime.utcnow()
                file_analysis.status = 'ERROR'
                db.commit()
                return

        # 阶段1: 场景检测抽帧 (5-30%)
        task.status = 'SIFTING'
        task.progress = 5
        db.commit()

        # 检查暂停
        check_pause()

        # ── 阶段0.5: 视频质量分析（face-quality-filter） ──
        # 在抽帧前分析视频整体质量，为后续人脸清晰度筛选提供参考
        video_quality_info = None
        try:
            fqc = config_manager.get("face_quality", {})
            if fqc.get("enabled", False):
                from src.core.video_quality_analyzer import VideoQualityAnalyzer
                qa = VideoQualityAnalyzer()
                video_quality_info = qa.analyze(file_path)
                analysis_details["video_quality"] = {
                    "resolution": list(video_quality_info.resolution),
                    "fps": video_quality_info.fps,
                    "avg_sharpness": round(video_quality_info.avg_sharpness, 2),
                    "threshold": round(video_quality_info.threshold, 2),
                    "is_low_quality": video_quality_info.is_low_quality,
                }
                logger.info(
                    f"视频质量分析完成: {video_quality_info.resolution[0]}x{video_quality_info.resolution[1]}, "
                    f"清晰度={video_quality_info.avg_sharpness:.1f}, 阈值={video_quality_info.threshold:.1f}"
                )
        except Exception as e:
            logger.warning(f"视频质量分析失败（非阻塞）: {e}")
            video_quality_info = None

        # 场景检测/抽帧的信号量：必须在 if 之外先初始化——续传路径不会获取它，
        # 若在 if 内初始化，finally 里读该变量会 UnboundLocalError（历史 bug）。
        #
        # 续传（帧已抽好）应整段跳过"场景检测/抽帧"：
        #   - goto_vlm_analysis（saved_progress>=30）：从 VLM 阶段继续
        #   - goto_clustering（saved_progress>=70）：从聚类阶段继续
        # 否则会重跑抽帧、并把进度错误地压回 30。
        _resume_skip_sifting = goto_vlm_analysis or goto_clustering
        sifting_semaphore = None
        if not _resume_skip_sifting:
            # ── 原有串行流程（opt-2 未启用时的兼容路径） ──

            # 获取场景检测信号量（限制 CPU 密集阶段的并发）
            sifting_semaphore = scheduler.sifting_semaphore
            logger.info(f"任务 {task_id} 等待场景检测信号量 (可用: {sifting_semaphore._value})")
            sifting_semaphore.acquire()
            logger.info(f"任务 {task_id} 获得场景检测信号量，开始分析")
        
        try:
            if _resume_skip_sifting:
                # 断点续传：帧已抽好，跳过场景检测/抽帧，直接用已保存的
                # frames_path 继续到后续阶段（不再重跑、不动信号量/进度）。
                logger.info(f"任务 {task_id} 断点续传：跳过场景检测/抽帧，直接进入后续阶段")
                raise _SkipSifting()
            # 使用场景检测
            from src.core.perf_tiers import active as tier_active, override_source
            _tier_info = tier_active(config_manager)
            analysis_details["perf"] = {
                "tier": _tier_info["tier"],
                "override": override_source(config_manager),
                "sample_rate": _tier_info["sample_rate"],
            }
            scene_detect_start = time.time()
            scenes = detect_scenes(file_path, sample_rate=_tier_info["sample_rate"])
            scene_detect_time = round(time.time() - scene_detect_start, 2)
            analysis_details["timing"]["scene_detection"] = scene_detect_time

            if not scenes or len(scenes) == 0:
                # 场景检测失败或覆盖率不足，回退到均匀抽帧
                # 根据视频时长动态调整抽帧数量
                video_path = Path(file_path)
                video_dir = video_path.parent
                video_name = video_path.stem
                video_output_dir = video_dir / f"{video_name}_frames"
                os.makedirs(video_output_dir, exist_ok=True)

                # 获取视频时长
                try:
                    probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                    duration = float(probe_result.stdout.strip()) if probe_result.stdout.strip() else 0
                except:
                    duration = 0

                # 根据时长计算抽帧数量：每分钟至少2帧，最少15帧，最多60帧
                if duration > 0:
                    num_frames = max(15, min(60, int(duration / 60 * 2)))
                else:
                    num_frames = 15

                analysis_details["scenes"] = [{"start": 0, "end": 0, "type": "fallback", "duration": duration, "frames": num_frames}]

                # 使用 ffmpeg 均匀抽帧
                frame_extract_start = time.time()
                interval = duration / num_frames if duration > 0 else 0

                for i in range(num_frames):
                    timestamp = i * interval
                    frame_path = video_output_dir / f"frame_{i:03d}.jpg"
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(timestamp),
                        '-i', file_path,
                        '-vframes', '1',
                        '-q:v', '2',
                        str(frame_path)
                    ]
                    subprocess.run(cmd, capture_output=True, check=False)

                frame_extract_time = round(time.time() - frame_extract_start, 2)
                analysis_details["timing"]["frame_extraction"] = frame_extract_time
                analysis_details["frames"] = [{"scene_index": 0, "count": num_frames}]
                frames_path = str(video_output_dir)

                logger.info(f"Fallback 均匀抽帧: {num_frames} 帧, 视频时长 {duration:.1f}s, 间隔 {interval:.1f}s")
            else:
                # 记录场景信息
                for scene in scenes:
                    analysis_details["scenes"].append({
                        "start": scene.get("start_frame", 0),
                        "end": scene.get("end_frame", 0),
                        "type": "scene"
                    })

                # 按场景抽帧
                video_config = config_manager.get_video_config()
                frames_per_scene = video_config.get('frames_per_scene', 5)

                # ── 最大帧数限制（控制长视频分析时间） ──
                max_frames = int(video_config.get('max_frames_per_video') or 0)
                # 档位密度缩放（perf_tiers，0.3）：先缩放再套上限
                from src.core.perf_tiers import apply_density
                frames_per_scene, max_frames = apply_density(
                    frames_per_scene, max_frames, _tier_info["tier"])
                if max_frames > 0 and len(scenes) * frames_per_scene > max_frames:
                    # 重新计算每场景帧数，确保总数不超过上限
                    frames_per_scene = max(1, max_frames // len(scenes))
                    logger.info(
                        f"长视频限制: {len(scenes)} 场景 × {frames_per_scene} 帧 = "
                        f"{len(scenes) * frames_per_scene} 帧（上限 {max_frames}）"
                    )

                # 默认放在原文件所在的文件夹
                video_path = Path(file_path)
                video_dir = video_path.parent
                video_name = video_path.stem
                video_output_dir = video_dir / f"{video_name}_frames"
                os.makedirs(video_output_dir, exist_ok=True)

                # 按场景抽帧
                frame_extract_start = time.time()
                frame_sampler = _get_frame_sampler()

                # 转换 scenes → units 格式（与 _extract_frames_process 一致）
                units = []
                for _idx, _s in enumerate(scenes):
                    units.append({
                        "unit_index": _idx,
                        "start_frame": _s.get("start_frame", _s.get("start", 0)),
                        "end_frame": _s.get("end_frame", _s.get("end", 0)),
                    })

                if hasattr(frame_sampler, 'sample_frames_per_unit'):
                    frame_results = frame_sampler.sample_frames_per_unit(
                        file_path, units, n_frames=frames_per_scene,
                        output_base_dir=str(video_output_dir)
                    )
                else:
                    frame_results = frame_sampler(
                        file_path, units, n_frames=frames_per_scene,
                        output_base_dir=str(video_output_dir)
                    )
                frame_extract_time = round(time.time() - frame_extract_start, 2)
                analysis_details["timing"]["frame_extraction"] = frame_extract_time

                # 记录抽帧信息
                for scene_idx, scene_frames in frame_results.items():
                    analysis_details["frames"].append({
                        "scene_index": scene_idx,
                        "count": len(scene_frames)
                    })

                frames_path = str(video_output_dir)
        except _SkipSifting:
            pass  # 续传路径：不做场景检测/抽帧
        finally:
            # 只有真正走过"场景检测/抽帧"阶段才释放信号量并回写进度=30；
            # 续传路径不碰（否则会把 >=30/>=70 的进度错误地压回 30）。
            if sifting_semaphore is not None:
                sifting_semaphore.release()
                logger.info(f"任务 {task_id} 释放场景检测信号量")

                task.progress = 30
                file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                db.commit()
        # End of if not goto_vlm_analysis

        # Face recognition (between sifting and VLM analysis) — multi-actor
        try:
            from src.core.face_recognition import FaceRecognizer
            recognizer = FaceRecognizer()

            # 优先使用两阶段视频人脸检测，失败则回退到帧目录方式
            actor_ids = None
            if file_path and Path(file_path).exists():
                try:
                    logger.info(f"任务 {task_id}: 使用两阶段视频人脸检测: {file_path}")
                    actor_ids = recognizer.recognize_actors_from_video(file_path, task_id, file_info["file_md5"])
                    actor_ids = [a for a in actor_ids if a is not None]
                    logger.info(f"任务 {task_id}: 两阶段检测识别到 {len(actor_ids)} 个人物")
                except Exception as video_e:
                    logger.warning(f"任务 {task_id}: 两阶段视频检测失败，回退到帧目录方式: {video_e}")
                    actor_ids = None

            if actor_ids is None:
                logger.info(f"任务 {task_id}: 使用帧目录方式识别人脸: {frames_path}")
                actor_ids = recognizer.recognize_actors(frames_path, task_id, file_info["file_md5"])
                actor_ids = [a for a in actor_ids if a is not None]

            if actor_ids:
                # 设置 actor_id 为第一个人物（向后兼容）
                task.actor_id = actor_ids[0]
                # 创建 TaskActor 关联记录
                for aid in actor_ids:
                    existing_ta = db.query(TaskActor).filter(
                        TaskActor.task_id == task.id,
                        TaskActor.actor_id == aid
                    ).first()
                    if not existing_ta:
                        db.add(TaskActor(task_id=task.id, actor_id=aid))

                # ── face-quality-filter: 计算人物平均人脸清晰度 ──
                for aid in actor_ids:
                    faces = db.query(ActorFace).filter(ActorFace.actor_id == aid).all()
                    if faces:
                        sharpness_values = [f.sharpness for f in faces if f.sharpness and f.sharpness > 0]
                        if sharpness_values:
                            avg_sharpness = sum(sharpness_values) / len(sharpness_values)
                            actor = db.query(Actor).filter(Actor.id == aid).first()
                            if actor:
                                actor.avg_face_sharpness = round(avg_sharpness, 2)
                                logger.info(
                                    f"人物 {aid} 平均人脸清晰度: {avg_sharpness:.1f} "
                                    f"(基于 {len(sharpness_values)} 张人脸)"
                                )

                db.commit()
        except Exception as fr_exc:
            logger.warning("Face recognition failed (non-blocking): %s", fr_exc)

        # 阶段2: 两阶段NSFW分析 (30-70%)
        task.status = 'ANALYZING'
        db.commit()
        
        # 获取 VLM 并发信号量（限制 GPU 密集型阶段的并发数）
        from src.core.scheduler import get_scheduler
        scheduler = get_scheduler()
        vlm_semaphore = scheduler.vlm_semaphore
        logger.info(f"任务 {task_id} 等待 VLM 信号量 (可用: {vlm_semaphore._value})")
        vlm_semaphore.acquire()
        logger.info(f"任务 {task_id} 获得 VLM 信号量，开始分析")
        
        try:
            # 使用两阶段检测器：先用 NSFW 预筛筛选，再用 VLM 分析
            from src.core.nsfw_detector import TwoStageNSFWAnalyzer
            
            vlm_start = time.time()
            
            def progress_callback(current, total, message):
                """更新任务进度
                
                注意：nsfw_detector.py 传递的 current 已经是百分比值（50-90），
                需要映射到 30-70% 的范围
                """
                # current 是 50-90 的百分比值，映射到 30-70%
                # 50 -> 30, 90 -> 70
                if current >= 100:
                    # 分析完成，设置为 70%
                    progress = 70
                else:
                    # 将 50-90 映射到 30-70
                    progress = 30 + int((current - 50) / 40 * 40)
                    progress = max(30, min(progress, 70))  # 限制在 30-70% 范围
                
                task.progress = progress
                db.commit()
                logger.debug(f"任务 {task_id} 进度更新: {progress}% (current={current}, total={total})")
            
            # 同 6115：函数后段的局部 import 使该名成为函数级局部变量，提前用别名绑定
            from src.core.vlm import create_vlm_client as _create_vlm_client
            vlm = _create_vlm_client(config=config_manager)
            analyzer = TwoStageNSFWAnalyzer(
                nsfw_threshold=0.5,
                vlm_client=vlm,
                config_manager=config_manager
            )
            
            # 获取帧文件列表
            frame_files = []
            for root, dirs, files in os.walk(frames_path):
                for file in sorted(files):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frame_files.append(os.path.join(root, file))
            
            if not frame_files:
                task.status = 'ERROR'
                task.progress = 30
                task.error_message = format_error("未找到帧文件")
                task.completed_at = datetime.datetime.utcnow()
                file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                db.commit()
                return
            
            # 档位与预估（perf_tiers，0.3）：resume 路径此处首次可达,防御式取值
            _tier = (analysis_details.get("perf") or {}).get("tier") or "HIGH"
            analysis_details.setdefault("perf", {})
            analysis_details["perf"]["frames"] = len(frame_files)
            from src.core.perf_tiers import estimated_minutes
            analysis_details["perf"]["eta_minutes"] = estimated_minutes(len(frame_files), _tier)

            # 主题提示词(两条路径共用; 主分析与场景分析保持一致)
            # 闸 1: 主题/配置提示词只含示例词,程序化追加完整词典闭集约束
            from src.core.vlm import OllamaVLMClient as _OVC
            _theme_prompt = _OVC.append_vocabulary_constraint(
                config_manager.get_analysis_prompt(_analysis_theme) or "")

            # 选帧/映射参数(两条路径共用)
            _theme_for_filter = _analysis_theme or config_manager.get_active_analysis_theme()
            _map_nsfw_filter = bool(config_manager.get(
                "analysis.mapping_nsfw_filter", _theme_for_filter == "成人内容"))
            _map_frames_cap = int(config_manager.get("analysis.mapping_frames_per_unit")
                                  or config_manager.get("video.frames_per_scene", 5) or 5)
            _map_min_nsfw = int(config_manager.get("analysis.mapping_nsfw_min_frames", 1) or 1)
            # 合并开关: 默认把"主分析 + 逐帧映射"两遍合并为每场景一次调用
            _merge_passes = bool(config_manager.get("analysis.merge_vlm_passes", True))

            # 标签质量门槛参数(闸2/3/4, 2026-09-30 批3, 依据 docs/数据质量/审计-2026-09-30)
            from src.core.dictionary import (enforce_vocabulary as _enforce_vocab,
                                             apply_frequency_gate as _freq_gate,
                                             filter_contradictions as _contradictions)
            _strict_vocab = bool(config_manager.get("analysis.strict_vocabulary", True))
            _min_support = int(config_manager.get("analysis.min_tag_scene_support", 2) or 2)
            _support_ratio = float(config_manager.get("analysis.tag_scene_support_ratio", 0.25) or 0.25)
            _max_tags_file = int(config_manager.get("analysis.max_tags_per_file", 30) or 30)
            _gate_kept = None  # 闸4 产物,mapped_tags 定稿时复用
            _std_tags: set = set()
            if _strict_vocab:
                _std_tags = {r[0] for r in db.query(DictionaryEntry.standard_tag).all()}
            if bool(config_manager.get("analysis.frame_quality_filter", True)):
                from src.core import frame_quality as _fq
                _fq.set_thresholds(
                    float(config_manager.get("analysis.frame_quality_black_mean", 16)),
                    float(config_manager.get("analysis.frame_quality_min_sharpness", 12)))
            else:
                _fq = None

            from src.db.models import TagFrameMapping
            # 批量插入: 不再逐标签 SELECT 查重 + db.add(逐条小写→锁竞争+写放大)。
            # 攒成 rows 后按场景用 INSERT ... ON CONFLICT DO NOTHING 一次写入,
            # 唯一约束(uq_task_tag_frame)负责去重, 语义与原查重一致。
            from sqlalchemy.dialects.sqlite import insert as _sqlite_insert

            _nsfw_det = None
            if _map_nsfw_filter:
                try:
                    from src.core.nsfw_detector import (
                        create_nsfw_detector, select_mapping_frames, select_scene_frames,
                    )
                    _nsfw_det = create_nsfw_detector(threshold=0.5)
                except Exception as _nn_exc:
                    logger.warning(f"任务 {task_id}: NSFW 检测器初始化失败, 退化为全帧分析: {_nn_exc}")
                    _nsfw_det = None
            _fallback_units = 0
            _nsfw_count = 0
            _total_count = len(frame_files)

            if _merge_passes:
                # ── 合并模式: 每个场景一次 VLM 调用 ──
                # NSFW 预筛选帧 → 一次调用出该场景标签 → 标签归到封面帧(NSFW 最高分)。
                # 替代旧的"全片主分析 + 逐帧映射"两遍遍历, 调用量约为原先 1/3。
                _frame_map = {}
                _mfile = Path(frames_path) / "frame_mappings.json"
                if _mfile.exists():
                    try:
                        _frame_map = json.loads(_mfile.read_text(encoding="utf-8"))
                    except Exception:
                        _frame_map = {}

                # 组装 (场景号, 帧列表); 帧目录若是平铺的(无场景切分), 整体当一个场景
                _pending_units = []
                for _unit in sorted(Path(frames_path).glob("unit_*")):
                    if not _unit.is_dir():
                        continue
                    _f = sorted(str(p) for p in _unit.glob("frame_*.jpg"))
                    if _f:
                        _pending_units.append((int(_unit.name.split("_")[1]), _f))
                if not _pending_units:
                    _f = sorted(str(p) for p in Path(frames_path).glob("frame_*.jpg"))
                    if _f:
                        _pending_units.append((0, _f))

                _scene_texts = []
                _pending_rows = []   # 闸4: 先攒后写,文件级证据门槛通过后统一落库
                _tag_unit_counts = {}  # 闸4: 标签 → 出现过的场景数
                logger.info(f"任务 {task_id}: 合并分析模式, 共 {len(_pending_units)} 个场景(每场景一次调用)")
                for _i, (_uidx, _uframes) in enumerate(_pending_units):
                    check_pause()

                    # 闸3: 垃圾帧(黑帧/糊帧/废帧)先剔除, 不送 NSFW 扫描与 VLM
                    if _fq is not None and _uframes:
                        _uframes, _junk = _fq.filter_frames(_uframes)
                        if not _uframes:
                            logger.info(f"任务 {task_id} 场景 {_uidx}: 全部帧为垃圾帧, 跳过")
                            continue

                    # 选帧 + 封面帧(NSFW 最高分; 检出不足则回退抽样)
                    _selected_frames = _uframes[:_map_frames_cap]
                    _cover_frame = _uframes[len(_uframes) // 2]
                    if _nsfw_det is not None:
                        try:
                            _scans = _nsfw_det.scan_frames(_uframes)
                            _nsfw_count += sum(1 for s in _scans if s.is_nsfw)
                            _selected_frames, _cover_frame, _sel_mode = select_scene_frames(
                                _uframes, _scans, _map_frames_cap, _map_min_nsfw)
                            if _sel_mode == "fallback":
                                _fallback_units += 1
                        except Exception as _ns_exc:
                            logger.warning(f"任务 {task_id} 场景 {_uidx}: NSFW 扫描失败, 退化为全帧: {_ns_exc}")

                    # 该场景一次 VLM 调用
                    _scene_text = vlm.analyze_frames_once(_selected_frames, _theme_prompt)
                    if not _scene_text:
                        continue
                    _scene_texts.append(_scene_text)

                    # 场景标签 → 封面帧(按标签浏览时的封面)
                    # 闸2 词典强制(非词典标签不入库) + 闸5 帧级互斥过滤
                    _tags = _parse_tags_from_vlm(_scene_text)
                    if _strict_vocab:
                        _tags, _off_vocab = _enforce_vocab(_tags, _std_tags)
                        if _off_vocab:
                            logger.info(f"任务 {task_id} 场景 {_uidx}: 非词典标签剔除 "
                                        f"{len(_off_vocab)} 个: {_off_vocab[:8]}")
                    _tags = _contradictions(_tags)
                    for _t in _tags:
                        _tag_unit_counts[_t] = _tag_unit_counts.get(_t, 0) + 1
                    if _tags and _cover_frame:
                        _finfo = _frame_map.get(str(_cover_frame)) or {}
                        if not isinstance(_finfo, dict):
                            _finfo = {"video_frame_number": _finfo}
                        _pending_rows.extend({
                            "task_id": task.id,
                            "entity_id": task.entity_id,
                            "tag_name": _tag,
                            "frame_path": _cover_frame,
                            "unit_index": _uidx,
                            "video_frame_number": _finfo.get("video_frame_number", 0),
                            "timestamp": _finfo.get("timestamp", 0.0),
                        } for _tag in _tags)
                    logger.info(
                        f"任务 {task_id} 场景 {_uidx} 合并分析完成: {len(_tags)} 标签"
                        f" (封面 {Path(_cover_frame).name if _cover_frame else '-'})")

                    if progress_callback:
                        progress_callback(
                            50 + int(40 * (_i + 1) / max(1, len(_pending_units))), 100,
                            f"合并分析中... ({_i + 1}/{len(_pending_units)} 场景)")

                # 闸4: 证据频率门槛 + 数量上限(文件级, 以实际分析的场景为母体)
                _gate_kept = _freq_gate(_tag_unit_counts, len(_scene_texts),
                                        _min_support, _support_ratio, _max_tags_file)
                _gated_out = len(_tag_unit_counts) - len(_gate_kept)
                if _gated_out:
                    logger.info(f"任务 {task_id}: 频率门槛剔除 {_gated_out} 个低证据标签"
                                f" (分析场景 {len(_scene_texts)})")
                if _pending_rows:
                    _pending_rows = [r for r in _pending_rows if r["tag_name"] in _gate_kept]
                    db.execute(_sqlite_insert(TagFrameMapping).on_conflict_do_nothing(
                        index_elements=["task_id", "tag_name", "frame_path"]), _pending_rows)
                db.commit()

                raw_text = "\n".join(_scene_texts)
                if _fallback_units:
                    logger.info(f"任务 {task_id}: 预筛未检出足量 NSFW, 回退抽样交 VLM 判定 {_fallback_units} 个场景")
                _nsfw_ratio = (_nsfw_count / _total_count) if _total_count else 0.0
                logger.info(
                    f"合并分析完成: {_nsfw_count}/{_total_count} 帧为NSFW ({_nsfw_ratio:.1%}), "
                    f"{len(_scene_texts)}/{len(_pending_units)} 场景有标签")
            else:
                # ── 旧两遍路径(analysis.merge_vlm_passes=false 时保留, 便于回退) ──
                result = analyzer.analyze(
                    frame_files, progress_callback=progress_callback, theme_name=_analysis_theme)
                _nsfw_count = result.nsfw_frames
                _total_count = result.total_frames
                raw_text = result.vlm_summary or ""
                logger.info(f"两阶段分析完成: {result.nsfw_frames}/{result.total_frames} 帧为NSFW ({result.nsfw_ratio:.1%})")

            vlm_time = round(time.time() - vlm_start, 2)
            analysis_details["timing"]["vlm_analysis"] = vlm_time
            analysis_details["vlm_raw"] = raw_text

            # 检查分析是否成功
            if not raw_text or len(raw_text.strip()) < 10:
                task.status = 'ERROR'
                task.progress = 30
                logger.warning(f"任务 {task_id} VLM 空结果细节: 长度={len(raw_text) if raw_text else 0}, NSFW帧数: {_nsfw_count}/{_total_count}")
                task.error_message = format_error("返回空结果或结果过短")
                task.completed_at = datetime.datetime.utcnow()
                file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
                db.commit()
                return

            # 确保进度为 70%（ANALYZING 阶段的上限）
            task.progress = 70
            task.status = 'ANALYZING'  # 保持状态一致
            file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
            db.commit()

            if not _merge_passes:
                # 逐场景分析并保存标签-帧映射（用于封面展示; 仅旧两遍路径需要）
                logger.info(f"任务 {task_id} 开始逐帧分析以建立标签-帧映射")
                _pending_rows = []       # 闸4: 先攒后写,文件级证据门槛通过后统一落库
                _tag_units = {}          # 闸4: 标签 → 出现过的场景(unit)集合
                _analyzed_units = 0      # 闸4: 实际分析的场景数(频率门槛母体)

                for unit_dir in sorted(Path(frames_path).glob("unit_*")):
                    if not unit_dir.is_dir():
                        continue

                    unit_index = int(unit_dir.name.split("_")[1])

                    # 选帧: NSFW 过滤 + 分数排序(封面 = 分数最高帧);
                    # 检出不足则回退抽样交 VLM 判定, 不丢弃场景
                    _unit_frames = sorted(str(p) for p in unit_dir.glob("frame_*.jpg"))
                    # 闸3: 垃圾帧(黑帧/糊帧/废帧)先剔除, 不送 NSFW 扫描与 VLM
                    if _fq is not None and _unit_frames:
                        _unit_frames, _junk = _fq.filter_frames(_unit_frames)
                        if not _unit_frames:
                            logger.info(f"任务 {task_id} 场景 {unit_index}: 全部帧为垃圾帧, 跳过")
                            continue
                    _selected_frames = None
                    if _unit_frames:
                        if _nsfw_det is not None:
                            try:
                                _scans = _nsfw_det.scan_frames(_unit_frames)
                                _selected_frames, _sel_mode = select_mapping_frames(
                                    _unit_frames, _scans, _map_frames_cap, _map_min_nsfw
                                )
                                if _sel_mode == "fallback":
                                    _fallback_units += 1
                                    logger.info(
                                        f"任务 {task_id} 场景 {unit_index}: 预筛检出不足"
                                        f"(<{_map_min_nsfw}), 回退抽样 {len(_selected_frames)} 帧交 VLM 判定")
                            except Exception as _ns_exc:
                                logger.warning(f"任务 {task_id} 场景 {unit_index}: NSFW 扫描失败, 退化为全帧: {_ns_exc}")
                                _selected_frames = _unit_frames[:_map_frames_cap]
                        else:
                            _selected_frames = _unit_frames[:_map_frames_cap]
                
                    # 逐帧分析该场景（使用与主分析同一主题的提示词; 仅分析的选中帧）
                    frame_tag_mappings = vlm.analyze_unit_frames_individually(
                        str(unit_dir), prompt=_theme_prompt, frame_paths=_selected_frames
                    )

                    # 攒批: 本场景所有 (帧, 标签) 行
                    _rows = []
                    for frame_path, frame_data in frame_tag_mappings.items():
                        # 解析标签：优先尝试 JSON 数组格式，回退到逗号分隔
                        tags_text = frame_data["tags"]
                        frame_number = frame_data.get("frame_number", 0)
                        timestamp = frame_data.get("timestamp", 0.0)
                        tags = _parse_tags_from_vlm(tags_text)
                        # 闸2 词典强制(非词典标签不入库) + 闸5 帧级互斥过滤
                        if _strict_vocab:
                            tags, _off_vocab = _enforce_vocab(tags, _std_tags)
                            if _off_vocab:
                                logger.info(f"任务 {task_id} 场景 {unit_index} "
                                            f"{Path(frame_path).name}: 非词典标签剔除 {_off_vocab}")
                        tags = _contradictions(tags)

                        for tag_name in tags:
                            _rows.append({
                                "task_id": task.id,
                                "entity_id": task.entity_id,  # 关键：设置 entity_id
                                "tag_name": tag_name,
                                "frame_path": frame_path,
                                "unit_index": unit_index,
                                "video_frame_number": frame_number,
                                "timestamp": timestamp,
                            })
                            _tag_units.setdefault(tag_name, set()).add(unit_index)
                    if frame_tag_mappings:
                        _analyzed_units += 1
                    _pending_rows.extend(_rows)
                    logger.info(f"任务 {task_id} 场景 {unit_index} 标签-帧映射暂存: "
                                f"{len(frame_tag_mappings)} 帧 ({len(_rows)} 条, 待文件级门槛)")

                # 闸4: 证据频率门槛 + 数量上限(文件级), 之后一次性落库
                _gate_kept = _freq_gate(
                    {t: len(u) for t, u in _tag_units.items()},
                    _analyzed_units, _min_support, _support_ratio, _max_tags_file)
                _gated_out = len(_tag_units) - len(_gate_kept)
                if _gated_out:
                    logger.info(f"任务 {task_id}: 频率门槛剔除 {_gated_out} 个低证据标签"
                                f" (分析场景 {_analyzed_units})")
                if _pending_rows:
                    _pending_rows = [r for r in _pending_rows if r["tag_name"] in _gate_kept]
                    _stmt = _sqlite_insert(TagFrameMapping).on_conflict_do_nothing(
                        index_elements=["task_id", "tag_name", "frame_path"]
                    )
                    db.execute(_stmt, _pending_rows)
                db.commit()
                logger.info(f"任务 {task_id} 标签-帧映射已保存: "
                            f"{len(_pending_rows)} 条 (频率门槛后)")

                if _fallback_units:
                    logger.info(f"任务 {task_id}: 预筛未检出足量 NSFW, 回退抽样交 VLM 判定 {_fallback_units} 个场景")
            
        finally:
            # 释放 VLM 信号量
            vlm_semaphore.release()
            logger.info(f"任务 {task_id} 释放 VLM 信号量")
        
        # 阶段3: 标签映射 (70-90%)
        task.status = 'CLUSTERING'
        task.progress = 70  # CLUSTERING 阶段起始进度
        db.commit()
        
        tag_mapping_start = time.time()
        dict_mgr = DictionaryManager(MD_MAPPING_PATH, db)
        mapped_tags, unmatched = dict_mgr.map_raw_text_to_tags(raw_text)
        # 闸 4: mapped_tags(NFO genre 直接来源)同样过证据频率门槛
        if _gate_kept is not None:
            _pre_gate = len(mapped_tags)
            mapped_tags = [t for t in mapped_tags if t in _gate_kept]
            if _pre_gate != len(mapped_tags):
                logger.info(f"任务 {task_id}: mapped_tags 频率门槛 {_pre_gate}→{len(mapped_tags)}")
        tag_mapping_time = round(time.time() - tag_mapping_start, 2)
        analysis_details["timing"]["tag_mapping"] = tag_mapping_time
        analysis_details["mapped_tags"] = mapped_tags
        analysis_details["unmatched"] = unmatched
        
        # 将未匹配标签写入审核队列
        for raw_label in unmatched:
            dict_mgr._add_to_audit_queue(raw_label)
        
        # 将 VLM 原始输出写入文件
        log_dir = Path(file_analysis.frames_path) if file_analysis.frames_path else Path(file_path).parent / f"{Path(file_path).stem}_frames"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "analysis_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        # 记录到 analysis_logs 表
        analysis_log = AnalysisLog(
            task_id=task.id,
            unit_index=0,  # 整体分析结果
            logfile=str(log_file)
        )
        db.add(analysis_log)
        
        file_analysis.mapped_tags = json.dumps(mapped_tags)
        file_analysis.frames_path = frames_path
        
        task.progress = 90  # CLUSTERING 阶段结束进度
        file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
        db.commit()
        
        # 阶段3.5: 人物名提取和交叉验证 (在 CLUSTERING 后，WRITING 前)
        try:
            from src.core.actor_name_extractor import ActorNameExtractor
            from src.core.vlm import create_vlm_client
            from src.core.config import ConfigManager
            
            extractor = ActorNameExtractor()
            config_manager = ConfigManager()
            vlm_client = create_vlm_client(config=config_manager)
            
            # 从文件名提取人物名
            filename_names = extractor.extract_from_filename(file_path)
            
            # 从水印提取人物名（VLM 在此阶段必定可用，因为 ANALYZING 阶段已成功）
            watermark_names = []
            if frames_path and os.path.exists(frames_path):
                try:
                    watermark_names = extractor.extract_from_watermark(frames_path, vlm_client)
                except Exception as e:
                    logger.warning(f"任务 {task_id} 水印识别失败: {e}")
            
            # 合并所有候选名称
            all_candidate_names = filename_names + watermark_names
            
            # 将识别结果写入 task_actor_names 表
            if all_candidate_names and task.entity_id:
                for name in set(all_candidate_names):
                    # 检查是否已存在
                    existing = db.query(TaskActorName).filter(
                        TaskActorName.entity_id == task.entity_id,
                        TaskActorName.actor_name == name
                    ).first()
                    
                    if not existing:
                        task_actor_name = TaskActorName(
                            entity_id=task.entity_id,
                            actor_name=name,
                            source='filename' if name in filename_names else 'watermark'
                        )
                        db.add(task_actor_name)
                
                db.commit()
                logger.info(f"任务 {task_id} 提取人物名: {all_candidate_names}")
                
                # 调用交叉验证，确认人物名
                confirmed_name = extractor.confirm_actor_name(
                    task.actor_id, 
                    all_candidate_names, 
                    db
                )
                
                if confirmed_name:
                    logger.info(f"任务 {task_id} 确认人物名: {confirmed_name}")
        except Exception as e:
            logger.warning(f"任务 {task_id} 人物名提取失败（非致命）: {e}")
            # 人物名提取失败不影响整体任务完成
        
        # 阶段4: 写入NFO文件 (90-100%)
        task.status = 'WRITING'
        task.progress = 90  # WRITING 阶段起始进度
        db.commit()
        
        from src.core.nfo_writer import generate_nfo

        # Query actor for collection name (Jellyfin <set> tag) and actor info
        collection_name = None
        actor_primary_name = None
        actor_aliases_list = []
        # 优先使用多人物关联，fallback 到 actor_id
        task_actor_records = db.query(TaskActor).filter(TaskActor.task_id == task.id).all() if task.id else []
        primary_actor = None
        if task_actor_records:
            primary_actor = db.query(Actor).filter(Actor.id == task_actor_records[0].actor_id).first()
        elif task.actor_id:
            primary_actor = db.query(Actor).filter(Actor.id == task.actor_id).first()
        if primary_actor:
            collection_name = primary_actor.name or primary_actor.code
            # 获取确认的人物名（优先使用 primary_name，否则使用 name）
            actor_primary_name = primary_actor.primary_name or primary_actor.name
            # 解析 aliases（JSON 数组）
            if primary_actor.aliases:
                try:
                    actor_aliases_list = json.loads(primary_actor.aliases)
                except:
                    actor_aliases_list = []

        nfo_writing_start = time.time()
        nfo_path = generate_nfo(
            file_path, 
            mapped_tags, 
            raw_text, 
            collection_name=collection_name,
            actor_name=actor_primary_name,
            actor_aliases=actor_aliases_list
        )
        nfo_writing_time = round(time.time() - nfo_writing_start, 2)
        analysis_details["timing"]["nfo_writing"] = nfo_writing_time

        # 验证 NFO 文件是否成功生成
        if not os.path.exists(nfo_path):
            task.status = 'ERROR'
            task.progress = 90
            logger.warning(f"任务 {task_id} NFO 写盘失败: {nfo_path}")
            task.error_message = format_error(f"NFO文件生成失败: {nfo_path}")
            task.completed_at = datetime.datetime.utcnow()
            file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
            db.commit()
            return

        # 阶段5: 人脸识别 (在COMPLETE之前执行) — multi-actor
        try:
            from src.core.face_recognition import FaceRecognizer

            recognizer = FaceRecognizer()
            actor_ids = None
            if recognizer.available:
                # 优先使用两阶段视频人脸检测
                if file_path and os.path.exists(file_path):
                    try:
                        logger.info(f"任务 {task_id}: 阶段5 - 使用两阶段视频人脸检测: {file_path}")
                        actor_ids = recognizer.recognize_actors_from_video(file_path, task.id, file_info["file_md5"])
                        actor_ids = [a for a in actor_ids if a is not None]
                        logger.info(f"任务 {task_id}: 两阶段检测识别到 {len(actor_ids)} 个人物")
                    except Exception as video_e:
                        logger.warning(f"任务 {task_id}: 两阶段视频检测失败，回退到帧目录方式: {video_e}")
                        actor_ids = None

                # 回退到帧目录方式
                if actor_ids is None and frames_path and os.path.exists(frames_path):
                    logger.info(f"任务 {task_id}: 阶段5 - 使用帧目录方式识别人脸: {frames_path}")
                    actor_ids = recognizer.recognize_actors(frames_path, task.id, file_info["file_md5"])
                    actor_ids = [a for a in actor_ids if a is not None]

                if actor_ids:
                    # 设置 actor_id 为第一个人物（向后兼容）
                    task.actor_id = actor_ids[0]
                    # 创建 TaskActor 关联记录
                    for aid in actor_ids:
                        existing_ta = db.query(TaskActor).filter(
                            TaskActor.task_id == task.id,
                            TaskActor.actor_id == aid
                        ).first()
                        if not existing_ta:
                            db.add(TaskActor(task_id=task.id, actor_id=aid))
                    logger.info(
                        "任务 %d 识别到 %d 个人物: %s",
                        task_id, len(actor_ids), actor_ids
                    )
                else:
                    if not frames_path or not os.path.exists(frames_path):
                        logger.warning(f"任务 {task_id} 帧目录不存在: {frames_path}")
                    else:
                        logger.info(f"任务 {task_id} 未检测到人脸")

                # 人脸扫描状态兜底写入(覆盖识别函数所有返回路径):
                # - 识别到人物 → complete
                # - 完整扫描过但无人物(快速存在性无脸/聚类为空等) → none
                # - actor_ids is None(模块不可用/未执行) → 不覆盖
                if actor_ids is not None and file_analysis is not None:
                    _scan = "complete" if actor_ids else "none"
                    file_analysis.face_scan_status = _scan
                    db.flush()
                    logger.info("任务 %d: 人脸扫描状态 = %s", task_id, _scan)
            else:
                if not recognizer.available:
                    logger.warning(f"任务 {task_id} 人脸识别模块不可用")
        except Exception as e:
            logger.error(f"任务 {task_id} 人脸识别失败: {e}")
            # 人脸识别失败不影响整体任务完成

        # ── 自动清理帧文件（ui-opt-8；配置键 frame_cleanup.*） ──
        _auto_clean_frames_if_enabled(
            config_manager.get("frame_cleanup", {}),
            file_analysis.frames_path,
            task_id,
        )

        # 完成
        task.status = 'COMPLETE'
        task.progress = 100
        task.completed_at = datetime.datetime.utcnow()

        # 同步 FileAnalysis 状态
        file_analysis.status = 'READY'
        file_analysis.progress = 100
        file_analysis.nfo_path = str(nfo_path)
        file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
        db.commit()
    except Exception as e:
        # 捕获任何异常，标记为错误状态
        logger.exception(f"任务分析异常(原始细节): {e}")
        if task:
            task.status = 'ERROR'
            task.completed_at = datetime.datetime.utcnow()
            task.error_message = format_error(str(e))

            # 同步 FileAnalysis 状态
            file_analysis.status = 'ERROR'
            file_analysis.analysis_details = json.dumps(analysis_details, ensure_ascii=False)
            db.commit()
    finally:
        db.close()


async def run_audit_worker(task_id: int):
    """
    异步版本的任务执行 Worker - 由调度器调用
    
    这个函数是异步的，但内部调用同步版本的 worker，
    在线程池中执行以避免阻塞事件循环。
    """
    # 在线程池中执行同步版本的 worker
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_audit_worker_sync, task_id)
