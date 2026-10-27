"""SQLite schema for ai-video-factory.

Five tables: tasks, montage_units, dictionary, audit_queue, nfo_writes.
No ORM — raw sqlite3 with parameterized queries.
"""

import sqlite3
import sys
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _default_db_path() -> Path:
    """默认库路径锚定数据根(path_manager)——裸相对路径在打包态
    (CWD=数据根)会落到 <根>/data/audit_cache.db 这个错误位置
    (真实库为 <根>/db/audit_cache.db)。解析失败回退旧相对路径。"""
    try:
        from src.core.path_manager import get_path_manager
        return Path(get_path_manager().db_file)
    except Exception:  # noqa: BLE001 - 循环导入/早期调用兜底
        return Path("data/audit_cache.db")


DB_PATH = _default_db_path()


def _ensure_db_path(path: Optional[str] = None) -> str:
    """Ensure the data directory exists and return the db path string."""
    db_path = Path(path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def get_engine(path: Optional[str] = None) -> str:
    """Return the database file path (creates directory if needed)."""
    return _ensure_db_path(path)


def get_connection(path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection with row factory."""
    conn = sqlite3.connect(_ensure_db_path(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")  # 与 ORM 层(models.py)保持一致
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _load_standard_tags(conn: sqlite3.Connection) -> None:
    """加载标准标签到 dictionary 表（首次初始化时调用）"""
    try:
        # 查找标准标签 JSON 文件
        if getattr(sys, 'frozen', False):
            # PyInstaller 打包环境：标准标签在 Contents/Resources/assets/
            # sys.executable 指向 Contents/MacOS/MediaTag
            base_path = Path(sys.executable).parent.parent / 'Resources'
        else:
            # 开发环境
            base_path = Path(__file__).parent.parent.parent
        
        json_path = base_path / 'assets' / 'standard_tags.json'
        
        if not json_path.exists():
            print(f"⚠ 标准标签文件不存在: {json_path}")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            tags_data = json.load(f)
        
        # 插入标签
        cursor = conn.cursor()
        for tag in tags_data:
            cursor.execute("""
                INSERT OR IGNORE INTO dictionary (category, standard_tag, aliases)
                VALUES (?, ?, ?)
            """, (tag['category'], tag['standard_tag'], tag['aliases']))
        
        conn.commit()
        print(f"✅ 已导入 {len(tags_data)} 个标准标签")
        
    except Exception as e:
        print(f"⚠ 导入标准标签失败: {e}")


def init_db(path: Optional[str] = None) -> sqlite3.Connection:
    """Create all tables if they don't exist. Return a connection.
    
    Auto-creates DB file on first run and imports standard tags.
    """
    conn = get_connection(path)
    cur = conn.cursor()

    # 检查是否是首次创建（dictionary 表为空）
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dictionary'")
    is_first_run = cursor.fetchone() is None

    # --- 1. tasks ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT    NOT NULL,
            file_name       TEXT    NOT NULL,
            file_size       INTEGER NOT NULL DEFAULT 0,
            file_md5        TEXT    UNIQUE,
            status          TEXT    NOT NULL DEFAULT 'scheduled',
            current_unit    INTEGER DEFAULT 0,
            total_units     INTEGER DEFAULT 0,
            frames_analyzed INTEGER DEFAULT 0,
            audit_progress  TEXT    DEFAULT '{}',
            error_message   TEXT,
            face_scan_status TEXT,  -- None/complete/none/skipped（人脸识别扫描状态）
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            completed_at    TEXT
        )
    """)

    # --- 2. montage_units ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS montage_units (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id           INTEGER NOT NULL REFERENCES tasks(id),
            unit_index        INTEGER NOT NULL,
            start_frame       INTEGER NOT NULL,
            end_frame         INTEGER NOT NULL,
            duration_seconds  REAL    DEFAULT 0.0,
            status            TEXT    NOT NULL DEFAULT 'pending',
            frame_paths       TEXT    DEFAULT '[]',
            raw_tags          TEXT    DEFAULT '[]',
            created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(task_id, unit_index)
        )
    """)

    # --- 3. dictionary ---
    # file_count/theme_id 为 ORM(DictionaryEntry) 必需列——2026-10-04 修复：
    # 新建库缺这两列导致词典 ORM 查询报 no such column（现网库早已手工演进补齐）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            category        TEXT    NOT NULL,
            standard_tag    TEXT    UNIQUE NOT NULL,
            aliases         TEXT    DEFAULT '[]',
            description     TEXT,
            file_count      INTEGER DEFAULT 0,
            theme_id        INTEGER,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS ix_dictionary_theme_id
        ON dictionary (theme_id)
    """)

    # --- 4. audit_queue ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_queue (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            category            TEXT    NOT NULL,
            raw_label           TEXT    NOT NULL,
            status              TEXT    NOT NULL DEFAULT 'pending',
            mapped_to           TEXT,
            ai_recommendation   TEXT    DEFAULT 'null',
            source_task_id      INTEGER REFERENCES tasks(id),
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            reviewed_at         TEXT
        )
    """)

    # --- 5. nfo_writes ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nfo_writes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id         INTEGER NOT NULL REFERENCES tasks(id),
            nfo_path        TEXT    NOT NULL,
            written_tags    TEXT    DEFAULT '[]',
            wrote_at        TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)


    # --- 6. face_segments (阶段3快速预扫描结果) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_segments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id             INTEGER NOT NULL REFERENCES tasks(id),
            unit_index          INTEGER DEFAULT 0,
            start_time          REAL    NOT NULL,
            end_time            REAL    NOT NULL,
            face_count_prescan  INTEGER DEFAULT 0,
            prescan_status      TEXT    NOT NULL DEFAULT 'pending',
            dense_status        TEXT    NOT NULL DEFAULT 'pending',
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(task_id, unit_index, start_time, end_time)
        )
    """)

    # --- 7. face_dense_frames (阶段3密集采样帧) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_dense_frames (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id             INTEGER NOT NULL REFERENCES tasks(id),
            segment_id          INTEGER NOT NULL REFERENCES face_segments(id),
            frame_index         INTEGER NOT NULL,
            timestamp           REAL    NOT NULL,
            frame_path          TEXT,
            detection_status    TEXT    NOT NULL DEFAULT 'pending',
            error_message       TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(task_id, segment_id, frame_index)
        )
    """)

    # --- 8. face_detections (阶段3高质量人脸检测结果) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_detections (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            dense_frame_id      INTEGER NOT NULL REFERENCES face_dense_frames(id),
            task_id             INTEGER NOT NULL REFERENCES tasks(id),
            face_index          INTEGER NOT NULL,
            confidence          REAL    NOT NULL DEFAULT 0.0,
            bbox                TEXT    DEFAULT '[]',
            landmarks           TEXT    DEFAULT '{}',
            quality_clarity     REAL    DEFAULT 0.0,
            quality_pose        REAL    DEFAULT 0.0,
            quality_brightness  REAL    DEFAULT 0.0,
            quality_overall     REAL    DEFAULT 0.0,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # --- 9. face_embeddings (阶段4人脸识别特征向量) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            face_detection_id   INTEGER NOT NULL REFERENCES face_detections(id),
            task_id             INTEGER NOT NULL REFERENCES tasks(id),
            embedding           BLOB    NOT NULL,
            embedding_dim       INTEGER DEFAULT 512,
            quality             REAL    DEFAULT 0.0,
            face_image_path     TEXT,
            recognition_status  TEXT    NOT NULL DEFAULT 'pending',
            error_message       TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # --- 索引 ---
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_segments_task ON face_segments(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_dense_frames_task ON face_dense_frames(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_dense_frames_segment ON face_dense_frames(segment_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_detections_task ON face_detections(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_detections_frame ON face_detections(dense_frame_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_task ON face_embeddings(task_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_detection ON face_embeddings(face_detection_id)")

    conn.commit()

    # 检查并添加 duration 列（如果不存在）
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'duration' not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN duration REAL DEFAULT 0.0")
        conn.commit()

    # 检查并添加 face_scan_status 列（人脸识别扫描状态）
    cursor.execute("PRAGMA table_info(tasks)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'face_scan_status' not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN face_scan_status TEXT")
        conn.commit()

    # 检查并添加 embedding 列（per-face 特征向量，用于跨文件人脸归一）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actor_faces'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(actor_faces)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'embedding' not in columns:
            conn.execute("ALTER TABLE actor_faces ADD COLUMN embedding TEXT")
            conn.commit()

    # 检查并添加 face_type 列（区分主脸/合并脸）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actor_faces'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(actor_faces)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'face_type' not in columns:
            conn.execute("ALTER TABLE actor_faces ADD COLUMN face_type VARCHAR DEFAULT 'main'")
            conn.commit()
        if 'main_face_id' not in columns:
            conn.execute("ALTER TABLE actor_faces ADD COLUMN main_face_id INTEGER REFERENCES actor_faces(id)")
            conn.commit()

    # 迁移 face_path 为相对路径
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actor_faces'")
    if cursor.fetchone() is not None:
        cursor.execute("SELECT face_path FROM actor_faces WHERE face_path LIKE '/Volumes/Data/actor_faces/%' LIMIT 1")
        if cursor.fetchone() is not None:
            conn.execute("""
                UPDATE actor_faces 
                SET face_path = REPLACE(face_path, '/Volumes/Data/actor_faces/', '')
                WHERE face_path LIKE '/Volumes/Data/actor_faces/%'
            """)
            conn.commit()

    # 检查并添加时空/检测维度列（全维度入库：帧号/时间戳/bbox/检测置信度/关键点/场景）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actor_faces'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(actor_faces)")
        columns = [row[1] for row in cursor.fetchall()]
        _af_migrations = [
            ("frame_id", "INTEGER"),
            ("timestamp", "REAL"),
            ("bbox", "TEXT"),
            ("det_score", "REAL"),
            ("kps", "TEXT"),
            ("scene_id", "INTEGER"),
            ("sex", "VARCHAR(2)"),
            ("age", "INTEGER"),
        ]
        for col, coltype in _af_migrations:
            if col not in columns:
                conn.execute(f"ALTER TABLE actor_faces ADD COLUMN {col} {coltype}")
        conn.commit()

    # 检查并添加 file_entities 列（media_type/exif 等，照片支持 Phase 2）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_entities'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(file_entities)")
        columns = [row[1] for row in cursor.fetchall()]
        _fe_migrations = [
            ("media_type", "VARCHAR(16) DEFAULT 'video'"),
            ("resolution", "VARCHAR(32)"),
            ("exif_data", "TEXT"),
            ("capture_time", "VARCHAR(64)"),
            ("analysis_theme", "VARCHAR(64)"),
        ]
        for col, coltype in _fe_migrations:
            if col not in columns:
                conn.execute(f"ALTER TABLE file_entities ADD COLUMN {col} {coltype}")
        conn.commit()

    # Phase 3: 标签主题系统 —— themes / theme_prompts 表 + dictionary.theme_id
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='themes'")
    if cursor.fetchone() is None:
        conn.execute("""
            CREATE TABLE themes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        VARCHAR(64) UNIQUE NOT NULL,
                description TEXT,
                is_default  INTEGER DEFAULT 0,
                label_mapping_path TEXT,
                faces_dir   TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # 创建默认主题（"日常"）
        conn.execute("INSERT INTO themes (name, description, is_default) VALUES ('日常', '默认主题', 1)")
        conn.commit()
    else:
        # 已存在的库不会走上面的建表 → 幂等补列（分析主题绑定：标签映射文件 + 人脸资源库）
        cursor.execute("PRAGMA table_info(themes)")
        _theme_cols = [row[1] for row in cursor.fetchall()]
        if 'label_mapping_path' not in _theme_cols:
            conn.execute("ALTER TABLE themes ADD COLUMN label_mapping_path TEXT")
        if 'faces_dir' not in _theme_cols:
            conn.execute("ALTER TABLE themes ADD COLUMN faces_dir TEXT")
        conn.commit()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='theme_prompts'")
    if cursor.fetchone() is None:
        conn.execute("""
            CREATE TABLE theme_prompts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                theme_id    INTEGER NOT NULL REFERENCES themes(id) ON DELETE CASCADE,
                prompt      TEXT NOT NULL,
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # 默认主题 prompt 占位（可在设置页编辑）
        conn.execute(
            "INSERT INTO theme_prompts (theme_id, prompt) "
            "SELECT id, '（默认提示词，可在设置页编辑）' FROM themes WHERE is_default=1"
        )
        conn.commit()

    # dictionary 表加 theme_id 列
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dictionary'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(dictionary)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'theme_id' not in columns:
            conn.execute("ALTER TABLE dictionary ADD COLUMN theme_id INTEGER")
            conn.commit()

    # Phase 7: 性能优化索引（高频查询字段）
    # 第三项支持多列（元组），用于覆盖索引
    _indexes = [
        ("idx_tasks_status", "tasks", "status"),
        ("idx_tasks_file_md5", "tasks", "file_md5"),
        ("idx_file_entities_media_type", "file_entities", "media_type"),
        ("idx_file_entities_analysis_theme", "file_entities", "analysis_theme"),
        ("idx_actor_faces_frame_id", "actor_faces", "frame_id"),
        ("idx_actor_faces_face_type", "actor_faces", "face_type"),
        # 标签覆盖量统计（GROUP BY tag_name + COUNT(DISTINCT entity_id)）的覆盖索引：
        # tag_frame_mappings 达百万行时，无此索引需回表逐行取 entity_id（实测 ~1.5s）；
        # 有则可做 index-only scan（实测 ~0.07s）。标签树/字典/标签浏览三处共用。
        ("idx_tfm_tag_entity", "tag_frame_mappings", ("tag_name", "entity_id")),
    ]
    for idx_name, table, cols in _indexes:
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name=?", (idx_name,))
            if cursor.fetchone() is None:
                col_sql = ", ".join(cols) if isinstance(cols, (list, tuple)) else cols
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col_sql})")
        except Exception as e:
            logger.warning(f"创建索引失败 {idx_name}: {e}")
    conn.commit()

    # 首次运行时导入标准标签
    if is_first_run:
        _load_standard_tags(conn)

    return conn


if __name__ == "__main__":
    db = init_db()
    print(f"Database initialized at {DB_PATH}")
    db.close()
