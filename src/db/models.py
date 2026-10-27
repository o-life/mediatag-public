from typing import Optional, Any
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, TEXT, DateTime, Float, ForeignKey, Boolean, UniqueConstraint, Index, event, Date, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
import datetime

Base = declarative_base()


class FileEntity(Base):
    """文件实体 - 以 MD5 为唯一标识，与路径解耦"""
    __tablename__ = 'file_entities'

    id = Column(Integer, primary_key=True)
    file_md5 = Column(String, unique=True, nullable=False, index=True)  # 内容唯一标识
    file_size = Column(Integer, nullable=True)   # 文件大小（字节）
    duration = Column(Float, nullable=True)      # 视频时长（秒，照片为 NULL）
    media_type = Column(String, nullable=True, default='video', index=True)  # video / photo
    resolution = Column(String, nullable=True)   # 分辨率（如 1920x1080，照片/视频）
    exif_data = Column(TEXT, nullable=True)      # 照片 EXIF 信息（JSON）
    capture_time = Column(String, nullable=True) # 照片拍摄时间
    analysis_theme = Column(String, nullable=True)  # 分析使用的主题（Phase 3）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    locations = relationship("FileLocation", back_populates="entity", cascade="all, delete-orphan")
    analyses = relationship("FileAnalysis", back_populates="entity", cascade="all, delete-orphan")
    actor_names = relationship("TaskActorName", back_populates="entity", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="entity")
    actor_faces = relationship("ActorFace", back_populates="entity")


class FileLocation(Base):
    """文件路径实例 - 同一文件可有多个路径"""
    __tablename__ = 'file_locations'

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='CASCADE'), nullable=False, index=True)
    file_path = Column(String, unique=True, nullable=False)  # 路径唯一
    file_name = Column(String, nullable=False)               # 文件名
    is_primary = Column(Integer, default=0)                  # 是否主路径（第一个导入的）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    entity = relationship("FileEntity", back_populates="locations")


class FileAnalysis(Base):
    """分析结果 - 视频文件的永久业务数据存储 (核心表)
    任务 (Task) 仅是作业过程，FileAnalysis 才是最终的视频档案。
    """
    __tablename__ = 'file_analysis'

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='CASCADE'), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey('tasks.id', ondelete='SET NULL'), nullable=True, index=True)  # 记录最后一次分析的任务 ID

    # 核心业务数据 (从 Task 剥离至此)
    mapped_tags = Column(TEXT, nullable=True)            # 映射后的标签（JSON）
    nfo_path = Column(String, nullable=True)             # 生成的 NFO 文件路径
    frames_path = Column(String, nullable=True)          # 抽帧目录路径
    analysis_details = Column(TEXT, nullable=True)       # 分析细节日志

    status = Column(String, nullable=False, default='PENDING')  # PENDING, READY, CONFIRMED, ERROR
    progress = Column(Float, default=0)

    # 人脸识别扫描状态（文件维度持久存储）:
    # None(未扫描) / complete(有人物) / none(已确认无人脸) / skipped(低清跳过)
    face_scan_status = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    entity = relationship("FileEntity", back_populates="analyses")
    task = relationship("Task", back_populates="file_analyses", cascade="none")


class TaskActorName(Base):
    """文件维度人物名 - 从文件名/水印提取的人物名"""
    __tablename__ = 'task_actor_names'

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='CASCADE'), nullable=False, index=True)
    actor_name = Column(String, nullable=False)
    source = Column(String, nullable=False)       # filename / watermark
    confidence = Column(Float, default=1.0)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    entity = relationship("FileEntity", back_populates="actor_names")


class Actor(Base):
    __tablename__ = 'actors'

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)  # ACTOR_001, ACTOR_002...
    name = Column(String, nullable=True)               # 用户可选填
    primary_name = Column(String, nullable=True)       # 确认的主名
    aliases = Column(TEXT, nullable=True)              # 别名列表（JSON数组）
    face_embedding = Column(TEXT, nullable=True)       # JSON 序列化的特征向量
    sample_count = Column(Integer, default=0)          # 样本数量
    cover_face = Column(String, nullable=True)         # 封面人脸图片路径（最佳正面照）
    avg_face_sharpness = Column(Float, nullable=True)  # 平均人脸清晰度（Laplacian方差）
    # 人脸完整度标注（训练数据积累）: complete(完整脸) / partial(局部/误检) / unknown(未标注)
    face_type = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    faces = relationship("ActorFace", back_populates="actor", cascade="all, delete-orphan")


class ActorFace(Base):
    __tablename__ = 'actor_faces'

    id = Column(Integer, primary_key=True)
    actor_id = Column(Integer, ForeignKey('actors.id'), nullable=False, index=True)
    task_id = Column(Integer, ForeignKey('tasks.id', ondelete='SET NULL'), nullable=True, index=True)   # 允许为NULL，删除任务时保留人脸记录
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='SET NULL'), nullable=True, index=True)  # 新增：关联文件实体
    file_md5 = Column(String, nullable=True, index=True)  # 视频文件MD5，保留向后兼容
    face_path = Column(String, nullable=False)            # 人脸图片路径（相对路径）
    yaw = Column(Float, nullable=True)   # 偏航角（左右转头）
    pitch = Column(Float, nullable=True) # 俯仰角（抬头低头）
    roll = Column(Float, nullable=True)  # 翻滚角（歪头）
    is_cover = Column(Integer, default=0)  # 是否为封面照
    sharpness = Column(Float, nullable=True)  # 人脸清晰度分数（Laplacian方差，原始值）
    quality_score = Column(Float, default=0)  # 人脸质量分数（0-1，综合评分）
    eyes_open_score = Column(Float, default=0)  # 睁眼评分（0=闭眼，1=睁眼）
    embedding = Column(TEXT, nullable=True)  # 帧级特征向量 JSON（512维），用于跨文件人脸归一
    face_type = Column(String, default='main')  # 'main' (主脸) | 'merged' (合并脸)
    main_face_id = Column(Integer, ForeignKey('actor_faces.id'), nullable=True)  # 合并脸指向主脸的 ID
    # ── 时空/检测维度（全维度入库，供聚类先验与结果溯源）──
    frame_id = Column(Integer, nullable=True, index=True)   # 帧号（视频内）
    timestamp = Column(Float, nullable=True)                # 时间戳（秒）
    bbox = Column(TEXT, nullable=True)                      # 人脸框 JSON [x1,y1,x2,y2]（原帧坐标）
    det_score = Column(Float, nullable=True)                # 检测置信度（0-1）
    kps = Column(TEXT, nullable=True)                       # 106 关键点 JSON [[x,y],...]
    scene_id = Column(Integer, nullable=True, index=True)   # 所属场景索引（场景切分结果）
    sex = Column(String, nullable=True)                     # 性别 'M'/'F'（genderage 模型）
    age = Column(Integer, nullable=True)                    # 年龄估计（genderage 模型）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    actor = relationship("Actor", back_populates="faces")
    task = relationship("Task", back_populates="actor_faces")
    entity = relationship("FileEntity", back_populates="actor_faces")
    main_face = relationship("ActorFace", remote_side=[id], backref="merged_faces")


class Task(Base):
    """纯作业调度表 - 仅记录任务状态，不存储业务数据"""
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='SET NULL'), nullable=True, index=True)  # 关联文件实体

    # [DEPRECATED] 冗余字段，建议从 FileEntity/FileAnalysis 获取
    file_path = Column(String, nullable=True)
    file_md5 = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)

    # 作业状态字段
    status = Column(String, default='PENDING')  # QUEUED, RUNNING, COMPLETE, ERROR
    progress = Column(Integer, default=0)       # 0-100
    error_message = Column(TEXT, nullable=True)      # 错误信息

    # 人脸识别扫描状态: None(未扫描) / complete(有人物) / none(已确认无人脸) / skipped(低清跳过)
    face_scan_status = Column(String, nullable=True, index=True)

    # [DEPRECATED] 保留字段用于向后兼容，优先使用 task_actors 关联表
    actor_id = Column(Integer, ForeignKey('actors.id'), nullable=True, index=True)

    # [影楼基础包 S3] 素材所属活动/项目（可空 = 未归类收件箱）
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=True, index=True)
    pinned = Column(Integer, default=0)  # 置顶（派发队列绝对优先；0/1；14 号方案优先按钮）

    # 调度器相关字段
    queued_at = Column(DateTime, nullable=True)   # 加入队列时间
    started_at = Column(DateTime, nullable=True)  # 开始执行时间
    completed_at = Column(DateTime, nullable=True)  # 完成时间

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    entity = relationship("FileEntity", back_populates="tasks")
    actor_faces = relationship("ActorFace", back_populates="task")
    file_analyses = relationship("FileAnalysis", back_populates="task", cascade="none")
    actor = relationship("Actor", foreign_keys=[actor_id])  # [DEPRECATED] 单人物关联
    task_actors = relationship("TaskActor", back_populates="task", cascade="all, delete-orphan")  # 多人物关联

    @property
    def actors(self):
        """获取任务关联的所有人物（通过 task_actors 关联表）"""
        return [ta.actor for ta in self.task_actors]


class TaskActor(Base):
    """任务-人物关联表（多对多）

    替代 Task.actor_id 的单人物限制，支持一个视频关联多个人物。
    """
    __tablename__ = 'task_actors'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False, index=True)
    actor_id = Column(Integer, ForeignKey('actors.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # 唯一约束：同一任务不能重复关联同一人物
    __table_args__ = (
        UniqueConstraint('task_id', 'actor_id', name='uq_task_actor'),
    )

    # Relationships
    task = relationship("Task", back_populates="task_actors")
    actor = relationship("Actor")


# ── 影楼基础包 S3.1：客单 / 活动类型 / 活动（项目）──────────────────

class Client(Base):
    """客户（委托方）：长期存在，可多次委托不同类型活动。"""
    __tablename__ = 'clients'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)          # 客户称呼（张三&李四 / 王府）
    contact = Column(String, nullable=True)        # 联系方式（可空；全本地存储）
    notes = Column(TEXT, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class EventType(Base):
    """活动类型词表：预置种子 + 用户自定义；停用不删除（历史活动引用保留）。"""
    __tablename__ = 'event_types'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    is_builtin = Column(Integer, default=0)        # 预置类型可停用不可删
    sort_order = Column(Integer, default=0)
    active = Column(Integer, default=1)


class Project(Base):
    """活动/项目：分类主轴——活动类型 + 拍摄日期 + 状态挂在活动层，
    素材（tasks）经 project_id 归属（可空 = 未归类收件箱）。

    2026-09-24 起「项目」= 文件夹语义：经 parent_id 支持任意层级嵌套，
    parent_id 为空即顶层文件夹；子文件夹 client_id 继承根活动。"""
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey('projects.id'), nullable=True, index=True)
    event_type_id = Column(Integer, ForeignKey('event_types.id'), nullable=True, index=True)
    name = Column(String, nullable=False)          # 默认自动生成"{客户名}·{类型}{日期}"
    event_date = Column(Date, nullable=True)       # 拍摄日期（时间维度主轴）
    package = Column(String, nullable=True)        # 套系（自由文本，可空）
    status = Column(String, nullable=True, default='拍摄中')
    notes = Column(TEXT, nullable=True)
    custom_fields = Column(TEXT, nullable=True, default='{}')  # 用户自定义字段（JSON：{键: 值}，可增删改）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    client = relationship("Client")
    event_type = relationship("EventType")
    parent = relationship("Project", remote_side=[id], backref="children")


class ProjectAsset(Base):
    """项目自持清单：项目 ↔ 内容（file_md5）成员关系——归属唯一存储。

    锚内容哈希而非任务/实体主键：任务增删、实体重建、路径迁移、删除重导
    都不影响归属——同内容再导入（md5 重算一致）自动恢复成员资格（tombstone）。
    (project_id, file_md5) 唯一；file_md5 不设全局唯一 → 同一内容可属多个项目
    （多归属，2026-09-30 用户裁定"项目自持清单"语义）。
    tasks.project_id 为影子列：停写保留一个版本作回滚通道，届时 drop。"""
    __tablename__ = 'project_assets'

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    file_md5 = Column(String, nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('project_id', 'file_md5', name='uq_project_assets_project_md5'),
    )

    project = relationship("Project")


class DictionaryEntry(Base):
    __tablename__ = 'dictionary'

    standard_tag = Column(String, primary_key=True)
    aliases = Column(TEXT)          # Comma separated trigger words
    category = Column(String, default='其他')  # 场景/体位/动作/人数/属性/服装/视觉重点
    file_count = Column(Integer, default=0)    # 关联文件数
    theme_id = Column(Integer, nullable=True, index=True)  # 所属主题（Phase 3）


class Theme(Base):
    """标签主题：主题 = VLM prompt + 标签字典（Phase 3）"""
    __tablename__ = 'themes'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)   # 主题名（如"日常"/"成人"）
    description = Column(String, nullable=True)           # 主题说明
    is_default = Column(Integer, default=0)               # 是否默认主题
    label_mapping_path = Column(TEXT, nullable=True)      # 该主题绑定的标签映射 md 文件路径
    faces_dir = Column(TEXT, nullable=True)               # 该主题的人脸资源库路径（第2轮接入识别流程）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class ThemePrompt(Base):
    """主题对应的 VLM prompt（Phase 3）"""
    __tablename__ = 'theme_prompts'

    id = Column(Integer, primary_key=True)
    theme_id = Column(Integer, ForeignKey('themes.id', ondelete='CASCADE'), nullable=False, index=True)
    prompt = Column(TEXT, nullable=False)                 # VLM 提示词
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class TagCategory(Base):
    __tablename__ = 'tag_categories'

    name = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AuditQueueItem(Base):
    __tablename__ = 'audit_queue'

    id = Column(Integer, primary_key=True)
    raw_label = Column(String, nullable=False)   # VLM 输出的原始标签
    count = Column(Integer, default=1)           # 出现次数
    ai_recommendation = Column(String)           # AI 推荐的标准标签
    status = Column(String, default='pending')   # pending/confirmed/rejected
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SystemConfig(Base):
    __tablename__ = 'system_config'

    key = Column(String, primary_key=True)
    value = Column(TEXT)
    description = Column(String)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class TagFrameMapping(Base):
    """标签与关键帧的映射关系表

    记录每个标签在哪些帧中被识别，用于封面展示。
    """
    __tablename__ = 'tag_frame_mappings'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id', ondelete='SET NULL'), nullable=True, index=True)  # 允许为NULL，删除任务时保留映射
    entity_id = Column(Integer, ForeignKey('file_entities.id', ondelete='SET NULL'), nullable=True, index=True)  # 关联文件实体
    tag_name = Column(String, nullable=False, index=True)  # 标签名称
    frame_path = Column(String, nullable=False)  # 帧图片路径
    unit_index = Column(Integer, nullable=True)  # 场景索引（unit_X）
    video_frame_number = Column(Integer, nullable=True)  # 视频中的帧号（用于精准定位播放）
    timestamp = Column(Float, nullable=True)  # 帧在视频中的时间戳（秒）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    task = relationship("Task")
    entity = relationship("FileEntity")

    # 唯一约束：同一任务的同一标签在同一帧只记录一次
    __table_args__ = (
        UniqueConstraint('task_id', 'tag_name', 'frame_path', name='uq_task_tag_frame'),
        # 标签树/字典页的覆盖量统计是 GROUP BY tag_name + COUNT(DISTINCT entity_id)：
        # 全表 1.6M 行时走覆盖索引可做 index-only scan（实测 1.5s → 0.07s）。
        Index('idx_tfm_tag_entity', 'tag_name', 'entity_id'),
    )


class AnalysisLog(Base):
    """VLM 分析日志 - 原始输出存储在文件系统，此表记录路径"""
    __tablename__ = 'analysis_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False, index=True)
    unit_index = Column(Integer, nullable=False)
    logfile = Column(String, nullable=False)  # 日志文件路径
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    task = relationship("Task")




def _get_default_db_path() -> str:
    """获取默认数据库路径（根据运行环境自动选择）"""
    try:
        from src.core.path_manager import get_path_manager
        return str(get_path_manager().db_file)
    except Exception:
        return 'data/audit_cache.db'


def get_db_engine(db_path: Optional[str] = None):
    if db_path is None:
        db_path = _get_default_db_path()
    engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={
            'check_same_thread': False,
            'timeout': 30,
        },
        pool_size=20,
        max_overflow=20,
        pool_pre_ping=True,
    )
    # 启用 WAL 模式以支持并发读写
    #
    # 注意（实测结论，见 openspec/changes/fix-sqlite-write-lock）：
    # 曾尝试在写闸内让事务发 `BEGIN IMMEDIATE`（关闭 pysqlite 隐式 BEGIN + begin
    # 事件显式发 BEGIN）。压测证明这**有害**：begin 事件对只读语句也会触发，
    # 使 SELECT 也进入事务并持有快照；随后同一事务写回时因 WAL 快照过期升级，
    # 反而**制造** database is locked（8 线程 ×20 次读后写：0 → 140 次锁错误）。
    # 故此处保持 pysqlite 默认（SELECT 不开启事务），写串行化只由写闸
    # （GuardedSession）承担。
    from sqlalchemy import event

    @event.listens_for(engine, 'connect')
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        # 写锁等待 30 秒（批量人脸识别并发写时避免 database is locked）
        cursor.execute('PRAGMA busy_timeout=30000;')
        # WAL 模式下 NORMAL 足够安全且性能更好
        cursor.execute('PRAGMA synchronous=NORMAL;')
        cursor.close()

    return engine


def init_db(db_path: Optional[str] = None):
    from sqlalchemy import text  # 函数内多处使用；提前导入防遮蔽（F823）
    if db_path is None:
        db_path = _get_default_db_path()
    
    print(f"[Database] 初始化数据库: {db_path}")
    
    # 确保数据库目录存在
    db_dir = Path(db_path).parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Database] 数据库目录已创建: {db_dir}")
    except Exception as e:
        print(f"[Database] 创建数据库目录失败: {e}")
        raise
    
    engine = get_db_engine(db_path)
    
    # 检查是否是首次创建（所有表都不存在）
    from sqlalchemy import inspect
    inspector = inspect(engine)
    is_first_run = not inspector.has_table('dictionary')
    print(f"[Database] 首次运行: {is_first_run}")
    
    Base.metadata.create_all(engine)
    print("[Database] 数据库表已创建")

    # Auto-migrate: add new columns to existing tables
    with engine.connect() as conn:
        # actors.avg_face_sharpness
        actors_cols = [c['name'] for c in inspector.get_columns('actors')]
        if 'avg_face_sharpness' not in actors_cols:
            conn.execute(create_column_sql('actors', 'avg_face_sharpness', 'REAL'))
            conn.commit()

        # actors.face_type（人脸完整度标注，训练数据积累）
        if 'face_type' not in actors_cols:
            conn.execute(create_column_sql('actors', 'face_type', 'TEXT'))
            conn.commit()

        # actor_faces.sharpness
        actor_faces_cols = [c['name'] for c in inspector.get_columns('actor_faces')]
        if 'sharpness' not in actor_faces_cols:
            conn.execute(create_column_sql('actor_faces', 'sharpness', 'REAL'))
            conn.commit()

        # actor_faces.eyes_open_score
        if 'eyes_open_score' not in actor_faces_cols:
            conn.execute(create_column_sql('actor_faces', 'eyes_open_score', 'REAL'))
            conn.commit()

        # tasks compatibility columns (restored for worker compatibility)
        tasks_cols = [c['name'] for c in inspector.get_columns('tasks')]
        for col_name, col_type in [
            ('mapped_tags', 'TEXT'),
            ('frames_path', 'TEXT'),
            ('nfo_path', 'TEXT'),
            ('analysis_details', 'TEXT'),
            ('pinned', 'INTEGER')
        ]:
            if col_name not in tasks_cols:
                conn.execute(create_column_sql('tasks', col_name, col_type))
                conn.commit()

        # tasks.face_scan_status（人脸扫描状态，任务维度过渡字段）
        if 'face_scan_status' not in tasks_cols:
            conn.execute(create_column_sql('tasks', 'face_scan_status', 'TEXT'))
            conn.commit()

        # file_analysis.face_scan_status（文件维度持久字段——无人脸记录核心）
        fa_cols = [c['name'] for c in inspector.get_columns('file_analysis')]
        if 'face_scan_status' not in fa_cols:
            conn.execute(create_column_sql('file_analysis', 'face_scan_status', 'TEXT'))
            conn.commit()

        # ── 方案 A：tag_frame_mappings.entity_id 回填（幂等，file 为核心）──
        # 旧管线按 task_id 写标签，新链路（标签浏览/详情）按 entity_id 读——
        # 只补空列：task → entity 的外键关系仍在时即可搬；已填/孤儿不动。
        tfm_cols = [c['name'] for c in inspector.get_columns('tag_frame_mappings')]
        if 'entity_id' in tfm_cols and 'task_id' in tfm_cols:
            orphans = conn.execute(text(
                "SELECT COUNT(*) FROM tag_frame_mappings "
                "WHERE entity_id IS NULL AND task_id IS NOT NULL")).scalar()
            if orphans:
                conn.execute(text(
                    "UPDATE tag_frame_mappings SET entity_id = "
                    "(SELECT entity_id FROM tasks WHERE tasks.id = tag_frame_mappings.task_id) "
                    "WHERE entity_id IS NULL AND task_id IS NOT NULL"))
                conn.commit()
                print(f"[Database] 标签关联回填 file 锚点: {orphans} 条")

        # ── 影楼基础包 S3.1：客单/活动/类型词表迁移（幂等）──
        _migrate_studio_tables(conn, inspector)

    # 首次运行或 dictionary 表为空时导入标准标签
    if is_first_run:
        print("[Database] 首次运行，导入标准标签...")
        _load_standard_tags_sqlalchemy(engine)
    else:
        # 检查 dictionary 表是否为空（可能是旧数据库未导入标签）
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM dictionary")).scalar()
            print(f"[Database] dictionary 表记录数: {result}")
            if result == 0:
                print("[Database] dictionary 表为空，导入标准标签...")
                _load_standard_tags_sqlalchemy(engine)

    return engine


def create_column_sql(table: str, column: str, column_type: str) -> Any:
    """Generate ALTER TABLE ADD COLUMN SQL statement."""
    from sqlalchemy import text
    return text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


# 预置活动类型词表（影楼基础包 S3；用户可在设置页增删自定义类型）
EVENT_TYPE_PRESETS = [
    "婚礼", "婚庆庆典", "宝宝宴·满月", "生日宴", "寿宴", "升学宴",
    "丧葬礼仪", "写真", "全家福", "开业庆典", "年会·活动", "会议·论坛", "其他",
]


def _seed_event_types(conn) -> int:
    """空 event_types 表时写入预置词表；返回写入条数（幂等）。"""
    count = conn.execute(text("SELECT COUNT(*) FROM event_types")).scalar()
    if count:
        return 0
    for i, name in enumerate(EVENT_TYPE_PRESETS, 1):
        conn.execute(
            text("INSERT INTO event_types (name, is_builtin, sort_order, active) VALUES (:n, 1, :s, 1)"),
            {"n": name, "s": i},
        )
    return len(EVENT_TYPE_PRESETS)


def _migrate_studio_tables(conn, inspector) -> None:
    """S3.1 迁移：老库 tasks 补 project_id 列 + 索引 + 词表种子（幂等）。

    新库由 create_all 直接带全（守卫自动跳过）；老库 tasks 缺列时 ALTER 补齐。
    """
    tasks_cols = [c['name'] for c in inspector.get_columns('tasks')]
    if 'project_id' not in tasks_cols:
        conn.execute(create_column_sql('tasks', 'project_id', 'INTEGER REFERENCES projects(id)'))
        conn.commit()
    projects_cols = [c['name'] for c in inspector.get_columns('projects')]
    if 'custom_fields' not in projects_cols:
        conn.execute(create_column_sql('projects', 'custom_fields', "TEXT DEFAULT '{}'"))
        conn.commit()
    if 'parent_id' not in projects_cols:
        conn.execute(create_column_sql('projects', 'parent_id', 'INTEGER REFERENCES projects(id)'))
        conn.commit()
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_tasks_project_id ON tasks (project_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_client_id ON projects (client_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_event_type_id ON projects (event_type_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_projects_parent_id ON projects (parent_id)"))
    # ── project_assets（归属清单，project-assets-roster 2026-09-30）──
    # 新库由 create_all 直带全；此处兜底建表（防 inspector 时序差）+ 索引 +
    # 影子列回填。回填幂等（INSERT OR IGNORE + 唯一约束），亦是回滚通道的
    # 再同步点：影子列停写期间若曾回滚，重跑本迁移即恢复清单一致。
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS project_assets (
            id INTEGER PRIMARY KEY,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            file_md5 VARCHAR NOT NULL,
            added_at DATETIME,
            CONSTRAINT uq_project_assets_project_md5 UNIQUE (project_id, file_md5)
        )"""))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_project_assets_project_id ON project_assets (project_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_project_assets_file_md5 ON project_assets (file_md5)"))
    # 影子列回填（幂等）：旧 tasks.project_id → 清单行。极老库的 tasks 可能
    # 尚无 entity_id/file_md5 列——按实际存在的列拼回填 SQL（双无则跳过，
    # 此类行本就无法归属，重导即恢复）。
    _has_entity = 'entity_id' in tasks_cols
    _has_md5 = 'file_md5' in tasks_cols
    if _has_entity or _has_md5:
        _md5_expr = ("COALESCE(e.file_md5, t.file_md5)" if (_has_entity and _has_md5)
                     else "e.file_md5" if _has_entity
                     else "t.file_md5")
        _join = "LEFT JOIN file_entities e ON t.entity_id = e.id" if _has_entity else ""
        conn.execute(text(f"""
            INSERT OR IGNORE INTO project_assets (project_id, file_md5)
            SELECT DISTINCT t.project_id, {_md5_expr}
            FROM tasks t {_join}
            WHERE t.project_id IS NOT NULL AND {_md5_expr} IS NOT NULL
        """))
    conn.commit()
    _seed_event_types(conn)


def _load_standard_tags_sqlalchemy(engine) -> None:
    """加载标准标签到 dictionary 表（首次初始化时调用）"""
    try:
        import sys
        import json
        from pathlib import Path
        
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
        
        # 使用 SQLAlchemy 插入标签
        from src.db.models import DictionaryEntry
        
        with get_session(engine) as session:
            for tag in tags_data:
                # 检查是否已存在
                existing = session.query(DictionaryEntry).filter(
                    DictionaryEntry.standard_tag == tag['standard_tag']
                ).first()
                
                if not existing:
                    entry = DictionaryEntry(
                        standard_tag=tag['standard_tag'],
                        category=tag['category'],
                        aliases=tag['aliases']
                    )
                    session.add(entry)
            
            session.commit()
        
        print(f"✅ 已导入 {len(tags_data)} 个标准标签")
        
    except Exception as e:
        print(f"⚠ 导入标准标签失败: {e}")
        import traceback
        traceback.print_exc()


def _statement_is_write(statement) -> bool:
    """该语句是否为写（DML）——用于把所有写操作纳入写闸。

    覆盖：ORM/Core 的 Insert/Update/Delete（`is_dml`），以及 `text()` 手写
    DML（按首关键字判定）。历史踩坑：VLM 标签写用 `db.execute(insert(...))`
    （Core execute），不经 `Session.flush()`，因而绕过了写闸。
    """
    if getattr(statement, "is_dml", False):
        return True
    try:
        from sqlalchemy.sql.elements import TextClause
        if isinstance(statement, TextClause):
            s = str(statement).strip().lstrip("(").strip()
            head = s.split(None, 1)[0].lower() if s else ""
            return head in {"insert", "update", "delete", "replace"}
    except Exception:
        pass
    return False


class GuardedSession(Session):
    """写闸保护的 Session（见 openspec/changes/fix-sqlite-write-lock，A 方案）。

    **事务级**串行化：从本次事务的**首次写**起持有写闸，直到
    `commit()` / `rollback()` / `close()` 释放。这样进程内同一时刻只有**一个
    写事务**在推进 —— 不只消除"写-写争抢"，也消除"某事务已释放写闸但事务仍
    开着（SQLite 写锁仍被持有）"造成的 `database is locked`。

    覆盖**所有**写入口：
    - `flush()`（ORM 脏对象 / 显式 flush）
    - `execute()` 与 `do_orm_execute`（Core `insert/update/delete`、ORM 批量
      `query().update()`、`text()` 手写 DML）—— 这些**不经过** `flush()`，
      是历史上绕过写闸、残留 `database is locked` 的根源。

    读操作（SELECT）不占写闸，WAL 快照读照常并行。
    前提：写事务要短（识别已按 actor 分批提交）。
    """

    _gate_held = False

    def _acquire_gate(self) -> None:
        if self._gate_held:
            return
        from src.db.write_gate import acquire_write_gate
        if not acquire_write_gate():
            raise TimeoutError("write_gate: 等待写闸超时（有长写事务未释放）")
        self._gate_held = True

    def _release_gate(self) -> None:
        if self._gate_held:
            from src.db.write_gate import release_write_gate
            release_write_gate()
            self._gate_held = False

    def execute(self, statement, *args, **kwargs):
        if _statement_is_write(statement):
            self._acquire_gate()
        return super().execute(statement, *args, **kwargs)

    def flush(self, *args, **kwargs):
        # 无待写对象时不占闸（读多写少的 session 不受影响）
        if not (self.new or self.dirty or self.deleted):
            return super().flush(*args, **kwargs)
        self._acquire_gate()
        try:
            return super().flush(*args, **kwargs)
        except Exception:
            # flush 失败 → 事务进入不确定状态，闸交 rollback/close 释放
            self._release_gate()
            raise

    def commit(self):
        self._acquire_gate()
        try:
            return super().commit()
        finally:
            self._release_gate()

    def rollback(self):
        try:
            return super().rollback()
        finally:
            self._release_gate()

    def close(self):
        try:
            return super().close()
        finally:
            self._release_gate()


@event.listens_for(GuardedSession, "do_orm_execute")
def _guarded_orm_execute(state):
    """ORM 执行钩子：批量 `query().update()/delete()`、`Session.execute()` 等
    不经 `flush()` 的写，也纳入写闸。"""
    try:
        if _statement_is_write(state.statement):
            state.session._acquire_gate()
    except Exception:
        raise


def session_factory(engine):
    """统一的 Session 工厂（写闸保护）。

    **所有**需要建 session 的地方都应走它（`get_session` / 各处 sessionmaker）——
    否则会绕过写闸，在并发下重新引入 `database is locked`（历史踩坑：
    scheduler 与若干 API 端点直接用裸 `sessionmaker(bind=engine)`）。

    expire_on_commit=False: commit 后不失效已加载对象属性，避免 worker 频繁
    commit 后再次查询同一对象（减少读库次数与锁竞争窗口）。
    """
    return sessionmaker(bind=engine, expire_on_commit=False, class_=GuardedSession)


def get_session(engine):
    return session_factory(engine)()
