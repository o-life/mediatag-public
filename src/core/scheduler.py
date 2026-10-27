"""
任务调度器 - 管理任务队列和并发控制

核心功能：
1. 任务队列管理（支持优先级）
2. 多阶段并发控制（MD5/场景检测/VLM 分析）
3. 自动调度（任务完成后自动启动下一个）
4. 错误处理和重试机制
"""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Set, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

# 统一走写闸保护的 Session 工厂（勿用裸 sessionmaker，否则绕过写闸 →
# 并发下重新引入 database is locked）
from src.db.models import session_factory

logger = logging.getLogger(__name__)


class SchedulerStatus(str, Enum):
    """调度器状态"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class TaskPriority(int, Enum):
    """任务优先级"""
    HIGH = 1
    NORMAL = 2
    LOW = 3


# 调度器启动时「可以加载执行」的任务状态：
#   - QUEUED：用户已显式激活，等待执行
#   - 中间态：上次运行被中断的孤儿任务，重置为 QUEUED 后继续
# 刻意不含：
#   - PENDING：已导入但未激活，须用户勾选后点「分析」
#   - PAUSED ：用户暂停，跨重启保持暂停，须用户主动恢复
#   - FAILED ：失败，须用户主动重试
QUEUED_TASK_STATUSES = ['QUEUED']
INTERRUPTED_TASK_STATUSES = ['SIFTING', 'ANALYZING', 'CLUSTERING', 'CHECKING', 'WRITING']


class TaskScheduler:
    """
    任务调度器 - 支持多阶段并发控制
    
    负责：
    1. 管理任务队列
    2. 控制并发执行（总并发 + 各阶段并发）
    3. 自动调度下一个任务
    """
    
    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        max_md5_concurrent: Optional[int] = None,
        max_sifting_concurrent: Optional[int] = None,
        max_vlm_concurrent: Optional[int] = None,
        queue_size: int = 10000,
        task_timeout: int = 3600,
        retry_on_failure: bool = True,
        max_retries: int = 3
    ):
        # 从配置文件读取性能方案
        from src.core.config import ConfigManager
        config_manager = ConfigManager()
        
        # 如果参数未提供，从配置文件读取
        if max_concurrent is None:
            max_concurrent = config_manager.get("scheduler.max_concurrent")
        if max_md5_concurrent is None:
            max_md5_concurrent = config_manager.get("scheduler.max_md5_concurrent")
        if max_sifting_concurrent is None:
            max_sifting_concurrent = config_manager.get("scheduler.max_sifting_concurrent")
        if max_vlm_concurrent is None:
            max_vlm_concurrent = config_manager.get("scheduler.max_vlm_concurrent")
        
        # 如果配置文件中没有具体参数，根据硬件自动计算最优配置
        if any(v is None for v in [max_concurrent, max_md5_concurrent, max_sifting_concurrent, max_vlm_concurrent]):
            from src.core.hardware_detector import HardwareDetector
            
            # 获取硬件推荐配置
            recommendations = HardwareDetector.get_recommendations()
            device_class = recommendations["device_class"]
            presets = recommendations["presets"]
            
            # 根据设备等级自动选择最优方案
            # ultra/max: 极致性能，pro: 性能均衡，basic: 极致节省
            if device_class in ["ultra", "max"]:
                preset = presets["performance"]
                scheme = "极致性能（自动）"
            elif device_class == "pro":
                preset = presets["balanced"]
                scheme = "性能均衡（自动）"
            else:  # basic
                preset = presets["economy"]
                scheme = "极致节省（自动）"
            
            # 应用预设值
            if max_concurrent is None:
                max_concurrent = preset["max_concurrent"]
            if max_md5_concurrent is None:
                max_md5_concurrent = preset["max_md5_concurrent"]
            if max_sifting_concurrent is None:
                max_sifting_concurrent = preset["max_sifting_concurrent"]
            if max_vlm_concurrent is None:
                max_vlm_concurrent = preset["max_vlm_concurrent"]
            
            logger.info(f"设备等级: {device_class}, 自动选择方案: {scheme}, 配置: {preset}")
        
        self.max_concurrent = max_concurrent
        self.max_md5_concurrent = max_md5_concurrent
        self.max_sifting_concurrent = max_sifting_concurrent
        self.max_vlm_concurrent = max_vlm_concurrent
        self.queue_size = queue_size
        self.task_timeout = task_timeout
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries

        # 短任务优先(全链路派发按时长升序, 即 SJF 纯排序方案)。
        # 关闭时中位排序键恒为 0, 退化为原先的 (优先级, task_id) FIFO 行为。
        self.short_task_first = bool(
            config_manager.get("scheduler.short_task_first", True)
        )
        # 派发策略（14 号方案定稿）：short_first / long_first / import_order
        # 新键缺省时从旧 short_task_first 迁移
        _ds = config_manager.get("scheduler.dispatch_strategy")
        self.dispatch_strategy = _ds or (
            "short_first" if self.short_task_first else "import_order")

        # task_id -> 时长(秒), 用于排序键; 顺带避免重复查库
        self._duration_cache: Dict[int, float] = {}

        # 任务队列（使用优先级队列）
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=queue_size)
        
        # 正在运行的任务
        self.running_tasks: Set[int] = set()
        
        # 任务重试计数
        self.task_retries: Dict[int, int] = {}
        
        # SQLite 锁冲突独立重排计数(不占用 task_retries 额度)
        self._lock_requeue_count: Dict[int, int] = {}
        
        # 单任务暂停控制
        self.paused_tasks: Set[int] = set()  # 被暂停的任务ID集合
        self._task_pause_events: Dict[int, asyncio.Event] = {}  # 每个任务的暂停事件

        # 已取消排队任务：派发时拦截丢弃（POP 后不丢弃集合成员——同一任务可能
        # 重复入队；也不随 stop/start 清空——防陈旧队列项复活）。用户显式重新
        # 入队（submit_task）时解除。
        self._cancelled_tasks: Set[int] = set()
        
        # 调度器状态
        self.status = SchedulerStatus.STOPPED
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 各阶段信号量（使用 threading.Semaphore 因为 worker 是同步函数）
        self.md5_semaphore = threading.Semaphore(max_md5_concurrent)
        self.sifting_semaphore = threading.Semaphore(max_sifting_concurrent)
        self.vlm_semaphore = threading.Semaphore(max_vlm_concurrent)
        
        # 调度器锁
        self.scheduler_lock = asyncio.Lock()
        
        # 调度任务（用于自动调度）
        self._scheduler_task: Optional[asyncio.Task] = None

        # ── 前置条件闸门（"就绪才派发"）──
        # 典型用法：VLM 模型没准备好时**暂停派发**，而不是让每条任务各自失败。
        # 未就绪 ⇒ 不消费队列、置 PAUSED，由哨兵任务定期复检并自动恢复。
        self._readiness_check: Optional[Callable[[], Tuple[bool, str]]] = None
        self._readiness_interval: float = 30.0
        self._readiness_paused: bool = False
        self._readiness_reason: str = ""
        self._readiness_checked_at: float = 0.0
        self._readiness_watch_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 线程池执行器（用于 CPU 密集型任务）
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        logger.info(
            f"TaskScheduler initialized: max_concurrent={max_concurrent}, "
            f"max_md5_concurrent={max_md5_concurrent}, "
            f"max_sifting_concurrent={max_sifting_concurrent}, "
            f"max_vlm_concurrent={max_vlm_concurrent}, "
            f"queue_size={queue_size}, task_timeout={task_timeout}"
        )
    
    def _sort_key(self, task_id: int, duration: Optional[float] = None) -> float:
        """返回队列排序用的时长键(越小越先派发)。

        派发策略（scheduler.dispatch_strategy，14 号方案）：
        - short_first：时长升序（未知 → inf 排末位）
        - long_first：时长降序（未知 → -inf 排末位，不抢最有把握的长任务）
        - import_order：恒 0.0（FIFO，按 task_id 稳定排序）
        置顶任务（tasks.pinned）：键整体 -1e9 → 恒排未置顶之前。
        """
        strategy = getattr(self, "dispatch_strategy", "short_first")
        if strategy == "import_order":
            key = 0.0
        else:
            if duration is None:
                duration = self._duration_cache.get(task_id)
            if duration is None:
                duration = self._lookup_duration(task_id)
            if duration and duration > 0:
                key = float(duration)
                if strategy == "long_first":
                    key = -key
            else:
                key = -1e9 if strategy == "long_first" else 1e9
        if self._is_pinned(task_id):
            key -= 1e9
        return key

    def _is_pinned(self, task_id: int) -> bool:
        """查任务是否置顶（tasks.pinned；查询失败按未置顶）。"""
        try:
            from src.db.models import Task, get_db_engine

            engine = get_db_engine()
            SessionLocal = session_factory(engine)
            session = SessionLocal()
            try:
                task = session.query(Task).filter(Task.id == task_id).first()
                return bool(task and getattr(task, "pinned", 0))
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"查询置顶状态失败 task={task_id}: {e}")
            return False

    async def resort_queue(self):
        """置顶状态变化后重建队列（PriorityQueue 不支持原地改序）：
        取出全部条目，按当前排序键重新入队。"""
        items = []
        while not self.task_queue.empty():
            items.append(await self.task_queue.get())
        for priority_value, _old_key, task_id in items:
            await self.task_queue.put(
                (priority_value, self._sort_key(task_id), task_id)
            )
        logger.info(f"队列已按当前排序键重排（{len(items)} 条）")

    def _lookup_duration(self, task_id: int) -> Optional[float]:
        """从数据库查任务时长(秒)并缓存; 查不到返回 None。"""
        try:
            from src.db.models import Task, get_db_engine

            engine = get_db_engine()
            SessionLocal = session_factory(engine)
            session = SessionLocal()
            try:
                task = session.query(Task).filter(Task.id == task_id).first()
                dur = getattr(task, "duration", None) if task else None
                if dur and dur > 0:
                    self._duration_cache[task_id] = float(dur)
                    return float(dur)
            finally:
                session.close()
        except Exception as e:
            logger.warning(f"查询任务 {task_id} 时长失败(按未知处理): {e}")
        return None

    async def submit_task(
        self,
        task_id: int,
        priority: TaskPriority = TaskPriority.NORMAL,
        duration: Optional[float] = None
    ) -> bool:
        """
        提交任务到队列

        Args:
            task_id: 任务ID
            priority: 任务优先级
            duration: 素材时长(秒), 供短任务优先排序; 不传则查库

        Returns:
            是否成功提交
        """
        if self.status == SchedulerStatus.STOPPED:
            logger.warning("调度器已停止，无法提交任务")
            return False

        if self.task_queue.full():
            logger.warning(f"任务队列已满，无法提交任务 {task_id}")
            return False

        # 排序键: (优先级, 时长, task_id) —— 优先级数字越小越优先;
        # 同优先级内时长升序(短任务优先); 再以 task_id 稳定排序
        # 显式重新入队 = 用户改了主意，解除之前的取消排队
        self._cancelled_tasks.discard(task_id)
        await self.task_queue.put(
            (priority.value, self._sort_key(task_id, duration), task_id)
        )

        logger.info(f"任务 {task_id} 已提交到队列，优先级: {priority.name}")

        # 如果调度器正在运行，尝试立即调度
        if self.status == SchedulerStatus.RUNNING:
            await self._schedule_next()

        return True

    
    async def submit_batch(self, task_ids: List[int], priority: TaskPriority = TaskPriority.NORMAL) -> int:
        """
        批量提交任务
        
        Args:
            task_ids: 任务 ID 列表
            priority: 任务优先级
            
        Returns:
            成功提交的任务数
        """
        submitted = 0
        for task_id in task_ids:
            if await self.submit_task(task_id, priority):
                submitted += 1
        
        logger.info(f"批量提交任务: {submitted}/{len(task_ids)}")
        return submitted
    
    async def start(self):
        """启动调度器"""
        if self.status == SchedulerStatus.RUNNING:
            logger.warning("调度器已经在运行")
            return
        
        self.status = SchedulerStatus.RUNNING
        
        # 记下事件循环：工作线程（ThreadPoolExecutor）请求"就绪性暂停"时要投回这里
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # 从数据库加载待处理任务到内存队列
        await self._load_pending_tasks()
        
        # 启动自动调度任务
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        self.status = SchedulerStatus.STOPPED
        
        # 取消调度任务
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 关闭线程池
        self.executor.shutdown(wait=False)
        
        logger.info("调度器已停止")
    
    async def pause(self):
        """暂停调度（**用户意图优先**：会接管"就绪性暂停"，哨兵不再自动恢复）"""
        was_readiness_pause = self._readiness_paused
        self._readiness_paused = False
        self._readiness_reason = ""

        if self.status == SchedulerStatus.RUNNING:
            self.status = SchedulerStatus.PAUSED
            logger.info("调度器已暂停")
        elif was_readiness_pause:
            logger.info("调度器保持暂停（就绪性暂停转由用户接管，不会自动恢复）")
    
    async def resume(self):
        """恢复调度"""
        if self.status == SchedulerStatus.PAUSED:
            self.status = SchedulerStatus.RUNNING
            logger.info("调度器已恢复")

            # 若暂停期间循环已退出（while 条件不成立），这里要重新拉起来
            if self._scheduler_task is None or self._scheduler_task.done():
                self._scheduler_task = asyncio.create_task(self._scheduler_loop())

            # 立即尝试调度
            await self._schedule_next()

    # ── 前置条件闸门（就绪才派发）─────────────────────────────────
    def set_readiness_check(self, check: Optional[Callable[[], Tuple[bool, str]]],
                            interval_seconds: Optional[float] = None) -> None:
        """安装"派发前置条件"检查：未就绪则暂停派发并定期复检。

        Args:
            check: ``() -> (是否就绪, 原因)``；传 ``None`` 关闭闸门。
                会在**线程池**里执行（可能含网络请求），因此必须是可以阻塞的函数。
            interval_seconds: 复检间隔；缺省 30s。
        """
        self._readiness_check = check
        if interval_seconds is not None:
            self._readiness_interval = float(interval_seconds)

    async def _readiness_ok(self) -> bool:
        """闸门判定：就绪 → True；未就绪 → 置 PAUSED 并返回 False（带节流）。"""
        now = time.monotonic()
        if now - self._readiness_checked_at < self._readiness_interval:
            # 刚探测过：保持当前判定，不重复打网络
            return not self._readiness_paused
        self._readiness_checked_at = now

        check = self._readiness_check
        if check is None:
            return True
        try:
            loop = asyncio.get_running_loop()
            ready, reason = await loop.run_in_executor(None, check)
        except Exception as exc:  # 探测本身出错 ⇒ 视为未就绪（宁可暂停）
            ready, reason = False, f"就绪性探测失败: {exc}"

        if ready:
            self._readiness_paused = False
            self._readiness_reason = ""
            return True
        self._pause_for_readiness(str(reason))
        return False

    def _pause_for_readiness(self, reason: str) -> None:
        """因前置条件不满足而暂停（幂等；保留任务在队列）。"""
        first = not self._readiness_paused
        self._readiness_paused = True
        self._readiness_reason = reason
        self.status = SchedulerStatus.PAUSED
        if first:
            logger.warning("调度器暂停（前置条件未满足）：%s —— 任务保留在队列，就绪后自动恢复", reason)
        self._spawn_readiness_watch()

    def request_readiness_pause(self, reason: str) -> None:
        """**线程安全**：请求"因前置条件未满足而暂停"。

        供工作线程（跑在 ThreadPoolExecutor 里）在发现模型缺失时调用 —— 例如
        `src/api/main.py` 的 VLM 预检。立即置 PAUSED 并挂上复检哨兵。
        """
        self._readiness_checked_at = time.monotonic()  # 刚判定过，别马上再探
        self._pause_for_readiness(reason)

    def _spawn_readiness_watch(self) -> None:
        """挂复检哨兵（同一时刻只有一个）。可在事件循环内或跨线程调用。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 跨线程：把挂载动作投回调度器所在的事件循环
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self._spawn_readiness_watch)
            return
        if self._readiness_watch_task is None or self._readiness_watch_task.done():
            self._readiness_watch_task = asyncio.create_task(self._readiness_watch())

    async def _readiness_watch(self) -> None:
        """定期复检前置条件；就绪后自动恢复派发。"""
        while self._readiness_paused and self.status == SchedulerStatus.PAUSED:
            await asyncio.sleep(self._readiness_interval)
            if not self._readiness_paused:
                return  # 已被用户手动恢复/暂停接管
            self._readiness_checked_at = 0.0  # 强制本次重新探测
            if await self._readiness_ok():
                self.status = SchedulerStatus.RUNNING
                logger.info("前置条件已满足，调度器恢复派发")
                if self._scheduler_task is None or self._scheduler_task.done():
                    self._scheduler_task = asyncio.create_task(self._scheduler_loop())
                return
    
    async def cancel_task(self, task_id: int) -> bool:
        """
        取消队列中的任务

        只把任务 ID 记入取消集，真正丢弃发生在派发点（_schedule_next）——
        asyncio.PriorityQueue 不支持按成员删除。幂等，可对不在队列的任务调用
        （兜住"库里已退回 PENDING 但内存队列还有残留"的窗口）。

        Args:
            task_id: 任务 ID

        Returns:
            是否成功取消
        """
        self._cancelled_tasks.add(int(task_id))
        logger.info(f"任务 {task_id} 已取消排队（派发时拦截）")
        return True
    
    async def pause_task(self, task_id: int) -> bool:
        """
        暂停单个任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否成功暂停
        """
        # 检查任务是否在运行中（内存状态或数据库状态）
        if task_id not in self.running_tasks:
            # 从数据库检查任务状态
            from src.db.repository import get_db
            db = get_db()
            try:
                from src.db.models import Task
                # 使用 SQLAlchemy 查询
                from src.db.models import get_db_engine
                engine = get_db_engine()
                SessionLocal = session_factory(engine)
                session = SessionLocal()
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task or task.status not in ('SIFTING', 'ANALYZING', 'CLUSTERING', 'CHECKING', 'WRITING'):
                    logger.warning(f"任务 {task_id} 不在运行中，无法暂停")
                    return False
                # 将任务添加到 running_tasks 以保持一致性
                self.running_tasks.add(task_id)
                session.close()
            finally:
                db.close()
        
        if task_id in self.paused_tasks:
            logger.warning(f"任务 {task_id} 已经暂停")
            return False
        
        # 标记任务为暂停状态
        self.paused_tasks.add(task_id)
        
        # 创建暂停事件（如果不存在）
        if task_id not in self._task_pause_events:
            self._task_pause_events[task_id] = asyncio.Event()
            self._task_pause_events[task_id].set()  # 初始为设置状态（运行中）
        
        # 清除事件，使任务等待
        self._task_pause_events[task_id].clear()
        
        logger.info(f"任务 {task_id} 已暂停")
        return True
    
    async def resume_task(self, task_id: int) -> bool:
        """
        恢复单个任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否成功恢复
        """
        # 检查内存中的暂停状态
        if task_id not in self.paused_tasks:
            # 从数据库检查任务状态
            from src.db.repository import get_db
            db = get_db()
            try:
                from src.db.models import Task, get_db_engine
                engine = get_db_engine()
                SessionLocal = session_factory(engine)
                session = SessionLocal()
                task = session.query(Task).filter(Task.id == task_id).first()
                # 允许恢复任何未完成的任务（包括 PAUSED 和运行中状态）
                if not task or task.status in ['COMPLETE', 'FAILED']:
                    logger.warning(f"任务 {task_id} 已完成或失败，无法恢复")
                    session.close()
                    return False
                # 将任务添加到内存暂停集合以保持一致性
                self.paused_tasks.add(task_id)
                session.close()
            finally:
                db.close()
        
        # 从暂停集合中移除
        self.paused_tasks.remove(task_id)
        
        # 设置事件，唤醒等待的任务
        if task_id not in self._task_pause_events:
            self._task_pause_events[task_id] = asyncio.Event()
        self._task_pause_events[task_id].set()
        
        logger.info(f"任务 {task_id} 已恢复")
        return True
    
    def is_task_paused(self, task_id: int) -> bool:
        """
        检查任务是否被暂停
        
        Args:
            task_id: 任务 ID
            
        Returns:
            任务是否被暂停
        """
        return task_id in self.paused_tasks
    
    async def check_task_pause(self, task_id: int):
        """
        检查任务暂停状态（在任务执行过程中调用）
        
        Args:
            task_id: 任务 ID
        """
        if task_id in self._task_pause_events:
            await self._task_pause_events[task_id].wait()
    
    async def _load_pending_tasks(self):
        """从数据库加载「已激活」与「被中断」的任务到内存队列。

        只加载 QUEUED（用户已激活）与孤儿中间态（上次运行被中断，重置为
        QUEUED 后继续）。不加载 PENDING / PAUSED / FAILED —— 分别为
        「未激活」「已暂停」「已失败」，都必须由用户在 UI 上主动操作。
        """
        try:
            from src.db.models import Task, get_db_engine
            
            engine = get_db_engine()
            SessionLocal = session_factory(engine)
            session = SessionLocal()
            
            # 只加载已激活 + 孤儿恢复（不含 PENDING/PAUSED/FAILED）
            loadable_statuses = QUEUED_TASK_STATUSES + INTERRUPTED_TASK_STATUSES
            pending_tasks = session.query(Task).filter(
                Task.status.in_(loadable_statuses)
            ).all()
            
            loaded_count = 0
            for task in pending_tasks:
                # 如果任务已经在运行集合中，跳过
                if task.id in self.running_tasks:
                    continue
                
                # 将处理中的任务重置为 QUEUED 状态
                if task.status in INTERRUPTED_TASK_STATUSES:
                    original_status = task.status
                    task.status = 'QUEUED'
                    logger.info(f"恢复孤儿任务 {task.id} (原状态: {original_status})")
                
                # 提交任务到队列(带时长 → 短任务优先排序)
                await self.task_queue.put(
                    (TaskPriority.NORMAL.value, self._sort_key(task.id, task.duration), task.id)
                )
                loaded_count += 1
            
            session.commit()
            session.close()
            
            if loaded_count > 0:
                logger.info(f"从数据库加载了 {loaded_count} 个已激活/待恢复任务到队列")
        except Exception as e:
            logger.error(f"加载待处理任务失败: {e}")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.status == SchedulerStatus.RUNNING:
            try:
                await self._schedule_next()
                await asyncio.sleep(0.1)  # 避免过度占用 CPU
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _schedule_next(self):
        """调度下一个任务"""
        if self.status != SchedulerStatus.RUNNING:
            return

        # 队列空时不探测（空闲不该因为模型缺失被标成"暂停"）
        if self.task_queue.empty():
            return

        # ── 前置条件闸门：未就绪就**不消费队列**，置 PAUSED 等复检 ──
        # 放在锁外：探测最长要等 HTTP 超时，不该阻塞其它任务的完成回调；
        # 放在取任务之前：否则"取出来再放回去"会打乱排序键与队列语义。
        if self._readiness_check is not None and not await self._readiness_ok():
            return

        async with self.scheduler_lock:
            # 检查是否可以调度更多任务
            if len(self.running_tasks) >= self.max_concurrent:
                return
            
            # 检查队列是否为空
            if self.task_queue.empty():
                return
            
            try:
                # 获取下一个任务(排序键: 优先级, 时长, task_id)
                _, _dur, task_id = await asyncio.wait_for(
                    self.task_queue.get(), timeout=0.1
                )
                
                # 检查任务是否被暂停
                if self.is_task_paused(task_id):
                    logger.info(f"任务 {task_id} 已暂停，跳过调度")
                    self.task_queue.task_done()
                    return

                # 检查任务是否已取消排队（如 batch-unqueue 退回 PENDING）
                if task_id in self._cancelled_tasks:
                    logger.info(f"任务 {task_id} 已取消排队，跳过调度")
                    self.task_queue.task_done()
                    return
                
                # 启动任务
                self.running_tasks.add(task_id)
                logger.info(f"开始执行任务 {task_id}")
                
                # 在线程池中执行任务
                asyncio.create_task(self._execute_task(task_id))
                
            except asyncio.TimeoutError:
                # 队列为空
                pass
            except Exception as e:
                logger.error(f"调度任务失败: {e}")
    
    async def _execute_task(self, task_id: int):
        """执行任务"""
        try:
            # 在线程池中执行同步任务
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._run_task_worker,
                task_id
            )
            
            logger.info(f"任务 {task_id} 执行完成")
            
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            
            # SQLite 并发写锁冲突是瞬时错误(另一 worker 短暂持锁)。
            # 不应消耗任务重试次数, 也不该立即标记 FAILED——等待后重新入队,
            # 避免"锁冲突→整任务回滚重跑"的放大效应(实测 16 并发下 265 次返工)。
            # submit_task 的 put 与 finally 的 task_done 平衡; finally 的
            # _schedule_next 会取到刚放回的任务再次执行。
            _err_str = str(e)
            if "database is locked" in _err_str or "database is locked" in _err_str.lower():
                # 锁定重排独立计数(不占用 max_retries 的普通重试额度),
                # 超限说明是持锁过久的真问题, 标记失败避免无限空转。
                _lock_retries = self._lock_requeue_count.get(task_id, 0)
                if _lock_retries >= 5:
                    logger.error(f"任务 {task_id} 数据库锁冲突超过 5 次, 标记失败")
                    await self._mark_task_failed(task_id, _err_str)
                    return
                self._lock_requeue_count[task_id] = _lock_retries + 1
                logger.warning(f"任务 {task_id} 数据库写锁冲突(瞬时), 延迟后重新入队 ({_lock_retries + 1}/5)")
                # 短暂退避(0.5-2s 随机)让持锁事务完成, 再重新调度
                import asyncio as _asyncio
                import random as _random
                await _asyncio.sleep(0.5 + _random.random() * 1.5)
                await self.submit_task(task_id, TaskPriority.NORMAL)
                return
            
            # 重试逻辑
            if self.retry_on_failure:
                retry_count = self.task_retries.get(task_id, 0)
                if retry_count < self.max_retries:
                    self.task_retries[task_id] = retry_count + 1
                    logger.info(f"任务 {task_id} 重试 ({retry_count + 1}/{self.max_retries})")
                    
                    # 重新提交到队列
                    await self.submit_task(task_id, TaskPriority.LOW)
                else:
                    logger.error(f"任务 {task_id} 重试次数已达上限")
                    # 标记任务为失败
                    await self._mark_task_failed(task_id, str(e))
            else:
                # 标记任务为失败
                await self._mark_task_failed(task_id, str(e))
        
        finally:
            # 从运行集合中移除
            self.running_tasks.discard(task_id)
            self.task_queue.task_done()
            
            # 尝试调度下一个任务
            await self._schedule_next()
    
    def _run_task_worker(self, task_id: int):
        """
        任务执行器（在线程池中运行）
        
        这个方法会被 main.py 中的 worker 函数覆盖
        """
        # 这个方法在 main.py 中会被覆盖
        raise NotImplementedError("Task worker not implemented")
    
    async def _mark_task_failed(self, task_id: int, error: str):
        """标记任务为失败"""
        try:
            from src.db.repository import get_db
            db = get_db()
            try:
                from src.db.models import Task
                from src.db.models import get_db_engine
                engine = get_db_engine()
                SessionLocal = session_factory(engine)
                session = SessionLocal()
                
                task = session.query(Task).filter(Task.id == task_id).first()
                if task:
                    task.status = 'FAILED'
                    task.error_message = error
                    task.completed_at = datetime.utcnow()
                    session.commit()
                    logger.info(f"任务 {task_id} 已标记为失败: {error}")
                
                session.close()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"标记任务失败时出错: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            调度器状态字典
        """
        return {
            "status": self.status.value,
            "running": len(self.running_tasks),
            "queued": self.task_queue.qsize(),
            "paused": len(self.paused_tasks),
            "max_concurrent": self.max_concurrent,
            "max_md5_concurrent": self.max_md5_concurrent,
            "max_sifting_concurrent": self.max_sifting_concurrent,
            "max_vlm_concurrent": self.max_vlm_concurrent,
            "short_task_first": self.short_task_first,
            "md5_available": self.md5_semaphore._value,
            "sifting_available": self.sifting_semaphore._value,
            "vlm_available": self.vlm_semaphore._value,
            "running_task_ids": list(self.running_tasks),
            "paused_task_ids": list(self.paused_tasks),
            # 就绪性暂停的原因（如"VLM 模型未安装"）；非就绪性暂停为空串
            "paused_reason": self._readiness_reason if self._readiness_paused else "",
            "queue_size": self.queue_size,
            "task_timeout": self.task_timeout,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries
        }
    
    def update_config(
        self,
        max_concurrent: Optional[int] = None,
        max_md5_concurrent: Optional[int] = None,
        max_sifting_concurrent: Optional[int] = None,
        max_vlm_concurrent: Optional[int] = None,
        task_timeout: Optional[int] = None,
        retry_on_failure: Optional[bool] = None,
        max_retries: Optional[int] = None
    ):
        """
        更新调度器配置
        
        Args:
            max_concurrent: 最大并发任务数
            max_md5_concurrent: MD5 计算并发数
            max_sifting_concurrent: 场景检测并发数
            max_vlm_concurrent: VLM 分析并发数
            task_timeout: 任务超时时间（秒）
            retry_on_failure: 失败时是否重试
            max_retries: 最大重试次数
        """
        if max_concurrent is not None:
            self.max_concurrent = max_concurrent
            self.semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(f"更新最大并发数: {max_concurrent}")
        
        if max_md5_concurrent is not None:
            self.max_md5_concurrent = max_md5_concurrent
            self.md5_semaphore = threading.Semaphore(max_md5_concurrent)
            logger.info(f"更新 MD5 并发数: {max_md5_concurrent}")
        
        if max_sifting_concurrent is not None:
            self.max_sifting_concurrent = max_sifting_concurrent
            self.sifting_semaphore = threading.Semaphore(max_sifting_concurrent)
            logger.info(f"更新场景检测并发数: {max_sifting_concurrent}")
        
        if max_vlm_concurrent is not None:
            self.max_vlm_concurrent = max_vlm_concurrent
            self.vlm_semaphore = threading.Semaphore(max_vlm_concurrent)
            logger.info(f"更新 VLM 并发数: {max_vlm_concurrent}")
        
        if task_timeout is not None:
            self.task_timeout = task_timeout
            logger.info(f"更新任务超时时间: {task_timeout}")
        
        if retry_on_failure is not None:
            self.retry_on_failure = retry_on_failure
            logger.info(f"更新失败重试: {retry_on_failure}")
        
        if max_retries is not None:
            self.max_retries = max_retries
            logger.info(f"更新最大重试次数: {max_retries}")


# 全局调度器实例
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """获取全局调度器实例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def set_scheduler(scheduler: TaskScheduler):
    """设置全局调度器实例"""
    global _scheduler
    _scheduler = scheduler
