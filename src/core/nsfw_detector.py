"""
NSFW 两阶段检测器

第一阶段：使用 NSFW 预筛模型快速扫描所有帧，筛选出包含 NSFW 内容的帧
第二阶段：仅对筛选出的帧调用 VLM 进行详细分析

这样可以：
1. 减少 VLM 调用次数，提升处理速度
2. 避免 VLM 看到非 NSFW 内容时触发安全过滤
3. 让 VLM 专注于分析明确的 NSFW 内容
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysis:
    """单帧分析结果"""
    frame_path: str
    is_nsfw: bool
    nsfw_score: float
    detections: List[str]  # 检测到的类别
    vlm_analysis: Optional[str] = None  # VLM 详细分析结果


@dataclass
class VideoAnalysisResult:
    """视频分析结果"""
    video_path: str
    total_frames: int
    nsfw_frames: int
    nsfw_ratio: float
    frame_results: List[FrameAnalysis]
    vlm_summary: Optional[str] = None


def sample_fallback_frames(
    frame_paths: List[str],
    max_samples: int = 12,
    per_scene: int = 3,
) -> List[str]:
    """预筛 0 检出兜底时抽取"代表性帧"做 VLM 判定。

    全部帧可能几十上百张(长视频), 全量喂 VLM 成本高; 而 0 检出场景
    里大多确为无成人内容(风景/日常/黑场), 少量覆盖各场景时间点的
    帧即可判定。路径形如 .../unit_N/frame_XXXXX.jpg 时可解析场景,
    按场景均匀抽; 无法解析时退化为全局限匀抽样。

    Args:
        frame_paths: 全部帧路径(平铺)
        max_samples: 抽样总帧数上限
        per_scene: 单场景最多抽取帧数
    """
    if len(frame_paths) <= max_samples:
        return list(frame_paths)

    # 解析 (场景序号, 帧序号) —— 兼容 unit_N/frame_M 与裸 frame_M
    import re as _re
    parsed = []
    has_unit = False
    for p in frame_paths:
        try:
            pp = Path(p)
            unit_m = _re.search(r"unit_(\d+)", str(pp.parent))
            frame_m = _re.search(r"frame_(\d+)", pp.stem)
            if unit_m:
                has_unit = True
            unit = int(unit_m.group(1)) if unit_m else 0
            fnum = int(frame_m.group(1)) if frame_m else 0
            parsed.append((unit, fnum, p))
        except Exception:
            parsed.append((0, 0, p))

    # 无任何 unit 场景信息(平铺帧目录) → 直接全局均匀取 max_samples 帧
    if not has_unit:
        step = len(frame_paths) / max_samples
        sampled = [frame_paths[int(i * step)] for i in range(max_samples)]
        logger.info(
            "0 检出兜底抽样(平铺目录): %d 帧 → %d 帧",
            len(frame_paths), len(sampled),
        )
        return sampled

    # 按场景分组并组内按帧号排序
    from collections import defaultdict
    groups = defaultdict(list)
    for unit, fnum, p in parsed:
        groups[unit].append((fnum, p))
    for g in groups.values():
        g.sort()

    # 先每场景取 1 帧(保证场景全覆盖), 再按预算给多帧场景补足到 per_scene。
    result = []
    scene_keys = sorted(groups.keys())

    # 第 1 轮: 每场景取中位帧(场景全覆盖优先)
    for unit in scene_keys:
        frames = groups[unit]
        mid = frames[len(frames) // 2][1]
        result.append(mid)

    # 第 2 轮: 若还有预算, 从帧数多的场景补帧到 per_scene
    if len(result) < max_samples:
        budget = max_samples - len(result)
        # 按场景帧数降序, 优先给帧数多的场景补
        for unit in sorted(groups.keys(), key=lambda k: -len(groups[k])):
            frames = groups[unit]
            have = sum(1 for p in result if p in [x[1] for x in frames])
            target = min(per_scene, len(frames))
            while have < target and budget > 0:
                # 均匀补: 在未选帧中取中位
                chosen = [x[1] for x in frames]
                picked = [p for p in result if p in chosen]
                remaining = [p for p in chosen if p not in picked]
                if not remaining:
                    break
                add = remaining[len(remaining) // 2]
                result.append(add)
                have += 1
                budget -= 1
            if budget <= 0:
                break

    # 兜底: 若抽样仍超限, 按全局均匀截断到 max_samples
    if len(result) > max_samples:
        step = len(result) / max_samples
        result = [result[int(i * step)] for i in range(max_samples)]

    logger.info(
        "0 检出兜底抽样: %d 帧 → %d 帧(场景数 %d, 上限 %d)",
        len(frame_paths), len(result), len(scene_keys), max_samples,
    )
    return result


def select_mapping_frames(
    unit_frames: List[str],
    scans: List[FrameAnalysis],
    frames_cap: int = 5,
    min_nsfw_frames: int = 1,
) -> Tuple[List[str], str]:
    """逐帧标签映射的选帧策略(与主分析口径一致)。

    - 检出足量 NSFW 帧 → 只送 NSFW 帧(按分数降序取前 frames_cap), 省 VLM 调用
    - 检出不足(含 320n 对男性向/男同内容系统性漏检导致的 0 检出)→ 不再丢弃
      整个场景, 而是抽样代表帧交 VLM 兜底判定, 避免该场景标签与封面双双丢失

    Args:
        unit_frames: 场景内全部帧路径(已排序)
        scans: 预筛对该场景的扫描结果
        frames_cap: 每场景送 VLM 的帧数上限
        min_nsfw_frames: 认定为"性活动场景"所需的 NSFW 帧数下限

    Returns:
        (selected_frames, mode)
        mode == "nsfw"     → 选中 NSFW 帧
        mode == "fallback" → 检出不足, 抽样代表帧交 VLM 兜底
    """
    nsfw = sorted(
        (s for s in scans if s.is_nsfw),
        key=lambda s: s.nsfw_score,
        reverse=True,
    )
    if nsfw and len(nsfw) >= min_nsfw_frames:
        return [s.frame_path for s in nsfw[:frames_cap]], "nsfw"
    return sample_fallback_frames(
        unit_frames, max_samples=frames_cap, per_scene=frames_cap
    ), "fallback"


def select_scene_frames(
    unit_frames: List[str],
    scans: List[FrameAnalysis],
    frames_cap: int = 5,
    min_nsfw_frames: int = 1,
) -> Tuple[List[str], Optional[str], str]:
    """合并分析模式的选帧 + 封面帧(每场景一次 VLM 调用)。

    在 select_mapping_frames 基础上额外交出封面帧: 场景标签将被归到该帧,
    作为按标签浏览时的封面。

    Returns:
        (selected_frames, cover_frame, mode)
        - mode == "nsfw": cover = NSFW 分数最高的帧(选中列表首项)
        - mode == "fallback": 无足量 NSFW, cover 取场景内中位帧作代表
    """
    selected, mode = select_mapping_frames(unit_frames, scans, frames_cap, min_nsfw_frames)
    if mode == "nsfw":
        cover = selected[0] if selected else None
    elif unit_frames:
        cover = unit_frames[len(unit_frames) // 2]
    else:
        cover = None
    return selected, cover, mode




class OpenNSFWDetector:
    """OpenNSFW（Yahoo open_nsfw / ONNX）NSFW 检测器 —— 可商用。

    许可：封装库 Apache-2.0，权重 Yahoo open_nsfw（BSD-2-Clause），模型随包内置、ONNX 推理
    （支持 CoreML）。输出**二分类 P(nsfw)**（无细分类别标签）。
    接口：`detect_frame()` / `scan_frames()`。
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._clf = None

    def _init(self):
        if self._clf is None:
            from opennsfw_onnx import NSFWClassifier
            self._clf = NSFWClassifier(threshold=self.threshold)
            try:
                self._clf.warmup()
            except Exception:
                pass

    def detect_frame(self, frame_path: str) -> Tuple[bool, float, List[str]]:
        self._init()
        try:
            score = float(self._clf.classify(frame_path).nsfw)
            return score >= self.threshold, score, []
        except Exception as e:
            logger.error(f"OpenNSFW 检测失败 {frame_path}: {e}")
            return False, 0.0, []

    def scan_frames(self, frame_paths: List[str]) -> List["FrameAnalysis"]:
        results = []
        for frame_path in frame_paths:
            is_nsfw, nsfw_score, detections = self.detect_frame(frame_path)
            results.append(FrameAnalysis(
                frame_path=frame_path,
                is_nsfw=is_nsfw,
                nsfw_score=nsfw_score,
                detections=detections,
            ))
        return results


def create_nsfw_detector(backend: Optional[str] = None, threshold: float = 0.5):
    """按配置创建 NSFW 检测器后端（**可回退**）。

    backend:
      - ``"opennsfw"``（**默认**，可商用：Apache-2.0 封装 + Yahoo open_nsfw BSD-2-Clause 权重）
      - ``"none"``     （跳过阶段 1，直接走 VLM 兜底判定）

    取自 config ``nsfw.detector``；缺配置时默认 ``opennsfw``。
    （历史后端 NudeNet 因 **AGPL-3.0** 已移除，不再随包分发。）
    """
    if backend is None:
        try:
            from src.core.config import ConfigManager
            backend = ConfigManager().get("nsfw.detector", "opennsfw")
        except Exception:
            backend = "opennsfw"
    backend = (backend or "opennsfw").lower()
    if backend == "none":
        return None
    return OpenNSFWDetector(threshold=threshold)


class TwoStageNSFWAnalyzer:
    """两阶段 NSFW 分析器"""
    
    def __init__(
        self,
        nsfw_threshold: float = 0.5,
        vlm_client=None,
        config_manager=None
    ):
        """
        初始化两阶段分析器

        Args:
            nsfw_threshold: NSFW 预筛阈值
            vlm_client: VLM 客户端实例
            config_manager: 配置管理器
        """
        self.detector = create_nsfw_detector(
            (config_manager.get("nsfw.detector", "opennsfw") if config_manager is not None else None),
            threshold=nsfw_threshold,
        )
        self.vlm_client = vlm_client
        self.config_manager = config_manager
    
    def analyze(
        self,
        frame_paths: List[str],
        progress_callback=None,
        theme_name: Optional[str] = None,
    ) -> VideoAnalysisResult:
        """
        执行两阶段分析
        
        Args:
            frame_paths: 帧图片路径列表
            progress_callback: 进度回调函数 callback(current, total, message)
            theme_name: 文件所属主题名，用于主题↔提示词联动
            
        Returns:
            VideoAnalysisResult
        """
        if not frame_paths:
            raise ValueError("帧列表为空")
        
        total_frames = len(frame_paths)
        logger.info(f"开始两阶段分析，共 {total_frames} 帧")
        
        # 阶段 1: NSFW 预筛（后端可配置：opennsfw / none）
        if progress_callback:
            progress_callback(0, 100, "阶段 1/2: NSFW 预筛中...")

        if self.detector is not None:
            frame_results = self.detector.scan_frames(frame_paths)
        else:
            # nsfw.detector=none → 跳过阶段 1，直接进入 VLM 兜底判定
            frame_results = []
            logger.info("NSFW 检测器已停用（nsfw.detector=none），直接进入 VLM 兜底判定")
        
        # 统计 NSFW 帧
        nsfw_frames = [fr for fr in frame_results if fr.is_nsfw]
        nsfw_count = len(nsfw_frames)
        nsfw_ratio = nsfw_count / total_frames if total_frames > 0 else 0.0
        
        logger.info(f"NSFW 预筛完成: {nsfw_count}/{total_frames} 帧为 NSFW ({nsfw_ratio:.1%})")
        
        # 阶段 2: VLM 详细分析（仅对 NSFW 帧，动态分批处理避免超出上下文限制）
        vlm_summary = None
        
        if nsfw_count > 0 and self.vlm_client:
            if progress_callback:
                progress_callback(50, 100, f"阶段 2/2: VLM 分析 {nsfw_count} 帧...")
            
            # 提取 NSFW 帧路径
            nsfw_frame_paths = [fr.frame_path for fr in nsfw_frames]
            
            # 动态计算每批帧数：基于帧文件大小估算 token 占用
            # 经验公式：1KB 图片 ≈ 100-150 tokens（考虑视觉编码开销）
            # 预留 5000 tokens 给 prompt 和输出，剩余分配给图片
            max_context_tokens = 35000  # 安全上限（低于 40960）
            reserved_tokens = 5000  # 预留给 prompt 和输出
            available_tokens = max_context_tokens - reserved_tokens
            
            # 计算每帧平均大小（KB）
            total_size_kb = sum(Path(p).stat().st_size for p in nsfw_frame_paths) / 1024
            avg_size_kb = total_size_kb / len(nsfw_frame_paths)
            
            # 估算每帧 token 数（1KB ≈ 120 tokens）
            tokens_per_frame = int(avg_size_kb * 120)
            
            # 计算动态 batch_size
            dynamic_batch_size = max(1, available_tokens // tokens_per_frame)
            # 限制在 3-15 帧之间，避免极端情况
            batch_size = min(15, max(3, dynamic_batch_size))
            
            logger.info(f"帧大小统计: 平均 {avg_size_kb:.1f}KB, 估算 {tokens_per_frame} tokens/帧, 动态批次大小: {batch_size}")
            
            vlm_summary = self._run_vlm_batches(
                nsfw_frame_paths, theme_name=theme_name, progress_callback=progress_callback
            )
            
            # 将 VLM 结果写入对应的 FrameAnalysis
            if vlm_summary:
                for fr in nsfw_frames:
                    fr.vlm_analysis = vlm_summary
        
        elif nsfw_count == 0 and self.vlm_client:
            # 预筛 0 检出不一定代表无成人内容（对男性向/男同性内容易漏），
            # 系统性漏检（实测 Gay 硬核视频全帧 0 检出）。此时降级让 VLM 兜底
            # 判定——但不必全量分析所有帧: 0 检出场景多为无成人内容,
            # 按场景/时间抽样代表帧即可覆盖; 抽样帧足够 VLM 判断是否成人。
            if progress_callback:
                progress_callback(50, 100, "阶段 2/2: 预筛未检出，VLM 兜底判定中...")
            
            _fallback_frames = self._sample_fallback_frames(frame_paths)
            logger.info("预筛未检出 NSFW，降级由 VLM 兜底判定 %d 帧", len(_fallback_frames))
            vlm_summary = self._run_vlm_batches(
                _fallback_frames, theme_name=theme_name, progress_callback=progress_callback
            )
            if not vlm_summary:
                logger.info("VLM 兜底判定无返回，视为无成人内容")
                vlm_summary = None
        
        if progress_callback:
            progress_callback(100, 100, "分析完成")
        
        return VideoAnalysisResult(
            video_path="",  # 由调用者设置
            total_frames=total_frames,
            nsfw_frames=nsfw_count,
            nsfw_ratio=nsfw_ratio,
            frame_results=frame_results,
            vlm_summary=vlm_summary
        )

    def _sample_fallback_frames(
        self,
        frame_paths: List[str],
        max_samples: int = 12,
        per_scene: int = 3,
    ) -> List[str]:
        """预筛 0 检出兜底抽样(实现见模块级 sample_fallback_frames)。"""
        # 闸 3: 垃圾帧(黑场/糊/废)先剔除——0 检出场景常见黑场,
        # 喂给 VLM 只会产出噪声标签(2026-09-30 审计实证)
        enabled = True
        try:
            if self.config_manager is not None:
                enabled = bool(self.config_manager.get("analysis.frame_quality_filter", True))
        except Exception:
            enabled = True
        if enabled:
            from src.core.frame_quality import filter_frames
            frame_paths, _dropped = filter_frames(frame_paths)
            if not frame_paths:
                logger.info("兜底抽样: 全部帧为垃圾帧, 跳过 VLM 兜底判定")
                return []
        return sample_fallback_frames(frame_paths, max_samples, per_scene)

    def _run_vlm_batches(
        self,
        frame_paths: List[str],
        theme_name: Optional[str] = None,
        progress_callback=None,
    ) -> Optional[str]:
        """对帧列表分批调用 VLM，返回合并后的分析文本。

        预筛检出 NSFW 帧(主路径)和 0 检出兜底(降级路径)共用此方法。
        动态分批避免一次塞入过多图片超出 VLM 上下文窗口。
        """
        if not frame_paths or not self.vlm_client:
            return None

        import tempfile
        import shutil

        # 动态计算每批帧数：基于帧文件大小估算 token 占用
        # 经验公式：1KB 图片 ≈ 100-150 tokens（考虑视觉编码开销）
        max_context_tokens = 35000  # 安全上限（低于 40960）
        reserved_tokens = 5000  # 预留给 prompt 和输出
        available_tokens = max_context_tokens - reserved_tokens

        try:
            total_size_kb = sum(Path(p).stat().st_size for p in frame_paths) / 1024
            avg_size_kb = total_size_kb / len(frame_paths)
        except OSError:
            avg_size_kb = 50.0  # 文件不可读时用保守默认值

        tokens_per_frame = int(avg_size_kb * 120)
        dynamic_batch_size = max(1, available_tokens // tokens_per_frame) if tokens_per_frame > 0 else 15
        # 限制在 3-15 帧之间，避免极端情况
        batch_size = min(15, max(3, dynamic_batch_size))

        logger.info(f"帧大小统计: 平均 {avg_size_kb:.1f}KB, 估算 {tokens_per_frame} tokens/帧, 动态批次大小: {batch_size}")

        all_vlm_results = []

        try:
            for i in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(frame_paths) + batch_size - 1) // batch_size

                logger.info(f"VLM 分析批次 {batch_num}/{total_batches}，帧数: {len(batch_paths)}")

                # 创建临时目录包含当前批次的帧
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 复制帧文件到临时目录
                    for frame_path in batch_paths:
                        src = Path(frame_path)
                        dst = Path(temp_dir) / src.name
                        try:
                            shutil.copy2(src, dst)
                        except OSError:
                            continue

                    # 调用 VLM 分析当前批次（按主题联动提示词）
                    batch_result = self.vlm_client.get_analysis(temp_dir, theme_name=theme_name)

                    if batch_result:
                        all_vlm_results.append(batch_result)
                        logger.info(f"批次 {batch_num} 分析完成，结果长度: {len(batch_result)}")

                # 更新进度
                if progress_callback:
                    progress = 50 + int((i + len(batch_paths)) / len(frame_paths) * 40)
                    progress_callback(progress, 100, f"VLM 分析中... ({i + len(batch_paths)}/{len(frame_paths)} 帧)")

            # 合并所有批次的结果
            if all_vlm_results:
                vlm_summary = "\n".join(all_vlm_results)
                logger.info(f"VLM 分析完成，总结果长度: {len(vlm_summary)}")
                return vlm_summary
            else:
                logger.warning("所有批次 VLM 分析均返回空结果")
                return None

        except Exception as e:
            logger.error(f"VLM 分析失败: {e}")
            return None
