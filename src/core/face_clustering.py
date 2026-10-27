"""Face clustering module.

使用 DBSCAN 算法进行人脸聚类，对噪声更鲁棒。
支持动态参数选择，根据视频特征自动调整聚类参数。
支持自监督参数选择，无需人工标注即可自动选择最优参数。
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.core.face_thresholds import DEFAULT_DISTANCE_THRESHOLD, DEFAULT_MIN_SAMPLES

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available — face clustering disabled")


def get_dynamic_params(video_features: Dict) -> Dict:
    """根据视频特征动态选择聚类参数

    阈值从 config.yaml 的 `face_recognition.dynamic_params` 读取，
    未配置时使用默认值。

    Args:
        video_features: 视频特征字典，包含以下键：
            - duration: 视频时长（秒）
            - n_faces: 人脸总数
            - n_scenes: 场景数量
            - face_density: 人脸密度（张/秒）

    Returns:
        包含 eps 和 min_samples 的参数字典
    """
    # 从 config 读取阈值（未配置时用默认值）
    try:
        from src.core.config import ConfigManager
        _dp = ConfigManager().get("face_recognition.dynamic_params", {}) or {}
        _density_low = _dp.get("density_thresholds", {}).get("low", 0.1)
        _density_med = _dp.get("density_thresholds", {}).get("medium", 0.3)
        _scene_low = _dp.get("scene_thresholds", {}).get("low", 10)
        _scene_med = _dp.get("scene_thresholds", {}).get("medium", 30)
        _eps_range = _dp.get("eps_range", {})
        _eps_min = _eps_range.get("min", 0.3)
        _eps_max = _eps_range.get("max", 0.5)
        _ms_range = _dp.get("min_samples_range", {})
        _ms_min = _ms_range.get("min", 3)
        _ms_max = _ms_range.get("max", 10)
    except Exception:
        _density_low, _density_med = 0.1, 0.3
        _scene_low, _scene_med = 10, 30
        _eps_min, _eps_max = 0.3, 0.5
        _ms_min, _ms_max = 3, 10

    n_faces = video_features.get('n_faces', 0)
    n_scenes = video_features.get('n_scenes', 0)
    face_density = video_features.get('face_density', 0)

    # 1. 基于人脸密度选择eps
    # 人脸密度低时，允许更大的距离阈值
    if face_density < _density_low:
        eps = _eps_max
    elif face_density < _density_med:
        eps = (_eps_min + _eps_max) / 2
    else:
        eps = _eps_min

    # 2. 基于场景数量选择min_samples
    # 场景少时，减少最小样本数要求
    if n_scenes < _scene_low:
        min_samples = _ms_min
    elif n_scenes < _scene_med:
        min_samples = int((_ms_min + _ms_max) / 2)
    else:
        min_samples = _ms_max

    # 3. 基于人脸总数调整
    # 人脸总数多时，收紧参数以提高精度
    if n_faces > 5000:
        eps = max(_eps_min, eps - 0.1)
        min_samples = max(_ms_min, min_samples)

    logger.info(
        "动态参数选择: face_density=%.3f, n_scenes=%d, n_faces=%d -> eps=%.2f, min_samples=%d",
        face_density, n_scenes, n_faces, eps, min_samples
    )

    return {
        'eps': eps,
        'min_samples': min_samples
    }


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> float:
    """评估聚类质量（使用轮廓系数）
    
    Args:
        features: 特征矩阵
        labels: 聚类标签
    
    Returns:
        轮廓系数，范围[-1, 1]，越高越好
    """
    if not _SKLEARN_AVAILABLE:
        return -1.0
    
    # 检查是否有有效的聚类
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 排除噪声
    
    if len(unique_labels) < 2:
        return -1.0  # 无效聚类（需要至少2个簇）
    
    # 排除噪声点
    mask = labels != -1
    if mask.sum() < 2:
        return -1.0
    
    try:
        score = silhouette_score(features[mask], labels[mask])
        return score
    except Exception as e:
        logger.warning("轮廓系数计算失败: %s", e)
        return -1.0


def auto_select_params(features: np.ndarray, param_grid: Optional[Dict] = None) -> Tuple[Dict, float]:
    """自监督参数选择：自动选择最优的DBSCAN参数
    
    Args:
        features: 特征矩阵
        param_grid: 参数网格，可选。默认测试多个参数组合
    
    Returns:
        (最优参数, 最优得分) 的元组
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("sklearn not available, returning default params")
        return {'eps': 0.5, 'min_samples': 3}, -1.0
    
    # 默认参数网格
    if param_grid is None:
        param_grid = {
            'eps': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
            'min_samples': [3, 5, 8, 10, 15]
        }
    
    best_score = -1.0
    best_params = {'eps': 0.5, 'min_samples': 3}
    
    total_combinations = len(param_grid['eps']) * len(param_grid['min_samples'])
    logger.info("开始自监督参数选择，共 %d 个参数组合", total_combinations)
    
    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            # 执行聚类
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clustering.fit_predict(features)
            
            # 评估
            score = evaluate_clustering(features, labels)
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
    
    logger.info(
        "自监督参数选择完成: eps=%.2f, min_samples=%d, 轮廓系数=%.4f",
        best_params['eps'], best_params['min_samples'], best_score
    )
    
    return best_params, best_score


def group_faces_into_tracks(
    faces: List[Dict],
    max_gap_frames: Optional[int] = None,
    emb_thresh: float = 0.55,
    keep_per_track: int = 2,
    max_gap_factor: float = 2.0,
) -> List[Dict]:
    """把人脸按"时间邻近 + 特征相近"串成轨迹，每条轨迹最多保留 ``keep_per_track`` 张代表脸。

    为什么需要（见 docs/架构/人脸识别-验证样本登记.md 的验收表）：
    聚类的最小单位应当是"一个人在一次出场里的一段"，而不是"每一张脸"。
    实测：采样加密后脸数从 15 涨到 54，固定阈值下能形成的簇也随之变多 ——
    **从正确的 2 个人物碎成 6 个**。

    判据为什么用 embedding 而不是框重叠（IoU）：
    先用 IoU 实现过一版，**实测无效**（54 张脸只并到 50 张）——2 秒间隔里人与镜头都在动，
    相邻样本的框重叠经常低于阈值，轨迹被打断成一段段。
    而实测数据支持用特征：**≤2 秒间隔的同人距离约 0.31，跨人约 0.855**，
    取 0.55 即可在短时间窗内干净分开 → 因此判据 = "时间邻近 **且** 特征距离 ≤ 阈值"。

    ``max_gap_frames`` 未给定时按输入自身的采样间隔中位数 × ``max_gap_factor`` 推导
    —— 轨迹窗口随采样密度自适应。

    注意：本函数只做"归并"，不做身份判定；跨出场段的同一人仍由后续聚类/匹配处理。

    Args:
        faces: 人脸列表，需含 ``frame_id`` 与 ``embedding``；缺失者各自独占一条轨迹
        max_gap_frames: 轨迹内允许的相邻样本帧号间隔；None → 按采样中位间隔推导
        emb_thresh: 归入同一轨迹所需的余弦距离上限（短时间窗内同人≈0.31 / 跨人≈0.855）
        keep_per_track: 每条轨迹保留的代表脸数（按 ``quality_score``/``det_score`` 降序取）
        max_gap_factor: 推导 ``max_gap_frames`` 时的系数

    Returns:
        代表脸列表（原 dict 的浅拷贝，附 ``track_id`` 与 ``track_size``），按时间排序
    """
    if not faces:
        return faces

    ordered = sorted(faces, key=lambda f: (f.get("frame_id") is None, f.get("frame_id") or 0))

    # 推导轨迹窗口：输入自身的采样间隔中位数 × 系数
    frame_ids = sorted({int(f["frame_id"]) for f in ordered if f.get("frame_id") is not None})
    if max_gap_frames is None:
        if len(frame_ids) >= 2:
            gaps = [b - a for a, b in zip(frame_ids, frame_ids[1:]) if b > a]
            median_gap = float(np.median(gaps)) if gaps else 0.0
        else:
            median_gap = 0.0
        max_gap_frames = max(1, int(round(median_gap * max_gap_factor)))

    def _unit(f: Dict):
        emb = f.get("embedding")
        if emb is None:
            return None
        v = np.asarray(emb, dtype=float)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def _rank(f: Dict) -> float:
        return float(f.get("quality_score") or 0.0) or float(f.get("det_score") or 0.0)

    tracks: List[List[Dict]] = []
    last_frame: List[Optional[int]] = []
    last_unit: List[Optional[np.ndarray]] = []

    for f in ordered:
        fid = f.get("frame_id")
        unit = _unit(f)
        best_i, best_dist = -1, emb_thresh
        if fid is not None and unit is not None:
            for i, (lf, lu) in enumerate(zip(last_frame, last_unit)):
                if lf is None or lu is None:
                    continue
                if abs(int(fid) - int(lf)) > max_gap_frames:
                    continue
                d = 1.0 - float(unit @ lu)
                if d <= best_dist:
                    best_i, best_dist = i, d
        if best_i >= 0:
            tracks[best_i].append(f)
            last_frame[best_i] = int(fid)
            last_unit[best_i] = unit
        else:
            tracks.append([f])
            last_frame.append(int(fid) if fid is not None else None)
            last_unit.append(unit)

    kept: List[Dict] = []
    for tid, track in enumerate(tracks):
        chosen = sorted(track, key=_rank, reverse=True)[:max(1, int(keep_per_track))]
        for c in sorted(chosen, key=lambda x: (x.get("frame_id") or 0)):
            out = dict(c)
            out["track_id"] = tid
            out["track_size"] = len(track)
            kept.append(out)

    kept.sort(key=lambda f: (f.get("frame_id") or 0))
    logger.info(
        "轨迹分组: %d 张脸 → %d 条轨迹 → 保留 %d 张代表脸 (窗口 %d 帧, 特征距离<=%.2f, 每轨<=%d)",
        len(faces), len(tracks), len(kept), max_gap_frames, emb_thresh, keep_per_track,
    )
    return kept


class FaceClusterer:
    """人脸聚类器：将检测到的人脸分组为不同人物

    使用 DBSCAN 算法：
    - 基于密度，对噪声更鲁棒
    - 不需要预先指定簇数
    - 自动识别噪声点
    
    支持动态参数选择：
    - 根据视频特征自动调整聚类参数
    - 适应不同类型的视频内容
    
    支持自监督参数选择：
    - 无需人工标注
    - 使用轮廓系数自动选择最优参数
    - 适用于完全未知的视频
    """

    def __init__(
        self,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        use_dynamic_params: bool = False,
        use_auto_params: bool = False,
        track_config: Optional[Dict] = None,
    ):
        """初始化聚类器

        Args:
            distance_threshold: DBSCAN 的 eps 参数（余弦距离）
            min_samples: DBSCAN 的 min_samples 参数（最小样本数）
            use_dynamic_params: 是否使用动态参数选择
            use_auto_params: 是否使用自监督参数选择
            track_config: 轨迹分组配置（enabled/max_gap_frames/iou_thresh/keep_per_track/
                max_gap_factor）。**默认关闭 = 与改动前行为完全一致。**
        """
        self.distance_threshold = distance_threshold
        self.min_samples = min_samples
        self.use_dynamic_params = use_dynamic_params
        self.use_auto_params = use_auto_params
        self._track_config = dict(track_config or {})
        self._video_features = None
        self._auto_score = None
        # 聚类质量信号（cluster_faces 后写入）：
        #   quality_tier: 'good' / 'medium' / 'poor'（embedding 质量分档）
        #   confidence: 0-1，聚类结果置信度（低置信意味着结果需人工复核）
        #   intra_median: 簇内距离中位（主要质量信号，无需人工标注）
        self.quality_tier: Optional[str] = None
        self.confidence: Optional[float] = None
        self.intra_median: Optional[float] = None
    
    def set_video_features(self, video_features: Dict) -> None:
        """设置视频特征，用于动态参数选择
        
        Args:
            video_features: 视频特征字典
        """
        self._video_features = video_features
        
        if self.use_dynamic_params and video_features:
            params = get_dynamic_params(video_features)
            self.distance_threshold = params['eps']
            self.min_samples = params['min_samples']
            logger.info(
                "动态更新参数: eps=%.2f, min_samples=%d",
                self.distance_threshold, self.min_samples
            )

    def cluster_faces(self, faces: List[Dict]) -> List[List[Dict]]:
        """将人脸列表聚类为多个人物组

        Args:
            faces: 检测到的人脸列表，每个包含 'embedding' 键（np.ndarray）

        Returns:
            聚类结果，每个子列表代表一个人物的人脸集合。
        """
        if not faces:
            return []

        # 轨迹分组：先把"时间连续 + 位置重合"的脸归并为轨迹、每条轨迹只留少量代表脸，
        # 再按代表脸聚类。目的：压住"碎"（输入点数变少）+ 提升代表脸稳定性。
        # 默认关闭（_track_config 为空）→ 行为与改动前一致。
        if self._track_config.get("enabled"):
            before = len(faces)
            faces = group_faces_into_tracks(
                faces,
                max_gap_frames=self._track_config.get("max_gap_frames"),
                emb_thresh=float(self._track_config.get("emb_thresh", 0.55)),
                keep_per_track=int(self._track_config.get("keep_per_track", 2)),
                max_gap_factor=float(self._track_config.get("max_gap_factor", 2.0)),
            )
            logger.info("轨迹分组: %d 张脸 → %d 张代表脸参与聚类", before, len(faces))
            if not faces:
                return []

        if not _SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, returning all faces as single group")
            return [faces]

        if len(faces) == 1:
            logger.info("Only 1 face detected, treating as single actor")
            return [faces]

        # 提取 embeddings
        E = np.array([f['embedding'] for f in faces], dtype=float)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        valid = norms[:, 0] > 0
        if not valid.all():
            logger.warning("%d 个零向量 embedding 被剔除", int((~valid).sum()))
            faces = [f for f, v in zip(faces, valid) if v]
            E = E[valid]
        
        if len(faces) == 1:
            return [faces]
        
        # 人脸数不足 min_samples 时跳过聚类（DBSCAN 无法形成有效簇）
        if len(faces) < self.min_samples:
            logger.info(
                "人脸数 %d < min_samples %d，跳过聚类，每张脸单独成簇",
                len(faces), self.min_samples
            )
            return [[f] for f in faces]
        
        # 归一化
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        
        # 计算余弦距离矩阵
        S = En @ En.T
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        D = np.clip(D, 0.0, 2.0)
        
        # 自监督参数选择
        if self.use_auto_params:
            logger.info("使用自监督参数选择")
            auto_params, self._auto_score = auto_select_params(D)
            self.distance_threshold = auto_params['eps']
            self.min_samples = auto_params['min_samples']
            logger.info(
                "自监督参数选择结果: eps=%.2f, min_samples=%d, 轮廓系数=%.4f",
                self.distance_threshold, self.min_samples, self._auto_score
            )
        
        logger.info(
            "DBSCAN 聚类: eps=%.3f, min_samples=%d, 距离分布: min=%.3f, median=%.3f, max=%.3f",
            self.distance_threshold, self.min_samples,
            D[D > 0].min() if len(D[D > 0]) > 0 else 0.0,
            np.median(D[D > 0]) if len(D[D > 0]) > 0 else 0.0,
            D[D > 0].max() if len(D[D > 0]) > 0 else 0.0
        )
        
        # DBSCAN 聚类（eps 自适应兜底 + 质量分档策略）
        # 若当前 eps 过紧导致噪声占比过高（如 config 里 eps=0.3 对距离中位 0.66 的
        # 低画质视频），则基于实际距离分布放宽 eps 重试，避免"人脸全判噪声、颗粒无收"。
        #
        # 分档策略（避免"低画质就出错"）：
        # - 用"全脸距离中位"预估 embedding 质量，决定 eps 放宽的上限：
        #   * 距离中位低（embedding 好）→ 可放宽到 0.55 以内（同人可并、跨人可分）
        #   * 距离中位高（embedding 差）→ 收紧放宽上限到 0.45，宁碎勿混——
        #     碎片化可后期合并修复，混人则污染人物库难修复
        # - 放宽受"过度合并检测"约束：最大簇内距离中位超档位上限则回退
        max_retries = 3
        labels = None
        best_labels = None  # 记录最近一次"未过度合并"的结果

        # 预估 embedding 质量：距离分布 p30 分位（低=好，高=差）
        # 注意：不能用全脸距离中位——它被跨人距离污染（多人视频中位接近 0.9），
        # 会误判优质视频为 poor。p30 接近真实"同人距离"（实测与人工标注吻合：
        # test_video 0.386 vs 0.268 / test_video_3 0.662 vs 0.664 / test_video_4 0.704 vs 0.681）
        dist_vals_all = D[D > 0]
        global_p30 = float(np.percentile(dist_vals_all, 30)) if len(dist_vals_all) > 0 else 0.5
        # 分档：<0.45 优质 / <0.68 中等 / >=0.68 低质（与 test_video/3/4 实测校准）
        if global_p30 < 0.45:
            quality_tier = 'good'
            eps_cap = 0.55   # 可放宽，同人可并
            conf_base = 0.9
        elif global_p30 < 0.68:
            quality_tier = 'medium'
            eps_cap = 0.48   # 保守放宽，宁碎勿混
            conf_base = 0.65
        else:
            quality_tier = 'poor'
            eps_cap = 0.42   # 极保守，只并非常近的脸
            conf_base = 0.4
        logger.info(
            "质量分档: 距离p30=%.3f -> %s (eps 上限 %.2f)",
            global_p30, quality_tier, eps_cap,
        )

        # 初始 eps 超过档位上限 → clamp 到上限（auto_params/动态参数可能选得过于宽松，
        # 在低质 embedding 上轮廓系数会被欺骗，必须受分档上限约束）
        if self.distance_threshold > eps_cap:
            logger.warning(
                "质量分档: eps %.3f 超过 %s 档上限 %.2f，clamp 到 %.2f",
                self.distance_threshold, quality_tier, eps_cap, eps_cap,
            )
            self.distance_threshold = eps_cap
        eps_backup = self.distance_threshold

        for attempt in range(max_retries):
            clustering = DBSCAN(
                eps=self.distance_threshold,
                min_samples=self.min_samples,
                metric='precomputed'
            )
            labels = clustering.fit_predict(D)
            n_noise = int(list(labels).count(-1))
            noise_ratio = n_noise / len(faces) if faces else 1.0

            # 过度合并检测：最大簇的簇内距离中位超过档位上限则判定混人
            largest_cluster_inner_median = None
            cluster_sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.array([])
            if len(cluster_sizes) > 0:
                largest_label = int(np.argmax(cluster_sizes))
                member_idx = np.where(labels == largest_label)[0]
                if len(member_idx) >= 3:
                    sub = D[np.ix_(member_idx, member_idx)]
                    vals = sub[sub > 0]
                    if len(vals):
                        largest_cluster_inner_median = float(np.median(vals))

            # 性别互斥检测：簇内同时出现两种明确性别 → 判定混人。
            # 性别是独立于 embedding 的强先验（genderage 模型输出），同一人性别恒定。
            # 仅在 poor 档启用（embedding 不可靠时性别兜底价值才大；good/medium 档
            # embedding 已足够可靠，避免误判干扰正常聚类）。
            # 阈值：少数性别占比 >= 25% 且 >= 5 张（低画质下 genderage 有 ~16% 误判率，
            # 少量误判不应触发；真混人时两性比例通常接近）。
            sex_conflict = False
            if quality_tier == 'poor':
                try:
                    for lab in set(labels.tolist()):
                        if lab == -1:
                            continue
                        mem = [faces[i].get("sex") for i in np.where(labels == lab)[0]]
                        cnt_m = sum(1 for s in mem if s == "M")
                        cnt_f = sum(1 for s in mem if s == "F")
                        total = cnt_m + cnt_f
                        if total >= 8 and cnt_m >= 5 and cnt_f >= 5:
                            ratio = min(cnt_m, cnt_f) / total
                            if ratio >= 0.25:
                                sex_conflict = True
                                logger.warning(
                                    "性别互斥检测: 簇%d 同时含 M×%d 和 F×%d (占比 %.0f%%)，判定混人",
                                    lab, cnt_m, cnt_f, ratio * 100,
                                )
                                break
                except Exception:
                    pass

            over_merged = (
                (largest_cluster_inner_median is not None and largest_cluster_inner_median > eps_cap)
                or sex_conflict
            )
            if over_merged:
                logger.warning(
                    "质量分档兜底: 过度合并检测触发 (簇内距离中位=%s > 上限 %.2f, 性别冲突=%s)，"
                    "回退到 eps=%.3f",
                    f"{largest_cluster_inner_median:.3f}" if largest_cluster_inner_median is not None else "N/A",
                    eps_cap, sex_conflict, eps_backup,
                )
                labels = best_labels if best_labels is not None else labels
                break

            if noise_ratio <= 0.3 or attempt == max_retries - 1:
                break

            best_labels = labels
            # 放宽 eps：受分档上限约束（低质视频收紧，避免混人）
            # 参照 p30（同人距离估计）而非全脸中位（被跨人距离污染）
            new_eps = min(max(self.distance_threshold + 0.05, global_p30 * 0.9), eps_cap)
            if new_eps <= self.distance_threshold:
                # 已到上限，无法再放宽
                break
            logger.warning(
                "质量分档兜底: 噪声占比 %.0f%% 过高, eps %.3f -> %.3f (距离p30 %.3f, 档位上限 %.2f)",
                noise_ratio * 100, self.distance_threshold, new_eps, global_p30, eps_cap,
            )
            self.distance_threshold = new_eps
        
        # 按标签分组（忽略噪声点 -1）
        clusters: Dict[int, List[Dict]] = {}
        for face, label in zip(faces, labels):
            if label == -1:
                continue  # 跳过噪声点
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(face)
        
        result = list(clusters.values())
        
        # 按簇大小排序（降序）
        result.sort(key=len, reverse=True)
        
        # ── 聚类质量信号（供主流程决定是否标记"需人工复核"）──
        # 簇内距离中位（所有簇的加权中位）：直接反映 embedding 质量
        intra_medians = []
        for c in result:
            idx = [i for i, f in enumerate(faces) if any(f is cf for cf in c)]
            if len(idx) >= 3:
                sub = D[np.ix_(idx, idx)]
                vals = sub[sub > 0]
                if len(vals):
                    intra_medians.append(float(np.median(vals)))
        self.intra_median = float(np.median(intra_medians)) if intra_medians else global_p30
        self.quality_tier = quality_tier
        # 置信度：基础档位分 × 噪声惩罚 × 碎片化惩罚
        noise_penalty = 1.0 - 0.4 * noise_ratio
        self.confidence = round(max(0.05, min(1.0, conf_base * noise_penalty)), 3)
        logger.info(
            "聚类质量: tier=%s, 簇内距离中位=%.3f, 噪声比=%.2f, 置信度=%.2f",
            quality_tier, self.intra_median, noise_ratio, self.confidence,
        )
        
        sizes = [len(c) for c in result]
        n_noise = list(labels).count(-1)
        logger.info(
            "DBSCAN 聚类完成: %d 簇, 噪声点 %d, top10=%s",
            len(result), n_noise, sizes[:10]
        )
        
        return result
    
    def get_auto_score(self) -> Optional[float]:
        """获取自监督参数选择的轮廓系数
        
        Returns:
            轮廓系数，如果未使用自监督参数选择则返回None
        """
        return self._auto_score
