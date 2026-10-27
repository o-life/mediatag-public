"""人脸聚类评估模块

实现纯度、覆盖率、误检率、碎片化度等评估指标，
符合《人脸识别系统过程控制方案》的要求。
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available — evaluation metrics disabled")


def calculate_purity(true_labels: List[str], pred_labels: List[int]) -> float:
    """计算纯度：actor内同人比例
    
    纯度 = 每个预测簇中，最主要真实标签的比例之和 / 总样本数
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表（-1表示噪声）
    
    Returns:
        纯度值，范围[0, 1]，越高越好
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("标签长度不一致")
    
    # 排除噪声点
    mask = pred_labels != -1
    if mask.sum() == 0:
        return 0.0
    
    true_labels = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
    pred_labels = pred_labels[mask]
    
    # 计算每个簇的纯度
    purity = 0.0
    for cluster_id in set(pred_labels):
        cluster_mask = pred_labels == cluster_id
        cluster_true = [true_labels[i] for i in range(len(true_labels)) if cluster_mask[i]]
        
        if cluster_true:
            # 最主要标签的数量
            most_common = Counter(cluster_true).most_common(1)[0][1]
            purity += most_common
    
    return purity / len(true_labels)


def calculate_coverage(true_labels: List[str], pred_labels: List[int]) -> float:
    """计算覆盖率：真实人物被找回比例
    
    覆盖率 = 至少有一个样本被正确聚类的真实人物数 / 总真实人物数
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表（-1表示噪声）
    
    Returns:
        覆盖率值，范围[0, 1]，越高越好
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("标签长度不一致")
    
    # 获取所有真实人物
    true_actors = set(true_labels)
    
    # 排除噪声点
    mask = pred_labels != -1
    if mask.sum() == 0:
        return 0.0
    
    # 统计每个真实人物是否被找回
    found_actors = set()
    for i in range(len(true_labels)):
        if mask[i] and pred_labels[i] != -1:
            found_actors.add(true_labels[i])
    
    return len(found_actors) / len(true_actors)


def calculate_false_positive_rate(true_labels: List[str], pred_labels: List[int]) -> float:
    """计算误检率：非脸进入actor的比例
    
    误检率 = 被错误分配到actor的样本数 / 总样本数
    
    Args:
        true_labels: 真实标签列表（'unknown'表示非脸）
        pred_labels: 预测标签列表（-1表示噪声）
    
    Returns:
        误检率，范围[0, 1]，越低越好
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("标签长度不一致")
    
    # 排除噪声点
    mask = pred_labels != -1
    if mask.sum() == 0:
        return 0.0
    
    # 统计错误分配的样本
    false_positives = 0
    for i in range(len(true_labels)):
        if mask[i] and true_labels[i] == 'unknown':
            false_positives += 1
    
    return false_positives / mask.sum()


def calculate_fragmentation(true_labels: List[str], pred_labels: List[int]) -> float:
    """计算碎片化度：同一人被拆成的actor数
    
    碎片化度 = 每个真实人物被拆成的簇数的平均值
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表（-1表示噪声）
    
    Returns:
        碎片化度，范围[1, +∞)，越低越好（1表示没有碎片化）
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("标签长度不一致")
    
    # 排除噪声点
    mask = pred_labels != -1
    if mask.sum() == 0:
        return 1.0
    
    # 统计每个真实人物被拆成的簇数
    actor_clusters = defaultdict(set)
    for i in range(len(true_labels)):
        if mask[i]:
            actor_clusters[true_labels[i]].add(pred_labels[i])
    
    if not actor_clusters:
        return 1.0
    
    # 计算平均碎片化度
    fragmentation = sum(len(clusters) for clusters in actor_clusters.values()) / len(actor_clusters)
    
    return max(1.0, fragmentation)


def calculate_intra_cluster_similarity(embeddings: np.ndarray, labels: List[int]) -> float:
    """计算簇内一致性：actor内两两相似度
    
    簇内一致性 = 所有有效簇内样本对相似度的平均值
    
    Args:
        embeddings: embedding矩阵
        labels: 预测标签列表（-1表示噪声）
    
    Returns:
        簇内一致性，范围[-1, 1]，越高越好
    """
    if not _SKLEARN_AVAILABLE:
        return 0.0
    
    # 排除噪声点
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    
    embeddings = embeddings[mask]
    labels = labels[mask]
    
    # 计算每个簇的平均相似度
    similarities = []
    for cluster_id in set(labels):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() < 2:
            continue
        
        cluster_embeddings = embeddings[cluster_mask]
        
        # 计算簇内两两相似度
        norm = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        normed = cluster_embeddings / norm
        
        sim_matrix = normed @ normed.T
        np.fill_diagonal(sim_matrix, 0)
        
        # 上三角的平均值
        n = len(cluster_embeddings)
        if n > 1:
            avg_sim = sim_matrix.sum() / (n * (n - 1))
            similarities.append(avg_sim)
    
    return np.mean(similarities) if similarities else 0.0


def calculate_inter_cluster_similarity(embeddings: np.ndarray, labels: List[int]) -> float:
    """计算跨actor相似度：不同actor间相似度
    
    跨actor相似度 = 不同簇间样本对相似度的平均值
    
    Args:
        embeddings: embedding矩阵
        labels: 预测标签列表（-1表示噪声）
    
    Returns:
        跨actor相似度，范围[-1, 1]，越低越好
    """
    if not _SKLEARN_AVAILABLE:
        return 0.0
    
    # 排除噪声点
    mask = labels != -1
    if mask.sum() < 2:
        return 0.0
    
    embeddings = embeddings[mask]
    labels = labels[mask]
    
    # 计算跨簇相似度
    similarities = []
    unique_labels = list(set(labels))
    
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            mask_i = labels == unique_labels[i]
            mask_j = labels == unique_labels[j]
            
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue
            
            emb_i = embeddings[mask_i]
            emb_j = embeddings[mask_j]
            
            # 计算两簇间的平均相似度
            norm_i = np.linalg.norm(emb_i, axis=1, keepdims=True)
            norm_j = np.linalg.norm(emb_j, axis=1, keepdims=True)
            
            norm_i = np.where(norm_i == 0, 1, norm_i)
            norm_j = np.where(norm_j == 0, 1, norm_j)
            
            normed_i = emb_i / norm_i
            normed_j = emb_j / norm_j
            
            sim_matrix = normed_i @ normed_j.T
            avg_sim = sim_matrix.mean()
            similarities.append(avg_sim)
    
    return np.mean(similarities) if similarities else 0.0


def evaluate_clustering(
    true_labels: List[str],
    pred_labels: List[int],
    embeddings: Optional[np.ndarray] = None
) -> Dict:
    """综合评估聚类结果
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        embeddings: embedding矩阵（可选，用于计算相似度指标）
    
    Returns:
        评估结果字典
    """
    results = {
        'purity': calculate_purity(true_labels, pred_labels),
        'coverage': calculate_coverage(true_labels, pred_labels),
        'false_positive_rate': calculate_false_positive_rate(true_labels, pred_labels),
        'fragmentation': calculate_fragmentation(true_labels, pred_labels),
    }
    
    # 如果有embedding，计算相似度指标
    if embeddings is not None and len(embeddings) > 0:
        results['intra_cluster_similarity'] = calculate_intra_cluster_similarity(embeddings, pred_labels)
        results['inter_cluster_similarity'] = calculate_inter_cluster_similarity(embeddings, pred_labels)
    
    # 计算NMI和ARI（如果有sklearn）
    if _SKLEARN_AVAILABLE:
        mask = pred_labels != -1
        if mask.sum() > 0 and len(set(pred_labels[mask])) > 1:
            results['nmi'] = normalized_mutual_info_score(
                [true_labels[i] for i in range(len(true_labels)) if mask[i]],
                pred_labels[mask]
            )
            results['ari'] = adjusted_rand_score(
                [true_labels[i] for i in range(len(true_labels)) if mask[i]],
                pred_labels[mask]
            )
        else:
            results['nmi'] = 0.0
            results['ari'] = 0.0
    
    # 添加统计信息
    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    n_noise = list(pred_labels).count(-1)
    results['n_clusters'] = n_clusters
    results['n_noise'] = n_noise
    results['noise_ratio'] = n_noise / len(pred_labels) if len(pred_labels) > 0 else 0.0
    
    return results


def generate_evaluation_report(
    video_name: str,
    results: Dict,
    params: Dict,
    output_path: Optional[str] = None
) -> str:
    """生成评估报告
    
    Args:
        video_name: 视频名称
        results: 评估结果
        params: 参数配置
        output_path: 输出路径（可选）
    
    Returns:
        报告内容
    """
    report = f"""# 人脸识别评估报告

## 基本信息
- 视频：{video_name}
- 评估时间：{np.datetime64('now')}

## 参数配置
- 算法：{params.get('algorithm', 'DBSCAN')}
- eps：{params.get('eps', 'N/A')}
- min_samples：{params.get('min_samples', 'N/A')}
- 特征：{params.get('feature_type', 'N/A')}

## 评估结果

### 核心指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 纯度 | {results['purity']:.4f} | actor内同人比例 |
| 覆盖率 | {results['coverage']:.4f} | 真实人物被找回比例 |
| 误检率 | {results['false_positive_rate']:.4f} | 非脸进入actor的比例 |
| 碎片化度 | {results['fragmentation']:.4f} | 同一人被拆成的actor数 |

### 相似度指标
| 指标 | 值 | 说明 |
|------|-----|------|
| 簇内一致性 | {results.get('intra_cluster_similarity', 'N/A'):.4f} | actor内两两相似度 |
| 跨actor相似度 | {results.get('inter_cluster_similarity', 'N/A'):.4f} | 不同actor间相似度 |

### 聚类统计
| 指标 | 值 |
|------|-----|
| 聚类数 | {results['n_clusters']} |
| 噪声点 | {results['n_noise']} |
| 噪声比例 | {results['noise_ratio']:.2%} |

### 其他指标
| 指标 | 值 |
|------|-----|
| NMI | {results.get('nmi', 'N/A'):.4f} |
| ARI | {results.get('ari', 'N/A'):.4f} |

## 结论
- 纯度达标：{'✓' if results['purity'] >= 0.8 else '✗'}
- 覆盖率达标：{'✓' if results['coverage'] >= 0.8 else '✗'}
- 误检率达标：{'✓' if results['false_positive_rate'] <= 0.05 else '✗'}
- 碎片化度达标：{'✓' if results['fragmentation'] <= 1.5 else '✗'}
"""
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("评估报告已保存到: %s", output_path)
    
    return report
