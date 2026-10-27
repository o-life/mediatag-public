"""纯度-覆盖率曲线模块

实现纯度-覆盖率曲线绘制和最优工作点选择，
符合《人脸识别系统过程控制方案》的要求。
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .face_evaluation import calculate_purity, calculate_coverage

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available — purity-coverage curve disabled")

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plotting disabled")


def generate_purity_coverage_curve(
    features: np.ndarray,
    true_labels: List[str],
    param_grid: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> Tuple[List[Dict], Dict]:
    """生成纯度-覆盖率曲线
    
    Args:
        features: 特征矩阵
        true_labels: 真实标签列表
        param_grid: 参数网格
        output_path: 输出路径（可选）
    
    Returns:
        (所有结果, 最优工作点) 的元组
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("sklearn not available")
        return [], {}
    
    # 默认参数网格
    if param_grid is None:
        param_grid = {
            'eps': np.arange(0.3, 0.7, 0.05).tolist(),
            'min_samples': [3, 5, 8, 10, 15]
        }
    
    results = []
    
    # 测试所有参数组合
    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            # 聚类
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            pred_labels = clustering.fit_predict(features)
            
            # 计算指标
            purity = calculate_purity(true_labels, pred_labels)
            coverage = calculate_coverage(true_labels, pred_labels)
            
            # 统计
            n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
            n_noise = list(pred_labels).count(-1)
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'purity': purity,
                'coverage': coverage,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })
    
    # 选择最优工作点（覆盖率>=80%时纯度最高）
    optimal = select_working_point(results, min_coverage=0.8)
    
    # 绘制曲线
    if output_path:
        plot_purity_coverage_curve(results, optimal, output_path)
    
    return results, optimal


def select_working_point(
    results: List[Dict],
    min_coverage: float = 0.8
) -> Dict:
    """选择最优工作点
    
    在覆盖率>=min_coverage的前提下，选择纯度最高的参数组合
    
    Args:
        results: 所有结果
        min_coverage: 最低覆盖率要求
    
    Returns:
        最优工作点
    """
    # 筛选满足覆盖率要求的结果
    valid_results = [r for r in results if r['coverage'] >= min_coverage]
    
    if not valid_results:
        # 如果没有满足要求的，选择覆盖率最高的
        if results:
            return max(results, key=lambda x: x['coverage'])
        return {}
    
    # 选择纯度最高的
    return max(valid_results, key=lambda x: x['purity'])


def plot_purity_coverage_curve(
    results: List[Dict],
    optimal: Dict,
    output_path: str
):
    """绘制纯度-覆盖率曲线
    
    Args:
        results: 所有结果
        optimal: 最优工作点
        output_path: 输出路径
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 提取数据
    coverages = [r['coverage'] for r in results]
    purities = [r['purity'] for r in results]
    eps_values = [r['eps'] for r in results]
    
    # 绘制散点图
    scatter = plt.scatter(coverages, purities, c=eps_values, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='eps')
    
    # 标记最优工作点
    if optimal:
        plt.scatter(
            optimal['coverage'], 
            optimal['purity'], 
            c='red', 
            s=200, 
            marker='star',
            label=f'最优: eps={optimal["eps"]:.2f}, min_samples={optimal["min_samples"]}'
        )
    
    # 绘制参考线
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='纯度=0.8')
    plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='覆盖率=0.8')
    
    # 设置标签
    plt.xlabel('覆盖率')
    plt.ylabel('纯度')
    plt.title('纯度-覆盖率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("曲线已保存到: %s", output_path)


def batch_generate_curves(
    videos: List[Dict],
    output_dir: str
) -> Dict:
    """批量生成多个视频的纯度-覆盖率曲线
    
    Args:
        videos: 视频列表，每个包含features和true_labels
        output_dir: 输出目录
    
    Returns:
        汇总结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    for video_info in videos:
        video_name = video_info['name']
        features = video_info['features']
        true_labels = video_info['true_labels']
        
        logger.info("处理视频: %s", video_name)
        
        # 生成曲线
        output_path = output_dir / f'{video_name}_purity_coverage.png'
        results, optimal = generate_purity_coverage_curve(
            features, true_labels, output_path=str(output_path)
        )
        
        summary[video_name] = {
            'optimal': optimal,
            'n_experiments': len(results),
            'curve_path': str(output_path)
        }
    
    return summary
