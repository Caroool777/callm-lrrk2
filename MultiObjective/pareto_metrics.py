"""
帕累托前沿评价指标模块 (Pareto Front Metrics)

提供多种帕累托前沿的评价指标，用于全面评估多目标优化算法的性能。

支持的评价指标：
1. Hypervolume (HV) - 超体积：衡量帕累托前沿覆盖的目标空间体积
2. Spacing - 间距：衡量解在前沿上的分布均匀性
3. Spread (Δ) - 分布广度：衡量解的分布范围和均匀性
4. IGD (Inverted Generational Distance) - 逆世代距离
5. GD (Generational Distance) - 世代距离
6. Coverage - 覆盖率指标
7. Maximum Spread - 最大扩展度
8. Pareto Front Diversity - 帕累托前沿多样性
9. AUC Top-1 - 优化过程中 Top-1 分子得分的曲线下面积
10. AUC Top-10 - 优化过程中 Top-10 分子平均得分的曲线下面积
11. AUC Top-100 - 优化过程中 Top-100 分子平均得分的曲线下面积

参考文献：
- Zitzler et al. "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results"
- Deb, K. "Multi-Objective Optimization using Evolutionary Algorithms" (2001)
- Coello et al. "Evolutionary Algorithms for Solving Multi-Objective Problems" (2007)
- Brown et al. "GuacaMol: Benchmarking Models for de Novo Molecular Design" (2019)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


def compute_hypervolume(pareto_front: List[Tuple], reference_point: Optional[List[float]] = None,
                        n_objectives: int = None) -> float:
    """
    计算帕累托前沿的超体积 (Hypervolume)
    
    超体积是衡量帕累托前沿质量的最重要指标之一，
    它度量了帕累托前沿与参考点之间所覆盖的目标空间体积。
    
    :param pareto_front: 帕累托前沿 [(objectives, mol, smiles), ...] 或 [objectives, ...]
    :param reference_point: 参考点，默认使用 [0, 0, ..., 0]
    :param n_objectives: 目标数量（如果未提供，从数据推断）
    :return: 超体积值（越大越好）
    """
    if not pareto_front:
        return 0.0
    
    # 提取目标向量
    if isinstance(pareto_front[0], tuple) and len(pareto_front[0]) >= 1:
        # 格式: [(objectives, mol, smiles), ...] 或 [(score, smiles, objectives), ...]
        if isinstance(pareto_front[0][0], (list, np.ndarray)):
            points = np.array([item[0] for item in pareto_front])
        elif len(pareto_front[0]) >= 3 and isinstance(pareto_front[0][2], (list, np.ndarray)):
            points = np.array([item[2] for item in pareto_front])
        else:
            points = np.array([list(item) for item in pareto_front])
    else:
        points = np.array(pareto_front)
    
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    
    n_points, n_obj = points.shape
    
    if n_objectives is None:
        n_objectives = n_obj
    
    if reference_point is None:
        reference_point = np.zeros(n_objectives)
    else:
        reference_point = np.array(reference_point)
    
    # 确保所有点都支配参考点（对于最大化问题）
    valid_points = points[np.all(points > reference_point, axis=1)]
    
    if len(valid_points) == 0:
        # 如果没有点支配参考点，计算绝对值的超体积
        valid_points = points
    
    if n_objectives == 2:
        # 2D 精确算法
        return _compute_hypervolume_2d(valid_points, reference_point)
    elif n_objectives == 3:
        # 3D 算法
        return _compute_hypervolume_3d(valid_points, reference_point)
    else:
        # 高维近似算法（Monte Carlo）
        return _compute_hypervolume_monte_carlo(valid_points, reference_point)


def _compute_hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """2D 超体积精确计算"""
    if len(points) == 0:
        return 0.0
    
    # 按第一个目标排序
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    hv = 0.0
    prev_y = reference_point[1]
    
    for i in range(len(sorted_points)):
        x = sorted_points[i, 0]
        y = sorted_points[i, 1]
        
        if i == len(sorted_points) - 1:
            # 最后一个点
            width = 1.0 - x if reference_point[0] == 0 else x - reference_point[0]
        else:
            width = sorted_points[i + 1, 0] - x
        
        height = y - reference_point[1]
        
        if width > 0 and height > 0:
            hv += width * height
    
    return hv


def _compute_hypervolume_3d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """3D 超体积计算（基于切片的近似方法）"""
    if len(points) == 0:
        return 0.0
    
    # 使用蒙特卡洛近似
    return _compute_hypervolume_monte_carlo(points, reference_point)


def _compute_hypervolume_monte_carlo(points: np.ndarray, reference_point: np.ndarray,
                                     n_samples: int = 10000) -> float:
    """Monte Carlo 近似超体积计算"""
    if len(points) == 0:
        return 0.0
    
    n_obj = points.shape[1]
    
    # 确定边界
    upper_bound = np.max(points, axis=0)
    lower_bound = reference_point
    
    # 计算包围盒体积
    box_volume = np.prod(upper_bound - lower_bound)
    
    if box_volume <= 0:
        return 0.0
    
    # 随机采样
    samples = np.random.uniform(lower_bound, upper_bound, (n_samples, n_obj))
    
    # 检查每个样本是否被至少一个点支配
    dominated_count = 0
    for sample in samples:
        for point in points:
            if np.all(point >= sample):
                dominated_count += 1
                break
    
    # 估计超体积
    hv = box_volume * (dominated_count / n_samples)
    return hv


def compute_spacing(pareto_front: List[Tuple], n_objectives: int = None) -> float:
    """
    计算帕累托前沿的间距 (Spacing)
    
    Spacing 衡量解在帕累托前沿上分布的均匀性。
    值越小表示分布越均匀。
    
    公式: S = sqrt(1/n * Σ(d_i - d_mean)^2)
    其中 d_i 是点 i 到最近邻点的距离
    
    :param pareto_front: 帕累托前沿
    :param n_objectives: 目标数量
    :return: 间距值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) < 2:
        return 0.0
    
    # 计算每个点到最近邻的距离
    distances = []
    for i, p1 in enumerate(points):
        min_dist = float('inf')
        for j, p2 in enumerate(points):
            if i != j:
                dist = np.sum(np.abs(p1 - p2))  # 曼哈顿距离
                min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    distances = np.array(distances)
    d_mean = np.mean(distances)
    
    # 计算标准差
    spacing = np.sqrt(np.mean((distances - d_mean) ** 2))
    
    return float(spacing)


def compute_spread(pareto_front: List[Tuple], n_objectives: int = None,
                   extreme_points: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    """
    计算帕累托前沿的分布广度 (Spread/Delta)
    
    Spread (Δ) 衡量解的分布范围和均匀性。
    考虑了极端点的距离和解之间的分布。
    
    公式: Δ = (d_f + d_l + Σ|d_i - d_mean|) / (d_f + d_l + (n-1)*d_mean)
    
    :param pareto_front: 帕累托前沿
    :param n_objectives: 目标数量
    :param extreme_points: (最小极端点, 最大极端点) 可选
    :return: 分布广度值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) < 2:
        return 1.0
    
    n_obj = points.shape[1]
    
    # 计算连续点之间的距离（按第一个目标排序）
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    consecutive_distances = []
    for i in range(len(sorted_points) - 1):
        dist = np.linalg.norm(sorted_points[i + 1] - sorted_points[i])
        consecutive_distances.append(dist)
    
    if not consecutive_distances:
        return 1.0
    
    d_mean = np.mean(consecutive_distances)
    
    # 计算极端点的距离
    if extreme_points is not None:
        d_f = np.linalg.norm(sorted_points[0] - extreme_points[0])
        d_l = np.linalg.norm(sorted_points[-1] - extreme_points[1])
    else:
        # 使用估计的极端点（单目标最优解）
        d_f = 0.0
        d_l = 0.0
        for m in range(n_obj):
            max_idx = np.argmax(points[:, m])
            min_idx = np.argmin(points[:, m])
            # 估计极端点贡献
            d_f += np.linalg.norm(sorted_points[0] - points[min_idx])
            d_l += np.linalg.norm(sorted_points[-1] - points[max_idx])
        d_f /= n_obj
        d_l /= n_obj
    
    # 计算 Spread
    numerator = d_f + d_l + np.sum(np.abs(np.array(consecutive_distances) - d_mean))
    denominator = d_f + d_l + (len(points) - 1) * d_mean
    
    if denominator == 0:
        return 1.0
    
    spread = numerator / denominator
    return float(spread)


def compute_igd(pareto_front: List[Tuple], reference_front: np.ndarray) -> float:
    """
    计算逆世代距离 (Inverted Generational Distance, IGD)
    
    IGD 衡量参考前沿到获得的帕累托前沿的平均距离。
    值越小表示算法获得的前沿越接近真实前沿。
    
    公式: IGD = (1/|P*|) * Σ min_y∈P d(x, y), 对于所有 x ∈ P*
    
    :param pareto_front: 获得的帕累托前沿
    :param reference_front: 参考/真实帕累托前沿
    :return: IGD 值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) == 0 or len(reference_front) == 0:
        return float('inf')
    
    # 对参考前沿中的每个点，找到帕累托前沿中最近的点
    total_distance = 0.0
    for ref_point in reference_front:
        min_dist = float('inf')
        for point in points:
            dist = np.linalg.norm(ref_point - point)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    igd = total_distance / len(reference_front)
    return float(igd)


def compute_gd(pareto_front: List[Tuple], reference_front: np.ndarray) -> float:
    """
    计算世代距离 (Generational Distance, GD)
    
    GD 衡量获得的帕累托前沿到参考前沿的平均距离。
    
    公式: GD = (1/|P|) * Σ min_x∈P* d(y, x), 对于所有 y ∈ P
    
    :param pareto_front: 获得的帕累托前沿
    :param reference_front: 参考/真实帕累托前沿
    :return: GD 值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) == 0 or len(reference_front) == 0:
        return float('inf')
    
    # 对帕累托前沿中的每个点，找到参考前沿中最近的点
    total_distance = 0.0
    for point in points:
        min_dist = float('inf')
        for ref_point in reference_front:
            dist = np.linalg.norm(point - ref_point)
            min_dist = min(min_dist, dist)
        total_distance += min_dist
    
    gd = total_distance / len(points)
    return float(gd)


def compute_maximum_spread(pareto_front: List[Tuple]) -> float:
    """
    计算最大扩展度 (Maximum Spread)
    
    衡量帕累托前沿在各个目标维度上的覆盖范围。
    
    公式: MS = sqrt(Σ(max_i - min_i)^2)
    
    :param pareto_front: 帕累托前沿
    :return: 最大扩展度值（越大越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) == 0:
        return 0.0
    
    max_vals = np.max(points, axis=0)
    min_vals = np.min(points, axis=0)
    
    spread = np.sqrt(np.sum((max_vals - min_vals) ** 2))
    return float(spread)


def compute_coverage_ratio(pareto_front: List[Tuple], 
                           comparison_front: List[Tuple]) -> float:
    """
    计算覆盖率 (Coverage Ratio / C-metric)
    
    计算 pareto_front 支配 comparison_front 中解的比例。
    
    :param pareto_front: 第一个帕累托前沿
    :param comparison_front: 第二个帕累托前沿（用于比较）
    :return: 覆盖率 [0, 1]（越高越好）
    """
    points1 = _extract_objectives(pareto_front)
    points2 = _extract_objectives(comparison_front)
    
    if len(points2) == 0:
        return 1.0
    
    dominated_count = 0
    for p2 in points2:
        for p1 in points1:
            if _dominates(p1, p2):
                dominated_count += 1
                break
    
    return dominated_count / len(points2)


def compute_diversity(pareto_front: List[Tuple]) -> float:
    """
    计算帕累托前沿的多样性 (Diversity)
    
    基于解之间的平均距离来衡量多样性。
    
    :param pareto_front: 帕累托前沿
    :return: 多样性值（越大越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) < 2:
        return 0.0
    
    # 计算所有点对之间的平均距离
    total_distance = 0.0
    count = 0
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            total_distance += dist
            count += 1
    
    if count == 0:
        return 0.0
    
    diversity = total_distance / count
    return float(diversity)


def compute_epsilon_indicator(pareto_front: List[Tuple], 
                               reference_front: np.ndarray) -> float:
    """
    计算 Epsilon 指标 (ε-indicator)
    
    ε-indicator 衡量将帕累托前沿平移到覆盖参考前沿所需的最小距离。
    
    :param pareto_front: 获得的帕累托前沿
    :param reference_front: 参考前沿
    :return: ε 值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) == 0 or len(reference_front) == 0:
        return float('inf')
    
    max_epsilon = float('-inf')
    
    for ref_point in reference_front:
        min_epsilon = float('inf')
        for point in points:
            # 计算每个目标维度的 epsilon
            epsilon = np.max(ref_point - point)
            min_epsilon = min(min_epsilon, epsilon)
        max_epsilon = max(max_epsilon, min_epsilon)
    
    return float(max_epsilon)


def compute_uniformity(pareto_front: List[Tuple]) -> float:
    """
    计算均匀性 (Uniformity)
    
    基于最近邻距离的标准差与均值的比值来衡量均匀性。
    值越接近0表示分布越均匀。
    
    :param pareto_front: 帕累托前沿
    :return: 均匀性值（越小越好）
    """
    points = _extract_objectives(pareto_front)
    
    if len(points) < 3:
        return 0.0
    
    # 计算每个点到最近邻的距离
    nn_distances = []
    for i, p1 in enumerate(points):
        min_dist = float('inf')
        for j, p2 in enumerate(points):
            if i != j:
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
        nn_distances.append(min_dist)
    
    nn_distances = np.array(nn_distances)
    mean_dist = np.mean(nn_distances)
    
    if mean_dist == 0:
        return 0.0
    
    # 变异系数（Coefficient of Variation）
    cv = np.std(nn_distances) / mean_dist
    return float(cv)


# ===================== AUC Top-K 指标 =====================

def compute_aggregated_score(objectives: List[float], method: str = 'geometric') -> float:
    """
    计算多目标的聚合得分
    
    :param objectives: 目标值列表
    :param method: 聚合方法 ('geometric', 'arithmetic', 'harmonic', 'min')
    :return: 聚合得分
    """
    objectives = [max(o, 1e-10) for o in objectives]  # 避免 log(0)
    
    if method == 'geometric':
        # 几何平均（推荐用于多目标优化）
        return float(np.exp(np.mean(np.log(objectives))))
    elif method == 'arithmetic':
        # 算术平均
        return float(np.mean(objectives))
    elif method == 'harmonic':
        # 调和平均
        return float(len(objectives) / np.sum(1.0 / np.array(objectives)))
    elif method == 'min':
        # 最小值（最保守的方法）
        return float(np.min(objectives))
    else:
        return float(np.mean(objectives))


def compute_top_k_scores(all_discovered: List[Tuple], k: int,
                         aggregation_method: str = 'geometric') -> List[float]:
    """
    计算优化过程中每个时间点的 Top-K 聚合得分
    
    :param all_discovered: 所有发现的分子 [(objectives, smiles, oracle_call_idx), ...]
    :param k: Top-K 的 K 值
    :param aggregation_method: 聚合方法
    :return: 每个 Oracle 调用点的 Top-K 平均得分列表
    """
    if not all_discovered:
        return []
    
    # 按 Oracle 调用索引排序
    sorted_discoveries = sorted(all_discovered, key=lambda x: x[2])
    
    # 计算每个分子的聚合得分
    scored_discoveries = []
    for objectives, smiles, call_idx in sorted_discoveries:
        score = compute_aggregated_score(objectives, aggregation_method)
        scored_discoveries.append((score, smiles, call_idx, objectives))
    
    # 计算每个时间点的 Top-K 得分
    top_k_scores = []
    current_molecules = []
    
    for score, smiles, call_idx, objectives in scored_discoveries:
        # 添加当前分子并按得分排序
        current_molecules.append((score, smiles))
        current_molecules.sort(key=lambda x: x[0], reverse=True)
        
        # 取 Top-K 的平均得分
        top_k = current_molecules[:k]
        if top_k:
            avg_score = np.mean([s for s, _ in top_k])
            top_k_scores.append((call_idx, avg_score))
    
    return top_k_scores


def compute_auc_top_k(all_discovered: List[Tuple], k: int,
                       max_oracle_calls: int = None,
                       aggregation_method: str = 'geometric',
                       normalize: bool = True) -> float:
    """
    计算 AUC Top-K 指标
    
    AUC Top-K 是优化过程中 Top-K 分子平均得分曲线下的面积。
    该指标衡量算法在整个优化过程中的性能，而不仅仅是最终结果。
    
    参考: GuacaMol benchmark (Brown et al., 2019)
    
    :param all_discovered: 所有发现的分子 [(objectives, smiles, oracle_call_idx), ...]
    :param k: Top-K 的 K 值 (1, 10, 100)
    :param max_oracle_calls: 最大 Oracle 调用次数（用于归一化）
    :param aggregation_method: 目标聚合方法
    :param normalize: 是否归一化 AUC 到 [0, 1] 范围
    :return: AUC Top-K 值（越大越好）
    """
    if not all_discovered:
        return 0.0
    
    # 计算 Top-K 得分序列
    top_k_scores = compute_top_k_scores(all_discovered, k, aggregation_method)
    
    if len(top_k_scores) < 2:
        return 0.0
    
    # 使用梯形法则计算 AUC
    call_indices = [item[0] for item in top_k_scores]
    scores = [item[1] for item in top_k_scores]
    
    # 计算 AUC
    auc = np.trapz(scores, call_indices)
    
    if normalize:
        # 归一化到 [0, 1]
        if max_oracle_calls is None:
            max_oracle_calls = max(call_indices)
        
        # 最大可能 AUC（假设所有分数都是 1.0）
        max_auc = max_oracle_calls * 1.0
        
        if max_auc > 0:
            auc = auc / max_auc
    
    return float(auc)


def compute_all_auc_metrics(all_discovered: List[Tuple],
                            max_oracle_calls: int = None,
                            aggregation_method: str = 'geometric') -> Dict[str, float]:
    """
    计算所有 AUC Top-K 指标
    
    :param all_discovered: 所有发现的分子
    :param max_oracle_calls: 最大 Oracle 调用次数
    :param aggregation_method: 目标聚合方法
    :return: 包含 auc_top_1, auc_top_10, auc_top_100 的字典
    """
    return {
        'auc_top_1': compute_auc_top_k(all_discovered, k=1, 
                                        max_oracle_calls=max_oracle_calls,
                                        aggregation_method=aggregation_method),
        'auc_top_10': compute_auc_top_k(all_discovered, k=10,
                                         max_oracle_calls=max_oracle_calls,
                                         aggregation_method=aggregation_method),
        'auc_top_100': compute_auc_top_k(all_discovered, k=100,
                                          max_oracle_calls=max_oracle_calls,
                                          aggregation_method=aggregation_method)
    }


def compute_top_k_final_scores(all_discovered: List[Tuple], k_values: List[int] = None,
                               aggregation_method: str = 'geometric') -> Dict[str, float]:
    """
    计算最终的 Top-K 得分（优化结束时）
    
    :param all_discovered: 所有发现的分子
    :param k_values: K 值列表，默认 [1, 10, 100]
    :param aggregation_method: 目标聚合方法
    :return: 包含各 Top-K 最终得分的字典
    """
    if k_values is None:
        k_values = [1, 10, 100]
    
    if not all_discovered:
        return {f'top_{k}_score': 0.0 for k in k_values}
    
    # 计算所有分子的聚合得分
    scored_molecules = []
    for objectives, smiles, call_idx in all_discovered:
        score = compute_aggregated_score(objectives, aggregation_method)
        scored_molecules.append((score, smiles, objectives))
    
    # 按得分排序
    scored_molecules.sort(key=lambda x: x[0], reverse=True)
    
    results = {}
    for k in k_values:
        top_k = scored_molecules[:k]
        if top_k:
            results[f'top_{k}_score'] = float(np.mean([s for s, _, _ in top_k]))
        else:
            results[f'top_{k}_score'] = 0.0
    
    return results


# ===================== 辅助函数 =====================

def _extract_objectives(pareto_front: List[Tuple]) -> np.ndarray:
    """从帕累托前沿中提取目标向量"""
    if not pareto_front:
        return np.array([])
    
    if isinstance(pareto_front[0], tuple):
        # 格式: [(objectives, mol, smiles), ...] 或 [(score, smiles, objectives), ...]
        if isinstance(pareto_front[0][0], (list, np.ndarray)):
            return np.array([item[0] for item in pareto_front])
        elif len(pareto_front[0]) >= 3 and isinstance(pareto_front[0][2], (list, np.ndarray)):
            return np.array([item[2] for item in pareto_front])
    
    return np.array(pareto_front)


def _dominates(p1: np.ndarray, p2: np.ndarray) -> bool:
    """检查 p1 是否帕累托支配 p2"""
    return np.all(p1 >= p2) and np.any(p1 > p2)


# ===================== 综合评价类 =====================

class ParetoMetrics:
    """
    帕累托前沿综合评价类
    
    提供一站式的帕累托前沿评价功能。
    """
    
    def __init__(self, reference_point: Optional[List[float]] = None,
                 reference_front: Optional[np.ndarray] = None,
                 objective_names: Optional[List[str]] = None):
        """
        初始化评价器
        
        :param reference_point: 超体积计算的参考点
        :param reference_front: 用于 IGD/GD 计算的参考前沿
        :param objective_names: 目标名称列表
        """
        self.reference_point = reference_point
        self.reference_front = reference_front
        self.objective_names = objective_names or []
    
    def evaluate_all(self, pareto_front: List[Tuple], 
                     n_objectives: int = None,
                     all_discovered: List[Tuple] = None,
                     max_oracle_calls: int = None) -> Dict[str, float]:
        """
        计算所有评价指标
        
        :param pareto_front: 帕累托前沿
        :param n_objectives: 目标数量
        :param all_discovered: 所有发现的分子（用于 AUC Top-K 计算）
        :param max_oracle_calls: 最大 Oracle 调用次数
        :return: 包含所有指标的字典
        """
        metrics = {}
        
        points = _extract_objectives(pareto_front)
        if len(points) == 0:
            return self._get_empty_metrics()
        
        if n_objectives is None:
            n_objectives = points.shape[1] if len(points.shape) > 1 else 1
        
        # 基本信息
        metrics['pareto_size'] = len(pareto_front)
        metrics['n_objectives'] = n_objectives
        
        # 超体积
        metrics['hypervolume'] = compute_hypervolume(
            pareto_front, self.reference_point, n_objectives
        )
        
        # 间距（均匀性）
        metrics['spacing'] = compute_spacing(pareto_front, n_objectives)
        
        # 分布广度
        metrics['spread'] = compute_spread(pareto_front, n_objectives)
        
        # 最大扩展度
        metrics['maximum_spread'] = compute_maximum_spread(pareto_front)
        
        # 多样性
        metrics['diversity'] = compute_diversity(pareto_front)
        
        # 均匀性（变异系数）
        metrics['uniformity'] = compute_uniformity(pareto_front)
        
        # 如果有参考前沿，计算 IGD 和 GD
        if self.reference_front is not None and len(self.reference_front) > 0:
            metrics['igd'] = compute_igd(pareto_front, self.reference_front)
            metrics['gd'] = compute_gd(pareto_front, self.reference_front)
            metrics['epsilon_indicator'] = compute_epsilon_indicator(
                pareto_front, self.reference_front
            )
        else:
            metrics['igd'] = None
            metrics['gd'] = None
            metrics['epsilon_indicator'] = None
        
        # AUC Top-K 指标（需要 all_discovered 数据）
        if all_discovered is not None and len(all_discovered) > 0:
            auc_metrics = compute_all_auc_metrics(
                all_discovered, 
                max_oracle_calls=max_oracle_calls
            )
            metrics.update(auc_metrics)
            
            # Top-K 最终得分
            top_k_scores = compute_top_k_final_scores(all_discovered)
            metrics.update(top_k_scores)
        else:
            metrics['auc_top_1'] = None
            metrics['auc_top_10'] = None
            metrics['auc_top_100'] = None
            metrics['top_1_score'] = None
            metrics['top_10_score'] = None
            metrics['top_100_score'] = None
        
        # 目标统计
        if len(points.shape) > 1:
            metrics['objective_means'] = points.mean(axis=0).tolist()
            metrics['objective_stds'] = points.std(axis=0).tolist()
            metrics['objective_maxs'] = points.max(axis=0).tolist()
            metrics['objective_mins'] = points.min(axis=0).tolist()
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """返回空的指标字典"""
        return {
            'pareto_size': 0,
            'n_objectives': 0,
            'hypervolume': 0.0,
            'spacing': 0.0,
            'spread': 1.0,
            'maximum_spread': 0.0,
            'diversity': 0.0,
            'uniformity': 0.0,
            'igd': None,
            'gd': None,
            'epsilon_indicator': None,
            'auc_top_1': None,
            'auc_top_10': None,
            'auc_top_100': None,
            'top_1_score': None,
            'top_10_score': None,
            'top_100_score': None,
            'objective_means': [],
            'objective_stds': [],
            'objective_maxs': [],
            'objective_mins': []
        }
    
    def print_metrics(self, metrics: Dict[str, float], 
                      objective_names: Optional[List[str]] = None) -> str:
        """
        格式化打印评价指标
        
        :param metrics: 指标字典
        :param objective_names: 目标名称列表
        :return: 格式化的字符串
        """
        if objective_names is None:
            objective_names = self.objective_names
        
        lines = []
        lines.append("=" * 60)
        lines.append("帕累托前沿评价指标 (Pareto Front Metrics)")
        lines.append("=" * 60)
        
        # 基本信息
        lines.append(f"帕累托前沿大小: {metrics.get('pareto_size', 0)}")
        lines.append(f"目标数量: {metrics.get('n_objectives', 0)}")
        lines.append("-" * 60)
        
        # 收敛性指标
        lines.append("【收敛性指标 (Convergence)】")
        lines.append(f"  Hypervolume (HV): {metrics.get('hypervolume', 0.0):.6f}")
        lines.append("    → 越大越好，衡量帕累托前沿覆盖的目标空间体积")
        
        if metrics.get('igd') is not None:
            lines.append(f"  IGD (逆世代距离): {metrics.get('igd', 0.0):.6f}")
            lines.append("    → 越小越好，衡量到真实帕累托前沿的距离")
        
        if metrics.get('gd') is not None:
            lines.append(f"  GD (世代距离): {metrics.get('gd', 0.0):.6f}")
            lines.append("    → 越小越好，衡量解的收敛程度")
        
        if metrics.get('epsilon_indicator') is not None:
            lines.append(f"  ε-indicator: {metrics.get('epsilon_indicator', 0.0):.6f}")
            lines.append("    → 越小越好，衡量需要平移多少才能覆盖参考前沿")
        
        lines.append("-" * 60)
        
        # 多样性指标
        lines.append("【多样性指标 (Diversity)】")
        lines.append(f"  Spacing (间距): {metrics.get('spacing', 0.0):.6f}")
        lines.append("    → 越小越好，衡量解分布的均匀性")
        
        lines.append(f"  Spread (Δ): {metrics.get('spread', 1.0):.6f}")
        lines.append("    → 越小越好，衡量解的分布广度和均匀性")
        
        lines.append(f"  Maximum Spread (最大扩展度): {metrics.get('maximum_spread', 0.0):.6f}")
        lines.append("    → 越大越好，衡量帕累托前沿的覆盖范围")
        
        lines.append(f"  Diversity (多样性): {metrics.get('diversity', 0.0):.6f}")
        lines.append("    → 越大越好，衡量解之间的平均距离")
        
        lines.append(f"  Uniformity (均匀性CV): {metrics.get('uniformity', 0.0):.6f}")
        lines.append("    → 越小越好，最近邻距离的变异系数")
        
        lines.append("-" * 60)
        
        # AUC Top-K 指标
        lines.append("【AUC Top-K 指标 (Optimization Progress)】")
        if metrics.get('auc_top_1') is not None:
            lines.append(f"  AUC Top-1: {metrics.get('auc_top_1', 0.0):.6f}")
            lines.append("    → 越大越好，优化过程中 Top-1 分子得分的曲线下面积")
        if metrics.get('auc_top_10') is not None:
            lines.append(f"  AUC Top-10: {metrics.get('auc_top_10', 0.0):.6f}")
            lines.append("    → 越大越好，优化过程中 Top-10 分子平均得分的曲线下面积")
        if metrics.get('auc_top_100') is not None:
            lines.append(f"  AUC Top-100: {metrics.get('auc_top_100', 0.0):.6f}")
            lines.append("    → 越大越好，优化过程中 Top-100 分子平均得分的曲线下面积")
        
        # Top-K 最终得分
        if metrics.get('top_1_score') is not None:
            lines.append("-" * 60)
            lines.append("【Top-K 最终得分 (Final Scores)】")
            lines.append(f"  Top-1 Score: {metrics.get('top_1_score', 0.0):.6f}")
            lines.append(f"  Top-10 Score: {metrics.get('top_10_score', 0.0):.6f}")
            lines.append(f"  Top-100 Score: {metrics.get('top_100_score', 0.0):.6f}")
        
        lines.append("-" * 60)
        
        # 目标统计
        if metrics.get('objective_means'):
            lines.append("【各目标统计】")
            obj_means = metrics.get('objective_means', [])
            obj_stds = metrics.get('objective_stds', [])
            obj_maxs = metrics.get('objective_maxs', [])
            obj_mins = metrics.get('objective_mins', [])
            
            for i in range(len(obj_means)):
                name = objective_names[i] if i < len(objective_names) else f"Obj{i+1}"
                lines.append(f"  {name}:")
                lines.append(f"    Mean={obj_means[i]:.4f}, Std={obj_stds[i]:.4f}")
                lines.append(f"    Max={obj_maxs[i]:.4f}, Min={obj_mins[i]:.4f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_metrics_evaluator(objective_names: List[str] = None,
                             reference_point: List[float] = None,
                             reference_front: np.ndarray = None) -> ParetoMetrics:
    """
    创建帕累托指标评价器的便捷函数
    
    :param objective_names: 目标名称列表
    :param reference_point: 超体积参考点
    :param reference_front: IGD/GD 参考前沿
    :return: ParetoMetrics 实例
    """
    return ParetoMetrics(
        reference_point=reference_point,
        reference_front=reference_front,
        objective_names=objective_names
    )


if __name__ == "__main__":
    # 测试代码
    print("测试帕累托前沿评价指标模块")
    print("-" * 40)
    
    # 创建测试数据（帕累托前沿）
    test_front = [
        ([0.9, 0.3, 0.5, 0.7], None, "mol1"),
        ([0.7, 0.8, 0.4, 0.6], None, "mol2"),
        ([0.5, 0.5, 0.9, 0.8], None, "mol3"),
        ([0.6, 0.7, 0.6, 0.5], None, "mol4"),
        ([0.8, 0.4, 0.7, 0.9], None, "mol5"),
    ]
    
    # 创建测试数据（all_discovered 格式：(objectives, smiles, oracle_call_idx)）
    test_discovered = [
        ([0.3, 0.2, 0.1, 0.4], "mol_1", 1),
        ([0.4, 0.3, 0.2, 0.5], "mol_2", 10),
        ([0.5, 0.4, 0.3, 0.6], "mol_3", 20),
        ([0.6, 0.5, 0.4, 0.7], "mol_4", 30),
        ([0.7, 0.6, 0.5, 0.8], "mol_5", 40),
        ([0.8, 0.7, 0.6, 0.9], "mol_6", 50),
        ([0.9, 0.3, 0.5, 0.7], "mol_7", 60),
        ([0.7, 0.8, 0.4, 0.6], "mol_8", 70),
        ([0.5, 0.5, 0.9, 0.8], "mol_9", 80),
        ([0.6, 0.7, 0.6, 0.5], "mol_10", 90),
        ([0.8, 0.4, 0.7, 0.9], "mol_11", 100),
    ]
    
    # 创建评价器
    evaluator = ParetoMetrics(
        objective_names=["similarity", "tpsa", "logp", "fluorine"]
    )
    
    # 计算所有指标（包括 AUC Top-K）
    metrics = evaluator.evaluate_all(
        test_front, 
        n_objectives=4,
        all_discovered=test_discovered,
        max_oracle_calls=100
    )
    
    # 打印结果
    print(evaluator.print_metrics(metrics))
    
    # 单独测试 AUC Top-K 函数
    print("\n" + "=" * 60)
    print("单独测试 AUC Top-K 函数")
    print("=" * 60)
    
    auc_1 = compute_auc_top_k(test_discovered, k=1, max_oracle_calls=100)
    auc_10 = compute_auc_top_k(test_discovered, k=10, max_oracle_calls=100)
    auc_100 = compute_auc_top_k(test_discovered, k=100, max_oracle_calls=100)
    
    print(f"AUC Top-1: {auc_1:.6f}")
    print(f"AUC Top-10: {auc_10:.6f}")
    print(f"AUC Top-100: {auc_100:.6f}")
    
    # 测试 Top-K 最终得分
    top_k_scores = compute_top_k_final_scores(test_discovered)
    print(f"\nTop-K Final Scores:")
    for k, v in top_k_scores.items():
        print(f"  {k}: {v:.6f}")
