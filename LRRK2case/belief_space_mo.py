"""
多目标信念空间 (Multi-Objective Belief Space) 模块 - 支持帕累托前沿

【文化算法 + 多目标优化核心】
与单目标版本不同，多目标信念空间存储的是帕累托前沿上的分子，
并使用帕累托支配关系来维护非支配解集。

实现的多目标文化算法机制：
1. Outlining (概括): 使用 MurckoScaffold 提取分子核心骨架
2. 知识聚合: 统计每种骨架对应的帕累托优势
3. Pareto Ranking: 使用帕累托支配关系进行排序
4. Influence (影响): 提供帕累托前沿上的最优骨架用于指导 BioT5 生成
5. Diversity (多样性): 通过拥挤距离维护解集多样性
"""

import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs


def pareto_dominates(obj1, obj2):
    """
    检查 obj1 是否帕累托支配 obj2
    
    :param obj1: 目标向量 1 (列表或数组)
    :param obj2: 目标向量 2 (列表或数组)
    :return: True 如果 obj1 帕累托支配 obj2
    
    帕累托支配定义：
    - obj1 在所有目标上都不劣于 obj2
    - obj1 在至少一个目标上严格优于 obj2
    """
    at_least_as_good = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
    return at_least_as_good and strictly_better


def compute_pareto_front(population_data):
    """
    计算帕累托前沿（非支配解集）
    
    :param population_data: [(objectives, mol, smiles), ...] 
                           其中 objectives 是目标向量
    :return: 帕累托前沿上的个体列表
    """
    if not population_data:
        return []
    
    pareto_front = []
    for candidate in population_data:
        is_dominated = False
        candidates_to_remove = []
        
        for i, existing in enumerate(pareto_front):
            if pareto_dominates(existing[0], candidate[0]):
                is_dominated = True
                break
            elif pareto_dominates(candidate[0], existing[0]):
                candidates_to_remove.append(i)
        
        if not is_dominated:
            # 移除被 candidate 支配的解
            for i in sorted(candidates_to_remove, reverse=True):
                pareto_front.pop(i)
            pareto_front.append(candidate)
    
    return pareto_front


def compute_pareto_ranks(population_data):
    """
    计算所有个体的帕累托等级（非支配排序）
    
    :param population_data: [(objectives, mol, smiles), ...]
    :return: [(rank, objectives, mol, smiles), ...] 按等级排序
    """
    if not population_data:
        return []
    
    remaining = list(population_data)
    ranked = []
    current_rank = 0
    
    while remaining:
        front = compute_pareto_front(remaining)
        for individual in front:
            ranked.append((current_rank, individual[0], individual[1], individual[2]))
            remaining.remove(individual)
        current_rank += 1
    
    return ranked


def compute_crowding_distance(front):
    """
    计算拥挤距离（用于保持帕累托前沿的多样性）
    
    :param front: 帕累托前沿上的个体列表 [(objectives, mol, smiles), ...]
    :return: 每个个体的拥挤距离列表
    """
    if len(front) <= 2:
        return [float('inf')] * len(front)
    
    n = len(front)
    n_objectives = len(front[0][0])
    distances = [0.0] * n
    
    for m in range(n_objectives):
        # 按第 m 个目标排序
        sorted_indices = sorted(range(n), key=lambda i: front[i][0][m])
        
        # 边界点距离设为无穷大
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # 计算中间点的拥挤距离
        obj_min = front[sorted_indices[0]][0][m]
        obj_max = front[sorted_indices[-1]][0][m]
        obj_range = obj_max - obj_min if obj_max > obj_min else 1e-6
        
        for i in range(1, n - 1):
            prev_obj = front[sorted_indices[i - 1]][0][m]
            next_obj = front[sorted_indices[i + 1]][0][m]
            distances[sorted_indices[i]] += (next_obj - prev_obj) / obj_range
    
    return distances


def select_by_crowding(front, n_select):
    """
    使用拥挤距离选择个体
    
    :param front: 帕累托前沿上的个体列表
    :param n_select: 需要选择的个体数量
    :return: 选中的个体列表
    """
    if len(front) <= n_select:
        return front
    
    distances = compute_crowding_distance(front)
    sorted_indices = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
    return [front[i] for i in sorted_indices[:n_select]]


class BeliefSpaceMO:
    """
    多目标信念空间 (Multi-Objective Belief Space)
    
    【帕累托优化核心】
    维护帕累托前沿上的非支配解集，通过拥挤距离保持多样性。
    """
    
    def __init__(self, capacity=50, scaffold_threshold=0.5, min_scaffold_samples=2,
                 n_objectives=4):
        """
        初始化多目标信念空间
        
        :param capacity: 保留的完整分子数量上限（帕累托前沿容量）
        :param scaffold_threshold: 骨架激活阈值
        :param min_scaffold_samples: 骨架最少需要的样本数
        :param n_objectives: 目标函数数量
        """
        self.capacity = capacity
        self.scaffold_threshold = scaffold_threshold
        self.min_scaffold_samples = min_scaffold_samples
        self.n_objectives = n_objectives
        
        # 【核心数据结构】骨架知识库
        self.scaffold_memory = {}  # {scaffold_smiles: [(objectives, smiles), ...]}
        self.active_mappa = {}     # {scaffold_smiles: avg_hypervolume} 筛选后的高分骨架
        
        # 帕累托前沿分子库
        self.pareto_front = []  # [(objectives, mol, smiles), ...]
        
        # 完整分子库（用于交叉操作）
        self.knowledge_repository = []
        
        # 骨架指纹缓存
        self.scaffold_fingerprints = {}

    def _get_scaffold_fingerprint(self, scaffold_smi):
        """获取骨架的 Morgan 指纹"""
        if scaffold_smi in self.scaffold_fingerprints:
            return self.scaffold_fingerprints[scaffold_smi]
        
        try:
            mol = Chem.MolFromSmiles(scaffold_smi)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            self.scaffold_fingerprints[scaffold_smi] = fp
            return fp
        except Exception:
            return None

    def _extract_scaffold(self, mol):
        """从分子中提取 Murcko 骨架"""
        if mol is None:
            return None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                return None
            scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
            return scaffold_smi
        except Exception:
            return None

    def _compute_scalar_score(self, objectives):
        """
        计算标量分数（用于排序和比较）
        使用几何平均数作为聚合分数
        """
        if not objectives:
            return 0.0
        # 避免零值导致的问题
        objectives = [max(o, 1e-6) for o in objectives]
        return np.exp(np.mean(np.log(objectives)))

    def update(self, population_data, current_generation=0):
        """
        更新多目标信念空间
        
        :param population_data: [(objectives, mol, smiles), ...]
                               其中 objectives 是目标向量
        :param current_generation: 当前代数
        """
        valid_pop = [x for x in population_data if x is not None and x[1] is not None]
        
        # === 第一步：更新帕累托前沿 ===
        combined = self.pareto_front + valid_pop
        self.pareto_front = compute_pareto_front(combined)
        
        # 如果帕累托前沿太大，使用拥挤距离裁剪
        if len(self.pareto_front) > self.capacity:
            self.pareto_front = select_by_crowding(self.pareto_front, self.capacity)
        
        # === 第二步：提取骨架并更新骨架知识库 ===
        for objectives, mol, smi in valid_pop:
            scaffold_smi = self._extract_scaffold(mol)
            if scaffold_smi:
                if scaffold_smi not in self.scaffold_memory:
                    self.scaffold_memory[scaffold_smi] = []
                self.scaffold_memory[scaffold_smi].append((objectives, smi))
        
        # === 第三步：筛选活跃骨架 ===
        self.active_mappa = {}
        for scaffold_smi, records in self.scaffold_memory.items():
            if len(records) >= self.min_scaffold_samples:
                # 计算该骨架的平均标量分数
                avg_score = np.mean([self._compute_scalar_score(obj) for obj, _ in records])
                if avg_score >= self.scaffold_threshold:
                    self.active_mappa[scaffold_smi] = avg_score
        
        # === 第四步：维护完整分子库（用于交叉操作）===
        # 使用帕累托等级 + 拥挤距离排序
        ranked = compute_pareto_ranks(valid_pop)
        combined_repo = self.knowledge_repository + [
            (r[1], r[2], r[3]) for r in ranked  # (objectives, mol, smiles)
        ]
        
        # 按标量分数排序
        combined_repo = sorted(
            combined_repo, 
            key=lambda x: self._compute_scalar_score(x[0]), 
            reverse=True
        )
        
        seen = set()
        unique_combined = []
        for item in combined_repo:
            if item[2] not in seen:
                unique_combined.append(item)
                seen.add(item[2])
        self.knowledge_repository = unique_combined[:self.capacity]

    def get_best_scaffold(self, top_k=1):
        """获取最优骨架用于指导 BioT5 生成"""
        if not self.active_mappa:
            return []
        
        sorted_scaffolds = sorted(
            self.active_mappa.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(scaffold, score, score, 0, 0) for scaffold, score in sorted_scaffolds[:top_k]]

    def get_exemplars(self, n=2):
        """从完整分子库随机获取 n 个样本（用于交叉操作）"""
        if len(self.knowledge_repository) == 0:
            return []
        n = min(len(self.knowledge_repository), n)
        return random.sample(self.knowledge_repository, n)

    def get_pareto_exemplars(self, n=2):
        """从帕累托前沿获取样本"""
        if len(self.pareto_front) == 0:
            return []
        n = min(len(self.pareto_front), n)
        return random.sample(self.pareto_front, n)

    def get_best(self):
        """获取分数最高的完整分子（使用标量聚合分数）"""
        if self.knowledge_repository:
            best = max(self.knowledge_repository, key=lambda x: self._compute_scalar_score(x[0]))
            return (self._compute_scalar_score(best[0]), best[1], best[2], best[0])
        return (0.0, None, "", [])

    def get_pareto_front_size(self):
        """获取帕累托前沿大小"""
        return len(self.pareto_front)

    def get_pareto_front_stats(self):
        """获取帕累托前沿的统计信息"""
        if not self.pareto_front:
            return None
        
        objectives_matrix = np.array([item[0] for item in self.pareto_front])
        return {
            'size': len(self.pareto_front),
            'obj_means': objectives_matrix.mean(axis=0).tolist(),
            'obj_maxs': objectives_matrix.max(axis=0).tolist(),
            'obj_mins': objectives_matrix.min(axis=0).tolist()
        }

    def compute_hypervolume(self, reference_point=None):
        """
        计算帕累托前沿的超体积（Hypervolume）
        
        :param reference_point: 参考点，默认使用 [0, 0, ..., 0]
        :return: 超体积值
        """
        if not self.pareto_front:
            return 0.0
        
        if reference_point is None:
            reference_point = [0.0] * self.n_objectives
        
        # 简单的2D超体积计算（适用于目标数 <= 2）
        # 对于高维情况，这里使用近似计算
        try:
            from scipy.spatial import ConvexHull
            points = np.array([item[0] for item in self.pareto_front])
            # 添加参考点
            all_points = np.vstack([points, reference_point])
            hull = ConvexHull(all_points)
            return hull.volume
        except Exception:
            # 如果计算失败，返回简单的聚合分数
            return sum(self._compute_scalar_score(item[0]) for item in self.pareto_front)

    def get_belief_summary(self):
        """获取信念空间的摘要信息"""
        top_scaffolds = self.get_best_scaffold(3)
        
        pareto_stats = self.get_pareto_front_stats()
        
        return {
            'total_scaffolds': len(self.scaffold_memory),
            'active_beliefs': len(self.active_mappa),
            'top_scaffolds': [
                {
                    'scaffold': s[0],
                    'effective_score': s[1],
                    'raw_score': s[2],
                    'boredom_penalty': 0,
                    'similarity_penalty': 0,
                    'dominance_count': 0
                } for s in top_scaffolds
            ],
            'pareto_front_size': len(self.pareto_front),
            'pareto_stats': pareto_stats,
            'best_molecule_score': self.get_best()[0] if self.knowledge_repository else 0.0,
            'boredom_penalized_scaffolds': 0,
            'similarity_penalized_scaffolds': 0,
            'boredom_events_count': 0,
            'similarity_events_count': 0
        }
