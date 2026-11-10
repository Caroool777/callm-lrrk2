"""
信念空间 (Belief Space) 模块 - 文化算法的核心组件

【文化算法核心】
与简单的"精英保留"策略不同，真正的信念空间存储的是"知识"（骨架模式），
而不是"个体"（完整分子）。

实现的文化算法机制：
1. Outlining (概括): 使用 MurckoScaffold 提取分子核心骨架
2. 知识聚合: 统计每种骨架对应的平均分数
3. Pruning (修剪): 只保留高分骨架作为"信念"
4. Influence (影响): 提供最优骨架用于指导 BioT5 生成
5. Boredom (无聊度): 追踪骨架主导代数，防止过早收敛
6. Similarity Penalty (相似性惩罚): 对与最优骨架相似的骨架施加惩罚，促进多样性
"""

import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs


class BeliefSpace:
    """
    信念空间 (Belief Space) - 文化算法的核心组件
    
    【无聊度机制 (Boredom Mechanism)】
    防止算法过早收敛到单一骨架，通过以下方式实现多样性探索：
    - 追踪每个骨架主导的代数 (dominance_count)
    - 当骨架连续主导超过阈值时，施加惩罚 (boredom_penalty)
    - 惩罚会累积，迫使算法探索新骨架
    - 惩罚随时间衰减 (boredom_decay)，允许骨架"恢复名誉"
    
    【相似性惩罚机制 (Similarity Penalty)】
    防止算法在同一骨架家族内打转：
    - 计算骨架之间的 Tanimoto 相似度
    - 对与当前最优骨架高度相似的骨架施加惩罚
    - 鼓励探索结构差异更大的骨架
    
    【子结构约束模式 (Substructure Constraint Mode)】
    为特定任务（如 scaffold_hop）启用子结构约束而非骨架约束：
    - 指定必须包含的子结构 (required_substructure)
    - 分子必须包含该子结构才能获得高分
    - 用于 scaffold_hop 任务确保分子包含目标药效团 (benzothiazole)
    """
    
    def __init__(self, capacity=50, scaffold_threshold=0.5, min_scaffold_samples=2,
                 boredom_threshold=15, boredom_penalty=0.1, boredom_decay=0.01,
                 similarity_threshold=0.7, similarity_penalty=0.05,
                 use_substructure_constraint=False, required_substructure=None):
        """
        初始化信念空间
        
        :param capacity: 保留的完整分子数量上限（用于交叉操作）
        :param scaffold_threshold: 骨架激活阈值，平均分超过此值才进入 active_mappa
        :param min_scaffold_samples: 骨架最少需要的样本数，避免单个高分分子误导信念
        :param boredom_threshold: 触发无聊度惩罚的连续主导代数（默认15代）
        :param boredom_penalty: 惩罚分数，达到阈值后从骨架分数中扣除（默认0.1）
        :param boredom_decay: 惩罚衰减速度，非主导时每代恢复的分数（默认0.01）
        :param similarity_threshold: 相似性阈值，超过此值的骨架会被惩罚（默认0.7）
        :param similarity_penalty: 相似性惩罚系数，惩罚 = 系数 * (相似度 - 阈值)（默认0.05）
        :param use_substructure_constraint: 是否使用子结构约束模式（用于 scaffold_hop 等任务）
        :param required_substructure: 必须包含的子结构 SMARTS/SMILES（如 benzothiazole: c1ccc2ncsc2c1）
        """
        self.capacity = capacity
        self.scaffold_threshold = scaffold_threshold
        self.min_scaffold_samples = min_scaffold_samples
        
        # 【子结构约束模式】
        self.use_substructure_constraint = use_substructure_constraint
        self.required_substructure = required_substructure
        self.substructure_pattern = None
        if required_substructure:
            self.substructure_pattern = Chem.MolFromSmarts(required_substructure)
            if self.substructure_pattern is None:
                # 尝试作为 SMILES 解析
                self.substructure_pattern = Chem.MolFromSmiles(required_substructure)
        
        # 【核心数据结构】骨架知识库
        self.scaffold_memory = {}  # {scaffold_smiles: [score1, score2, ...]}
        self.active_mappa = {}     # {scaffold_smiles: avg_score} 筛选后的高分骨架
        
        # 完整分子库（用于交叉操作，兼容旧逻辑）
        self.knowledge_repository = []
        
        # 【无聊度机制】
        self.boredom_threshold = boredom_threshold
        self.boredom_penalty = boredom_penalty
        self.boredom_decay = boredom_decay
        
        self.scaffold_dominance_count = {}  # {scaffold: 连续主导代数}
        self.scaffold_penalty = {}          # {scaffold: 当前惩罚值}
        self.last_dominant_scaffold = None  # 上一代的主导骨架
        self.boredom_events = []            # 记录无聊度触发事件
        
        # 【相似性惩罚机制】
        self.similarity_threshold = similarity_threshold
        self.similarity_penalty = similarity_penalty
        self.scaffold_fingerprints = {}     # {scaffold_smiles: fingerprint} 缓存指纹
        self.similarity_penalty_cache = {}  # {scaffold: 当前相似性惩罚值}
        self.similarity_events = []         # 记录相似性惩罚事件
        
        # 【子结构约束统计】
        self.substructure_hits = 0          # 包含目标子结构的分子数
        self.substructure_misses = 0        # 不包含目标子结构的分子数

    def has_required_substructure(self, mol):
        """检查分子是否包含必需的子结构"""
        if not self.use_substructure_constraint or self.substructure_pattern is None:
            return True  # 未启用子结构约束，默认通过
        
        if mol is None:
            return False
        
        try:
            return mol.HasSubstructMatch(self.substructure_pattern)
        except Exception:
            return False

    def _get_scaffold_fingerprint(self, scaffold_smi):
        """获取骨架的 Morgan 指纹（用于相似性计算）"""
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

    def _calculate_similarity(self, scaffold1, scaffold2):
        """计算两个骨架之间的 Tanimoto 相似度"""
        fp1 = self._get_scaffold_fingerprint(scaffold1)
        fp2 = self._get_scaffold_fingerprint(scaffold2)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def _extract_scaffold(self, mol):
        """【Outlining 核心】从分子中提取 Murcko 骨架"""
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

    def update(self, population_data, current_generation=0):
        """更新信念空间 - 实现完整的文化算法知识提取流程
        
        【子结构约束模式】
        当启用子结构约束时，优先保留包含目标子结构的分子，
        同时禁用传统的骨架约束（因为 scaffold_hop 需要骨架多样性）
        """
        valid_pop = [x for x in population_data if x is not None and x[1] is not None]
        
        # === 【子结构约束模式】过滤和统计 ===
        if self.use_substructure_constraint and self.substructure_pattern is not None:
            # 分离包含/不包含目标子结构的分子
            with_substructure = []
            without_substructure = []
            for item in valid_pop:
                score, mol, smi = item
                if self.has_required_substructure(mol):
                    with_substructure.append(item)
                    self.substructure_hits += 1
                else:
                    without_substructure.append(item)
                    self.substructure_misses += 1
            
            # 优先使用包含子结构的分子，但也保留一些不包含的（保持多样性）
            if with_substructure:
                # 有包含子结构的分子，主要使用这些
                valid_pop = with_substructure
                # 可选：也加入一小部分不包含子结构的高分分子作为探索
                if without_substructure:
                    top_without = sorted(without_substructure, key=lambda x: x[0], reverse=True)[:5]
                    valid_pop = valid_pop + top_without
        
        # === 第一步：Outlining - 提取骨架并更新骨架知识库 ===
        for score, mol, smi in valid_pop:
            scaffold_smi = self._extract_scaffold(mol)
            if scaffold_smi:
                if scaffold_smi not in self.scaffold_memory:
                    self.scaffold_memory[scaffold_smi] = []
                self.scaffold_memory[scaffold_smi].append(score)
        
        # === 第二步：Pruning - 筛选高分骨架作为活跃信念 ===
        self.active_mappa = {}
        for scaffold_smi, scores in self.scaffold_memory.items():
            avg_score = np.mean(scores)
            if avg_score >= self.scaffold_threshold and len(scores) >= self.min_scaffold_samples:
                self.active_mappa[scaffold_smi] = avg_score
        
        # === 第三步【新增】：无聊度机制 - 惩罚长期主导的骨架 ===
        self._update_boredom(current_generation)
        
        # === 第四步【新增】：相似性惩罚 - 惩罚与最优骨架相似的骨架 ===
        self._update_similarity_penalty(current_generation)
        
        # === 第五步：同时维护完整分子库 ===
        combined = self.knowledge_repository + valid_pop
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
        seen = set()
        unique_combined = []
        for item in combined:
            if item[2] not in seen:
                unique_combined.append(item)
                seen.add(item[2])
        self.knowledge_repository = unique_combined[:self.capacity]

    def _update_boredom(self, current_generation):
        """【无聊度机制核心】更新骨架的无聊度状态"""
        if not self.active_mappa:
            return
        
        # 计算每个骨架的有效分数（原始分数 - 惩罚）
        effective_scores = {}
        for scaffold, raw_score in self.active_mappa.items():
            penalty = self.scaffold_penalty.get(scaffold, 0.0)
            effective_scores[scaffold] = raw_score - penalty
        
        # 找出当前有效分数最高的骨架（主导骨架）
        current_dominant = max(effective_scores.items(), key=lambda x: x[1])[0]
        
        # 更新主导计数
        if current_dominant == self.last_dominant_scaffold:
            self.scaffold_dominance_count[current_dominant] = \
                self.scaffold_dominance_count.get(current_dominant, 0) + 1
        else:
            self.scaffold_dominance_count[current_dominant] = 1
        
        # 检查是否触发无聊度惩罚
        dominance_count = self.scaffold_dominance_count.get(current_dominant, 0)
        if dominance_count >= self.boredom_threshold:
            current_penalty = self.scaffold_penalty.get(current_dominant, 0.0)
            new_penalty = current_penalty + self.boredom_penalty
            self.scaffold_penalty[current_dominant] = new_penalty
            
            self.boredom_events.append({
                'generation': current_generation,
                'scaffold': current_dominant,
                'dominance_count': dominance_count,
                'new_penalty': new_penalty,
                'raw_score': self.active_mappa.get(current_dominant, 0),
                'effective_score': self.active_mappa.get(current_dominant, 0) - new_penalty
            })
            
            print(f"  ⚠️ [Boredom] 骨架 '{current_dominant[:30]}...' 已主导 {dominance_count} 代，"
                  f"施加惩罚 {new_penalty:.3f}")
            
            # 重置主导计数
            self.scaffold_dominance_count[current_dominant] = 0
        
        # 对所有非当前主导的骨架，衰减惩罚
        for scaffold in list(self.scaffold_penalty.keys()):
            if scaffold != current_dominant:
                current_penalty = self.scaffold_penalty[scaffold]
                new_penalty = max(0.0, current_penalty - self.boredom_decay)
                if new_penalty > 0:
                    self.scaffold_penalty[scaffold] = new_penalty
                else:
                    del self.scaffold_penalty[scaffold]
        
        self.last_dominant_scaffold = current_dominant

    def _update_similarity_penalty(self, current_generation):
        """【相似性惩罚机制核心】更新骨架的相似性惩罚"""
        if not self.active_mappa or len(self.active_mappa) < 2:
            return
        
        # 找出当前原始分数最高的骨架（不考虑任何惩罚）
        top_scaffold = max(self.active_mappa.items(), key=lambda x: x[1])[0]
        top_score = self.active_mappa[top_scaffold]
        
        # 重新计算所有骨架的相似性惩罚
        new_similarity_penalties = {}
        similarity_penalized_count = 0
        
        for scaffold, raw_score in self.active_mappa.items():
            if scaffold == top_scaffold:
                # 最优骨架不惩罚自己
                new_similarity_penalties[scaffold] = 0.0
                continue
            
            # 计算与最优骨架的相似度
            similarity = self._calculate_similarity(scaffold, top_scaffold)
            
            if similarity > self.similarity_threshold:
                # 相似度超过阈值，施加惩罚
                # 惩罚 = 系数 * (相似度 - 阈值) / (1 - 阈值) * 原始分数
                # 这样惩罚与相似度成正比，且考虑了骨架本身的分数
                penalty_factor = (similarity - self.similarity_threshold) / (1 - self.similarity_threshold)
                penalty = self.similarity_penalty * penalty_factor * raw_score
                new_similarity_penalties[scaffold] = penalty
                similarity_penalized_count += 1
            else:
                new_similarity_penalties[scaffold] = 0.0
        
        # 更新缓存
        self.similarity_penalty_cache = new_similarity_penalties
        
        # 记录事件（仅当有显著变化时）
        if similarity_penalized_count > 0 and current_generation % 20 == 0:
            self.similarity_events.append({
                'generation': current_generation,
                'top_scaffold': top_scaffold[:50],
                'penalized_count': similarity_penalized_count,
                'threshold': self.similarity_threshold
            })

    def get_effective_score(self, scaffold):
        """获取骨架的有效分数（原始分数 - 无聊度惩罚 - 相似性惩罚）"""
        raw_score = self.active_mappa.get(scaffold, 0.0)
        boredom_penalty = self.scaffold_penalty.get(scaffold, 0.0)
        similarity_penalty = self.similarity_penalty_cache.get(scaffold, 0.0)
        return raw_score - boredom_penalty - similarity_penalty

    def get_best_scaffold(self, top_k=1):
        """【Influence 核心】获取最优骨架用于指导 BioT5 生成（考虑无聊度惩罚和相似性惩罚）"""
        if not self.active_mappa:
            return []
        
        effective_scaffolds = []
        for scaffold, raw_score in self.active_mappa.items():
            effective_score = self.get_effective_score(scaffold)
            boredom_pen = self.scaffold_penalty.get(scaffold, 0.0)
            similarity_pen = self.similarity_penalty_cache.get(scaffold, 0.0)
            effective_scaffolds.append((scaffold, effective_score, raw_score, boredom_pen, similarity_pen))
        
        sorted_scaffolds = sorted(effective_scaffolds, key=lambda x: x[1], reverse=True)
        
        # 返回格式：(scaffold, effective_score, raw_score, boredom_penalty, similarity_penalty)
        return sorted_scaffolds[:top_k]

    def get_exemplars(self, n=2):
        """从完整分子库随机获取 n 个样本（用于交叉操作）"""
        if len(self.knowledge_repository) == 0:
            return []
        n = min(len(self.knowledge_repository), n)
        return random.sample(self.knowledge_repository, n)

    def get_best(self):
        """获取分数最高的完整分子"""
        if self.knowledge_repository:
            return self.knowledge_repository[0]
        return (0.0, None, "")
    
    def get_substructure_constraint(self):
        """获取子结构约束信息（用于 BioT5 prompt）
        
        返回必须包含的子结构 SMILES/SMARTS，用于指导 BioT5 生成
        """
        if self.use_substructure_constraint and self.required_substructure:
            return self.required_substructure
        return None
    
    def get_substructure_stats(self):
        """获取子结构约束统计信息"""
        return {
            'enabled': self.use_substructure_constraint,
            'pattern': self.required_substructure,
            'hits': self.substructure_hits,
            'misses': self.substructure_misses,
            'hit_rate': self.substructure_hits / max(1, self.substructure_hits + self.substructure_misses)
        }
    
    def get_belief_summary(self):
        """获取信念空间的摘要信息（包含无聊度和相似性惩罚状态）"""
        top_scaffolds = self.get_best_scaffold(3)
        
        top_scaffold_info = []
        for item in top_scaffolds:
            scaffold, effective_score, raw_score, boredom_pen, similarity_pen = item
            dominance = self.scaffold_dominance_count.get(scaffold, 0)
            top_scaffold_info.append({
                'scaffold': scaffold,
                'effective_score': effective_score,
                'raw_score': raw_score,
                'boredom_penalty': boredom_pen,
                'similarity_penalty': similarity_pen,
                'dominance_count': dominance
            })
        
        # 统计被相似性惩罚的骨架数量
        similarity_penalized_count = sum(1 for v in self.similarity_penalty_cache.values() if v > 0)
        
        # 子结构约束统计
        substructure_stats = self.get_substructure_stats() if self.use_substructure_constraint else None
        
        return {
            'total_scaffolds': len(self.scaffold_memory),
            'active_beliefs': len(self.active_mappa),
            'top_scaffolds': top_scaffold_info,
            'best_molecule_score': self.get_best()[0] if self.knowledge_repository else 0.0,
            'boredom_penalized_scaffolds': len(self.scaffold_penalty),
            'similarity_penalized_scaffolds': similarity_penalized_count,
            'boredom_events_count': len(self.boredom_events),
            'similarity_events_count': len(self.similarity_events),
            'substructure_stats': substructure_stats
        }
