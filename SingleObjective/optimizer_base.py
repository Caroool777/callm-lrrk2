"""
文化算法优化器基类 (Cultural Algorithm Base)

提供分子优化的通用框架，包括：
- Oracle 评估与缓存
- AUC 计算
- BioT5 集成
- 信念空间管理
- 无聊度机制
- 配置加载与应用
- Top-K 分子图输出

子类只需实现任务特定的配置即可。
"""

import bisect
import json
import random
import os
import time
import numpy as np
import pandas as pd
from abc import ABC
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from belief_space import BeliefSpace
from llm_backends import create_molecule_editor, LLMBackendFactory
import crossover as co
import mutate as mu


def load_config(task_name=None, base_dir=None):
    """
    加载配置，支持继承机制
    
    :param task_name: 任务名称 (如 "jnk3", "drd2", "gsk3b", "qed")
    :param base_dir: 配置文件所在目录
    :return: 合并后的配置字典
    """
    import json
    
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    
    # 1. 加载基础配置
    base_config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 已加载基础配置: {base_config_path}")
    else:
        config = {}
    
    # 2. 合并任务特定配置
    if task_name:
        task_config_path = os.path.join(base_dir, f"config_{task_name}.json")
        if os.path.exists(task_config_path):
            with open(task_config_path, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            config.update(task_config)
            print(f"✅ 已合并任务配置: {task_config_path}")
    
    override_path = os.environ.get("CALLM_ABLATION_CONFIG")
    if override_path and os.path.exists(override_path):
        with open(override_path, 'r', encoding='utf-8') as f:
            override_config = json.load(f)
        config.update(override_config)
        print(f"Loaded ablation override config: {override_path}")
    
    return config


def load_initial_population(config, base_dir=None):
    """
    从配置加载初始种群
    
    :param config: 配置字典
    :param base_dir: 基础目录
    :return: SMILES 列表
    """
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    
    smiles_file = config.get("initial_smiles_file", "../250k_rndm_zinc_drugs_clean_3.csv")
    if not os.path.isabs(smiles_file):
        smiles_file = os.path.join(base_dir, smiles_file)
    
    pop_size = config.get("pop_size", 100)
    
    if os.path.exists(smiles_file):
        print(f"从文件加载初始分子: {smiles_file}")
        df = pd.read_csv(smiles_file)
        if 'smiles' in df.columns:
            all_smiles = df['smiles'].tolist()
        else:
            all_smiles = df.iloc[:, 0].tolist()
        initial_pop = random.sample(all_smiles, min(pop_size, len(all_smiles)))
        print(f"已加载 {len(initial_pop)} 个初始分子")
        return initial_pop
    else:
        print(f"警告: 找不到文件 {smiles_file}，使用默认初始种群")
        return [
            "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2",
            "CCOc1ccc(cc1)NC(=O)C",
            "CCN(CC)C(=O)Cc1ccc(cc1)Cl",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        ]


class CulturalAlgorithmBase(ABC):
    """
    文化算法优化器基类
    
    【文化算法核心】
    结合 BioT5 分子编辑模型、骨架知识提取、动态 Prompt 约束，
    在分子空间中搜索具有高活性的分子。
    
    【无聊度机制】
    引入无聊度机制防止算法过早收敛到单一骨架，
    当某骨架连续主导多代后施加惩罚，促进探索多样性。
    """
    
    # 子类需要设置的任务名称
    TASK_NAME = "base"
    VALID_ABLATION_MODES = {
        "full",
        "no_belief_space",
        "no_scaffold_influence",
        "no_llm",
        "no_biot5",
        "no_ga_operators",
        "no_diversity_penalty",
    }
    
    def __init__(self, oracle_func, task_name=None, max_oracle_calls=500,
                 boredom_threshold=15, boredom_penalty=0.1, boredom_decay=0.01,
                 similarity_threshold=0.7, similarity_penalty=0.05, device=None,
                 ablation_mode="full"):
        """
        初始化优化器
        
        :param oracle_func: Oracle 评分函数
        :param task_name: 任务名称（用于 BioT5 prompt）
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param boredom_threshold: 无聊度阈值
        :param boredom_penalty: 无聊度惩罚值
        :param boredom_decay: 惩罚衰减速度
        :param similarity_threshold: 相似性阈值（0-1），超过此值的骨架会被惩罚
        :param similarity_penalty: 相似性惩罚系数
        :param device: 指定 GPU 设备 (如 "cuda:0", "cuda:1")，默认自动选择
        """
        self.oracle = oracle_func
        self.task_name = task_name or self.TASK_NAME
        self.max_oracle_calls = max_oracle_calls
        self.current_oracle_calls = 0
        self.ablation_mode = self._normalize_ablation_mode(ablation_mode)
        self.seed = None
        self.run_config = {}
        self.runtime_seconds = 0.0
        self._validate_ablation_mode()
        self.device = device  # 保存设备信息
        
        # 无聊度机制参数
        self.boredom_threshold = boredom_threshold
        self.boredom_penalty = boredom_penalty
        self.boredom_decay = boredom_decay
        
        # 相似性惩罚参数
        self.similarity_threshold = similarity_threshold
        self.similarity_penalty = similarity_penalty
        
        # 初始化 LLM 后端 - 使用多样性生成配置
        self.llm_editor = create_molecule_editor(
            backend=LLMBackendFactory.get_default_backend(),
            task=self.task_name,
            num_return_sequences=3,
            num_beams=5,
            temperature=0.8,
            do_sample=True,
            device=device
        )
        
        # BioT5 每代处理的精英分子数量
        self.llm_editor_count = 10
        
        # 【文化算法核心】初始化信念空间
        self.belief_space = BeliefSpace(
            capacity=20,
            scaffold_threshold=0.5,
            min_scaffold_samples=2,
            boredom_threshold=boredom_threshold,
            boredom_penalty=boredom_penalty,
            boredom_decay=boredom_decay,
            similarity_threshold=similarity_threshold,
            similarity_penalty=similarity_penalty
        )
        
        self.population_size = 50
        self.offspring_size = 30
        self.mutation_rate = 0.5
        
        # Oracle 缓存，避免重复评估
        self.score_cache = {}
        
        # 记录所有经 Oracle 评估过的唯一分子，用于 AUC / Top-K 统计
        # 这里保留 score=0 的分子，避免 top-100 AUC 在早期被异常抬高。
        self.all_discovered = []  # [(score, canonical_smiles, oracle_call_idx), ...]
        
        # 记录收敛曲线数据：(oracle_calls, top_10_avg_score)
        self.convergence_data = []  # [(oracle_calls, avg_score), ...]
        self.convergence_log_freq = 10  # 每隔多少次记录一次
        
        # 是否启用骨架约束的 Prompt（文化算法 Influence）
        self.use_scaffold_constraint = True
        
        # 【子结构约束模式】用于 scaffold_hop 等任务
        self.use_substructure_constraint = False
        self.required_substructure = None

    def apply_config(self, config):
        """
        从配置字典应用参数
        
        :param config: 配置字典
        """
        # ChatGPT/OpenAI 参数
        self.run_config = dict(config)
        self.ablation_mode = self._normalize_ablation_mode(config.get("ablation_mode", "full"))
        self.seed = config.get("seed", None)
        self.max_oracle_calls = config.get("max_oracle_calls", self.max_oracle_calls)
        self._validate_ablation_mode()

        self.llm_editor = LLMBackendFactory.create_from_config(config)
        self.llm_editor.set_task(self.task_name)
        self.llm_editor_count = config.get(
            "llm_elite_count",
            config.get("llm_editor_count", config.get("biot5_elite_count", self.llm_editor_count))
        )
        
        # 种群参数
        self.population_size = config.get("pop_size", self.population_size)
        self.offspring_size = int(self.population_size * config.get("crossover_rate", 0.8))
        
        # 信念空间配置（支持新旧两种键名）
        belief_capacity = config.get("belief_capacity", self.population_size // 2)
        scaffold_threshold = config.get("scaffold_threshold", 
                                        config.get("belief_space_scaffold_threshold", 0.5))
        min_scaffold_samples = config.get("min_scaffold_samples", 2)
        
        # 相似性惩罚配置
        self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
        self.similarity_penalty = config.get("similarity_penalty", self.similarity_penalty)
        
        # 无聊度机制配置（从配置文件读取）
        self.boredom_threshold = config.get("boredom_threshold", self.boredom_threshold)
        self.boredom_penalty = config.get("boredom_penalty", self.boredom_penalty)
        self.boredom_decay = config.get("boredom_decay", self.boredom_decay)
        belief_boredom_penalty = 0.0 if self.ablation_mode == "no_diversity_penalty" else self.boredom_penalty
        belief_similarity_penalty = 0.0 if self.ablation_mode == "no_diversity_penalty" else self.similarity_penalty
        
        # 变异率配置（从配置文件读取）
        self.mutation_rate = config.get("mutation_rate", self.mutation_rate)
        
        # 【子结构约束模式】用于 scaffold_hop 等任务
        self.use_substructure_constraint = config.get("use_substructure_constraint", False)
        self.required_substructure = config.get("required_substructure", None)
        
        self.belief_space = BeliefSpace(
            capacity=belief_capacity,
            scaffold_threshold=scaffold_threshold,
            min_scaffold_samples=min_scaffold_samples,
            boredom_threshold=self.boredom_threshold,
            boredom_penalty=belief_boredom_penalty,
            boredom_decay=self.boredom_decay,
            similarity_threshold=self.similarity_threshold,
            similarity_penalty=belief_similarity_penalty,
            use_substructure_constraint=self.use_substructure_constraint,
            required_substructure=self.required_substructure
        )
        
        self.use_scaffold_constraint = config.get("use_scaffold_constraint", True)
        
        # 打印配置摘要
        print(f"LLM Editor 参数: elite_count={self.llm_editor_count}")
        print(f"信念空间: capacity={belief_capacity}, "
              f"scaffold_threshold={scaffold_threshold}, "
              f"min_samples={min_scaffold_samples}, "
              f"use_constraint={self.use_scaffold_constraint}")
        print(f"无聊度机制: threshold={self.boredom_threshold}, "
              f"penalty={self.boredom_penalty}, "
              f"decay={self.boredom_decay}")
        print(f"相似性惩罚: threshold={self.similarity_threshold}, "
              f"penalty={self.similarity_penalty}")
        print(f"Ablation mode: {self.ablation_mode}")
        if self.ablation_mode == "no_diversity_penalty":
            print("Diversity penalties disabled: boredom_penalty=0, similarity_penalty=0")
        print(f"变异率: mutation_rate={self.mutation_rate}")
        
        # 【子结构约束模式】打印配置
        if self.use_substructure_constraint:
            print(f"🔬 子结构约束模式: 已启用")
            print(f"   必需子结构: {self.required_substructure}")
            print(f"   骨架约束: {'禁用 (使用子结构约束代替)' if not self.use_scaffold_constraint else '启用'}")

    def _validate_ablation_mode(self):
        if self.ablation_mode not in self.VALID_ABLATION_MODES:
            raise ValueError(
                f"Unsupported single-objective ablation_mode={self.ablation_mode}. "
                f"Valid modes: {sorted(self.VALID_ABLATION_MODES)}"
            )

    def _normalize_ablation_mode(self, mode):
        if mode == "no_biot5":
            return "no_llm"
        return mode or "full"

    def _canonicalize_mol(self, mol):
        if mol is None:
            return None
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
            return smi or None
        except Exception:
            return None

    def _deduplicate_valid_mols(self, mols):
        """Remove None, invalid RDKit molecules, and duplicate canonical SMILES before oracle evaluation."""
        unique = []
        seen = set()
        for mol in mols:
            smi = self._canonicalize_mol(mol)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            unique.append(mol)
        return unique

    def _data_from_all_discovered(self):
        rows = []
        for score, smi, _ in self.all_discovered:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rows.append((float(score), mol, Chem.MolToSmiles(mol, canonical=True)))
        return rows

    def _get_single_selection_pool(self, population_data, k=None):
        """Build the selection pool for each single-objective ablation mode.

        no_belief_space bypasses belief-space memory and uses a transient top-score
        pool from current population plus all_discovered. Other modes prefer the
        belief-space repository and fall back to the current population.
        """
        if self.ablation_mode == "no_belief_space":
            pool = list(population_data or []) + self._data_from_all_discovered()
        else:
            pool = list(getattr(self.belief_space, "knowledge_repository", []) or [])
            if not pool:
                pool = list(population_data or [])
        pool = [x for x in pool if x is not None and x[1] is not None]
        pool = sorted(pool, key=lambda x: float(x[0]), reverse=True)
        seen = set()
        unique = []
        for item in pool:
            if item[2] in seen:
                continue
            seen.add(item[2])
            unique.append(item)
        return unique[:k] if k is not None else unique

    def _select_single_parents_for_crossover(self, population_data, n_parents=2):
        """Select crossover parents, with GA disabled in no_ga_operators mode."""
        if self.ablation_mode == "no_ga_operators":
            return []
        if self.ablation_mode == "no_belief_space":
            pool = self._get_single_selection_pool(population_data)
            return random.sample(pool, n_parents) if len(pool) >= n_parents else []
        parents = self.belief_space.get_exemplars(n_parents)
        if len(parents) < n_parents:
            pool = self._get_single_selection_pool(population_data)
            return random.sample(pool, n_parents) if len(pool) >= n_parents else []
        return parents

    def _select_single_elites_for_llm(self, population_data, elite_count):
        """Select LLM elites according to the active ablation mode."""
        if self.ablation_mode == "no_llm":
            return []
        if self.ablation_mode == "no_belief_space":
            return sorted(list(population_data or []), key=lambda x: float(x[0]), reverse=True)[:elite_count]
        elites = self.belief_space.get_exemplars(elite_count)
        if not elites:
            elites = self._get_single_selection_pool(population_data, elite_count)
        return elites

    def _get_single_scaffold_constraint_for_llm(self):
        """Return LLM scaffold/substructure constraint, or None when that influence is ablated."""
        if self.ablation_mode in {"no_belief_space", "no_scaffold_influence", "no_llm"}:
            return None
        if self.use_substructure_constraint and self.required_substructure:
            print(f"  [Substructure] Using substructure constraint: {self.required_substructure}")
            return self.required_substructure
        if self.use_scaffold_constraint:
            top_scaffolds = self.belief_space.get_best_scaffold(top_k=1)
            if top_scaffolds:
                constraint = top_scaffolds[0][0]
                print(f"  [Scaffold] Using scaffold constraint: {constraint[:40]}...")
                return constraint
        return None

    def _get_best_single_result(self, population_data=None):
        if self.ablation_mode != "no_belief_space":
            return self.belief_space.get_best()
        pool = self._get_single_selection_pool(population_data)
        return pool[0] if pool else (0.0, None, "")

    def evaluate(self, mols):
        """评估分子列表，返回 (score, mol, smiles) 元组列表

        使用 Canonical SMILES 作为缓存键，确保同一分子不被重复评估。
        同时严格遵守 max_oracle_calls 预算，不允许 batch 内越界。
        """
        results = []
        for mol in mols:
            if mol is None:
                continue

            try:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                if not canonical_smi:
                    continue

                if canonical_smi in self.score_cache:
                    score = self.score_cache[canonical_smi]
                else:
                    if self.current_oracle_calls >= self.max_oracle_calls:
                        break

                    score = self.oracle.score(canonical_smi)
                    self.score_cache[canonical_smi] = score
                    self.current_oracle_calls += 1

                    # 记录所有真正调用过 Oracle 的唯一分子，包括 score=0。
                    self.all_discovered.append((float(score), canonical_smi, self.current_oracle_calls))

                    if self.current_oracle_calls % self.convergence_log_freq == 0 or self.current_oracle_calls == 1:
                        self._log_convergence_point()

                results.append((score, mol, canonical_smi))
            except Exception as e:
                print(f"Eval Error: {e}")
                continue

        return results
    
    def _log_convergence_point(self):
        """记录当前的收敛曲线数据点。

        与 PMO 的日志口径一致：对当前已有分子中最好的 min(k, n)
        个分子取均值。正式 AUC 由 compute_top_k_auc 按每 100 次调用
        的 PMO 官方采样规则独立计算。
        """
        if len(self.all_discovered) == 0:
            return

        all_scores = sorted((float(item[0]) for item in self.all_discovered), reverse=True)
        top_1 = all_scores[0] if all_scores else 0.0
        top_10 = float(np.mean(all_scores[:10]))
        top_100 = float(np.mean(all_scores[:100]))

        self.convergence_data.append((self.current_oracle_calls, top_1, top_10, top_100))

    def compute_top_k_auc(self, k, freq_log=100, finish=True):
        """Compute Top-K AUC exactly as PMO mol_opt/main/optimizer.py::top_auc.

        PMO samples every freq_log unique valid molecules, integrates the
        running Top-K mean with the trapezoid rule, and normalizes by the fixed
        oracle budget. Formal reporting passes finish=True.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if freq_log <= 0:
            raise ValueError("freq_log must be a positive integer")
        if len(self.all_discovered) == 0 or self.max_oracle_calls <= 0:
            return 0.0

        max_calls = int(self.max_oracle_calls)
        ordered = sorted(self.all_discovered, key=lambda item: item[2])[:max_calls]
        n_evaluated = len(ordered)
        auc_sum = 0.0
        previous = 0.0
        called = 0

        for idx in range(freq_log, n_evaluated, freq_log):
            scores = sorted(
                (float(item[0]) for item in ordered[:idx]), reverse=True
            )[:k]
            top_k_now = float(np.mean(scores))
            auc_sum += freq_log * (top_k_now + previous) / 2.0
            previous = top_k_now
            called = idx

        final_scores = sorted(
            (float(item[0]) for item in ordered), reverse=True
        )[:k]
        top_k_now = float(np.mean(final_scores)) if final_scores else 0.0
        auc_sum += (n_evaluated - called) * (top_k_now + previous) / 2.0
        if finish and n_evaluated < max_calls:
            auc_sum += (max_calls - n_evaluated) * top_k_now
        return float(auc_sum / max_calls)

    def get_top_k_molecules(self, k=10):
        """
        获取打分最高的 K 个分子
        
        :param k: 返回的分子数量
        :return: [(score, smiles), ...] 列表，按分数降序排列
        
        注意：使用 Canonical SMILES 进行去重，避免同一分子的不同表示被重复计入
        """
        if len(self.all_discovered) == 0:
            return []
        
        # 按分数降序排序
        sorted_mols = sorted(self.all_discovered, key=lambda x: x[0], reverse=True)
        
        # 使用 Canonical SMILES 去重
        seen_canonical = set()
        unique_top = []
        for score, smi, _ in sorted_mols:
            # 转换为 Canonical SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            
            if canonical_smi not in seen_canonical:
                unique_top.append((score, canonical_smi))  # 返回规范化的 SMILES
                seen_canonical.add(canonical_smi)
                if len(unique_top) >= k:
                    break
        
        return unique_top

    def _run_ablation_impl(self, initial_smiles_list):
        """Run single-objective CALLM with ablation-mode switches."""
        start_time = time.time()
        print(f"Initializing Population... ablation_mode={self.ablation_mode}")
        initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles_list if s]
        initial_mols = self._deduplicate_valid_mols(initial_mols)

        if not initial_mols:
            print("Error: No valid molecules in initial population!")
            return (0.0, None, ""), 0.0, 0.0

        print(f"Valid initial molecules: {len(initial_mols)}")
        population_data = self.evaluate(initial_mols)
        if not population_data:
            print("Error: Failed to evaluate initial population!")
            return (0.0, None, ""), 0.0, 0.0

        if self.ablation_mode != "no_belief_space":
            self.belief_space.update(population_data, current_generation=0)
        else:
            pool = self._get_single_selection_pool(population_data)
            best_score = pool[0][0] if pool else 0.0
            print(f"[no_belief_space] transient pool size={len(pool)} best_score={best_score:.4f}")

        generation = 0
        max_stale_generations = 100
        stale_count = 0
        last_calls = 0
        use_crossover = self.ablation_mode != "no_ga_operators"
        use_llm = self.ablation_mode != "no_llm"
        use_random_mutation = self.ablation_mode != "no_ga_operators"

        while self.current_oracle_calls < self.max_oracle_calls:
            best_score = self._get_best_single_result(population_data)[0]
            if self.ablation_mode == "no_belief_space":
                pool = self._get_single_selection_pool(population_data)
                print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                      f"Best: {best_score:.4f} | Transient Pool: {len(pool)}")
            else:
                belief_summary = self.belief_space.get_belief_summary()
                penalty_info = ""
                if belief_summary['boredom_penalized_scaffolds'] > 0 or belief_summary['similarity_penalized_scaffolds'] > 0:
                    penalty_info = f" | Boredom: {belief_summary['boredom_penalized_scaffolds']}, SimPen: {belief_summary['similarity_penalized_scaffolds']}"
                print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                      f"Best: {best_score:.4f} | Active Beliefs: {belief_summary['active_beliefs']}{penalty_info}")

            offspring_mols = []
            if use_crossover:
                crossover_attempts = self.offspring_size if self.ablation_mode == "no_llm" else max(1, self.offspring_size // 3)
                for _ in range(crossover_attempts):
                    parents_data = self._select_single_parents_for_crossover(population_data, 2)
                    if len(parents_data) == 2:
                        child = co.crossover(parents_data[0][1], parents_data[1][1])
                        if child:
                            offspring_mols.append(child)

            if use_llm:
                rounds = 3 if self.ablation_mode == "no_ga_operators" else 1
                for round_idx in range(rounds):
                    elite_count = min(max(self.llm_editor_count * (round_idx + 1), 1), max(len(population_data), 1))
                    elite_data = self._select_single_elites_for_llm(population_data, elite_count)
                    if not elite_data:
                        break
                    elites = [item[1] for item in elite_data if item[1] is not None]
                    constraint_for_llm = self._get_single_scaffold_constraint_for_llm()
                    try:
                        mutated_elites = self.llm_editor.edit(elites, scaffold_constraints=constraint_for_llm)
                        mutated_elites = [m for m in mutated_elites if m is not None]
                        print(f"  ChatGPT generated {len(mutated_elites)} candidates (input {len(elites)})")
                        offspring_mols.extend(mutated_elites)
                    except Exception as e:
                        print(f"ChatGPT error: {e}")
                    if len(offspring_mols) >= self.offspring_size:
                        break

            if use_random_mutation:
                retry_limit = self.offspring_size * (6 if self.ablation_mode == "no_llm" else 3)
                retries = 0
                while len(offspring_mols) < self.offspring_size and retries < retry_limit:
                    parents = self._get_single_selection_pool(population_data)
                    if not parents:
                        print("Warning: random mutation fallback has no parents.")
                        break
                    parent = random.choice(parents)
                    child = mu.mutate(parent[1], mutation_rate=self.mutation_rate)
                    if child:
                        offspring_mols.append(child)
                    retries += 1

            offspring_mols = self._deduplicate_valid_mols(offspring_mols)
            if not offspring_mols:
                print(f"Warning: empty offspring in {self.ablation_mode}; falling back to current top molecules.")
                offspring_mols = [item[1] for item in self._get_single_selection_pool(population_data, self.offspring_size)]

            offspring_data = self.evaluate(offspring_mols)
            if offspring_data and self.ablation_mode != "no_belief_space":
                self.belief_space.update(offspring_data, current_generation=generation)

            if self.current_oracle_calls == last_calls:
                stale_count += 1
                if stale_count >= max_stale_generations:
                    print(f"Warning: No progress for {max_stale_generations} generations. Stopping.")
                    break
            else:
                stale_count = 0
                last_calls = self.current_oracle_calls

            if self.ablation_mode == "no_belief_space":
                next_gen_candidates = offspring_data + self._get_single_selection_pool(population_data)
            else:
                next_gen_candidates = offspring_data + self.belief_space.knowledge_repository
            next_gen_candidates = sorted(next_gen_candidates, key=lambda x: x[0], reverse=True)
            population_data = next_gen_candidates[:self.population_size] or population_data
            generation += 1

        self._log_convergence_point()
        top_10_auc = self.compute_top_k_auc(10)
        top_100_auc = self.compute_top_k_auc(100)
        self.runtime_seconds = time.time() - start_time
        self._print_final_summary()
        return self._get_best_single_result(population_data), top_10_auc, top_100_auc

    def run(self, initial_smiles_list):
        """运行优化算法"""
        return self._run_ablation_impl(initial_smiles_list)
        print("Initializing Population...")
        initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles_list if s]
        initial_mols = [m for m in initial_mols if m is not None]
        
        if not initial_mols:
            print("Error: No valid molecules in initial population!")
            return (0.0, None, ""), 0.0, 0.0
        
        print(f"Valid initial molecules: {len(initial_mols)}")
        population_data = self.evaluate(initial_mols)
        
        if not population_data:
            print("Error: Failed to evaluate initial population!")
            return (0.0, None, ""), 0.0, 0.0
        
        self.belief_space.update(population_data, current_generation=0)
        
        generation = 0
        max_stale_generations = 100
        stale_count = 0
        last_calls = 0
        
        while self.current_oracle_calls < self.max_oracle_calls:
            best_score = self.belief_space.get_best()[0]
            
            # 【文化算法】打印信念空间摘要
            belief_summary = self.belief_space.get_belief_summary()
            
            # 构建惩罚状态信息
            penalty_info = ""
            if belief_summary['boredom_penalized_scaffolds'] > 0 or belief_summary['similarity_penalized_scaffolds'] > 0:
                penalty_info = f" | Boredom: {belief_summary['boredom_penalized_scaffolds']}, SimPen: {belief_summary['similarity_penalized_scaffolds']}"
            
            print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                  f"Best: {best_score:.4f} | Active Beliefs: {belief_summary['active_beliefs']}{penalty_info}")

            offspring_mols = []
            
            # --- 策略 A: Crossover ---
            for _ in range(self.offspring_size // 3):
                parents_data = self.belief_space.get_exemplars(2)
                if len(parents_data) == 2:
                    child = co.crossover(parents_data[0][1], parents_data[1][1])
                    if child:
                        offspring_mols.append(child)

            # --- 策略 B: BioT5 Mutation with Scaffold/Substructure Constraint ---
            elite_count = min(self.llm_editor_count, len(self.belief_space.knowledge_repository))
            elite_data = self.belief_space.get_exemplars(elite_count)
            
            if elite_data:
                elites = [item[1] for item in elite_data]
                
                # 【子结构约束模式】优先使用子结构约束
                constraint_for_llm = None
                
                if self.use_substructure_constraint and self.required_substructure:
                    # 使用子结构约束（如 benzothiazole）
                    constraint_for_llm = self.required_substructure
                    print(f"  [Substructure] 使用子结构约束: {constraint_for_llm}")
                elif self.use_scaffold_constraint:
                    # 使用传统骨架约束
                    top_scaffolds = self.belief_space.get_best_scaffold(top_k=1)
                    if top_scaffolds:
                        constraint_for_llm = top_scaffolds[0][0]
                        effective_score = top_scaffolds[0][1]
                        raw_score = top_scaffolds[0][2]
                        boredom_pen = top_scaffolds[0][3]
                        similarity_pen = top_scaffolds[0][4]
                        
                        penalty_parts = []
                        if boredom_pen > 0:
                            penalty_parts.append(f"bore:{boredom_pen:.3f}")
                        if similarity_pen > 0:
                            penalty_parts.append(f"sim:{similarity_pen:.3f}")
                        penalty_info = f" ({', '.join(penalty_parts)})" if penalty_parts else ""
                        
                        print(f"  [Scaffold] 使用骨架约束: {constraint_for_llm[:40]}... "
                              f"(eff: {effective_score:.4f}, raw: {raw_score:.4f}){penalty_info}")
                
                try:
                    mutated_elites = self.llm_editor.edit(
                        elites,
                        scaffold_constraints=constraint_for_llm
                    )
                    mutated_elites = [m for m in mutated_elites if m is not None]
                    print(f"  ChatGPT 生成了 {len(mutated_elites)} 个候选分子 (输入 {len(elites)} 个)")
                    offspring_mols.extend(mutated_elites)
                except Exception as e:
                    print(f"ChatGPT error: {e}")

            # --- 策略 C: Random Mutation ---
            while len(offspring_mols) < self.offspring_size:
                parent = self.belief_space.get_exemplars(1)
                if parent:
                    if self.current_oracle_calls >= self.max_oracle_calls:
                        break
                    child = mu.mutate(parent[0][1], mutation_rate=self.mutation_rate)
                    if child:
                        offspring_mols.append(child)
                else:
                    break

            # 评估后代（evaluate 内部会严格遵守预算）
            offspring_data = self.evaluate(offspring_mols)
            if offspring_data:
                self.belief_space.update(offspring_data, current_generation=generation)
            
            # 检测是否卡住
            if self.current_oracle_calls == last_calls:
                stale_count += 1
                if stale_count >= max_stale_generations:
                    print(f"Warning: No progress for {max_stale_generations} generations. Stopping.")
                    break
            else:
                stale_count = 0
                last_calls = self.current_oracle_calls
            
            # 生成下一代种群
            top_knowledge = self.belief_space.knowledge_repository
            next_gen_candidates = offspring_data + top_knowledge
            next_gen_candidates = sorted(next_gen_candidates, key=lambda x: x[0], reverse=True)
            population_data = next_gen_candidates[:self.population_size]
            
            generation += 1

        # 记录最终数据点
        self._log_convergence_point()
        
        # 计算 AUC 指标
        top_10_auc = self.compute_top_k_auc(10)
        top_100_auc = self.compute_top_k_auc(100)
        
        # 打印最终信念空间状态
        self._print_final_summary()
        
        return self.belief_space.get_best(), top_10_auc, top_100_auc

    def _print_final_summary(self):
        """打印最终信念空间状态"""
        print("\n" + "="*60)
        if self.ablation_mode == "no_belief_space":
            pool_size = len(self._data_from_all_discovered())
            best = self._get_best_single_result()
            print("Belief space disabled for no_belief_space ablation.")
            print(f"  Transient pool size: {pool_size}")
            print(f"  Best score: {best[0]:.4f}")
            print("="*60)
            return
        print("信念空间最终状态:")
        final_summary = self.belief_space.get_belief_summary()
        print(f"  总骨架数: {final_summary['total_scaffolds']}")
        print(f"  活跃信念数: {final_summary['active_beliefs']}")
        print(f"  Top-3 骨架 (考虑所有惩罚后):")
        
        for i, scaffold_info in enumerate(final_summary['top_scaffolds']):
            scaffold = scaffold_info['scaffold']
            eff_score = scaffold_info['effective_score']
            raw_score = scaffold_info['raw_score']
            boredom_pen = scaffold_info['boredom_penalty']
            similarity_pen = scaffold_info['similarity_penalty']
            dominance = scaffold_info['dominance_count']
            
            penalty_parts = []
            if boredom_pen > 0:
                penalty_parts.append(f"bore:{boredom_pen:.3f}")
            if similarity_pen > 0:
                penalty_parts.append(f"sim:{similarity_pen:.3f}")
            penalty_str = f", {', '.join(penalty_parts)}" if penalty_parts else ""
            
            print(f"    {i+1}. {scaffold[:50]}...")
            print(f"       eff: {eff_score:.4f}, raw: {raw_score:.4f}{penalty_str}, dominance: {dominance}")
        
        # 打印无聊度事件统计
        if self.belief_space.boredom_events:
            print(f"\n  无聊度触发事件: {len(self.belief_space.boredom_events)} 次")
            print("  最近 5 次事件:")
            for event in self.belief_space.boredom_events[-5:]:
                print(f"    Gen {event['generation']}: {event['scaffold'][:30]}... "
                      f"(主导 {event['dominance_count']} 代, 惩罚累计: {event['new_penalty']:.3f})")
        else:
            print(f"\n  无聊度触发事件: 0 次 (没有骨架长期主导)")
        
        # 打印相似性惩罚统计
        print(f"\n  相似性惩罚骨架数: {final_summary['similarity_penalized_scaffolds']}")
        
        # 【子结构约束模式】打印统计
        if final_summary.get('substructure_stats'):
            stats = final_summary['substructure_stats']
            print(f"\n  子结构约束统计:")
            print(f"    必需子结构: {stats['pattern']}")
            print(f"    命中次数: {stats['hits']}")
            print(f"    未命中次数: {stats['misses']}")
            print(f"    命中率: {stats['hit_rate']*100:.1f}%")
        
        print("="*60)

    def print_results(self, task_name, best_score, best_smi, top_10_auc, top_100_auc,
                       output_dir=None, save_images=True, save_curve=True):
        """
        打印最终优化结果并保存 Top-10 分子图
        
        :param task_name: 任务名称
        :param best_score: 最佳分数
        :param best_smi: 最佳分子 SMILES
        :param top_10_auc: Top-10 AUC
        :param top_100_auc: Top-100 AUC
        :param output_dir: 输出目录（默认为 {task_name}_top_molecules）
        :param save_images: 是否保存分子图
        :param save_curve: 是否保存收敛曲线图
        """
        print("\n" + "="*60)
        print(f"{task_name} Optimization Finished.")
        print("="*60)
        print(f"Best Score: {best_score:.4f}")
        print(f"Best SMILES: {best_smi}")
        print(f"Total Oracle Calls: {self.current_oracle_calls}")
        print(f"Total Evaluated Unique Molecules: {len(self.all_discovered)}")
        print(f"Total Cached Molecules: {len(self.score_cache)}")
        print("-"*60)
        print(f"Top-10 AUC:  {top_10_auc:.4f}")
        print(f"Top-100 AUC: {top_100_auc:.4f}")
        print("-"*60)
        
        # 获取 Top 10 分子
        top_mols = self.get_top_k_molecules(k=10)
        
        # 输出 Top 10 分子
        print("\nTop 10 Molecules:")
        print("-"*60)
        for i, (score, smi) in enumerate(top_mols):
            print(f"  {i+1}. Score={score:.4f} | {smi}")
        print("="*60)
        
        # 保存分子图
        if save_images and top_mols:
            self._save_top_molecules_images(
                task_name=task_name,
                top_mols=top_mols,
                output_dir=output_dir
            )
        
        # 保存收敛曲线图
        if save_curve and self.convergence_data:
            self._save_convergence_curve(
                task_name=task_name,
                output_dir=output_dir
            )
        
        # 保存评价指标到 JSON 文件
        self._save_metrics(
            task_name=task_name,
            best_score=best_score,
            best_smi=best_smi,
            top_10_auc=top_10_auc,
            top_100_auc=top_100_auc,
            output_dir=output_dir
        )
    
    def _compute_single_diversity_metrics(self):
        """Compute scaffold entropy and mean pairwise Tanimoto distance."""
        scaffolds = []
        fps = []
        for _, smi, _ in self.all_discovered:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold is not None and scaffold.GetNumAtoms() > 0:
                    scaffolds.append(Chem.MolToSmiles(scaffold, canonical=True))
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            except Exception as e:
                print(f"Warning: diversity metric failed: {e}")
        entropy = 0.0
        if scaffolds:
            _, counts = np.unique(scaffolds, return_counts=True)
            probs = counts.astype(float) / float(np.sum(counts))
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        mean_distance = 0.0
        if len(fps) > 1:
            distances = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    distances.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                    if len(distances) >= 10000:
                        break
                if len(distances) >= 10000:
                    break
            mean_distance = float(np.mean(distances)) if distances else 0.0
        return len(set(scaffolds)), entropy, mean_distance

    def _save_metrics(self, task_name, best_score, best_smi, top_10_auc, top_100_auc, output_dir=None):
        """
        保存评价指标到 JSON 文件
        """
        import json
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), f"{task_name.lower()}_top_molecules")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取 Top-K 分子
        top_1 = self.get_top_k_molecules(k=1)
        top_10 = self.get_top_k_molecules(k=10)
        top_100 = self.get_top_k_molecules(k=100)
        
        # 计算 Top-K 平均分数
        top_1_score = top_1[0][0] if top_1 else 0.0
        top_10_avg = np.mean([s for s, _ in top_10]) if top_10 else 0.0
        top_100_avg = np.mean([s for s, _ in top_100]) if top_100 else 0.0
        unique_scaffold_count, scaffold_entropy, mean_tanimoto_distance = self._compute_single_diversity_metrics()
        top_1_auc = self.compute_top_k_auc(1)
        
        # 构建评价指标字典
        metrics = {
            "task_name": task_name,
            "ablation_mode": self.ablation_mode,
            "seed": self.seed,
            "max_oracle_calls": self.max_oracle_calls,
            "best_score": float(best_score),
            "best_smiles": best_smi,
            "total_oracle_calls": self.current_oracle_calls,
            "total_evaluated_unique_molecules": len(self.all_discovered),
            "total_cached_molecules": len(self.score_cache),
            "auc_top_1": float(top_1_auc),
            "auc_top_10": float(top_10_auc),
            "auc_top_100": float(top_100_auc),
            "top_1_score": float(top_1_score),
            "top_10_avg_score": float(top_10_avg),
            "top_100_avg_score": float(top_100_avg),
            "unique_scaffold_count": int(unique_scaffold_count),
            "scaffold_entropy": float(scaffold_entropy),
            "mean_pairwise_tanimoto_distance": float(mean_tanimoto_distance),
            "runtime_seconds": float(self.runtime_seconds),
            "convergence_steps": len(self.convergence_data),
        }
        
        # 保存 JSON 文件
        metrics_path = os.path.join(output_dir, f"{task_name.lower()}_metrics.json")
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 已保存评价指标: {metrics_path}")
        except Exception as e:
            print(f"  ⚠️ 保存评价指标失败: {e}")
        
        # 保存所有发现的分子数据
        try:
            with open(os.path.join(output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            run_config = dict(self.run_config)
            run_config.update({
                "task_name": task_name,
                "ablation_mode": self.ablation_mode,
                "seed": self.seed,
                "max_oracle_calls": self.max_oracle_calls,
            })
            with open(os.path.join(output_dir, "run_config.json"), 'w', encoding='utf-8') as f:
                json.dump(run_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: failed to save standard ablation JSON outputs: {e}")

        self._save_all_discovered(task_name, output_dir)
        self._save_standard_single_outputs(output_dir)
    
    def _save_standard_single_outputs(self, output_dir):
        """Save ablation-standard CSV files with fixed names."""
        try:
            rows = [
                {
                    "oracle_call_idx": int(call_idx),
                    "smiles": smiles,
                    "score": float(score),
                    "ablation_mode": self.ablation_mode,
                    "seed": self.seed,
                }
                for score, smiles, call_idx in self.all_discovered
            ]
            pd.DataFrame(rows).to_csv(os.path.join(output_dir, "all_discovered.csv"), index=False)

            conv_rows = []
            for point in self.convergence_data:
                if len(point) >= 4:
                    oracle_calls, top1, top10, top100 = point[:4]
                else:
                    oracle_calls, top10 = point[:2]
                    scores = sorted((float(x[0]) for x in self.all_discovered if x[2] <= oracle_calls), reverse=True)
                    top1 = scores[0] if scores else 0.0
                    top100 = sum(scores[:100]) / 100.0 if scores else 0.0
                conv_rows.append({
                    "oracle_calls": int(oracle_calls),
                    "top1": float(top1),
                    "top10": float(top10),
                    "top100": float(top100),
                })
            pd.DataFrame(conv_rows).to_csv(os.path.join(output_dir, "convergence.csv"), index=False)

            top_rows = []
            for rank, (score, smi) in enumerate(self.get_top_k_molecules(k=100), start=1):
                scaffold_smi = ""
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if scaffold is not None and scaffold.GetNumAtoms() > 0:
                            scaffold_smi = Chem.MolToSmiles(scaffold, canonical=True)
                    except Exception:
                        scaffold_smi = ""
                top_rows.append({
                    "rank": rank,
                    "smiles": smi,
                    "score": float(score),
                    "scaffold_smiles": scaffold_smi,
                })
            pd.DataFrame(top_rows).to_csv(os.path.join(output_dir, "top_molecules.csv"), index=False)
        except Exception as e:
            print(f"  Warning: failed to save standard ablation CSV outputs: {e}")

    def _save_all_discovered(self, task_name, output_dir):
        """保存所有发现的分子数据"""
        if len(self.all_discovered) == 0:
            return
        
        try:
            all_mols_data = []
            for score, smiles, call_idx in self.all_discovered:
                all_mols_data.append({
                    'oracle_call': call_idx,
                    'score': float(score),
                    'smiles': smiles
                })
            
            all_mols_df = pd.DataFrame(all_mols_data)
            all_mols_path = os.path.join(output_dir, f"{task_name.lower()}_all_discovered.csv")
            all_mols_df.to_csv(all_mols_path, index=False)
            print(f"  ✅ 已保存所有发现的分子: {all_mols_path} ({len(all_mols_data)} 个分子)")
        except Exception as e:
            print(f"  ⚠️ 保存所有发现的分子失败: {e}")

    def _save_top_molecules_images(self, task_name, top_mols, output_dir=None):
        """
        保存 Top-K 分子的图像
        
        :param task_name: 任务名称
        :param top_mols: [(score, smiles), ...] 列表
        :param output_dir: 输出目录
        """
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), f"{task_name.lower()}_top_molecules")
        
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📁 保存分子图到: {output_dir}")
        
        # 转换为 RDKit Mol 对象
        mols = []
        legends = []
        valid_data = []
        
        for i, (score, smi) in enumerate(top_mols):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                legends.append(f"#{i+1}: {score:.4f}")
                valid_data.append((i+1, score, smi))
        
        if not mols:
            print("⚠️ 没有有效分子可以绘制")
            return
        
        # 1. 保存单独的分子图
        for i, (rank, score, smi) in enumerate(valid_data):
            mol = mols[i]
            img_path = os.path.join(output_dir, f"top_{rank:02d}_{score:.4f}.png")
            try:
                img = Draw.MolToImage(mol, size=(400, 400))
                img.save(img_path)
            except Exception as e:
                print(f"  ⚠️ 保存分子 #{rank} 图像失败: {e}")
        
        # 2. 保存合并的网格图
        grid_path = os.path.join(output_dir, f"{task_name.lower()}_top10_grid.png")
        try:
            # 计算网格尺寸
            n_mols = len(mols)
            n_cols = min(5, n_mols)
            n_rows = (n_mols + n_cols - 1) // n_cols
            
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=n_cols,
                subImgSize=(300, 300),
                legends=legends
            )
            img.save(grid_path)
            print(f"  ✅ 已保存网格图: {grid_path}")
        except Exception as e:
            print(f"  ⚠️ 保存网格图失败: {e}")
        
        # 3. 保存 CSV 结果文件
        csv_path = os.path.join(output_dir, f"{task_name.lower()}_top10_results.csv")
        try:
            results_df = pd.DataFrame([
                {"rank": rank, "score": score, "smiles": smi}
                for rank, score, smi in valid_data
            ])
            results_df.to_csv(csv_path, index=False)
            print(f"  ✅ 已保存结果 CSV: {csv_path}")
        except Exception as e:
            print(f"  ⚠️ 保存 CSV 失败: {e}")
        
        print(f"  ✅ 已保存 {len(valid_data)} 个分子图")
    
    def _save_convergence_curve(self, task_name, output_dir=None):
        """
        保存收敛曲线图：Oracle Calls vs. Average Score of Top-10 Molecules
        
        :param task_name: 任务名称
        :param output_dir: 输出目录
        """
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), f"{task_name.lower()}_top_molecules")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(self.convergence_data) < 2:
            print("⚠️ 收敛曲线数据点不足，跳过绘图")
            return
        
        # 准备数据
        oracle_calls = [point[0] for point in self.convergence_data]
        avg_scores = [point[1] for point in self.convergence_data]
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制曲线
        ax.plot(oracle_calls, avg_scores, 'b-', linewidth=2, marker='o', markersize=3, 
                label='Avg Score of Top-10', alpha=0.8)
        
        # 填充曲线下方区域
        ax.fill_between(oracle_calls, avg_scores, alpha=0.2, color='blue')
        
        # 设置标签和标题
        ax.set_xlabel('Oracle Calls', fontsize=12)
        ax.set_ylabel('Average Score of Top-10 Molecules', fontsize=12)
        ax.set_title(f'{task_name} - Convergence Curve', fontsize=14, fontweight='bold')
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 设置图例
        ax.legend(loc='lower right', fontsize=10)
        
        # 设置 x 轴范围从 0 开始
        ax.set_xlim(left=0)
        
        # 设置 y 轴范围 [0, 1] 或根据数据动态调整
        y_min = max(0, min(avg_scores) - 0.05)
        y_max = min(1.0, max(avg_scores) + 0.05)
        ax.set_ylim(y_min, y_max)
        
        # 添加最终分数标注
        final_calls = oracle_calls[-1]
        final_score = avg_scores[-1]
        ax.annotate(f'Final: {final_score:.4f}',
                   xy=(final_calls, final_score),
                   xytext=(final_calls - final_calls*0.15, final_score - 0.03),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   color='red')
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存图片
        curve_path = os.path.join(output_dir, f"{task_name.lower()}_convergence_curve.png")
        try:
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ 已保存收敛曲线图: {curve_path}")
        except Exception as e:
            print(f"  ⚠️ 保存收敛曲线图失败: {e}")
        finally:
            plt.close(fig)
        
        # 同时保存收敛数据到 CSV
        csv_path = os.path.join(output_dir, f"{task_name.lower()}_convergence_data.csv")
        try:
            convergence_df = pd.DataFrame({
                'oracle_calls': oracle_calls,
                'avg_top10_score': avg_scores
            })
            convergence_df.to_csv(csv_path, index=False)
            print(f"  ✅ 已保存收敛曲线数据: {csv_path}")
        except Exception as e:
            print(f"  ⚠️ 保存收敛曲线数据失败: {e}")
