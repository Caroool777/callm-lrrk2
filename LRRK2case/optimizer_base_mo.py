"""
多目标文化算法优化器基类 (Multi-Objective Cultural Algorithm Base)

提供多目标分子优化的通用框架，包括：
- 多目标 Oracle 评估
- 帕累托前沿维护
- 超体积 (Hypervolume) 计算
- BioT5 集成
- 多目标信念空间管理

子类只需实现任务特定的多目标评估函数即可。
"""

import heapq
import json
import random
import os
import pickle
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
# 添加 SingleObjective 目录到 Python 路径，以便导入 crossover 和 mutate 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SingleObjective'))

from belief_space_mo import BeliefSpaceMO, compute_pareto_front, compute_pareto_ranks, pareto_dominates
from pareto_metrics import (
    ParetoMetrics, compute_hypervolume, compute_spacing, compute_spread,
    compute_auc_top_k, compute_all_auc_metrics, compute_top_k_final_scores
)

# 从 SingleObjective 目录导入交叉和变异操作
try:
    import crossover as co
    import mutate as mu
    from llm_backends import LLMBackendFactory
except ImportError:
    # 如果导入失败，尝试从相对路径导入
    from SingleObjective import crossover as co
    from SingleObjective import mutate as mu
    from SingleObjective.llm_backends import LLMBackendFactory


def load_config(task_name=None, base_dir=None):
    """
    加载配置，支持继承机制
    """
    import json
    
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    
    # 加载基础配置
    base_config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Loaded base config: {base_config_path}")
    else:
        # 尝试从 SingleObjective 加载
        single_config_path = os.path.join(base_dir, '..', 'SingleObjective', 'config.json')
        if os.path.exists(single_config_path):
            with open(single_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Loaded base config: {single_config_path}")
        else:
            config = {}
    
    # 合并任务特定配置
    if task_name:
        task_config_path = os.path.join(base_dir, f"config_{task_name}.json")
        if os.path.exists(task_config_path):
            with open(task_config_path, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            config.update(task_config)
            print(f"Loaded task config: {task_config_path}")
    
    override_path = os.environ.get("CALLM_ABLATION_CONFIG")
    if override_path and os.path.exists(override_path):
        with open(override_path, 'r', encoding='utf-8') as f:
            override_config = json.load(f)
        config.update(override_config)
        print(f"Loaded ablation override config: {override_path}")
    
    return config


def load_initial_population(config, base_dir=None):
    """从配置加载初始种群"""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    
    smiles_file = config.get("initial_smiles_file") or "../250k_rndm_zinc_drugs_clean_3.csv"
    if not os.path.isabs(smiles_file):
        smiles_file = os.path.join(base_dir, smiles_file)
    
    pop_size = config.get("pop_size", 100)
    
    if os.path.isfile(smiles_file):
        print(f"从文件加载初始分子: {smiles_file}")
        df = pd.read_csv(smiles_file)
        if 'smiles' in df.columns:
            all_smiles = df['smiles'].tolist()
        elif 'SMILES' in df.columns:
            all_smiles = df['SMILES'].tolist()
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
        ]


class CulturalAlgorithmMOBase(ABC):
    """
    多目标文化算法优化器基类
    
    【帕累托优化核心】
    使用帕累托支配关系维护非支配解集，
    通过拥挤距离保持解集多样性。
    """
    
    TASK_NAME = "multi_objective_base"
    VALID_ABLATION_MODES = {
        "full",
        "no_belief_space",
        "no_scaffold_influence",
        "no_llm",
        "no_biot5",
        "no_ga_operators",
        "scalarized_selection",
    }
    OBJECTIVE_NAMES = ["obj1", "obj2"]  # 子类需要覆盖
    
    def __init__(self, oracle_funcs, task_name=None, max_oracle_calls=10000,
                 device=None, ablation_mode="full"):
        """
        初始化多目标优化器
        
        :param oracle_funcs: Oracle 评分函数列表（每个目标一个）
        :param task_name: 任务名称
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: 指定 GPU 设备
        """
        self.oracle_funcs = oracle_funcs
        self.n_objectives = len(oracle_funcs)
        self.task_name = task_name or self.TASK_NAME
        self.max_oracle_calls = max_oracle_calls
        self.current_oracle_calls = 0
        self.ablation_mode = self._normalize_ablation_mode(ablation_mode)
        self.seed = None
        self.run_config = {}
        self.runtime_seconds = 0.0
        self.selection_type = "pareto"
        self._validate_ablation_mode()
        self.device = device
        
        # Created by apply_config(). Direct run() calls initialize it lazily.
        self.llm_editor = None
        self.llm_editor_count = 10
        
        # 初始化多目标信念空间
        self.belief_space = BeliefSpaceMO(
            capacity=100,
            scaffold_threshold=0.3,  # 多目标任务使用较低阈值
            min_scaffold_samples=2,
            n_objectives=self.n_objectives
        )
        
        self.population_size = 100
        self.offspring_size = 80
        self.mutation_rate = 0.5
        
        # Oracle 缓存
        self.score_cache = {}
        
        # 记录所有发现的分子
        self.all_discovered = []  # [(objectives, smiles, oracle_call_idx), ...]
        self.discovery_metadata = {}  # canonical_smiles -> oracle/property diagnostics
        
        # 收敛曲线数据
        self.convergence_data = []
        self.convergence_log_freq = 10
        self.generation_source_history = []
        
        # 是否启用骨架约束
        self.use_scaffold_constraint = True
        
        # 帕累托前沿评价指标器
        self.pareto_metrics = ParetoMetrics(
            objective_names=self.OBJECTIVE_NAMES,
            reference_point=[0.0] * self.n_objectives
        )

    def apply_config(self, config):
        """从配置字典应用参数"""
        # LLM 后端参数
        self.run_config = dict(config)
        self.checkpoint_dir = config.get("checkpoint_dir")
        self.checkpoint_every_generations = max(
            1, int(config.get("checkpoint_every_generations", 1))
        )
        self.resume_checkpoint = bool(config.get("resume_checkpoint", True))
        self.ablation_mode = self._normalize_ablation_mode(config.get("ablation_mode", "full"))
        self.seed = config.get("seed", None)
        self.max_oracle_calls = config.get("max_oracle_calls", self.max_oracle_calls)
        self._validate_ablation_mode()
        self.selection_type = "scalarized_geometric_mean" if self.ablation_mode == "scalarized_selection" else "pareto"

        # 配置 LLM 编辑器
        self.llm_editor = LLMBackendFactory.create_from_config(config)
        self.llm_editor.set_task(self.task_name)
        self.llm_editor_count = config.get(
            "llm_elite_count",
            config.get("biot5_elite_count", self.llm_editor_count)
        )
        
        # 种群参数
        self.population_size = config.get("pop_size", self.population_size)
        if "offspring_fraction" in config:
            self.offspring_fraction = float(config["offspring_fraction"])
        else:
            self.offspring_fraction = float(config.get("crossover_rate", 0.8))
            if "crossover_rate" in config:
                print(
                    "Warning: crossover_rate is a legacy name and now means offspring_fraction. "
                    "Use offspring_fraction plus crossover_fraction for new configs."
                )
        self.offspring_size = int(self.population_size * self.offspring_fraction)
        self.crossover_fraction = float(config.get("crossover_fraction", 1.0 / 3.0))
        self.crossover_attempts_config = config.get("crossover_attempts")
        self.mutation_rate = config.get("mutation_rate", self.mutation_rate)
        if hasattr(co, "configure_mol_filters"):
            co.configure_mol_filters(
                min_atom_count=config.get("crossover_min_atom_count", 6),
                max_atom_count=config.get("crossover_max_atom_count", 70),
                max_ring_size=config.get("crossover_max_ring_size", 8),
            )
        
        # 信念空间配置
        belief_capacity = config.get("belief_capacity", self.population_size)
        scaffold_threshold = config.get("scaffold_threshold", 0.3)
        
        self.belief_space = BeliefSpaceMO(
            capacity=belief_capacity,
            scaffold_threshold=scaffold_threshold,
            min_scaffold_samples=config.get("min_scaffold_samples", 2),
            n_objectives=self.n_objectives
        )
        
        self.use_scaffold_constraint = config.get("use_scaffold_constraint", True)
        
        # 更新帕累托评价指标器
        self.pareto_metrics = ParetoMetrics(
            objective_names=self.OBJECTIVE_NAMES,
            reference_point=[0.0] * self.n_objectives
        )
        
        print(f"LLM 参数: backend={self.llm_editor.backend_name}, "
              f"beams={self.llm_editor.num_beams}, "
              f"sequences={self.llm_editor.num_return_sequences}, "
              f"temp={self.llm_editor.temperature}")
        print(f"多目标参数: n_objectives={self.n_objectives}, "
              f"objectives={self.OBJECTIVE_NAMES}")

    def _checkpoint_path(self):
        if not getattr(self, "checkpoint_dir", None):
            return None
        checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, "stage_checkpoint.pkl")

    def _save_generation_checkpoint(self, population_data, generation, complete=False):
        """Persist resumable optimizer state without serializing the loaded LLM model."""
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None:
            return
        try:
            population_records = []
            for objectives, mol, smiles in population_data or []:
                if mol is None or not smiles:
                    continue
                population_records.append((list(objectives), str(smiles)))
            llm_generation_count = None
            if self.llm_editor is not None and hasattr(self.llm_editor, "_generation_count"):
                llm_generation_count = int(self.llm_editor._generation_count)
            state = {
                "checkpoint_version": 1,
                "task_name": self.task_name,
                "objective_names": list(self.OBJECTIVE_NAMES),
                "n_objectives": self.n_objectives,
                "generation": int(generation),
                "complete": bool(complete),
                "current_oracle_calls": int(self.current_oracle_calls),
                "population_records": population_records,
                "score_cache": self.score_cache,
                "all_discovered": self.all_discovered,
                "discovery_metadata": self.discovery_metadata,
                "convergence_data": self.convergence_data,
                "generation_source_history": self.generation_source_history,
                "belief_space": self.belief_space,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "llm_generation_count": llm_generation_count,
            }
            temporary_path = checkpoint_path + ".tmp"
            with open(temporary_path, "wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, checkpoint_path)

            summary_path = os.path.join(os.path.dirname(checkpoint_path), "checkpoint_summary.json")
            summary_tmp = summary_path + ".tmp"
            with open(summary_tmp, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "checkpoint_version": 1,
                        "task_name": self.task_name,
                        "generation": int(generation),
                        "complete": bool(complete),
                        "current_oracle_calls": int(self.current_oracle_calls),
                        "max_oracle_calls": int(self.max_oracle_calls),
                        "population_size": len(population_records),
                        "discovered_count": len(self.all_discovered),
                        "checkpoint_file": checkpoint_path,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(summary_tmp, summary_path)
        except Exception as exc:
            print(f"Warning: failed to save generation checkpoint: {exc}")

    def _load_generation_checkpoint(self):
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None or not self.resume_checkpoint or not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path, "rb") as handle:
                state = pickle.load(handle)
            if state.get("task_name") != self.task_name:
                raise ValueError(
                    f"checkpoint task mismatch: {state.get('task_name')} != {self.task_name}"
                )
            if list(state.get("objective_names", [])) != list(self.OBJECTIVE_NAMES):
                raise ValueError("checkpoint objective names do not match the current task")

            self.current_oracle_calls = int(state.get("current_oracle_calls", 0))
            self.score_cache = dict(state.get("score_cache", {}))
            self.all_discovered = list(state.get("all_discovered", []))
            self.discovery_metadata = dict(state.get("discovery_metadata", {}))
            self.convergence_data = list(state.get("convergence_data", []))
            self.generation_source_history = list(state.get("generation_source_history", []))
            if state.get("belief_space") is not None:
                self.belief_space = state["belief_space"]

            population_data = []
            for objectives, smiles in state.get("population_records", []):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    population_data.append((list(objectives), mol, smiles))
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            llm_generation_count = state.get("llm_generation_count")
            if (
                llm_generation_count is not None
                and self.llm_editor is not None
                and hasattr(self.llm_editor, "_generation_count")
            ):
                self.llm_editor._generation_count = int(llm_generation_count)
            print(
                "Resumed checkpoint: "
                f"{checkpoint_path} (generation={state.get('generation', 0)}, "
                f"calls={self.current_oracle_calls}, complete={bool(state.get('complete', False))})"
            )
            return population_data, int(state.get("generation", 0))
        except Exception as exc:
            raise RuntimeError(f"Could not resume checkpoint {checkpoint_path}: {exc}") from exc

    def _ensure_llm_editor(self):
        if self.llm_editor is not None:
            return
        self.llm_editor = LLMBackendFactory.create_from_config(self.run_config)
        self.llm_editor.set_task(self.task_name)

    def _validate_ablation_mode(self):
        if self.ablation_mode not in self.VALID_ABLATION_MODES:
            raise ValueError(
                f"Unsupported multi-objective ablation_mode={self.ablation_mode}. "
                f"Valid modes: {sorted(self.VALID_ABLATION_MODES)}"
            )

    def _normalize_ablation_mode(self, mode):
        if mode == "no_biot5":
            return "no_llm"
        return mode or "full"

    def _aggregate_objectives(self, objectives, method="geometric"):
        """Aggregate objectives for scalarized selection ablation."""
        if not objectives:
            return 0.0
        values = np.array(objectives, dtype=float)
        if method == "arithmetic":
            return float(np.mean(values))
        if method == "min":
            return float(np.min(values))
        values = np.maximum(values, 1e-10)
        return float(np.exp(np.mean(np.log(values))))

    def _sort_population_scalarized(self, population_data):
        return sorted(
            list(population_data or []),
            key=lambda x: self._aggregate_objectives(x[0], method="geometric"),
            reverse=True,
        )

    def _canonicalize_mol(self, mol):
        if mol is None:
            return None
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
            return smi or None
        except Exception:
            return None

    def _deduplicate_valid_mols(self, mols):
        unique = []
        seen = set()
        for mol in mols:
            smi = self._canonicalize_mol(mol)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            unique.append(mol)
        return unique

    def _accept_offspring_child(self, child, parent_smiles=None):
        if child is None:
            return False
        child_smiles = self._canonicalize_mol(child)
        if not child_smiles:
            return False
        if parent_smiles and child_smiles == parent_smiles:
            return False
        if child_smiles in self.score_cache:
            return False
        return True

    def _record_discovery_metadata(self, canonical_smi, metadata):
        if canonical_smi and metadata:
            self.discovery_metadata[canonical_smi] = dict(metadata)

    def _get_discovery_metadata(self, smiles):
        if not smiles:
            return {}
        metadata = self.discovery_metadata.get(smiles)
        if metadata is not None:
            return dict(metadata)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            canonical_smi = Chem.MolToSmiles(mol, canonical=True)
            return dict(self.discovery_metadata.get(canonical_smi, {}))
        except Exception:
            return {}

    def _get_rank_tiebreaker(self, smiles):
        metadata = self._get_discovery_metadata(smiles)
        for key in ("raw_pic50_tiebreaker", "raw_predicted_pic50", "raw_predicted_pactivity"):
            try:
                value = float(metadata.get(key, float("-inf")))
                if np.isfinite(value):
                    return value
            except Exception:
                continue
        return float("-inf")

    def _rank_pareto_summaries(self, pareto_summaries):
        return sorted(
            pareto_summaries,
            key=lambda item: (float(item[0]), self._get_rank_tiebreaker(item[1])),
            reverse=True,
        )

    def _append_metadata_to_row(self, row, smiles):
        for key, value in self._get_discovery_metadata(smiles).items():
            output_key = key if key not in row else f"diagnostic_{key}"
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, ensure_ascii=False)
            row[output_key] = value
        return row

    def _data_from_all_discovered(self):
        rows = []
        for objectives, smi, _ in self.all_discovered:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rows.append((objectives, mol, Chem.MolToSmiles(mol, canonical=True)))
        return rows

    def _get_mo_selection_pool(self, population_data, k=None):
        """Return pool for crossover/BioT5 according to multi-objective ablation mode."""
        if self.ablation_mode == "no_belief_space":
            pool = list(population_data or []) + self._data_from_all_discovered()
            pool = self._sort_population_scalarized(pool)
        elif self.ablation_mode == "scalarized_selection":
            pool = list(getattr(self.belief_space, "knowledge_repository", []) or []) + list(population_data or [])
            pool = self._sort_population_scalarized(pool)
        else:
            pool = list(getattr(self.belief_space, "pareto_front", []) or []) + list(getattr(self.belief_space, "knowledge_repository", []) or [])
            if not pool:
                pool = list(population_data or [])
        seen = set()
        unique = []
        for item in pool:
            if item is None or item[1] is None:
                continue
            if item[2] in seen:
                continue
            seen.add(item[2])
            unique.append(item)
        return unique[:k] if k is not None else unique

    def _select_mo_parents_for_crossover(self, population_data, n_parents=2):
        if self.ablation_mode == "no_ga_operators":
            return []
        if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
            pool = self._get_mo_selection_pool(population_data)
            return random.sample(pool, n_parents) if len(pool) >= n_parents else []
        parents = self.belief_space.get_pareto_exemplars(n_parents)
        if len(parents) < n_parents:
            pool = self._get_mo_selection_pool(population_data)
            return random.sample(pool, n_parents) if len(pool) >= n_parents else []
        return parents

    def _select_mo_elites_for_llm(self, population_data, elite_count):
        if self.ablation_mode == "no_llm":
            return []
        if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
            return self._sort_population_scalarized(population_data)[:elite_count]
        elites = self.belief_space.get_exemplars(elite_count)
        return elites if elites else self._get_mo_selection_pool(population_data, elite_count)

    def _get_mo_scaffold_constraint_for_llm(self):
        if self.ablation_mode in {"no_scaffold_influence", "no_belief_space", "no_llm"}:
            return None
        if self.use_scaffold_constraint:
            top_scaffolds = self.belief_space.get_best_scaffold(top_k=1)
            if top_scaffolds:
                constraint = top_scaffolds[0][0]
                print(f"  [Scaffold] Using scaffold constraint: {constraint[:40]}...")
                return constraint
        return None

    def _get_best_mo_result(self, population_data=None):
        if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
            pool = self._get_mo_selection_pool(population_data)
            if pool:
                best = pool[0]
                return (self._aggregate_objectives(best[0]), best[1], best[2], best[0])
            return (0.0, None, "", [])
        return self.belief_space.get_best()

    @abstractmethod
    def evaluate_objectives(self, smiles):
        """
        评估单个分子的多目标分数
        
        子类必须实现此方法
        
        :param smiles: SMILES 字符串
        :return: 目标值列表 [obj1, obj2, ...]
        """
        pass

    def evaluate(self, mols):
        """评估分子列表，返回 (objectives, mol, smiles) 元组列表"""
        results = []
        for mol in mols:
            if mol is None:
                continue
            try:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                if not canonical_smi:
                    continue
                
                if canonical_smi in self.score_cache:
                    objectives = self.score_cache[canonical_smi]
                else:
                    if self.current_oracle_calls >= self.max_oracle_calls:
                        break
                    objectives = self.evaluate_objectives(canonical_smi)
                    self.score_cache[canonical_smi] = objectives
                    self.current_oracle_calls += 1
                    
                    # 记录有效分子
                    self.all_discovered.append((objectives, canonical_smi, self.current_oracle_calls))
                    
                    # 记录收敛曲线数据点
                    if self.current_oracle_calls % self.convergence_log_freq == 0:
                        self._log_convergence_point()
                
                results.append((objectives, mol, canonical_smi))
            except Exception as e:
                print(f"Eval Error: {e}")
                continue
        return results

    def _log_convergence_point(self):
        """记录当前的收敛曲线数据点"""
        if len(self.all_discovered) == 0:
            return
        
        # 计算当前帕累托前沿的平均分数
        pareto_front = compute_pareto_front([
            (obj, None, smi) for obj, smi, _ in self.all_discovered
        ])
        
        if pareto_front:
            # 使用几何平均作为聚合分数
            avg_scores = []
            for item in pareto_front:
                objectives = [max(o, 1e-6) for o in item[0]]
                geo_mean = np.exp(np.mean(np.log(objectives)))
                avg_scores.append(geo_mean)
            
            avg_pareto_score = np.mean(avg_scores)
            self.convergence_data.append((
                self.current_oracle_calls, 
                avg_pareto_score,
                len(pareto_front)
            ))

    def get_top_k_molecules(self, k=10):
        """获取帕累托前沿上的 Top-K 分子"""
        if len(self.all_discovered) == 0:
            return []
        
        pareto_front = compute_pareto_front([
            (obj, None, smi) for obj, smi, _ in self.all_discovered
        ])
        
        # 按聚合分数排序
        def compute_score(objectives):
            objectives = [max(o, 1e-6) for o in objectives]
            return np.exp(np.mean(np.log(objectives)))
        
        summaries = [(compute_score(item[0]), item[2], item[0]) for item in pareto_front]
        return self._rank_pareto_summaries(summaries)[:k]

    def compute_pareto_metrics(self, pareto_front=None):
        """
        计算帕累托前沿的综合评价指标
        
        :param pareto_front: 帕累托前沿（如果为 None，使用当前发现的所有分子）
        :return: 评价指标字典
        """
        if pareto_front is None:
            if len(self.all_discovered) == 0:
                return self.pareto_metrics._get_empty_metrics()
            
            pareto_front = compute_pareto_front([
                (obj, None, smi) for obj, smi, _ in self.all_discovered
            ])
        
        # 使用评价指标器计算所有指标（包括 AUC Top-K）
        metrics = self.pareto_metrics.evaluate_all(
            pareto_front, 
            self.n_objectives,
            all_discovered=self.all_discovered,
            max_oracle_calls=self.max_oracle_calls
        )
        
        return metrics

    def print_pareto_metrics(self, metrics=None, pareto_front=None):
        """
        打印帕累托前沿评价指标
        
        :param metrics: 预先计算的指标字典（可选）
        :param pareto_front: 帕累托前沿（可选）
        """
        if metrics is None:
            metrics = self.compute_pareto_metrics(pareto_front)
        
        print(self.pareto_metrics.print_metrics(metrics, self.OBJECTIVE_NAMES))

    def _run_ablation_impl(self, initial_smiles_list):
        """Run multi-objective CALLM with ablation-mode selection/operator switches."""
        start_time = time.time()
        restored = self._load_generation_checkpoint()
        if restored is not None:
            population_data, generation = restored
            if not population_data and self.all_discovered:
                population_data = self._data_from_all_discovered()[-self.population_size :]
        else:
            print(f"Initializing Population for Multi-Objective Optimization... ablation_mode={self.ablation_mode}")
            print(f"selection_type={self.selection_type}")
            initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles_list if s]
            initial_mols = self._deduplicate_valid_mols(initial_mols)
            if not initial_mols:
                print("Error: No valid molecules in initial population!")
                return (0.0, None, "", []), [], 0.0, {}

            print(f"Valid initial molecules: {len(initial_mols)}")
            population_data = self.evaluate(initial_mols)
            if not population_data:
                print("Error: Failed to evaluate initial population!")
                return (0.0, None, "", []), [], 0.0, {}

            if self.ablation_mode != "no_belief_space":
                self.belief_space.update(population_data, current_generation=0)
            else:
                pool = self._get_mo_selection_pool(population_data)
                best = self._aggregate_objectives(pool[0][0]) if pool else 0.0
                print(f"[no_belief_space] transient pool size={len(pool)} scalarized_best={best:.4f}")

            generation = 0
            self._save_generation_checkpoint(population_data, generation, complete=False)
        max_stale_generations = 200
        stale_count = 0
        last_calls = self.current_oracle_calls
        use_crossover = self.ablation_mode != "no_ga_operators"
        use_llm = self.ablation_mode != "no_llm"
        use_random_mutation = self.ablation_mode != "no_ga_operators"
        if use_llm:
            self._ensure_llm_editor()

        while self.current_oracle_calls < self.max_oracle_calls:
            best = self._get_best_mo_result(population_data)
            best_score = best[0]
            if self.ablation_mode == "no_belief_space":
                pool = self._get_mo_selection_pool(population_data)
                print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                      f"BestScalar: {best_score:.4f} | Transient Pool: {len(pool)}")
            else:
                belief_summary = self.belief_space.get_belief_summary()
                pareto_size = belief_summary.get('pareto_front_size', 0)
                print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                      f"Best: {best_score:.4f} | Pareto Size: {pareto_size} | "
                      f"Active Beliefs: {belief_summary['active_beliefs']} | selection_type={self.selection_type}")

            offspring_mols = []
            source_counts = {
                "generation": generation,
                "crossover_candidates": 0,
                "biot5_candidates": 0,
                "random_mutation_candidates": 0,
                "deduplicated_offspring": 0,
                "evaluated_offspring": 0,
            }
            if use_crossover:
                if self.crossover_attempts_config is not None:
                    crossover_attempts = int(self.crossover_attempts_config)
                elif self.ablation_mode == "no_llm":
                    crossover_attempts = self.offspring_size
                else:
                    crossover_attempts = max(1, int(self.offspring_size * self.crossover_fraction))
                for _ in range(crossover_attempts):
                    parents_data = self._select_mo_parents_for_crossover(population_data, 2)
                    if len(parents_data) == 2:
                        child = co.crossover(parents_data[0][1], parents_data[1][1])
                        if self._accept_offspring_child(child):
                            offspring_mols.append(child)
                            source_counts["crossover_candidates"] += 1

            if use_llm:
                rounds = 3 if self.ablation_mode == "no_ga_operators" else 1
                for round_idx in range(rounds):
                    elite_count = min(max(self.llm_editor_count * (round_idx + 1), 1), max(len(population_data), 1))
                    elite_data = self._select_mo_elites_for_llm(population_data, elite_count)
                    if not elite_data:
                        break
                    elites = [item[1] for item in elite_data if item[1] is not None]
                    constraint_for_llm = self._get_mo_scaffold_constraint_for_llm()
                    try:
                        mutated_elites = self.llm_editor.edit(elites, scaffold_constraints=constraint_for_llm)
                        mutated_elites = [m for m in mutated_elites if m is not None]
                        print(f"  BioT5 generated {len(mutated_elites)} candidates")
                        offspring_mols.extend(mutated_elites)
                        source_counts["biot5_candidates"] += len(mutated_elites)
                    except Exception as e:
                        print(f"LLM error: {e}")
                    if len(offspring_mols) >= self.offspring_size:
                        break

            if use_random_mutation:
                retry_limit = self.offspring_size * (6 if self.ablation_mode == "no_llm" else 3)
                retries = 0
                while len(offspring_mols) < self.offspring_size and retries < retry_limit:
                    pool = self._get_mo_selection_pool(population_data)
                    if not pool:
                        print("Warning: random mutation fallback has no parents.")
                        break
                    parent = random.choice(pool)
                    child = mu.mutate(parent[1], mutation_rate=self.mutation_rate)
                    if self._accept_offspring_child(child, parent_smiles=parent[2]):
                        offspring_mols.append(child)
                        source_counts["random_mutation_candidates"] += 1
                    retries += 1

            offspring_mols = self._deduplicate_valid_mols(offspring_mols)
            source_counts["deduplicated_offspring"] = len(offspring_mols)
            if not offspring_mols:
                print(f"Warning: empty offspring in {self.ablation_mode}; falling back to current top molecules.")
                offspring_mols = [item[1] for item in self._get_mo_selection_pool(population_data, self.offspring_size)]

            offspring_data = self.evaluate(offspring_mols)
            source_counts["evaluated_offspring"] = len(offspring_data)
            self.generation_source_history.append(dict(source_counts))
            print(
                "  Generation sources: crossover={crossover_candidates}, BioT5={biot5_candidates}, "
                "random_mutation={random_mutation_candidates}, dedup={deduplicated_offspring}, "
                "evaluated={evaluated_offspring}".format(**source_counts)
            )
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
                next_gen_candidates = offspring_data + self._get_mo_selection_pool(population_data)
            else:
                next_gen_candidates = offspring_data + self.belief_space.knowledge_repository
            if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
                population_data = self._sort_population_scalarized(next_gen_candidates)[:self.population_size] or population_data
            else:
                ranked = compute_pareto_ranks(next_gen_candidates)
                population_data = [(r[1], r[2], r[3]) for r in ranked[:self.population_size]] or population_data
            generation += 1
            if generation % self.checkpoint_every_generations == 0:
                self._save_generation_checkpoint(population_data, generation, complete=False)

        self._log_convergence_point()
        pareto_front_data = compute_pareto_front([(obj, None, smi) for obj, smi, _ in self.all_discovered])
        final_pareto = [
            (self._aggregate_objectives(item[0]), item[2], item[0])
            for item in pareto_front_data
        ]
        final_pareto = self._rank_pareto_summaries(final_pareto)
        pareto_metrics = self.compute_pareto_metrics(pareto_front_data)
        pareto_metrics["full_pareto_front_size"] = len(final_pareto)
        pareto_metrics["top100_is_ranked_summary_only"] = True
        pareto_metrics["selection_type"] = self.selection_type
        pareto_metrics["ablation_mode"] = self.ablation_mode
        pareto_metrics["generation_source_totals"] = self._generation_source_totals()
        hypervolume = pareto_metrics.get('hypervolume', 0.0)
        self.runtime_seconds = time.time() - start_time
        self._save_generation_checkpoint(population_data, generation, complete=True)
        self._print_final_summary(pareto_metrics)
        return self._get_best_mo_result(population_data), final_pareto, hypervolume, pareto_metrics

    def run(self, initial_smiles_list):
        """运行多目标优化算法"""
        return self._run_ablation_impl(initial_smiles_list)

    def _generation_source_totals(self):
        totals = {
            "crossover_candidates": 0,
            "biot5_candidates": 0,
            "random_mutation_candidates": 0,
            "deduplicated_offspring": 0,
            "evaluated_offspring": 0,
        }
        for row in self.generation_source_history:
            for key in totals:
                totals[key] += int(row.get(key, 0) or 0)
        totals["generations_recorded"] = len(self.generation_source_history)
        return totals

    def _print_final_summary(self, pareto_metrics=None):
        """打印最终信念空间状态和帕累托评价指标"""
        print("\n" + "="*60)
        if self.ablation_mode == "no_belief_space":
            print("Belief space disabled for no_belief_space ablation.")
            print(f"  selection_type={self.selection_type}")
            print(f"  transient pool size: {len(self._data_from_all_discovered())}")
            print("="*60)
            if pareto_metrics is not None:
                self.print_pareto_metrics(pareto_metrics)
            return
        print("多目标信念空间最终状态:")
        final_summary = self.belief_space.get_belief_summary()
        print(f"  总骨架数: {final_summary['total_scaffolds']}")
        print(f"  活跃信念数: {final_summary['active_beliefs']}")
        print(f"  帕累托前沿大小: {final_summary['pareto_front_size']}")
        
        if final_summary.get('pareto_stats'):
            stats = final_summary['pareto_stats']
            print(f"  帕累托前沿目标统计:")
            for i, name in enumerate(self.OBJECTIVE_NAMES):
                print(f"    {name}: mean={stats['obj_means'][i]:.4f}, "
                      f"max={stats['obj_maxs'][i]:.4f}, min={stats['obj_mins'][i]:.4f}")
        
        print("="*60)
        
        # 打印完整的帕累托评价指标
        if pareto_metrics is not None:
            self.print_pareto_metrics(pareto_metrics)

    def print_results(self, task_name, best_score, best_smi, pareto_front, hypervolume,
                      best_objectives=None, output_dir=None, save_images=True,
                      pareto_metrics=None):
        """打印最终优化结果"""
        print("\n" + "="*60)
        print(f"{task_name} Multi-Objective Optimization Finished.")
        print("="*60)
        print(f"Best Aggregated Score: {best_score:.4f}")
        print(f"Best SMILES: {best_smi}")
        if best_objectives:
            print(f"Best Objectives: {[f'{name}={val:.4f}' for name, val in zip(self.OBJECTIVE_NAMES, best_objectives)]}")
        print(f"Total Oracle Calls: {self.current_oracle_calls}")
        print(f"Pareto Front Size: {len(pareto_front)}")
        print("-"*60)
        
        # 打印帕累托前沿评价指标
        print("【帕累托前沿评价指标】")
        if pareto_metrics:
            print(f"  Hypervolume (HV): {pareto_metrics.get('hypervolume', hypervolume):.6f}")
            print(f"  Spacing (间距): {pareto_metrics.get('spacing', 0.0):.6f}")
            print(f"  Spread (Δ): {pareto_metrics.get('spread', 1.0):.6f}")
            print(f"  Maximum Spread: {pareto_metrics.get('maximum_spread', 0.0):.6f}")
            print(f"  Diversity: {pareto_metrics.get('diversity', 0.0):.6f}")
            print(f"  Uniformity (CV): {pareto_metrics.get('uniformity', 0.0):.6f}")
            
            # AUC Top-K 指标
            if pareto_metrics.get('auc_top_1') is not None:
                print("-" * 40)
                print("【AUC Top-K 指标】")
                print(f"  AUC Top-1: {pareto_metrics.get('auc_top_1', 0.0):.6f}")
                print(f"  AUC Top-10: {pareto_metrics.get('auc_top_10', 0.0):.6f}")
                print(f"  AUC Top-100: {pareto_metrics.get('auc_top_100', 0.0):.6f}")
            
            # Top-K 最终得分
            if pareto_metrics.get('top_1_score') is not None:
                print("-" * 40)
                print("【Top-K 最终得分】")
                print(f"  Top-1 Score: {pareto_metrics.get('top_1_score', 0.0):.6f}")
                print(f"  Top-10 Score: {pareto_metrics.get('top_10_score', 0.0):.6f}")
                print(f"  Top-100 Score: {pareto_metrics.get('top_100_score', 0.0):.6f}")
        else:
            print(f"  Hypervolume: {hypervolume:.4f}")
        print("-"*60)
        
        # 输出帕累托前沿上的 Top 10 分子
        print("\nPareto Front Top 10 Molecules:")
        print("-"*60)
        for i, (score, smi, objectives) in enumerate(pareto_front[:10]):
            obj_str = ", ".join([f"{name}={val:.4f}" for name, val in zip(self.OBJECTIVE_NAMES, objectives)])
            print(f"  {i+1}. Score={score:.4f} | {obj_str}")
            print(f"      SMILES: {smi}")
        print("="*60)
        
        # 保存结果
        if save_images and pareto_front:
            self._save_pareto_front(task_name, pareto_front, output_dir, pareto_metrics)

    def _save_standard_mo_outputs(self, task_name, pareto_front, output_dir, pareto_metrics=None):
        """Save ablation-standard metrics/all_discovered/pareto/convergence/run_config files."""
        try:
            metrics = dict(pareto_metrics or {})
            best_score = pareto_front[0][0] if pareto_front else 0.0
            metrics.update({
                "task_name": task_name,
                "ablation_mode": self.ablation_mode,
                "seed": self.seed,
                "max_oracle_calls": self.max_oracle_calls,
                "total_oracle_calls": self.current_oracle_calls,
                "total_evaluated_unique_molecules": len(self.all_discovered),
                "best_aggregated_score": float(best_score),
                "hypervolume": float(metrics.get("hypervolume", 0.0)),
                "spacing": float(metrics.get("spacing", 0.0)),
                "spread": float(metrics.get("spread", 0.0)),
                "maximum_spread": float(metrics.get("maximum_spread", 0.0)),
                "diversity": float(metrics.get("diversity", 0.0)),
                "pareto_front_size": int(metrics.get("pareto_size", len(pareto_front))),
                "runtime_seconds": float(self.runtime_seconds),
                "selection_type": self.selection_type,
                "generation_source_totals": self._generation_source_totals(),
            })
            for key in ["auc_top_1", "auc_top_10", "auc_top_100", "top_1_score", "top_10_score", "top_100_score"]:
                metrics[key] = float(metrics.get(key, 0.0) or 0.0)
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            rows = []
            for objectives, smiles, call_idx in self.all_discovered:
                row = {
                    "oracle_call_idx": int(call_idx),
                    "smiles": smiles,
                    "aggregated_score": float(self._aggregate_objectives(objectives)),
                    "ablation_mode": self.ablation_mode,
                    "seed": self.seed,
                }
                for i, value in enumerate(objectives, start=1):
                    row[f"objective_{i}"] = float(value)
                row = self._append_metadata_to_row(row, smiles)
                rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(output_dir, "all_discovered.csv"), index=False)

            front_rows = []
            for score, smi, objectives in pareto_front:
                row = {
                    "smiles": smi,
                    "aggregated_score": float(score),
                    "pareto_rank": 0,
                    "is_pareto_front": True,
                }
                for i, value in enumerate(objectives, start=1):
                    row[f"objective_{i}"] = float(value)
                row = self._append_metadata_to_row(row, smi)
                front_rows.append(row)
            pd.DataFrame(front_rows).to_csv(os.path.join(output_dir, "pareto_front.csv"), index=False)
            pd.DataFrame(front_rows[:100]).to_csv(os.path.join(output_dir, "pareto_front_top100.csv"), index=False)

            conv_rows = []
            scores_by_time = sorted(self.all_discovered, key=lambda x: x[2])
            call_points = sorted(set([x[2] for x in scores_by_time] + [self.current_oracle_calls]))
            top1_heap = []
            top10_heap = []
            top100_heap = []
            pf = []
            cursor = 0
            hv_samples = None
            hv_dominated = None
            if self.n_objectives >= 3:
                hv_rng = np.random.default_rng(0)
                hv_samples = hv_rng.uniform(0.0, 1.0, size=(100000, self.n_objectives))
                hv_dominated = np.zeros(len(hv_samples), dtype=bool)

            def push_top(heap, value, capacity):
                if len(heap) < capacity:
                    heapq.heappush(heap, value)
                elif value > heap[0]:
                    heapq.heapreplace(heap, value)

            for call_idx in call_points:
                while cursor < len(scores_by_time) and scores_by_time[cursor][2] <= call_idx:
                    objectives, smiles, _ = scores_by_time[cursor]
                    score = self._aggregate_objectives(objectives)
                    push_top(top1_heap, score, 1)
                    push_top(top10_heap, score, 10)
                    push_top(top100_heap, score, 100)

                    candidate = (objectives, None, smiles)
                    is_dominated = False
                    remove_indices = []
                    for index, existing in enumerate(pf):
                        if pareto_dominates(existing[0], objectives):
                            is_dominated = True
                            break
                        if pareto_dominates(objectives, existing[0]):
                            remove_indices.append(index)
                    if not is_dominated:
                        for index in reversed(remove_indices):
                            pf.pop(index)
                        pf.append(candidate)
                        if hv_samples is not None:
                            clipped = np.clip(np.asarray(objectives, dtype=float), 0.0, 1.0)
                            if np.any(clipped > 0.0):
                                hv_dominated |= np.all(clipped[None, :] >= hv_samples, axis=1)
                    cursor += 1

                if hv_samples is not None:
                    hv = float(np.mean(hv_dominated))
                else:
                    hv = 0.0
                    try:
                        hv = compute_hypervolume(pf, n_objectives=self.n_objectives)
                    except Exception:
                        hv = 0.0
                conv_rows.append({
                    "oracle_calls": int(call_idx),
                    "top1": float(top1_heap[0]) if top1_heap else 0.0,
                    "top10": float(sum(top10_heap) / 10.0) if top10_heap else 0.0,
                    "top100": float(sum(top100_heap) / 100.0) if top100_heap else 0.0,
                    "hypervolume": float(hv),
                    "pareto_front_size": int(len(pf)),
                })
            pd.DataFrame(conv_rows).to_csv(os.path.join(output_dir, "convergence.csv"), index=False)
            pd.DataFrame(self.generation_source_history).to_csv(
                os.path.join(output_dir, "generation_sources.csv"), index=False
            )

            run_config = dict(self.run_config)
            run_config.update({
                "task_name": task_name,
                "ablation_mode": self.ablation_mode,
                "seed": self.seed,
                "max_oracle_calls": self.max_oracle_calls,
                "selection_type": self.selection_type,
                "generation_methods": ["random_crossover", "biot5_edit", "random_mutation"],
                "generation_source_totals": self._generation_source_totals(),
            })
            with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: failed to save standard MO ablation outputs: {e}")

    def _save_pareto_front(self, task_name, pareto_front, output_dir=None, pareto_metrics=None):
        """保存帕累托前沿"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), f"{task_name.lower()}_results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 CSV 结果
        csv_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_front.csv")
        results = []
        for i, (score, smi, objectives) in enumerate(pareto_front):
            row = {"rank": i + 1, "aggregated_score": score, "smiles": smi}
            for name, val in zip(self.OBJECTIVE_NAMES, objectives):
                row[name] = val
            row = self._append_metadata_to_row(row, smi)
            results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path, index=False)
        print(f"  Saved full Pareto front: {csv_path}")
        top100_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_front_top100.csv")
        results_df.head(100).to_csv(top100_path, index=False)
        print(f"  Saved Pareto Top-100 ranked summary: {top100_path}")
        
        # 保存帕累托评价指标到 JSON 文件
        if pareto_metrics:
            import json
            metrics_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_metrics.json")
            # 转换为可序列化的格式
            serializable_metrics = {}
            for k, v in pareto_metrics.items():
                if isinstance(v, (np.ndarray, np.floating, np.integer)):
                    serializable_metrics[k] = float(v) if np.isscalar(v) else v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.floating, np.integer)):
                    serializable_metrics[k] = [float(x) for x in v]
                else:
                    serializable_metrics[k] = v
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            print(f"  Saved Pareto metrics: {metrics_path}")
        
        # 保存分子图
        mols = []
        legends = []
        for i, (score, smi, _) in enumerate(pareto_front[:10]):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                legends.append(f"#{i+1}: {score:.3f}")
        
        if mols:
            grid_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_grid.png")
            try:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=5,
                    subImgSize=(300, 300),
                    legends=legends
                )
                img.save(grid_path)
                print(f"  Saved Pareto grid image: {grid_path}")
            except Exception as e:
                print(f"  Warning: failed to save grid image: {e}")
        
        # 保存帕累托前沿可视化（2D 散点图）
        if len(self.OBJECTIVE_NAMES) >= 2:
            self._save_pareto_scatter(task_name, pareto_front, output_dir, pareto_metrics)
        
        # 保存 AUC Top-K 曲线图和数据
        if len(self.all_discovered) > 0:
            self._save_auc_top_k_curves(task_name, output_dir)
            # 保存所有发现的分子数据
            self._save_all_discovered_molecules(task_name, output_dir)
        self._save_standard_mo_outputs(task_name, pareto_front, output_dir, pareto_metrics)

    def _save_all_discovered_molecules(self, task_name, output_dir):
        """保存所有发现的分子数据到 CSV 文件"""
        if len(self.all_discovered) == 0:
            return
        
        try:
            all_mols_data = []
            for objectives, smiles, call_idx in self.all_discovered:
                row = {
                    'oracle_call': call_idx,
                    'smiles': smiles,
                    'aggregated_score': float(np.prod(objectives) ** (1.0 / len(objectives)))
                }
                for i, name in enumerate(self.OBJECTIVE_NAMES):
                    row[name] = objectives[i]
                row = self._append_metadata_to_row(row, smiles)
                all_mols_data.append(row)
            
            all_mols_df = pd.DataFrame(all_mols_data)
            all_mols_path = os.path.join(output_dir, f"{task_name.lower()}_all_discovered.csv")
            all_mols_df.to_csv(all_mols_path, index=False)
            print(f"  Saved all discovered molecules: {all_mols_path} ({len(all_mols_data)} molecules)")
        except Exception as e:
            print(f"  Warning: failed to save all discovered molecules: {e}")

    def _save_pareto_scatter(self, task_name, pareto_front, output_dir, pareto_metrics=None):
        """保存帕累托前沿的散点图"""
        if len(pareto_front) < 2:
            return
        
        # 使用前两个目标绘制散点图
        obj1 = [item[2][0] for item in pareto_front]
        obj2 = [item[2][1] for item in pareto_front]
        
        plt.figure(figsize=(12, 10))
        plt.scatter(obj1, obj2, c='blue', alpha=0.6, s=50, label='Pareto Front')
        
        # 标记 Top 5
        for i, (score, smi, objectives) in enumerate(pareto_front[:5]):
            plt.annotate(f"#{i+1}", (objectives[0], objectives[1]), fontsize=8)
        
        plt.xlabel(self.OBJECTIVE_NAMES[0], fontsize=12)
        plt.ylabel(self.OBJECTIVE_NAMES[1], fontsize=12)
        plt.title(f'{task_name} - Pareto Front', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 在图中添加评价指标文本框
        if pareto_metrics:
            metrics_text = (
                f"Pareto Metrics:\n"
                f"  Size: {pareto_metrics.get('pareto_size', len(pareto_front))}\n"
                f"  HV: {pareto_metrics.get('hypervolume', 0.0):.4f}\n"
                f"  Spacing: {pareto_metrics.get('spacing', 0.0):.4f}\n"
                f"  Spread(Δ): {pareto_metrics.get('spread', 0.0):.4f}\n"
                f"  Diversity: {pareto_metrics.get('diversity', 0.0):.4f}\n"
                f"  Uniformity: {pareto_metrics.get('uniformity', 0.0):.4f}"
            )
            # 添加 AUC Top-K 指标（如果可用）
            if pareto_metrics.get('auc_top_1') is not None:
                metrics_text += (
                    f"\nAUC Metrics:\n"
                    f"  AUC Top-1: {pareto_metrics.get('auc_top_1', 0.0):.4f}\n"
                    f"  AUC Top-10: {pareto_metrics.get('auc_top_10', 0.0):.4f}\n"
                    f"  AUC Top-100: {pareto_metrics.get('auc_top_100', 0.0):.4f}"
                )
            plt.gcf().text(0.72, 0.15, metrics_text, fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          verticalalignment='bottom')
        
        scatter_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_scatter.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Pareto scatter plot: {scatter_path}")
        
        # 如果有 4 个目标，绘制额外的目标对散点图
        if len(self.OBJECTIVE_NAMES) >= 4:
            self._save_multi_objective_scatter(task_name, pareto_front, output_dir, pareto_metrics)
    
    def _save_multi_objective_scatter(self, task_name, pareto_front, output_dir, pareto_metrics=None):
        """保存多目标散点图矩阵"""
        n_obj = min(len(self.OBJECTIVE_NAMES), 4)
        
        fig, axes = plt.subplots(n_obj - 1, n_obj - 1, figsize=(12, 12))
        fig.suptitle(f'{task_name} - Pareto Front (All Objectives)', fontsize=14, fontweight='bold')
        
        for i in range(n_obj - 1):
            for j in range(n_obj - 1):
                ax = axes[i, j] if n_obj > 2 else axes
                
                if j > i:
                    ax.axis('off')
                    continue
                
                obj_x = [item[2][j] for item in pareto_front]
                obj_y = [item[2][i + 1] for item in pareto_front]
                
                ax.scatter(obj_x, obj_y, c='blue', alpha=0.5, s=20)
                ax.set_xlabel(self.OBJECTIVE_NAMES[j], fontsize=8)
                ax.set_ylabel(self.OBJECTIVE_NAMES[i + 1], fontsize=8)
                ax.tick_params(axis='both', labelsize=7)
                ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        matrix_path = os.path.join(output_dir, f"{task_name.lower()}_pareto_matrix.png")
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved multi-objective scatter matrix: {matrix_path}")

    def _save_auc_top_k_curves(self, task_name, output_dir):
        """保存 AUC Top-K 曲线图和数据"""
        from pareto_metrics import compute_top_k_scores
        
        if len(self.all_discovered) < 2:
            return
        
        # 计算 Top-1, Top-10, Top-100 的得分曲线
        top_1_curve = compute_top_k_scores(self.all_discovered, k=1)
        top_10_curve = compute_top_k_scores(self.all_discovered, k=10)
        top_100_curve = compute_top_k_scores(self.all_discovered, k=100)
        
        if not top_1_curve:
            return
        
        # 保存 AUC Top-K 曲线数据到 CSV 文件
        try:
            # 合并所有曲线数据
            all_calls = sorted(set(
                [c for c, _ in top_1_curve] + 
                [c for c, _ in top_10_curve] + 
                [c for c, _ in top_100_curve]
            ))
            
            # 创建字典便于查找
            top_1_dict = dict(top_1_curve)
            top_10_dict = dict(top_10_curve)
            top_100_dict = dict(top_100_curve)
            
            curve_data = []
            for call in all_calls:
                curve_data.append({
                    'oracle_calls': call,
                    'top_1_score': top_1_dict.get(call, None),
                    'top_10_score': top_10_dict.get(call, None),
                    'top_100_score': top_100_dict.get(call, None)
                })
            
            curve_df = pd.DataFrame(curve_data)
            curve_csv_path = os.path.join(output_dir, f"{task_name.lower()}_auc_top_k_data.csv")
            curve_df.to_csv(curve_csv_path, index=False)
            print(f"  Saved AUC Top-K curve data: {curve_csv_path}")
        except Exception as e:
            print(f"  Warning: failed to save AUC Top-K curve data: {e}")
        
        plt.figure(figsize=(12, 8))
        
        # 绘制三条曲线
        if top_1_curve:
            calls_1, scores_1 = zip(*top_1_curve)
            plt.plot(calls_1, scores_1, 'b-', linewidth=2, label='Top-1', marker='o', markersize=3, alpha=0.7)
        
        if top_10_curve:
            calls_10, scores_10 = zip(*top_10_curve)
            plt.plot(calls_10, scores_10, 'g-', linewidth=2, label='Top-10', marker='s', markersize=3, alpha=0.7)
        
        if top_100_curve:
            calls_100, scores_100 = zip(*top_100_curve)
            plt.plot(calls_100, scores_100, 'r-', linewidth=2, label='Top-100', marker='^', markersize=3, alpha=0.7)
        
        plt.xlabel('Oracle Calls', fontsize=12)
        plt.ylabel('Aggregated Score (Geometric Mean)', fontsize=12)
        plt.title(f'{task_name} - Top-K Score vs Oracle Calls\n(AUC Top-K Curves)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 添加 AUC 值标注
        auc_1 = compute_auc_top_k(self.all_discovered, k=1, max_oracle_calls=self.max_oracle_calls)
        auc_10 = compute_auc_top_k(self.all_discovered, k=10, max_oracle_calls=self.max_oracle_calls)
        auc_100 = compute_auc_top_k(self.all_discovered, k=100, max_oracle_calls=self.max_oracle_calls)
        
        auc_text = (
            f"AUC Values:\n"
            f"  Top-1: {auc_1:.4f}\n"
            f"  Top-10: {auc_10:.4f}\n"
            f"  Top-100: {auc_100:.4f}"
        )
        plt.gcf().text(0.15, 0.75, auc_text, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                      verticalalignment='top')
        
        curve_path = os.path.join(output_dir, f"{task_name.lower()}_auc_top_k_curves.png")
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved AUC Top-K curve plot: {curve_path}")
