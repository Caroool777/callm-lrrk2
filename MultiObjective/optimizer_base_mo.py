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

import json
import random
import os
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

import sys
# 添加 SingleObjective 目录到 Python 路径，以便导入 crossover 和 mutate 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SingleObjective'))

from belief_space_mo import BeliefSpaceMO, compute_pareto_front, compute_pareto_ranks
from pareto_metrics import (
    ParetoMetrics, compute_hypervolume, compute_spacing, compute_spread,
    compute_auc_top_k, compute_all_auc_metrics, compute_top_k_final_scores
)

# 从 SingleObjective 目录导入交叉和变异操作
try:
    import crossover as co
    import mutate as mu
except ImportError:
    # 如果导入失败，尝试从相对路径导入
    from SingleObjective import crossover as co
    from SingleObjective import mutate as mu


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
        print(f"✅ 已加载基础配置: {base_config_path}")
    else:
        # 尝试从 SingleObjective 加载
        single_config_path = os.path.join(base_dir, '..', 'SingleObjective', 'config.json')
        if os.path.exists(single_config_path):
            with open(single_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 已加载基础配置: {single_config_path}")
        else:
            config = {}
    
    # 合并任务特定配置
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
    """从配置加载初始种群"""
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
        ]


class BioT5MO:
    """
    BioT5 多目标版本 - 支持多目标优化任务的 Prompt
    """
    
    def __init__(self, num_beams=5, num_return_sequences=3, temperature=0.8, 
                 do_sample=True, device=None):
        print("Loading BioT5 model for Multi-Objective Optimization...")
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import selfies as sf
        
        self.sf = sf
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"BioT5 使用设备: {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.model = T5ForConditionalGeneration.from_pretrained(
            'QizhiPei/biot5-base-text2mol'
        ).to(self.device)

        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.do_sample = do_sample
        
        # 多目标任务描述
        self.task2description = {
            'sitagliptin_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Sitagliptin-like properties:\n'
                '1. High structural similarity to Sitagliptin (a DPP-4 inhibitor)\n'
                '2. Optimal TPSA (Topological Polar Surface Area) around 80-100\n'
                '3. LogP in the range of 1-3 for good oral bioavailability\n'
                '4. Appropriate number of fluorine atoms (2-3 for metabolic stability)\n'
                'The generated molecule should balance all these properties.\n\n'
            ),
            'zaleplon_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Zaleplon-like properties:\n'
                '1. High structural similarity to Zaleplon (a sedative-hypnotic drug, ECFP4)\n'
                '2. Match the target molecular formula C19H17N3O2\n'
                'The generated molecule should balance structural similarity with the exact molecular formula.\n\n'
            ),
            'amlodipine_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Amlodipine-like properties:\n'
                '1. High structural similarity to Amlodipine (a calcium channel blocker, ECFP4)\n'
                '2. Optimal number of rings (target: 3 rings for drug-likeness)\n'
                'The generated molecule should balance structural similarity with appropriate ring count.\n\n'
            ),
            'fexofenadine_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Fexofenadine-like properties:\n'
                '1. High structural similarity to Fexofenadine (an antihistamine, Atom Pair fingerprint)\n'
                '2. Optimal TPSA (Topological Polar Surface Area) around 90 or higher\n'
                '3. LogP around 4 or lower for good absorption\n'
                'The generated molecule should balance all these pharmacokinetic properties.\n\n'
            ),
            'osimertinib_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Osimertinib-like properties:\n'
                '1. High structural similarity to Osimertinib (a third-generation EGFR TKI, FCFP4)\n'
                '2. Optimal TPSA (Topological Polar Surface Area) around 100 or higher\n'
                '3. LogP around 1 or lower for good drug-likeness\n'
                'The generated molecule should balance structural similarity with optimal physicochemical properties.\n\n'
            ),
            'perindopril_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Perindopril-like properties:\n'
                '1. High structural similarity to Perindopril (an ACE inhibitor for hypertension, ECFP4)\n'
                '2. Optimal number of aromatic rings (target: 2 aromatic rings)\n'
                'The generated molecule should balance structural similarity with appropriate aromatic ring count.\n\n'
            ),
            'ranolazine_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for Ranolazine-like properties:\n'
                '1. High structural similarity to Ranolazine (an anti-anginal agent, Atom Pair fingerprint)\n'
                '2. Optimal logP (target: <= 7)\n'
                '3. Optimal TPSA (Topological Polar Surface Area, target: >= 95)\n'
                '4. Appropriate number of fluorine atoms (target: 1 fluorine atom)\n'
                'The generated molecule should balance structural similarity with optimal physicochemical properties.\n\n'
            ),
            'lrrk2_wt_activity_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for LRRK2 wild-type inhibitor discovery:\n'
                '1. High predicted LRRK2 wild-type activity from a learned activity oracle\n'
                '2. Good overall medicinal chemistry quality as a weak regularizer\n'
                'The generated molecule should improve predicted target activity while remaining drug-like.\n\n'
            ),
            'lrrk2_wt_lead_repair_mpo': (
                'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule '
                'that optimizes multiple objectives for LRRK2 wild-type lead repair:\n'
                '1. High predicted LRRK2 wild-type activity from a learned activity oracle\n'
                '2. Improved BBB/CNS-oriented physicochemical profile\n'
                '3. Balanced lipophilicity and polarity as a medicinal chemistry regularizer\n'
                'The generated molecule should preserve target activity while improving CNS exposure-related properties.\n\n'
            ),
        }
        
        # 带骨架约束的任务描述模板
        self.task2constrained_description = {
            'sitagliptin_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Sitagliptin-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Sitagliptin\n'
                '2. TPSA around 80-100\n'
                '3. LogP in range 1-3\n'
                '4. 2-3 fluorine atoms\n\n'
            ),
            'zaleplon_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Zaleplon-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Zaleplon\n'
                '2. Match molecular formula C19H17N3O2\n\n'
            ),
            'amlodipine_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Amlodipine-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Amlodipine\n'
                '2. Maintain approximately 3 rings\n\n'
            ),
            'fexofenadine_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Fexofenadine-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Fexofenadine\n'
                '2. TPSA >= 90\n'
                '3. LogP <= 4\n\n'
            ),
            'osimertinib_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Osimertinib-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Osimertinib\n'
                '2. TPSA >= 100\n'
                '3. LogP <= 1\n\n'
            ),
            'perindopril_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Perindopril-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Perindopril\n'
                '2. Maintain approximately 2 aromatic rings\n\n'
            ),
            'ranolazine_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for Ranolazine-like multi-property optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Structural similarity to Ranolazine\n'
                '2. logP <= 7\n'
                '3. TPSA >= 95\n'
                '4. 1 fluorine atom\n\n'
            ),
            'lrrk2_wt_activity_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for LRRK2 wild-type inhibitor optimization (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Predicted LRRK2 wild-type activity\n'
                '2. Medicinal chemistry quality as a weak regularizer\n\n'
            ),
            'lrrk2_wt_lead_repair_mpo': (
                'Definition: You are given a molecule SELFIES. '
                'The core scaffold {scaffold} has been identified as effective for LRRK2 wild-type lead repair (Belief Knowledge). '
                'Your job is to generate a new SELFIES molecule by modifying ONLY the side chains or substituents '
                'while strictly preserving the core scaffold structure to optimize:\n'
                '1. Predicted LRRK2 wild-type activity\n'
                '2. BBB/CNS-oriented properties\n'
                '3. Balanced lipophilicity and polarity\n'
                'Preserve the core scaffold and improve side chains for CNS exposure while maintaining LRRK2 wild-type activity.\n\n'
            ),
        }
        
        self.task = None

    def _build_prompt(self, task_name, selfies_input, scaffold_constraint=None):
        """构建动态 Prompt"""
        if scaffold_constraint:
            try:
                scaffold_selfies = self.sf.encoder(scaffold_constraint)
            except Exception:
                scaffold_selfies = scaffold_constraint
            
            task_definition = self.task2constrained_description.get(
                task_name, 
                self.task2description.get(task_name, '')
            ).format(scaffold=scaffold_selfies)
        else:
            task_definition = self.task2description.get(task_name, '')
        
        task_input = f'Now complete the following example -\nInput: <bom>{selfies_input}<eom>\nOutput: '
        return task_definition + task_input

    def edit(self, smiles_list, scaffold_constraints=None):
        """使用 BioT5 模型编辑分子列表"""
        if not self.task:
            print("Error: Task not set for BioT5.")
            return []
            
        task_name = self.task[0]
        
        # 处理 scaffold_constraints 参数
        if scaffold_constraints is None:
            constraints = [None] * len(smiles_list)
        elif isinstance(scaffold_constraints, str):
            constraints = [scaffold_constraints] * len(smiles_list)
        else:
            constraints = scaffold_constraints
            if len(constraints) < len(smiles_list):
                constraints = constraints + [None] * (len(smiles_list) - len(constraints))

        editted_molecules = []
        for i, MOL in enumerate(smiles_list):
            if MOL is None: 
                continue
            SMILES = Chem.MolToSmiles(MOL, canonical=True)

            try:
                selfies_input = self.sf.encoder(SMILES)
            except Exception:
                print("could not encode input smiles", SMILES)
                continue
            
            scaffold = constraints[i] if i < len(constraints) else None
            model_input = self._build_prompt(task_name, selfies_input, scaffold)
            
            input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids.to(self.device)
            
            if self.do_sample:
                outputs = self.model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=50,
                    top_p=0.95,
                    early_stopping=True
                )
            else:
                outputs = self.model.generate(
                    input_ids,
                    max_length=512,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=True
                )
            
            for output in outputs:
                output_selfies = self.tokenizer.decode(output, skip_special_tokens=True).replace(' ', '')
                
                try:
                    output_smiles = self.sf.decoder(output_selfies)
                    mol = Chem.MolFromSmiles(output_smiles)
                    if mol:
                        editted_molecules.append(mol)
                except Exception:
                    continue
                    
        return editted_molecules


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
        self.ablation_mode = ablation_mode
        self.seed = None
        self.run_config = {}
        self.runtime_seconds = 0.0
        self.selection_type = "pareto"
        self._validate_ablation_mode()
        self.device = device
        
        # 初始化 BioT5 模型
        self.biot5_model = BioT5MO(
            num_beams=5,
            num_return_sequences=3,
            temperature=0.8,
            do_sample=True,
            device=device
        )
        self.biot5_model.task = [self.task_name]
        
        self.biot5_elite_count = 10
        
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
        
        # 收敛曲线数据
        self.convergence_data = []
        self.convergence_log_freq = 10
        
        # 是否启用骨架约束
        self.use_scaffold_constraint = True
        
        # 帕累托前沿评价指标器
        self.pareto_metrics = ParetoMetrics(
            objective_names=self.OBJECTIVE_NAMES,
            reference_point=[0.0] * self.n_objectives
        )

    def apply_config(self, config):
        """从配置字典应用参数"""
        # BioT5 参数
        self.run_config = dict(config)
        self.ablation_mode = config.get("ablation_mode", "full")
        self.seed = config.get("seed", None)
        self.max_oracle_calls = config.get("max_oracle_calls", self.max_oracle_calls)
        self._validate_ablation_mode()
        self.selection_type = "scalarized_geometric_mean" if self.ablation_mode == "scalarized_selection" else "pareto"

        self.biot5_model.num_beams = config.get("biot5_num_beams", 5)
        self.biot5_model.num_return_sequences = config.get("biot5_num_return_sequences", 3)
        self.biot5_model.temperature = config.get("biot5_temperature", 0.8)
        self.biot5_model.do_sample = config.get("biot5_do_sample", True)
        self.biot5_elite_count = config.get("biot5_elite_count", 10)
        
        # 种群参数
        self.population_size = config.get("pop_size", self.population_size)
        self.offspring_size = int(self.population_size * config.get("crossover_rate", 0.8))
        self.mutation_rate = config.get("mutation_rate", self.mutation_rate)
        
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
        
        print(f"BioT5 参数: beams={self.biot5_model.num_beams}, "
              f"sequences={self.biot5_model.num_return_sequences}, "
              f"temp={self.biot5_model.temperature}")
        print(f"多目标参数: n_objectives={self.n_objectives}, "
              f"objectives={self.OBJECTIVE_NAMES}")

    def _validate_ablation_mode(self):
        if self.ablation_mode not in self.VALID_ABLATION_MODES:
            raise ValueError(
                f"Unsupported multi-objective ablation_mode={self.ablation_mode}. "
                f"Valid modes: {sorted(self.VALID_ABLATION_MODES)}"
            )

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

    def _select_mo_elites_for_biot5(self, population_data, elite_count):
        if self.ablation_mode == "no_biot5":
            return []
        if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
            return self._sort_population_scalarized(population_data)[:elite_count]
        elites = self.belief_space.get_exemplars(elite_count)
        return elites if elites else self._get_mo_selection_pool(population_data, elite_count)

    def _get_mo_scaffold_constraint_for_biot5(self):
        if self.ablation_mode in {"no_scaffold_influence", "no_belief_space", "no_biot5"}:
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
        
        sorted_front = sorted(pareto_front, key=lambda x: compute_score(x[0]), reverse=True)
        
        return [(compute_score(item[0]), item[2], item[0]) for item in sorted_front[:k]]

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
        max_stale_generations = 200
        stale_count = 0
        last_calls = 0
        use_crossover = self.ablation_mode != "no_ga_operators"
        use_biot5 = self.ablation_mode != "no_biot5"
        use_random_mutation = self.ablation_mode != "no_ga_operators"

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
            if use_crossover:
                crossover_attempts = self.offspring_size if self.ablation_mode == "no_biot5" else max(1, self.offspring_size // 3)
                for _ in range(crossover_attempts):
                    parents_data = self._select_mo_parents_for_crossover(population_data, 2)
                    if len(parents_data) == 2:
                        child = co.crossover(parents_data[0][1], parents_data[1][1])
                        if child:
                            offspring_mols.append(child)

            if use_biot5:
                rounds = 3 if self.ablation_mode == "no_ga_operators" else 1
                for round_idx in range(rounds):
                    elite_count = min(max(self.biot5_elite_count * (round_idx + 1), 1), max(len(population_data), 1))
                    elite_data = self._select_mo_elites_for_biot5(population_data, elite_count)
                    if not elite_data:
                        break
                    elites = [item[1] for item in elite_data if item[1] is not None]
                    constraint_for_biot5 = self._get_mo_scaffold_constraint_for_biot5()
                    try:
                        mutated_elites = self.biot5_model.edit(elites, scaffold_constraints=constraint_for_biot5)
                        mutated_elites = [m for m in mutated_elites if m is not None]
                        print(f"  BioT5 generated {len(mutated_elites)} candidates")
                        offspring_mols.extend(mutated_elites)
                    except Exception as e:
                        print(f"BioT5 error: {e}")
                    if len(offspring_mols) >= self.offspring_size:
                        break

            if use_random_mutation:
                retry_limit = self.offspring_size * (6 if self.ablation_mode == "no_biot5" else 3)
                retries = 0
                while len(offspring_mols) < self.offspring_size and retries < retry_limit:
                    pool = self._get_mo_selection_pool(population_data)
                    if not pool:
                        print("Warning: random mutation fallback has no parents.")
                        break
                    parent = random.choice(pool)
                    child = mu.mutate(parent[1], mutation_rate=self.mutation_rate)
                    if child:
                        offspring_mols.append(child)
                    retries += 1

            offspring_mols = self._deduplicate_valid_mols(offspring_mols)
            if not offspring_mols:
                print(f"Warning: empty offspring in {self.ablation_mode}; falling back to current top molecules.")
                offspring_mols = [item[1] for item in self._get_mo_selection_pool(population_data, self.offspring_size)]

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
                next_gen_candidates = offspring_data + self._get_mo_selection_pool(population_data)
            else:
                next_gen_candidates = offspring_data + self.belief_space.knowledge_repository
            if self.ablation_mode in {"no_belief_space", "scalarized_selection"}:
                population_data = self._sort_population_scalarized(next_gen_candidates)[:self.population_size] or population_data
            else:
                ranked = compute_pareto_ranks(next_gen_candidates)
                population_data = [(r[1], r[2], r[3]) for r in ranked[:self.population_size]] or population_data
            generation += 1

        self._log_convergence_point()
        if self.ablation_mode == "scalarized_selection":
            final_pareto = [
                (self._aggregate_objectives(obj), smi, obj)
                for obj, smi, _ in sorted(
                    self.all_discovered,
                    key=lambda x: self._aggregate_objectives(x[0]),
                    reverse=True,
                )[:100]
            ]
        else:
            final_pareto = self.get_top_k_molecules(k=100)

        pareto_front_data = compute_pareto_front([(obj, None, smi) for obj, smi, _ in self.all_discovered])
        pareto_metrics = self.compute_pareto_metrics(pareto_front_data)
        pareto_metrics["selection_type"] = self.selection_type
        pareto_metrics["ablation_mode"] = self.ablation_mode
        hypervolume = pareto_metrics.get('hypervolume', 0.0)
        self.runtime_seconds = time.time() - start_time
        self._print_final_summary(pareto_metrics)
        return self._get_best_mo_result(population_data), final_pareto, hypervolume, pareto_metrics

    def run(self, initial_smiles_list):
        """运行多目标优化算法"""
        return self._run_ablation_impl(initial_smiles_list)
        print("Initializing Population for Multi-Objective Optimization...")
        initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles_list if s]
        initial_mols = [m for m in initial_mols if m is not None]
        
        if not initial_mols:
            print("Error: No valid molecules in initial population!")
            return (0.0, None, "", []), [], 0.0
        
        print(f"Valid initial molecules: {len(initial_mols)}")
        population_data = self.evaluate(initial_mols)
        
        if not population_data:
            print("Error: Failed to evaluate initial population!")
            return (0.0, None, "", []), [], 0.0
        
        self.belief_space.update(population_data, current_generation=0)
        
        generation = 0
        max_stale_generations = 200  # 增加容忍度，避免过早停止
        stale_count = 0
        last_calls = 0
        
        while self.current_oracle_calls < self.max_oracle_calls:
            best = self.belief_space.get_best()
            best_score = best[0]
            
            belief_summary = self.belief_space.get_belief_summary()
            pareto_size = belief_summary.get('pareto_front_size', 0)
            
            print(f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                  f"Best: {best_score:.4f} | Pareto Size: {pareto_size} | "
                  f"Active Beliefs: {belief_summary['active_beliefs']}")

            offspring_mols = []
            
            # --- 策略 A: Crossover from Pareto Front ---
            for _ in range(self.offspring_size // 3):
                parents_data = self.belief_space.get_pareto_exemplars(2)
                if len(parents_data) == 2:
                    child = co.crossover(parents_data[0][1], parents_data[1][1])
                    if child:
                        offspring_mols.append(child)

            # --- 策略 B: BioT5 Mutation with Scaffold Constraint ---
            elite_count = min(self.biot5_elite_count, len(self.belief_space.knowledge_repository))
            elite_data = self.belief_space.get_exemplars(elite_count)
            
            if elite_data:
                elites = [item[1] for item in elite_data]
                
                constraint_for_biot5 = None
                if self.use_scaffold_constraint:
                    top_scaffolds = self.belief_space.get_best_scaffold(top_k=1)
                    if top_scaffolds:
                        constraint_for_biot5 = top_scaffolds[0][0]
                        print(f"  [Scaffold] 使用骨架约束: {constraint_for_biot5[:40]}...")
                
                try:
                    mutated_elites = self.biot5_model.edit(
                        elites,
                        scaffold_constraints=constraint_for_biot5
                    )
                    mutated_elites = [m for m in mutated_elites if m is not None]
                    print(f"  BioT5 生成了 {len(mutated_elites)} 个候选分子")
                    offspring_mols.extend(mutated_elites)
                except Exception as e:
                    print(f"BioT5 error: {e}")

            # --- 策略 C: Random Mutation ---
            while len(offspring_mols) < self.offspring_size:
                parent = self.belief_space.get_exemplars(1)
                if parent:
                    child = mu.mutate(parent[0][1], mutation_rate=self.mutation_rate)
                    if child:
                        offspring_mols.append(child)
                else:
                    break

            # 评估后代
            offspring_data = self.evaluate(offspring_mols)
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
            
            # 使用帕累托排序选择
            ranked = compute_pareto_ranks(next_gen_candidates)
            population_data = [(r[1], r[2], r[3]) for r in ranked[:self.population_size]]
            
            generation += 1

        # 记录最终数据点
        self._log_convergence_point()
        
        # 获取最终帕累托前沿
        final_pareto = self.get_top_k_molecules(k=100)
        
        # 计算帕累托前沿评价指标
        pareto_front_data = compute_pareto_front([
            (obj, None, smi) for obj, smi, _ in self.all_discovered
        ])
        pareto_metrics = self.compute_pareto_metrics(pareto_front_data)
        
        # 计算超体积（如果可用）
        hypervolume = pareto_metrics.get('hypervolume', 0.0)
        
        # 打印最终信念空间状态和帕累托评价指标
        self._print_final_summary(pareto_metrics)
        
        return self.belief_space.get_best(), final_pareto, hypervolume, pareto_metrics

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
                front_rows.append(row)
            pd.DataFrame(front_rows).to_csv(os.path.join(output_dir, "pareto_front.csv"), index=False)

            conv_rows = []
            scores_by_time = sorted(self.all_discovered, key=lambda x: x[2])
            for call_idx in sorted(set([x[2] for x in scores_by_time] + [self.current_oracle_calls])):
                upto = [x for x in scores_by_time if x[2] <= call_idx]
                agg_scores = sorted([self._aggregate_objectives(x[0]) for x in upto], reverse=True)
                pf = compute_pareto_front([(obj, None, smi) for obj, smi, _ in upto])
                hv = 0.0
                try:
                    hv = compute_hypervolume(pf, n_objectives=self.n_objectives)
                except Exception:
                    hv = 0.0
                conv_rows.append({
                    "oracle_calls": int(call_idx),
                    "top1": float(agg_scores[0]) if agg_scores else 0.0,
                    "top10": float(sum(agg_scores[:10]) / 10.0) if agg_scores else 0.0,
                    "top100": float(sum(agg_scores[:100]) / 100.0) if agg_scores else 0.0,
                    "hypervolume": float(hv),
                    "pareto_front_size": int(len(pf)),
                })
            pd.DataFrame(conv_rows).to_csv(os.path.join(output_dir, "convergence.csv"), index=False)

            run_config = dict(self.run_config)
            run_config.update({
                "task_name": task_name,
                "ablation_mode": self.ablation_mode,
                "seed": self.seed,
                "max_oracle_calls": self.max_oracle_calls,
                "selection_type": self.selection_type,
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
            results.append(row)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_path, index=False)
        print(f"  ✅ 已保存帕累托前沿: {csv_path}")
        
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
            print(f"  ✅ 已保存帕累托评价指标: {metrics_path}")
        
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
                print(f"  ✅ 已保存帕累托前沿网格图: {grid_path}")
            except Exception as e:
                print(f"  ⚠️ 保存网格图失败: {e}")
        
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
                all_mols_data.append(row)
            
            all_mols_df = pd.DataFrame(all_mols_data)
            all_mols_path = os.path.join(output_dir, f"{task_name.lower()}_all_discovered.csv")
            all_mols_df.to_csv(all_mols_path, index=False)
            print(f"  ✅ 已保存所有发现的分子: {all_mols_path} ({len(all_mols_data)} 个分子)")
        except Exception as e:
            print(f"  ⚠️ 保存所有发现的分子失败: {e}")

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
        print(f"  ✅ 已保存帕累托前沿散点图: {scatter_path}")
        
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
        print(f"  ✅ 已保存多目标散点图矩阵: {matrix_path}")

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
            print(f"  ✅ 已保存 AUC Top-K 曲线数据: {curve_csv_path}")
        except Exception as e:
            print(f"  ⚠️ 保存 AUC Top-K 曲线数据失败: {e}")
        
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
        print(f"  ✅ 已保存 AUC Top-K 曲线图: {curve_path}")
