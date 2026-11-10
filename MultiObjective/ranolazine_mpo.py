"""
Ranolazine MPO 多目标优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 + 帕累托优化的多目标分子优化框架。

Ranolazine MPO 任务包含以下目标 (根据 GuacaMol 官方定义):
1. Ranolazine 相似度 (Atom Pair Fingerprint, ClippedScoreModifier threshold=0.7)
2. logP 分数 (MaxGaussian μ=7, σ=1)
3. TPSA 分数 (MaxGaussian μ=95, σ=20)
4. 氟原子计数 (Gaussian μ=1, σ=1)

Ranolazine 是一种抗心绞痛药物，用于治疗慢性心绞痛。

参考文献：
- TDC Benchmark: https://tdcommons.ai/functions/oracles/#multi-property-objective-mpo
- GuacaMol Benchmark: Brown et al. "GuacaMol: benchmarking models for de novo 
  molecular design." J. Chem. Inf. Model. 2019.
"""

import os
import random
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


from optimizer_base_mo import CulturalAlgorithmMOBase, load_config, load_initial_population


# Ranolazine 的 SMILES 结构
RANOLAZINE_SMILES = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"

# 目标 logP 值 (MaxGaussian μ=7, σ=1) - logP <= 7 得满分
TARGET_LOGP = 7.0

# 目标 TPSA 值 (MaxGaussian μ=95, σ=20) - TPSA >= 95 得满分
TARGET_TPSA = 95.0

# 目标氟原子数量 (Gaussian μ=1, σ=1)
TARGET_FLUORINE = 1


class RanolazineMPOOracleTDC:
    """
    使用 TDC Oracle 的 Ranolazine MPO 评估器
    
    直接调用 TDC 的 Ranolazine_MPO oracle
    """
    
    def __init__(self):
        print("正在初始化 Ranolazine MPO Oracle (TDC)...")
        
        try:
            from tdc import Oracle
            self.tdc_oracle = Oracle(name='Ranolazine_MPO')
            self.use_tdc = True
            print("✅ TDC Ranolazine_MPO Oracle 加载成功")
        except Exception as e:
            print(f"⚠️ TDC Oracle 加载失败: {e}")
            print("将使用本地实现")
            self.use_tdc = False
        
        # 计算 Ranolazine 的指纹 (Atom Pair - GuacaMol 官方定义)
        self.ranolazine_mol = Chem.MolFromSmiles(RANOLAZINE_SMILES)
        self.ranolazine_ap = Pairs.GetAtomPairFingerprint(self.ranolazine_mol)
        
        print(f"Ranolazine SMILES: {RANOLAZINE_SMILES}")
        print(f"指纹类型: Atom Pair (AP)")
        print(f"目标 logP: <= {TARGET_LOGP}")
        print(f"目标 TPSA: >= {TARGET_TPSA}")
        print(f"目标氟原子数量: {TARGET_FLUORINE}")
        print("✅ Ranolazine MPO Oracle 初始化完成")
    
    def compute_similarity(self, mol):
        """
        计算与 Ranolazine 的 Tanimoto 相似度 (Atom Pair)
        
        使用 ClippedScoreModifier (upper_x=0.7):
        - similarity >= 0.7: score = 1.0
        - similarity < 0.7: score = similarity / 0.7
        
        :param mol: RDKit Mol 对象
        :return: float, 相似度分数 (0-1)
        """
        try:
            ap = Pairs.GetAtomPairFingerprint(mol)
            sim = DataStructs.TanimotoSimilarity(self.ranolazine_ap, ap)
            
            # ClippedScoreModifier at 0.7
            threshold = 0.7
            if sim >= threshold:
                return 1.0
            else:
                return sim / threshold
        except Exception:
            return 0.0
    
    def compute_logp_score(self, mol):
        """
        计算 logP 分数
        
        使用 MaxGaussian modifier (μ=7, σ=1):
        - logP <= 7: score = 1.0
        - logP > 7: score = exp(-0.5 * ((logP - 7) / 1)^2)
        
        :param mol: RDKit Mol 对象
        :return: float, logP 分数 (0-1)
        """
        try:
            logp = Descriptors.MolLogP(mol)
            mu = TARGET_LOGP  # 7.0
            sigma = 1.0
            
            # MaxGaussian: 小于等于 mu 得满分
            if logp <= mu:
                return 1.0
            else:
                return np.exp(-0.5 * ((logp - mu) / sigma) ** 2)
        except Exception:
            return 0.0
    
    def compute_tpsa_score(self, mol):
        """
        计算 TPSA 分数
        
        使用 MaxGaussian modifier (μ=95, σ=20):
        - TPSA >= 95: score = 1.0
        - TPSA < 95: score = exp(-0.5 * ((TPSA - 95) / 20)^2)
        
        :param mol: RDKit Mol 对象
        :return: float, TPSA 分数 (0-1)
        """
        try:
            tpsa = Descriptors.TPSA(mol)
            mu = TARGET_TPSA  # 95.0
            sigma = 20.0
            
            # MaxGaussian: 大于等于 mu 得满分
            if tpsa >= mu:
                return 1.0
            else:
                return np.exp(-0.5 * ((tpsa - mu) / sigma) ** 2)
        except Exception:
            return 0.0
    
    def compute_fluorine_score(self, mol):
        """
        计算氟原子计数分数
        
        使用高斯函数，目标是 TARGET_FLUORINE 个氟原子
        
        :param mol: RDKit Mol 对象
        :return: float, 氟原子分数 (0-1)
        """
        try:
            f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
            # 高斯函数
            score = np.exp(-0.5 * (f_count - TARGET_FLUORINE) ** 2)
            return score
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        根据 GuacaMol 官方定义，Ranolazine MPO 包含 4 个目标：
        1. 与 Ranolazine 的相似度 (Atom Pair, ClippedScoreModifier 0.7)
        2. logP 分数 (MaxGaussian μ=7, σ=1)
        3. TPSA 分数 (MaxGaussian μ=95, σ=20)
        4. 氟原子分数 (Gaussian μ=1, σ=1)
        
        :param smiles: SMILES 字符串
        :return: [similarity, logp_score, tpsa_score, fluorine_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0, 0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_logp_score(mol),
            self.compute_tpsa_score(mol),
            self.compute_fluorine_score(mol)
        ]
    
    def score_aggregated(self, smiles):
        """
        计算聚合分数
        
        如果 TDC 可用，直接使用 TDC oracle
        否则使用本地几何平均
        
        :param smiles: SMILES 字符串
        :return: float
        """
        if self.use_tdc:
            try:
                score = self.tdc_oracle(smiles)
                return score if score is not None else 0.0
            except Exception:
                pass
        
        objectives = self.score_all(smiles)
        objectives = [max(o, 1e-6) for o in objectives]
        return np.exp(np.mean(np.log(objectives)))


class CulturalAlgorithmRanolazineMPO(CulturalAlgorithmMOBase):
    """
    Ranolazine MPO 任务的多目标文化算法优化器
    
    根据 GuacaMol 官方定义，包含 4 个目标
    """
    
    TASK_NAME = "ranolazine_mpo"
    OBJECTIVE_NAMES = ["similarity", "logp", "tpsa", "fluorine"]
    
    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        """
        初始化 Ranolazine MPO 优化器
        
        :param oracle: RanolazineMPOOracleTDC 实例
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: GPU 设备
        """
        self.mpo_oracle = oracle
        
        # 创建各个目标的单独 Oracle 函数 (4个目标)
        oracle_funcs = [
            lambda smi, o=oracle: o.compute_similarity(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_logp_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_tpsa_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_fluorine_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
        ]
        
        super().__init__(
            oracle_funcs=oracle_funcs,
            task_name=self.TASK_NAME,
            max_oracle_calls=max_oracle_calls,
            device=device
        )
    
    def evaluate_objectives(self, smiles):
        """
        评估单个分子的多目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, logp_score, tpsa_score, fluorine_score]
        """
        return self.mpo_oracle.score_all(smiles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ranolazine MPO 多目标分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--max_calls', type=int, default=10000, help='最大 Oracle 调用次数')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_dir)')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="ranolazine_mpo")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_calls is not None:
        config["max_oracle_calls"] = args.max_calls
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    config["device"] = args.device
    
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"随机种子: {seed}")
    
    # 2. 准备多目标 Oracle (使用 TDC)
    oracle = RanolazineMPOOracleTDC()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建多目标优化器
    max_calls = config.get("max_oracle_calls", args.max_calls)
    print(f"✅ Oracle 最大调用次数设置为: {max_calls}")
    optimizer = CulturalAlgorithmRanolazineMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行多目标优化
    print("\n" + "="*60)
    print("开始 Ranolazine MPO 多目标优化")
    print("="*60)
    
    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best
    
    # 6. 打印结果
    output_dir = config.get("output_dir", "ranolazine_mpo_results")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results(
        "Ranolazine_MPO", 
        best_score, 
        best_smi, 
        pareto_front, 
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics
    )

