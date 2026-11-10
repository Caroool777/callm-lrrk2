"""
Osimertinib MPO 多目标优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 + 帕累托优化的多目标分子优化框架。

Osimertinib MPO 任务包含以下目标：
1. Osimertinib 相似度 (Tanimoto Similarity, FCFP4)
2. TPSA (Topological Polar Surface Area) 优化
3. logP 优化

Osimertinib 是一种第三代 EGFR 酪氨酸激酶抑制剂，用于治疗 EGFR T790M 突变阳性的非小细胞肺癌。

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
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs

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


# Osimertinib 的 SMILES 结构 (EGFR T790M 抑制剂)
OSIMERTINIB_SMILES = "CN1C=C(C2=CC=CC=C21)C3=NC(=NC=C3)NC4=C(C=C(C(=C4)NC(=O)C=C)N(C)CCN(C)C)OC"

# 目标属性参数 (基于 GuacaMol MPO 定义)
TARGET_TPSA = 100.0  # 目标 TPSA 值
TARGET_LOGP = 1.0    # 目标 logP 值


class OsimertinibMPOOracleTDC:
    """
    使用 TDC Oracle 的 Osimertinib MPO 评估器
    
    直接调用 TDC 的 Osimertinib_MPO oracle，同时提供各个目标的独立评分
    
    目标函数：
    1. 与 Osimertinib 的相似度 (FCFP4 Tanimoto)
    2. TPSA (Topological Polar Surface Area) - MaxGaussian(100, 2)
    3. logP - MinGaussian(1, 2)
    """
    
    def __init__(self):
        print("正在初始化 Osimertinib MPO Oracle (TDC)...")
        
        # 尝试加载 TDC Oracle
        try:
            from tdc import Oracle
            self.tdc_oracle = Oracle(name='Osimertinib_MPO')
            self.use_tdc = True
            print("✅ TDC Osimertinib_MPO Oracle 加载成功")
        except Exception as e:
            print(f"⚠️ TDC Oracle 加载失败: {e}")
            raise RuntimeError("需要 TDC Oracle 来运行此任务")
        
        # 计算 Osimertinib 的指纹用于单独评估相似度
        # 使用 FCFP4 (Feature Connectivity Fingerprint, radius=2, useFeatures=True)
        self.osimertinib_mol = Chem.MolFromSmiles(OSIMERTINIB_SMILES)
        self.osimertinib_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.osimertinib_mol, radius=2, nBits=2048, useFeatures=True
        )
        
        print(f"Osimertinib SMILES: {OSIMERTINIB_SMILES}")
        print(f"目标 TPSA: {TARGET_TPSA}")
        print(f"目标 logP: {TARGET_LOGP}")
        print("✅ Osimertinib MPO Oracle 初始化完成")
    
    def compute_similarity(self, mol):
        """
        计算与 Osimertinib 的 Tanimoto 相似度 (FCFP4)
        
        :param mol: RDKit Mol 对象
        :return: float, 相似度分数 (0-1)
        """
        try:
            # FCFP4: Morgan fingerprint with features (useFeatures=True)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048, useFeatures=True
            )
            return DataStructs.TanimotoSimilarity(self.osimertinib_fp, fp)
        except Exception:
            return 0.0
    
    def compute_tpsa_score(self, mol):
        """
        计算 TPSA 分数
        
        使用 MaxGaussian(100, 2) 函数
        目标是使 TPSA >= 100，超过100后保持满分
        
        :param mol: RDKit Mol 对象
        :return: float, TPSA 分数 (0-1)
        """
        try:
            tpsa = Descriptors.TPSA(mol)
            # MaxGaussian: 当 TPSA >= target 时分数为1，低于target时按高斯衰减
            if tpsa >= TARGET_TPSA:
                return 1.0
            else:
                # 高斯衰减，sigma = 2 (实际按比例)
                sigma = TARGET_TPSA * 0.1  # 约10%的容差
                return np.exp(-0.5 * ((tpsa - TARGET_TPSA) / sigma) ** 2)
        except Exception:
            return 0.0
    
    def compute_logp_score(self, mol):
        """
        计算 logP 分数
        
        使用 MinGaussian(1, 2) 函数
        目标是使 logP <= 1，低于1后保持满分
        
        :param mol: RDKit Mol 对象
        :return: float, logP 分数 (0-1)
        """
        try:
            logp = Descriptors.MolLogP(mol)
            # MinGaussian: 当 logP <= target 时分数为1，高于target时按高斯衰减
            if logp <= TARGET_LOGP:
                return 1.0
            else:
                # 高斯衰减
                sigma = 2.0  # 容差参数
                return np.exp(-0.5 * ((logp - TARGET_LOGP) / sigma) ** 2)
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, tpsa_score, logp_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_tpsa_score(mol),
            self.compute_logp_score(mol)
        ]
    
    def score_aggregated(self, smiles):
        """
        计算聚合分数
        
        使用 TDC oracle 的官方聚合分数
        
        :param smiles: SMILES 字符串
        :return: float
        """
        try:
            score = self.tdc_oracle(smiles)
            return score if score is not None else 0.0
        except Exception:
            # 回退到几何平均
            objectives = self.score_all(smiles)
            objectives = [max(o, 1e-6) for o in objectives]
            return np.exp(np.mean(np.log(objectives)))


class CulturalAlgorithmOsimertinibMPO(CulturalAlgorithmMOBase):
    """
    Osimertinib MPO 任务的多目标文化算法优化器
    """
    
    TASK_NAME = "osimertinib_mpo"
    OBJECTIVE_NAMES = ["similarity", "tpsa", "logp"]
    
    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        """
        初始化 Osimertinib MPO 优化器
        
        :param oracle: OsimertinibMPOOracleTDC 实例
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: GPU 设备
        """
        self.mpo_oracle = oracle
        
        # 创建各个目标的单独 Oracle 函数
        oracle_funcs = [
            lambda smi, o=oracle: o.compute_similarity(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_tpsa_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_logp_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
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
        :return: [similarity, tpsa_score, logp_score]
        """
        return self.mpo_oracle.score_all(smiles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Osimertinib MPO 多目标分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--max_calls', type=int, default=10000, help='最大 Oracle 调用次数')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_dir)')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="osimertinib_mpo")
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
    
    # 2. 准备多目标 Oracle (仅使用 TDC)
    oracle = OsimertinibMPOOracleTDC()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建多目标优化器
    max_calls = config.get("max_oracle_calls", args.max_calls)
    print(f"✅ Oracle 最大调用次数设置为: {max_calls}")
    optimizer = CulturalAlgorithmOsimertinibMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行多目标优化
    print("\n" + "="*60)
    print("开始 Osimertinib MPO 多目标优化")
    print("="*60)
    
    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best
    
    # 6. 打印结果
    output_dir = config.get("output_dir", "osimertinib_mpo_results")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results(
        "Osimertinib_MPO", 
        best_score, 
        best_smi, 
        pareto_front, 
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics
    )

