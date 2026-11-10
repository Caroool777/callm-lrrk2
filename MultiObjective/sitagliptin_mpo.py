"""
Sitagliptin MPO 多目标优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 + 帕累托优化的多目标分子优化框架。

Sitagliptin MPO 任务包含以下目标：
1. Sitagliptin 相似度 (Tanimoto Similarity)
2. TPSA (Topological Polar Surface Area) - 目标范围 80-100
3. LogP - 目标范围 1-3
4. 氟原子数量 - 目标为 2-3 个

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


# Sitagliptin 的 SMILES 结构
SITAGLIPTIN_SMILES = "Fc1cc(c(F)cc1F)C[C@@H](N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"

# 目标 TPSA 和 LogP 范围
TARGET_TPSA_MIN = 80.0
TARGET_TPSA_MAX = 100.0
TARGET_LOGP_MIN = 1.0
TARGET_LOGP_MAX = 3.0
TARGET_FLUORINE_MIN = 2
TARGET_FLUORINE_MAX = 3


class SitagliptinMPOOracle:
    """
    Sitagliptin MPO 多目标 Oracle
    
    返回四个目标分数：
    1. Sitagliptin 相似度 (0-1)
    2. TPSA 分数 (0-1)
    3. LogP 分数 (0-1)
    4. 氟原子分数 (0-1)
    """
    
    def __init__(self):
        print("正在初始化 Sitagliptin MPO Oracle...")
        
        # 计算 Sitagliptin 的指纹
        self.sitagliptin_mol = Chem.MolFromSmiles(SITAGLIPTIN_SMILES)
        self.sitagliptin_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.sitagliptin_mol, radius=2, nBits=2048
        )
        
        print(f"Sitagliptin SMILES: {SITAGLIPTIN_SMILES}")
        print(f"目标 TPSA 范围: {TARGET_TPSA_MIN}-{TARGET_TPSA_MAX}")
        print(f"目标 LogP 范围: {TARGET_LOGP_MIN}-{TARGET_LOGP_MAX}")
        print(f"目标氟原子数: {TARGET_FLUORINE_MIN}-{TARGET_FLUORINE_MAX}")
        print("✅ Sitagliptin MPO Oracle 初始化完成")
    
    def compute_similarity(self, mol):
        """计算与 Sitagliptin 的 Tanimoto 相似度"""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(self.sitagliptin_fp, fp)
        except Exception:
            return 0.0
    
    def compute_tpsa_score(self, mol):
        """
        计算 TPSA 分数
        
        使用高斯函数，目标范围内得分接近 1，范围外衰减
        """
        try:
            tpsa = Descriptors.TPSA(mol)
            
            if TARGET_TPSA_MIN <= tpsa <= TARGET_TPSA_MAX:
                return 1.0
            else:
                # 高斯衰减
                distance = min(abs(tpsa - TARGET_TPSA_MIN), abs(tpsa - TARGET_TPSA_MAX))
                return np.exp(-0.5 * (distance / 20) ** 2)
        except Exception:
            return 0.0
    
    def compute_logp_score(self, mol):
        """
        计算 LogP 分数
        
        使用高斯函数，目标范围内得分接近 1，范围外衰减
        """
        try:
            logp = Descriptors.MolLogP(mol)
            
            if TARGET_LOGP_MIN <= logp <= TARGET_LOGP_MAX:
                return 1.0
            else:
                # 高斯衰减
                distance = min(abs(logp - TARGET_LOGP_MIN), abs(logp - TARGET_LOGP_MAX))
                return np.exp(-0.5 * (distance / 1.0) ** 2)
        except Exception:
            return 0.0
    
    def compute_fluorine_score(self, mol):
        """
        计算氟原子数量分数
        
        目标是 2-3 个氟原子
        """
        try:
            fluorine_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
            
            if TARGET_FLUORINE_MIN <= fluorine_count <= TARGET_FLUORINE_MAX:
                return 1.0
            elif fluorine_count == 1 or fluorine_count == 4:
                return 0.75
            elif fluorine_count == 0 or fluorine_count == 5:
                return 0.5
            else:
                return max(0.1, 1.0 - 0.15 * abs(fluorine_count - 2.5))
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, tpsa_score, logp_score, fluorine_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0, 0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_tpsa_score(mol),
            self.compute_logp_score(mol),
            self.compute_fluorine_score(mol)
        ]
    
    def score_aggregated(self, smiles):
        """
        计算聚合分数（几何平均）
        
        :param smiles: SMILES 字符串
        :return: float
        """
        objectives = self.score_all(smiles)
        # 避免零值
        objectives = [max(o, 1e-6) for o in objectives]
        return np.exp(np.mean(np.log(objectives)))


class CulturalAlgorithmSitagliptinMPO(CulturalAlgorithmMOBase):
    """
    Sitagliptin MPO 任务的多目标文化算法优化器
    """
    
    TASK_NAME = "sitagliptin_mpo"
    OBJECTIVE_NAMES = ["similarity", "tpsa", "logp", "fluorine"]
    
    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        """
        初始化 Sitagliptin MPO 优化器
        
        :param oracle: SitagliptinMPOOracle 实例
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: GPU 设备
        """
        self.mpo_oracle = oracle
        
        # 创建各个目标的单独 Oracle 函数
        oracle_funcs = [
            lambda smi, o=oracle: o.compute_similarity(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_tpsa_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_logp_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
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
        :return: [similarity, tpsa, logp, fluorine]
        """
        return self.mpo_oracle.score_all(smiles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sitagliptin MPO 多目标分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--max_calls', type=int, default=10000, help='最大 Oracle 调用次数')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_dir)')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="sitagliptin_mpo")
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
    
    # 2. 准备多目标 Oracle
    oracle = SitagliptinMPOOracle()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建多目标优化器
    max_calls = config.get("max_oracle_calls", args.max_calls)
    print(f"✅ Oracle 最大调用次数设置为: {max_calls}")  # 添加调试输出
    optimizer = CulturalAlgorithmSitagliptinMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行多目标优化
    print("\n" + "="*60)
    print("开始 Sitagliptin MPO 多目标优化")
    print("="*60)
    
    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best
    
    # 6. 打印结果
    output_dir = config.get("output_dir", "sitagliptin_mpo_results")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results(
        "Sitagliptin_MPO", 
        best_score, 
        best_smi, 
        pareto_front, 
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics
    )

