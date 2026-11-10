"""
Amlodipine MPO 多目标优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 + 帕累托优化的多目标分子优化框架。

Amlodipine MPO 任务包含以下目标：
1. Amlodipine 相似度 (Tanimoto Similarity, ECFP4)
2. 环数量 - 目标为 3 个环

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
from tdc import Oracle

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


# Amlodipine 的 SMILES 结构
# Amlodipine 是一种钙通道阻滞剂，用于治疗高血压和心绞痛
AMLODIPINE_SMILES = "CCOC(=O)C1=C(COCCN)NC(C)=C(C1c1ccccc1Cl)C(=O)OC"

# 目标环数量
TARGET_NUM_RINGS = 3
TARGET_NUM_RINGS_SIGMA = 0.5


class AmlodipineMPOOracle:
    """
    使用 TDC Oracle 的 Amlodipine MPO 评估器
    
    直接调用 TDC 的 Amlodipine_MPO oracle
    
    返回两个目标分数：
    1. Amlodipine 相似度 (0-1) - ECFP4 Tanimoto 相似度
    2. 环数量分数 (0-1) - 使用高斯函数评估，目标为 3 个环
    """
    
    def __init__(self):
        print("正在初始化 Amlodipine MPO Oracle (TDC)...")
        
        # 加载 TDC Oracle
        self.tdc_oracle = Oracle(name='Amlodipine_MPO')
        print("✅ TDC Amlodipine_MPO Oracle 加载成功")
        
        # 计算 Amlodipine 的指纹用于单独评估相似度
        self.amlodipine_mol = Chem.MolFromSmiles(AMLODIPINE_SMILES)
        self.amlodipine_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.amlodipine_mol, radius=2, nBits=2048
        )
        
        # 验证 Amlodipine 本身的环数量
        amlodipine_num_rings = rdMolDescriptors.CalcNumRings(self.amlodipine_mol)
        
        print(f"Amlodipine SMILES: {AMLODIPINE_SMILES}")
        print(f"Amlodipine 环数量: {amlodipine_num_rings}")
        print(f"目标环数量: {TARGET_NUM_RINGS} (sigma={TARGET_NUM_RINGS_SIGMA})")
        print("✅ Amlodipine MPO Oracle 初始化完成")
    
    def compute_similarity(self, mol):
        """
        计算与 Amlodipine 的 Tanimoto 相似度 (ECFP4)
        
        :param mol: RDKit Mol 对象
        :return: float, 相似度分数 (0-1)
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(self.amlodipine_fp, fp)
        except Exception:
            return 0.0
    
    def compute_rings_score(self, mol):
        """
        计算环数量分数
        
        使用高斯函数，目标为 3 个环
        Gaussian(3, 0.5) - 来自 GuacaMol Benchmark
        
        :param mol: RDKit Mol 对象
        :return: float, 环数量分数 (0-1)
        """
        try:
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            
            # 高斯函数: exp(-0.5 * ((x - mu) / sigma)^2)
            score = np.exp(-0.5 * ((num_rings - TARGET_NUM_RINGS) / TARGET_NUM_RINGS_SIGMA) ** 2)
            
            return score
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, rings_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_rings_score(mol)
        ]
    
    def score_aggregated(self, smiles):
        """
        计算聚合分数，直接使用 TDC oracle
        
        :param smiles: SMILES 字符串
        :return: float
        """
        try:
            score = self.tdc_oracle(smiles)
            return score if score is not None else 0.0
        except Exception:
            return 0.0


class CulturalAlgorithmAmlodipineMPO(CulturalAlgorithmMOBase):
    """
    Amlodipine MPO 任务的多目标文化算法优化器
    """
    
    TASK_NAME = "amlodipine_mpo"
    OBJECTIVE_NAMES = ["similarity", "rings"]
    
    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        """
        初始化 Amlodipine MPO 优化器
        
        :param oracle: AmlodipineMPOOracle 实例
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: GPU 设备
        """
        self.mpo_oracle = oracle
        
        # 创建各个目标的单独 Oracle 函数
        oracle_funcs = [
            lambda smi, o=oracle: o.compute_similarity(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_rings_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
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
        :return: [similarity, rings_score]
        """
        return self.mpo_oracle.score_all(smiles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Amlodipine MPO 多目标分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--max_calls', type=int, default=10000, help='最大 Oracle 调用次数')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_dir)')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="amlodipine_mpo")
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
    
    # 2. 准备 TDC Oracle
    oracle = AmlodipineMPOOracle()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建多目标优化器
    max_calls = config.get("max_oracle_calls", args.max_calls)
    print(f"✅ Oracle 最大调用次数设置为: {max_calls}")
    optimizer = CulturalAlgorithmAmlodipineMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行多目标优化
    print("\n" + "="*60)
    print("开始 Amlodipine MPO 多目标优化")
    print("="*60)
    
    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best
    
    # 6. 打印结果
    output_dir = config.get("output_dir", "amlodipine_mpo_results")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results(
        "Amlodipine_MPO", 
        best_score, 
        best_smi, 
        pareto_front, 
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics
    )

