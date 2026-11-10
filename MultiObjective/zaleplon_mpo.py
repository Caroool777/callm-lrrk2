"""
Zaleplon MPO 多目标优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 + 帕累托优化的多目标分子优化框架。

Zaleplon MPO 任务包含以下目标：
1. Zaleplon 相似度 (Tanimoto Similarity, ECFP4)
2. 分子式匹配 (目标分子式: C19H17N3O2)

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


# Zaleplon 的 SMILES 结构
ZALEPLON_SMILES = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"

# 目标分子式
TARGET_FORMULA = "C19H17N3O2"


class ZaleplonMPOOracle:
    """
    Zaleplon MPO 多目标 Oracle
    
    返回两个目标分数：
    1. Zaleplon 相似度 (0-1) - ECFP4 Tanimoto 相似度
    2. 分子式匹配分数 (0-1) - 与目标分子式 C19H17N3O2 的匹配程度
    """
    
    def __init__(self):
        print("正在初始化 Zaleplon MPO Oracle...")
        
        # 计算 Zaleplon 的指纹 (ECFP4)
        self.zaleplon_mol = Chem.MolFromSmiles(ZALEPLON_SMILES)
        self.zaleplon_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.zaleplon_mol, radius=2, nBits=2048
        )
        
        # 解析目标分子式
        self.target_formula = TARGET_FORMULA
        self.target_formula_dict = self._parse_formula(TARGET_FORMULA)
        
        print(f"Zaleplon SMILES: {ZALEPLON_SMILES}")
        print(f"目标分子式: {TARGET_FORMULA}")
        print("✅ Zaleplon MPO Oracle 初始化完成")
    
    def _parse_formula(self, formula):
        """
        解析分子式字符串为原子计数字典
        
        :param formula: 分子式字符串，如 "C19H17N3O2"
        :return: 字典，如 {'C': 19, 'H': 17, 'N': 3, 'O': 2}
        """
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        result = {}
        for element, count in matches:
            if element:
                result[element] = int(count) if count else 1
        return result
    
    def _get_mol_formula_dict(self, mol):
        """
        获取分子的原子计数字典
        
        :param mol: RDKit Mol 对象
        :return: 字典，原子符号到计数的映射
        """
        formula_dict = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
        
        # 添加氢原子计数
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == 'H')
        if h_count > 0:
            formula_dict['H'] = h_count
        
        return formula_dict
    
    def compute_similarity(self, mol):
        """
        计算与 Zaleplon 的 Tanimoto 相似度 (ECFP4)
        
        :param mol: RDKit Mol 对象
        :return: float, 相似度分数 (0-1)
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(self.zaleplon_fp, fp)
        except Exception:
            return 0.0
    
    def compute_formula_score(self, mol):
        """
        计算分子式匹配分数
        
        使用高斯函数基于原子差异计算分数
        目标分子式: C19H17N3O2
        
        :param mol: RDKit Mol 对象
        :return: float, 分子式匹配分数 (0-1)
        """
        try:
            mol_formula_dict = self._get_mol_formula_dict(mol)
            
            # 计算所有相关元素的差异
            all_elements = set(self.target_formula_dict.keys()) | set(mol_formula_dict.keys())
            
            total_diff = 0
            for element in all_elements:
                target_count = self.target_formula_dict.get(element, 0)
                mol_count = mol_formula_dict.get(element, 0)
                total_diff += abs(target_count - mol_count)
            
            # 使用指数衰减函数
            # 当差异为0时，分数为1
            # 差异越大，分数越低
            score = np.exp(-0.5 * total_diff)
            
            return score
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, formula_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_formula_score(mol)
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


class ZaleplonMPOOracleTDC:
    """
    使用 TDC Oracle 的 Zaleplon MPO 评估器
    
    直接调用 TDC 的 Zaleplon_MPO oracle
    """
    
    def __init__(self):
        print("正在初始化 Zaleplon MPO Oracle (TDC)...")
        
        try:
            from tdc import Oracle
            self.tdc_oracle = Oracle(name='Zaleplon_MPO')
            self.use_tdc = True
            print("✅ TDC Zaleplon_MPO Oracle 加载成功")
        except Exception as e:
            print(f"⚠️ TDC Oracle 加载失败: {e}")
            print("将使用本地实现")
            self.use_tdc = False
            self._init_local_oracle()
        
        # 计算 Zaleplon 的指纹用于单独评估相似度
        self.zaleplon_mol = Chem.MolFromSmiles(ZALEPLON_SMILES)
        self.zaleplon_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.zaleplon_mol, radius=2, nBits=2048
        )
        self.target_formula_dict = self._parse_formula(TARGET_FORMULA)
        
        print(f"Zaleplon SMILES: {ZALEPLON_SMILES}")
        print(f"目标分子式: {TARGET_FORMULA}")
        print("✅ Zaleplon MPO Oracle 初始化完成")
    
    def _init_local_oracle(self):
        """初始化本地 Oracle 作为备用"""
        pass
    
    def _parse_formula(self, formula):
        """解析分子式字符串为原子计数字典"""
        import re
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        result = {}
        for element, count in matches:
            if element:
                result[element] = int(count) if count else 1
        return result
    
    def _get_mol_formula_dict(self, mol):
        """获取分子的原子计数字典"""
        formula_dict = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
        
        mol_with_h = Chem.AddHs(mol)
        h_count = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == 'H')
        if h_count > 0:
            formula_dict['H'] = h_count
        
        return formula_dict
    
    def compute_similarity(self, mol):
        """计算与 Zaleplon 的 Tanimoto 相似度"""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return DataStructs.TanimotoSimilarity(self.zaleplon_fp, fp)
        except Exception:
            return 0.0
    
    def compute_formula_score(self, mol):
        """计算分子式匹配分数"""
        try:
            mol_formula_dict = self._get_mol_formula_dict(mol)
            all_elements = set(self.target_formula_dict.keys()) | set(mol_formula_dict.keys())
            
            total_diff = 0
            for element in all_elements:
                target_count = self.target_formula_dict.get(element, 0)
                mol_count = mol_formula_dict.get(element, 0)
                total_diff += abs(target_count - mol_count)
            
            score = np.exp(-0.5 * total_diff)
            return score
        except Exception:
            return 0.0
    
    def score_all(self, smiles):
        """
        评估分子的所有目标分数
        
        :param smiles: SMILES 字符串
        :return: [similarity, formula_score]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0, 0.0]
        
        return [
            self.compute_similarity(mol),
            self.compute_formula_score(mol)
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


class CulturalAlgorithmZaleplonMPO(CulturalAlgorithmMOBase):
    """
    Zaleplon MPO 任务的多目标文化算法优化器
    """
    
    TASK_NAME = "zaleplon_mpo"
    OBJECTIVE_NAMES = ["similarity", "formula"]
    
    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        """
        初始化 Zaleplon MPO 优化器
        
        :param oracle: ZaleplonMPOOracle 或 ZaleplonMPOOracleTDC 实例
        :param max_oracle_calls: 最大 Oracle 调用次数
        :param device: GPU 设备
        """
        self.mpo_oracle = oracle
        
        # 创建各个目标的单独 Oracle 函数
        oracle_funcs = [
            lambda smi, o=oracle: o.compute_similarity(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
            lambda smi, o=oracle: o.compute_formula_score(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else 0.0,
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
        :return: [similarity, formula_score]
        """
        return self.mpo_oracle.score_all(smiles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Zaleplon MPO 多目标分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--max_calls', type=int, default=10000, help='最大 Oracle 调用次数')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_dir)')
    parser.add_argument('--use_tdc', action='store_true', help='使用 TDC Oracle')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="zaleplon_mpo")
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
    if args.use_tdc:
        oracle = ZaleplonMPOOracleTDC()
    else:
        oracle = ZaleplonMPOOracle()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建多目标优化器
    max_calls = config.get("max_oracle_calls", args.max_calls)
    print(f"✅ Oracle 最大调用次数设置为: {max_calls}")
    optimizer = CulturalAlgorithmZaleplonMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行多目标优化
    print("\n" + "="*60)
    print("开始 Zaleplon MPO 多目标优化")
    print("="*60)
    
    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best
    
    # 6. 打印结果
    output_dir = config.get("output_dir", "zaleplon_mpo_results")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results(
        "Zaleplon_MPO", 
        best_score, 
        best_smi, 
        pareto_front, 
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics
    )

