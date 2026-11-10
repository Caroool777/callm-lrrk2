"""
QED (药物相似性) 分子优化脚本

基于文化算法 (Cultural Algorithm) + BioT5 的分子优化框架，
继承自 CulturalAlgorithmBase 基类。

QED (Quantitative Estimate of Drug-likeness) 使用 RDKit 内置函数计算。
"""

import os
import random
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.QED import qed

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


from optimizer_base import CulturalAlgorithmBase, load_config, load_initial_population


class QEDOracle:
    """QED Oracle - 使用 RDKit 计算分子的药物相似性分数"""
    
    def __init__(self):
        print("正在初始化 QED Oracle (基于 RDKit)...")
        print("✅ QED Oracle 初始化完成")
    
    def score(self, smiles):
        if not smiles or not isinstance(smiles, str):
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        try:
            return qed(mol)
        except Exception:
            return 0.0


class CulturalAlgorithmQED(CulturalAlgorithmBase):
    """QED 任务的文化算法优化器
    
    注意：QED 任务使用较高的骨架阈值 (0.6)，已在 config_qed.json 中配置
    """
    
    TASK_NAME = "qed"
    
    def __init__(self, oracle_func, max_oracle_calls=500,
                 boredom_threshold=15, boredom_penalty=0.1, boredom_decay=0.01,
                 similarity_threshold=0.7, similarity_penalty=0.05, device=None):
        super().__init__(
            oracle_func=oracle_func,
            task_name=self.TASK_NAME,
            max_oracle_calls=max_oracle_calls,
            boredom_threshold=boredom_threshold,
            boredom_penalty=boredom_penalty,
            boredom_decay=boredom_decay,
            similarity_threshold=similarity_threshold,
            similarity_penalty=similarity_penalty,
            device=device
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='QED 分子优化')
    parser.add_argument('--device', type=str, default=None, help='指定GPU设备 (如 cuda:0, cuda:1)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (overrides config)')
    parser.add_argument('--max_calls', type=int, default=None, help='maximum oracle calls (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory (overrides config output_image_dir)')
    args = parser.parse_args()
    
    # 1. 加载配置
    config = load_config(task_name="qed")
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_calls is not None:
        config["max_oracle_calls"] = args.max_calls
    if args.output_dir is not None:
        config["output_image_dir"] = args.output_dir
    config["device"] = args.device
    
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"随机种子: {seed}")
    
    # 2. 准备 Oracle
    oracle = QEDOracle()
    
    # 3. 加载初始种群
    initial_pop = load_initial_population(config)
    
    # 4. 创建优化器
    optimizer = CulturalAlgorithmQED(
        oracle_func=oracle,
        max_oracle_calls=config.get("max_oracle_calls", 500),
        boredom_threshold=config.get("boredom_threshold", 15),
        boredom_penalty=config.get("boredom_penalty", 0.1),
        boredom_decay=config.get("boredom_decay", 0.01),
        similarity_threshold=config.get("similarity_threshold", 0.7),
        similarity_penalty=config.get("similarity_penalty", 0.05),
        device=args.device
    )
    
    optimizer.apply_config(config)
    
    # 5. 运行优化
    (best_score, best_mol, best_smi), top_10_auc, top_100_auc = optimizer.run(initial_pop)
    
    # 6. 打印结果并保存分子图
    output_dir = config.get("output_image_dir", "qed_top_molecules")
    # 将相对路径转换为基于脚本目录的绝对路径
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)
    optimizer.print_results("QED", best_score, best_smi, top_10_auc, top_100_auc,
                           output_dir=output_dir)

