import sys
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem.QED import qed as rdkit_qed


def is_chemically_valid(smiles):
    """
    检查分子是否化学合理
    
    过滤掉含有化学上不稳定或不合理结构的分子，包括：
    - 次卤酸酯 (O-X, X=Cl, Br, I)：极不稳定的结构
    - 过氧化物 (O-O)：不稳定
    - 叠氮化物 (N=N=N)：爆炸性
    - 酰卤 (C(=O)X) 在某些情况下可接受，但需谨慎
    
    :param smiles: SMILES 字符串
    :return: bool, True 表示化学合理
    """
    if smiles is None:
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    # 使用 SMARTS 模式检测不稳定结构
    unstable_patterns = [
        # 次卤酸酯 (Hypohalite esters): O-Cl, O-Br, O-I
        ('[OX2][Cl,Br,I]', '次卤酸酯 (O-X)'),
        # 卤素直接连接氧 (反向)
        ('[Cl,Br,I][OX2]', '卤素-氧键'),
        # N-卤化物 (N-Cl, N-Br, N-I) - 不稳定
        ('[NX3][Cl,Br,I]', 'N-卤化物'),
        # 过氧化物 (O-O) - 不稳定，但排除酯类
        ('[OX2][OX2]', '过氧化物 (O-O)'),
        # 三键氮连接卤素
        ('[N+]#[C-][Cl,Br,I]', '异腈卤化物'),
    ]
    
    for smarts, description in unstable_patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            # 可以取消注释下面的行来调试
            # print(f"  [Filter] 检测到不稳定结构 '{description}': {smiles[:50]}...")
            return False
    
    return True


class QEDOracle:
    """
    QED (Quantitative Estimate of Drug-likeness) Oracle
    使用 RDKit 内置的 QED 计算方法
    
    QED 评分范围: 0.0 - 1.0
    - 接近 1.0 表示具有良好的类药性
    - 接近 0.0 表示类药性较差
    
    Reference:
    - Bickerton, G.R., Paolini, G.V., Besnard, J., Muresan, S. & Hopkins, A.L. (2012)
      Quantifying the chemical beauty of drugs.
      Nature Chemistry, 4, 90-98.
    """
    def __init__(self):
        print("正在初始化 QED Oracle (基于 RDKit)...")

    def score(self, smiles):
        """
        评估分子的 QED 分数
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return rdkit_qed(mol)
        except Exception as e:
            return 0.0


class JNK3Oracle:
    """
    JNK3 抑制剂预测 Oracle。
    使用了 Therapeutics Data Commons (TDC) 的预训练模型。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - JNK3 Task Details: https://tdcommons.ai/functions/oracles/#jnk3
    """
    def __init__(self):
        print("正在初始化 TDC JNK3 Oracle (首次运行可能需要下载模型权重)...")
        # 'JNK3' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='JNK3')

    def score(self, smiles):
        """
        评估分子的 JNK3 活性概率。
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class DRD2Oracle:
    """
    DRD2 (Dopamine Receptor D2) 活性预测 Oracle。
    使用了 Therapeutics Data Commons (TDC) 的预训练模型。
    
    DRD2 是一种 G 蛋白偶联受体，是治疗精神分裂症、帕金森病等
    神经精神疾病的重要药物靶点。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - DRD2 Task Details: https://tdcommons.ai/functions/oracles/#drd2
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示预测对 DRD2 有高活性
    - 接近 0.0 表示预测对 DRD2 活性较低
    """
    def __init__(self):
        print("正在初始化 TDC DRD2 Oracle (首次运行可能需要下载模型权重)...")
        # 'DRD2' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='DRD2')

    def score(self, smiles):
        """
        评估分子的 DRD2 活性概率。
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class GSK3BOracle:
    """
    GSK3β (Glycogen Synthase Kinase 3 Beta) 抑制剂预测 Oracle。
    使用了 Therapeutics Data Commons (TDC) 的预训练模型。
    
    GSK3β 是一种丝氨酸/苏氨酸蛋白激酶，参与多种细胞过程，包括：
    - 糖原代谢
    - 细胞分化
    - 细胞凋亡
    
    GSK3β 是治疗阿尔茨海默病、2型糖尿病、双相情感障碍等疾病的
    重要药物靶点。抑制 GSK3β 已被证明对神经退行性疾病具有潜在
    治疗作用。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - GSK3β Task Details: https://tdcommons.ai/functions/oracles/#gsk3b
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示预测对 GSK3β 有高抑制活性
    - 接近 0.0 表示预测对 GSK3β 抑制活性较低
    """
    def __init__(self):
        print("正在初始化 TDC GSK3β Oracle (首次运行可能需要下载模型权重)...")
        # 'GSK3B' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='GSK3B')
        self._error_count = 0
        self._max_error_log = 5

    def score(self, smiles):
        """
        评估分子的 GSK3β 抑制活性概率。
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            self._error_count += 1
            if self._error_count <= self._max_error_log:
                print(f"[GSK3BOracle] 评分失败 #{self._error_count}: smiles={smiles}, error={e}")
            elif self._error_count == self._max_error_log + 1:
                print("[GSK3BOracle] 后续错误将静默，仅返回 0.0")
            return 0.0


class IsomersC7H8N2O2Oracle:
    """
    Isomers C7H8N2O2 Oracle - 分子式匹配任务
    
    目标：生成符合分子式 C7H8N2O2 的分子异构体
    
    这是一个组合优化任务，目标是找到具有特定分子式的有效分子。
    分子式 C7H8N2O2 对应的分子量约为 168.15 g/mol。
    
    评分基于分子与目标分子式的匹配程度：
    - 1.0: 完全匹配目标分子式
    - 0.0-1.0: 部分匹配（基于元素组成的几何平均分数）
    
    评分使用 TDC 的 Isomer_scoring 类，它计算生成分子的元素组成
    与目标分子式 C7H8N2O2 之间的相似度。
    
    Source / Citation:
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Isomers Task: https://tdcommons.ai/functions/oracles/#isomers
    """
    def __init__(self):
        print("正在初始化 TDC Isomers C7H8N2O2 Oracle...")
        # 直接从 tdc.chem_utils.oracle.oracle 导入 isomers_c7h8n2o2
        # 因为 TDC 0.3.8 版本的 chem_utils/__init__.py 没有正确导出这个函数
        from tdc.chem_utils.oracle.oracle import isomers_c7h8n2o2
        self.oracle = isomers_c7h8n2o2
        print("✅ Isomers C7H8N2O2 Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 C7H8N2O2 分子式的匹配程度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            score = self.oracle(smiles)
            return score
        except Exception as e:
            return 0.0


class IsomersC9H10N2O2PF2ClOracle:
    """
    Isomers C9H10N2O2PF2Cl Oracle - 分子式匹配任务
    
    目标：生成符合分子式 C9H10N2O2PF2Cl 的分子异构体
    - 9 个碳原子
    - 10 个氢原子
    - 2 个氮原子
    - 2 个氧原子
    - 1 个磷原子
    - 2 个氟原子
    - 1 个氯原子
    
    这是一个组合优化任务，目标是找到具有特定分子式的有效分子。
    
    评分基于分子与目标分子式的匹配程度：
    - 1.0: 完全匹配目标分子式
    - 0.0-1.0: 部分匹配（基于元素组成的几何平均分数）
    
    评分使用 TDC 的 Isomer_scoring 类，它计算生成分子的元素组成
    与目标分子式 C9H10N2O2PF2Cl 之间的相似度。
    
    Source / Citation:
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Isomers Task: https://tdcommons.ai/functions/oracles/#isomers
    """
    def __init__(self):
        print("正在初始化 TDC Isomers C9H10N2O2PF2Cl Oracle...")
        # 使用 TDC 的 Isomer_scoring 类来创建自定义分子式评分器
        from tdc.chem_utils.oracle.oracle import Isomer_scoring
        self.oracle = Isomer_scoring("C9H10N2O2PF2Cl")
        print("✅ Isomers C9H10N2O2PF2Cl Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 C9H10N2O2PF2Cl 分子式的匹配程度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            score = self.oracle(smiles)
            return score
        except Exception as e:
            return 0.0


class DecoHopOracle:
    """
    Deco Hop Oracle - 装饰跳跃 (Decoration Hopping) 优化任务
    使用了 Therapeutics Data Commons (TDC) 的预训练模型。
    
    Deco Hop (Decoration Hopping) 是一种分子优化任务，目标是在保持
    分子核心骨架不变的情况下，通过修改侧链或取代基（装饰基团）来
    优化分子的性质。
    
    这是药物化学中常用的优化策略，也称为 "bioisosteric replacement"
    或 "scaffold decoration"。目标是找到与参考分子具有相似骨架但
    具有更好性质的新分子。
    
    评分基于：
    - 与参考分子骨架的相似性
    - 药物性质的优化程度
    - 化学结构的合理性
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Deco Hop Task: https://tdcommons.ai/functions/oracles/#deco-hop
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示在保持骨架相似性的同时优化了性质
    - 接近 0.0 表示未能有效优化或骨架差异过大
    """
    def __init__(self, use_chemical_filter=False):
        print("正在初始化 TDC Deco Hop Oracle...")
        from tdc import Oracle
        self.oracle = Oracle(name='Deco Hop')
        self.use_chemical_filter = use_chemical_filter
        self.filtered_count = 0  # 统计被过滤的分子数量
        if use_chemical_filter:
            print("✅ Deco Hop Oracle 初始化完成 (已启用化学合理性过滤器)")
        else:
            print("✅ Deco Hop Oracle 初始化完成 (未启用化学合理性过滤器)")

    def score(self, smiles):
        """
        评估分子的 Deco Hop 分数
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # 化学合理性过滤：拒绝含有不稳定结构的分子
            if self.use_chemical_filter and not is_chemically_valid(smiles):
                self.filtered_count += 1
                return 0.0
            
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class ScaffoldHopOracle:
    """
    Scaffold Hop Oracle - 骨架跳跃 (Scaffold Hopping) 优化任务
    使用了 Therapeutics Data Commons (TDC) 的预训练模型。
    
    Scaffold Hop (Scaffold Hopping) 是一种分子优化任务，目标是找到具有
    不同核心骨架但保持相似药理活性的分子。这与 Deco Hop 相反：
    - Deco Hop: 保持骨架，改变侧链
    - Scaffold Hop: 改变骨架，保持药理活性
    
    这是药物化学中突破专利限制和发现新化学实体 (NCE) 的重要策略。
    
    评分基于：
    - 与参考分子的药效团相似性 (而非骨架相似性)
    - 骨架的新颖性 (骨架应该不同)
    - 药物性质的保持或优化
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Scaffold Hop Task: https://tdcommons.ai/functions/oracles/#scaffold-hop
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示成功找到具有不同骨架但保持活性的分子
    - 接近 0.0 表示骨架相似度过高或活性丢失
    """
    def __init__(self, use_chemical_filter=False):
        print("正在初始化 TDC Scaffold Hop Oracle...")
        # Use TDC's public API so the installed version resolves the canonical
        # benchmark objective.
        self.oracle = Oracle(name='Scaffold Hop')
        self.use_chemical_filter = use_chemical_filter
        self.filtered_count = 0  # 统计被过滤的分子数量
        if use_chemical_filter:
            print("✅ Scaffold Hop Oracle 初始化完成 (已启用化学合理性过滤器)")
        else:
            print("✅ Scaffold Hop Oracle 初始化完成 (未启用化学合理性过滤器)")

    def score(self, smiles):
        """
        评估分子的 Scaffold Hop 分数
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # 化学合理性过滤：拒绝含有不稳定结构的分子
            if self.use_chemical_filter and not is_chemically_valid(smiles):
                self.filtered_count += 1
                return 0.0
            
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return float(score)
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class CelecoxibRediscoveryOracle:
    """
    Celecoxib Rediscovery Oracle - 塞来昔布重发现任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Celecoxib (塞来昔布) 是一种选择性 COX-2 抑制剂，用于治疗关节炎和急性疼痛。
    它是辉瑞公司开发的非甾体抗炎药 (NSAID)。
    
    Celecoxib 的 SMILES: Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1
    
    该任务是一个分子重发现任务，目标是重新发现（生成）与 Celecoxib 结构
    高度相似的分子。评分基于生成分子与 Celecoxib 之间的 Tanimoto 相似度。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Celecoxib Rediscovery Task: https://tdcommons.ai/functions/oracles/#celecoxib-rediscovery
    
    评分范围: 0.0 - 1.0
    - 1.0 表示生成的分子与 Celecoxib 完全相同
    - 接近 1.0 表示高度相似
    - 接近 0.0 表示相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Celecoxib Rediscovery Oracle...")
        # 'Celecoxib_Rediscovery' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Celecoxib_Rediscovery')
        print("✅ Celecoxib Rediscovery Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Celecoxib 的相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class ThiothixeneRediscoveryOracle:
    """
    Thiothixene Rediscovery Oracle - 泰尔登重发现任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Thiothixene (泰尔登/噻吨嗪) 是一种噻吨类 (thioxanthene) 抗精神病药物，
    主要用于治疗精神分裂症。它是一种多巴胺受体拮抗剂。
    
    Thiothixene 的 SMILES: CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1
    
    该任务是一个分子重发现任务，目标是重新发现（生成）与 Thiothixene 结构
    高度相似的分子。评分基于生成分子与 Thiothixene 之间的 Tanimoto 相似度。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Thiothixene Rediscovery Task: https://tdcommons.ai/functions/oracles/#thiothixene-rediscovery
    
    评分范围: 0.0 - 1.0
    - 1.0 表示生成的分子与 Thiothixene 完全相同
    - 接近 1.0 表示高度相似
    - 接近 0.0 表示相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Thiothixene Rediscovery Oracle...")
        # 'Thiothixene_Rediscovery' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Thiothixene_Rediscovery')
        print("✅ Thiothixene Rediscovery Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Thiothixene 的相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class TroglitazoneRediscoveryOracle:
    """
    Troglitazone Rediscovery Oracle - 曲格列酮重发现任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Troglitazone (曲格列酮) 是一种噻唑烷二酮类 (thiazolidinedione) 药物，
    曾用于治疗2型糖尿病。它是一种 PPARγ (过氧化物酶体增殖物激活受体γ) 激动剂。
    注：该药物因肝毒性已从市场撤回，但其结构仍具有研究价值。
    
    Troglitazone 的 SMILES: Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O
    
    该任务是一个分子重发现任务，目标是重新发现（生成）与 Troglitazone 结构
    高度相似的分子。评分基于生成分子与 Troglitazone 之间的 Tanimoto 相似度。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Troglitazone Rediscovery Task: https://tdcommons.ai/functions/oracles/#troglitazone-rediscovery
    
    评分范围: 0.0 - 1.0
    - 1.0 表示生成的分子与 Troglitazone 完全相同
    - 接近 1.0 表示高度相似
    - 接近 0.0 表示相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Troglitazone Rediscovery Oracle...")
        # 'Troglitazone_Rediscovery' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Troglitazone_Rediscovery')
        print("✅ Troglitazone Rediscovery Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Troglitazone 的相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class AlbuterolSimilarityOracle:
    """
    Albuterol Similarity Oracle - 沙丁胺醇相似性任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Albuterol (沙丁胺醇，又名 Salbutamol) 是一种短效 β2 肾上腺素受体激动剂，
    主要用于治疗哮喘和慢性阻塞性肺疾病 (COPD)。它是一种支气管扩张剂。
    
    Albuterol 的 SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
    
    该任务是一个分子相似性任务，目标是生成与 Albuterol 结构相似的分子。
    与 Rediscovery 任务不同，Similarity 任务允许更大的结构变异，
    目标是找到具有相似药理活性但可能具有更好性质的分子。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Albuterol Similarity Task: https://tdcommons.ai/functions/oracles/#albuterol-similarity
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示高度相似
    - 接近 0.0 表示相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Albuterol Similarity Oracle...")
        # 'Albuterol_Similarity' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Albuterol_Similarity')
        print("✅ Albuterol Similarity Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Albuterol 的相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class MestranolSimilarityOracle:
    """
    Mestranol Similarity Oracle - 美雌醇相似性任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Mestranol (美雌醇) 是一种合成雌激素，曾广泛用于口服避孕药中。
    它是炔雌醇 (Ethinylestradiol) 的 3-甲基醚前药，在体内被代谢为炔雌醇发挥作用。
    
    Mestranol 的 SMILES: COc1ccc2c(c1)CCC1C2CCC2(C)C(O)CCC12C#C
    
    该任务是一个分子相似性任务，目标是生成与 Mestranol 结构相似的分子。
    与 Rediscovery 任务不同，Similarity 任务允许更大的结构变异，
    目标是找到具有相似药理活性但可能具有更好性质的分子。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Mestranol Similarity Task: https://tdcommons.ai/functions/oracles/#mestranol-similarity
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示高度相似
    - 接近 0.0 表示相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Mestranol Similarity Oracle...")
        # 'Mestranol_Similarity' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Mestranol_Similarity')
        print("✅ Mestranol Similarity Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Mestranol 的相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class Median1Oracle:
    """
    Median 1 Oracle - 中位数分子1优化任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Median 1 任务是一个多目标优化任务，目标是生成同时与两个参考分子
    (Camphor 和 Menthol) 具有相似结构的分子。
    
    参考分子:
    - Camphor (樟脑): CC1(C)C2CCC1(C)C(=O)C2
    - Menthol (薄荷醇): CC(C)C1CCC(C)CC1O
    
    评分计算方式：取分子与两个参考分子 Tanimoto 相似度的几何平均值。
    这要求生成的分子需要同时兼顾两个参考分子的结构特征。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Median 1 Task: https://tdcommons.ai/functions/oracles/#median-1
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示与两个参考分子都高度相似
    - 接近 0.0 表示与至少一个参考分子相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Median 1 Oracle...")
        # 'Median1' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Median1')
        print("✅ Median 1 Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Camphor 和 Menthol 的综合相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class Median2Oracle:
    """
    Median 2 Oracle - 中位数分子2优化任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Median 2 任务是一个多目标优化任务，目标是生成同时与两个参考分子
    (Tadalafil 和 Sildenafil) 具有相似结构的分子。
    
    参考分子:
    - Tadalafil (他达拉非): 用于治疗勃起功能障碍的 PDE5 抑制剂
      SMILES: O=C1N(C)C(=O)c2c1cc(c1ccc3OCOc3c1)c1[nH]c3ccccc3c21
    - Sildenafil (西地那非): 用于治疗勃起功能障碍的 PDE5 抑制剂
      SMILES: CCCc1nn(C)c2c1nc([nH]c2=O)c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1
    
    评分计算方式：取分子与两个参考分子 Tanimoto 相似度的几何平均值。
    这要求生成的分子需要同时兼顾两个参考分子的结构特征。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Median 2 Task: https://tdcommons.ai/functions/oracles/#median-2
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示与两个参考分子都高度相似
    - 接近 0.0 表示与至少一个参考分子相似度很低
    """
    def __init__(self):
        print("正在初始化 TDC Median 2 Oracle...")
        # 'Median2' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Median2')
        print("✅ Median 2 Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子与 Tadalafil 和 Sildenafil 的综合相似度
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


class ValsartanSmartsOracle:
    """
    Valsartan SMARTS Oracle - 基于 SMARTS 模式匹配的分子优化任务
    使用了 Therapeutics Data Commons (TDC) 的评分函数。
    
    Valsartan 是一种血管紧张素 II 受体拮抗剂 (ARB)，用于治疗高血压和心力衰竭。
    
    该任务基于 SMARTS 模式匹配评分，目标是生成包含 Valsartan 关键药效团的分子。
    评分基于分子与特定 SMARTS 模式的匹配程度。
    
    Source / Citation: 
    - TDC Codebase: https://github.com/mims-harvard/TDC
    - Valsartan SMARTS Task: https://tdcommons.ai/functions/oracles/#valsartan-smarts
    
    评分范围: 0.0 - 1.0
    - 接近 1.0 表示分子包含目标 SMARTS 模式
    - 接近 0.0 表示分子不包含目标模式
    """
    def __init__(self):
        print("正在初始化 TDC Valsartan SMARTS Oracle...")
        # 'Valsartan_SMARTS' 是 TDC 中定义的标准 Oracle 名称
        self.oracle = Oracle(name='Valsartan_SMARTS')
        print("✅ Valsartan SMARTS Oracle 初始化完成")

    def score(self, smiles):
        """
        评估分子的 Valsartan SMARTS 匹配分数
        :param smiles: SMILES 字符串
        :return: float (0.0 到 1.0)
        """
        try:
            # TDC 的 Oracle 直接返回 0-1 之间的分数
            score = self.oracle(smiles)
            return score
        except Exception as e:
            # print(f"Oracle error for {smiles}: {e}")
            return 0.0


# 简单的单例测试
if __name__ == "__main__":
    print("="*50)
    print("测试 JNK3 Oracle")
    print("="*50)
    scorer_jnk3 = JNK3Oracle()
    test_smi = "CC1=CC=C(C=C1)C"
    print(f"Test SMILES: {test_smi}")
    print(f"JNK3 Score: {scorer_jnk3.score(test_smi)}")
    
    print("\n" + "="*50)
    print("测试 QED Oracle")
    print("="*50)
    scorer_qed = QEDOracle()
    # 测试一个类药分子
    drug_like_smi = "CC(=O)Oc1ccccc1C(=O)O"  # 阿司匹林
    print(f"Test SMILES (Aspirin): {drug_like_smi}")
    print(f"QED Score: {scorer_qed.score(drug_like_smi)}")
    print(f"Test SMILES (Random): {test_smi}")
    print(f"QED Score: {scorer_qed.score(test_smi)}")
    
    print("\n" + "="*50)
    print("测试 DRD2 Oracle")
    print("="*50)
    scorer_drd2 = DRD2Oracle()
    print(f"Test SMILES: {test_smi}")
    print(f"DRD2 Score: {scorer_drd2.score(test_smi)}")
    
    print("\n" + "="*50)
    print("测试 GSK3β Oracle")
    print("="*50)
    scorer_gsk3b = GSK3BOracle()
    print(f"Test SMILES: {test_smi}")
    print(f"GSK3β Score: {scorer_gsk3b.score(test_smi)}")
    
    print("\n" + "="*50)
    print("测试 Isomers C7H8N2O2 Oracle")
    print("="*50)
    scorer_isomers = IsomersC7H8N2O2Oracle()
    # 测试一个符合 C7H8N2O2 分子式的分子
    test_smi_isomer = "Cc1ncc(C(N)=O)c(N)n1"  # 可能的 C7H8N2O2 异构体
    print(f"Test SMILES: {test_smi_isomer}")
    print(f"Isomers C7H8N2O2 Score: {scorer_isomers.score(test_smi_isomer)}")
    
    print("\n" + "="*50)
    print("测试 Isomers C9H10N2O2PF2Cl Oracle")
    print("="*50)
    scorer_isomers_c9 = IsomersC9H10N2O2PF2ClOracle()
    # 测试一个示例分子
    test_smi_c9 = "CC(C)NC(=O)c1ccc(F)c(F)c1Cl"  # 示例分子
    print(f"Test SMILES: {test_smi_c9}")
    print(f"Isomers C9H10N2O2PF2Cl Score: {scorer_isomers_c9.score(test_smi_c9)}")
    
    print("\n" + "="*50)
    print("测试 Deco Hop Oracle")
    print("="*50)
    scorer_deco_hop = DecoHopOracle()
    print(f"Test SMILES: {test_smi}")
    print(f"Deco Hop Score: {scorer_deco_hop.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Scaffold Hop Oracle")
    print("="*50)
    scorer_scaffold_hop = ScaffoldHopOracle()
    print(f"Test SMILES: {test_smi}")
    print(f"Scaffold Hop Score: {scorer_scaffold_hop.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Valsartan SMARTS Oracle")
    print("="*50)
    scorer_valsartan = ValsartanSmartsOracle()
    # 测试一个含有联苯-四氮唑结构的分子
    valsartan_smi = "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(=O)O)C(C)C"
    print(f"Test SMILES (Valsartan-like): {valsartan_smi}")
    print(f"Valsartan SMARTS Score: {scorer_valsartan.score(valsartan_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Valsartan SMARTS Score: {scorer_valsartan.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Celecoxib Rediscovery Oracle")
    print("="*50)
    scorer_celecoxib = CelecoxibRediscoveryOracle()
    # 测试 Celecoxib 本身，应该返回 1.0
    celecoxib_smi = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    print(f"Test SMILES (Celecoxib): {celecoxib_smi}")
    print(f"Celecoxib Rediscovery Score: {scorer_celecoxib.score(celecoxib_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Celecoxib Rediscovery Score: {scorer_celecoxib.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Thiothixene Rediscovery Oracle")
    print("="*50)
    scorer_thiothixene = ThiothixeneRediscoveryOracle()
    # 测试 Thiothixene 本身，应该返回 1.0
    thiothixene_smi = "CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1"
    print(f"Test SMILES (Thiothixene): {thiothixene_smi}")
    print(f"Thiothixene Rediscovery Score: {scorer_thiothixene.score(thiothixene_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Thiothixene Rediscovery Score: {scorer_thiothixene.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Troglitazone Rediscovery Oracle")
    print("="*50)
    scorer_troglitazone = TroglitazoneRediscoveryOracle()
    # 测试 Troglitazone 本身，应该返回 1.0
    troglitazone_smi = "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O"
    print(f"Test SMILES (Troglitazone): {troglitazone_smi}")
    print(f"Troglitazone Rediscovery Score: {scorer_troglitazone.score(troglitazone_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Troglitazone Rediscovery Score: {scorer_troglitazone.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Albuterol Similarity Oracle")
    print("="*50)
    scorer_albuterol = AlbuterolSimilarityOracle()
    # 测试 Albuterol 本身，应该返回 1.0
    albuterol_smi = "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"
    print(f"Test SMILES (Albuterol): {albuterol_smi}")
    print(f"Albuterol Similarity Score: {scorer_albuterol.score(albuterol_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Albuterol Similarity Score: {scorer_albuterol.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Mestranol Similarity Oracle")
    print("="*50)
    scorer_mestranol = MestranolSimilarityOracle()
    # 测试 Mestranol 本身，应该返回 1.0
    mestranol_smi = "COc1ccc2c(c1)CCC1C2CCC2(C)C(O)CCC12C#C"
    print(f"Test SMILES (Mestranol): {mestranol_smi}")
    print(f"Mestranol Similarity Score: {scorer_mestranol.score(mestranol_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Mestranol Similarity Score: {scorer_mestranol.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Median 1 Oracle")
    print("="*50)
    scorer_median1 = Median1Oracle()
    # 测试 Camphor，应该返回较高分数
    camphor_smi = "CC1(C)C2CCC1(C)C(=O)C2"
    print(f"Test SMILES (Camphor): {camphor_smi}")
    print(f"Median 1 Score: {scorer_median1.score(camphor_smi)}")
    # 测试 Menthol，应该返回较高分数
    menthol_smi = "CC(C)C1CCC(C)CC1O"
    print(f"Test SMILES (Menthol): {menthol_smi}")
    print(f"Median 1 Score: {scorer_median1.score(menthol_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Median 1 Score: {scorer_median1.score(test_smi)}")

    print("\n" + "="*50)
    print("测试 Median 2 Oracle")
    print("="*50)
    scorer_median2 = Median2Oracle()
    # 测试 Tadalafil，应该返回较高分数
    tadalafil_smi = "O=C1N(C)C(=O)c2c1cc(c1ccc3OCOc3c1)c1[nH]c3ccccc3c21"
    print(f"Test SMILES (Tadalafil): {tadalafil_smi}")
    print(f"Median 2 Score: {scorer_median2.score(tadalafil_smi)}")
    # 测试 Sildenafil，应该返回较高分数
    sildenafil_smi = "CCCc1nn(C)c2c1nc([nH]c2=O)c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"
    print(f"Test SMILES (Sildenafil): {sildenafil_smi}")
    print(f"Median 2 Score: {scorer_median2.score(sildenafil_smi)}")
    # 测试一个随机分子
    print(f"Test SMILES (Random): {test_smi}")
    print(f"Median 2 Score: {scorer_median2.score(test_smi)}")
