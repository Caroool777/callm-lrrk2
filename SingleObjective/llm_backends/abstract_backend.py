"""
LLM 后端抽象基类 - CALLM 框架统一接口

定义所有 LLM 后端必须实现的接口：
- BioT5 (本地 Hugging Face 模型)

这是策略模式的核心，允许运行时切换不同的 LLM 实现。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
from rdkit import Chem


class LLMBackend(ABC):
    """
    LLM 后端的抽象基类
    
    BioT5 后端实现应继承此类。
    """
    
    def __init__(
        self,
        num_beams: int = 5,
        num_return_sequences: int = 3,
        temperature: float = 0.8,
        do_sample: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 LLM 后端
        
        Args:
            num_beams: Beam search 的束数量
            num_return_sequences: 每个输入生成多少个输出
            temperature: 采样温度（越高越随机）
            do_sample: 是否启用采样
            device: 设备标识符（GPU/CPU）
        """
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = device
        self.task: Optional[tuple] = None
        self.task2description: Dict[str, str] = {}
        self.task2constrained_description: Dict[str, str] = {}
    
    def set_task(self, task_name: str) -> None:
        """
        设置优化任务
        
        Args:
            task_name: 任务名称
        
        Raises:
            ValueError: 如果任务不存在
        """
        if task_name not in self.task2description:
            raise ValueError(
                f"Task '{task_name}' not found in {self.__class__.__name__}. "
                f"Available tasks: {list(self.task2description.keys())}"
            )
        self.task = (task_name,)
        print(f"[{self.__class__.__name__}] Task set to: {task_name}")
    
    @abstractmethod
    def edit(
        self,
        smiles_list: List,
        scaffold_constraints: Optional[Union[str, List[str]]] = None
    ) -> List:
        """
        编辑分子列表，生成候选分子
        
        【文化算法增强】支持骨架约束，实现"保留骨架，优化侧链"的填空题模式。
        
        Args:
            smiles_list: RDKit Mol 对象列表
            scaffold_constraints: 骨架约束（SMILES），可选
                - None: 无约束
                - str: 所有分子使用同一个约束
                - list: 每个分子使用对应的约束
        
        Returns:
            编辑后的 Mol 对象列表
        
        Raises:
            RuntimeError: 如果任务未设置或其他运行时错误
        """
        pass
    
    @abstractmethod
    def _build_prompt(
        self,
        task_name: str,
        selfies_input: str,
        scaffold_constraint: Optional[str] = None
    ) -> str:
        """
        构建模型输入 Prompt
        
        这是文化算法的核心 - 将信念空间知识注入生成过程。
        
        Args:
            task_name: 任务名称
            selfies_input: 输入分子的 SELFIES 表示
            scaffold_constraint: 骨架约束（来自信念空间）
        
        Returns:
            完整的 Prompt 字符串
        """
        pass
    
    @abstractmethod
    def _generate_candidate_mols(
        self,
        task_prompt: str,
        input_smiles: str,
        input_selfies: str,
        scaffold_constraint: Optional[str] = None,
    ) -> List[Chem.Mol]:
        """
        使用 LLM 生成候选分子
        
        Args:
            task_prompt: 完整的任务描述 Prompt
            input_smiles: 输入分子的标准 SMILES
            input_selfies: 输入分子的 SELFIES 表示
            scaffold_constraint: 骨架约束 SMILES
        
        Returns:
            生成的 RDKit Mol 对象列表
        """
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """返回后端名称"""
        pass
    
    @abstractmethod
    def apply_config(self, config: Dict) -> None:
        """
        应用配置文件中的参数
        
        Args:
            config: 配置字典
        """
        pass
