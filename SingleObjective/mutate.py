import random
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

# [已修复] 修正导入路径
import crossover as co 

# [已修复] 删除缺失文件的导入
# from main.molleo.utils import get_fp_scores
# from main.molleo.network import obtain_model_pred

rdBase.DisableLog('rdApp.error')

def delete_atom():
    choices = ['[*:1]~[D1:2]>>[*:1]', '[*:1]~[D2:2]~[*:3]>>[*:1]-[*:3]',
               '[*:1]~[D3:2](~[*;!H0:3])~[*:4]>>[*:1]-[*:3]-[*:4]',
               '[*:1]~[D4:2](~[*;!H0:3])(~[*;!H0:4])~[*:5]>>[*:1]-[*:3]-[*:4]-[*:5]',
               '[*:1]~[D4:2](~[*;!H0;!H1:3])(~[*:4])~[*:5]>>[*:1]-[*:3](-[*:4])-[*:5]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    return np.random.choice(choices, p=p)

def append_atom():
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
               ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
               ['triple', ['C', 'N'], 2 * [1.0 / 2.0]]]
    p_BO = [0.60, 0.35, 0.05]
    index = np.random.choice(list(range(3)), p=p_BO)
    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)
    if BO == 'single': rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)
    if BO == 'double': rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X', '=' + new_atom)
    if BO == 'triple': rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X', '#' + new_atom)
    return rxn_smarts

def insert_atom():
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
               ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
               ['triple', ['C'], [1.0]]]
    p_BO = [0.60, 0.35, 0.05]
    index = np.random.choice(list(range(3)), p=p_BO)
    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)
    if BO == 'single': rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X', new_atom)
    if BO == 'double': rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X', new_atom)
    if BO == 'triple': rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X', new_atom)
    return rxn_smarts

def change_bond_order():
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]', '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
               '[*:1]#[*:2]>>[*:1]=[*:2]', '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    p = [0.45, 0.45, 0.05, 0.05]
    return np.random.choice(choices, p=p)

def delete_cyclic_bond():
    return '[*:1]@[*:2]>>([*:1].[*:2])'

def add_ring():
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
               '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1']
    p = [0.05, 0.05, 0.45, 0.45]
    return np.random.choice(choices, p=p)

def change_atom(mol):
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
    
    # [修复] 找出分子中实际存在的原子类型，避免无限循环
    available_X = [c for c in choices if mol.HasSubstructMatch(Chem.MolFromSmarts('[' + c + ']'))]
    if not available_X:
        # 如果没有可替换的原子，返回一个无效的反应（不会匹配任何东西）
        return '[#0:1]>>[#0:1]'
    
    X = random.choice(available_X)
    Y = np.random.choice(choices, p=p)
    
    # 确保 Y 和 X 不同，最多尝试 10 次
    attempts = 0
    while Y == X and attempts < 10:
        Y = np.random.choice(choices, p=p)
        attempts += 1
    
    return '[X:1]>>[Y:1]'.replace('X', X).replace('Y', Y)


def mutate(mol, mutation_rate, mol_lm=None, net=None):
    if mol is None: return None
    parent_smiles = Chem.MolToSmiles(mol, canonical=True)
    work_mol = Chem.RWMol(mol)
    try:
        Chem.Kekulize(work_mol, clearAromaticFlags=True)
    except ValueError:
        return None

    # [已修复] 这一块依赖 utils 和 network，且主程序未调用，直接注释掉避免报错
    if mol_lm is not None:
        print("Warning: mol_lm functionality disabled due to missing dependencies.")
        # 原有的逻辑需要 utils.get_fp_scores 等，此处省略
        pass 

    if random.random() > mutation_rate:
        return None

    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(work_mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        try:
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            new_mol_trial = rxn.RunReactants((work_mol,))
            new_mols = []
            for m in new_mol_trial:
                m = m[0]
                if co.mol_ok(m) and co.ring_OK(m):
                    new_mols.append(m)
            if len(new_mols) > 0:
                child = random.choice(new_mols)
                child_smiles = Chem.MolToSmiles(child, canonical=True)
                if child_smiles != parent_smiles:
                    return child
        except Exception:
            continue

    return None
