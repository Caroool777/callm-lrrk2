import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')


def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring

    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None
    # [修复] 删除不可达的 return None


def cut_ring(mol):

    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None

    return None


CROSSOVER_FILTER_CONFIG = {
    "min_atom_count": 6,
    "max_atom_count": 70,
    "max_ring_size": 6,
}


def configure_mol_filters(min_atom_count=None, max_atom_count=None, max_ring_size=None):
    """Configure deterministic crossover filters."""
    if min_atom_count is not None:
        CROSSOVER_FILTER_CONFIG["min_atom_count"] = int(min_atom_count)
    if max_atom_count is not None:
        CROSSOVER_FILTER_CONFIG["max_atom_count"] = int(max_atom_count)
    if max_ring_size is not None:
        CROSSOVER_FILTER_CONFIG["max_ring_size"] = int(max_ring_size)


def ring_OK(mol):
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return True
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    return max(ring_sizes) <= CROSSOVER_FILTER_CONFIG["max_ring_size"]


def mol_ok(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    atom_count = mol.GetNumAtoms()
    if atom_count < CROSSOVER_FILTER_CONFIG["min_atom_count"]:
        return False
    if atom_count > CROSSOVER_FILTER_CONFIG["max_atom_count"]:
        return False
    if not ring_OK(mol):
        return False
    return True


def crossover_ring(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    prod = rxn1.RunReactants((fa, fb))
                    if len(prod) > 0:
                        new_mol_trial.append(prod[0])
                    #new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def crossover_non_ring(parent_A, parent_B):

    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                prod = rxn.RunReactants((fa, fb))
                if len(prod) > 0:
                    new_mol_trial.append(prod[0])
                #new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None


def crossover(parent_A, parent_B):
    parent_smiles = [Chem.MolToSmiles(parent_A, canonical=True), Chem.MolToSmiles(parent_B, canonical=True)]
    
    # [修复] 复制分子对象，避免修改原始对象
    parent_A = Chem.RWMol(parent_A)
    parent_B = Chem.RWMol(parent_B)
    
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.5:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    return None
