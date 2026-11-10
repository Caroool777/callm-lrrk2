"""Structure-preserving operators for the TDC PMO Deco Hop task."""

from itertools import product
import random
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.error")


REFERENCE_SMILES = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
CORE_SMARTS = "[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12"
BANNED_DECO_SMARTS = (
    "CS([#6])(=O)=O",
    "[#7]-c1ccc2ncsc2c1",
)

CORE_QUERY = Chem.MolFromSmarts(CORE_SMARTS)
BANNED_QUERIES = tuple(Chem.MolFromSmarts(value) for value in BANNED_DECO_SMARTS)

# Prefixes end at the oxygen attached to the fixed core.
O_TAILS = (
    "CO", "CCO", "CCCO", "CCCCO", "CCCCCO", "CC(C)O", "CC(C)CO",
    "CCC(C)O", "COCO", "CCOCO", "CCCOCO", "C(F)(F)CO", "NCCO",
    "CN(C)CCO", "C1CC1CO", "C1CCC1O", "c1ccccc1O", "c1ccccc1CO",
    "c1ccncc1O", "c1ncccc1CO", "CC#CCO", "C=CCO", "CC(F)CO",
    "CC(Cl)CO", "OCCO",
)

# Fragments start at the atom bonded to the exocyclic nitrogen in CORE_SMARTS.
N_DECORATIONS = (
    "c3ccccc3", "c3cccc(F)c3", "c3cccc(Cl)c3", "c3cccc(Br)c3",
    "c3cccc(C)c3", "c3cccc(OC)c3", "c3cccc(C#N)c3", "c3cccc(C(F)(F)F)c3",
    "c3ccncc3", "c3cnccc3", "c3ncccc3", "c3nccnc3", "c3cncnc3",
    "c3ncncc3", "c3ccnc(F)c3", "c3ccnc(C)c3", "c3ccnnc3",
    "c3cnncc3", "c3ccoc3", "c3ccsc3", "c3nocc3", "c3nscc3",
    "c3ccc4ccccc4c3", "c3ccc4ncccc4c3", "c3ccc4occc4c3",
    "c3ccc4sccc4c3", "c3ccc4[nH]ccc4c3", "c3nc4ccccc4n3",
    "c3ncc4ccccc4n3", "c3ccc(C(=O)N)cc3",
)

# Fragments replace the complete tBu-sulfone decoration. Every choice keeps the
# aromatic core carbon substituted, which preserves the [c;h0] atom constraint.
RING_SUBSTITUENTS = (
    "F", "Cl", "Br", "C", "CC", "CCC", "CCCC", "C(C)C", "C(C)(C)C",
    "C(F)(F)F", "C#N", "C=O", "C(=O)N", "C(=O)NC", "C(=O)N(C)C",
    "C(=O)OC", "C(=O)OCC", "OC", "OCC", "OCCC", "N", "NC", "NCC",
    "N(C)C", "S(=O)(=O)N", "S(=O)(=O)NC", "c3ccccc3", "c3ccncc3",
    "C3CC3", "C3CCC3",
)

Genotype = Tuple[str, str, str]


def is_valid_decohop_mol(mol: Optional[Chem.Mol]) -> bool:
    if mol is None or CORE_QUERY is None or any(query is None for query in BANNED_QUERIES):
        return False
    try:
        Chem.SanitizeMol(mol)
        return mol.HasSubstructMatch(CORE_QUERY) and not any(
            mol.HasSubstructMatch(query) for query in BANNED_QUERIES
        )
    except Exception:
        return False


def is_valid_decohop_candidate(smiles: str) -> bool:
    return is_valid_decohop_mol(Chem.MolFromSmiles(smiles))


def _build_mol(genotype: Genotype) -> Optional[Chem.Mol]:
    o_tail, n_decoration, ring_substituent = genotype
    smiles = f"{o_tail}c1cc2ncnc(N{n_decoration})c2cc1{ring_substituent}"
    mol = Chem.MolFromSmiles(smiles)
    if not is_valid_decohop_mol(mol):
        return None
    return mol


class DecoHopTemplateGenerator:
    """Enumerate and recombine only the three editable Deco Hop R-group sites."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(int(seed))
        self._catalog: List[Tuple[str, Genotype]] = []
        self._genotype_by_smiles: Dict[str, Genotype] = {}
        for genotype in product(O_TAILS, N_DECORATIONS, RING_SUBSTITUENTS):
            mol = _build_mol(genotype)
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if smiles in self._genotype_by_smiles:
                continue
            self._genotype_by_smiles[smiles] = genotype
            self._catalog.append((smiles, genotype))
        if len(self._catalog) < 10000:
            raise RuntimeError(
                f"Deco Hop template catalog has only {len(self._catalog)} valid molecules."
            )

    @property
    def catalog_size(self) -> int:
        return len(self._catalog)

    def _mol_for_genotype(self, genotype: Genotype) -> Optional[Chem.Mol]:
        mol = _build_mol(genotype)
        return Chem.Mol(mol) if mol is not None else None

    def _genotype_for_mol(self, mol: Optional[Chem.Mol]) -> Optional[Genotype]:
        if mol is None:
            return None
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None
        return self._genotype_by_smiles.get(smiles)

    def initial_population(self, count: int) -> List[Chem.Mol]:
        count = min(int(count), len(self._catalog))
        entries = self.rng.sample(self._catalog, count)
        return [Chem.MolFromSmiles(smiles) for smiles, _ in entries]

    def random_candidate(self, excluded_smiles: Iterable[str] = ()) -> Optional[Chem.Mol]:
        excluded = set(excluded_smiles)
        for _ in range(100):
            smiles, _ = self.rng.choice(self._catalog)
            if smiles not in excluded:
                return Chem.MolFromSmiles(smiles)
        return None

    def novel_candidates(self, count: int, excluded_smiles: Iterable[str]) -> List[Chem.Mol]:
        excluded = set(excluded_smiles)
        available = [entry for entry in self._catalog if entry[0] not in excluded]
        if not available:
            return []
        selected = self.rng.sample(available, min(int(count), len(available)))
        return [Chem.MolFromSmiles(smiles) for smiles, _ in selected]

    def crossover(self, parent_a: Chem.Mol, parent_b: Chem.Mol) -> Optional[Chem.Mol]:
        genotype_a = self._genotype_for_mol(parent_a)
        genotype_b = self._genotype_for_mol(parent_b)
        if genotype_a is None or genotype_b is None:
            return self.random_candidate()
        parent_smiles = {
            Chem.MolToSmiles(parent_a, canonical=True),
            Chem.MolToSmiles(parent_b, canonical=True),
        }
        for _ in range(20):
            mask = self.rng.choice(((True, False, False), (False, True, False),
                                    (False, False, True), (True, True, False),
                                    (True, False, True), (False, True, True)))
            genotype = tuple(
                genotype_a[index] if mask[index] else genotype_b[index]
                for index in range(3)
            )
            child = self._mol_for_genotype(genotype)
            if child is not None and Chem.MolToSmiles(child, canonical=True) not in parent_smiles:
                return child
        return self.random_candidate(parent_smiles)

    def mutate(self, parent: Chem.Mol, mutation_rate: float = 1.0) -> Optional[Chem.Mol]:
        if self.rng.random() > float(mutation_rate):
            return None
        genotype = self._genotype_for_mol(parent)
        if genotype is None:
            return self.random_candidate()
        choices = (O_TAILS, N_DECORATIONS, RING_SUBSTITUENTS)
        parent_smiles = Chem.MolToSmiles(parent, canonical=True)
        for _ in range(20):
            site = self.rng.randrange(3)
            replacement = self.rng.choice(choices[site])
            if replacement == genotype[site]:
                continue
            mutated = list(genotype)
            mutated[site] = replacement
            child = self._mol_for_genotype(tuple(mutated))
            if child is not None and Chem.MolToSmiles(child, canonical=True) != parent_smiles:
                return child
        return self.random_candidate((parent_smiles,))
