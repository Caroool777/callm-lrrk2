#!/usr/bin/env python3
"""Run the existing two-stage LRRK2 workflow with five weighted seed lineages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator

rdBase.DisableLog("rdApp.warning")

ROOT = Path(__file__).resolve().parents[1]
LRRK2_DIR = ROOT / "LRRK2case"
PYTHON_BIN = Path(sys.executable)
MASTER_SEED = 20260820
FINAL_TARGET = 20_000
MODEL_NAME = "QizhiPei/biot5-base-text2mol"
MODEL_PATH = ROOT / "lrrk2_qsar_oracle" / "lrrk2_qsar_oracle_model.pkl"
STAGE1_ENTRY = LRRK2_DIR / "lrrk2_wt_activity_mpo.py"
STAGE2_ENTRY = LRRK2_DIR / "lrrk2_wt_lead_repair_mpo.py"
STAGE1_CANONICAL = "lrrk2_wt_activity_mpo_all_discovered_canonical.csv"
STAGE2_QUALITY = "lrrk2_wt_lead_repair_mpo_structure_quality.csv"

SEEDS = [
    {
        "name": "DNL201_GNE-0877",
        "smiles": "CNc1nc(Nc2cn(C(C)(C)C#N)nc2C)ncc1C(F)(F)F",
        "weight": 0.30,
    },
    {
        "name": "PF-06447475",
        "smiles": "N#Cc1cccc(-c2c[nH]c3ncnc(N4CCOCC4)c23)c1",
        "weight": 0.25,
    },
    {
        "name": "MLi-2",
        "smiles": "C[C@@H]1CN(c2cc(-c3n[nH]c4ccc(OC5(C)CC5)cc34)ncn2)C[C@H](C)O1",
        "weight": 0.20,
    },
    {
        "name": "GNE-7915",
        "smiles": "CCNc1nc(Nc2cc(F)c(C(=O)N3CCOCC3)cc2OC)ncc1C(F)(F)F",
        "weight": 0.15,
    },
    {
        "name": "LRRK2_compound18_YXD",
        "smiles": "Cn1cc(-c2cccc(C(=O)Nc3cc(N4CC[C@H](C(C)(C)O)C4)cnc3C(F)(F)F)n2)cn1",
        "weight": 0.10,
    },
]

METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}
ALLOWED_ATOMIC_NUMBERS = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
REACTIVE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "ketene_or_cumulene": "[C,c]=[C]=[C,N,O,S]",
    "heterocumulene": "[N,O,S]=[C]=[N,O,S]",
}
COMPILED_REACTIVE = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in REACTIVE_SMARTS.items()
}
LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_log(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "run.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = float("-inf")) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def geometric_mean(values: Sequence[float]) -> float:
    values = [max(float(value), 1e-12) for value in values]
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def validate_and_write_seeds(run_dir: Path) -> List[Dict[str, Any]]:
    validated = []
    seen = set()
    total_weight = sum(seed["weight"] for seed in SEEDS)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Seed weights sum to {total_weight}, expected 1.0")

    for seed in SEEDS:
        mol = Chem.MolFromSmiles(seed["smiles"])
        if mol is None:
            raise ValueError(f"RDKit could not parse seed {seed['name']}")
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        roundtrip = Chem.MolFromSmiles(canonical)
        if roundtrip is None:
            raise ValueError(f"Canonical seed failed round-trip: {seed['name']}")
        Chem.SanitizeMol(roundtrip)
        roundtrip_canonical = Chem.MolToSmiles(roundtrip, canonical=True, isomericSmiles=True)
        if canonical != roundtrip_canonical:
            raise ValueError(f"Seed connectivity changed on round-trip: {seed['name']}")
        inchikey = Chem.MolToInchiKey(mol)
        if not inchikey or inchikey in seen:
            raise ValueError(f"Missing or duplicate seed InChIKey: {seed['name']}")
        seen.add(inchikey)

        chiral_centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False
        )
        specified = [(idx, label) for idx, label in chiral_centers if label != "?"]
        if seed["name"] in {"MLi-2", "LRRK2_compound18_YXD"}:
            if "@" not in seed["smiles"] or "@" not in canonical or not specified:
                raise ValueError(f"Specified stereochemistry was not preserved for {seed['name']}")

        record = dict(seed)
        record.update(
            {
                "canonical_isomeric_smiles": canonical,
                "inchikey": inchikey,
                "rdkit_parse": True,
                "rdkit_sanitize": True,
                "connectivity_roundtrip_preserved": True,
                "specified_chiral_centers": len(specified),
                "chiral_centers": [
                    {"atom_index": int(idx), "assignment": label} for idx, label in chiral_centers
                ],
            }
        )
        validated.append(record)

    payload = {
        "random_seed": MASTER_SEED,
        "weight_sum": total_weight,
        "seed_count": len(validated),
        "seeds": validated,
    }
    atomic_json(run_dir / "seeds.yaml", payload)
    seed_input_dir = run_dir / "inputs"
    seed_input_dir.mkdir(parents=True, exist_ok=True)
    for seed in validated:
        pd.DataFrame(
            [{"smiles": seed["canonical_isomeric_smiles"], "parent_seed": seed["name"]}]
        ).to_csv(seed_input_dir / f"{slugify(seed['name'])}.csv", index=False)
    return validated


def sample_allocations(total: int, random_seed: int, guarantee_all: bool = False) -> Dict[str, int]:
    rng = np.random.default_rng(random_seed)
    names = [seed["name"] for seed in SEEDS]
    weights = np.array([seed["weight"] for seed in SEEDS], dtype=float)
    sampled = rng.choice(len(names), size=int(total), replace=True, p=weights)
    counts = np.bincount(sampled, minlength=len(names)).astype(int)
    if guarantee_all:
        for index, count in enumerate(counts):
            if count == 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def clean_stage_templates() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stage1 = read_json(LRRK2_DIR / "config_lrrk2_wt_activity_mpo.json", {})
    stage2 = read_json(LRRK2_DIR / "config_lrrk2_wt_lead_repair_mpo.json", {})
    for key in ["initial_smiles_file", "output_dir", "seed"]:
        stage1.pop(key, None)
    for key in [
        "initial_smiles_file", "stage1_seed_file", "output_dir", "seed",
        "parent_lead_smiles", "lead_pool_smiles",
    ]:
        stage2.pop(key, None)
    return stage1, stage2


def write_resolved_config(
    run_dir: Path,
    validated_seeds: List[Dict[str, Any]],
    stage1_allocations: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stage1, stage2 = clean_stage_templates()
    resolved = {
        "run_name": run_dir.name,
        "random_seed": MASTER_SEED,
        "derived_seed_policy": "master + stage offset + round*100 + seed index",
        "target_final_unique_molecules": FINAL_TARGET,
        "stage1_sampling_events": int(sum(stage1_allocations.values())),
        "stage1_sampling_method": "probability sampling with fixed master RNG",
        "stage1_actual_parent_allocations": stage1_allocations,
        "seeds": [
            {
                "name": row["name"],
                "smiles": row["smiles"],
                "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                "inchikey": row["inchikey"],
                "weight": row["weight"],
            }
            for row in validated_seeds
        ],
        "stage1_entry": str(STAGE1_ENTRY),
        "stage2_entry": str(STAGE2_ENTRY),
        "stage1_base_config_without_io": stage1,
        "stage2_base_config_without_io": stage2,
        "stage2_thresholds": {
            "clean_candidate_min_aggregated_score": 0.85,
            "clean_candidate_min_activity_score": 0.55,
            "clean_candidate_min_bbb_cns": 0.8,
            "clean_candidate_min_logp_window": 0.8,
            "clean_candidate_min_tpsa_window": 0.8,
            "clean_candidate_min_ad_rank": 1,
        },
        "generation_model": {
            "name": MODEL_NAME,
            "local_checkpoint": str(MODEL_PATH),
            "sha256": sha256_file(MODEL_PATH),
        },
        "stage2_incremental_round_budget": args.stage2_round_budget,
        "stage2_max_rounds": args.max_stage2_rounds,
        "gpus": args.gpus,
        "checkpoint_every_generations": 1,
        "old_output_policy": "read only for exclusion; never used as generation input",
    }
    serialized = json.dumps(resolved, ensure_ascii=False)
    old_parent = "C=Cc1cc2c(cc1-c1cncc(C(N)=O)c1F)CN(C)C2=C"
    if old_parent in serialized or "250k_rndm_zinc" in serialized:
        raise ValueError("Resolved config still contains an old seed source")
    atomic_json(run_dir / "resolved_config.json", resolved)
    return resolved


def record_environment(run_dir: Path) -> None:
    versions = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model_name": MODEL_NAME,
        "model_checkpoint": str(MODEL_PATH),
        "model_checkpoint_sha256": sha256_file(MODEL_PATH),
    }
    for module_name in ["torch", "transformers", "selfies", "sklearn"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["cuda_version"] = torch.version.cuda
        versions["cuda_device_count"] = int(torch.cuda.device_count())
        versions["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        versions["cuda_error"] = str(exc)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        versions["nvidia_smi"] = result.stdout.strip()
    except Exception as exc:
        versions["nvidia_smi"] = str(exc)
    atomic_json(run_dir / "environment_versions.json", versions)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout
    (run_dir / "git_status_initial.txt").write_text(status, encoding="utf-8")


def record_git_diff(run_dir: Path) -> None:
    relative_paths = [
        Path("LRRK2case/optimizer_base_mo.py"),
        Path("LRRK2case/lrrk2_wt_activity_mpo.py"),
        Path("LRRK2case/lrrk2_wt_lead_repair_mpo.py"),
        Path("scripts/run_lrrk2_regeneration.py"),
        Path("scripts/validate_lrrk2_generation.py"),
    ]
    sections = []
    for relative in relative_paths:
        path = ROOT / relative
        if not path.exists():
            continue
        tracked_result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked_result.returncode == 0:
            sections.append(
                subprocess.run(
                    ["git", "diff", "--no-ext-diff", "--", str(relative)],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout
            )
            continue
        file_lines = path.read_text(encoding="utf-8").splitlines(True)
        sections.append(
            f"diff --git a/{relative} b/{relative}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{relative}\n"
            f"@@ -0,0 +1,{len(file_lines)} @@\n"
        )
        sections.extend("+" + line for line in file_lines)
    (run_dir / "git_diff.patch").write_text("".join(sections), encoding="utf-8")
    final_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    (run_dir / "git_status_final.txt").write_text(
        final_status,
        encoding="utf-8",
    )


def old_output_inchikeys(run_dir: Path) -> set:
    cache_path = run_dir / "old_output_inchikeys.txt"
    if cache_path.exists():
        return {line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    patterns = [
        "MultiResult/lrrk2*/**/*all_discovered*.csv",
        "MultiResult/lrrk2*/**/*clean_candidates*.csv",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    paths = sorted(set(path for path in paths if path.is_file()))
    keys = set()
    source_manifest = []
    for path in paths:
        count = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                smiles_column = next(
                    (
                        name for name in ["canonical_smiles", "canonical_SMILES", "smiles", "SMILES"]
                        if name in reader.fieldnames
                    ),
                    None,
                )
                if smiles_column is None:
                    continue
                for row in reader:
                    mol = Chem.MolFromSmiles(str(row.get(smiles_column) or ""))
                    if mol is None:
                        continue
                    key = Chem.MolToInchiKey(mol)
                    if key:
                        keys.add(key)
                        count += 1
        except Exception as exc:
            print(f"Warning: could not index old output {path}: {exc}")
        source_manifest.append({"path": str(path), "parsed_molecules": count})
    cache_path.write_text("\n".join(sorted(keys)) + "\n", encoding="utf-8")
    atomic_json(run_dir / "old_output_exclusion_manifest.json", source_manifest)
    print(f"Indexed {len(keys)} unique structures from {len(paths)} previous LRRK2 output files")
    return keys


def standardize_and_qc(smiles: Any) -> Tuple[Optional[Chem.Mol], str, str, List[str]]:
    text = str(smiles or "").strip()
    if not text:
        return None, "", "", ["empty_smiles"]
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None, "", "", ["rdkit_parse_failed"]
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "", "", ["rdkit_sanitize_failed"]

    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return None, "", "", ["fragment_sanitize_failed"]
    organic_fragments = [
        fragment for fragment in fragments if any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
    ]
    if len(organic_fragments) > 1:
        return None, "", "", ["multiple_organic_components"]
    try:
        mol = LARGEST_FRAGMENT.choose(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "", "", ["largest_fragment_failed"]
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "", "", ["empty_molecule"]
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return None, "", "", ["no_organic_component"]

    reasons = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 0:
            reasons.append("dummy_or_wildcard_atom")
        if atomic_num in METAL_ATOMIC_NUMBERS:
            reasons.append("metal_atom")
        if atomic_num not in ALLOWED_ATOMIC_NUMBERS:
            reasons.append(f"unsupported_element_{atom.GetSymbol()}")
        if atom.GetNumRadicalElectrons():
            reasons.append("radical")
    for bond in mol.GetBonds():
        symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        if symbols & {"F", "Cl", "Br", "I"} and symbols & {"N", "O"}:
            reasons.append("hetero_halogen_bond")
    for atom in mol.GetAtoms():
        double_bonds = [
            bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(double_bonds) >= 2 and atom.IsInRing():
            ring_sizes = [
                len(ring) for ring in mol.GetRingInfo().AtomRings() if atom.GetIdx() in ring
            ]
            if ring_sizes and min(ring_sizes) <= 4:
                reasons.append("small_ring_cumulene")
    for name, query in COMPILED_REACTIVE.items():
        if query is not None and mol.HasSubstructMatch(query):
            reasons.append(name)
    if reasons:
        return None, "", "", sorted(set(reasons))

    try:
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        reparsed = Chem.MolFromSmiles(canonical)
        if reparsed is None:
            return None, "", "", ["canonical_roundtrip_failed"]
        Chem.SanitizeMol(reparsed)
        inchikey = Chem.MolToInchiKey(reparsed)
        if not inchikey:
            return None, "", "", ["inchikey_failed"]
        return reparsed, canonical, inchikey, []
    except Exception:
        return None, "", "", ["final_sanitize_failed"]


def make_stage1_specs(
    work_dir: Path,
    validated: List[Dict[str, Any]],
    allocations: Dict[str, int],
    smoke: bool,
) -> List[Dict[str, Any]]:
    base = read_json(LRRK2_DIR / "config_lrrk2_wt_activity_mpo.json", {})
    specs = []
    for index, seed in enumerate(validated):
        name = seed["name"]
        target = int(allocations[name])
        slug = slugify(name)
        output_dir = work_dir / "batches" / f"stage1_{slug}"
        config = dict(base)
        config.update(
            {
                "seed": MASTER_SEED + 100 + index,
                "max_oracle_calls": target,
                "pop_size": min(24, max(4, target)) if smoke else 120,
                "belief_capacity": min(24, max(4, target)) if smoke else 120,
                "initial_smiles_file": str(work_dir / "inputs" / f"{slug}.csv"),
                "output_dir": str(output_dir),
                "parent_seed": name,
                "generation_batch": f"stage1_{slug}",
                "checkpoint_dir": str(output_dir / "checkpoints"),
                "checkpoint_every_generations": 1,
                "resume_checkpoint": True,
                "export_selected_sdf": False,
                "selected_sdf_generate_3d": False,
                "biot5_model_name": MODEL_NAME,
            }
        )
        config_path = output_dir / "batch_config.json"
        atomic_json(config_path, config)
        specs.append(
            {
                "stage": 1,
                "name": name,
                "slug": slug,
                "target": target,
                "seed": config["seed"],
                "config": config_path,
                "output": output_dir,
                "entry": STAGE1_ENTRY,
                "canonical": output_dir / STAGE1_CANONICAL,
            }
        )
    return specs


def stage2_seed_input_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def make_stage2_specs(
    work_dir: Path,
    validated: List[Dict[str, Any]],
    allocations: Dict[str, int],
    input_paths: Dict[str, Path],
    round_index: int,
    smoke: bool,
    shard_plan: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    base = read_json(LRRK2_DIR / "config_lrrk2_wt_lead_repair_mpo.json", {})
    specs = []
    for index, seed in enumerate(validated):
        name = seed["name"]
        target = int(allocations[name])
        slug = slugify(name)
        input_path = input_paths[name]
        input_count = stage2_seed_input_count(input_path)
        if input_count < 1:
            raise ValueError(f"No stage-1 input molecules for {name}: {input_path}")
        shard_count = max(1, int((shard_plan or {}).get(name, 1)))
        base_target, remainder = divmod(target, shard_count)
        shard_targets = [
            base_target + (1 if shard_index < remainder else 0)
            for shard_index in range(shard_count)
        ]
        for shard_index, shard_target in enumerate(shard_targets):
            shard_suffix = (
                f"_shard{shard_index + 1:02d}of{shard_count:02d}"
                if shard_count > 1
                else ""
            )
            batch_label = f"stage2_round{round_index:03d}_{slug}{shard_suffix}"
            output_dir = work_dir / "batches" / batch_label
            pop_size = min(24, max(4, shard_target, input_count)) if smoke else 120
            config = dict(base)
            config.update(
                {
                    "seed": MASTER_SEED + 10_000 + round_index * 1_000 + index * 100 + shard_index,
                    "max_oracle_calls": shard_target,
                    "pop_size": pop_size,
                    "belief_capacity": min(pop_size, 96),
                    "initial_smiles_file": "",
                    "stage1_seed_file": str(input_path),
                    "stage1_seed_top_k": min(pop_size - 1, input_count),
                    "stage1_seed_sort_by": "aggregated_score",
                    "require_stage1_seed_file": True,
                    "parent_lead_smiles": seed["canonical_isomeric_smiles"],
                    "lead_pool_smiles": [],
                    "initial_population_strategy": "stage1_only",
                    "initial_parent_lead_count": pop_size,
                    "output_dir": str(output_dir),
                    "parent_seed": name,
                    "generation_batch": batch_label,
                    "checkpoint_dir": str(output_dir / "checkpoints"),
                    "checkpoint_every_generations": 1,
                    "resume_checkpoint": True,
                    "export_selected_sdf": False,
                    "selected_sdf_generate_3d": False,
                    "biot5_model_name": MODEL_NAME,
                }
            )
            config_path = output_dir / "batch_config.json"
            atomic_json(config_path, config)
            specs.append(
                {
                    "stage": 2,
                    "name": name,
                    "slug": slug,
                    "target": shard_target,
                    "seed": config["seed"],
                    "config": config_path,
                    "output": output_dir,
                    "entry": STAGE2_ENTRY,
                    "quality": output_dir / STAGE2_QUALITY,
                    "round": round_index,
                    "batch_label": batch_label,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                }
            )
    if not smoke and round_index == 4:
        pf_name = validated[1]["name"]
        slow_spec = next(
            (
                spec
                for spec in specs
                if spec["name"] == pf_name
                and spec.get("shard_count") == 4
                and spec.get("shard_index") == 3
            ),
            None,
        )
        if slow_spec is not None:
            checkpoint_summary = read_json(
                slow_spec["output"] / "checkpoints" / "checkpoint_summary.json",
                {},
            )
            completed_calls = int(checkpoint_summary.get("current_oracle_calls", 0))
            original_target = int(slow_spec["target"])
            if 0 < completed_calls < original_target:
                slow_config = read_json(slow_spec["config"], {})
                slow_config["max_oracle_calls"] = completed_calls
                slow_config["split_recovery_original_target"] = original_target
                atomic_json(slow_spec["config"], slow_config)
                slow_spec["target"] = completed_calls

                remaining = original_target - completed_calls
                supplement_count = 4
                supplement_base, supplement_remainder = divmod(remaining, supplement_count)
                for supplement_index in range(supplement_count):
                    supplement_target = supplement_base + (
                        1 if supplement_index < supplement_remainder else 0
                    )
                    batch_label = (
                        "stage2_round004_pf_06447475_"
                        f"supplement{supplement_index + 1:02d}of{supplement_count:02d}"
                    )
                    output_dir = work_dir / "batches" / batch_label
                    supplement_seed = (
                        MASTER_SEED + 900_000 + round_index * 100 + supplement_index
                    )
                    config = dict(slow_config)
                    config.update(
                        {
                            "seed": supplement_seed,
                            "max_oracle_calls": supplement_target,
                            "output_dir": str(output_dir),
                            "generation_batch": batch_label,
                            "checkpoint_dir": str(output_dir / "checkpoints"),
                            "split_recovery_from": str(slow_spec["output"]),
                        }
                    )
                    config_path = output_dir / "batch_config.json"
                    atomic_json(config_path, config)
                    specs.append(
                        {
                            "stage": 2,
                            "name": pf_name,
                            "slug": slugify(pf_name),
                            "target": supplement_target,
                            "seed": supplement_seed,
                            "config": config_path,
                            "output": output_dir,
                            "entry": STAGE2_ENTRY,
                            "quality": output_dir / STAGE2_QUALITY,
                            "round": round_index,
                            "batch_label": batch_label,
                            "shard_index": supplement_index,
                            "shard_count": supplement_count,
                            "split_recovery": True,
                        }
                    )
    return specs


def append_command(run_dir: Path, command: Sequence[str], env_gpu: str) -> None:
    path = run_dir / "commands.sh"
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(env_gpu)} TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + shlex.join([str(value) for value in command]) + "\n")


def batch_complete(spec: Dict[str, Any], target: int) -> bool:
    metrics_path = spec["output"] / "metrics.json"
    expected = spec.get("canonical") if spec["stage"] == 1 else spec.get("quality")
    if not metrics_path.exists() or not expected or not Path(expected).exists():
        return False
    metrics = read_json(metrics_path, {})
    return int(metrics.get("total_oracle_calls", 0)) >= int(target)


def run_specs(
    specs: List[Dict[str, Any]],
    gpus: List[str],
    run_dir: Path,
    target_override: Optional[Dict[str, int]] = None,
    phase_label: str = "",
) -> None:
    queued = []
    for position, spec in enumerate(specs):
        target = int((target_override or {}).get(spec["name"], spec["target"]))
        if batch_complete(spec, target):
            print(f"[{phase_label}] Reusing complete batch {spec['output']} at {target} calls")
            continue
        queued.append((position, spec, target))

    pending = []
    available_gpus = list(gpus)

    def start_batch(position: int, spec: Dict[str, Any], target: int, gpu: str) -> None:
        spec["output"].mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON_BIN),
            str(spec["entry"]),
            "--config",
            str(spec["config"]),
            "--max_calls",
            str(target),
            "--seed",
            str(spec["seed"]),
            "--output_dir",
            str(spec["output"]),
            "--device",
            "cuda:0",
        ]
        append_command(run_dir, command, gpu)
        log_path = spec["output"] / "batch.log"
        log_handle = log_path.open("ab")
        log_handle.write(
            (f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
             f"{phase_label} target={target} gpu={gpu} =====\n").encode("utf-8")
        )
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTHONHASHSEED": str(spec["seed"]),
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        pending.append((process, spec, target, gpu, log_handle))
        print(
            f"[{phase_label}] Started {spec['name']} stage {spec['stage']} "
            f"target={target} gpu={gpu} pid={process.pid}"
        )

    while queued and available_gpus:
        position, spec, target = queued.pop(0)
        start_batch(position, spec, target, available_gpus.pop(0))

    last_update = 0.0
    while pending or queued:
        remaining = []
        for process, spec, target, gpu, log_handle in pending:
            code = process.poll()
            if code is None:
                remaining.append((process, spec, target, gpu, log_handle))
                continue
            log_handle.close()
            if code != 0:
                tail = ""
                log_path = spec["output"] / "batch.log"
                try:
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                except Exception:
                    pass
                raise RuntimeError(
                    f"Batch failed ({spec['name']} stage {spec['stage']}, exit {code}):\n{tail}"
                )
            if not batch_complete(spec, target):
                metrics = read_json(spec["output"] / "metrics.json", {})
                raise RuntimeError(
                    f"Batch ended without target count: {spec['name']} target={target} metrics={metrics}"
                )
            print(f"[{phase_label}] Completed {spec['name']} stage {spec['stage']} at {target} calls")
            available_gpus.append(gpu)
        pending = remaining
        while queued and available_gpus:
            position, spec, target = queued.pop(0)
            start_batch(position, spec, target, available_gpus.pop(0))
        now = time.time()
        if pending and now - last_update >= 30:
            labels = ", ".join(
                f"{spec['name']}:pid{process.pid}" for process, spec, _, _, _ in pending
            )
            print(f"[{phase_label}] Running: {labels}")
            last_update = now
        if pending or queued:
            time.sleep(5)


def write_stage1_combined(
    work_dir: Path,
    specs: List[Dict[str, Any]],
    validated: List[Dict[str, Any]],
    old_keys: set,
) -> Tuple[Dict[str, Path], Dict[str, Any], List[Dict[str, Any]]]:
    seed_keys = {row["inchikey"] for row in validated}
    raw_rows = []
    best_by_key: Dict[str, Dict[str, Any]] = {}
    rejections = []
    for spec in specs:
        path = Path(spec["canonical"])
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            record = row.to_dict()
            smiles = record.get("smiles") or record.get("canonical_smiles")
            raw_id = f"ST1_RAW_{len(raw_rows) + 1:08d}"
            raw_rows.append(
                {
                    "molecule_id": raw_id,
                    "SMILES": smiles,
                    "parent_seed": spec["name"],
                    "generation_batch": f"stage1_{spec['slug']}",
                    "stage1_score": record.get("aggregated_score", ""),
                    "oracle_call": record.get("oracle_call_best", record.get("oracle_call_first_seen", "")),
                }
            )
            _, canonical, key, reasons = standardize_and_qc(smiles)
            if not reasons and key in seed_keys:
                reasons = ["exact_seed_copy"]
            if not reasons and key in old_keys:
                reasons = ["old_output_overlap"]
            if reasons:
                rejections.append(
                    {
                        "stage": "stage1",
                        "rejected_id": raw_id,
                        "SMILES": smiles,
                        "parent_seed": spec["name"],
                        "generation_batch": f"stage1_{spec['slug']}",
                        "reason": ";".join(reasons),
                    }
                )
                continue
            valid = dict(record)
            valid.update(
                {
                    "molecule_id": raw_id.replace("RAW", "VALID"),
                    "smiles": canonical,
                    "canonical_isomeric_smiles": canonical,
                    "inchikey": key,
                    "parent_seed": spec["name"],
                    "generation_batch": f"stage1_{spec['slug']}",
                    "stage1_score": as_float(record.get("aggregated_score"), 0.0),
                }
            )
            previous = best_by_key.get(key)
            if previous is None or valid["stage1_score"] > previous["stage1_score"]:
                best_by_key[key] = valid

    raw_path = work_dir / "stage1_raw.smi"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["molecule_id", "SMILES", "parent_seed", "generation_batch", "stage1_score", "oracle_call"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    valid_rows = sorted(
        best_by_key.values(),
        key=lambda row: (as_float(row.get("stage1_score"), 0.0), row["inchikey"]),
        reverse=True,
    )
    for index, row in enumerate(valid_rows, start=1):
        row["molecule_id"] = f"ST1_VALID_{index:08d}"
    valid_frame = pd.DataFrame(valid_rows)
    valid_frame.to_csv(work_dir / "stage1_valid.csv", index=False)
    with (work_dir / "stage1_valid.smi").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["molecule_id", "SMILES", "parent_seed"])
        for row in valid_rows:
            writer.writerow([row["molecule_id"], row["canonical_isomeric_smiles"], row["parent_seed"]])

    per_seed = {}
    feedback_dir = work_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    for seed in validated:
        name = seed["name"]
        path = feedback_dir / f"stage1_valid_{slugify(name)}.csv"
        subset = [row for row in valid_rows if row["parent_seed"] == name]
        pd.DataFrame(
            [
                {
                    "smiles": row["canonical_isomeric_smiles"],
                    "aggregated_score": row["stage1_score"],
                    "parent_seed": name,
                }
                for row in subset
            ]
        ).to_csv(path, index=False)
        per_seed[name] = path
        if not subset:
            raise RuntimeError(f"Stage 1 produced no valid unique molecules for {name}")

    stats = {
        "raw_discovered": len(raw_rows),
        "standardized_valid_unique": len(valid_rows),
        "rejected": len(rejections),
        "duplicates_removed": max(0, len(raw_rows) - len(rejections) - len(valid_rows)),
        "raw_by_parent": dict(Counter(row["parent_seed"] for row in raw_rows)),
        "valid_by_parent": dict(Counter(row["parent_seed"] for row in valid_rows)),
    }
    atomic_json(
        work_dir / "checkpoints" / "stage1_manifest.json",
        {
            "complete": True,
            "batch_outputs": [str(spec["output"]) for spec in specs],
            "stage1_raw": str(raw_path),
            "stage1_valid": str(work_dir / "stage1_valid.csv"),
            "statistics": stats,
        },
    )
    return per_seed, stats, rejections


def stage2_sort_key(row: Dict[str, Any]) -> Tuple[float, ...]:
    ad_value = str(row.get("applicability_domain") or row.get("oracle_confidence") or "").lower()
    ad_rank = {"high": 3, "medium": 2, "low": 1, "very_low": 0}.get(ad_value, -1)
    return (
        as_float(row.get("aggregated_score")),
        as_float(row.get("qsar_activity_ad_penalized"), as_float(row.get("activity_score"))),
        as_float(row.get("raw_predicted_pic50")),
        float(ad_rank),
        as_float(row.get("regularizer_score")),
        as_float(row.get("nearest_train_similarity")),
        as_float(row.get("bbb_cns_score")),
        as_float(row.get("logp_window_score")),
        as_float(row.get("tpsa_window_score")),
    )


def combine_stage2(
    work_dir: Path,
    specs: List[Dict[str, Any]],
    validated: List[Dict[str, Any]],
    old_keys: set,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    seed_keys = {row["inchikey"] for row in validated}
    all_rows = []
    rejections = []
    for spec in specs:
        quality_path = Path(spec["quality"])
        frame = pd.read_csv(quality_path)
        for _, series in frame.iterrows():
            row = series.to_dict()
            smiles = row.get("smiles") or row.get("canonical_smiles") or row.get("canonical_SMILES")
            scored_id = f"ST2_{len(all_rows) + 1:09d}"
            _, canonical, key, qc_reasons = standardize_and_qc(smiles)
            reasons = [
                reason
                for reason in str(row.get("clean_candidate_failure_reasons") or "").split(";")
                if reason and reason.lower() != "nan"
            ]
            if not truthy(row.get("clean_candidate_pass")) and not reasons:
                reasons.append("stage2_threshold_failure")
            if qc_reasons:
                reasons.extend(qc_reasons)
            if key and key in seed_keys:
                reasons.append("exact_seed_copy")
            if key and key in old_keys:
                reasons.append("old_output_overlap")
            reasons = sorted(set(reasons))
            row.update(
                {
                    "scored_molecule_id": scored_id,
                    "original_smiles": smiles,
                    "canonical_isomeric_smiles": canonical,
                    "inchikey": key,
                    "parent_seed": spec["name"],
                    "generation_batch": spec.get(
                        "batch_label", f"stage2_round{spec['round']:03d}_{spec['slug']}"
                    ),
                    "generation_round": spec["round"],
                    "stage2_score": as_float(row.get("aggregated_score"), 0.0),
                    "final_qc_failure_reasons": ";".join(reasons),
                    "final_qc_pass": not reasons,
                    "global_duplicate": False,
                    "final_eligible": False,
                }
            )
            all_rows.append(row)

    ordered = sorted(all_rows, key=stage2_sort_key, reverse=True)
    best_by_key = {}
    for row in ordered:
        key = row.get("inchikey")
        if not key or not row.get("final_qc_pass"):
            continue
        if key in best_by_key:
            row["global_duplicate"] = True
            row["final_qc_failure_reasons"] = "global_duplicate"
            row["final_qc_pass"] = False
            continue
        best_by_key[key] = row
        row["final_eligible"] = True

    eligible = list(best_by_key.values())
    eligible.sort(key=stage2_sort_key, reverse=True)
    for row in all_rows:
        if not row.get("final_eligible"):
            rejections.append(
                {
                    "stage": "stage2",
                    "rejected_id": row["scored_molecule_id"],
                    "SMILES": row.get("original_smiles", ""),
                    "parent_seed": row["parent_seed"],
                    "generation_batch": row["generation_batch"],
                    "reason": row.get("final_qc_failure_reasons") or "global_duplicate",
                }
            )

    all_frame = pd.DataFrame(all_rows)
    temporary = work_dir / "stage2_all_scored.csv.tmp"
    all_frame.to_csv(temporary, index=False)
    os.replace(temporary, work_dir / "stage2_all_scored.csv")
    stats = {
        "processed": len(all_rows),
        "stage2_clean_before_global_dedup": sum(
            1 for row in all_rows if truthy(row.get("clean_candidate_pass"))
        ),
        "strict_eligible_unique": len(eligible),
        "global_duplicates": sum(1 for row in all_rows if row.get("global_duplicate")),
        "rejected": len(rejections),
        "processed_by_parent": dict(Counter(row["parent_seed"] for row in all_rows)),
        "eligible_by_parent": dict(Counter(row["parent_seed"] for row in eligible)),
    }
    atomic_json(
        work_dir / "checkpoints" / "stage2_manifest.json",
        {
            "complete": len(eligible) >= FINAL_TARGET,
            "batch_outputs": [str(spec["output"]) for spec in specs],
            "stage2_all_scored": str(work_dir / "stage2_all_scored.csv"),
            "statistics": stats,
        },
    )
    return eligible, stats, rejections


def write_feedback(
    work_dir: Path,
    eligible: List[Dict[str, Any]],
    stage1_inputs: Dict[str, Path],
    validated: List[Dict[str, Any]],
) -> Dict[str, Path]:
    feedback_dir = work_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    by_parent = defaultdict(list)
    for row in eligible:
        by_parent[row["parent_seed"]].append(row)
    result = {}
    for seed in validated:
        name = seed["name"]
        rows = sorted(by_parent[name], key=stage2_sort_key, reverse=True)
        path = feedback_dir / f"stage2_feedback_{slugify(name)}.csv"
        feedback_rows = [
            {
                "smiles": row["canonical_isomeric_smiles"],
                "aggregated_score": row["stage2_score"],
                "parent_seed": name,
            }
            for row in rows[:5000]
        ]
        seen = {row["smiles"] for row in feedback_rows}
        if len(feedback_rows) < 120:
            stage1_frame = pd.read_csv(stage1_inputs[name])
            for record in stage1_frame.to_dict("records"):
                smiles = str(record.get("smiles") or "")
                if not smiles or smiles in seen:
                    continue
                feedback_rows.append(
                    {
                        "smiles": smiles,
                        "aggregated_score": as_float(record.get("aggregated_score"), 0.0),
                        "parent_seed": name,
                    }
                )
                seen.add(smiles)
                if len(feedback_rows) >= 120:
                    break
        if feedback_rows:
            pd.DataFrame(feedback_rows[:5000]).to_csv(path, index=False)
            result[name] = path
        else:
            result[name] = stage1_inputs[name]
    return result


def sa_score(mol: Chem.Mol) -> float:
    try:
        try:
            from rdkit.Contrib.SA_Score import sascorer
        except Exception:
            sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
            if str(sa_dir) not in sys.path:
                sys.path.append(str(sa_dir))
            import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


def build_alert_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


def stereochemistry_status(mol: Chem.Mol) -> str:
    centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False
    )
    assigned = sum(1 for _, label in centers if label != "?")
    unassigned = sum(1 for _, label in centers if label == "?")
    if unassigned:
        return f"unassigned_centers:{unassigned};assigned_centers:{assigned}"
    if assigned:
        return f"specified_centers:{assigned}"
    return "no_tetrahedral_stereocenters"


def final_properties(
    selected: List[Dict[str, Any]], validated: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    seed_fps = [
        MORGAN_GENERATOR.GetFingerprint(Chem.MolFromSmiles(seed["canonical_isomeric_smiles"]))
        for seed in validated
    ]
    alert_catalog = build_alert_catalog()
    result = []
    for rank, source in enumerate(selected, start=1):
        mol = Chem.MolFromSmiles(source["canonical_isomeric_smiles"])
        Chem.SanitizeMol(mol)
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, seed_fps)
        alerts = [match.GetDescription() for match in alert_catalog.GetMatches(mol)]
        existing_alerts = [
            value for value in str(source.get("structure_risk_flags") or "").split(";")
            if value and value.lower() != "nan"
        ]
        alerts = sorted(set(existing_alerts + [f"PAINS:{value}" for value in alerts]))
        qed_value = float(QED.qed(mol))
        activity = as_float(
            source.get("qsar_activity_ad_penalized"),
            as_float(source.get("activity_score"), 0.0),
        )
        row = {
            "molecule_id": f"LRRK2_{rank:05d}",
            "canonical_isomeric_smiles": source["canonical_isomeric_smiles"],
            "inchikey": source["inchikey"],
            "parent_seed": source["parent_seed"],
            "generation_batch": source["generation_batch"],
            "generation_round": int(source.get("generation_round", 0)),
            "stage1_score": geometric_mean([activity, qed_value]),
            "stage2_score": as_float(source.get("stage2_score"), 0.0),
            "LRRK2_qsar_activity_ad_penalized": activity,
            "LRRK2_raw_predicted_pIC50": as_float(source.get("raw_predicted_pic50"), float("nan")),
            "LRRK2_applicability_domain": source.get("applicability_domain", ""),
            "LRRK2_nearest_train_similarity": as_float(source.get("nearest_train_similarity"), float("nan")),
            "bbb_cns_score": as_float(source.get("bbb_cns_score"), 0.0),
            "logp_window_score": as_float(source.get("logp_window_score"), 0.0),
            "tpsa_window_score": as_float(source.get("tpsa_window_score"), 0.0),
            "MW": float(Descriptors.MolWt(mol)),
            "cLogP": float(Crippen.MolLogP(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "ring_count": int(Lipinski.RingCount(mol)),
            "QED": qed_value,
            "SA_score": sa_score(mol),
            "structural_alerts": ";".join(alerts),
            "stereochemistry_status": stereochemistry_status(mol),
            "max_similarity_to_any_seed": max(similarities),
            "selection_rank": rank,
            "source_scored_molecule_id": source["scored_molecule_id"],
        }
        result.append(row)
        if rank % 2000 == 0:
            print(f"Computed final properties for {rank}/{len(selected)} molecules")
    return result


def write_final_exports(run_dir: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = run_dir / "lrrk2_final_20000.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    smi_path = run_dir / "lrrk2_final_20000.smi"
    with smi_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["molecule_id", "SMILES", "parent_seed"])
        for row in rows:
            writer.writerow([row["molecule_id"], row["canonical_isomeric_smiles"], row["parent_seed"]])

    sdf_path = run_dir / "lrrk2_final_20000.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    try:
        for index, row in enumerate(rows, start=1):
            mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", row["molecule_id"])
            for key, value in row.items():
                if value is None:
                    continue
                mol.SetProp(str(key), str(value))
            writer.write(mol)
            if index % 2000 == 0:
                print(f"Wrote SDF {index}/{len(rows)}")
    finally:
        writer.close()


def distribution(values: Iterable[Any]) -> Dict[str, float]:
    finite_values = []
    for value in values:
        try:
            numeric = float(value)
            if np.isfinite(numeric):
                finite_values.append(numeric)
        except Exception:
            continue
    array = np.array(finite_values, dtype=float)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def collect_generation_sources(specs: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = Counter()
    for spec in specs:
        metrics = read_json(spec["output"] / "metrics.json", {})
        for key, value in metrics.get("generation_source_totals", {}).items():
            totals[key] += int(value or 0)
    return dict(totals)


def write_rejections(run_dir: Path, rejections: List[Dict[str, Any]]) -> Dict[str, int]:
    reason_counter = Counter()
    for row in rejections:
        reasons = [reason for reason in str(row.get("reason") or "").split(";") if reason]
        for reason in reasons:
            reason_counter[reason] += 1
    expanded = []
    for row in rejections:
        reasons = [reason for reason in str(row.get("reason") or "").split(";") if reason]
        for reason in reasons or ["unknown"]:
            item = dict(row)
            item["reason"] = reason
            item["reason_total_count"] = reason_counter[reason]
            expanded.append(item)
    pd.DataFrame(
        expanded,
        columns=[
            "stage", "rejected_id", "SMILES", "parent_seed",
            "generation_batch", "reason", "reason_total_count",
        ],
    ).to_csv(run_dir / "rejection_reasons.csv", index=False)
    return dict(reason_counter)


def format_distribution_table(distributions: Dict[str, Dict[str, float]]) -> str:
    headers = ["Property", "Min", "P05", "Median", "Mean", "P95", "Max"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, stats in distributions.items():
        lines.append(
            "| " + " | ".join(
                [
                    name,
                    f"{stats.get('min', float('nan')):.3f}",
                    f"{stats.get('p05', float('nan')):.3f}",
                    f"{stats.get('median', float('nan')):.3f}",
                    f"{stats.get('mean', float('nan')):.3f}",
                    f"{stats.get('p95', float('nan')):.3f}",
                    f"{stats.get('max', float('nan')):.3f}",
                ]
            ) + " |"
        )
    return "\n".join(lines)


def write_quality_report(
    run_dir: Path,
    validated: List[Dict[str, Any]],
    stage1_stats: Dict[str, Any],
    stage2_stats: Dict[str, Any],
    final_rows: List[Dict[str, Any]],
    rejection_counts: Dict[str, int],
    stage1_specs: List[Dict[str, Any]],
    stage2_specs: List[Dict[str, Any]],
) -> None:
    distributions = {
        key: distribution(row[key] for row in final_rows)
        for key in [
            "MW", "cLogP", "TPSA", "HBD", "HBA", "rotatable_bonds",
            "formal_charge", "max_similarity_to_any_seed", "stage1_score", "stage2_score",
        ]
    }
    final_by_parent = Counter(row["parent_seed"] for row in final_rows)
    alert_counts = Counter()
    for row in final_rows:
        for alert in str(row["structural_alerts"]).split(";"):
            if alert:
                alert_counts[alert] += 1
    unspecified = sum(
        1 for row in final_rows if row["stereochemistry_status"].startswith("unassigned_centers")
    )
    soft_interval_counts = {
        "MW_250_480": sum(250 <= row["MW"] <= 480 for row in final_rows),
        "cLogP_1_4.5": sum(1.0 <= row["cLogP"] <= 4.5 for row in final_rows),
        "TPSA_35_110": sum(35 <= row["TPSA"] <= 110 for row in final_rows),
        "HBD_0_3": sum(0 <= row["HBD"] <= 3 for row in final_rows),
        "RotB_0_8": sum(0 <= row["rotatable_bonds"] <= 8 for row in final_rows),
        "charge_minus1_0_plus1": sum(row["formal_charge"] in {-1, 0, 1} for row in final_rows),
    }
    previous_stage1 = read_json(ROOT / "MultiResult/lrrk2_wt_activity_mpo_results/metrics.json", {})
    previous_stage2 = read_json(
        ROOT / "MultiResult/lrrk2_wt_lead_repair_mpo_results_filtered_10000/metrics.json", {}
    )
    previous_clean = read_json(
        ROOT / "MultiResult/lrrk2_wt_lead_repair_mpo_results_filtered_10000"
        / "lrrk2_wt_lead_repair_mpo_clean_candidate_summary.json",
        {},
    )
    commands = (run_dir / "commands.sh").read_text(encoding="utf-8") if (run_dir / "commands.sh").exists() else ""

    lines = [
        "# LRRK2 Five-Seed Regeneration Quality Report",
        "",
        "## Seed Validation",
        "",
        "| Seed | Weight | Canonical isomeric SMILES | InChIKey | Specified chiral centers |",
        "|---|---:|---|---|---:|",
    ]
    for seed in validated:
        lines.append(
            f"| {seed['name']} | {seed['weight']:.2f} | {seed['canonical_isomeric_smiles']} "
            f"| {seed['inchikey']} | {seed['specified_chiral_centers']} |"
        )
    lines.extend(
        [
            "",
            f"All five inputs parsed and sanitized with RDKit; weight sum = {sum(s['weight'] for s in validated):.2f}. "
            "Canonical round-trip connectivity passed and specified stereochemistry was preserved.",
            "",
            "## Workflow",
            "",
            f"- Stage 1 entry: {STAGE1_ENTRY}",
            f"- Stage 2 entry: {STAGE2_ENTRY}",
            f"- Generation model: {MODEL_NAME}",
            f"- QSAR checkpoint: {MODEL_PATH} (SHA256 {sha256_file(MODEL_PATH)})",
            "- Stage 1 objectives: AD-aware LRRK2 activity and QED regularizer.",
            "- Stage 2 objectives: AD-penalized LRRK2 activity, BBB/CNS, logP window, TPSA window.",
            "- Existing stage-2 clean thresholds and structural block rules were not lowered.",
            "",
            "### Commands",
            "",
        ]
    )
    lines.extend("    " + command for command in commands.rstrip().splitlines())
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Stage 1 raw discovered: {stage1_stats['raw_discovered']:,}",
            f"- Stage 1 RDKit/QC valid unique: {stage1_stats['standardized_valid_unique']:,}",
            f"- Stage 1 duplicates removed: {stage1_stats['duplicates_removed']:,}",
            f"- Stage 2 processed/scored: {stage2_stats['processed']:,}",
            f"- Stage 2 clean before global InChIKey dedup: {stage2_stats['stage2_clean_before_global_dedup']:,}",
            f"- Stage 2 global duplicates: {stage2_stats['global_duplicates']:,}",
            f"- Stage 2 strict eligible unique: {stage2_stats['strict_eligible_unique']:,}",
            f"- Final score-ranked unique molecules: {len(final_rows):,}",
            "",
            "## Parent Contributions",
            "",
            "| Parent seed | Stage 1 raw | Stage 1 valid unique | Stage 2 processed | Final |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for seed in validated:
        name = seed["name"]
        lines.append(
            f"| {name} | {stage1_stats['raw_by_parent'].get(name, 0):,} "
            f"| {stage1_stats['valid_by_parent'].get(name, 0):,} "
            f"| {stage2_stats['processed_by_parent'].get(name, 0):,} "
            f"| {final_by_parent.get(name, 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Rejections",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count:,} |")
    lines.extend(
        [
            "",
            "## Property Distributions",
            "",
            format_distribution_table(distributions),
            "",
            "Soft medicinal-chemistry interval coverage:",
            "",
        ]
    )
    for name, count in soft_interval_counts.items():
        lines.append(f"- {name}: {count:,}/{len(final_rows):,} ({count / len(final_rows):.2%})")
    lines.extend(
        [
            "",
            "## Similarity, Stereo, And Alerts",
            "",
            "- Exact copies of the five input seeds: 0",
            f"- Molecules with at least one unassigned tetrahedral center: {unspecified:,}",
            "- Maximum ECFP4-to-seed similarity distribution is included above.",
            "",
            "| Structural alert | Count |",
            "|---|---:|",
        ]
    )
    if alert_counts:
        for alert, count in alert_counts.most_common():
            lines.append(f"| {alert} | {count:,} |")
    else:
        lines.append("| None | 0 |")
    lines.extend(
        [
            "",
            "## Generation Sources",
            "",
            f"- Stage 1: {json.dumps(collect_generation_sources(stage1_specs), sort_keys=True)}",
            f"- Stage 2: {json.dumps(collect_generation_sources(stage2_specs), sort_keys=True)}",
            "",
            "## Previous LRRK2 Comparison",
            "",
            f"- Previous stage 1: {int(previous_stage1.get('total_oracle_calls', 0)):,} unique scored, "
            f"runtime {as_float(previous_stage1.get('runtime_seconds'), 0.0):.1f} s.",
            f"- Previous stage 2: {int(previous_stage2.get('total_oracle_calls', 0)):,} unique scored; "
            f"{int(previous_clean.get('clean_candidates', 0)):,} passed clean thresholds.",
            f"- Current stage 2: {stage2_stats['processed']:,} scored; "
            f"{stage2_stats['strict_eligible_unique']:,} strict unique candidates before final top-20,000 selection.",
            "- Previous files were read only to report this comparison and to exclude structure overlap.",
            "",
            "## Top 100",
            "",
            "| Rank | Molecule ID | Parent | Stage 2 | LRRK2 score | MW | cLogP | TPSA | Max seed similarity | SMILES |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in final_rows[:100]:
        lines.append(
            f"| {row['selection_rank']} | {row['molecule_id']} | {row['parent_seed']} "
            f"| {row['stage2_score']:.4f} | {row['LRRK2_qsar_activity_ad_penalized']:.4f} "
            f"| {row['MW']:.1f} | {row['cLogP']:.2f} | {row['TPSA']:.1f} "
            f"| {row['max_similarity_to_any_seed']:.3f} | {row['canonical_isomeric_smiles']} |"
        )
    lines.extend(
        [
            "",
            "## Selection And Recovery",
            "",
            "- Final molecules were sorted by the existing stage-2 balanced ranking tuple; file order and generation time were not used.",
            "- Each stage batch has an atomic generation checkpoint and checkpoint summary.",
            "- Completed batches were never restarted; incremental stage-2 rounds used prior strict candidates as inputs.",
            "- One incomplete, unmanifested 107-call checkpoint from the original round-2 compound18 layout "
            "was preserved at batches/stage2_round002_lrrk2_compound18_yxd but excluded from all official "
            "stage-2 counts and outputs after fixing the undersized feedback pool.",
            "- The stage-2 shard layout and bounded GPU scheduler were adjusted to reduce runtime skew. "
            "These changes only partition each weighted per-parent call budget; model checkpoints, scoring "
            "objectives, clean thresholds, and global selection remained unchanged.",
            "- The unfinished long-tail PF-06447475 round-4 shard was frozen at its atomic checkpoint; "
            "its remaining calls were partitioned across four supplemental PF shards, preserving the "
            "original per-parent allocation and exact 70,000-call round budget.",
            "- Full validation results are appended below by the acceptance script.",
            "",
        ]
    )
    (run_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_run(
    run_dir: Path,
    validated: List[Dict[str, Any]],
    eligible: List[Dict[str, Any]],
    stage1_stats: Dict[str, Any],
    stage2_stats: Dict[str, Any],
    stage1_rejections: List[Dict[str, Any]],
    stage2_rejections: List[Dict[str, Any]],
    stage1_specs: List[Dict[str, Any]],
    stage2_specs: List[Dict[str, Any]],
) -> None:
    if len(eligible) < FINAL_TARGET:
        raise RuntimeError(f"Cannot finalize: only {len(eligible)} strict unique molecules")
    selected = sorted(eligible, key=stage2_sort_key, reverse=True)[:FINAL_TARGET]
    final_rows = final_properties(selected, validated)
    if len(final_rows) != FINAL_TARGET:
        raise RuntimeError(f"Final property count is {len(final_rows)}, expected {FINAL_TARGET}")
    write_final_exports(run_dir, final_rows)
    rejection_counts = write_rejections(
        run_dir, stage1_rejections + stage2_rejections
    )

    statistics = {
        "random_seed": MASTER_SEED,
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "final_count": len(final_rows),
        "final_unique_inchikeys": len({row["inchikey"] for row in final_rows}),
        "final_by_parent": dict(Counter(row["parent_seed"] for row in final_rows)),
        "rejection_reason_counts": rejection_counts,
        "stage1_generation_sources": collect_generation_sources(stage1_specs),
        "stage2_generation_sources": collect_generation_sources(stage2_specs),
        "selection": "existing stage2 clean-candidate balanced sort tuple, descending",
    }
    atomic_json(run_dir / "generation_statistics.json", statistics)
    write_quality_report(
        run_dir,
        validated,
        stage1_stats,
        stage2_stats,
        final_rows,
        rejection_counts,
        stage1_specs,
        stage2_specs,
    )
    record_git_diff(run_dir)


def copy_seed_inputs(source_run: Path, target: Path) -> None:
    target_inputs = target / "inputs"
    target_inputs.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        source = source_run / "inputs" / f"{slugify(seed['name'])}.csv"
        target_path = target_inputs / source.name
        target_path.write_bytes(source.read_bytes())


def run_smoke(
    run_dir: Path,
    validated: List[Dict[str, Any]],
    gpus: List[str],
    old_keys: set,
) -> None:
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    copy_seed_inputs(run_dir, smoke_dir)
    allocations = sample_allocations(150, MASTER_SEED + 1, guarantee_all=True)
    stage1_specs = make_stage1_specs(smoke_dir, validated, allocations, smoke=True)
    half = {name: max(1, count // 2) for name, count in allocations.items()}
    run_specs(stage1_specs, gpus, run_dir, target_override=half, phase_label="smoke-stage1-partial")
    run_specs(stage1_specs, gpus, run_dir, phase_label="smoke-stage1-resume")
    stage1_inputs, stage1_stats, stage1_rejections = write_stage1_combined(
        smoke_dir, stage1_specs, validated, old_keys
    )

    stage2_allocations = sample_allocations(150, MASTER_SEED + 2, guarantee_all=True)
    stage2_specs = make_stage2_specs(
        smoke_dir, validated, stage2_allocations, stage1_inputs, 0, smoke=True
    )
    half_stage2 = {name: max(1, count // 2) for name, count in stage2_allocations.items()}
    run_specs(stage2_specs, gpus, run_dir, target_override=half_stage2, phase_label="smoke-stage2-partial")
    run_specs(stage2_specs, gpus, run_dir, phase_label="smoke-stage2-resume")
    eligible, stage2_stats, stage2_rejections = combine_stage2(
        smoke_dir, stage2_specs, validated, old_keys
    )

    frame = pd.read_csv(smoke_dir / "stage2_all_scored.csv")
    all_parent_names = set(frame["parent_seed"].dropna().astype(str))
    checkpoint_summaries = list(smoke_dir.glob("batches/**/checkpoint_summary.json"))
    resumed_logs = []
    for path in smoke_dir.glob("batches/**/batch.log"):
        if "Resumed checkpoint:" in path.read_text(encoding="utf-8", errors="replace"):
            resumed_logs.append(str(path))
    checks = {
        "all_five_seeds_loaded": all_parent_names == {seed["name"] for seed in validated},
        "weights_correct": math.isclose(sum(seed["weight"] for seed in validated), 1.0),
        "stage1_completed": stage1_stats["raw_discovered"] > 0,
        "stage1_to_stage2": stage2_stats["processed"] > 0,
        "stage2_completed": all(batch_complete(spec, spec["target"]) for spec in stage2_specs),
        "rdkit_outputs_valid": not any(
            standardize_and_qc(smiles)[3]
            for smiles in frame["canonical_isomeric_smiles"].dropna().astype(str).head(100)
        ),
        "lineage_present": not frame["parent_seed"].isna().any(),
        "checkpoint_files_present": len(checkpoint_summaries) == 10,
        "resume_observed": len(resumed_logs) == 10,
        "no_old_input_path": all(
            str(spec["config"]).startswith(str(smoke_dir)) for spec in stage1_specs + stage2_specs
        ),
        "required_stage2_fields": {
            "parent_seed", "generation_batch", "aggregated_score",
            "qsar_activity_ad_penalized", "clean_candidate_pass",
        }.issubset(frame.columns),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Smoke test failed: {checks}")
    atomic_json(
        smoke_dir / "smoke_test_report.json",
        {
            "passed": True,
            "checks": checks,
            "stage1_allocations": allocations,
            "stage2_allocations": stage2_allocations,
            "stage1_statistics": stage1_stats,
            "stage2_statistics": stage2_stats,
            "strict_eligible_unique": len(eligible),
            "stage1_rejections": len(stage1_rejections),
            "stage2_rejections": len(stage2_rejections),
            "resumed_logs": resumed_logs,
        },
    )
    print(f"Smoke test passed: {smoke_dir / 'smoke_test_report.json'}")


def load_existing_stage2_specs(manifest_path: Path) -> List[Dict[str, Any]]:
    data = read_json(manifest_path, [])
    result = []
    for row in data:
        row["config"] = Path(row["config"])
        row["output"] = Path(row["output"])
        row["entry"] = Path(row["entry"])
        row["quality"] = Path(row["quality"])
        result.append(row)
    return result


def save_stage2_specs(manifest_path: Path, specs: List[Dict[str, Any]]) -> None:
    serializable = []
    for spec in specs:
        row = dict(spec)
        for key in ["config", "output", "entry", "quality"]:
            row[key] = str(row[key])
        serializable.append(row)
    atomic_json(manifest_path, serializable)


def run_full(
    run_dir: Path,
    validated: List[Dict[str, Any]],
    args: argparse.Namespace,
    gpus: List[str],
    old_keys: set,
) -> None:
    allocations = sample_allocations(20_000, MASTER_SEED, guarantee_all=True)
    write_resolved_config(run_dir, validated, allocations, args)
    stage1_specs = make_stage1_specs(run_dir, validated, allocations, smoke=False)
    run_specs(stage1_specs, gpus, run_dir, phase_label="full-stage1")
    stage1_inputs, stage1_stats, stage1_rejections = write_stage1_combined(
        run_dir, stage1_specs, validated, old_keys
    )
    print(f"Stage 1 complete: raw={stage1_stats['raw_discovered']} valid_unique={stage1_stats['standardized_valid_unique']}")

    manifest_path = run_dir / "checkpoints" / "stage2_batch_specs.json"
    stage2_specs = load_existing_stage2_specs(manifest_path)
    if stage2_specs:
        eligible, stage2_stats, stage2_rejections = combine_stage2(
            run_dir, stage2_specs, validated, old_keys
        )
    else:
        eligible, stage2_stats, stage2_rejections = [], {
            "processed": 0,
            "stage2_clean_before_global_dedup": 0,
            "strict_eligible_unique": 0,
            "global_duplicates": 0,
            "rejected": 0,
            "processed_by_parent": {},
            "eligible_by_parent": {},
        }, []

    next_round = 0 if not stage2_specs else max(spec["round"] for spec in stage2_specs) + 1
    feedback = write_feedback(run_dir, eligible, stage1_inputs, validated)
    fine_shard_plan = (
        {
            validated[0]["name"]: 4,
            validated[1]["name"]: 4,
            validated[2]["name"]: 4,
            validated[3]["name"]: 3,
            validated[4]["name"]: 8,
        }
        if len(gpus) >= 8
        else None
    )
    legacy_shard_plan = (
        {
            validated[0]["name"]: 2,
            validated[1]["name"]: 2,
            validated[2]["name"]: 2,
            validated[4]["name"]: 2,
        }
        if len(gpus) >= 8
        else None
    )
    while len(eligible) < FINAL_TARGET:
        if next_round >= args.max_stage2_rounds:
            raise RuntimeError(
                f"Stage 2 reached max rounds ({args.max_stage2_rounds}) with {len(eligible)} eligible"
            )
        legacy_round3_resume = (
            next_round == 3
            and (
                run_dir / "batches"
                / "stage2_round003_dnl201_gne_0877_shard01of02"
            ).exists()
        )
        round_budget = 50_000 if legacy_round3_resume else args.stage2_round_budget
        round_shard_plan = legacy_shard_plan if legacy_round3_resume else fine_shard_plan
        round_allocations = sample_allocations(
            round_budget,
            MASTER_SEED + 20_000 + next_round,
            guarantee_all=True,
        )
        new_specs = make_stage2_specs(
            run_dir,
            validated,
            round_allocations,
            feedback,
            next_round,
            smoke=False,
            shard_plan=round_shard_plan,
        )
        run_specs(new_specs, gpus, run_dir, phase_label=f"full-stage2-round{next_round:03d}")
        stage2_specs.extend(new_specs)
        save_stage2_specs(manifest_path, stage2_specs)
        eligible, stage2_stats, stage2_rejections = combine_stage2(
            run_dir, stage2_specs, validated, old_keys
        )
        print(
            f"Stage 2 round {next_round} complete: processed={stage2_stats['processed']} "
            f"clean_pre_dedup={stage2_stats['stage2_clean_before_global_dedup']} "
            f"strict_unique={len(eligible)}/{FINAL_TARGET}"
        )
        feedback = write_feedback(run_dir, eligible, stage1_inputs, validated)
        next_round += 1

    finalize_run(
        run_dir,
        validated,
        eligible,
        stage1_stats,
        stage2_stats,
        stage1_rejections,
        stage2_rejections,
        stage1_specs,
        stage2_specs,
    )
    print(f"Full run finalized with exactly {FINAL_TARGET} molecules")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / "runs" / "lrrk2_regeneration_5seeds_20000"),
    )
    parser.add_argument("--mode", choices=["init", "smoke", "full", "all"], default="all")
    parser.add_argument("--gpus", default="1,2,3,4,5")
    parser.add_argument("--stage2-round-budget", type=int, default=50_000)
    parser.add_argument("--max-stage2-rounds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    setup_run_log(run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Master random seed: {MASTER_SEED}")
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Required callm Python environment not found: {PYTHON_BIN}")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be provided")

    validated = validate_and_write_seeds(run_dir)
    allocations = sample_allocations(20_000, MASTER_SEED, guarantee_all=True)
    write_resolved_config(run_dir, validated, allocations, args)
    environment_path = run_dir / "environment_versions.json"
    if environment_path.exists():
        print(f"Reusing recorded environment: {environment_path}")
    else:
        record_environment(run_dir)
    old_keys = old_output_inchikeys(run_dir)
    record_git_diff(run_dir)

    if args.mode == "init":
        print("Initialization complete")
        return 0
    if args.mode in {"smoke", "all"}:
        run_smoke(run_dir, validated, gpus, old_keys)
    if args.mode in {"full", "all"}:
        smoke_report = read_json(run_dir / "smoke" / "smoke_test_report.json", {})
        if not smoke_report.get("passed"):
            raise RuntimeError("Full run requires a passing smoke test")
        run_full(run_dir, validated, args, gpus, old_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

