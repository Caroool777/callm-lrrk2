#!/usr/bin/env python3
"""Resumable eight-seed LRRK2 WT Stage-1 -> Stage-2 production workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
LRRK2_DIR = ROOT / "LRRK2case"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(LRRK2_DIR) not in sys.path:
    sys.path.insert(0, str(LRRK2_DIR))

import run_lrrk2_regeneration as legacy  # noqa: E402
from lrrk2_qsar_mpo_utils import compute_property_panel  # noqa: E402

rdBase.DisableLog("rdApp.warning")

PYTHON_BIN = Path(sys.executable)
SEED_CONFIG = ROOT / "configs" / "lrrk2_stage1_group_seeds.yaml"
STAGE2_CONFIG = ROOT / "configs" / "lrrk2_stage2_20k.yaml"
STAGE1_BASE = LRRK2_DIR / "config_lrrk2_wt_activity_mpo.json"
STAGE2_BASE = LRRK2_DIR / "config_lrrk2_wt_lead_repair_mpo.json"
STAGE1_ENTRY = LRRK2_DIR / "lrrk2_wt_activity_mpo.py"
STAGE2_ENTRY = LRRK2_DIR / "lrrk2_wt_lead_repair_mpo.py"
ORACLE_DIR = ROOT / "lrrk2_qsar_oracle"
MODEL_NAME = "QizhiPei/biot5-base-text2mol"
STREAM_SEEDS = [41, 42, 43, 44, 45]
FINAL_TARGET = 20_000
STAGE1_HANDOFF = 50
STAGE1_CANONICAL = "lrrk2_wt_activity_mpo_all_discovered_canonical.csv"
STAGE2_QUALITY = "lrrk2_wt_lead_repair_mpo_structure_quality.csv"
STAGE1_PARETO = "lrrk2_wt_activity_mpo_pareto_front.csv"
STAGE2_PARETO = "lrrk2_wt_lead_repair_mpo_pareto_front.csv"
SOURCE_LABEL = "FIXED_USER_VERIFIED_LRRK2_SEED_SET"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_seed_config() -> List[Dict[str, Any]]:
    payload = yaml.safe_load(SEED_CONFIG.read_text(encoding="utf-8"))
    seeds = list(payload.get("seeds", []))
    if len(seeds) != 8:
        raise ValueError(f"Expected exactly 8 fixed seeds, found {len(seeds)}")
    return seeds


def raw_sa_score(mol: Chem.Mol) -> float:
    try:
        try:
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        except Exception:
            sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
            if str(sa_dir) not in sys.path:
                sys.path.append(str(sa_dir))
            import sascorer  # type: ignore
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")


def canonical_isomeric(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def validate_and_write_seeds(run_dir: Path) -> List[Dict[str, Any]]:
    import selfies as sf

    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seed_config()
    ids = [str(row["seed_id"]) for row in seeds]
    if len(set(ids)) != 8:
        raise ValueError(f"Seed IDs are not unique: {ids}")
    weights = [float(row["weight"]) for row in seeds]
    if not all(np.isfinite(value) and value > 0 for value in weights):
        raise ValueError(f"Seed weights must be finite and positive: {weights}")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Seed weights sum to {sum(weights):.16f}, expected 1.0")

    validated: List[Dict[str, Any]] = []
    seen_canonical = set()
    for source in seeds:
        seed_id = str(source["seed_id"])
        input_smiles = str(source["input_smiles"])
        parse_status = "PASS"
        sanitize_status = "PASS"
        try:
            mol = Chem.MolFromSmiles(input_smiles)
            if mol is None:
                raise ValueError("MolFromSmiles returned None")
        except Exception as exc:
            raise ValueError(f"RDKit parse failed for {seed_id}: {input_smiles}: {exc}") from exc
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(f"RDKit sanitize failed for {seed_id}: {input_smiles}: {exc}") from exc

        canonical = canonical_isomeric(mol)
        reparsed = Chem.MolFromSmiles(canonical)
        if reparsed is None:
            raise ValueError(f"Canonical SMILES reparsing failed for {seed_id}: {canonical}")
        Chem.SanitizeMol(reparsed)
        if canonical_isomeric(reparsed) != canonical:
            raise ValueError(f"Canonical molecular graph changed for {seed_id}")
        if canonical in seen_canonical:
            raise ValueError(f"Two fixed seeds collapse to the same structure: {seed_id}")
        seen_canonical.add(canonical)

        selfies_roundtrip_status = "PASS"
        graph_equivalence_status = "PASS"
        try:
            encoded = sf.encoder(canonical)
            decoded = sf.decoder(encoded)
            decoded_mol = Chem.MolFromSmiles(decoded)
            if decoded_mol is None:
                raise ValueError("SELFIES-decoded SMILES did not parse")
            Chem.SanitizeMol(decoded_mol)
            decoded_canonical = canonical_isomeric(decoded_mol)
            if decoded_canonical != canonical:
                graph_equivalence_status = "FAIL"
                raise ValueError(
                    f"SELFIES graph mismatch: before={canonical}, after={decoded_canonical}"
                )
        except Exception as exc:
            selfies_roundtrip_status = "FAIL"
            raise ValueError(f"SELFIES round-trip failed for {seed_id}: {exc}") from exc

        if seed_id == "LY2023-001" and "/C=N/" not in canonical:
            raise ValueError(
                "LY2023-001 E/Z annotation was not preserved in canonical isomeric SMILES: "
                + canonical
            )

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        record = {
            "seed_id": seed_id,
            "input_smiles": input_smiles,
            "canonical_smiles": canonical,
            "inchikey": Chem.MolToInchiKey(mol),
            "weight": float(source["weight"]),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "MW": float(Descriptors.MolWt(mol)),
            "cLogP": float(Crippen.MolLogP(mol)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "QED": float(QED.qed(mol)),
            "Murcko_scaffold": canonical_isomeric(scaffold_mol) if scaffold_mol.GetNumAtoms() else "",
            "rdkit_parse_status": parse_status,
            "rdkit_sanitize_status": sanitize_status,
            "selfies_roundtrip_status": selfies_roundtrip_status,
            "graph_equivalence_status": graph_equivalence_status,
            "validation_status": "PASS",
            "source": SOURCE_LABEL,
        }
        validated.append(record)

        family_path = inputs_dir / f"{legacy.slugify(seed_id)}.csv"
        atomic_csv(
            pd.DataFrame([{"smiles": canonical, "seed_family": seed_id, "weight": source["weight"]}]),
            family_path,
        )

    if len(validated) != 8 or len(seen_canonical) != 8:
        raise RuntimeError("Seed integrity check did not produce exactly 8 unique valid structures")
    atomic_csv(pd.DataFrame(validated), inputs_dir / "lrrk2_stage1_seed_pool.csv")
    atomic_text(
        inputs_dir / "lrrk2_stage1_seed_pool.smi",
        "".join(f"{row['canonical_smiles']}\t{row['seed_id']}\n" for row in validated),
    )
    legacy.atomic_json(
        inputs_dir / "seed_validation.json",
        {
            "passed": True,
            "seed_count": 8,
            "unique_canonical_count": 8,
            "weight_sum": sum(weights),
            "source": SOURCE_LABEL,
            "records": validated,
        },
    )
    return validated


def multinomial_allocations(total: int, stream_seed: int, validated: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    rng = np.random.default_rng(stream_seed)
    weights = np.array([float(row["weight"]) for row in validated], dtype=float)
    counts = rng.multinomial(int(total), weights)
    if total >= len(validated):
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return {row["seed_id"]: int(count) for row, count in zip(validated, counts)}


def make_stage1_specs(
    work_dir: Path,
    validated: Sequence[Dict[str, Any]],
    budget_per_stream: int,
    smoke: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    base = legacy.read_json(STAGE1_BASE, {})
    specs: List[Dict[str, Any]] = []
    allocation_manifest: Dict[str, Dict[str, int]] = {}
    streams = [41] if smoke else STREAM_SEEDS
    for stream_seed in streams:
        if smoke:
            allocations = {row["seed_id"]: max(4, int(round(budget_per_stream * row["weight"]))) for row in validated}
        else:
            allocations = multinomial_allocations(budget_per_stream, stream_seed, validated)
        allocation_manifest[str(stream_seed)] = allocations
        for row in validated:
            family = row["seed_id"]
            target = int(allocations[family])
            slug = legacy.slugify(family)
            output_dir = work_dir / "stage1" / f"run_{stream_seed}" / slug
            config = dict(base)
            config.update(
                {
                    "seed": stream_seed,
                    "max_oracle_calls": target,
                    "pop_size": min(24, max(4, target)) if smoke else 120,
                    "belief_capacity": min(24, max(4, target)) if smoke else 120,
                    "initial_smiles_file": str(work_dir / "inputs" / f"{slug}.csv"),
                    "output_dir": str(output_dir),
                    "parent_seed": family,
                    "seed_family": family,
                    "run_seed": stream_seed,
                    "generation_batch": f"stage1_run{stream_seed}_{slug}",
                    "checkpoint_dir": str(output_dir / "checkpoints"),
                    "checkpoint_every_generations": 1,
                    "resume_checkpoint": True,
                    "export_selected_sdf": False,
                    "selected_sdf_generate_3d": False,
                    "biot5_model_name": MODEL_NAME,
                }
            )
            config_path = output_dir / "batch_config.json"
            legacy.atomic_json(config_path, config)
            specs.append(
                {
                    "stage": 1,
                    "name": family,
                    "seed_family": family,
                    "stream_seed": stream_seed,
                    "slug": slug,
                    "target": target,
                    "seed": stream_seed,
                    "config": config_path,
                    "output": output_dir,
                    "entry": STAGE1_ENTRY,
                    "canonical": output_dir / STAGE1_CANONICAL,
                    "batch_label": config["generation_batch"],
                }
            )
    return specs, allocation_manifest


def descriptor_record(mol: Chem.Mol) -> Dict[str, Any]:
    panel = compute_property_panel(Chem.MolToSmiles(mol, canonical=True), regularizer="sa")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "cLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "QED": float(QED.qed(mol)),
        "SA_score": raw_sa_score(mol),
        "SA_objective": float(panel["regularizer_score"]),
        "BBB_score": float(panel["bbb_cns_score"]),
        "CNS_score": float(panel["bbb_cns_score"]),
        "Murcko_scaffold": canonical_isomeric(scaffold) if scaffold.GetNumAtoms() else "",
    }


def write_smi(path: Path, rows: Sequence[Dict[str, Any]], id_column: str = "compound_id") -> None:
    atomic_text(
        path,
        "".join(f"{row['canonical_smiles']}\t{row[id_column]}\n" for row in rows),
    )


def write_optional_parquet(frame: pd.DataFrame, path: Path) -> bool:
    try:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        return True
    except Exception as exc:
        print(f"Parquet export unavailable for {path}: {exc}")
        return False

def canonical_set_from_csv(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return set()
    column = next(
        (name for name in ["canonical_smiles", "smiles", "canonical_SMILES", "SMILES"] if name in frame.columns),
        None,
    )
    if column is None:
        return set()
    result = set()
    for value in frame[column].dropna().astype(str):
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            result.add(canonical_isomeric(mol))
    return result


def append_generation_history(specs: Sequence[Dict[str, Any]], path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        source = spec["output"] / "generation_sources.csv"
        if not source.exists():
            continue
        for row in pd.read_csv(source).to_dict("records"):
            rows.append(
                {
                    "stage": spec["stage"],
                    "run_seed": spec["stream_seed"],
                    "seed_family": spec["seed_family"],
                    "generation_batch": spec["batch_label"],
                    **row,
                }
            )
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return len(rows)


def stage1_record(
    source: Dict[str, Any],
    spec: Dict[str, Any],
    mol: Chem.Mol,
    canonical: str,
    inchikey: str,
    is_pareto: bool,
) -> Dict[str, Any]:
    pred_pic50 = legacy.as_float(source.get("raw_predicted_pic50"), float("nan"))
    activity = legacy.as_float(
        source.get("qsar_activity_score"),
        legacy.as_float(source.get("lrrk2_wt_activity"), float("nan")),
    )
    raw_activity = legacy.as_float(source.get("activity_score"), float("nan"))
    if not np.isfinite(raw_activity) and np.isfinite(pred_pic50):
        raw_activity = float(np.clip((pred_pic50 - 6.0) / 3.0, 0.0, 1.0))
    pred_ic50 = float(10 ** (9.0 - pred_pic50)) if np.isfinite(pred_pic50) else float("nan")
    return {
        "compound_id": "",
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "stage": "stage1",
        "run_seed": int(spec["stream_seed"]),
        "generation": "",
        "seed_family": spec["seed_family"],
        "parent_id_1": "",
        "parent_id_2": "",
        "parent_smiles_1": "",
        "parent_smiles_2": "",
        "source_operator": "unavailable_in_current_callm_archive",
        "generation_batch": spec["batch_label"],
        "pred_pIC50": pred_pic50,
        "pred_IC50": pred_ic50,
        "raw_activity_score": raw_activity,
        "qsar_activity_score_ad_penalized": activity,
        "AD_similarity": legacy.as_float(source.get("nearest_train_similarity"), float("nan")),
        "AD_category": str(source.get("applicability_domain") or source.get("oracle_confidence") or ""),
        **descriptor_record(mol),
        "Pareto_front": bool(is_pareto),
        "Pareto_rank": 0 if is_pareto else 1,
        "crowding_distance": "",
        "selection_score": legacy.as_float(source.get("aggregated_score"), 0.0),
        "stage1_score": legacy.as_float(source.get("aggregated_score"), 0.0),
        "oracle_call_first_seen": source.get("oracle_call_first_seen", ""),
        "oracle_call_best": source.get("oracle_call_best", ""),
        "validity_status": "PASS",
    }


def build_stage1_archive(
    work_dir: Path,
    specs: Sequence[Dict[str, Any]],
    validated: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seed_keys = {row["inchikey"] for row in validated}
    best_by_key: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    duplicate_log: List[Dict[str, Any]] = []
    raw_evaluated = 0

    for spec in specs:
        frame = pd.read_csv(spec["canonical"])
        pareto_set = canonical_set_from_csv(spec["output"] / STAGE1_PARETO)
        for source in frame.to_dict("records"):
            raw_evaluated += 1
            raw_smiles = str(source.get("smiles") or source.get("canonical_smiles") or "")
            mol, canonical, key, reasons = legacy.standardize_and_qc(raw_smiles)
            if key and key in seed_keys:
                reasons = list(reasons) + ["initial_reference_seed_not_generated"]
            if reasons or mol is None:
                rejected.append(
                    {
                        "stage": "stage1",
                        "run_seed": spec["stream_seed"],
                        "seed_family": spec["seed_family"],
                        "generation_batch": spec["batch_label"],
                        "smiles": raw_smiles,
                        "reason": ";".join(sorted(set(reasons or ["invalid_molecule"]))),
                    }
                )
                continue
            record = stage1_record(source, spec, mol, canonical, key, canonical in pareto_set)
            previous = best_by_key.get(key)
            if previous is None:
                best_by_key[key] = record
                continue
            keep_new = (
                record["stage1_score"],
                record["qsar_activity_score_ad_penalized"],
            ) > (
                previous["stage1_score"],
                previous["qsar_activity_score_ad_penalized"],
            )
            kept = record if keep_new else previous
            removed = previous if keep_new else record
            best_by_key[key] = kept
            duplicate_log.append(
                {
                    "inchikey": key,
                    "canonical_smiles": canonical,
                    "kept_generation_batch": kept["generation_batch"],
                    "removed_generation_batch": removed["generation_batch"],
                    "reason": "cross_batch_canonical_duplicate",
                }
            )

    rows = sorted(
        best_by_key.values(),
        key=lambda row: (
            bool(row["Pareto_front"]),
            float(row["stage1_score"]),
            float(row["qsar_activity_score_ad_penalized"]),
            float(row["QED"]),
            row["inchikey"],
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["compound_id"] = f"CALLM_LRRK2_S1_{index:06d}"

    stage_dir = work_dir / "stage1"
    frame = pd.DataFrame(rows)
    atomic_csv(frame, stage_dir / "all_valid_unique.csv")
    write_optional_parquet(frame, stage_dir / "all_valid_unique.parquet")
    write_smi(stage_dir / "all_valid_unique.smi", rows)
    atomic_csv(pd.DataFrame(rejected), stage_dir / "rejected_invalid.csv")
    atomic_csv(pd.DataFrame(duplicate_log), stage_dir / "duplicate_log.csv")
    generation_rows = append_generation_history(specs, stage_dir / "generation_history.jsonl")
    stats = {
        "raw_evaluated": raw_evaluated,
        "valid_unique": len(rows),
        "rejected": len(rejected),
        "duplicates": len(duplicate_log),
        "generation_history_rows": generation_rows,
        "by_stream": dict(Counter(str(row["run_seed"]) for row in rows)),
        "by_seed_family": dict(Counter(row["seed_family"] for row in rows)),
    }
    legacy.atomic_json(
        stage_dir / "checkpoint_manifest.json",
        {
            "complete": all(legacy.batch_complete(spec, spec["target"]) for spec in specs),
            "batch_checkpoints": [str(spec["output"] / "checkpoints" / "stage_checkpoint.pkl") for spec in specs],
            "archive": str(stage_dir / "all_valid_unique.csv"),
            "statistics": stats,
        },
    )
    return rows, stats


def group_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def select_stage2_parents(
    work_dir: Path,
    stage1_rows: Sequence[Dict[str, Any]],
    count: int = STAGE1_HANDOFF,
) -> List[Dict[str, Any]]:
    if len(stage1_rows) < count:
        raise RuntimeError(f"Stage 1 has only {len(stage1_rows)} valid generated molecules; need {count}")
    ordered = sorted(
        stage1_rows,
        key=lambda row: (
            bool(row.get("Pareto_front")),
            legacy.as_float(row.get("stage1_score"), 0.0),
            legacy.as_float(row.get("qsar_activity_score_ad_penalized"), 0.0),
            {"high": 3, "medium": 2, "low": 1, "very_low": 0}.get(str(row.get("AD_category", "")).lower(), -1),
            legacy.as_float(row.get("AD_similarity"), 0.0),
            legacy.as_float(row.get("QED"), 0.0),
            str(row.get("inchikey", "")),
        ),
        reverse=True,
    )
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    scaffold_counts: Counter = Counter()
    for cap in [1, 2, 4, 8, 1_000_000]:
        for row in ordered:
            if row["inchikey"] in selected_keys:
                continue
            scaffold = row.get("Murcko_scaffold") or f"acyclic:{row['inchikey']}"
            if scaffold_counts[scaffold] >= cap:
                continue
            selected.append(dict(row))
            selected_keys.add(row["inchikey"])
            scaffold_counts[scaffold] += 1
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Diversity selector returned {len(selected)}, expected {count}")
    for index, row in enumerate(selected, start=1):
        row["stage2_parent_rank"] = index
        row["stage2_parent_id"] = f"CALLM_LRRK2_S2_PARENT_{index:03d}"

    stage_dir = work_dir / "stage1"
    atomic_csv(pd.DataFrame(selected), stage_dir / "stage1_to_stage2_seeds.csv")
    atomic_text(
        stage_dir / "stage1_to_stage2_seeds.smi",
        "".join(f"{row['canonical_smiles']}\t{row['stage2_parent_id']}\n" for row in selected),
    )
    parent_dir = stage_dir / "stage2_parent_pools"
    parent_dir.mkdir(parents=True, exist_ok=True)
    for family, rows in sorted(group_by(selected, "seed_family").items()):
        atomic_csv(
            pd.DataFrame(
                {
                    "smiles": row["canonical_smiles"],
                    "aggregated_score": row["stage1_score"],
                    "stage2_parent_id": row["stage2_parent_id"],
                    "seed_family": family,
                }
                for row in rows
            ),
            parent_dir / f"{legacy.slugify(family)}.csv",
        )
    return selected


def family_allocations(
    total: int,
    families: Sequence[str],
    family_sizes: Dict[str, int],
    random_seed: int,
) -> Dict[str, int]:
    probabilities = np.array([float(family_sizes[name]) for name in families], dtype=float)
    probabilities /= probabilities.sum()
    counts = np.random.default_rng(random_seed).multinomial(int(total), probabilities)
    if total >= len(families):
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[index] += 1
    return {name: int(count) for name, count in zip(families, counts)}


def make_stage2_specs(
    work_dir: Path,
    selected: Sequence[Dict[str, Any]],
    budget_per_stream: int,
    smoke: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    base = legacy.read_json(STAGE2_BASE, {})
    grouped = group_by(selected, "seed_family")
    families = sorted(grouped)
    sizes = {name: len(grouped[name]) for name in families}
    streams = [41] if smoke else STREAM_SEEDS
    specs: List[Dict[str, Any]] = []
    allocations_manifest: Dict[str, Dict[str, int]] = {}
    parent_dir = work_dir / "stage1" / "stage2_parent_pools"
    for stream_seed in streams:
        if smoke:
            allocations = {name: max(4, int(round(budget_per_stream * sizes[name] / len(selected)))) for name in families}
        else:
            allocations = family_allocations(budget_per_stream, families, sizes, stream_seed + 10_000)
        allocations_manifest[str(stream_seed)] = allocations
        for family in families:
            target = int(allocations[family])
            if target <= 0:
                continue
            slug = legacy.slugify(family)
            input_path = parent_dir / f"{slug}.csv"
            input_count = len(grouped[family])
            output_dir = work_dir / "stage2" / f"run_{stream_seed}" / slug
            batch_label = f"stage2_run{stream_seed}_{slug}"
            config = dict(base)
            config.update(
                {
                    "seed": stream_seed,
                    "max_oracle_calls": target,
                    "pop_size": min(24, max(4, target, input_count)) if smoke else 120,
                    "belief_capacity": min(24, max(4, target, input_count)) if smoke else 96,
                    # A non-empty explicit path prevents the legacy default seed fallback.
                    "initial_smiles_file": str(input_path),
                    "stage1_seed_file": str(input_path),
                    "stage1_seed_top_k": input_count,
                    "stage1_seed_sort_by": "aggregated_score",
                    "require_stage1_seed_file": True,
                    "parent_lead_smiles": "",
                    "lead_pool_smiles": [],
                    "initial_population_strategy": "stage1_only",
                    "initial_parent_lead_count": min(120, input_count),
                    "output_dir": str(output_dir),
                    "parent_seed": family,
                    "seed_family": family,
                    "run_seed": stream_seed,
                    "generation_batch": batch_label,
                    "checkpoint_dir": str(output_dir / "checkpoints"),
                    "checkpoint_every_generations": 1,
                    "resume_checkpoint": True,
                    "export_selected_sdf": False,
                    "selected_sdf_generate_3d": False,
                    "biot5_model_name": MODEL_NAME,
                    "regularizer_type": "sa",
                    "objective_names": [
                        "qsar_activity_ad_penalized",
                        "bbb_cns",
                        "logp_window",
                        "tpsa_window",
                        "regularizer",
                    ],
                }
            )
            config_path = output_dir / "batch_config.json"
            legacy.atomic_json(config_path, config)
            specs.append(
                {
                    "stage": 2,
                    "name": family,
                    "seed_family": family,
                    "stream_seed": stream_seed,
                    "slug": slug,
                    "target": target,
                    "seed": stream_seed,
                    "config": config_path,
                    "output": output_dir,
                    "entry": STAGE2_ENTRY,
                    "quality": output_dir / STAGE2_QUALITY,
                    "batch_label": batch_label,
                    "round": 0,
                    "input_path": input_path,
                    "input_count": input_count,
                }
            )
    return specs, allocations_manifest


def serialize_specs(path: Path, specs: Sequence[Dict[str, Any]]) -> None:
    serializable = []
    for spec in specs:
        row = dict(spec)
        for key in ["config", "output", "entry", "quality", "input_path", "canonical"]:
            if key in row:
                row[key] = str(row[key])
        serializable.append(row)
    legacy.atomic_json(path, serializable)


def load_specs(path: Path) -> List[Dict[str, Any]]:
    rows = legacy.read_json(path, [])
    for row in rows:
        for key in ["config", "output", "entry", "quality", "input_path", "canonical"]:
            if key in row:
                row[key] = Path(row[key])
    return rows


def extend_stage2_specs(
    specs: List[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    increment_per_stream: int,
    round_index: int,
) -> Dict[str, Dict[str, int]]:
    grouped = group_by(selected, "seed_family")
    families = sorted(grouped)
    sizes = {name: len(grouped[name]) for name in families}
    manifest: Dict[str, Dict[str, int]] = {}
    for stream_seed in STREAM_SEEDS:
        additions = family_allocations(
            increment_per_stream,
            families,
            sizes,
            stream_seed + 10_000 + round_index * 100,
        )
        manifest[str(stream_seed)] = additions
        for spec in specs:
            if spec["stream_seed"] != stream_seed:
                continue
            addition = int(additions.get(spec["seed_family"], 0))
            if addition <= 0:
                continue
            spec["target"] = int(spec["target"]) + addition
            spec["round"] = round_index
            config = legacy.read_json(spec["config"], {})
            config["max_oracle_calls"] = int(spec["target"])
            config["production_extension_round"] = round_index
            legacy.atomic_json(spec["config"], config)
    return manifest


def stage2_record(
    source: Dict[str, Any],
    spec: Dict[str, Any],
    mol: Chem.Mol,
    canonical: str,
    inchikey: str,
    is_pareto: bool,
    parent_pool_ids: str,
) -> Dict[str, Any]:
    pred_pic50 = legacy.as_float(source.get("raw_predicted_pic50"), float("nan"))
    raw_activity = legacy.as_float(source.get("activity_score"), float("nan"))
    if not np.isfinite(raw_activity) and np.isfinite(pred_pic50):
        raw_activity = float(np.clip((pred_pic50 - 6.0) / 3.0, 0.0, 1.0))
    activity = legacy.as_float(
        source.get("diagnostic_qsar_activity_ad_penalized"),
        legacy.as_float(source.get("qsar_activity_ad_penalized"), float("nan")),
    )
    pred_ic50 = float(10 ** (9.0 - pred_pic50)) if np.isfinite(pred_pic50) else float("nan")
    properties = descriptor_record(mol)
    properties["SA_objective"] = legacy.as_float(
        source.get("regularizer_score"), properties["SA_objective"]
    )
    properties["BBB_score"] = legacy.as_float(source.get("bbb_cns_score"), properties["BBB_score"])
    properties["CNS_score"] = properties["BBB_score"]
    return {
        "compound_id": "",
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "stage": "stage2",
        "run_seed": int(spec["stream_seed"]),
        "generation": "",
        "seed_family": spec["seed_family"],
        "parent_id_1": "",
        "parent_id_2": "",
        "parent_pool_ids": parent_pool_ids,
        "source_operator": "unavailable_in_current_callm_archive",
        "generation_batch": spec["batch_label"],
        "pred_pIC50": pred_pic50,
        "pred_IC50": pred_ic50,
        "raw_activity_score": raw_activity,
        "qsar_activity_score_ad_penalized": activity,
        "AD_similarity": legacy.as_float(source.get("nearest_train_similarity"), float("nan")),
        "AD_category": str(source.get("applicability_domain") or source.get("oracle_confidence") or ""),
        **properties,
        "Pareto_front": bool(is_pareto),
        "Pareto_rank": 0 if is_pareto else 1,
        "crowding_distance": "",
        "aggregated_score": legacy.as_float(source.get("aggregated_score"), 0.0),
        "qsar_activity_ad_penalized": activity,
        "raw_predicted_pic50": pred_pic50,
        "nearest_train_similarity": legacy.as_float(source.get("nearest_train_similarity"), float("nan")),
        "applicability_domain": str(source.get("applicability_domain") or source.get("oracle_confidence") or ""),
        "stage2_score": legacy.as_float(source.get("aggregated_score"), 0.0),
        "selection_score": legacy.as_float(source.get("aggregated_score"), 0.0),
        "logp_window_score": legacy.as_float(source.get("logp_window_score"), 0.0),
        "tpsa_window_score": legacy.as_float(source.get("tpsa_window_score"), 0.0),
        "bbb_cns_score": legacy.as_float(source.get("bbb_cns_score"), 0.0),
        "regularizer_score": legacy.as_float(source.get("regularizer_score"), 0.0),
        "clean_candidate_pass": legacy.truthy(source.get("clean_candidate_pass")),
        "structure_risk_pass": legacy.truthy(source.get("structure_risk_pass", True)),
        "new_stage2_offspring": True,
        "validity_status": "PASS",
    }


def build_stage2_archive(
    work_dir: Path,
    specs: Sequence[Dict[str, Any]],
    validated: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    original_keys = {row["inchikey"] for row in validated}
    parent_keys = {row["inchikey"] for row in selected}
    parent_ids_by_family = {
        family: ";".join(row["stage2_parent_id"] for row in rows)
        for family, rows in group_by(selected, "seed_family").items()
    }
    best_by_key: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    duplicate_log: List[Dict[str, Any]] = []
    raw_evaluated = 0
    for spec in specs:
        frame = pd.read_csv(spec["quality"])
        pareto_set = canonical_set_from_csv(spec["output"] / STAGE2_PARETO)
        for source in frame.to_dict("records"):
            raw_evaluated += 1
            raw_smiles = str(
                source.get("smiles")
                or source.get("canonical_smiles")
                or source.get("canonical_SMILES")
                or ""
            )
            mol, canonical, key, reasons = legacy.standardize_and_qc(raw_smiles)
            reasons = list(reasons)
            if "structure_risk_pass" in source and not legacy.truthy(source.get("structure_risk_pass")):
                reasons.append("existing_callm_structure_risk_block")
            if key and key in original_keys:
                reasons.append("exact_original_seed_copy")
            if key and key in parent_keys:
                reasons.append("unchanged_stage2_input_parent")
            if reasons or mol is None:
                rejected.append(
                    {
                        "stage": "stage2",
                        "run_seed": spec["stream_seed"],
                        "seed_family": spec["seed_family"],
                        "generation_batch": spec["batch_label"],
                        "smiles": raw_smiles,
                        "reason": ";".join(sorted(set(reasons or ["invalid_molecule"]))),
                    }
                )
                continue
            record = stage2_record(
                source,
                spec,
                mol,
                canonical,
                key,
                canonical in pareto_set,
                parent_ids_by_family.get(spec["seed_family"], ""),
            )
            previous = best_by_key.get(key)
            if previous is None:
                best_by_key[key] = record
                continue
            keep_new = legacy.stage2_sort_key(record) > legacy.stage2_sort_key(previous)
            kept = record if keep_new else previous
            removed = previous if keep_new else record
            best_by_key[key] = kept
            duplicate_log.append(
                {
                    "inchikey": key,
                    "canonical_smiles": canonical,
                    "kept_generation_batch": kept["generation_batch"],
                    "removed_generation_batch": removed["generation_batch"],
                    "reason": "cross_stream_or_family_duplicate",
                }
            )

    rows = sorted(best_by_key.values(), key=legacy.stage2_sort_key, reverse=True)
    for index, row in enumerate(rows, start=1):
        row["compound_id"] = f"CALLM_LRRK2_S2_ARCHIVE_{index:06d}"
    stage_dir = work_dir / "stage2"
    frame = pd.DataFrame(rows)
    atomic_csv(frame, stage_dir / "all_valid_unique.csv")
    write_optional_parquet(frame, stage_dir / "all_valid_unique.parquet")
    write_smi(stage_dir / "all_valid_unique.smi", rows)
    atomic_csv(pd.DataFrame(rejected), stage_dir / "rejected_invalid.csv")
    atomic_csv(pd.DataFrame(duplicate_log), stage_dir / "duplicate_log.csv")
    generation_rows = append_generation_history(specs, stage_dir / "generation_history.jsonl")
    raw_proposals = 0
    history_path = stage_dir / "generation_history.jsonl"
    for line in history_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        raw_proposals += sum(
            int(item.get(key, 0) or 0)
            for key in ["crossover_candidates", "biot5_candidates", "random_mutation_candidates"]
        )
    stats = {
        "raw_proposal_count": raw_proposals,
        "evaluated_count": raw_evaluated,
        "valid_unique_count": len(rows),
        "rejected_count": len(rejected),
        "duplicate_count": len(duplicate_log),
        "generation_history_rows": generation_rows,
        "by_stream": dict(Counter(str(row["run_seed"]) for row in rows)),
        "by_seed_family": dict(Counter(row["seed_family"] for row in rows)),
    }
    legacy.atomic_json(
        stage_dir / "checkpoint_manifest.json",
        {
            "complete": len(rows) >= FINAL_TARGET,
            "batch_checkpoints": [str(spec["output"] / "checkpoints" / "stage_checkpoint.pkl") for spec in specs],
            "archive": str(stage_dir / "all_valid_unique.csv"),
            "statistics": stats,
        },
    )
    return rows, stats


def prepare_sdf_molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("MolFromSmiles returned None")
    Chem.SanitizeMol(mol)
    aromatic_bond_indices = {
        bond.GetIdx() for bond in mol.GetBonds() if bond.GetIsAromatic()
    }
    Chem.Kekulize(mol, clearAromaticFlags=True)
    AllChem.Compute2DCoords(mol)
    for bond in mol.GetBonds():
        if (
            bond.GetIdx() in aromatic_bond_indices
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            bond.SetStereo(Chem.BondStereo.STEREOANY)
            bond.SetBondDir(Chem.BondDir.EITHERDOUBLE)
    return mol


def sdf_roundtrip_matches(smiles: str) -> bool:
    expected = Chem.MolFromSmiles(smiles)
    if expected is None:
        return False
    try:
        exported = prepare_sdf_molecule(smiles)
        mol_block = Chem.MolToMolBlock(
            exported, includeStereo=True, kekulize=False, forceV3000=False
        )
        reloaded = Chem.MolFromMolBlock(
            mol_block, removeHs=False, sanitize=True, strictParsing=True
        )
        return (
            reloaded is not None
            and Chem.MolToInchiKey(Chem.RemoveHs(reloaded))
            == Chem.MolToInchiKey(expected)
        )
    except Exception:
        return False


def select_final_rows(
    rows: Sequence[Dict[str, Any]],
    target: int,
    roundtrip_exclusions_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if len(rows) < target:
        raise RuntimeError(f"Stage 2 archive has {len(rows)} unique valid new molecules; need {target}")
    ordered = sorted(rows, key=legacy.stage2_sort_key, reverse=True)
    selected: List[Dict[str, Any]] = []
    seen = set()
    scaffold_counts: Counter = Counter()
    compatibility: Dict[str, bool] = {}
    roundtrip_exclusions: Dict[str, Dict[str, Any]] = {}
    # Keep a soft scaffold cap first, then fill strictly by the established score tuple.
    for cap in [20, 50, 1_000_000]:
        for row in ordered:
            if row["inchikey"] in seen:
                continue
            key = row["inchikey"]
            if key not in compatibility:
                compatibility[key] = sdf_roundtrip_matches(row["canonical_smiles"])
            if not compatibility[key]:
                roundtrip_exclusions.setdefault(
                    key,
                    {
                        "archive_compound_id": row.get("compound_id", ""),
                        "canonical_smiles": row["canonical_smiles"],
                        "inchikey": key,
                        "reason": "sdf_full_inchikey_roundtrip_mismatch",
                    },
                )
                continue
            scaffold = row.get("Murcko_scaffold") or f"acyclic:{row['inchikey']}"
            if scaffold_counts[scaffold] >= cap:
                continue
            selected.append(dict(row))
            seen.add(row["inchikey"])
            scaffold_counts[scaffold] += 1
            if len(selected) == target:
                break
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(f"Final selection returned {len(selected)}, expected {target}")
    if roundtrip_exclusions_path is not None:
        roundtrip_exclusions_path.parent.mkdir(parents=True, exist_ok=True)
        exclusion_columns = [
            "archive_compound_id", "canonical_smiles", "inchikey", "reason"
        ]
        atomic_csv(
            pd.DataFrame(list(roundtrip_exclusions.values()), columns=exclusion_columns),
            roundtrip_exclusions_path,
        )
    for index, row in enumerate(selected, start=1):
        row["compound_id"] = f"CALLM_LRRK2_S2_{index:06d}"
        row["final_rank"] = index
    return selected


def write_final_exports(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Path]:
    final_dir = run_dir / "final_docking_20000"
    final_dir.mkdir(parents=True, exist_ok=True)
    csv_path = final_dir / "lrrk2_stage2_20000.csv"
    smi_path = final_dir / "lrrk2_stage2_20000_for_docking.smi"
    sdf_path = final_dir / "lrrk2_stage2_20000.sdf"
    scaffold_path = final_dir / "lrrk2_stage2_20000_scaffolds.csv"
    frame = pd.DataFrame(rows)
    atomic_csv(frame, csv_path)
    write_smi(smi_path, rows)

    scaffold_frequency = Counter(row.get("Murcko_scaffold") or "" for row in rows)
    scaffold_rows = [
        {
            "compound_id": row["compound_id"],
            "canonical_smiles": row["canonical_smiles"],
            "Murcko_scaffold": row.get("Murcko_scaffold", ""),
            "scaffold_frequency": scaffold_frequency[row.get("Murcko_scaffold") or ""],
            "seed_family": row["seed_family"],
            "final_rank": row["final_rank"],
        }
        for row in rows
    ]
    atomic_csv(pd.DataFrame(scaffold_rows), scaffold_path)

    temporary_sdf = sdf_path.with_suffix(".sdf.tmp")
    writer = Chem.SDWriter(str(temporary_sdf))
    # Mol blocks use Kekule single/double bonds for aromatic systems. Without
    # an explicit "either" marker, InChI can infer spurious E/Z layers from the
    # 2D coordinates of those representation-only double bonds. Kekulize a
    # copy ourselves and mark only aromatic-derived double bonds as unknown;
    # genuine non-aromatic E/Z and atom stereochemistry remain untouched.
    writer.SetKekulize(False)
    try:
        for row in rows:
            try:
                mol = prepare_sdf_molecule(row["canonical_smiles"])
            except Exception as exc:
                raise RuntimeError(
                    f"SDF export could not prepare {row['compound_id']}: {exc}"
                ) from exc
            mol.SetProp("_Name", row["compound_id"])
            for key, value in row.items():
                if value is None:
                    continue
                if isinstance(value, float) and not np.isfinite(value):
                    continue
                mol.SetProp(str(key), str(value))
            writer.write(mol)
    finally:
        writer.close()
    os.replace(temporary_sdf, sdf_path)
    return {
        "directory": final_dir,
        "csv": csv_path,
        "smi": smi_path,
        "sdf": sdf_path,
        "scaffolds": scaffold_path,
    }


def parse_smi(path: Path) -> List[Tuple[str, str]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"SMI line {line_number} does not have exactly two tab-separated fields")
        records.append((fields[0].strip(), fields[1].strip()))
    return records


def validate_exports(
    run_dir: Path,
    validated: Sequence[Dict[str, Any]],
    selected_parents: Sequence[Dict[str, Any]],
    target: int = FINAL_TARGET,
    final_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    final_dir = final_dir or run_dir / "final_docking_20000"
    csv_path = final_dir / "lrrk2_stage2_20000.csv"
    smi_path = final_dir / "lrrk2_stage2_20000_for_docking.smi"
    sdf_path = final_dir / "lrrk2_stage2_20000.sdf"
    frame = pd.read_csv(csv_path)
    smi_rows = parse_smi(smi_path)
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    sdf_mols = list(supplier)
    sdf_ids = [mol.GetProp("compound_id") if mol is not None and mol.HasProp("compound_id") else "" for mol in sdf_mols]
    csv_ids = frame["compound_id"].astype(str).tolist()
    smi_ids = [row[1] for row in smi_rows]

    parsed_keys = []
    invalid = []
    noncanonical = []
    for index, (smiles, compound_id) in enumerate(smi_rows):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("MolFromSmiles returned None")
            Chem.SanitizeMol(mol)
            canonical = canonical_isomeric(mol)
            if canonical != smiles:
                noncanonical.append(compound_id)
            parsed_keys.append(Chem.MolToInchiKey(mol))
        except Exception as exc:
            invalid.append({"index": index, "compound_id": compound_id, "error": str(exc)})
    original_keys = {row["inchikey"] for row in validated}
    parent_keys = {row["inchikey"] for row in selected_parents}
    required_columns = {
        "compound_id", "canonical_smiles", "stage", "run_seed", "generation",
        "seed_family", "parent_id_1", "parent_id_2", "source_operator",
        "pred_pIC50", "pred_IC50", "raw_activity_score",
        "qsar_activity_score_ad_penalized", "AD_similarity", "AD_category",
        "MW", "cLogP", "TPSA", "HBD", "HBA", "rotatable_bonds",
        "formal_charge", "QED", "SA_score", "BBB_score", "CNS_score",
        "Murcko_scaffold", "Pareto_front", "Pareto_rank", "final_rank",
    }
    provenance_columns = ["run_seed", "seed_family", "generation_batch"]
    provenance_complete = all(
        frame[column].notna().all()
        and frame[column].astype(str).str.strip().ne("").all()
        for column in provenance_columns
    )
    sdf_structure_keys = []
    for mol in sdf_mols:
        if mol is None:
            continue
        Chem.SanitizeMol(mol)
        sdf_structure_keys.append(Chem.MolToInchiKey(Chem.RemoveHs(mol)))

    checks = {
        "A": {"passed": len(smi_rows) == target, "detail": f"smi_records={len(smi_rows)}"},
        "B": {"passed": len(set(smi_ids)) == target and all(smi_ids), "detail": f"unique_ids={len(set(smi_ids))}"},
        "C": {"passed": len(set(parsed_keys)) == target and not noncanonical, "detail": f"unique_structures={len(set(parsed_keys))}, noncanonical={len(noncanonical)}"},
        "D": {"passed": not invalid and len(parsed_keys) == target, "detail": f"invalid={len(invalid)}"},
        "E": {"passed": not invalid and len(parsed_keys) == target, "detail": "RDKit sanitization completed for every SMI record"},
        "F": {"passed": csv_ids == smi_ids and len(csv_ids) == target, "detail": f"csv_rows={len(csv_ids)}"},
        "G": {"passed": sdf_ids == smi_ids and len(sdf_ids) == target, "detail": f"sdf_records={len(sdf_ids)}"},
        "H": {"passed": not (set(parsed_keys) & original_keys), "detail": f"original_seed_copies={len(set(parsed_keys) & original_keys)}"},
        "I": {"passed": not (set(parsed_keys) & parent_keys), "detail": f"unchanged_stage2_parents={len(set(parsed_keys) & parent_keys)}"},
        "J": {"passed": frame["canonical_smiles"].notna().all() and frame["canonical_smiles"].astype(str).str.strip().ne("").all(), "detail": f"blank={int(frame['canonical_smiles'].isna().sum())}"},
        "K": {"passed": required_columns.issubset(frame.columns), "detail": f"missing={sorted(required_columns - set(frame.columns))}"},
        "L": {"passed": set(frame["stage"].astype(str)) == {"stage2"}, "detail": f"stages={sorted(set(frame['stage'].astype(str)))}"},
        "M": {"passed": provenance_complete, "detail": f"required_provenance={provenance_columns}; exact per-offspring parent/operator unavailable in current CALLM core"},
        "N": {"passed": math.isclose(sum(float(row["weight"]) for row in validated), 1.0, rel_tol=0.0, abs_tol=1e-12), "detail": f"weight_sum={sum(float(row['weight']) for row in validated):.12f}"},
        "O": {"passed": len(validated) == 8 and all(row["rdkit_parse_status"] == "PASS" and row["rdkit_sanitize_status"] == "PASS" for row in validated), "detail": f"validated_seeds={len(validated)}"},
        "P": {"passed": len(validated) == 8 and all(row["selfies_roundtrip_status"] == "PASS" and row["graph_equivalence_status"] == "PASS" for row in validated), "detail": "all eight SELFIES graph round-trips passed"},
        "Q": {"passed": not invalid and len(parsed_keys) == target, "detail": "complete docking SMI reloaded by RDKit"},
        "R": {"passed": len(frame) == target and csv_ids == smi_ids, "detail": f"csv_rows={len(frame)}"},
        "S": {"passed": len(sdf_mols) == target and sdf_ids == smi_ids and sdf_structure_keys == parsed_keys, "detail": f"sdf_rows={len(sdf_mols)}, structure_matches={sdf_structure_keys == parsed_keys}"},
    }
    for item in checks.values():
        item["passed"] = bool(item["passed"])
    report = {
        "passed": all(item["passed"] for item in checks.values()),
        "target": target,
        "checks": checks,
        "invalid_sample": invalid[:20],
        "noncanonical_sample": noncanonical[:20],
    }
    legacy.atomic_json(final_dir / "validation_report.json", report)
    if not report["passed"]:
        failed = {key: value for key, value in checks.items() if not value["passed"]}
        raise RuntimeError(f"Final validation failed: {failed}")
    return report


def version_or_unavailable(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return f"unavailable: {exc}"


def write_manifest_and_summary(
    run_dir: Path,
    validated: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    stage1_stats: Dict[str, Any],
    stage2_stats: Dict[str, Any],
    final_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    stage1_allocations: Dict[str, Dict[str, int]],
    stage2_allocations: Dict[str, Dict[str, int]],
) -> None:
    final_dir = run_dir / "final_docking_20000"
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=20, check=False,
        )
        gpu_models = gpu_result.stdout.strip()
    except Exception as exc:
        gpu_models = f"unavailable: {exc}"
    oracle_metadata = legacy.read_json(ORACLE_DIR / "lrrk2_qsar_oracle_metadata.json", {})
    roundtrip_exclusions_path = final_dir / "sdf_roundtrip_exclusions.csv"
    if roundtrip_exclusions_path.exists():
        roundtrip_exclusion_count = len(pd.read_csv(roundtrip_exclusions_path))
    else:
        roundtrip_exclusion_count = 0
    scaffold_count = len({row.get("Murcko_scaffold") or "" for row in final_rows})
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "working_tree_status": git_status,
        "python_version": sys.version,
        "rdkit_version": rdBase.rdkitVersion,
        "torch_version": torch_version,
        "transformers_version": version_or_unavailable("transformers"),
        "selfies_version": version_or_unavailable("selfies"),
        "CUDA_version": cuda_version,
        "GPU_model": gpu_models,
        "BioT5_model": MODEL_NAME,
        "QSAR_oracle_path": str(ORACLE_DIR),
        "QSAR_oracle_version": oracle_metadata.get("oracle_version"),
        "Stage_1_configuration": str(SEED_CONFIG),
        "Stage_2_configuration": str(STAGE2_CONFIG),
        "original_seed_ids": [row["seed_id"] for row in validated],
        "original_input_smiles": [row["input_smiles"] for row in validated],
        "original_canonical_smiles": [row["canonical_smiles"] for row in validated],
        "original_seed_weights": [row["weight"] for row in validated],
        "random_seeds": STREAM_SEEDS,
        "stage1_allocations": stage1_allocations,
        "stage2_initial_allocations": stage2_allocations,
        "Stage_1_generated_count": stage1_stats["raw_evaluated"],
        "Stage_1_valid_count": stage1_stats["raw_evaluated"] - stage1_stats["rejected"],
        "Stage_1_unique_count": stage1_stats["valid_unique"],
        "Stage_2_raw_proposal_count": stage2_stats["raw_proposal_count"],
        "Stage_2_valid_count": stage2_stats["evaluated_count"] - stage2_stats["rejected_count"],
        "Stage_2_unique_count": stage2_stats["valid_unique_count"],
        "final_docking_count": len(final_rows),
        "number_of_unique_Murcko_scaffolds": scaffold_count,
        "SDF_roundtrip_exclusion_count": roundtrip_exclusion_count,
        "lineage_note": "Exact original family/run/batch provenance is retained. The current CALLM core does not persist exact per-offspring parent IDs or source operator.",
    }
    legacy.atomic_json(final_dir / "manifest.json", manifest)

    distributions = {}
    for column in ["pred_pIC50", "qsar_activity_score_ad_penalized", "MW", "cLogP", "TPSA", "HBD", "HBA", "rotatable_bonds", "SA_score", "BBB_score"]:
        values = pd.to_numeric(pd.Series([row.get(column) for row in final_rows]), errors="coerce").dropna()
        distributions[column] = {
            "min": float(values.min()), "median": float(values.median()),
            "mean": float(values.mean()), "max": float(values.max()),
        }
    by_family = Counter(row["seed_family"] for row in final_rows)
    lines = [
        "# LRRK2 Stage 1 -> Stage 2 Production Summary", "",
        f"- Stage 1 evaluated: {stage1_stats['raw_evaluated']:,}",
        f"- Stage 1 valid unique: {stage1_stats['valid_unique']:,}",
        f"- Stage 2 raw proposals: {stage2_stats['raw_proposal_count']:,}",
        f"- Stage 2 evaluated: {stage2_stats['evaluated_count']:,}",
        f"- Stage 2 valid unique new offspring: {stage2_stats['valid_unique_count']:,}",
        f"- Final docking export: {len(final_rows):,}",
        f"- Unique Murcko scaffolds in final export: {scaffold_count:,}", "",
        "## Seed Validation", "",
    ]
    lines.extend(
        f"- {row['seed_id']}: PASS; canonical {row['canonical_smiles']}; weight {row['weight']:.2f}"
        for row in validated
    )
    lines.extend(["", "## Stage 2 Parent Selection", "", f"- Selected {len(selected)} generated Stage-1 molecules using Pareto quality, AD-aware score, and a deterministic Murcko scaffold cap.", "- Original reference seeds were not reinjected.", "", "## Final Family Provenance", ""])
    lines.extend(f"- {name}: {count:,} ({count / len(final_rows):.2%})" for name, count in sorted(by_family.items()))
    lines.extend(["", "## Property Distributions", ""])
    lines.extend(f"- {name}: {json.dumps(values, sort_keys=True)}" for name, values in distributions.items())
    lines.extend(
        [
            "", "## Warnings", "",
            "- Exact per-offspring parent IDs, per-molecule generation numbers, and source operators are not emitted by the current CALLM optimizer; run seed, original family, batch, aggregate operator histories, and parent-pool IDs are retained.",
            "- The original family weights are implemented as per-stream multinomial family call-budget allocation, matching the repository's established independent-family production pattern.",
            f"- {roundtrip_exclusion_count} ranked candidates remain in the Stage-2 archive but were skipped in the final pool because their 2D SDF mol blocks did not preserve the full standardized InChIKey on RDKit round-trip.",
            "- No docking was performed. The SDF contains sanitized 2D structures for downstream LigPrep/Maestro processing.",
            "", "## Final Files", "",
        ]
    )
    for path in sorted(final_dir.iterdir()):
        lines.append(f"- {path}")
    atomic_text(final_dir / "run_summary.md", "\n".join(lines) + "\n")


def copy_seed_inputs(source_run: Path, target_run: Path, validated: Sequence[Dict[str, Any]]) -> None:
    target_inputs = target_run / "inputs"
    target_inputs.mkdir(parents=True, exist_ok=True)
    for row in validated:
        name = f"{legacy.slugify(row['seed_id'])}.csv"
        target = target_inputs / name
        if not target.exists():
            target.write_bytes((source_run / "inputs" / name).read_bytes())


def write_resolved_config(
    run_dir: Path,
    args: argparse.Namespace,
    validated: Sequence[Dict[str, Any]],
    stage1_allocations: Dict[str, Dict[str, int]],
    stage2_allocations: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    stage1_base, stage2_base = legacy.clean_stage_templates()
    payload = {
        "run_id": run_dir.name,
        "workflow": "LRRK2 WT Stage 1 -> Stage 2",
        "generator": MODEL_NAME,
        "oracle_dir": str(ORACLE_DIR),
        "stream_seeds": STREAM_SEEDS,
        "stage1_budget_per_stream": args.stage1_budget_per_stream,
        "stage2_initial_budget_per_stream": args.stage2_budget_per_stream,
        "stage2_increment_per_stream": args.stage2_increment_per_stream,
        "stage1_handoff_count": STAGE1_HANDOFF,
        "final_target": FINAL_TARGET,
        "seed_family_sampling": "per-stream multinomial CALLM call-budget allocation",
        "stage1_allocations": stage1_allocations,
        "stage2_initial_allocations": stage2_allocations or {},
        "stage1_base_config_without_io": stage1_base,
        "stage2_base_config_without_io": stage2_base,
        "stage2_overrides": {
            "regularizer_type": "sa",
            "objective_names": [
                "qsar_activity_ad_penalized", "bbb_cns", "logp_window",
                "tpsa_window", "regularizer",
            ],
            "parent_lead_smiles": "",
            "initial_population_strategy": "stage1_only",
        },
        "seeds": list(validated),
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = [
        "DNL201_GNE-0877", "PF-06447475", "MLi-2", "GNE-7915",
        "LRRK2_compound18_YXD", "250k_rndm_zinc",
    ]
    leftovers = [value for value in forbidden if value in serialized]
    if leftovers:
        raise RuntimeError(f"Resolved production configuration contains old seed/input values: {leftovers}")
    legacy.atomic_json(run_dir / "resolved_config.json", payload)


def run_smoke(
    run_dir: Path,
    validated: Sequence[Dict[str, Any]],
    gpus: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    smoke_dir = run_dir / "smoke"
    copy_seed_inputs(run_dir, smoke_dir, validated)
    stage1_specs, stage1_allocations = make_stage1_specs(
        smoke_dir, validated, args.smoke_stage1_budget, smoke=True
    )
    partial = {spec["name"]: max(1, int(spec["target"]) // 2) for spec in stage1_specs}
    legacy.run_specs(stage1_specs, gpus, smoke_dir, target_override=partial, phase_label="smoke-stage1-partial")
    legacy.run_specs(stage1_specs, gpus, smoke_dir, phase_label="smoke-stage1-resume")
    stage1_rows, stage1_stats = build_stage1_archive(smoke_dir, stage1_specs, validated)
    smoke_parent_count = min(16, len(stage1_rows))
    selected = select_stage2_parents(smoke_dir, stage1_rows, smoke_parent_count)

    stage2_specs, stage2_allocations = make_stage2_specs(
        smoke_dir, selected, args.smoke_stage2_budget, smoke=True
    )
    partial2 = {spec["name"]: max(1, int(spec["target"]) // 2) for spec in stage2_specs}
    legacy.run_specs(stage2_specs, gpus, smoke_dir, target_override=partial2, phase_label="smoke-stage2-partial")
    legacy.run_specs(stage2_specs, gpus, smoke_dir, phase_label="smoke-stage2-resume")
    stage2_rows, stage2_stats = build_stage2_archive(
        smoke_dir, stage2_specs, validated, selected
    )
    if not stage2_rows:
        raise RuntimeError("Smoke Stage 2 produced no unique valid new molecules")
    smoke_target = min(10, len(stage2_rows))
    smoke_final = select_final_rows(stage2_rows, smoke_target)
    paths = write_final_exports(smoke_dir, smoke_final)
    validation = validate_exports(
        smoke_dir, validated, selected, target=smoke_target, final_dir=paths["directory"]
    )
    resumed = []
    for path in smoke_dir.glob("stage*/run_*/*/batch.log"):
        if "Resumed checkpoint:" in path.read_text(encoding="utf-8", errors="replace"):
            resumed.append(str(path))
    checks = {
        "all_eight_seeds_loaded": set(stage1_stats["by_seed_family"]) == {row["seed_id"] for row in validated},
        "weight_sum_one": math.isclose(sum(float(row["weight"]) for row in validated), 1.0, abs_tol=1e-12),
        "stage1_completed": all(legacy.batch_complete(spec, spec["target"]) for spec in stage1_specs),
        "stage1_archive_created": (smoke_dir / "stage1" / "all_valid_unique.csv").exists(),
        "stage1_to_stage2_handoff": len(selected) == smoke_parent_count,
        "no_original_seed_reinjection": all(
            not str(legacy.read_json(spec["config"], {}).get("parent_lead_smiles", "")).strip()
            for spec in stage2_specs
        ),
        "stage2_five_objectives": all(
            legacy.read_json(spec["config"], {}).get("objective_names")
            == ["qsar_activity_ad_penalized", "bbb_cns", "logp_window", "tpsa_window", "regularizer"]
            and legacy.read_json(spec["config"], {}).get("regularizer_type") == "sa"
            for spec in stage2_specs
        ),
        "stage2_completed": all(legacy.batch_complete(spec, spec["target"]) for spec in stage2_specs),
        "stage2_archive_created": (smoke_dir / "stage2" / "all_valid_unique.csv").exists(),
        "resume_observed": len(resumed) == len(stage1_specs) + len(stage2_specs),
        "smoke_exports_valid": validation["passed"],
        "all_outputs_have_family_lineage": all(row.get("seed_family") for row in stage2_rows),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "stage1_allocations": stage1_allocations,
        "stage2_allocations": stage2_allocations,
        "stage1_statistics": stage1_stats,
        "stage2_statistics": stage2_stats,
        "resumed_batch_logs": resumed,
    }
    legacy.atomic_json(smoke_dir / "smoke_test_report.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Smoke test failed: {checks}")
    return report


def run_full(
    run_dir: Path,
    validated: Sequence[Dict[str, Any]],
    gpus: List[str],
    args: argparse.Namespace,
) -> None:
    stage1_specs, stage1_allocations = make_stage1_specs(
        run_dir, validated, args.stage1_budget_per_stream, smoke=False
    )
    serialize_specs(run_dir / "stage1" / "batch_specs.json", stage1_specs)
    write_resolved_config(run_dir, args, validated, stage1_allocations)
    legacy.run_specs(stage1_specs, gpus, run_dir, phase_label="production-stage1")
    stage1_rows, stage1_stats = build_stage1_archive(run_dir, stage1_specs, validated)
    selected = select_stage2_parents(run_dir, stage1_rows, STAGE1_HANDOFF)

    specs_path = run_dir / "stage2" / "batch_specs.json"
    if specs_path.exists():
        stage2_specs = load_specs(specs_path)
        resolved = legacy.read_json(run_dir / "resolved_config.json", {})
        stage2_allocations = resolved.get("stage2_initial_allocations", {})
    else:
        stage2_specs, stage2_allocations = make_stage2_specs(
            run_dir, selected, args.stage2_budget_per_stream, smoke=False
        )
        serialize_specs(specs_path, stage2_specs)
        write_resolved_config(
            run_dir, args, validated, stage1_allocations, stage2_allocations
        )

    legacy.run_specs(stage2_specs, gpus, run_dir, phase_label="production-stage2-initial")
    stage2_rows, stage2_stats = build_stage2_archive(
        run_dir, stage2_specs, validated, selected
    )
    current_round = max(int(spec.get("round", 0)) for spec in stage2_specs)
    while len(stage2_rows) < FINAL_TARGET:
        current_round += 1
        if current_round > args.max_stage2_rounds:
            raise RuntimeError(
                f"Stage 2 reached max extension rounds with {len(stage2_rows)} unique valid new molecules"
            )
        additions = extend_stage2_specs(
            stage2_specs, selected, args.stage2_increment_per_stream, current_round
        )
        serialize_specs(specs_path, stage2_specs)
        extension_path = run_dir / "stage2" / "extension_allocations.json"
        extension_manifest = legacy.read_json(extension_path, {})
        extension_manifest[str(current_round)] = additions
        legacy.atomic_json(extension_path, extension_manifest)
        legacy.run_specs(
            stage2_specs, gpus, run_dir,
            phase_label=f"production-stage2-extension-{current_round:02d}",
        )
        stage2_rows, stage2_stats = build_stage2_archive(
            run_dir, stage2_specs, validated, selected
        )
        print(
            f"Stage 2 extension {current_round}: "
            f"{len(stage2_rows):,}/{FINAL_TARGET:,} unique valid new molecules"
        )

    final_rows = select_final_rows(
        stage2_rows,
        FINAL_TARGET,
        run_dir / "final_docking_20000" / "sdf_roundtrip_exclusions.csv",
    )
    write_final_exports(run_dir, final_rows)
    validation = validate_exports(run_dir, validated, selected, FINAL_TARGET)
    write_manifest_and_summary(
        run_dir, validated, selected, stage1_stats, stage2_stats,
        final_rows, args, stage1_allocations, stage2_allocations,
    )
    report = legacy.read_json(run_dir / "final_docking_20000" / "validation_report.json", {})
    report["passed"] = bool(validation["passed"])
    report["manifest_present"] = (run_dir / "final_docking_20000" / "manifest.json").exists()
    report["run_summary_present"] = (run_dir / "final_docking_20000" / "run_summary.md").exists()
    legacy.atomic_json(run_dir / "final_docking_20000" / "validation_report.json", report)


def load_validated_seed_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "inputs" / "lrrk2_stage1_seed_pool.csv"
    return pd.read_csv(path).to_dict("records")


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(ROOT / "results" / f"lrrk2_stage1_stage2_20k_{timestamp}"))
    parser.add_argument("--mode", choices=["init", "smoke", "full", "all", "validate"], default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--stage1-budget-per-stream", type=int, default=5000)
    parser.add_argument("--stage2-budget-per-stream", type=int, default=5000)
    parser.add_argument("--stage2-increment-per-stream", type=int, default=1000)
    parser.add_argument("--max-stage2-rounds", type=int, default=20)
    parser.add_argument("--smoke-stage1-budget", type=int, default=64)
    parser.add_argument("--smoke-stage2-budget", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy.setup_run_log(run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Mode: {args.mode}")
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"CALLM Python environment not found: {PYTHON_BIN}")
    required = [SEED_CONFIG, STAGE2_CONFIG, STAGE1_BASE, STAGE2_BASE, STAGE1_ENTRY, STAGE2_ENTRY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required workflow files are missing: {missing}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")

    if args.mode == "validate":
        validated = load_validated_seed_rows(run_dir)
        selected = pd.read_csv(run_dir / "stage1" / "stage1_to_stage2_seeds.csv").to_dict("records")
        validate_exports(run_dir, validated, selected, FINAL_TARGET)
        print("Validation passed")
        return 0

    validated = validate_and_write_seeds(run_dir)
    if not (run_dir / "environment_versions.json").exists():
        legacy.record_environment(run_dir)
    stage1_allocations = {
        str(seed): multinomial_allocations(args.stage1_budget_per_stream, seed, validated)
        for seed in STREAM_SEEDS
    }
    write_resolved_config(run_dir, args, validated, stage1_allocations)
    if args.mode == "init":
        print("Initialization and eight-seed validation complete")
        return 0
    if args.mode in {"smoke", "all"}:
        run_smoke(run_dir, validated, gpus, args)
        print("Smoke test passed")
    if args.mode in {"full", "all"}:
        if not legacy.read_json(run_dir / "smoke" / "smoke_test_report.json", {}).get("passed"):
            raise RuntimeError("Full production requires a passing smoke test in this run directory")
        run_full(run_dir, validated, gpus, args)
        print("RUN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

