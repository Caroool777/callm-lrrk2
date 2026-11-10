"""
Reusable LRRK2 QSAR activity oracle and property utilities for MPO tasks.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.QED import qed as rdkit_qed


DEFAULT_QSAR_SCORE_COLUMN = "qsar_activity_score_ad_penalized"
DEFAULT_QSAR_RAW_COLUMN = "pred_pIC50"


DEFAULT_STRUCTURE_RISK_BLOCK_FLAGS = [
    "N-F_substitution",
    "N-O_substitution",
    "explicit_SH_valence",
    "sulfur_multiple_bond",
    "radical",
    "unusual_elements:H",
]
DEFAULT_STRUCTURE_RISK_BLOCK_PREFIXES = [
    "hetero_halogen_bond:",
    "hetero_hetero_bond:",
]
DEFAULT_STRUCTURE_RISK_ALLOWED_ELEMENTS = {
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "I",
    "N",
    "O",
    "P",
    "S",
}


def structure_risk_filter_config(config: Optional[Dict] = None) -> Tuple[List[str], List[str]]:
    config = config or {}
    block_flags = config.get("structure_filter_block_flags", DEFAULT_STRUCTURE_RISK_BLOCK_FLAGS)
    block_prefixes = config.get("structure_filter_block_prefixes", DEFAULT_STRUCTURE_RISK_BLOCK_PREFIXES)
    return [str(flag) for flag in block_flags], [str(prefix) for prefix in block_prefixes]


def identify_structure_risk_flags(smiles_or_mol) -> List[str]:
    """Return simple structural-risk flags used for lead-repair filtering."""
    if isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
        smiles = Chem.MolToSmiles(mol, canonical=True)
    else:
        smiles = str(smiles_or_mol or "").strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ["invalid_rdkit"]

    flags = set()
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    unusual = sorted(elements - DEFAULT_STRUCTURE_RISK_ALLOWED_ELEMENTS)
    if unusual:
        flags.add("unusual_elements:" + ";".join(unusual))
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        flags.add("formal_charge")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        flags.add("radical")

    for bond in mol.GetBonds():
        symbol_a = bond.GetBeginAtom().GetSymbol()
        symbol_b = bond.GetEndAtom().GetSymbol()
        pair = "-".join(sorted([symbol_a, symbol_b]))
        if pair in {"Br-N", "Cl-N", "F-N", "F-O", "F-S"}:
            flags.add(f"hetero_halogen_bond:{pair}")
        if pair in {"N-O", "N-S", "O-S"} and bond.GetBondTypeAsDouble() != 1.0:
            flags.add(f"hetero_hetero_bond:{pair}")
        if pair == "F-N":
            flags.add("N-F_substitution")

    if "N(O)" in smiles or "NO" in smiles:
        flags.add("N-O_substitution")
    if "[SH]" in smiles or "=[SH]" in smiles:
        flags.add("explicit_SH_valence")
    if "#S" in smiles or "=[SH]" in smiles:
        flags.add("sulfur_multiple_bond")

    return sorted(flags)


def structure_flags_are_blocking(
    flags: Sequence[str],
    block_flags: Optional[Sequence[str]] = None,
    block_prefixes: Optional[Sequence[str]] = None,
) -> bool:
    block_flag_set = set(block_flags or DEFAULT_STRUCTURE_RISK_BLOCK_FLAGS)
    prefixes = tuple(block_prefixes or DEFAULT_STRUCTURE_RISK_BLOCK_PREFIXES)
    for flag in flags:
        if flag in block_flag_set:
            return True
        if prefixes and str(flag).startswith(prefixes):
            return True
    return False


def annotate_structure_quality(
    row: Dict[str, object],
    block_flags: Optional[Sequence[str]] = None,
    block_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    annotated = dict(row)
    smiles = str(
        row.get("canonical_smiles")
        or row.get("canonical_SMILES")
        or row.get("smiles")
        or row.get("SMILES")
        or ""
    )
    flags = identify_structure_risk_flags(smiles)
    is_blocking = structure_flags_are_blocking(flags, block_flags=block_flags, block_prefixes=block_prefixes)
    annotated["structure_risk_flags"] = ";".join(flags)
    annotated["structure_risk_flag_count"] = len(flags)
    annotated["structure_risk_pass"] = not is_blocking
    return annotated


def _candidate_sort_float(row: Dict[str, object], key: str, default: float = float("-inf")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _candidate_ad_rank(row: Dict[str, object]) -> int:
    value = str(row.get("applicability_domain") or row.get("oracle_confidence") or "").lower()
    return {"high": 3, "medium": 2, "low": 1, "very_low": 0}.get(value, -1)


def candidate_threshold_config(config: Optional[Dict] = None) -> Dict[str, float]:
    config = config or {}
    return {
        "min_aggregated_score": float(config.get("clean_candidate_min_aggregated_score", 0.85)),
        "min_activity_score": float(config.get("clean_candidate_min_activity_score", 0.55)),
        "min_bbb_cns": float(config.get("clean_candidate_min_bbb_cns", 0.8)),
        "min_logp_window": float(config.get("clean_candidate_min_logp_window", 0.8)),
        "min_tpsa_window": float(config.get("clean_candidate_min_tpsa_window", 0.8)),
        "min_ad_rank": float(config.get("clean_candidate_min_ad_rank", 1)),
    }


def annotate_clean_candidate_quality(row: Dict[str, object], config: Optional[Dict] = None) -> Dict[str, object]:
    thresholds = candidate_threshold_config(config)
    annotated = dict(row)
    activity = _candidate_sort_float(
        row,
        "qsar_activity_ad_penalized",
        _candidate_sort_float(row, "lrrk2_wt_activity", _candidate_sort_float(row, "activity_score", 0.0)),
    )
    bbb_cns = _candidate_sort_float(row, "bbb_cns_score", _candidate_sort_float(row, "bbb_cns", 1.0))
    logp_window = _candidate_sort_float(row, "logp_window_score", _candidate_sort_float(row, "logp_window", 1.0))
    tpsa_window = _candidate_sort_float(row, "tpsa_window_score", _candidate_sort_float(row, "tpsa_window", 1.0))
    aggregated = _candidate_sort_float(row, "aggregated_score", 0.0)
    ad_rank = float(_candidate_ad_rank(row))
    failure_reasons = []
    if not bool(row.get("structure_risk_pass")):
        failure_reasons.append("structure_risk")
    if aggregated < thresholds["min_aggregated_score"]:
        failure_reasons.append("low_aggregated_score")
    if activity < thresholds["min_activity_score"]:
        failure_reasons.append("low_activity_score")
    if bbb_cns < thresholds["min_bbb_cns"]:
        failure_reasons.append("low_bbb_cns")
    if logp_window < thresholds["min_logp_window"]:
        failure_reasons.append("low_logp_window")
    if tpsa_window < thresholds["min_tpsa_window"]:
        failure_reasons.append("low_tpsa_window")
    if ad_rank < thresholds["min_ad_rank"]:
        failure_reasons.append("low_applicability_domain")
    annotated["clean_candidate_pass"] = not failure_reasons
    annotated["clean_candidate_failure_reasons"] = ";".join(failure_reasons)
    annotated["clean_candidate_activity_score"] = activity
    annotated["clean_candidate_ad_rank"] = ad_rank
    return annotated


def clean_candidate_sort_key(row: Dict[str, object]) -> Tuple[float, ...]:
    """Sort clean lead-repair candidates by balanced score before raw potency tie-breakers."""
    return (
        _candidate_sort_float(row, "aggregated_score"),
        _candidate_sort_float(row, "qsar_activity_ad_penalized", _candidate_sort_float(row, "activity_score")),
        _candidate_sort_float(row, "raw_predicted_pic50"),
        float(_candidate_ad_rank(row)),
        _candidate_sort_float(row, "regularizer_score"),
        _candidate_sort_float(row, "nearest_train_similarity"),
        _candidate_sort_float(row, "bbb_cns_score", _candidate_sort_float(row, "bbb_cns")),
        _candidate_sort_float(row, "logp_window_score", _candidate_sort_float(row, "logp_window")),
        _candidate_sort_float(row, "tpsa_window_score", _candidate_sort_float(row, "tpsa_window")),
    )


def export_structure_filtered_candidate_tables(
    canonical_rows: Sequence[Dict[str, object]],
    output_dir: str,
    output_prefix: str,
    config: Optional[Dict] = None,
) -> Dict[str, object]:
    import json
    import pandas as pd

    config = config or {}
    block_flags, block_prefixes = structure_risk_filter_config(config)
    annotated_rows = [
        annotate_clean_candidate_quality(
            annotate_structure_quality(row, block_flags=block_flags, block_prefixes=block_prefixes),
            config=config,
        )
        for row in canonical_rows
    ]
    clean_rows = [row for row in annotated_rows if bool(row.get("clean_candidate_pass"))]
    structure_flagged_rows = [row for row in annotated_rows if not bool(row.get("structure_risk_pass"))]
    threshold_rejected_rows = [
        row
        for row in annotated_rows
        if bool(row.get("structure_risk_pass")) and not bool(row.get("clean_candidate_pass"))
    ]
    clean_rows = sorted(clean_rows, key=clean_candidate_sort_key, reverse=True)

    for rank, row in enumerate(clean_rows, start=1):
        row["clean_rank"] = rank

    os.makedirs(output_dir, exist_ok=True)
    quality_path = os.path.join(
        output_dir,
        config.get("structure_quality_filename", f"{output_prefix}_structure_quality.csv"),
    )
    flagged_path = os.path.join(
        output_dir,
        config.get("flagged_candidate_filename", f"{output_prefix}_flagged_candidates.csv"),
    )
    clean_path = os.path.join(
        output_dir,
        config.get("clean_candidate_filename", f"{output_prefix}_clean_candidates.csv"),
    )
    summary_path = os.path.join(
        output_dir,
        config.get("clean_candidate_summary_filename", f"{output_prefix}_clean_candidate_summary.json"),
    )

    pd.DataFrame(annotated_rows).to_csv(quality_path, index=False)
    pd.DataFrame(structure_flagged_rows).to_csv(flagged_path, index=False)
    pd.DataFrame(clean_rows).to_csv(clean_path, index=False)

    flag_counts: Dict[str, int] = {}
    for row in annotated_rows:
        for flag in str(row.get("structure_risk_flags") or "").split(";"):
            if not flag:
                continue
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    summary = {
        "total_candidates": len(annotated_rows),
        "clean_candidates": len(clean_rows),
        "structure_flagged_candidates": len(structure_flagged_rows),
        "threshold_rejected_candidates": len(threshold_rejected_rows),
        "block_flags": block_flags,
        "block_prefixes": block_prefixes,
        "candidate_thresholds": candidate_threshold_config(config),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda item: item[1], reverse=True)),
        "quality_csv": quality_path,
        "flagged_csv": flagged_path,
        "clean_csv": clean_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    structure_pass_rows = [row for row in annotated_rows if bool(row.get("structure_risk_pass"))]
    structure_pass_rows = sorted(structure_pass_rows, key=clean_candidate_sort_key, reverse=True)

    return {
        "annotated_rows": annotated_rows,
        "clean_rows": clean_rows,
        "flagged_rows": structure_flagged_rows,
        "threshold_rejected_rows": threshold_rejected_rows,
        "structure_pass_rows": structure_pass_rows,
        "summary": summary,
        "quality_csv": quality_path,
        "flagged_csv": flagged_path,
        "clean_csv": clean_path,
        "summary_json": summary_path,
    }


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def window_score(value: float, lower: float, upper: float, sigma: float) -> float:
    if lower <= value <= upper:
        return 1.0
    distance = min(abs(value - lower), abs(value - upper))
    sigma = max(float(sigma), 1e-6)
    return clip01(float(np.exp(-0.5 * (distance / sigma) ** 2)))


def tpsa_window_score(value: float, lower: float = 20.0, upper: float = 70.0, sigma: float = 12.0) -> float:
    return window_score(value, lower, upper, sigma)


def logp_window_score(value: float, lower: float = 1.5, upper: float = 3.5, sigma: float = 0.75) -> float:
    return window_score(value, lower, upper, sigma)


def cns_mpo_like_score(
    mol: Chem.Mol,
    logp_window: Optional[Dict[str, float]] = None,
    tpsa_window: Optional[Dict[str, float]] = None,
    mw_window: Optional[Dict[str, float]] = None,
    hbd_upper: int = 1,
) -> float:
    if mol is None:
        return 0.0

    logp_cfg = logp_window or {"lower": 1.5, "upper": 3.5, "sigma": 0.75}
    tpsa_cfg = tpsa_window or {"lower": 20.0, "upper": 70.0, "sigma": 12.0}
    mw_cfg = mw_window or {"lower": 250.0, "upper": 450.0, "sigma": 35.0}

    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    hbd_score = 1.0 if hbd <= hbd_upper else float(np.exp(-0.8 * (hbd - hbd_upper)))

    return float(
        np.mean(
            [
                logp_window_score(logp, **logp_cfg),
                tpsa_window_score(tpsa, **tpsa_cfg),
                window_score(mw, **mw_cfg),
                clip01(hbd_score),
            ]
        )
    )


def simple_synth_druglike_score(mol: Chem.Mol) -> float:
    if mol is None:
        return 0.0
    try:
        return clip01(rdkit_qed(mol))
    except Exception:
        return 0.0


def synthetic_accessibility_score(mol: Chem.Mol) -> float:
    """Normalized SA score; high means easier to synthesize."""
    if mol is None:
        return 0.0
    try:
        try:
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        except Exception:
            sa_dir = Path(RDConfig.RDContribDir) / "SA_Score"
            if str(sa_dir) not in sys.path:
                sys.path.append(str(sa_dir))
            import sascorer  # type: ignore
        raw_sa = float(sascorer.calculateScore(mol))
        return clip01((10.0 - raw_sa) / 9.0)
    except Exception:
        heavy = max(1, mol.GetNumHeavyAtoms())
        rings = float(Lipinski.RingCount(mol))
        rotors = float(Lipinski.NumRotatableBonds(mol))
        return clip01(1.0 - (0.015 * heavy + 0.08 * rings + 0.025 * rotors))


def compute_property_panel(
    smiles: str,
    logp_window: Optional[Dict[str, float]] = None,
    tpsa_window: Optional[Dict[str, float]] = None,
    regularizer: str = "qed",
) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "bbb_cns_score": 0.0,
            "logp_window_score": 0.0,
            "tpsa_window_score": 0.0,
            "regularizer_score": 0.0,
            "raw_logp": 0.0,
            "raw_tpsa": 0.0,
            "raw_mw": 0.0,
        }

    raw_logp = float(Crippen.MolLogP(mol))
    raw_tpsa = float(Descriptors.TPSA(mol))
    raw_mw = float(Descriptors.MolWt(mol))
    regularizer_key = str(regularizer or "qed").lower()
    if regularizer_key in {"qed", "druglike", "drug_likeness", "qed_druglike"}:
        regularizer_score = simple_synth_druglike_score(mol)
    elif regularizer_key in {"sa", "sas", "sa_score", "synthetic_accessibility"}:
        regularizer_score = synthetic_accessibility_score(mol)
    else:
        raise ValueError(f"Unknown regularizer_type: {regularizer}")

    return {
        "bbb_cns_score": cns_mpo_like_score(mol, logp_window=logp_window, tpsa_window=tpsa_window),
        "logp_window_score": logp_window_score(raw_logp, **(logp_window or {"lower": 1.5, "upper": 3.5, "sigma": 0.75})),
        "tpsa_window_score": tpsa_window_score(raw_tpsa, **(tpsa_window or {"lower": 20.0, "upper": 70.0, "sigma": 12.0})),
        "regularizer_score": regularizer_score,
        "raw_logp": raw_logp,
        "raw_tpsa": raw_tpsa,
        "raw_mw": raw_mw,
    }


def load_smiles_from_csv(
    csv_path: str,
    fallback_column: int = 0,
    top_k: Optional[int] = None,
    sort_by: str = "aggregated_score",
) -> List[str]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False, kind="mergesort")
    if top_k is not None and int(top_k) > 0:
        df = df.head(int(top_k))
    if "smiles" in df.columns:
        smiles = df["smiles"].tolist()
    elif "canonical_smiles" in df.columns:
        smiles = df["canonical_smiles"].tolist()
    elif "canonical_SMILES" in df.columns:
        smiles = df["canonical_SMILES"].tolist()
    elif "SMILES" in df.columns:
        smiles = df["SMILES"].tolist()
    else:
        smiles = df.iloc[:, fallback_column].tolist()
    return [s for s in smiles if isinstance(s, str) and s.strip()]


def build_lead_centered_initial_population(
    config: Dict,
    task_dir: str,
    default_csv: Optional[str] = None,
) -> List[str]:
    """Build lead-repair initial population and record seed-source metadata."""
    explicit_leads = []
    metadata = {
        "stage1_seed_file": config.get("stage1_seed_file") or config.get("stage1_output_csv"),
        "stage1_seed_file_resolved": "",
        "stage1_seed_file_exists": False,
        "stage1_seed_loaded_count": 0,
        "initial_smiles_file": config.get("initial_smiles_file") or "",
        "initial_smiles_loaded_count": 0,
        "lead_pool_count": 0,
        "parent_lead_present": False,
        "parent_lead_mutation_seed_count": 0,
        "parent_lead_included": False,
        "final_initial_population_size": 0,
        "require_stage1_seed_file": bool(config.get("require_stage1_seed_file", False)),
        "initial_population_strategy": str(config.get("initial_population_strategy", "random_parent_lead_mutation")),
    }

    parent_lead = config.get("parent_lead_smiles")
    if isinstance(parent_lead, str) and parent_lead.strip():
        explicit_leads.append(parent_lead.strip())
        metadata["parent_lead_present"] = True

    lead_pool = config.get("lead_pool_smiles", [])
    if isinstance(lead_pool, list):
        lead_pool_items = [s.strip() for s in lead_pool if isinstance(s, str) and s.strip()]
        explicit_leads.extend(lead_pool_items)
        metadata["lead_pool_count"] = len(lead_pool_items)

    initial_smiles_file = config.get("initial_smiles_file")
    if not initial_smiles_file and default_csv:
        initial_smiles_file = default_csv

    if initial_smiles_file:
        csv_path = initial_smiles_file
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(task_dir, csv_path)
        metadata["initial_smiles_file"] = csv_path
        if os.path.exists(csv_path):
            initial_smiles = load_smiles_from_csv(csv_path)
            explicit_leads.extend(initial_smiles)
            metadata["initial_smiles_loaded_count"] = len(initial_smiles)

    stage1_seed_file = config.get("stage1_seed_file") or config.get("stage1_output_csv")
    if stage1_seed_file:
        stage1_path = stage1_seed_file
        if not os.path.isabs(stage1_path):
            stage1_path = os.path.join(task_dir, stage1_path)
        metadata["stage1_seed_file_resolved"] = stage1_path
        metadata["stage1_seed_file_exists"] = os.path.exists(stage1_path)
        if os.path.exists(stage1_path):
            stage1_top_k = config.get("stage1_seed_top_k", config.get("stage1_seed_count"))
            stage1_sort_by = config.get("stage1_seed_sort_by", "aggregated_score")
            stage1_smiles = load_smiles_from_csv(
                stage1_path,
                top_k=stage1_top_k,
                sort_by=stage1_sort_by,
            )
            explicit_leads.extend(stage1_smiles)
            metadata["stage1_seed_loaded_count"] = len(stage1_smiles)
            print(f"Loaded {len(stage1_smiles)} stage-1 seed molecules: {stage1_path}")
        elif config.get("require_stage1_seed_file", False):
            config["initial_population_metadata"] = metadata
            raise FileNotFoundError(f"Required stage-1 seed file not found: {stage1_path}")
        else:
            print(f"Warning: stage-1 seed file not found; using lead-centered seeds only: {stage1_path}")

    canonical = []
    seen = set()
    for smiles in explicit_leads:
        canonical_smiles = canonicalize_smiles(smiles)
        if canonical_smiles is None or canonical_smiles in seen:
            continue
        seen.add(canonical_smiles)
        canonical.append(canonical_smiles)

    target_count = int(config.get("initial_parent_lead_count", config.get("pop_size", len(canonical) or 1)))
    strategy = str(config.get("initial_population_strategy", "random_parent_lead_mutation")).lower()
    parent_can = canonicalize_smiles(parent_lead) if parent_lead else None
    before_mutation_count = len(canonical)
    if parent_can and strategy in {"random_parent_lead_mutation", "lead_mutation", "mutated_parent"}:
        parent_mol = Chem.MolFromSmiles(parent_can)
        single_dir = Path(task_dir).parent / "SingleObjective"
        if str(single_dir) not in sys.path:
            sys.path.append(str(single_dir))
        try:
            import mutate as mutate_module  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Could not import SingleObjective/mutate.py for initial population: {exc}")
        random.seed(int(config.get("seed", 42)))
        mutation_rate = float(config.get("initial_mutation_rate", 1.0))
        max_attempts = int(config.get("initial_population_max_attempts", target_count * 80))
        attempts = 0
        while len(canonical) < target_count and attempts < max_attempts:
            attempts += 1
            child = mutate_module.mutate(parent_mol, mutation_rate=mutation_rate)
            if child is None:
                continue
            child_can = canonicalize_smiles(Chem.MolToSmiles(child, canonical=True))
            if child_can and child_can != parent_can and child_can not in seen:
                seen.add(child_can)
                canonical.append(child_can)
    metadata["parent_lead_mutation_seed_count"] = max(0, len(canonical) - before_mutation_count)
    if parent_can and parent_can not in seen:
        canonical.insert(0, parent_can)
        metadata["parent_lead_included"] = True
    elif parent_can:
        metadata["parent_lead_included"] = True

    final_population = canonical[:target_count]
    metadata["final_initial_population_size"] = len(final_population)
    config["initial_population_metadata"] = metadata
    return final_population




def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def deduplicate_ranked_molecules_by_canonical_smiles(
    ranked_molecules: Sequence[Tuple[float, str, Sequence[float]]],
) -> List[Tuple[float, str, Sequence[float]]]:
    deduplicated: List[Tuple[float, str, Sequence[float]]] = []
    seen = set()

    for score, smiles, objectives in ranked_molecules:
        canonical_smiles = canonicalize_smiles(smiles)
        if canonical_smiles is None or canonical_smiles in seen:
            continue
        seen.add(canonical_smiles)
        deduplicated.append((float(score), canonical_smiles, list(objectives)))

    return deduplicated


def export_canonical_deduplicated_discovered_csv(
    all_discovered: Sequence[Tuple[Sequence[float], str, int]],
    output_csv_path: str,
    objective_names: Optional[Sequence[str]] = None,
    metadata_by_smiles: Optional[Dict[str, Dict[str, object]]] = None,
) -> int:
    import pandas as pd

    rows = collect_canonical_deduplicated_discovered_rows(
        all_discovered=all_discovered,
        objective_names=objective_names,
        metadata_by_smiles=metadata_by_smiles,
    )
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    return len(rows)


def collect_canonical_deduplicated_discovered_rows(
    all_discovered: Sequence[Tuple[Sequence[float], str, int]],
    objective_names: Optional[Sequence[str]] = None,
    metadata_by_smiles: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    objective_names = list(objective_names or [])
    metadata_by_smiles = metadata_by_smiles or {}
    grouped: Dict[str, Dict[str, object]] = {}

    for objectives, smiles, oracle_call in all_discovered:
        canonical_smiles = canonicalize_smiles(smiles)
        if canonical_smiles is None:
            continue

        objectives = [float(value) for value in objectives]
        aggregated_score = float(np.prod(objectives) ** (1.0 / len(objectives))) if objectives else 0.0
        metadata = dict(metadata_by_smiles.get(canonical_smiles, {}))

        if canonical_smiles not in grouped:
            grouped[canonical_smiles] = {
                "canonical_smiles": canonical_smiles,
                "best_smiles": canonical_smiles,
                "aggregated_score": aggregated_score,
                "objectives": objectives,
                "oracle_call_first_seen": int(oracle_call),
                "oracle_call_best": int(oracle_call),
                "n_occurrences": 1,
                "metadata": metadata,
            }
            continue

        grouped[canonical_smiles]["n_occurrences"] += 1
        grouped[canonical_smiles]["oracle_call_first_seen"] = min(
            int(grouped[canonical_smiles]["oracle_call_first_seen"]),
            int(oracle_call),
        )
        if aggregated_score > float(grouped[canonical_smiles]["aggregated_score"]):
            grouped[canonical_smiles]["aggregated_score"] = aggregated_score
            grouped[canonical_smiles]["objectives"] = objectives
            grouped[canonical_smiles]["best_smiles"] = canonical_smiles
            grouped[canonical_smiles]["oracle_call_best"] = int(oracle_call)
            grouped[canonical_smiles]["metadata"] = metadata

    def metadata_tiebreaker(record: Dict[str, object]) -> float:
        metadata = dict(record.get("metadata") or {})
        for key in ("raw_pic50_tiebreaker", "raw_predicted_pic50", "raw_predicted_pactivity"):
            try:
                value = float(metadata.get(key, float("-inf")))
                if np.isfinite(value):
                    return value
            except Exception:
                continue
        return float("-inf")

    rows = []
    ranked_records = sorted(
        grouped.values(),
        key=lambda item: (float(item["aggregated_score"]), metadata_tiebreaker(item)),
        reverse=True,
    )
    for rank, record in enumerate(ranked_records, start=1):
        row = {
            "rank": rank,
            "canonical_smiles": record["canonical_smiles"],
            "smiles": record["best_smiles"],
            "aggregated_score": float(record["aggregated_score"]),
            "n_occurrences": int(record["n_occurrences"]),
            "oracle_call_first_seen": int(record["oracle_call_first_seen"]),
            "oracle_call_best": int(record["oracle_call_best"]),
        }
        for idx, value in enumerate(record["objectives"]):
            obj_name = objective_names[idx] if idx < len(objective_names) else f"objective_{idx + 1}"
            row[obj_name] = float(value)
        for key, value in dict(record.get("metadata") or {}).items():
            output_key = key if key not in row else f"diagnostic_{key}"
            if isinstance(value, (list, tuple, dict, set)):
                value = str(value)
            row[output_key] = value
        rows.append(row)

    return rows


def canonical_discovered_rows_to_ranked_molecules(
    canonical_rows: Sequence[Dict[str, object]],
    objective_names: Optional[Sequence[str]] = None,
) -> List[Tuple[float, str, Sequence[float]]]:
    objective_names = list(objective_names or [])
    ranked_molecules: List[Tuple[float, str, Sequence[float]]] = []

    for row in canonical_rows:
        smiles = str(row.get("smiles") or row.get("canonical_smiles") or "")
        if not smiles:
            continue

        objectives = []
        for idx, obj_name in enumerate(objective_names):
            value = row.get(obj_name)
            if value is None:
                value = row.get(f"objective_{idx + 1}")
            if value is None:
                break
            objectives.append(float(value))

        if objective_names and len(objectives) != len(objective_names):
            continue

        reserved = {
            "rank",
            "canonical_smiles",
            "smiles",
            "aggregated_score",
            "n_occurrences",
            "oracle_call_first_seen",
            "oracle_call_best",
            *objective_names,
        }
        metadata = {key: value for key, value in row.items() if key not in reserved}
        ranked_molecules.append((float(row.get("aggregated_score", 0.0)), smiles, objectives, metadata))

    return ranked_molecules


def _prepare_mol_for_sdf(smiles: str, generate_3d: bool = True) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    if generate_3d:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            status = AllChem.EmbedMolecule(mol, params)
            if status == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                    except Exception:
                        pass
            else:
                AllChem.Compute2DCoords(mol)
        except Exception:
            try:
                AllChem.Compute2DCoords(mol)
            except Exception:
                pass
    else:
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass

    return mol


def export_ranked_molecules_to_sdf(
    ranked_molecules: Sequence[Tuple[float, str, Sequence[float]]],
    output_sdf_path: str,
    objective_names: Optional[Sequence[str]] = None,
    max_molecules: int = 20,
    generate_3d: bool = True,
    per_molecule_dir: Optional[str] = None,
) -> int:
    objective_names = list(objective_names or [])
    selected = list(ranked_molecules[: max(0, int(max_molecules))])
    if not selected:
        return 0

    os.makedirs(os.path.dirname(output_sdf_path), exist_ok=True)
    if per_molecule_dir:
        os.makedirs(per_molecule_dir, exist_ok=True)

    writer = Chem.SDWriter(output_sdf_path)
    written = 0

    try:
        for rank, item in enumerate(selected, start=1):
            if len(item) >= 4:
                score, smiles, objectives, metadata = item[0], item[1], item[2], dict(item[3] or {})
            else:
                score, smiles, objectives = item[0], item[1], item[2]
                metadata = {}
            mol = _prepare_mol_for_sdf(smiles, generate_3d=generate_3d)
            if mol is None:
                continue

            mol.SetProp("_Name", f"rank_{rank}")
            mol.SetProp("rank", str(rank))
            mol.SetProp("smiles", smiles)
            mol.SetProp("aggregated_score", f"{float(score):.6f}")
            for idx, value in enumerate(objectives):
                obj_name = objective_names[idx] if idx < len(objective_names) else f"objective_{idx + 1}"
                mol.SetProp(obj_name, f"{float(value):.6f}")
            for key, value in metadata.items():
                if value is None:
                    continue
                prop_name = str(key)
                if mol.HasProp(prop_name):
                    prop_name = f"diagnostic_{prop_name}"
                if isinstance(value, float):
                    mol.SetProp(prop_name, f"{value:.6g}")
                else:
                    mol.SetProp(prop_name, str(value))

            writer.write(mol)
            written += 1

            if per_molecule_dir:
                single_path = os.path.join(per_molecule_dir, f"rank_{rank:03d}.sdf")
                single_writer = Chem.SDWriter(single_path)
                single_writer.write(mol)
                single_writer.close()
    finally:
        writer.close()

    return written


@dataclass
class ActivityScore:
    raw_value: float
    normalized_score: float
    diagnostics: Dict[str, object] = field(default_factory=dict)


class LRRK2QSAROracleAdapter:
    """Adapter around the packaged LRRK2 QSAR oracle for CALLM MPO scoring."""

    def __init__(
        self,
        oracle_dir: Optional[str] = None,
        score_column: str = DEFAULT_QSAR_SCORE_COLUMN,
        raw_value_column: str = DEFAULT_QSAR_RAW_COLUMN,
    ):
        self.oracle_dir = self._resolve_oracle_dir(oracle_dir)
        self.score_column = score_column
        self.raw_value_column = raw_value_column
        self._score_cache: Dict[str, ActivityScore] = {}
        self._qsar_module = self._load_qsar_module(self.oracle_dir)
        self._oracle = self._qsar_module.load_oracle(self.oracle_dir)

    @staticmethod
    def _resolve_oracle_dir(oracle_dir: Optional[str]) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        if not oracle_dir:
            return repo_root / "lrrk2_qsar_oracle"
        path = Path(oracle_dir)
        if not path.is_absolute():
            repo_relative = (repo_root / path).resolve()
            if repo_relative.exists():
                return repo_relative
            task_relative = (Path(__file__).resolve().parent / path).resolve()
            if task_relative.exists():
                return task_relative
            path = repo_relative
        return path.resolve()

    @staticmethod
    def _load_qsar_module(oracle_dir: Path):
        module_path = oracle_dir / "lrrk2_qsar_oracle.py"
        if not module_path.exists():
            raise FileNotFoundError(f"LRRK2 QSAR oracle module not found: {module_path}")

        spec = importlib.util.spec_from_file_location("_callm_lrrk2_qsar_oracle", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load LRRK2 QSAR oracle module from: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            if value is None:
                return default
            numeric = float(value)
            if np.isnan(numeric):
                return default
            return numeric
        except Exception:
            return default

    def score(self, smiles: str) -> ActivityScore:
        return self.score_many([smiles])[0]

    def score_smiles(self, smiles: str) -> Tuple[float, float]:
        result = self.score(smiles)
        return result.raw_value, result.normalized_score

    def score_many(self, smiles_list: Sequence[str]) -> List[ActivityScore]:
        if not smiles_list:
            return []

        results: List[Optional[ActivityScore]] = [None] * len(smiles_list)
        uncached_smiles: List[str] = []
        positions_by_key: Dict[str, List[int]] = {}

        for idx, smiles in enumerate(smiles_list):
            canonical_smiles = canonicalize_smiles(smiles)
            if canonical_smiles is None:
                results[idx] = ActivityScore(raw_value=float("-inf"), normalized_score=0.0, diagnostics={"valid_smiles": False})
                continue

            cached = self._score_cache.get(canonical_smiles)
            if cached is not None:
                results[idx] = cached
                continue

            positions_by_key.setdefault(canonical_smiles, []).append(idx)
            if len(positions_by_key[canonical_smiles]) == 1:
                uncached_smiles.append(canonical_smiles)

        if uncached_smiles:
            import pandas as pd

            scored = self._qsar_module.score_dataframe(
                pd.DataFrame({"SMILES": uncached_smiles}),
                smiles_col="SMILES",
                oracle=self._oracle,
            )
            if self.score_column not in scored.columns:
                raise ValueError(f"QSAR score column not found: {self.score_column}")
            if self.raw_value_column not in scored.columns:
                raise ValueError(f"QSAR raw value column not found: {self.raw_value_column}")

            for canonical_smiles, row in zip(uncached_smiles, scored.to_dict("records")):
                valid = bool(row.get("valid_smiles", True))
                raw_value = self._to_float(row.get(self.raw_value_column), float("-inf"))
                normalized_score = clip01(self._to_float(row.get(self.score_column), 0.0)) if valid else 0.0
                activity_score = ActivityScore(
                    raw_value=raw_value,
                    normalized_score=normalized_score,
                    diagnostics={
                        "valid_smiles": valid,
                        "canonical_SMILES": row.get("canonical_SMILES", canonical_smiles),
                        "activity_score": self._to_float(row.get("activity_score"), 0.0),
                        "nearest_train_similarity": self._to_float(row.get("nearest_train_similarity"), float("nan")),
                        "applicability_domain": row.get("applicability_domain", "invalid"),
                        "oracle_confidence": row.get("oracle_confidence", "invalid"),
                    },
                )
                self._score_cache[canonical_smiles] = activity_score
                for idx in positions_by_key.get(canonical_smiles, []):
                    results[idx] = activity_score

        invalid = ActivityScore(raw_value=float("-inf"), normalized_score=0.0, diagnostics={"valid_smiles": False})
        return [result if result is not None else invalid for result in results]
