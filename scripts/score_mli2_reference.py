#!/usr/bin/env python3
"""Score only MLi-2 with the archived CALLM Stage-2 property configuration.

Existing generated-molecule tables are read for comparison, never rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LRRK2case"))
from lrrk2_qsar_mpo_utils import (
    LRRK2QSAROracleAdapter,
    annotate_clean_candidate_quality,
    annotate_structure_quality,
    compute_property_panel,
    structure_risk_filter_config,
)
from run_lrrk2_stage1_stage2_20k import descriptor_record

RUN = ROOT / "results/lrrk2_stage1_stage2_20k_20260902T101731Z"
EXPECTED_KEY = "ATUUNJCZCOMUKD-OKILXGFUSA-N"
FOUR_OBJECTIVES = ["qsar_activity_ad_penalized", "bbb_cns", "logp_window", "tpsa_window"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values):
    return float(np.prod(values) ** (1.0 / len(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/lrrk2_regeneration_5seeds_20000/inputs/mli_2.csv")
    parser.add_argument("--stage2-csv", type=Path, default=RUN / "final_docking_20000/lrrk2_stage2_20000.csv")
    parser.add_argument("--resolved-config", type=Path, default=RUN / "resolved_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "MultiResult/mli2_reference_properties")
    args = parser.parse_args()
    resolved = json.loads(args.resolved_config.read_text())
    config = dict(resolved["stage2_base_config_without_io"])
    config.update(resolved["stage2_overrides"])
    objectives = config["objective_names"]
    if objectives != FOUR_OBJECTIVES + ["regularizer"] or config["regularizer_type"] != "sa":
        raise ValueError("This export requires the archived five-objective SA Stage-2 configuration.")

    source = pd.read_csv(args.input)
    if len(source) != 1 or source.iloc[0]["parent_seed"] != "MLi-2":
        raise ValueError("Expected exactly one MLi-2 input row.")
    smiles = str(source.iloc[0]["smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.MolToInchiKey(mol) != EXPECTED_KEY:
        raise ValueError("MLi-2 structure/stereochemistry does not match the verified InChIKey.")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    stage2_hash = sha256(args.stage2_csv)
    stage2 = pd.read_csv(args.stage2_csv)
    if stage2["compound_id"].duplicated().any() or set(stage2["stage"]) != {"stage2"}:
        raise ValueError("Stage-2 reference table has duplicate IDs or unexpected stages.")

    print("Scoring MLi-2 with the existing QSAR model and property functions", flush=True)
    adapter = LRRK2QSAROracleAdapter()
    module = adapter._qsar_module
    oracle_dir = ROOT / "lrrk2_qsar_oracle"
    model_path = oracle_dir / "lrrk2_qsar_oracle_model.pkl"
    environment = json.loads((args.resolved_config.parent / "environment_versions.json").read_text())
    model_hash = sha256(model_path)
    if model_hash != environment["model_checkpoint_sha256"]:
        raise ValueError("QSAR model differs from the archived Stage-2 model.")
    if rdBase.rdkitVersion != environment["rdkit"]:
        raise ValueError("RDKit version differs from the archived Stage-2 descriptor environment.")
    scored = module.score_dataframe(pd.DataFrame({"SMILES": [canonical]}), oracle=adapter._oracle).iloc[0].to_dict()
    # The archived AD multipliers differ from current source for lower-confidence
    # molecules. MLi-2 must be high AD: both versions then use the raw score.
    if scored["applicability_domain"] != "high":
        raise ValueError("MLi-2 is no longer high AD; historical AD scoring needs explicit reconciliation.")

    descriptors = descriptor_record(mol)
    raw_sa = float(sascorer.calculateScore(mol))  # fail instead of silently using a proxy
    panel = compute_property_panel(canonical, logp_window=config["logp_window"],
                                   tpsa_window=config["tpsa_window"], regularizer="sa")
    if not np.isclose(panel["regularizer_score"], (10.0 - raw_sa) / 9.0, atol=1e-12):
        raise ValueError("SA normalization does not match RDKit SA_Score.")
    activity = float(scored["qsar_activity_score_ad_penalized"])
    mapping = dict(zip(FOUR_OBJECTIVES, [activity, panel["bbb_cns_score"],
                                       panel["logp_window_score"], panel["tpsa_window_score"]]))
    mapping["regularizer"] = panel["regularizer_score"]
    risk_flags, risk_prefixes = structure_risk_filter_config(config)
    row = annotate_structure_quality({"canonical_smiles": canonical}, risk_flags, risk_prefixes)
    if not row["structure_risk_pass"]:
        raise ValueError("Verified MLi-2 unexpectedly fails the archived structural filter.")
    aggregate = geometric_mean([mapping[name] for name in objectives])
    train = pd.read_csv(oracle_dir / "lrrk2_train_smiles.csv")
    train_matches = train[train["canonical_SMILES"] == canonical]
    row.update({
        **scored, **descriptors, **panel,
        "compound_id": "MLi-2", "compound_role": "reference", "CAS": "1627091-47-7",
        "canonical_smiles": canonical, "inchikey": EXPECTED_KEY,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "pred_IC50": scored["pred_IC50_nM"], "pred_IC50_unit": "nM",
        "raw_predicted_pic50": scored["pred_pIC50"], "raw_activity_score": scored["activity_score"],
        "qsar_activity_ad_penalized": activity,
        "AD_similarity": scored["nearest_train_similarity"], "AD_category": scored["applicability_domain"],
        "SA_score": raw_sa, "SA_objective": panel["regularizer_score"],
        "BBB_score": panel["bbb_cns_score"], "CNS_score": panel["bbb_cns_score"],
        "regularizer_type": "sa", "objective_names": ";".join(objectives),
        "aggregated_score": aggregate, "stage2_score": aggregate, "selection_score": aggregate,
        "aggregated_score_4obj": geometric_mean([mapping[name] for name in FOUR_OBJECTIVES]),
        "in_qsar_training_set_exact_smiles": not train_matches.empty,
        "training_exact_match_count": len(train_matches),
        "source_input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
    })
    for name, value in mapping.items():
        row[f"objective_{name}"] = value
    row = annotate_clean_candidate_quality(row, config)
    numeric_keys = ["pred_pIC50", "pred_IC50", "MW", "cLogP", "TPSA", "HBD", "HBA",
                    "rotatable_bonds", "formal_charge", "QED", "SA_score", "SA_objective",
                    "BBB_score", "CNS_score", "qsar_activity_score_ad_penalized",
                    "AD_similarity", "logp_window_score", "tpsa_window_score", "stage2_score"]
    if not all(np.isfinite(float(row[key])) for key in numeric_keys):
        raise ValueError("A required property is missing or nonfinite.")

    # Reuse stored values: no generated molecule is rescored.
    comparisons = []
    for key in numeric_keys:
        values = pd.to_numeric(stage2[key], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite reference values in {key}")
        value = float(row[key])
        comparisons.append({"property": key, "MLi_2": value, "stage2_n": len(values),
                            "stage2_min": values.min(), "stage2_median": values.median(),
                            "stage2_mean": values.mean(), "stage2_max": values.max(),
                            "stage2_count_below_MLi2": int((values < value - 1e-10).sum()),
                            "stage2_count_equal_MLi2": int(np.isclose(values, value, rtol=0, atol=1e-10).sum()),
                            "stage2_count_above_MLi2": int((values > value + 1e-10).sum())})
    comparison = pd.DataFrame(comparisons)

    # Audit the aggregate using stored component scores; this also guards against
    # accidentally comparing a four-objective archive with this five-objective row.
    columns = ["qsar_activity_score_ad_penalized", "bbb_cns_score", "logp_window_score",
               "tpsa_window_score", "regularizer_score"]
    reconstructed = np.prod(stage2[columns].to_numpy(dtype=float), axis=1) ** 0.2
    aggregate_matches = np.isclose(reconstructed, stage2["stage2_score"], rtol=1e-9, atol=1e-10)
    if not aggregate_matches.all():
        raise ValueError("Archived Stage-2 aggregate does not match its five stored objectives.")
    current_ad = [module.ad_penalized_activity_score(a, d) for a, d in
                  zip(stage2["raw_activity_score"], stage2["AD_category"])]
    sa_matches = np.isclose(stage2["SA_objective"], (10.0 - stage2["SA_score"]) / 9.0,
                            rtol=1e-9, atol=1e-10)
    sa_discrepancies = stage2.loc[~sa_matches, ["compound_id", "canonical_smiles", "SA_score",
                                              "SA_objective", "generation_batch"]].copy()
    sa_discrepancies["SA_objective_from_exported_SA_score"] = (10.0 - sa_discrepancies["SA_score"]) / 9.0
    # These pre-structure current scores document drift; never replace archived scores.
    audit = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_csv": str(args.stage2_csv), "reference_rows": len(stage2),
        "stage2_sha256_before": stage2_hash, "stage2_sha256_after": sha256(args.stage2_csv),
        "model_sha256": model_hash, "model_matches_archived_run": True,
        "rdkit_matches_archived_run": True, "python": sys.version, "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__, "pandas": pd.__version__,
        "lightgbm": version("lightgbm"), "scikit_learn": version("scikit-learn"),
        "stage2_aggregate_rows_verified": int(aggregate_matches.sum()),
        "stage2_SA_normalization_discrepancy_count": len(sa_discrepancies),
        "stage2_AD_counts": {str(key): int(value) for key, value in stage2["AD_category"].value_counts().items()},
        "current_ad_before_structure_differs_from_archived_count": int((~np.isclose(
            current_ad, stage2["qsar_activity_score_ad_penalized"], rtol=1e-9, atol=1e-10)).sum()),
        "MLi2_AD_version_effect": "none: high AD in both versions; no structural penalty",
        "training_exact_matches": train_matches.to_dict("records"),
        "config": {key: config[key] for key in ["objective_names", "regularizer_type", "logp_window", "tpsa_window"]},
        "structure_reference": "https://pubchem.ncbi.nlm.nih.gov/compound/78319901",
        "verified_inchikey": EXPECTED_KEY,
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in [
            args.input, args.resolved_config, oracle_dir / "lrrk2_train_fingerprints.npy",
            oracle_dir / "lrrk2_train_smiles.csv", oracle_dir / "lrrk2_qsar_oracle.py",
            ROOT / "LRRK2case/lrrk2_qsar_mpo_utils.py",
            ROOT / "scripts/run_lrrk2_stage1_stage2_20k.py", Path(__file__).resolve()] if path.is_relative_to(ROOT)},
    }
    if audit["stage2_sha256_before"] != audit["stage2_sha256_after"]:
        raise ValueError("Reference table changed during calculation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Put shared Stage-2 property columns first, without inventing generation/rank metadata.
    ordered = [key for key in stage2.columns if key in row]
    ordered += [key for key in row if key not in ordered]
    pd.DataFrame([row], columns=ordered).to_csv(args.output_dir / "mli2_properties.csv", index=False)
    comparison.to_csv(args.output_dir / "mli2_vs_stage2_summary.csv", index=False)
    sa_discrepancies.to_csv(args.output_dir / "stage2_sa_consistency_findings.csv", index=False)
    (args.output_dir / "calculation_manifest.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "mli2.smi").write_text(f"{canonical}\tMLi-2\n")
    sdf_mol = Chem.Mol(mol)
    AllChem.Compute2DCoords(sdf_mol)
    sdf_mol.SetProp("_Name", "MLi-2")
    for key, value in row.items():
        sdf_mol.SetProp(key, str(value))
    with Chem.SDWriter(str(args.output_dir / "mli2_properties.sdf")) as writer:
        writer.write(sdf_mol)
    recovered = next(iter(Chem.SDMolSupplier(str(args.output_dir / "mli2_properties.sdf"))))
    if recovered is None or Chem.MolToInchiKey(recovered) != EXPECTED_KEY:
        raise ValueError("SDF round-trip changed MLi-2 stereochemistry.")
    print(json.dumps({key: row[key] for key in ["compound_id"] + numeric_keys}, indent=2), flush=True)
    print(f"Wrote MLi-2 outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
