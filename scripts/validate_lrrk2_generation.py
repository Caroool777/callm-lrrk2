#!/usr/bin/env python3
"""Validate all acceptance conditions for the five-seed LRRK2 regeneration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_lrrk2_regeneration import (  # noqa: E402
    FINAL_TARGET,
    MASTER_SEED,
    SEEDS,
    stage2_sort_key,
    standardize_and_qc,
)

rdBase.DisableLog("rdApp.warning")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_check(checks: List[Dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def parse_smi(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_sdf(path: Path) -> List[Dict[str, str]]:
    records = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for index, mol in enumerate(supplier, start=1):
        if mol is None:
            records.append({"_invalid_index": str(index)})
            continue
        try:
            Chem.SanitizeMol(mol)
            props = mol.GetPropsAsDict(includePrivate=True, includeComputed=False)
            props["_structure_smiles"] = Chem.MolToSmiles(
                Chem.RemoveHs(mol), canonical=True, isomericSmiles=True
            )
            records.append({key: str(value) for key, value in props.items()})
        except Exception:
            records.append({"_invalid_index": str(index)})
    return records


def top_stage2_inchikeys(path: Path, target: int) -> List[str]:
    eligible = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("final_eligible", "")).strip().lower() == "true":
                eligible.append(row)
    eligible.sort(key=stage2_sort_key, reverse=True)
    return [row["inchikey"] for row in eligible[:target]]


def validate(run_dir: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    csv_path = run_dir / "lrrk2_final_20000.csv"
    smi_path = run_dir / "lrrk2_final_20000.smi"
    sdf_path = run_dir / "lrrk2_final_20000.sdf"
    stage1_raw = run_dir / "stage1_raw.smi"
    stage1_valid = run_dir / "stage1_valid.smi"
    stage2_all = run_dir / "stage2_all_scored.csv"

    required_paths = [
        run_dir / "seeds.yaml",
        run_dir / "resolved_config.json",
        stage1_raw,
        stage1_valid,
        stage2_all,
        csv_path,
        smi_path,
        sdf_path,
        run_dir / "generation_statistics.json",
        run_dir / "rejection_reasons.csv",
        run_dir / "run.log",
        run_dir / "quality_report.md",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    add_check(checks, "required_output_files", not missing, f"missing={missing}")
    if missing:
        return {"passed": False, "checks": checks}

    frame = pd.read_csv(csv_path)
    smi_rows = parse_smi(smi_path)
    sdf_rows = parse_sdf(sdf_path)

    add_check(checks, "csv_count_20000", len(frame) == FINAL_TARGET, str(len(frame)))
    add_check(checks, "smi_count_20000", len(smi_rows) == FINAL_TARGET, str(len(smi_rows)))
    add_check(checks, "sdf_count_20000", len(sdf_rows) == FINAL_TARGET, str(len(sdf_rows)))

    required_columns = {
        "molecule_id",
        "canonical_isomeric_smiles",
        "inchikey",
        "parent_seed",
        "generation_batch",
        "stage1_score",
        "stage2_score",
        "LRRK2_qsar_activity_ad_penalized",
        "MW",
        "cLogP",
        "TPSA",
        "HBD",
        "HBA",
        "rotatable_bonds",
        "formal_charge",
        "ring_count",
        "QED",
        "SA_score",
        "structural_alerts",
        "stereochemistry_status",
        "max_similarity_to_any_seed",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    add_check(checks, "required_csv_fields", not missing_columns, f"missing={missing_columns}")

    csv_ids = frame["molecule_id"].astype(str).tolist()
    smi_ids = [row.get("molecule_id", "") for row in smi_rows]
    sdf_ids = [row.get("molecule_id", "") for row in sdf_rows]
    add_check(
        checks,
        "molecule_ids_identical",
        csv_ids == smi_ids == sdf_ids,
        f"csv={len(set(csv_ids))} smi={len(set(smi_ids))} sdf={len(set(sdf_ids))}",
    )

    valid_count = 0
    standardized_keys = []
    canonical_mismatches = 0
    inchikey_mismatches = 0
    multifragment = 0
    invalid_indices = []
    for index, row in frame.iterrows():
        smiles = str(row.get("canonical_isomeric_smiles") or "")
        mol, canonical, key, reasons = standardize_and_qc(smiles)
        if reasons or mol is None:
            invalid_indices.append((int(index), reasons))
            continue
        valid_count += 1
        standardized_keys.append(key)
        if canonical != smiles:
            canonical_mismatches += 1
        if key != str(row.get("inchikey") or ""):
            inchikey_mismatches += 1
        if "." in smiles:
            multifragment += 1
    add_check(
        checks,
        "all_rdkit_parse_and_sanitize",
        valid_count == FINAL_TARGET,
        f"valid={valid_count} invalid_sample={invalid_indices[:5]}",
    )
    add_check(
        checks,
        "canonical_smiles_match",
        canonical_mismatches == 0,
        f"mismatches={canonical_mismatches}",
    )
    add_check(
        checks,
        "inchikeys_match_standardization",
        inchikey_mismatches == 0,
        f"mismatches={inchikey_mismatches}",
    )
    add_check(
        checks,
        "unique_standardized_inchikeys",
        len(set(standardized_keys)) == FINAL_TARGET,
        f"unique={len(set(standardized_keys))}",
    )
    add_check(checks, "no_multifragment_errors", multifragment == 0, f"count={multifragment}")
    add_check(
        checks,
        "no_empty_smiles",
        frame["canonical_isomeric_smiles"].fillna("").astype(str).str.strip().ne("").all(),
        f"empty={int(frame['canonical_isomeric_smiles'].fillna('').astype(str).str.strip().eq('').sum())}",
    )

    seeds_payload = read_json(run_dir / "seeds.yaml")
    seed_rows = seeds_payload.get("seeds", [])
    seed_names = {seed["name"] for seed in seed_rows}
    seed_keys = {seed["inchikey"] for seed in seed_rows}
    expected_names = {seed["name"] for seed in SEEDS}
    weight_sum = sum(float(seed["weight"]) for seed in seed_rows)
    add_check(
        checks,
        "five_new_seeds_only",
        seed_names == expected_names and len(seed_rows) == 5,
        f"names={sorted(seed_names)}",
    )
    add_check(
        checks,
        "seed_weight_sum_1",
        math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12),
        f"sum={weight_sum:.12f}",
    )
    exact_seed_count = len(set(standardized_keys) & seed_keys)
    add_check(checks, "no_exact_seed_copy", exact_seed_count == 0, f"count={exact_seed_count}")

    old_keys_path = run_dir / "old_output_inchikeys.txt"
    old_keys = {
        line.strip()
        for line in old_keys_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    old_overlap = len(set(standardized_keys) & old_keys)
    add_check(checks, "no_previous_output_overlap", old_overlap == 0, f"count={old_overlap}")

    parent_values = set(frame["parent_seed"].dropna().astype(str))
    parent_missing = int(frame["parent_seed"].isna().sum()) + int(
        frame["parent_seed"].fillna("").astype(str).str.strip().eq("").sum()
    )
    add_check(
        checks,
        "parent_seed_complete_and_valid",
        parent_missing == 0 and parent_values.issubset(expected_names),
        f"missing={parent_missing} values={sorted(parent_values)}",
    )

    resolved = read_json(run_dir / "resolved_config.json")
    resolved_text = json.dumps(resolved, ensure_ascii=False)
    old_parent = "C=Cc1cc2c(cc1-c1cncc(C(N)=O)c1F)CN(C)C2=C"
    config_clean = old_parent not in resolved_text and "250k_rndm_zinc" not in resolved_text
    batch_config_issues = []
    for path in run_dir.glob("batches/**/batch_config.json"):
        text = path.read_text(encoding="utf-8")
        if old_parent in text or "250k_rndm_zinc" in text:
            batch_config_issues.append(str(path))
    add_check(
        checks,
        "no_old_seed_in_configs",
        config_clean and not batch_config_issues,
        f"batch_issues={batch_config_issues}",
    )
    allocations = resolved.get("stage1_actual_parent_allocations", {})
    add_check(
        checks,
        "stage1_probability_allocations_total_20000",
        sum(int(value) for value in allocations.values()) == FINAL_TARGET
        and set(allocations) == expected_names,
        f"allocations={allocations}",
    )
    add_check(
        checks,
        "master_random_seed",
        int(resolved.get("random_seed", -1)) == MASTER_SEED,
        str(resolved.get("random_seed")),
    )

    sdf_invalid = sum("_invalid_index" in row for row in sdf_rows)
    sdf_map = {row.get("molecule_id", ""): row for row in sdf_rows if "molecule_id" in row}
    sdf_property_mismatches = 0
    structure_mismatches = 0
    for _, row in frame.iterrows():
        molecule_id = str(row["molecule_id"])
        sdf = sdf_map.get(molecule_id)
        if sdf is None:
            sdf_property_mismatches += 1
            continue
        for key in ["parent_seed", "generation_batch", "inchikey"]:
            if str(row[key]) != str(sdf.get(key, "")):
                sdf_property_mismatches += 1
                break
        if abs(float(row["stage2_score"]) - float(sdf.get("stage2_score", "nan"))) > 1e-8:
            sdf_property_mismatches += 1
        _, sdf_canonical, _, sdf_reasons = standardize_and_qc(sdf.get("_structure_smiles", ""))
        if sdf_reasons or sdf_canonical != str(row["canonical_isomeric_smiles"]):
            structure_mismatches += 1
    add_check(checks, "sdf_all_valid", sdf_invalid == 0, f"invalid={sdf_invalid}")
    add_check(
        checks,
        "sdf_csv_properties_correspond",
        sdf_property_mismatches == 0,
        f"mismatches={sdf_property_mismatches}",
    )
    add_check(
        checks,
        "sdf_csv_structures_correspond",
        structure_mismatches == 0,
        f"mismatches={structure_mismatches}",
    )

    stage1_manifest = read_json(run_dir / "checkpoints" / "stage1_manifest.json")
    stage2_manifest = read_json(run_dir / "checkpoints" / "stage2_manifest.json")
    stage1_checkpoint_paths = list(
        run_dir.glob("batches/stage1_*/checkpoints/checkpoint_summary.json")
    )
    incomplete_stage1 = []
    for path in stage1_checkpoint_paths:
        summary = read_json(path)
        if not summary.get("complete"):
            incomplete_stage1.append(str(path))

    stage2_specs = read_json(run_dir / "checkpoints" / "stage2_batch_specs.json")
    incomplete_stage2 = []
    official_stage2_outputs = set()
    for spec in stage2_specs:
        output_dir = Path(spec["output"]).resolve()
        official_stage2_outputs.add(output_dir)
        checkpoint_path = output_dir / "checkpoints" / "stage_checkpoint.pkl"
        summary_path = output_dir / "checkpoints" / "checkpoint_summary.json"
        summary = read_json(summary_path)
        if (
            not checkpoint_path.exists()
            or not summary.get("complete")
            or int(summary.get("current_oracle_calls", 0)) < int(spec["target"])
        ):
            incomplete_stage2.append(str(output_dir))

    stage1_batch_count = len(stage1_checkpoint_paths)
    stage2_batch_count = len(stage2_specs)
    checkpoint_pass = (
        stage1_manifest.get("complete")
        and stage2_manifest.get("complete")
        and stage1_batch_count == 5
        and stage2_batch_count >= 5
        and not incomplete_stage1
        and not incomplete_stage2
    )
    add_check(
        checks,
        "stage_checkpoints_complete",
        checkpoint_pass,
        (
            f"stage1={stage1_batch_count} official_stage2={stage2_batch_count} "
            f"incomplete_stage1={len(incomplete_stage1)} "
            f"incomplete_official_stage2={len(incomplete_stage2)}"
        ),
    )

    superseded_output = (
        run_dir / "batches" / "stage2_round002_lrrk2_compound18_yxd"
    ).resolve()
    add_check(
        checks,
        "superseded_partial_excluded_from_manifest",
        superseded_output not in official_stage2_outputs,
        f"superseded={superseded_output} official_stage2={stage2_batch_count}",
    )

    top_keys = top_stage2_inchikeys(stage2_all, FINAL_TARGET)
    selected_keys = frame["inchikey"].astype(str).tolist()
    add_check(
        checks,
        "selected_by_stage2_score_not_file_order",
        selected_keys == top_keys,
        f"selected={len(selected_keys)} ranked={len(top_keys)}",
    )
    score_order = frame["stage2_score"].astype(float).tolist()
    add_check(
        checks,
        "final_selection_rank_consistent",
        frame["selection_rank"].astype(int).tolist() == list(range(1, FINAL_TARGET + 1))
        and all(score_order[index] >= score_order[index + 1] - 1e-15 for index in range(len(score_order) - 1)),
        "selection_rank is sequential and primary stage2 score is non-increasing",
    )

    threshold_failures = {
        "stage2_score": int((frame["stage2_score"].astype(float) < 0.85 - 1e-12).sum()),
        "activity": int((frame["LRRK2_qsar_activity_ad_penalized"].astype(float) < 0.55 - 1e-12).sum()),
        "bbb": int((frame["bbb_cns_score"].astype(float) < 0.8 - 1e-12).sum()),
        "logp_window": int((frame["logp_window_score"].astype(float) < 0.8 - 1e-12).sum()),
        "tpsa_window": int((frame["tpsa_window_score"].astype(float) < 0.8 - 1e-12).sum()),
    }
    add_check(
        checks,
        "existing_stage2_thresholds_preserved",
        not any(threshold_failures.values()),
        str(threshold_failures),
    )

    statistics = read_json(run_dir / "generation_statistics.json")
    add_check(
        checks,
        "statistics_final_count",
        statistics.get("final_count") == FINAL_TARGET
        and statistics.get("final_unique_inchikeys") == FINAL_TARGET,
        f"final={statistics.get('final_count')} unique={statistics.get('final_unique_inchikeys')}",
    )

    passed = all(check["passed"] for check in checks)
    result = {
        "passed": passed,
        "run_dir": str(run_dir),
        "rdkit_version": rdBase.rdkitVersion,
        "final_count": len(frame),
        "unique_inchikeys": len(set(standardized_keys)),
        "parent_contributions": dict(Counter(frame["parent_seed"].astype(str))),
        "checks": checks,
    }
    return result


def write_results(run_dir: Path, result: Dict[str, Any]) -> None:
    output_path = run_dir / "acceptance_results.json"
    temporary = output_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.flush()
    temporary.replace(output_path)

    lines = [
        "",
        "## Automatic Acceptance Validation",
        "",
        f"- Overall: {'PASS' if result['passed'] else 'FAIL'}",
        f"- Final count: {result.get('final_count', 0):,}",
        f"- Unique standardized InChIKeys: {result.get('unique_inchikeys', 0):,}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        detail = str(check["detail"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {detail} |")
    report_path = run_dir / "quality_report.md"
    marker = "\n## Automatic Acceptance Validation"
    existing_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in existing_report:
        existing_report = existing_report.split(marker, 1)[0].rstrip()
    report_path.write_text(
        existing_report + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write("\nAUTOMATIC ACCEPTANCE RESULT: " + ("PASS" if result["passed"] else "FAIL") + "\n")
        for check in result["checks"]:
            handle.write(
                f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['detail']}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / "runs" / "lrrk2_regeneration_5seeds_20000"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    result = validate(run_dir)
    write_results(run_dir, result)
    print(f"Acceptance result: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"Results: {run_dir / 'acceptance_results.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

