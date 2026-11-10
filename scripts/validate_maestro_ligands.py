#!/usr/bin/env python3
"""Validate the clean 2D and explicit-H 3D Maestro import SDF files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--maestro-dir", type=Path, required=True)
    return parser.parse_args()


def check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def inspect_sdf(path: Path, require_3d: bool) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    invalid: list[str] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for index, mol in enumerate(supplier, start=1):
        if mol is None:
            invalid.append(f"record_{index}:parse_failure")
            continue
        try:
            Chem.SanitizeMol(mol)
            molecule_id = mol.GetProp("molecule_id") if mol.HasProp("molecule_id") else ""
            if not molecule_id:
                invalid.append(f"record_{index}:missing_molecule_id")
                continue
            if len(Chem.GetMolFrags(Chem.RemoveHs(mol))) != 1:
                invalid.append(f"{molecule_id}:multiple_fragments")
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                invalid.append(f"{molecule_id}:dummy_atom")
            if any(
                bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing()
                for bond in mol.GetBonds()
            ):
                invalid.append(f"{molecule_id}:ring_triple_bond")
            heavy = Chem.RemoveHs(mol)
            canonical = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
            graph_only = Chem.Mol(heavy)
            graph_only.RemoveAllConformers()
            key = Chem.MolToInchiKey(graph_only)
            record = {
                "molecule_id": molecule_id,
                "canonical_isomeric_smiles": canonical,
                "inchikey": key,
                "source_inchikey_property": (
                    mol.GetProp("inchikey") if mol.HasProp("inchikey") else ""
                ),
                "parent_seed": mol.GetProp("parent_seed") if mol.HasProp("parent_seed") else "",
                "stage1_score": mol.GetProp("stage1_score") if mol.HasProp("stage1_score") else "",
                "stage2_score": mol.GetProp("stage2_score") if mol.HasProp("stage2_score") else "",
            }
            records.append(record)
            counts["valid"] += 1
            explicit_h = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
            counts["with_explicit_h"] += int(explicit_h > 0)
            if mol.GetNumConformers():
                conformer = mol.GetConformer()
                counts["with_conformer"] += 1
                counts["with_3d_flag"] += int(conformer.Is3D())
                coords = conformer.GetPositions()
                finite = all(
                    math.isfinite(float(value)) for row in coords for value in row
                )
                counts["with_finite_coordinates"] += int(finite)
            if require_3d and explicit_h == 0:
                invalid.append(f"{molecule_id}:no_explicit_hydrogens")
            if require_3d and (not mol.GetNumConformers() or not mol.GetConformer().Is3D()):
                invalid.append(f"{molecule_id}:no_3d_conformer")
        except Exception as exc:
            invalid.append(f"record_{index}:{exc}")
    return {"records": records, "counts": dict(counts), "invalid": invalid}


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.maestro_dir.resolve()
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    source_ids = [row["molecule_id"] for row in source]
    source_keys = [row["inchikey"] for row in source]
    source_smiles = [row["canonical_isomeric_smiles"] for row in source]

    path_2d = output_dir / "lrrk2_maestro_ligprep_input_20000.sdf"
    path_3d = output_dir / "lrrk2_maestro_3d_as_generated_20000.sdf"
    chunks = sorted((output_dir / "chunks").glob("lrrk2_maestro_3d_*.sdf"))
    checks: list[dict[str, Any]] = []
    check(checks, "required_files_exist", path_2d.exists() and path_3d.exists(), f"2d={path_2d.exists()} 3d={path_3d.exists()}")
    check(checks, "source_count_20000", len(source) == 20000, str(len(source)))

    result_2d = inspect_sdf(path_2d, require_3d=False)
    result_3d = inspect_sdf(path_3d, require_3d=True)
    rows_2d = result_2d["records"]
    rows_3d = result_3d["records"]
    check(checks, "clean_2d_count_20000", len(rows_2d) == 20000, str(len(rows_2d)))
    check(checks, "clean_3d_count_20000", len(rows_3d) == 20000, str(len(rows_3d)))
    check(checks, "clean_2d_all_valid", not result_2d["invalid"], str(result_2d["invalid"][:5]))
    check(checks, "clean_3d_all_valid", not result_3d["invalid"], str(result_3d["invalid"][:5]))
    ids_2d = [row["molecule_id"] for row in rows_2d]
    ids_3d = [row["molecule_id"] for row in rows_3d]
    check(checks, "ids_match_source_and_order", source_ids == ids_2d == ids_3d, f"unique_3d={len(set(ids_3d))}")
    keys_2d = [row["inchikey"] for row in rows_2d]
    keys_3d = [row["inchikey"] for row in rows_3d]
    check(
        checks,
        "ligprep_input_inchikeys_preserved_and_unique",
        source_keys == keys_2d and len(set(keys_2d)) == 20000,
        f"unique={len(set(keys_2d))}",
    )
    source_key_props_3d = [row["source_inchikey_property"] for row in rows_3d]
    check(
        checks,
        "3d_source_identity_properties_preserved",
        source_keys == source_key_props_3d,
        "SDF inchikey properties match the curated source",
    )
    smiles_2d = [row["canonical_isomeric_smiles"] for row in rows_2d]
    smiles_3d = [row["canonical_isomeric_smiles"] for row in rows_3d]
    check(
        checks,
        "ligprep_input_canonical_stereo_preserved",
        source_smiles == smiles_2d,
        "2D LigPrep input exactly matches source CSV",
    )
    source_connectivity = [
        Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False
        )
        for smiles in source_smiles
    ]
    geometry_connectivity = [
        Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False
        )
        for smiles in smiles_3d
    ]
    check(
        checks,
        "3d_connectivity_preserved",
        source_connectivity == geometry_connectivity,
        "3D geometry has the same non-stereo molecular graphs",
    )
    specified_stereo_preserved = sum(
        Chem.MolFromSmiles(target).HasSubstructMatch(
            Chem.MolFromSmiles(source_smiles[index]), useChirality=True
        )
        for index, target in enumerate(smiles_3d)
    )
    check(
        checks,
        "3d_specified_stereochemistry_preserved",
        specified_stereo_preserved == 20000,
        f"preserved={specified_stereo_preserved}",
    )
    geometry_resolved_stereo = sum(
        source_smiles[index] != smiles for index, smiles in enumerate(smiles_3d)
    )
    parent_ok = all(
        row["parent_seed"] == source[index]["parent_seed"]
        for index, row in enumerate(rows_3d)
    )
    scores_ok = all(
        row["stage1_score"] == source[index]["stage1_score"]
        and row["stage2_score"] == source[index]["stage2_score"]
        for index, row in enumerate(rows_3d)
    )
    check(checks, "parent_seed_properties_preserved", parent_ok, str(parent_ok))
    check(checks, "stage_scores_preserved", scores_ok, str(scores_ok))
    counts_3d = result_3d["counts"]
    check(checks, "all_have_explicit_hydrogens", counts_3d.get("with_explicit_h", 0) == 20000, str(counts_3d.get("with_explicit_h", 0)))
    check(checks, "all_have_3d_coordinates", counts_3d.get("with_3d_flag", 0) == 20000 and counts_3d.get("with_finite_coordinates", 0) == 20000, str(counts_3d))

    chunk_records: list[dict[str, str]] = []
    chunk_sizes = []
    chunk_invalid = []
    for chunk in chunks:
        inspected = inspect_sdf(chunk, require_3d=True)
        chunk_records.extend(inspected["records"])
        chunk_sizes.append(len(inspected["records"]))
        chunk_invalid.extend(inspected["invalid"])
    check(checks, "twenty_chunks_of_1000", len(chunks) == 20 and chunk_sizes == [1000] * 20, f"chunks={len(chunks)} sizes={chunk_sizes}")
    check(checks, "chunks_all_valid", not chunk_invalid, str(chunk_invalid[:5]))
    check(checks, "chunks_match_combined_3d", [row["molecule_id"] for row in chunk_records] == ids_3d, f"chunk_records={len(chunk_records)}")

    manifest_path = output_dir / "maestro_preparation_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    check(checks, "manifest_20000_success", len(manifest) == 20000 and all(row["status"] == "success" for row in manifest), f"rows={len(manifest)} failed={sum(row['status'] != 'success' for row in manifest)}")
    check(checks, "manifest_ids_match", [row["molecule_id"] for row in manifest] == source_ids, "manifest order matches source")

    passed = all(item["passed"] for item in checks)
    report = {
        "passed": passed,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdBase.rdkitVersion,
        "checks": checks,
        "counts_2d": result_2d["counts"],
        "counts_3d": result_3d["counts"],
        "3d_geometry_resolved_unspecified_stereo_count": geometry_resolved_stereo,
        "3d_geometry_inferred_unique_inchikeys": len(set(keys_3d)),
    }
    (output_dir / "validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# Maestro ligand preparation validation",
        "",
        f"Overall: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        detail = item["detail"].replace("|", "\\|")
        markdown.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | {detail} |")
    markdown.extend(
        [
            "",
            f"3D geometry realizes previously unspecified stereochemistry in {geometry_resolved_stereo} records; those coordinates contain {len(set(keys_3d))} inferred full InChIKeys.",
            "",
            "## Usage note",
            "",
            "Use `lrrk2_maestro_ligprep_input_20000.sdf` as the canonical Maestro import and LigPrep/Epik input. It preserves all 20,000 unique source identities exactly. The 3D SDF is a coordinate preview: three-dimensional geometry necessarily realizes one arbitrary orientation at unspecified stereocenters, so it is not a production docking-state enumeration.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
