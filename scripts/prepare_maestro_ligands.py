#!/usr/bin/env python3
"""Prepare validated LRRK2 structures for import into Schrodinger Maestro.

The script deliberately preserves each input graph, formal charge, tautomer, and
specified stereochemistry.  It creates both a clean 2D LigPrep input and a 3D,
explicit-hydrogen SDF for direct Maestro import.  The latter is an as-generated
state and is not a substitute for project-specific LigPrep/Epik enumeration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


DEFAULT_SEED = 20260820
DEFAULT_CHUNK_SIZE = 1000
REQUIRED_COLUMNS = {
    "molecule_id",
    "canonical_isomeric_smiles",
    "inchikey",
    "parent_seed",
    "stage1_score",
    "stage2_score",
}
METALS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete chunk files whose checkpoint says they succeeded.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identity(mol: Chem.Mol) -> tuple[str, str]:
    heavy = Chem.RemoveHs(mol)
    Chem.SanitizeMol(heavy)
    smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    # InChI can infer an otherwise unspecified stereo layer from 3D coordinates.
    # Identity here is the source molecular graph, so compute the key without a
    # conformer while retaining all explicitly assigned chiral tags.
    graph_only = Chem.Mol(heavy)
    graph_only.RemoveAllConformers()
    return smiles, Chem.MolToInchiKey(graph_only)


def validate_source(row: dict[str, str]) -> Chem.Mol:
    smiles = row["canonical_isomeric_smiles"].strip()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("parse_or_sanitize_failure")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multiple_fragments")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("dummy_or_wildcard_atom")
    if any(atom.GetAtomicNum() in METALS for atom in mol.GetAtoms()):
        raise ValueError("metal_atom")
    if any(
        bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing()
        for bond in mol.GetBonds()
    ):
        raise ValueError("ring_triple_bond")
    canonical, key = canonical_identity(mol)
    if canonical != smiles:
        raise ValueError(f"canonical_smiles_mismatch:{canonical}")
    if key != row["inchikey"].strip():
        raise ValueError(f"inchikey_mismatch:{key}")
    return mol


def set_properties(mol: Chem.Mol, row: dict[str, str], extra: dict[str, Any]) -> None:
    mol.SetProp("_Name", row["molecule_id"])
    for key, value in row.items():
        if value is not None:
            mol.SetProp(str(key), str(value))
    for key, value in extra.items():
        mol.SetProp(str(key), str(value))


def embed_3d(mol: Chem.Mol, seed: int) -> tuple[Chem.Mol, str]:
    mol_h = Chem.AddHs(mol, addCoords=False)
    attempts = (
        ("ETKDGv3", False, 0),
        ("ETKDGv3_random_coords", True, 9973),
        ("ETKDGv3_random_coords_retry", True, 19997),
    )
    selected = ""
    for label, random_coords, offset in attempts:
        mol_h.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int((seed + offset) % 2147483647)
        params.enforceChirality = True
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.useRandomCoords = random_coords
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            selected = label
            break
    if not selected:
        mol_h.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int((seed + 31337) % 2147483647)
        params.enforceChirality = True
        params.useRandomCoords = True
        params.ignoreSmoothingFailures = True
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False
        params.maxIterations = 1000
        params.boxSizeMult = 10.0
        params.numThreads = 1
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            selected = "ETKDGv3_relaxed_distance_geometry"
    if not selected:
        raise ValueError("3d_embedding_failure")

    force_field = "none"
    optimization_status = "not_attempted"
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        force_field = "MMFF94s"
        result = AllChem.MMFFOptimizeMolecule(
            mol_h, mmffVariant="MMFF94s", maxIters=500, nonBondedThresh=100.0
        )
        optimization_status = "converged" if result == 0 else "max_iterations"
    elif AllChem.UFFHasAllMoleculeParams(mol_h):
        force_field = "UFF"
        result = AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
        optimization_status = "converged" if result == 0 else "max_iterations"

    conformer = mol_h.GetConformer()
    conformer.Set3D(True)
    coordinates = conformer.GetPositions()
    if not all(math.isfinite(float(value)) for row in coordinates for value in row):
        raise ValueError("non_finite_3d_coordinates")
    return mol_h, f"{selected}|{force_field}|{optimization_status}"


def prepare_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    chunk_number = payload["chunk_number"]
    rows = payload["rows"]
    output_dir = Path(payload["output_dir"])
    random_seed = payload["random_seed"]
    expected_sha = payload["input_sha256"]
    chunk_path = output_dir / "chunks" / f"lrrk2_maestro_3d_{chunk_number:03d}.sdf"
    checkpoint_path = output_dir / "checkpoints" / f"chunk_{chunk_number:03d}.json"

    if payload["resume"] and chunk_path.exists() and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") == "complete"
            and checkpoint.get("input_sha256") == expected_sha
            and checkpoint.get("count") == len(rows)
            and checkpoint.get("output_sha256") == sha256(chunk_path)
        ):
            return checkpoint

    temporary = chunk_path.with_suffix(".sdf.partial")
    writer = Chem.SDWriter(str(temporary))
    manifest_rows: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    try:
        for local_index, row in enumerate(rows):
            molecule_id = row["molecule_id"]
            try:
                source = validate_source(row)
                molecule_seed = (random_seed + int(row["_global_index"])) % 2147483647
                mol_h, method = embed_3d(source, molecule_seed)
                canonical, key = canonical_identity(mol_h)
                if canonical != row["canonical_isomeric_smiles"] or key != row["inchikey"]:
                    raise ValueError("identity_changed_during_3d_preparation")
                embed_method, force_field, optimization_status = method.split("|")
                stereo = Chem.FindMolChiralCenters(
                    source, includeUnassigned=True, includeCIP=True
                )
                unassigned = sum(label == "?" for _, label in stereo)
                set_properties(
                    mol_h,
                    row,
                    {
                        "maestro_preparation_level": "rdkit_3d_as_generated_state",
                        "protonation_state_policy": "preserved_as_generated",
                        "tautomer_state_policy": "preserved_as_generated",
                        "stereochemistry_policy": "specified_preserved_unassigned_not_enumerated",
                        "unassigned_stereocenters_rdkit": unassigned,
                        "explicit_hydrogens": sum(
                            atom.GetAtomicNum() == 1 for atom in mol_h.GetAtoms()
                        ),
                        "embed_method": embed_method,
                        "force_field": force_field,
                        "optimization_status": optimization_status,
                        "preparation_random_seed": molecule_seed,
                    },
                )
                writer.write(mol_h)
                stats[f"embed_method:{embed_method}"] += 1
                stats[f"force_field:{force_field}"] += 1
                stats[f"optimization:{optimization_status}"] += 1
                stats["success"] += 1
                manifest_rows.append(
                    {
                        "molecule_id": molecule_id,
                        "status": "success",
                        "embed_method": embed_method,
                        "force_field": force_field,
                        "optimization_status": optimization_status,
                        "unassigned_stereocenters_rdkit": str(unassigned),
                        "reason": "",
                    }
                )
            except Exception as exc:
                failures.append({"molecule_id": molecule_id, "reason": str(exc)})
                manifest_rows.append(
                    {
                        "molecule_id": molecule_id,
                        "status": "failed",
                        "embed_method": "",
                        "force_field": "",
                        "optimization_status": "",
                        "unassigned_stereocenters_rdkit": "",
                        "reason": str(exc),
                    }
                )
                stats["failed"] += 1
    finally:
        writer.close()

    if failures:
        temporary.unlink(missing_ok=True)
        result = {
            "status": "failed",
            "chunk_number": chunk_number,
            "count": len(rows),
            "success": stats["success"],
            "failures": failures,
            "input_sha256": expected_sha,
            "manifest_rows": manifest_rows,
        }
        checkpoint_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    os.replace(temporary, chunk_path)
    result = {
        "status": "complete",
        "chunk_number": chunk_number,
        "count": len(rows),
        "success": stats["success"],
        "stats": dict(stats),
        "input_sha256": expected_sha,
        "output_sha256": sha256(chunk_path),
        "manifest_rows": manifest_rows,
    }
    checkpoint_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_2d_input(rows: list[dict[str, str]], output_path: Path) -> None:
    temporary = output_path.with_suffix(".sdf.partial")
    writer = Chem.SDWriter(str(temporary))
    try:
        for row in rows:
            mol = validate_source(row)
            AllChem.Compute2DCoords(mol)
            set_properties(
                mol,
                row,
                {
                    "maestro_preparation_level": "clean_2d_ligprep_input",
                    "protonation_state_policy": "preserved_as_generated",
                    "tautomer_state_policy": "preserved_as_generated",
                    "stereochemistry_policy": "specified_preserved_unassigned_not_enumerated",
                },
            )
            writer.write(mol)
    finally:
        writer.close()
    os.replace(temporary, output_path)


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise SystemExit("--workers and --chunk-size must be positive")
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != 20000:
        raise SystemExit(f"Expected 20000 source molecules, found {len(rows)}")
    if len({row["molecule_id"] for row in rows}) != len(rows):
        raise SystemExit("Source molecule_id values are not unique")

    input_sha = sha256(input_csv)
    for index, row in enumerate(rows):
        row["_global_index"] = str(index)
        validate_source(row)

    two_d_path = output_dir / "lrrk2_maestro_ligprep_input_20000.sdf"
    if not args.resume or not two_d_path.exists():
        write_2d_input(rows, two_d_path)

    payloads = []
    for start in range(0, len(rows), args.chunk_size):
        chunk_number = start // args.chunk_size + 1
        payloads.append(
            {
                "chunk_number": chunk_number,
                "rows": rows[start : start + args.chunk_size],
                "output_dir": str(output_dir),
                "random_seed": args.random_seed,
                "input_sha256": input_sha,
                "resume": args.resume,
            }
        )

    print(
        f"Preparing {len(rows)} molecules in {len(payloads)} chunks "
        f"with {args.workers} workers",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_chunk, payload): payload for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"chunk {result['chunk_number']:03d}: {result['status']} "
                f"({result.get('success', 0)}/{result['count']})",
                flush=True,
            )
    results.sort(key=lambda item: item["chunk_number"])
    failures = [result for result in results if result["status"] != "complete"]
    if failures:
        compact = [
            {
                "chunk_number": result["chunk_number"],
                "count": result["count"],
                "success": result.get("success", 0),
                "failures": result.get("failures", []),
            }
            for result in failures
        ]
        print(json.dumps(compact, indent=2), file=sys.stderr)
        return 2

    combined_path = output_dir / "lrrk2_maestro_3d_as_generated_20000.sdf"
    temporary_combined = combined_path.with_suffix(".sdf.partial")
    with temporary_combined.open("wb") as destination:
        for result in results:
            source = output_dir / "chunks" / (
                f"lrrk2_maestro_3d_{result['chunk_number']:03d}.sdf"
            )
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, destination, length=1024 * 1024)
    os.replace(temporary_combined, combined_path)

    manifest_path = output_dir / "maestro_preparation_manifest.csv"
    fields = [
        "molecule_id",
        "status",
        "embed_method",
        "force_field",
        "optimization_status",
        "unassigned_stereocenters_rdkit",
        "reason",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["manifest_rows"])

    aggregate: Counter[str] = Counter()
    for result in results:
        aggregate.update(result.get("stats", {}))
    summary = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(input_csv),
        "source_csv_sha256": input_sha,
        "random_seed": args.random_seed,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "molecule_count": len(rows),
        "chunk_count": len(results),
        "policy": {
            "graph": "preserved",
            "formal_charge": "preserved_as_generated",
            "tautomer": "preserved_as_generated",
            "specified_stereochemistry": "preserved",
            "unassigned_stereochemistry": "not_enumerated",
            "protonation_enumeration": "not_performed_use_LigPrep_for_project_pH",
        },
        "statistics": dict(aggregate),
        "files": {
            "ligprep_input_2d_sdf": str(two_d_path),
            "maestro_import_3d_sdf": str(combined_path),
            "manifest_csv": str(manifest_path),
            "chunks_directory": str(output_dir / "chunks"),
        },
        "sha256": {
            two_d_path.name: sha256(two_d_path),
            combined_path.name: sha256(combined_path),
            manifest_path.name: sha256(manifest_path),
        },
        "rdkit_version": rdBase.rdkitVersion,
        "python_version": sys.version,
    }
    (output_dir / "maestro_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Complete: {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
