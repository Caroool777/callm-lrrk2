#!/usr/bin/env python3
"""Replace nonphysical ring-alkyne structures for a 20k Maestro docking set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem

from run_lrrk2_regeneration import final_properties, stage2_sort_key, truthy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--stage2-csv", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_ring_triple(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid source SMILES: {smiles}")
    return any(
        bond.GetBondType() == Chem.BondType.TRIPLE and bond.IsInRing()
        for bond in mol.GetBonds()
    )


def main() -> int:
    args = parse_args()
    with args.final_csv.open("r", encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 20000:
        raise SystemExit(f"Expected 20000 original molecules, found {len(original)}")

    excluded = [
        row for row in original if has_ring_triple(row["canonical_isomeric_smiles"])
    ]
    retained = [
        row for row in original if not has_ring_triple(row["canonical_isomeric_smiles"])
    ]
    original_keys = {row["inchikey"] for row in original}
    if len(original_keys) != len(original):
        raise SystemExit("Original final CSV is not unique by InChIKey")

    eligible: list[dict[str, Any]] = []
    with args.stage2_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if not truthy(row.get("final_eligible")):
                continue
            if not row.get("inchikey") or row["inchikey"] in original_keys:
                continue
            if has_ring_triple(row["canonical_isomeric_smiles"]):
                continue
            eligible.append(row)
    eligible.sort(key=stage2_sort_key, reverse=True)
    if len(eligible) < len(excluded):
        raise SystemExit(
            f"Need {len(excluded)} replacements but only {len(eligible)} are eligible"
        )
    replacement_sources = eligible[: len(excluded)]

    seeds_payload = json.loads(args.seeds.read_text(encoding="utf-8"))
    validated_seeds = seeds_payload["seeds"]
    replacement_rows = final_properties(replacement_sources, validated_seeds)
    excluded.sort(key=lambda row: int(row["selection_rank"]))

    curated: list[dict[str, Any]] = []
    for row in retained:
        item = dict(row)
        item["source_final_molecule_id"] = row["molecule_id"]
        item["docking_library_action"] = "retained"
        item["replaces_invalid_molecule_id"] = ""
        curated.append(item)
    for removed, row in zip(excluded, replacement_rows):
        item = dict(row)
        item["source_final_molecule_id"] = ""
        item["docking_library_action"] = "stage2_ranked_replacement"
        item["replaces_invalid_molecule_id"] = removed["molecule_id"]
        curated.append(item)

    curated.sort(
        key=lambda row: (
            0 if row["docking_library_action"] == "retained" else 1,
            int(row.get("selection_rank") or 0),
        )
    )
    for rank, row in enumerate(curated, start=1):
        row["molecule_id"] = f"LRRK2_DOCK_{rank:05d}"
        row["selection_rank"] = rank

    keys = [str(row["inchikey"]) for row in curated]
    if len(curated) != 20000 or len(set(keys)) != 20000:
        raise SystemExit(
            f"Curated set failed size/uniqueness: count={len(curated)} unique={len(set(keys))}"
        )
    remaining_ring_triples = sum(
        has_ring_triple(str(row["canonical_isomeric_smiles"])) for row in curated
    )
    if remaining_ring_triples:
        raise SystemExit(f"Curated set still has {remaining_ring_triples} ring triple bonds")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(original[0]) + [
        "source_final_molecule_id",
        "docking_library_action",
        "replaces_invalid_molecule_id",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(curated)

    audit_fields = [
        "excluded_molecule_id",
        "excluded_inchikey",
        "excluded_smiles",
        "reason",
        "replacement_molecule_id",
        "replacement_source_scored_molecule_id",
        "replacement_inchikey",
        "replacement_smiles",
        "replacement_parent_seed",
        "replacement_stage2_score",
    ]
    with args.audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        replacements_by_removed = {
            row["replaces_invalid_molecule_id"]: row
            for row in curated
            if row["docking_library_action"] == "stage2_ranked_replacement"
        }
        for removed in excluded:
            replacement = replacements_by_removed[removed["molecule_id"]]
            writer.writerow(
                {
                    "excluded_molecule_id": removed["molecule_id"],
                    "excluded_inchikey": removed["inchikey"],
                    "excluded_smiles": removed["canonical_isomeric_smiles"],
                    "reason": "ring_triple_bond_nonphysical_3d",
                    "replacement_molecule_id": replacement["molecule_id"],
                    "replacement_source_scored_molecule_id": replacement[
                        "source_scored_molecule_id"
                    ],
                    "replacement_inchikey": replacement["inchikey"],
                    "replacement_smiles": replacement["canonical_isomeric_smiles"],
                    "replacement_parent_seed": replacement["parent_seed"],
                    "replacement_stage2_score": replacement["stage2_score"],
                }
            )

    summary = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_count": len(original),
        "excluded_count": len(excluded),
        "exclusion_reason": "ring_triple_bond_nonphysical_3d",
        "retained_count": len(retained),
        "replacement_count": len(replacement_rows),
        "curated_count": len(curated),
        "curated_unique_inchikeys": len(set(keys)),
        "remaining_ring_triple_bonds": remaining_ring_triples,
        "replacement_parent_distribution": dict(
            Counter(row["parent_seed"] for row in replacement_rows)
        ),
        "original_final_csv": str(args.final_csv.resolve()),
        "original_final_csv_sha256": file_sha256(args.final_csv),
        "stage2_csv": str(args.stage2_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": file_sha256(args.output_csv),
        "audit_csv": str(args.audit_csv.resolve()),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
