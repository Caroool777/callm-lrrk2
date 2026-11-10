#!/usr/bin/env python3
"""Repeatable A-S validator for the eight-seed LRRK2 Stage-2 docking export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_lrrk2_stage1_stage2_20k import FINAL_TARGET, validate_exports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    validated = pd.read_csv(
        run_dir / "inputs" / "lrrk2_stage1_seed_pool.csv"
    ).to_dict("records")
    selected = pd.read_csv(
        run_dir / "stage1" / "stage1_to_stage2_seeds.csv"
    ).to_dict("records")
    report = validate_exports(run_dir, validated, selected, FINAL_TARGET)
    print(f"A-S validation passed: {report['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

