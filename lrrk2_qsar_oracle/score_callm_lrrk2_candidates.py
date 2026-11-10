from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from lrrk2_qsar_oracle import DEFAULT_MODEL_DIR, score_dataframe


CALLM_SCORE_COLUMN = "qsar_activity_score_ad_penalized"
CALLM_RANK_COLUMN = "callm_lrrk2_rank"
CALLM_INPUT_ROW_COLUMN = "callm_input_row"

SMILES_COLUMN_PRIORITY = [
    "SMILES",
    "smiles",
    "canonical_SMILES",
    "canonical_smiles",
    "mol",
]


def detect_smiles_column(columns: list[str]) -> str:
    """Detect the most likely SMILES column for CALLM candidate files."""
    for column in SMILES_COLUMN_PRIORITY:
        if column in columns:
            return column
    raise ValueError(
        "Could not identify a SMILES column. Expected one of: "
        + ", ".join(SMILES_COLUMN_PRIORITY)
    )


def detect_name_column(columns: list[str]) -> str | None:
    """Prefer common molecule name columns when available."""
    if "Name" in columns:
        return "Name"
    if "name" in columns:
        return "name"
    return None


def rank_scored_candidates(
    scored: pd.DataFrame,
    rank_by: str = CALLM_SCORE_COLUMN,
) -> pd.DataFrame:
    """Sort candidates for CALLM selection and add a deterministic 1-based rank.

    Ranking is descending by the selected score column. Ties are resolved by
    nearest training-set similarity and QED when those columns are available.
    A stable sort is used so repeated runs are deterministic.
    """
    if rank_by not in scored.columns:
        raise ValueError(f"Cannot rank by missing column: {rank_by}")

    ranked = scored.copy()
    if CALLM_RANK_COLUMN in ranked.columns:
        ranked = ranked.drop(columns=[CALLM_RANK_COLUMN])

    # Numeric coercion makes invalid/NaN scores reliably sort last. Ties are
    # resolved by nearest training-set similarity and QED when available.
    ranked["_callm_rank_score"] = pd.to_numeric(ranked[rank_by], errors="coerce")

    sort_columns = ["_callm_rank_score"]
    ascending = [False]
    for tie_breaker in ["nearest_train_similarity", "QED"]:
        if tie_breaker in ranked.columns:
            helper_col = f"_callm_rank_{tie_breaker}"
            ranked[helper_col] = pd.to_numeric(ranked[tie_breaker], errors="coerce")
            sort_columns.append(helper_col)
            ascending.append(False)

    ranked = ranked.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    ranked = ranked.drop(columns=[col for col in ranked.columns if col.startswith("_callm_rank_")])
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, CALLM_RANK_COLUMN, range(1, len(ranked) + 1))
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and rank CALLM-generated LRRK2 candidate molecules."
    )
    parser.add_argument("--input", required=True, help="CALLM candidate CSV file.")
    parser.add_argument(
        "--output",
        default="callm_candidates_scored.csv",
        help="Output scored CSV file. Default: callm_candidates_scored.csv.",
    )
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Oracle artifact directory. Default: lrrk2_qsar_oracle/.",
    )
    parser.add_argument(
        "--rank_by",
        default=CALLM_SCORE_COLUMN,
        help=(
            "Column used for descending ranking. "
            f"Default: {CALLM_SCORE_COLUMN}."
        ),
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help=(
            "Preserve input row order and do not add the callm_lrrk2_rank column. "
            "This is useful when CALLM expects output rows to match input rows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, keep_default_na=False)
    smiles_col = detect_smiles_column(list(df.columns))
    name_col = detect_name_column(list(df.columns))

    scored = score_dataframe(
        df,
        smiles_col=smiles_col,
        name_col=name_col,
        model_dir=args.model_dir,
    )
    if CALLM_INPUT_ROW_COLUMN not in scored.columns:
        scored.insert(0, CALLM_INPUT_ROW_COLUMN, range(1, len(scored) + 1))
    if not args.no_sort:
        scored = rank_scored_candidates(scored, rank_by=args.rank_by)
    scored.to_csv(output_path, index=False)

    valid_count = int(scored["valid_smiles"].astype(bool).sum())
    invalid_count = int((~scored["valid_smiles"].astype(bool)).sum())
    ad_distribution = scored["applicability_domain"].value_counts(dropna=False)
    penalized = pd.to_numeric(scored[CALLM_SCORE_COLUMN], errors="coerce")

    print("CALLM LRRK2 candidate scoring completed")
    print(f"Input path: {input_path.resolve()}")
    print(f"SMILES column: {smiles_col}")
    if args.no_sort:
        print("Ranking: disabled; output preserves input order")
    else:
        print(f"Ranking: descending by {args.rank_by}; rank column: {CALLM_RANK_COLUMN}")
    print(f"Original row column: {CALLM_INPUT_ROW_COLUMN}")
    print(f"Total molecules: {len(scored)}")
    print(f"Valid SMILES: {valid_count}")
    print(f"Invalid SMILES: {invalid_count}")
    print("Applicability domain distribution:")
    if len(scored) == 0:
        print("<empty>")
        print(f"{CALLM_SCORE_COLUMN} distribution: <empty>")
    else:
        print(ad_distribution.to_string())
        print(
            f"{CALLM_SCORE_COLUMN} distribution: "
            f"min={penalized.min():.6g}, "
            f"median={penalized.median():.6g}, "
            f"mean={penalized.mean():.6g}, "
            f"max={penalized.max():.6g}"
        )
    if not args.no_sort:
        print(f"Ranking column: {CALLM_RANK_COLUMN}")
    print(f"Use {CALLM_SCORE_COLUMN} as the CALLM LRRK2 activity objective.")
    print("Keep activity_score and pred_pIC50 for analysis only.")
    print(f"Output path: {output_path.resolve()}")


if __name__ == "__main__":
    main()
