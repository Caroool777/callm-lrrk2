from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep the smoke test lightweight on shared machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lrrk2_qsar_oracle import ORACLE_OUTPUT_COLUMNS, load_oracle, score_dataframe
from lrrk2_qsar_oracle import ad_penalized_activity_score
from score_callm_lrrk2_candidates import (
    CALLM_INPUT_ROW_COLUMN,
    CALLM_RANK_COLUMN,
    CALLM_SCORE_COLUMN,
    rank_scored_candidates,
)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    assert np.isclose(ad_penalized_activity_score(0.5, "medium"), 0.425)
    assert np.isclose(ad_penalized_activity_score(0.5, "low"), 0.275)
    assert np.isclose(ad_penalized_activity_score(0.5, "very_low"), 0.15)
    assert np.isclose(ad_penalized_activity_score(1.0, "very_low"), 0.25)
    assert ad_penalized_activity_score(0.5, "invalid") == 0.0

    # Empty/all-invalid batches should not need model artifacts. This prevents
    # CALLM loops from crashing when a generation/filtering step produces no
    # valid molecules.
    missing_model_dir = ROOT / "definitely_missing_model_dir"
    empty = score_dataframe(pd.DataFrame({"SMILES": []}), model_dir=missing_model_dir)
    assert empty.empty
    assert all(column in empty.columns for column in ORACLE_OUTPUT_COLUMNS)

    all_invalid = score_dataframe(
        pd.DataFrame({"SMILES": ["not_a_smiles", "also_bad"]}, index=[101, 205]),
        model_dir=missing_model_dir,
    )
    assert list(all_invalid.index) == [101, 205]
    assert all_invalid["applicability_domain"].tolist() == ["invalid", "invalid"]
    assert all_invalid[CALLM_SCORE_COLUMN].tolist() == [0.0, 0.0]

    oracle = load_oracle(ROOT)

    # Non-default DataFrame index regression: no NaNs or row misalignment.
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "bad"],
            "SMILES": [
                "c1ccccc1",
                "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
                "not_a_smiles",
            ],
        },
        index=[10, 20, 42],
    )
    scored = score_dataframe(df, oracle=oracle)
    assert list(scored.index) == [10, 20, 42]
    assert scored.loc[10, "canonical_SMILES"] == "c1ccccc1"
    assert bool(scored.loc[10, "valid_smiles"])
    assert not bool(scored.loc[42, "valid_smiles"])
    assert not scored.loc[[10, 20], "pred_pIC50"].isna().any()
    assert not scored.loc[[10, 20], "applicability_domain"].isna().any()
    assert float(scored.loc[42, CALLM_SCORE_COLUMN]) == 0.0

    # Training-set molecule should have nearest_train_similarity = 1 and high AD.
    train = pd.read_csv(ROOT / "lrrk2_train_smiles.csv")
    train_score = score_dataframe(
        pd.DataFrame({"SMILES": [train.loc[0, "canonical_SMILES"]]}),
        oracle=oracle,
    )
    assert np.isclose(float(train_score.loc[0, "nearest_train_similarity"]), 1.0, atol=1e-6)
    assert train_score.loc[0, "applicability_domain"] == "high"

    # Ranking helper regression: sorted descending and deterministic 1-based rank.
    rank_input = scored.copy()
    rank_input.insert(0, CALLM_INPUT_ROW_COLUMN, range(1, len(rank_input) + 1))
    ranked = rank_scored_candidates(rank_input)
    assert ranked.columns[0] == CALLM_RANK_COLUMN
    assert ranked[CALLM_RANK_COLUMN].tolist() == [1, 2, 3]
    ranked_scores = pd.to_numeric(ranked[CALLM_SCORE_COLUMN], errors="coerce").fillna(-1).tolist()
    assert ranked_scores == sorted(ranked_scores, reverse=True)

    print("All LRRK2 QSAR oracle regression tests passed.")


if __name__ == "__main__":
    main()
