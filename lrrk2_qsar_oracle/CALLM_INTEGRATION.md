# CALLM LRRK2 Scoring Integration

## Objective column

For generated molecule scoring and ranking, use:

```text
qsar_activity_score_ad_penalized
```

Do not optimize raw `activity_score` directly. Keep `pred_pIC50`, `activity_score`, `nearest_train_similarity`, and `applicability_domain` for diagnostics.

## Option A: CLI scoring and ranking

Place this package folder in your CALLM project and run:

```bash
python score_callm_lrrk2_candidates.py \
  --input callm_candidates.csv \
  --output callm_candidates_scored.csv \
  --model_dir .
```

Default CLI behavior sorts descending by `qsar_activity_score_ad_penalized` and adds:

```text
callm_lrrk2_rank
callm_input_row
```

`callm_input_row` is 1-based and points back to the original input CSV row order before sorting.

Use `--no_sort` when CALLM requires output rows to stay aligned with the input CSV:

```bash
python score_callm_lrrk2_candidates.py \
  --input callm_candidates.csv \
  --output callm_candidates_scored_input_order.csv \
  --model_dir . \
  --no_sort
```

## Option B: Python batch API

Load the oracle once and reuse it:

```python
from pathlib import Path
import pandas as pd

from lrrk2_qsar_oracle import load_oracle, score_dataframe

ORACLE_DIR = Path("path/to/lrrk2_qsar_oracle_callm_ready_v1_1_1")
_ORACLE = load_oracle(ORACLE_DIR)


def score_lrrk2_candidates(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    return score_dataframe(df, smiles_col=smiles_col, oracle=_ORACLE)


def rank_lrrk2_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.sort_values(
        by="qsar_activity_score_ad_penalized",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
```

The Python API preserves the input DataFrame index and row order. Sort only after scores have been attached to candidate IDs.

## Minimal scalar scoring hook

```python
from pathlib import Path
import pandas as pd

from lrrk2_qsar_oracle import load_oracle, score_dataframe

_ORACLE = load_oracle(Path("path/to/lrrk2_qsar_oracle_callm_ready_v1_1_1"))


def lrrk2_qsar_score(smiles: str) -> float:
    scored = score_dataframe(pd.DataFrame({"SMILES": [smiles]}), oracle=_ORACLE)
    return float(scored.loc[0, "qsar_activity_score_ad_penalized"])
```

## Regression test

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python test_lrrk2_qsar_oracle_regression.py
```
