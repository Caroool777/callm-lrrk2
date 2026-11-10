# LRRK2 QSAR Oracle v1.1.1 CALLM bugfix

This is a scoring-code robustness patch for CALLM integration. The trained LightGBM model weights, training set, and fingerprint artifacts were not retrained or altered.

## Fixed

1. `score_dataframe()` now builds the scored table with an explicit `ORACLE_OUTPUT_COLUMNS` schema and the original input index.
2. Oracle output columns are copied back by row position, not by pandas index-label alignment. This fixes filtered, sliced, shuffled, and duplicate-index DataFrames.
3. Empty candidate batches return an empty scored table with the full oracle schema instead of raising `KeyError`.
4. Empty and all-invalid candidate batches do not require model artifact loading, so CALLM loops can safely pass empty batches.
5. `load_oracle()` sets LightGBM inference to `n_jobs=1` for predictable CALLM resource usage. This does not change predictions.
6. The CALLM wrapper sorts descending by `qsar_activity_score_ad_penalized` by default, adds `callm_lrrk2_rank`, and preserves the original input row as `callm_input_row`.
7. Use `--no_sort` when CALLM expects output rows to remain in the exact same order as the input CSV.

## Recommended CALLM objective

Use:

```text
qsar_activity_score_ad_penalized
```

Keep `activity_score`, `pred_pIC50`, `nearest_train_similarity`, and `applicability_domain` for diagnostics and reporting.

## Regression test

Run from the artifact folder:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python test_lrrk2_qsar_oracle_regression.py
```

Expected output:

```text
All LRRK2 QSAR oracle regression tests passed.
```
