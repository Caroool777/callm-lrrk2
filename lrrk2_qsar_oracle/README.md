# LRRK2 QSAR Activity Oracle v1.1.2

This oracle is intended for the CALLM LRRK2 case to score generated molecules with a predicted LRRK2 activity signal.

## Model

- Model: LightGBM `LGBMRegressor`
- Molecular representation: Morgan fingerprint
- Radius: `2`
- Bits: `2048`
- Target: `pIC50`
- IC50 unit: nM

## Inputs

The oracle accepts a CSV file containing a SMILES column.

## Outputs

Key activity outputs:

- `pred_pIC50`: predicted pIC50 from the QSAR model
- `pred_IC50_nM`: predicted IC50 in nM
- `activity_score`: raw 0-1 score from predicted pIC50, for analysis only
- `qsar_activity_score_ad_penalized`: AD-aware and structure-penalized score for CALLM optimization
- `nearest_train_similarity`: maximum Tanimoto similarity to training molecules
- `applicability_domain`: high, medium, low, very_low, or invalid
- `oracle_confidence`: confidence label matching the applicability domain

Structure-risk outputs:

- `n_fragments`: number of molecular fragments
- `is_multifragment`: true for salts or disconnected structures
- `contains_metal`: true for common metal or metalloid atoms
- `formal_charge`: RDKit formal charge
- `has_abnormal_charge`: true when absolute formal charge is greater than 1, or for standalone charged ions such as `[NH4+]`
- `structure_warning`: semicolon-separated warning labels

Descriptor outputs:

- `MW`
- `logP`
- `TPSA`
- `HBD`
- `HBA`
- `RotB`
- `RingCount`
- `QED`

## Raw Activity Score

The raw activity score is retained for analysis:

```text
activity_score = clip((pred_pIC50 - 6.0) / 3.0, 0, 1)
```

This maps `pred_pIC50 = 6` to 0, `pred_pIC50 = 7` to about 0.333, `pred_pIC50 = 8` to about 0.667, and `pred_pIC50 >= 9` to 1.

The predicted IC50 is reported as:

```text
pred_IC50_nM = 10 ** (9 - pred_pIC50)
```

## CALLM Optimization Score

CALLM should use:

```text
qsar_activity_score_ad_penalized
```

Do not use raw `activity_score` as the primary optimization objective. Keep `activity_score` and `pred_pIC50` for analysis and reporting.

## Applicability Domain

Applicability-domain confidence is estimated from the maximum Tanimoto similarity between the query molecule's Morgan fingerprint and all training-set fingerprints.

| nearest_train_similarity | applicability_domain | oracle_confidence |
| --- | --- | --- |
| `>= 0.60` | `high` | `high` |
| `0.40 - 0.60` | `medium` | `medium` |
| `0.30 - 0.40` | `low` | `low` |
| `< 0.30` | `very_low` | `very_low` |

Low-similarity molecules are still scored, but their predictions should be interpreted cautiously.

## AD Penalty

The raw activity score is penalized by applicability domain:

| applicability_domain | AD-penalized score |
| --- | --- |
| `high` | raw score |
| `medium` | `min(score * 0.85, 0.70)` |
| `low` | `min(score * 0.55, 0.45)` |
| `very_low` | `min(score * 0.30, 0.25)` |
| `invalid` | `0.0` |

## Structure Penalty

Version 1.1.2 modestly relaxes the AD multipliers while retaining the previous caps. Model weights and pIC50 predictions are unchanged. Results produced with v1.1.1 retain their original scores; this version has not been calibrated against an independent activity set.

The AD-penalized score is further reduced for structure-risk flags:

- Multifragment molecule: `score = min(score * 0.2, 0.10)`
- Contains metal: `score = min(score * 0.2, 0.10)`
- Abnormal charge: `score = min(score * 0.5, 0.20)`

The final `qsar_activity_score_ad_penalized` is clipped to 0-1. Molecules are not deleted because of low confidence or structure flags; they are labeled and penalized.

## Usage

From the project root:

```bash
python lrrk2_qsar_oracle.py --input input_molecules.csv --smiles_col SMILES --output scored_molecules.csv
```

From inside the `lrrk2_qsar_oracle/` artifact directory:

```bash
python lrrk2_qsar_oracle.py --input input_molecules.csv --smiles_col SMILES --output scored_molecules.csv
```

CALLM wrapper, default behavior writes a score-descending ranked CSV:

```bash
python score_callm_lrrk2_candidates.py --input callm_candidates.csv --output callm_candidates_scored.csv
```

If CALLM needs output rows to remain in the exact same order as the input candidate table, disable sorting:

```bash
python score_callm_lrrk2_candidates.py --input callm_candidates.csv --output callm_candidates_scored_input_order.csv --no_sort
```

## Files

- `lrrk2_qsar_oracle_model.pkl`: trained final LightGBM model
- `lrrk2_qsar_oracle_metadata.json`: model metadata and provenance
- `lrrk2_train_fingerprints.npy`: training Morgan fingerprints
- `lrrk2_train_smiles.csv`: training molecules and activity values
- `lrrk2_qsar_oracle_self_check.csv`: training-set self-check predictions
- `self_test_scored_clean_dataset_v1_1.csv`: v1.1 command-line self-test output


## CALLM Integration Bugfix Note

Version 1.1.1 introduced a `score_dataframe` fix for CALLM batch scoring:

- filtered or sliced DataFrames with non-default indices are scored by row position, so output rows stay aligned with input rows;
- empty candidate batches return an empty scored table with the complete oracle schema instead of raising `KeyError`;
- the core Python API `score_dataframe(...)` preserves the original input row order;
- the CLI wrapper sorts candidates by `qsar_activity_score_ad_penalized` by default and adds `callm_lrrk2_rank`; use `--no_sort` if CALLM needs the original row order:
- loaded LightGBM models are configured with `n_jobs=1` for predictable CALLM inference resource usage; this does not change predictions.

```bash
python score_callm_lrrk2_candidates.py --input callm_candidates.csv --output callm_candidates_scored.csv
python score_callm_lrrk2_candidates.py --input callm_candidates.csv --output callm_candidates_scored_input_order.csv --no_sort
```

For direct CALLM integration, keep `score_dataframe` output in the original order and sort only after the scoring result has been attached to candidate IDs.

## Notes

This oracle comes from a QSAR model. Predictions are not experimental validation results. Molecules with low applicability domain confidence require careful interpretation and may need manual review. Downstream candidate selection should combine this oracle with docking score, ALA1950/GLU1948 interactions, logP, TPSA, BBB, SA_score, and scaffold diversity.

## Paper-ready wording

A LightGBM-based QSAR oracle trained on 4,927 curated LRRK2 inhibitors was used to estimate the activity of generated molecules. Molecular structures were represented by Morgan fingerprints, and the predicted pIC50 values were transformed into a normalized activity score for CALLM optimization. Applicability-domain confidence was estimated using the maximum Tanimoto similarity to the training molecules.
