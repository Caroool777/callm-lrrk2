# CALLM BioT5 Backend

CALLM LRRK2 experiments use a local BioT5 molecule editor only. Remote API backends are intentionally unsupported so LRRK2 runs are reproducible and do not depend on external services.

## Files

```text
SingleObjective/llm_backends/
├── __init__.py             # BioT5-only factory and compatibility wrapper
├── abstract_backend.py     # shared molecule-editor interface
├── biot5_backend.py        # local Hugging Face BioT5 implementation
├── prompts_multi.py        # multi-objective task prompts, including LRRK2
└── prompts_single.py       # single-objective task prompts
```

## Usage

```python
from llm_backends import create_molecule_editor

editor = create_molecule_editor("biot5", task="lrrk2_wt_lead_repair_mpo")
editor.apply_config(config)
candidates = editor.edit(mols, scaffold_constraints=scaffold_smiles)
```

Configuration should use:

```json
{
  "llm_backend": "biot5",
  "biot5_model_name": "QizhiPei/biot5-base-text2mol",
  "biot5_num_beams": 5,
  "biot5_num_return_sequences": 3,
  "biot5_temperature": 0.8,
  "biot5_do_sample": true,
  "biot5_elite_count": 10
}
```

If `llm_backend` is missing, the factory defaults to `biot5`. If any other backend name is requested, the factory raises `ValueError`.

## LRRK2 Notes

- Stage 1 uses `lrrk2_wt_activity_mpo` and optimizes AD-penalized LRRK2 QSAR activity plus a weak regularizer.
- Stage 2 uses `lrrk2_wt_lead_repair_mpo` and optimizes exactly four Pareto objectives: AD-penalized activity, BBB/CNS-likeness, logP window fit, and TPSA window fit.
- Raw predicted pIC50 is metadata and a same-score tie-breaker only; it is not a Pareto objective.
- Scaffold constraints are injected into the BioT5 prompt when enabled by the task config.
