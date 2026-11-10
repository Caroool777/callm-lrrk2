# CALLM

CALLM combines cultural algorithms with molecular generation models for single-objective and multi-objective molecular optimization, including LRRK2 case studies.

## Project Structure

| Path | Purpose |
| --- | --- |
| `SingleObjective/` | Single-objective optimization, belief spaces, crossover, and mutation operators |
| `MultiObjective/` | Multi-objective optimization, Pareto fronts, and evaluation metrics |
| `LRRK2case/` | LRRK2 activity optimization and lead repair |
| `lrrk2_qsar_oracle/` | LRRK2 QSAR scoring code, model, training fingerprints, and metadata |
| `configs/` | Configuration files for the two-stage LRRK2 workflow |
| `scripts/` | Workflow execution, result validation, docking input preparation, and plotting |
| `250k_rndm_zinc_drugs_clean_3.csv` | Initial molecular library |

## Current Status

LRRK2 QSAR scoring can run independently. The local BioT5 backend in `SingleObjective/llm_backends/` is included and is used by the single-objective and LRRK2 generation workflows.

`MultiObjective/optimizer_base_mo.py` uses a separate BioT5 implementation. Molecular generation requires PyTorch, Transformers, SELFIES, and model weights. The full molecular generation workflow has not yet been validated.

## Installation

Use Python 3.10 or 3.11 and create a new environment on your machine:

```bash
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, create a Conda environment with `conda env create -f environment.yml`, activate it, and run `python -m pip install -r requirements.txt` to install the task dependencies. GPU execution requires a PyTorch installation compatible with your hardware.

## LRRK2 QSAR Scoring

For scoring only, install the smaller dependency set:

```bash
python -m pip install -r lrrk2_qsar_oracle/requirements.txt
python lrrk2_qsar_oracle/score_callm_lrrk2_candidates.py --input candidates.csv --output scored_candidates.csv
python lrrk2_qsar_oracle/test_lrrk2_qsar_oracle_regression.py
```

The input CSV must contain a `SMILES` or `smiles` column. Use `qsar_activity_score_ad_penalized` as the optimization objective; retain raw activity predictions and applicability-domain information for analysis. See the [Oracle documentation](lrrk2_qsar_oracle/README.md) for details.

## Optimization Entry Points

Use the following commands to list tasks and inspect command-line options:

```bash
python SingleObjective/run_parallel.py --list
python SingleObjective/qed.py --help
python MultiObjective/zaleplon_mpo.py --help
python scripts/run_lrrk2_stage1_stage2_20k.py --help
```

Available general optimization tasks:

- Single-objective: `qed`, `jnk3`, `gsk3b`, `drd2`, `celecoxib_rediscovery`, `albuterol_similarity`, `mestranol_similarity`, `median1`, `median2`, `scaffold_hop`, `deco_hop`, `valsartan_smarts`.
- Multi-objective: `sitagliptin_mpo`, `zaleplon_mpo`, `amlodipine_mpo`, `osimertinib_mpo`, `perindopril_mpo`, `ranolazine_mpo`.

Linux / Bash users can run `bash run_experiments.sh help` to view batch execution commands. The LRRK2 workflow runs subprocesses in the Python environment used to launch the script.

Run `python scripts/test_lrrk2_stage1_stage2_20k.py` for lightweight regression checks of seed validation, budget allocation, stage handoff, and exports. These checks run without loading the generation model.

## Files and Release

The project includes source code, configuration files, the initial molecular library, and Oracle models. The `.gitignore` file excludes caches, local virtual environments, assistant settings, credentials, backups, and generated results. Files already tracked by Git must be handled through the corresponding commit changes. Experiment outputs are regenerated during execution.

The project code is licensed under the [MIT License](LICENSE.txt).
Third-party code, datasets, and models remain subject to their respective licenses and terms of use.
