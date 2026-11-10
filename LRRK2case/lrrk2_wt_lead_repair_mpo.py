"""
LRRK2 WT lead repair / co-optimization task.
"""

import json
import os
import platform
import random
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from typing import Any, Dict, List, Tuple

from optimizer_base_mo import CulturalAlgorithmMOBase, load_config
from lrrk2_qsar_mpo_utils import (
    canonical_discovered_rows_to_ranked_molecules,
    collect_canonical_deduplicated_discovered_rows,
    LRRK2QSAROracleAdapter,
    build_lead_centered_initial_population,
    compute_property_panel,
    clip01,
    deduplicate_ranked_molecules_by_canonical_smiles,
    export_ranked_molecules_to_sdf,
    export_structure_filtered_candidate_tables,
    identify_structure_risk_flags,
    structure_flags_are_blocking,
    structure_risk_filter_config,
)

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")


def set_seed(seed=42):
    """Set random seeds and deterministic backend flags for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass


def log_reproducibility_versions():
    """Log package versions that affect reproducibility."""
    import importlib

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ["numpy", "rdkit", "torch", "transformers"]:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "not_installed"
    print("Reproducibility versions:", json.dumps(versions, sort_keys=True))
    return versions
    return versions


def load_run_config(default_task_name, config_path=None):
    config = load_config(task_name=default_task_name)
    if not config_path:
        return config

    resolved_path = config_path
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.join(os.path.dirname(__file__), config_path)

    with open(resolved_path, "r", encoding="utf-8") as handle:
        override_config = json.load(handle)

    config.update(override_config)
    print(f"Loaded override config: {resolved_path}")
    return config


DEFAULT_LEAD_REPAIR_OBJECTIVE_NAMES = [
    "qsar_activity_ad_penalized",
    "bbb_cns",
    "logp_window",
    "tpsa_window",
]


class LRRK2WTLeadRepairMPOOracle:
    """Explicit LRRK2 WT lead-repair oracle with structural-risk filtering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parent_seed = str(config.get("parent_seed", ""))
        self.generation_batch = str(config.get("generation_batch", ""))
        self.activity_oracle = LRRK2QSAROracleAdapter(
            oracle_dir=config.get("qsar_oracle_dir"),
            score_column=config.get("qsar_oracle_score_column", "qsar_activity_score_ad_penalized"),
            raw_value_column=config.get("qsar_oracle_raw_column", "pred_pIC50"),
        )
        self.logp_window = config.get("logp_window", {"lower": 1.5, "upper": 3.5, "sigma": 0.75})
        self.tpsa_window = config.get("tpsa_window", {"lower": 20.0, "upper": 70.0, "sigma": 12.0})
        self.regularizer_type = str(config.get("regularizer_type", "qed")).lower()
        configured_objectives = list(config.get("objective_names", DEFAULT_LEAD_REPAIR_OBJECTIVE_NAMES))
        if "raw_pic50_tiebreaker" in configured_objectives:
            print(
                "Warning: raw_pic50_tiebreaker is no longer a Pareto objective; "
                "it will be recorded as diagnostics and used only for same-score ranking ties."
            )
            configured_objectives = [name for name in configured_objectives if name != "raw_pic50_tiebreaker"]
        self.objective_names = configured_objectives
        self.pic50_tiebreaker_min = float(config.get("raw_pic50_tiebreaker_min", 6.0))
        self.pic50_tiebreaker_max = float(config.get("raw_pic50_tiebreaker_max", 11.0))
        supported = set(DEFAULT_LEAD_REPAIR_OBJECTIVE_NAMES + ["regularizer"])
        unknown = [name for name in self.objective_names if name not in supported]
        if unknown:
            raise ValueError(f"Unknown lead-repair objective(s): {unknown}")
        if not self.objective_names:
            raise ValueError("Lead-repair objective_names must contain at least one Pareto objective.")
        self.structure_filter_enabled = bool(config.get("structure_filter_enabled", True))
        self.structure_filter_apply_oracle_penalty = bool(
            config.get("structure_filter_apply_oracle_penalty", True)
        )
        self.structure_filter_penalty_factor = float(config.get("structure_filter_penalty_factor", 0.0))
        self.structure_filter_block_flags, self.structure_filter_block_prefixes = structure_risk_filter_config(config)

    def _raw_pic50_tiebreaker(self, raw_pic50: float) -> float:
        if not np.isfinite(raw_pic50):
            return 0.0
        width = max(1e-9, self.pic50_tiebreaker_max - self.pic50_tiebreaker_min)
        return clip01((raw_pic50 - self.pic50_tiebreaker_min) / width)

    def _objective_mapping(self, activity_score: float, raw_pic50: float, panel: Dict[str, float]) -> Dict[str, float]:
        return {
            "qsar_activity_ad_penalized": activity_score,
            "raw_pic50_tiebreaker": self._raw_pic50_tiebreaker(raw_pic50),
            "bbb_cns": panel["bbb_cns_score"],
            "logp_window": panel["logp_window_score"],
            "tpsa_window": panel["tpsa_window_score"],
            "regularizer": panel["regularizer_score"],
        }

    def _structure_risk_metadata(self, smiles: str) -> Dict[str, Any]:
        if not self.structure_filter_enabled:
            return {
                "structure_risk_flags": "",
                "structure_risk_flag_count": 0,
                "structure_risk_pass": True,
                "structure_filter_enabled": False,
                "structure_penalty_applied": False,
                "structure_penalty_factor": 1.0,
            }
        flags = identify_structure_risk_flags(smiles)
        blocking = structure_flags_are_blocking(
            flags,
            block_flags=self.structure_filter_block_flags,
            block_prefixes=self.structure_filter_block_prefixes,
        )
        return {
            "structure_risk_flags": ";".join(flags),
            "structure_risk_flag_count": len(flags),
            "structure_risk_pass": not blocking,
            "structure_filter_enabled": True,
            "structure_penalty_applied": bool(blocking and self.structure_filter_apply_oracle_penalty),
            "structure_penalty_factor": self.structure_filter_penalty_factor if blocking else 1.0,
        }

    def _apply_structure_penalty(self, objectives: List[float], structure_meta: Dict[str, Any]) -> List[float]:
        if not structure_meta.get("structure_penalty_applied"):
            return objectives
        factor = clip01(float(structure_meta.get("structure_penalty_factor", 0.0)))
        return [clip01(float(value) * factor) for value in objectives]

    def score_all(self, smiles: str) -> Tuple[List[float], Dict[str, float]]:
        raw_activity, activity_score = self.activity_oracle.score_smiles(smiles)
        panel = compute_property_panel(
            smiles,
            logp_window=self.logp_window,
            tpsa_window=self.tpsa_window,
            regularizer=self.regularizer_type,
        )
        mapping = self._objective_mapping(activity_score, raw_activity, panel)
        raw_objectives = [clip01(mapping[name]) for name in self.objective_names]
        structure_meta = self._structure_risk_metadata(smiles)
        objectives = self._apply_structure_penalty(raw_objectives, structure_meta)
        meta = {
            "parent_seed": self.parent_seed,
            "generation_batch": self.generation_batch,
            "raw_predicted_pic50": raw_activity,
            "qsar_activity_ad_penalized": activity_score,
            "qsar_activity_score": activity_score,
            "raw_pic50_tiebreaker": self._raw_pic50_tiebreaker(raw_activity),
            "bbb_cns_score": panel["bbb_cns_score"],
            "logp_window_score": panel["logp_window_score"],
            "tpsa_window_score": panel["tpsa_window_score"],
            "regularizer_score": panel["regularizer_score"],
            "regularizer_type": self.regularizer_type,
            "raw_logp": panel["raw_logp"],
            "raw_tpsa": panel["raw_tpsa"],
            "raw_mw": panel["raw_mw"],
            "objective_names": ";".join(self.objective_names),
            **structure_meta,
        }
        for name, value in zip(self.objective_names, raw_objectives):
            meta[f"unfiltered_objective_{name}"] = value
        for name, value in zip(self.objective_names, objectives):
            meta[f"objective_{name}"] = value
        return objectives, meta

    def score_many(self, smiles_list: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        activity_results = self.activity_oracle.score_many(smiles_list)
        batch_scores = []
        batch_meta = []
        for smiles, activity_result in zip(smiles_list, activity_results):
            panel = compute_property_panel(
                smiles,
                logp_window=self.logp_window,
                tpsa_window=self.tpsa_window,
                regularizer=self.regularizer_type,
            )
            mapping = self._objective_mapping(activity_result.normalized_score, activity_result.raw_value, panel)
            raw_objectives = [clip01(mapping[name]) for name in self.objective_names]
            structure_meta = self._structure_risk_metadata(smiles)
            objectives = self._apply_structure_penalty(raw_objectives, structure_meta)
            meta = {
                "parent_seed": self.parent_seed,
                "generation_batch": self.generation_batch,
                "raw_predicted_pic50": activity_result.raw_value,
                "qsar_activity_ad_penalized": activity_result.normalized_score,
                "qsar_activity_score": activity_result.normalized_score,
                "raw_pic50_tiebreaker": self._raw_pic50_tiebreaker(activity_result.raw_value),
                "bbb_cns_score": panel["bbb_cns_score"],
                "logp_window_score": panel["logp_window_score"],
                "tpsa_window_score": panel["tpsa_window_score"],
                "regularizer_score": panel["regularizer_score"],
                "regularizer_type": self.regularizer_type,
                "raw_logp": panel["raw_logp"],
                "raw_tpsa": panel["raw_tpsa"],
                "raw_mw": panel["raw_mw"],
                "objective_names": ";".join(self.objective_names),
                **structure_meta,
                **activity_result.diagnostics,
            }
            for name, value in zip(self.objective_names, raw_objectives):
                meta[f"unfiltered_objective_{name}"] = value
            for name, value in zip(self.objective_names, objectives):
                meta[f"objective_{name}"] = value
            batch_scores.append(objectives)
            batch_meta.append(meta)
        return batch_scores, batch_meta


class CulturalAlgorithmLRRK2WTLeadRepairMPO(CulturalAlgorithmMOBase):
    TASK_NAME = "lrrk2_wt_lead_repair_mpo"
    OBJECTIVE_NAMES = DEFAULT_LEAD_REPAIR_OBJECTIVE_NAMES

    def __init__(self, oracle, max_oracle_calls=10000, device=None, objective_names=None):
        self.mpo_oracle = oracle
        self.OBJECTIVE_NAMES = list(objective_names or oracle.objective_names)
        super().__init__(
            oracle_funcs=[
                (lambda smi, idx=i: oracle.score_all(smi)[0][idx])
                for i in range(len(self.OBJECTIVE_NAMES))
            ],
            task_name=self.TASK_NAME,
            max_oracle_calls=max_oracle_calls,
            device=device,
        )

    def evaluate_objectives(self, smiles):
        return self.mpo_oracle.score_all(smiles)[0]

    def evaluate(self, mols):
        results = []
        canonical_records = []
        uncached_smiles = []
        seen_uncached = set()

        for mol in mols:
            if mol is None:
                continue
            try:
                canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                if not canonical_smi:
                    continue
                canonical_records.append((mol, canonical_smi))
                if canonical_smi not in self.score_cache and canonical_smi not in seen_uncached:
                    seen_uncached.add(canonical_smi)
                    uncached_smiles.append(canonical_smi)
            except Exception as e:
                print(f"Eval Error: {e}")
                continue

        remaining_budget = max(0, self.max_oracle_calls - self.current_oracle_calls)
        uncached_to_score = uncached_smiles[:remaining_budget]
        if uncached_to_score:
            batch_objectives, batch_meta = self.mpo_oracle.score_many(uncached_to_score)
            for canonical_smi, objectives, metadata in zip(uncached_to_score, batch_objectives, batch_meta):
                self.score_cache[canonical_smi] = objectives
                self._record_discovery_metadata(canonical_smi, metadata)
                self.current_oracle_calls += 1
                self.all_discovered.append((objectives, canonical_smi, self.current_oracle_calls))
                if self.current_oracle_calls % self.convergence_log_freq == 0:
                    self._log_convergence_point()

        for mol, canonical_smi in canonical_records:
            objectives = self.score_cache.get(canonical_smi)
            if objectives is None:
                continue
            results.append((objectives, mol, canonical_smi))

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LRRK2 WT lead repair MPO")
    parser.add_argument("--device", type=str, default=None, help="GPU device, e.g. cuda:0")
    parser.add_argument("--max_calls", type=int, default=10000, help="Maximum oracle calls")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config override path, relative to LRRK2case/ unless absolute.",
    )
    parser.add_argument("--ablation_mode", type=str, default=None, help="Ablation mode override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_run_config("lrrk2_wt_lead_repair_mpo", config_path=args.config)
    if args.ablation_mode:
        config["ablation_mode"] = args.ablation_mode
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_calls is not None:
        config["max_oracle_calls"] = args.max_calls
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.device:
        config["device"] = args.device
        config["biot5_device"] = args.device
    seed = config.get("seed", 42)
    set_seed(seed)
    config["software_versions"] = log_reproducibility_versions()
    print(f"Random seed: {seed}")

    if "offspring_fraction" in config and "crossover_rate" in config:
        config["crossover_rate_deprecated"] = config.pop("crossover_rate")

    initial_pop = build_lead_centered_initial_population(
        config,
        task_dir=os.path.dirname(__file__),
        default_csv="../oracle/lrrk2_seed_molecules.csv",
    )
    if not initial_pop:
        raise ValueError("Lead repair task requires at least one valid parent or seed SMILES.")

    oracle = LRRK2WTLeadRepairMPOOracle(config)
    max_calls = config.get("max_oracle_calls", args.max_calls)
    optimizer = CulturalAlgorithmLRRK2WTLeadRepairMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device,
        objective_names=oracle.objective_names,
    )
    optimizer.apply_config(config)
    optimizer.run_config.update({
        "stage": "stage2_lrrk2_wt_lead_repair_mpo",
        "generation_methods": ["random_crossover", "biot5_edit", "random_mutation"],
        "seed_source": "stage1_seed_file_plus_parent_lead_mutation",
        "stage1_seed_required": bool(config.get("require_stage1_seed_file", False)),
        "initial_population_metadata": config.get("initial_population_metadata", {}),
        "biot5_only": True,
        "scaffold_constraint_enabled": bool(config.get("use_scaffold_constraint", False)),
    })

    print("\n" + "=" * 60)
    print("Starting LRRK2 WT lead repair MPO")
    print("=" * 60)

    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best

    raw_pareto_size = len(pareto_front)
    pareto_front = deduplicate_ranked_molecules_by_canonical_smiles(pareto_front)
    dedup_removed = raw_pareto_size - len(pareto_front)
    if dedup_removed > 0:
        print(
            f"Canonical SMILES dedup removed {dedup_removed} duplicate entries "
            f"from the complete final Pareto front ({raw_pareto_size} -> {len(pareto_front)})."
        )
    pareto_metrics["full_pareto_size_before_canonical_dedup"] = raw_pareto_size
    pareto_metrics["full_pareto_size_after_canonical_dedup"] = len(pareto_front)
    pareto_metrics["canonical_dedup_removed"] = dedup_removed
    pareto_metrics["top100_is_ranked_summary_only"] = True
    if pareto_front:
        best_score, best_smi, best_objectives = pareto_front[0]

    output_dir = config.get("output_dir", "../MultiResult/lrrk2_wt_lead_repair_mpo_results")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    optimizer.print_results(
        "LRRK2_WT_Lead_Repair_MPO",
        best_score,
        best_smi,
        pareto_front,
        hypervolume,
        best_objectives=best_objectives,
        output_dir=output_dir,
        pareto_metrics=pareto_metrics,
    )

    canonical_all_discovered_path = os.path.join(
        output_dir,
        "lrrk2_wt_lead_repair_mpo_all_discovered_canonical.csv",
    )
    canonical_rows_data = collect_canonical_deduplicated_discovered_rows(
        optimizer.all_discovered,
        objective_names=optimizer.OBJECTIVE_NAMES,
        metadata_by_smiles=optimizer.discovery_metadata,
    )
    pd.DataFrame(canonical_rows_data).to_csv(canonical_all_discovered_path, index=False)
    canonical_rows = len(canonical_rows_data)
    print(
        "Exported canonical-deduplicated discovered molecules: "
        f"{canonical_all_discovered_path} ({canonical_rows} unique molecules)"
    )

    filtered_exports = export_structure_filtered_candidate_tables(
        canonical_rows_data,
        output_dir=output_dir,
        output_prefix="lrrk2_wt_lead_repair_mpo",
        config=config,
    )
    optimizer.run_config["structure_filter_summary"] = filtered_exports["summary"]
    run_config_path = os.path.join(output_dir, "run_config.json")
    try:
        with open(run_config_path, "w", encoding="utf-8") as handle:
            json.dump(optimizer.run_config, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"  Warning: failed to update run_config with structure filter summary: {exc}")
    print(
        "Exported structure-filtered candidates: "
        f"clean={len(filtered_exports['clean_rows'])}, "
        f"flagged={len(filtered_exports['flagged_rows'])}"
    )
    print(f"  Clean candidates: {filtered_exports['clean_csv']}")
    print(f"  Flagged candidates: {filtered_exports['flagged_csv']}")

    if config.get("export_selected_sdf", True):
        sdf_path = os.path.join(
            output_dir,
            config.get("selected_sdf_filename", "lrrk2_wt_lead_repair_selected.sdf"),
        )
        per_molecule_dir = None
        if config.get("export_selected_sdf_individual_files", False):
            per_molecule_dir = os.path.join(
                output_dir,
                config.get("selected_sdf_individual_dir", "selected_sdf_files"),
            )
        if config.get("selected_sdf_use_clean_candidates", True):
            sdf_source_rows = filtered_exports["clean_rows"]
            min_selected = int(config.get("selected_sdf_top_k", 20))
            if len(sdf_source_rows) < min_selected:
                sdf_source_rows = filtered_exports.get("structure_pass_rows", sdf_source_rows)
                print(
                    "  Warning: clean candidate threshold yielded fewer selected molecules; "
                    "using structure-pass candidates for SDF fallback."
                )
        else:
            sdf_source_rows = canonical_rows_data
        sdf_ranked_molecules = canonical_discovered_rows_to_ranked_molecules(
            sdf_source_rows,
            objective_names=optimizer.OBJECTIVE_NAMES,
        )
        written = export_ranked_molecules_to_sdf(
            sdf_ranked_molecules,
            output_sdf_path=sdf_path,
            objective_names=optimizer.OBJECTIVE_NAMES,
            max_molecules=config.get("selected_sdf_top_k", 20),
            generate_3d=config.get("selected_sdf_generate_3d", True),
            per_molecule_dir=per_molecule_dir,
        )
        print(f"Exported {written} selected clean molecules to SDF: {sdf_path}")
