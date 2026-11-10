"""
LRRK2 WT activity multi-objective optimization task.
"""

import json
import os
import random

import numpy as np
from rdkit import Chem
from rdkit import rdBase

from optimizer_base_mo import CulturalAlgorithmMOBase, load_config, load_initial_population
from lrrk2_qsar_mpo_utils import (
    canonical_discovered_rows_to_ranked_molecules,
    collect_canonical_deduplicated_discovered_rows,
    LRRK2QSAROracleAdapter,
    compute_property_panel,
    deduplicate_ranked_molecules_by_canonical_smiles,
    export_canonical_deduplicated_discovered_csv,
    export_ranked_molecules_to_sdf,
)

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class LRRK2WTActivityMPOOracle:
    def __init__(self, config):
        self.parent_seed = str(config.get("parent_seed", ""))
        self.generation_batch = str(config.get("generation_batch", ""))
        self.activity_oracle = LRRK2QSAROracleAdapter(
            oracle_dir=config.get("qsar_oracle_dir"),
            score_column=config.get("qsar_oracle_score_column", "qsar_activity_score_ad_penalized"),
            raw_value_column=config.get("qsar_oracle_raw_column", "pred_pIC50"),
        )
        self.regularizer_type = config.get("regularizer_type", "qed")

    def score_all(self, smiles):
        raw_activity, activity_score = self.activity_oracle.score_smiles(smiles)
        panel = compute_property_panel(smiles, regularizer=self.regularizer_type)
        return [activity_score, panel["regularizer_score"]], {
            "raw_predicted_pic50": raw_activity,
            "qsar_activity_score": activity_score,
            "regularizer_score": panel["regularizer_score"],
        }

    def score_many(self, smiles_list):
        activity_results = self.activity_oracle.score_many(smiles_list)
        batch_scores = []
        batch_meta = []
        for smiles, activity_result in zip(smiles_list, activity_results):
            panel = compute_property_panel(smiles, regularizer=self.regularizer_type)
            batch_scores.append([activity_result.normalized_score, panel["regularizer_score"]])
            batch_meta.append(
                {
                    "parent_seed": self.parent_seed,
                    "generation_batch": self.generation_batch,
                    "raw_predicted_pic50": activity_result.raw_value,
                    "qsar_activity_score": activity_result.normalized_score,
                    "regularizer_score": panel["regularizer_score"],
                    **activity_result.diagnostics,
                }
            )
        return batch_scores, batch_meta


class CulturalAlgorithmLRRK2WTActivityMPO(CulturalAlgorithmMOBase):
    TASK_NAME = "lrrk2_wt_activity_mpo"
    OBJECTIVE_NAMES = ["lrrk2_wt_activity", "regularizer"]

    def __init__(self, oracle, max_oracle_calls=10000, device=None):
        self.mpo_oracle = oracle
        super().__init__(
            oracle_funcs=[
                lambda smi: oracle.score_all(smi)[0][0],
                lambda smi: oracle.score_all(smi)[0][1],
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

    parser = argparse.ArgumentParser(description="LRRK2 WT activity MPO")
    parser.add_argument("--device", type=str, default=None, help="GPU device, e.g. cuda:0")
    parser.add_argument("--max_calls", type=int, default=10000, help="Maximum oracle calls")
    parser.add_argument("--config", type=str, default=None, help="Optional config override path")
    parser.add_argument("--ablation_mode", type=str, default=None, help="Ablation mode override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_config(task_name="lrrk2_wt_activity_mpo")
    if args.config:
        config_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    if args.ablation_mode:
        config["ablation_mode"] = args.ablation_mode
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_calls is not None:
        config["max_oracle_calls"] = args.max_calls
    if args.output_dir:
        config["output_dir"] = args.output_dir
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    oracle = LRRK2WTActivityMPOOracle(config)
    initial_pop = load_initial_population(config)

    max_calls = config.get("max_oracle_calls", args.max_calls)
    optimizer = CulturalAlgorithmLRRK2WTActivityMPO(
        oracle=oracle,
        max_oracle_calls=max_calls,
        device=args.device,
    )
    optimizer.apply_config(config)
    optimizer.run_config.update({
        "stage": "stage1_lrrk2_wt_activity_mpo",
        "generation_methods": ["random_crossover", "biot5_edit", "random_mutation"],
        "seed_source": "initial_smiles_file",
        "biot5_only": True,
        "scaffold_constraint_enabled": bool(config.get("use_scaffold_constraint", False)),
    })

    print("\n" + "=" * 60)
    print("Starting LRRK2 WT activity MPO")
    print("=" * 60)

    best, pareto_front, hypervolume, pareto_metrics = optimizer.run(initial_pop)
    best_score, best_mol, best_smi, best_objectives = best

    raw_pareto_size = len(pareto_front)
    pareto_front = deduplicate_ranked_molecules_by_canonical_smiles(pareto_front)
    dedup_removed = raw_pareto_size - len(pareto_front)
    if dedup_removed > 0:
        print(
            f"Canonical SMILES dedup removed {dedup_removed} duplicate entries "
            f"from the final Pareto summary ({raw_pareto_size} -> {len(pareto_front)})."
        )

    if pareto_front:
        best_score, best_smi, best_objectives = pareto_front[0]
        pareto_front_data = [(objectives, None, smi) for score, smi, objectives in pareto_front]
        pareto_metrics = optimizer.compute_pareto_metrics(pareto_front_data)
        pareto_metrics["pareto_size_before_canonical_dedup"] = raw_pareto_size
        pareto_metrics["pareto_size_after_canonical_dedup"] = len(pareto_front)
        pareto_metrics["canonical_dedup_removed"] = dedup_removed
        hypervolume = pareto_metrics.get("hypervolume", hypervolume)

    output_dir = config.get("output_dir", "../MultiResult/lrrk2_wt_activity_mpo_results")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    optimizer.print_results(
        "LRRK2_WT_Activity_MPO",
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
        "lrrk2_wt_activity_mpo_all_discovered_canonical.csv",
    )
    canonical_rows_data = collect_canonical_deduplicated_discovered_rows(
        optimizer.all_discovered,
        objective_names=optimizer.OBJECTIVE_NAMES,
        metadata_by_smiles=optimizer.discovery_metadata,
    )
    canonical_rows = export_canonical_deduplicated_discovered_csv(
        optimizer.all_discovered,
        output_csv_path=canonical_all_discovered_path,
        objective_names=optimizer.OBJECTIVE_NAMES,
        metadata_by_smiles=optimizer.discovery_metadata,
    )
    print(
        "Exported canonical-deduplicated discovered molecules: "
        f"{canonical_all_discovered_path} ({canonical_rows} unique molecules)"
    )

    if config.get("export_selected_sdf", True):
        sdf_path = os.path.join(
            output_dir,
            config.get("selected_sdf_filename", "lrrk2_wt_activity_selected.sdf"),
        )
        per_molecule_dir = None
        if config.get("export_selected_sdf_individual_files", False):
            per_molecule_dir = os.path.join(
                output_dir,
                config.get("selected_sdf_individual_dir", "selected_sdf_files"),
            )
        sdf_ranked_molecules = canonical_discovered_rows_to_ranked_molecules(
            canonical_rows_data,
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
        print(f"Exported {written} selected molecules to SDF: {sdf_path}")
