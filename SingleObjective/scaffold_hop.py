#!/usr/bin/env python
"""Run CALLM on the TDC PMO scaffold_hop task under Structure-based/."""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SINGLE_OBJECTIVE_DIR = os.path.join(REPO_ROOT, "SingleObjective")

for path in (SINGLE_OBJECTIVE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import crossover as co  # noqa: E402
import mutate as mu  # noqa: E402
from optimizer_base import CulturalAlgorithmBase, load_config, load_initial_population  # noqa: E402
from oracle_utils import ScaffoldHopOracle  # noqa: E402


class CulturalAlgorithmScaffoldHop(CulturalAlgorithmBase):
    """Single-objective scaffold_hop optimizer with source logging."""

    TASK_NAME = "scaffold_hop"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_source_history = []
        self.generated_candidate_history = []

    def apply_config(self, config):
        super().apply_config(config)
        self.run_config.update(
            {
                "task_name": self.TASK_NAME,
                "oracle": "tdc.Oracle(name='Scaffold Hop')",
                "generation_methods": [
                    "random_crossover",
                    "biot5_edit",
                    "random_mutation",
                ],
                "save_all_generated_molecules": True,
                "biot5_only": False,
                "pmo_auc_frequency": 100,
                "pmo_auc_finish": True,
            }
        )

    def _record_generated_candidates(self, generation, source, mols):
        for mol in mols:
            smiles = self._canonicalize_mol(mol)
            self.generated_candidate_history.append(
                {
                    "candidate_idx": len(self.generated_candidate_history) + 1,
                    "generation": int(generation),
                    "source": source,
                    "smiles": smiles or "",
                    "valid_rdkit": bool(smiles),
                }
            )

    def _generation_source_totals(self):
        totals = {
            "crossover_candidates": 0,
            "biot5_candidates": 0,
            "random_mutation_candidates": 0,
            "deduplicated_offspring": 0,
            "evaluated_offspring": 0,
        }
        for row in self.generation_source_history:
            for key in totals:
                totals[key] += int(row.get(key, 0) or 0)
        totals["generations_recorded"] = len(self.generation_source_history)
        totals["generated_candidates_recorded"] = len(self.generated_candidate_history)
        totals["generated_candidate_sources"] = dict(
            Counter(row["source"] for row in self.generated_candidate_history)
        )
        return totals

    def _run_ablation_impl(self, initial_smiles_list):
        start_time = time.time()
        print(f"Initializing Population... ablation_mode={self.ablation_mode}")
        initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles_list if s]
        initial_mols = self._deduplicate_valid_mols(initial_mols)

        if not initial_mols:
            print("Error: No valid molecules in initial population!")
            return (0.0, None, ""), 0.0, 0.0

        print(f"Valid initial molecules: {len(initial_mols)}")
        population_data = self.evaluate(initial_mols)
        if not population_data:
            print("Error: Failed to evaluate initial population!")
            return (0.0, None, ""), 0.0, 0.0

        if self.ablation_mode != "no_belief_space":
            self.belief_space.update(population_data, current_generation=0)
        else:
            pool = self._get_single_selection_pool(population_data)
            best_score = pool[0][0] if pool else 0.0
            print(f"[no_belief_space] transient pool size={len(pool)} best_score={best_score:.4f}")

        generation = 0
        max_stale_generations = 100
        stale_count = 0
        last_calls = 0
        use_crossover = self.ablation_mode != "no_ga_operators"
        use_llm = self.ablation_mode != "no_llm"
        use_random_mutation = self.ablation_mode != "no_ga_operators"

        while self.current_oracle_calls < self.max_oracle_calls:
            best_score = self._get_best_single_result(population_data)[0]
            if self.ablation_mode == "no_belief_space":
                pool = self._get_single_selection_pool(population_data)
                print(
                    f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                    f"Best: {best_score:.4f} | Transient Pool: {len(pool)}"
                )
            else:
                belief_summary = self.belief_space.get_belief_summary()
                penalty_info = ""
                if (
                    belief_summary["boredom_penalized_scaffolds"] > 0
                    or belief_summary["similarity_penalized_scaffolds"] > 0
                ):
                    penalty_info = (
                        f" | Boredom: {belief_summary['boredom_penalized_scaffolds']}, "
                        f"SimPen: {belief_summary['similarity_penalized_scaffolds']}"
                    )
                print(
                    f"Gen {generation} | Calls: {self.current_oracle_calls}/{self.max_oracle_calls} | "
                    f"Best: {best_score:.4f} | Active Beliefs: {belief_summary['active_beliefs']}{penalty_info}"
                )

            offspring_mols = []
            source_counts = {
                "generation": generation,
                "crossover_candidates": 0,
                "biot5_candidates": 0,
                "random_mutation_candidates": 0,
                "deduplicated_offspring": 0,
                "evaluated_offspring": 0,
                }

            if use_crossover:
                crossover_attempts = (
                    self.offspring_size
                    if self.ablation_mode == "no_llm"
                    else max(1, self.offspring_size // 3)
                )
                for _ in range(crossover_attempts):
                    parents_data = self._select_single_parents_for_crossover(population_data, 2)
                    if len(parents_data) == 2:
                        child = co.crossover(parents_data[0][1], parents_data[1][1])
                        if child:
                            offspring_mols.append(child)
                            source_counts["crossover_candidates"] += 1
                            self._record_generated_candidates(generation, "random_crossover", [child])

            if use_llm:
                rounds = 3 if self.ablation_mode == "no_ga_operators" else 1
                for round_idx in range(rounds):
                    elite_count = min(
                        max(self.llm_editor_count * (round_idx + 1), 1),
                        max(len(population_data), 1),
                    )
                    elite_data = self._select_single_elites_for_llm(population_data, elite_count)
                    if not elite_data:
                        break
                    elites = [item[1] for item in elite_data if item[1] is not None]
                    constraint_for_llm = self._get_single_scaffold_constraint_for_llm()
                    try:
                        mutated_elites = self.llm_editor.edit(elites, scaffold_constraints=constraint_for_llm)
                        mutated_elites = [m for m in mutated_elites if m is not None]
                        print(f"  BioT5 generated {len(mutated_elites)} candidates (input {len(elites)})")
                        offspring_mols.extend(mutated_elites)
                        source_counts["biot5_candidates"] += len(mutated_elites)
                        self._record_generated_candidates(generation, "biot5_edit", mutated_elites)
                    except Exception as exc:
                        print(f"LLM error: {exc}")
                    if len(offspring_mols) >= self.offspring_size:
                        break

            if use_random_mutation:
                retry_limit = self.offspring_size * (6 if self.ablation_mode == "no_llm" else 3)
                retries = 0
                while len(offspring_mols) < self.offspring_size and retries < retry_limit:
                    parents = self._get_single_selection_pool(population_data)
                    if not parents:
                        print("Warning: random mutation fallback has no parents.")
                        break
                    parent = random.choice(parents)
                    child = mu.mutate(parent[1], mutation_rate=self.mutation_rate)
                    if child:
                        offspring_mols.append(child)
                        source_counts["random_mutation_candidates"] += 1
                        self._record_generated_candidates(generation, "random_mutation", [child])
                    retries += 1

            offspring_mols = self._deduplicate_valid_mols(offspring_mols)
            source_counts["deduplicated_offspring"] = len(offspring_mols)
            if not offspring_mols:
                print(f"Warning: empty offspring in {self.ablation_mode}; falling back to current top molecules.")
                offspring_mols = [
                    item[1] for item in self._get_single_selection_pool(population_data, self.offspring_size)
                ]
                self._record_generated_candidates(generation, "fallback_current_top", offspring_mols)

            offspring_data = self.evaluate(offspring_mols)
            source_counts["evaluated_offspring"] = len(offspring_data)
            self.generation_source_history.append(dict(source_counts))
            print(
                "  Generation sources: crossover={crossover_candidates}, BioT5={biot5_candidates}, "
                "random_mutation={random_mutation_candidates}, dedup={deduplicated_offspring}, "
                "evaluated={evaluated_offspring}".format(**source_counts)
            )

            if offspring_data and self.ablation_mode != "no_belief_space":
                self.belief_space.update(offspring_data, current_generation=generation)

            if self.current_oracle_calls == last_calls:
                stale_count += 1
                if stale_count >= max_stale_generations:
                    print(f"Warning: No progress for {max_stale_generations} generations. Stopping.")
                    break
            else:
                stale_count = 0
                last_calls = self.current_oracle_calls

            if self.ablation_mode == "no_belief_space":
                next_gen_candidates = offspring_data + self._get_single_selection_pool(population_data)
            else:
                next_gen_candidates = offspring_data + self.belief_space.knowledge_repository
            next_gen_candidates = sorted(next_gen_candidates, key=lambda x: x[0], reverse=True)
            population_data = next_gen_candidates[: self.population_size] or population_data
            generation += 1

        self._log_convergence_point()
        top_10_auc = self.compute_top_k_auc(10)
        top_100_auc = self.compute_top_k_auc(100)
        self.runtime_seconds = time.time() - start_time
        self._print_final_summary()
        return self._get_best_single_result(population_data), top_10_auc, top_100_auc

    def print_results(
        self,
        task_name,
        best_score,
        best_smi,
        top_10_auc,
        top_100_auc,
        output_dir=None,
        save_images=True,
        save_curve=True,
    ):
        super().print_results(
            task_name=task_name,
            best_score=best_score,
            best_smi=best_smi,
            top_10_auc=top_10_auc,
            top_100_auc=top_100_auc,
            output_dir=output_dir,
            save_images=save_images,
            save_curve=save_curve,
        )
        self._save_structure_based_outputs(task_name, output_dir)

    def _save_structure_based_outputs(self, task_name, output_dir):
        if output_dir is None:
            output_dir = os.path.join(SCRIPT_DIR, "result")
        os.makedirs(output_dir, exist_ok=True)

        source_path = os.path.join(output_dir, "generation_sources.csv")
        pd.DataFrame(self.generation_source_history).to_csv(source_path, index=False)
        print(f"  Saved generation source counts: {source_path}")

        candidates_path = os.path.join(output_dir, "generated_candidates.csv")
        pd.DataFrame(self.generated_candidate_history).to_csv(candidates_path, index=False)
        print(f"  Saved generated candidates: {candidates_path}")

        totals = self._generation_source_totals()
        for filename in ("metrics.json", f"{task_name.lower()}_metrics.json", "run_config.json"):
            path = os.path.join(output_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data["generation_source_totals"] = totals
                data["generation_sources_csv"] = "generation_sources.csv"
                data["generated_candidates_csv"] = "generated_candidates.csv"
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"  Warning: failed to patch {filename}: {exc}")


def resolve_output_dir(value, relative_to_script=True):
    output_dir = value or "result"
    if not os.path.isabs(output_dir):
        base_dir = SCRIPT_DIR if relative_to_script else os.getcwd()
        output_dir = os.path.join(base_dir, output_dir)
    return os.path.abspath(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run CALLM BioT5 on TDC PMO scaffold_hop.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to config seed.")
    parser.add_argument("--max_calls", type=int, default=None, help="Oracle budget. Defaults to config max_oracle_calls.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to Structure-based/result.")
    args = parser.parse_args()

    config = load_config(task_name="scaffold_hop", base_dir=SCRIPT_DIR)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_calls is not None:
        config["max_oracle_calls"] = args.max_calls
    if args.device is not None:
        config["device"] = args.device
        config["biot5_device"] = args.device

    if args.output_dir is None:
        output_dir = resolve_output_dir(config.get("output_image_dir", "result"), relative_to_script=True)
    else:
        output_dir = resolve_output_dir(args.output_dir, relative_to_script=False)
    config["output_image_dir"] = output_dir

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    os.makedirs(output_dir, exist_ok=True)

    use_chemical_filter = bool(config.get("use_chemical_filter", False))
    if use_chemical_filter:
        raise ValueError(
            "Strict PMO runs must not apply an extra chemical filter to the TDC oracle."
        )
    oracle = ScaffoldHopOracle(use_chemical_filter=False)
    initial_smiles = load_initial_population(config, base_dir=SCRIPT_DIR)

    optimizer = CulturalAlgorithmScaffoldHop(
        oracle_func=oracle,
        task_name="scaffold_hop",
        max_oracle_calls=int(config.get("max_oracle_calls", 10000)),
        boredom_threshold=config.get("boredom_threshold", 15),
        boredom_penalty=config.get("boredom_penalty", 0.1),
        boredom_decay=config.get("boredom_decay", 0.01),
        similarity_threshold=config.get("similarity_threshold", 0.7),
        similarity_penalty=config.get("similarity_penalty", 0.05),
        device=config.get("device", config.get("biot5_device")),
        ablation_mode=config.get("ablation_mode", "full"),
    )
    optimizer.apply_config(config)
    optimizer.run_config.update(
        {
            "script_path": os.path.abspath(__file__),
            "output_dir": output_dir,
            "config_dir": SCRIPT_DIR,
        }
    )

    best, top_10_auc, top_100_auc = optimizer.run(initial_smiles)
    best_score, _, best_smi = best
    optimizer.print_results(
        task_name="scaffold_hop",
        best_score=best_score,
        best_smi=best_smi,
        top_10_auc=top_10_auc,
        top_100_auc=top_100_auc,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
