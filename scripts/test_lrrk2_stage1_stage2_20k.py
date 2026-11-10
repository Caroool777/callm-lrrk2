#!/usr/bin/env python3

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

import run_lrrk2_stage1_stage2_20k as workflow


class LRRK2ProductionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)
        self.validated = workflow.validate_and_write_seeds(self.run_dir)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_fixed_seeds_validate_and_roundtrip(self):
        self.assertEqual(len(self.validated), 8)
        self.assertEqual(len({row["canonical_smiles"] for row in self.validated}), 8)
        self.assertTrue(all(row["validation_status"] == "PASS" for row in self.validated))
        self.assertTrue(all(row["selfies_roundtrip_status"] == "PASS" for row in self.validated))
        ly = next(row for row in self.validated if row["seed_id"] == "LY2023-001")
        self.assertIn("/C=N/", ly["canonical_smiles"])

    def test_weights_and_multinomial_allocations(self):
        self.assertTrue(
            math.isclose(
                sum(float(row["weight"]) for row in self.validated),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        for stream_seed in workflow.STREAM_SEEDS:
            allocation = workflow.multinomial_allocations(5000, stream_seed, self.validated)
            self.assertEqual(sum(allocation.values()), 5000)
            self.assertEqual(set(allocation), {row["seed_id"] for row in self.validated})
            self.assertTrue(all(value > 0 for value in allocation.values()))

    def test_stage1_lineage_checkpoint_and_stage2_handoff(self):
        specs, _ = workflow.make_stage1_specs(self.run_dir, self.validated, 5000)
        self.assertEqual(len(specs), 40)
        self.assertEqual({spec["stream_seed"] for spec in specs}, set(workflow.STREAM_SEEDS))
        for spec in specs:
            config = workflow.legacy.read_json(spec["config"], {})
            self.assertTrue(config["resume_checkpoint"])
            self.assertEqual(config["checkpoint_every_generations"], 1)
            self.assertEqual(config["parent_seed"], spec["seed_family"])

        stage1_rows = []
        for index in range(60):
            stage1_rows.append(
                {
                    "compound_id": f"S1_{index:03d}",
                    "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles("C" * (index + 1))),
                    "inchikey": f"TESTKEY{index:04d}",
                    "seed_family": self.validated[index % 8]["seed_id"],
                    "Pareto_front": index < 30,
                    "stage1_score": 1.0 - index / 1000.0,
                    "qsar_activity_score_ad_penalized": 0.9 - index / 1000.0,
                    "AD_category": "high",
                    "AD_similarity": 0.8,
                    "QED": 0.7,
                    "Murcko_scaffold": f"scaffold_{index % 25}",
                }
            )
        selected = workflow.select_stage2_parents(self.run_dir, stage1_rows, 50)
        self.assertEqual(len(selected), 50)
        self.assertEqual(len({row["inchikey"] for row in selected}), 50)
        self.assertTrue(all(row["seed_family"] for row in selected))
        stage2_specs, _ = workflow.make_stage2_specs(self.run_dir, selected, 5000)
        for spec in stage2_specs:
            config = workflow.legacy.read_json(spec["config"], {})
            self.assertEqual(config["parent_lead_smiles"], "")
            self.assertEqual(config["initial_population_strategy"], "stage1_only")
            self.assertEqual(config["regularizer_type"], "sa")
            self.assertEqual(len(config["objective_names"]), 5)

    def test_small_export_count_dedup_and_validation(self):
        final_smiles = ["CCO", "CCN", "CCC"]
        rows = []
        for index, smiles in enumerate(final_smiles, start=1):
            mol = Chem.MolFromSmiles(smiles)
            canonical = workflow.canonical_isomeric(mol)
            properties = workflow.descriptor_record(mol)
            rows.append(
                {
                    "compound_id": f"CALLM_LRRK2_S2_{index:06d}",
                    "canonical_smiles": canonical,
                    "inchikey": Chem.MolToInchiKey(mol),
                    "stage": "stage2",
                    "run_seed": 41,
                    "generation": "",
                    "seed_family": "LY2023-25",
                    "parent_id_1": "",
                    "parent_id_2": "",
                    "source_operator": "unavailable_in_current_callm_archive",
                    "generation_batch": "unit_test",
                    "pred_pIC50": 7.5,
                    "pred_IC50": 31.6,
                    "raw_activity_score": 0.5,
                    "qsar_activity_score_ad_penalized": 0.4,
                    "AD_similarity": 0.6,
                    "AD_category": "high",
                    **properties,
                    "Pareto_front": True,
                    "Pareto_rank": 0,
                    "final_rank": index,
                }
            )
        workflow.write_final_exports(self.run_dir, rows)
        report = workflow.validate_exports(
            self.run_dir,
            self.validated,
            [{"inchikey": self.validated[0]["inchikey"]}],
            target=3,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(workflow.parse_smi(self.run_dir / "final_docking_20000" / "lrrk2_stage2_20000_for_docking.smi")), 3)


if __name__ == "__main__":
    unittest.main()

