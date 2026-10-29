from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from decision_criticality.artifacts import ASSETS, FAMILIES, load_json, load_models
from decision_criticality.evaluation import paired_intervals, select_threshold
from decision_criticality.experiment import evaluate_pairs, run_study, summarize


class EvaluationTests(unittest.TestCase):
    def test_artifacts_load_and_models_replay(self):
        torch.set_num_threads(1)
        self.assertEqual(len(load_models()), 12)
        self.assertEqual(load_json(ASSETS / "reference.json")["pairs"], 28200)

    def test_actual_actions_have_separate_targets_and_shared_control(self):
        pool = [{"probabilities": ["0", "51/100", "49/100", "0", "0", "0"], "view": {"audit": {}}},
                {"probabilities": ["0", "1/2", "1/2", "0", "0", "0"], "view": {"audit": {}}}]
        pairs = [{"game_id": "instance", "observer": 0, "role": "seer", "root": "r0", "context": "original",
                  "edit": "changed_status", "reference_index": 0, "edited_index": 1}]
        predictions = {"baseline_42": np.asarray([[0.51, 0.49, 0, 0, 0], [0.49, 0.51, 0, 0, 0]]),
                       "independent_42": np.asarray([[0.51, 0.49, 0, 0, 0], [0.51, 0.49, 0, 0, 0]])}
        thresholds = {name: [0.01, 0.01] for name in predictions}
        rows = evaluate_pairs(pool, pairs, predictions, thresholds, {"instance": 1})
        self.assertEqual(rows["baseline_42"][0]["material_harm"], 1)
        self.assertEqual(rows["independent_42"][0]["material_harm"], 0)
        self.assertEqual(rows["baseline_42"][0]["common_action_regret"], rows["independent_42"][0]["common_action_regret"])
        self.assertEqual(summarize(rows, thresholds)["baseline_42"]["all"]["material_harm"]["f1"], 1)
        self.assertGreater(select_threshold([0], [0]), 1)

    def test_bootstrap_weights_whole_instances_and_checks_alignment(self):
        fields = ("regret_error", "common_action_error", "edited_regret", "reference_regret",
                  "incremental_reference_regret", "any_harm_error", "material_harm_error", "suboptimal_action")
        rows = {f"{family}_42": [
            {**dict.fromkeys(fields, 0.0), "game_id": instance, "data_seed": 1, "observer": 0, "root": str(index),
             "context": "original", "edit": "changed_status", "near_tie": False, "edited_tie": False,
             "policy_dependent": False, "regret_error": float(family == "independent" and index == 2)}
            for index, instance in enumerate(("first", "first", "second"))] for family in FAMILIES}
        result = paired_intervals(rows, FAMILIES, [42], samples=100)
        primary = next(row for row in result if row["contrast"] == "independent_minus_baseline" and row["metric"] == "regret_error")
        self.assertEqual(primary["difference"], 0.5)
        rows["uniform_42"].reverse()
        with self.assertRaises(ValueError):
            paired_intervals(rows, FAMILIES, [42], samples=10)

    def test_end_to_end_evaluation_writes_replayable_artifacts(self):
        with TemporaryDirectory() as temporary:
            output, summary = run_study(Path(temporary) / "evaluation", games_per_seed=1,
                                        data_seeds=(45101,), bootstrap_samples=10)
            self.assertEqual(summary["games"], 1)
            self.assertEqual(summary["pairs"], 235)
            self.assertTrue(all(summary["verification"].values()))
            for name in ("protocol.json", "instances.json", "observations.json", "pairs.json", "probabilities.npz",
                         "case_readouts.csv", "paired_intervals.csv", "summary.json", "report.md"):
                self.assertTrue((output / name).is_file())
            with self.assertRaises(FileExistsError):
                run_study(output, games_per_seed=1, data_seeds=(45101,), bootstrap_samples=10)