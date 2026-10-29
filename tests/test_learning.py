import unittest

import numpy as np
import torch

from decision_criticality.example import generate_instance
from decision_criticality.learning import build_dataset, fit_model
from decision_criticality.models import GraphActionModel


class LearningTests(unittest.TestCase):
    def test_matched_training_and_split_guard(self):
        torch.set_num_threads(1)
        random = np.random.default_rng(42)
        first, second = generate_instance(random, "train"), generate_instance(random, "validation")
        train, validation = build_dataset([first]), build_dataset([second])
        model, fit = fit_model(GraphActionModel(), train, validation, 42, epochs=1)
        self.assertEqual(fit["optimizer_steps"], 1)
        self.assertEqual(fit["row_presentations"], 25)
        self.assertEqual(len(fit["history"]), 2)
        self.assertIsInstance(model, GraphActionModel)
        with self.assertRaises(ValueError):
            fit_model(model, train, train, 42, epochs=1)