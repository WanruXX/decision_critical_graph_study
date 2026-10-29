from copy import deepcopy
import unittest

import numpy as np
import torch

from decision_criticality.example import (
    build_pairs, contextualize, distinct_reports, edit_evidence, generate_instance,
    instance_signature, normalize_evidence, pack_observations, posterior, reference_posterior, sample_instances,
)
from decision_criticality.models import GraphActionModel, predict


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.instance = generate_instance(np.random.default_rng(42), "example")

    def test_exact_posterior_and_normalization_agree(self):
        for view in self.instance["views"]:
            self.assertEqual(posterior(view), reference_posterior(view))
            self.assertEqual(posterior(view), posterior(normalize_evidence(view)))

    def test_duplicate_and_verified_attribution_are_protected(self):
        for view in self.instance["views"]:
            root = next(iter(distinct_reports(view)))
            copied = contextualize(view, "forwarded_copy", root)
            changed = edit_evidence(copied, root, "remove_one_copy")
            for before, after in zip(pack_observations([copied]), pack_observations([changed])):
                self.assertTrue(torch.equal(before, after))
            for audited_root in view["audit"]:
                changed = edit_evidence(view, audited_root, "changed_author")
                self.assertEqual(posterior(view), posterior(changed))
                for before, after in zip(pack_observations([view]), pack_observations([changed])):
                    self.assertTrue(torch.equal(before, after))

    def test_pairs_and_model_shapes(self):
        pool, pairs = build_pairs([self.instance])
        self.assertEqual(len(pairs), 235)
        self.assertTrue(all(0 <= pair["edited_index"] < len(pool) for pair in pairs))
        packed = pack_observations([pool[0]["view"]])
        probabilities = predict(GraphActionModel(), packed)
        self.assertEqual(probabilities.shape, (1, 5))
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)

    def test_inconsistent_copies_and_impossible_evidence_are_rejected(self):
        view = deepcopy(self.instance["views"][0])
        view["reports"].append({**view["reports"][0], "wolf_claim": not view["reports"][0]["wolf_claim"]})
        for oracle in (posterior, reference_posterior):
            with self.assertRaises(ValueError):
                oracle(view)
        view = deepcopy(self.instance["views"][0])
        view["inspection"] = (view["observer"], True)
        for oracle in (posterior, reference_posterior):
            with self.assertRaises(ValueError):
                oracle(view)

    def test_split_sampling_and_hidden_metadata_do_not_leak_into_features(self):
        first = sample_instances((51,), 2)
        signatures = {instance_signature(instance) for instance in first}
        second = sample_instances((51,), 2, excluded=signatures)
        self.assertFalse(signatures.intersection(instance_signature(instance) for instance in second))
        view = self.instance["views"][0]
        tagged = {**view, "true_wolf": 5, "roles": [0] * 6, "context": "independent_conflict",
                  "condition": "changed_status", "oracle_posterior": [1, 0, 0, 0, 0, 0]}
        for original, changed in zip(pack_observations([view]), pack_observations([tagged])):
            self.assertTrue(torch.equal(original, changed))