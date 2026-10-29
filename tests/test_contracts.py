from fractions import Fraction
import unittest

from decision_criticality.contracts import consequence_readout, evaluate_decision


class DecisionContractTests(unittest.TestCase):
    def test_tied_actions_can_have_different_reference_harm(self):
        result = evaluate_decision((Fraction(3, 5), Fraction(2, 5)), (Fraction(1, 2), Fraction(1, 2)), 1, 0)
        self.assertEqual(result["edited_regret"], 0)
        self.assertEqual(result["reference_regret"], 0.2)
        self.assertEqual((result["possible_any_harm"], result["unavoidable_any_harm"]), (1, 0))
        self.assertTrue(result["policy_dependent"])

    def test_preexisting_loss_is_not_incremental_damage(self):
        result = evaluate_decision((3, 2), (3, 2), 1, 1)
        self.assertEqual(result["reference_regret"], 1)
        self.assertEqual(result["incremental_reference_regret"], 0)
        prediction = consequence_readout((0.4, 0.6), (0.4, 0.6))
        self.assertEqual(prediction["predicted_regret"], 0)

    def test_material_boundary_is_exact_and_optimal_bounds_do_not_bound_bad_actions(self):
        result = evaluate_decision((Fraction(51, 100), Fraction(49, 100), 0), (1, 0, 0), 1)
        self.assertEqual(result["material_harm"], 1)
        self.assertGreater(result["reference_regret"], result["maximum_optimal_regret"])
        self.assertEqual(result["exact_reference_regret"], "1/50")

    def test_incremental_loss_can_be_negative_and_invalid_inputs_fail(self):
        self.assertEqual(evaluate_decision((3, 2), (3, 2), 0, 1)["incremental_reference_regret"], -1)
        with self.assertRaises(ValueError):
            evaluate_decision((1, 2), (1,), 0)
        with self.assertRaises(ValueError):
            consequence_readout((1, 2), (1, float("nan")))