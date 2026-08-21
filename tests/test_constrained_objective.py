import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constrained_objective import (
    calculate_soft_constraint_metrics,
    parse_constrained_objective,
    probability_at_least_k,
    soft_object_pass,
)


class ConstrainedObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "regions_gev": [[25, 40], [40, 120]],
                    "region_weights": [0.5, 0.5],
                    "allowed_deficits": [0.005, 0.005],
                }
            }
        )

    def test_at_least_two_probability_matches_exact_binary_decisions(self):
        probabilities = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        measured = probability_at_least_k(probabilities, mask, k=2)
        self.assertTrue(torch.equal(measured, torch.tensor([1.0, 0.0, 1.0])))

    def test_padded_objects_do_not_change_event_probability(self):
        probabilities = torch.tensor([[0.8, 0.7, 0.9]])
        mask = torch.tensor([[True, True, False]])
        measured = probability_at_least_k(probabilities, mask, k=2)
        self.assertAlmostEqual(float(measured), 0.56, places=6)

    def test_surrogate_has_gradient_near_threshold(self):
        scores = torch.tensor([[0.49, 0.51]], requires_grad=True)
        probabilities = soft_object_pass(
            scores,
            threshold=0.5,
            temperature=0.05,
            classifier_name="nn_only",
        )
        event_probability = probability_at_least_k(
            probabilities,
            torch.ones_like(probabilities, dtype=torch.bool),
            k=2,
        )
        event_probability.backward()
        self.assertTrue(torch.all(torch.isfinite(scores.grad)))
        self.assertTrue(torch.all(scores.grad > 0))

    def test_or_branch_is_exactly_one_for_tob_passing_objects(self):
        scores = torch.tensor([[0.1, 0.9]])
        tob_pt = torch.tensor([[50.0, 10.0]])
        probabilities = soft_object_pass(
            scores,
            threshold=0.5,
            temperature=0.05,
            classifier_name="tob_nn_or",
            tob_pt_gev_values=tob_pt,
            tob_threshold_gev=40.0,
        )
        self.assertEqual(float(probabilities[0, 0]), 1.0)
        self.assertLess(float(probabilities[0, 1]), 1.0)

    def test_soft_metrics_use_event_fpr_and_object_efficiency(self):
        probabilities = torch.tensor(
            [[0.8, 0.7], [0.9, 0.6]], requires_grad=True
        )
        object_mask = torch.ones_like(probabilities, dtype=torch.bool)
        signal_mask = torch.tensor([[False, False], [True, True]])
        background_mask = torch.tensor([True, False])
        truth_pt = torch.tensor([[0.0, 0.0], [30.0, 50.0]])
        baseline = torch.tensor([[False, False], [True, True]])
        metrics = calculate_soft_constraint_metrics(
            probabilities,
            object_mask,
            signal_mask,
            background_mask,
            truth_pt,
            baseline,
            self.config,
        )
        self.assertAlmostEqual(float(metrics.event_fpr.detach()), 0.56, places=6)
        self.assertEqual(tuple(metrics.violations.shape), (3,))
        (-metrics.objective + metrics.violations.sum()).backward()
        self.assertIsNotNone(probabilities.grad)

    def test_reference_guard_uses_the_stricter_efficiency_floor(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "regions_gev": [[25, 120]],
                    "region_weights": [1.0],
                    "allowed_deficits": [0.005],
                    "minimum_region_advantages": [0.0025],
                    "reference_model_allowed_deficits": [0.005],
                }
            }
        )
        probabilities = torch.tensor([[0.80, 0.70]], requires_grad=True)
        reference = torch.tensor([[0.90, 0.80]])
        object_mask = torch.ones_like(probabilities, dtype=torch.bool)
        signal_mask = torch.ones_like(probabilities, dtype=torch.bool)
        background_mask = torch.tensor([True])
        truth_pt = torch.tensor([[30.0, 50.0]])
        baseline = torch.tensor([[True, False]])
        metrics = calculate_soft_constraint_metrics(
            probabilities,
            object_mask,
            signal_mask,
            background_mask,
            truth_pt,
            baseline,
            config,
            reference_object_pass_probabilities=reference,
        )
        # Baseline + 0.0025 is 0.5025; reference - 0.005 is 0.845.
        self.assertAlmostEqual(float(metrics.required_efficiencies[0]), 0.845)
        self.assertAlmostEqual(float(metrics.region_margins[0].detach()), -0.095)
        self.assertAlmostEqual(float(metrics.violations[1].detach()), 0.095)

    def test_positive_baseline_margin_is_capped_at_unit_efficiency(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "regions_gev": [[60, 120]],
                    "region_weights": [1.0],
                    "minimum_region_advantages": [0.0025],
                }
            }
        )
        probabilities = torch.tensor([[1.0, 1.0]], requires_grad=True)
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        metrics = calculate_soft_constraint_metrics(
            probabilities,
            mask,
            mask,
            torch.tensor([True]),
            torch.tensor([[80.0, 90.0]]),
            torch.tensor([[True, True]]),
            config,
        )
        self.assertEqual(float(metrics.required_efficiencies[0]), 1.0)
        self.assertEqual(float(metrics.region_margins[0].detach()), 0.0)

    def test_separate_dual_rates_fall_back_to_legacy_rate(self):
        legacy = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "dual_learning_rate": 3.0,
                }
            }
        )
        self.assertEqual(legacy.fpr_dual_learning_rate, 3.0)
        self.assertEqual(legacy.region_dual_learning_rate, 3.0)

        separate = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "dual_learning_rate": 3.0,
                    "fpr_dual_learning_rate": 1.0,
                    "region_dual_learning_rate": 50.0,
                }
            }
        )
        self.assertEqual(separate.fpr_dual_learning_rate, 1.0)
        self.assertEqual(separate.region_dual_learning_rate, 50.0)


if __name__ == "__main__":
    unittest.main()
