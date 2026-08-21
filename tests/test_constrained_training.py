import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classifiers import parse_classifier
from constrained_objective import (
    SoftConstraintMetrics,
    parse_constrained_objective,
)
from constrained_training import (
    DualState,
    _is_better_hard_candidate,
    constrained_primal_loss,
    initialize_fpr_multiplier_from_gradients,
    parameter_gradient_pair_statistics,
    parameter_gradient_norm,
    update_dual_state,
)
from event_data import EventBatch


class ConstrainedTrainingTests(unittest.TestCase):
    def test_dual_update_projects_to_nonnegative_finite_interval(self):
        state = DualState(torch.tensor([0.1, 0.2, 0.0]))
        update_dual_state(
            state,
            torch.tensor([-1.0, 2.0, 100.0]),
            learning_rate=0.5,
            maximum=3.0,
        )
        self.assertTrue(torch.equal(state.multipliers, torch.tensor([0.0, 1.2, 3.0])))

    def test_primal_loss_uses_detached_dual_prices(self):
        objective = torch.tensor(0.3, requires_grad=True)
        violations = torch.tensor([0.1, -0.2], requires_grad=True)
        metrics = SoftConstraintMetrics(
            objective=objective,
            event_fpr=torch.tensor(0.0),
            region_efficiencies=torch.zeros(1),
            baseline_efficiencies=torch.zeros(1),
            region_deltas=torch.zeros(1),
            violations=violations,
            valid_regions=torch.ones(1, dtype=torch.bool),
        )
        dual = DualState(torch.tensor([2.0, 3.0], requires_grad=True))
        loss = constrained_primal_loss(metrics, dual)
        loss.backward()
        self.assertAlmostEqual(float(objective.grad), -1.0)
        self.assertTrue(torch.equal(violations.grad, torch.tensor([2.0, 3.0])))
        self.assertIsNone(dual.multipliers.grad)

    def test_feasible_checkpoint_is_preferred_before_objective(self):
        infeasible = {
            "constraints_satisfied": False,
            "objective_value": 0.9,
            "minimum_margin": -0.1,
        }
        feasible = {
            "constraints_satisfied": True,
            "objective_value": 0.1,
            "minimum_margin": 0.0,
        }
        self.assertTrue(_is_better_hard_candidate(feasible, infeasible))
        self.assertFalse(_is_better_hard_candidate(infeasible, feasible))

    def test_gradient_norm_does_not_populate_parameter_gradients(self):
        model = torch.nn.Linear(1, 1)
        value = model(torch.ones(2, 1)).sum()
        norm = parameter_gradient_norm(
            value,
            model.parameters(),
            retain_graph=False,
        )
        self.assertGreater(float(norm), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_gradient_pair_statistics_reports_direction(self):
        parameter = torch.tensor([2.0, -1.0], requires_grad=True)
        first = torch.sum(parameter ** 2)
        second = 3.0 * first
        first_norm, second_norm, cosine = parameter_gradient_pair_statistics(
            first,
            second,
            [parameter],
            retain_graph=False,
        )
        self.assertGreater(float(first_norm), 0.0)
        self.assertGreater(float(second_norm), float(first_norm))
        self.assertAlmostEqual(float(cosine), 1.0, places=6)
        self.assertIsNone(parameter.grad)

    def test_gradient_balance_selects_training_only_ratio(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())
        batch = EventBatch(
            features=torch.tensor(
                [
                    [[1.0], [0.5]],
                    [[0.8], [0.4]],
                    [[-0.2], [-0.5]],
                    [[-0.4], [-0.8]],
                ]
            ),
            labels=torch.tensor(
                [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            truth_pt_gev=torch.tensor(
                [[30.0, 35.0], [45.0, 50.0], [0.0, 0.0], [0.0, 0.0]]
            ),
            tob_pt_gev=torch.tensor(
                [[30.0, 35.0], [45.0, 50.0], [5.0, 6.0], [5.0, 6.0]]
            ),
            object_mask=torch.ones(4, 2, dtype=torch.bool),
            signal_object_mask=torch.tensor(
                [[True, True], [True, True], [False, False], [False, False]]
            ),
            background_event_mask=torch.tensor([False, False, True, True]),
            event_numbers=torch.arange(4),
        )
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "regions_gev": [[25.0, 120.0]],
                    "region_weights": [1.0],
                    "allowed_deficits": [0.005],
                    "initial_fpr_multiplier_mode": "gradient_balance",
                    "initial_fpr_multiplier": 1.0,
                    "gradient_balance_batches": 1,
                    "max_multiplier": 5.0,
                }
            }
        )
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "nn_only",
                    "target_fpr": 0.005,
                    "trigger_objects": 2,
                }
            }
        )
        selected, diagnostic = initialize_fpr_multiplier_from_gradients(
            model,
            [batch],
            classifier,
            config,
            fixed_nn_threshold=0.5,
            fixed_tob_threshold=None,
            baseline_threshold_gev=20.0,
        )
        self.assertEqual(diagnostic["batches_measured"], 1)
        self.assertGreater(diagnostic["recommended_unclipped"], 0.0)
        self.assertIn("gradient_cosine_similarity", diagnostic["measurements"][0])
        self.assertAlmostEqual(
            selected,
            min(diagnostic["recommended_unclipped"], config.max_multiplier),
        )


if __name__ == "__main__":
    unittest.main()
