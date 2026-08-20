import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constrained_objective import SoftConstraintMetrics
from constrained_training import (
    DualState,
    _is_better_hard_candidate,
    constrained_primal_loss,
    update_dual_state,
)


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


if __name__ == "__main__":
    unittest.main()
