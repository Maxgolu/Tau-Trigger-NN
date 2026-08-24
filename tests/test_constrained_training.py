import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classifiers import parse_classifier
from constrained_objective import (
    SoftConstraintMetrics,
    parse_constrained_objective,
)
from constrained_training import (
    DualState,
    HardNegativeMemoryBank,
    _is_better_hard_candidate,
    budget_searched_metrics,
    constraint_resolution_warnings,
    constrained_primal_loss,
    initialize_fpr_multiplier_from_gradients,
    parameter_gradient_pair_statistics,
    parameter_gradient_norm,
    resolve_constrained_or_budget,
    update_dual_state,
)
from constrained_validation import (
    build_constraint_crossfit_rows,
    calculate_cross_fitted_hard_metrics,
)
from event_data import EventBatch
from model import DynamicMLP


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

    def test_dual_update_uses_separate_constraint_rates(self):
        state = DualState(torch.zeros(3))
        update_dual_state(
            state,
            torch.tensor([0.1, 0.1, -0.1]),
            learning_rate=1.0,
            region_learning_rate=10.0,
            maximum=3.0,
        )
        self.assertTrue(torch.equal(state.multipliers, torch.tensor([0.1, 1.0, 0.0])))

    def test_resolution_warning_detects_sub_object_slack(self):
        messages = constraint_resolution_warnings(
            {
                "constraint_margins": [0.0001, 0.01],
                "region_efficiency_resolutions": [0.0005, 0.001],
            },
            ((25.0, 40.0), (60.0, 120.0)),
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("below one-object resolution", messages[0])

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

    def test_infeasible_checkpoint_selection_prefers_certified_margin(self):
        high_objective_large_violation = {
            "constraints_satisfied": False,
            "objective_value": 0.9,
            "minimum_margin": 0.0,
            "minimum_certified_margin": -0.10,
            "feasibility": {"mode": "one_sided_confidence"},
        }
        lower_objective_small_violation = {
            "constraints_satisfied": False,
            "objective_value": 0.1,
            "minimum_margin": -0.01,
            "minimum_certified_margin": -0.01,
            "feasibility": {"mode": "one_sided_confidence"},
        }
        self.assertTrue(
            _is_better_hard_candidate(
                lower_objective_small_violation,
                high_objective_large_violation,
            )
        )
        self.assertFalse(
            _is_better_hard_candidate(
                high_objective_large_violation,
                lower_objective_small_violation,
            )
        )

    def test_legacy_infeasible_checkpoint_order_remains_objective_first(self):
        higher_objective = {
            "constraints_satisfied": False,
            "objective_value": 0.9,
            "minimum_margin": -0.10,
        }
        safer_objective = {
            "constraints_satisfied": False,
            "objective_value": 0.1,
            "minimum_margin": -0.01,
        }
        self.assertTrue(_is_better_hard_candidate(higher_objective, safer_objective))

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

    def test_fixed_fpr_initialization_skips_ill_conditioned_gradient_ratio(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "initial_fpr_multiplier_mode": "fixed",
                    "initial_fpr_multiplier": 0.0,
                }
            }
        )
        selected, diagnostic = initialize_fpr_multiplier_from_gradients(
            torch.nn.Linear(1, 1),
            [],
            None,
            config,
            fixed_nn_threshold=0.5,
            fixed_tob_threshold=None,
            baseline_threshold_gev=20.0,
        )
        self.assertEqual(selected, 0.0)
        self.assertEqual(diagnostic["batches_measured"], 0)
        self.assertIsNone(diagnostic["recommended_unclipped"])

    def test_dynamic_model_exposes_logits_without_changing_probabilities(self):
        model = DynamicMLP(2, [3])
        inputs = torch.tensor([[0.2, -0.1], [1.0, 2.0]])
        logits = model.forward_logits(inputs)
        self.assertTrue(torch.allclose(model(inputs), torch.sigmoid(logits)))
        self.assertIn("network.0.weight", model.state_dict())
        self.assertIn("network.2.weight", model.state_dict())

    def test_hard_negative_bank_keeps_largest_centered_offsets(self):
        bank = HardNegativeMemoryBank(3)
        bank.update(torch.tensor([0.1, 0.5]))
        bank.update(torch.tensor([0.2, 0.8]))
        self.assertEqual(len(bank), 3)
        self.assertTrue(
            torch.equal(bank.values, torch.tensor([0.8, 0.5, 0.2]))
        )

    def test_cross_fitted_constraints_measure_held_out_training_events(self):
        rows = []
        scores = []
        for event in range(8):
            for obj, score in enumerate((0.05 + event * 0.01, 0.10 + event * 0.01)):
                rows.append(
                    {
                        "eventNumber": event,
                        "tob_index": obj,
                        "Type": "BKG",
                        "signal": 0,
                        "truth_pt": 0.0,
                        "tob_pt": 5.0 + score,
                    }
                )
                scores.append(score)
        for event in range(8, 16):
            for obj, score in enumerate((0.70, 0.80)):
                rows.append(
                    {
                        "eventNumber": event,
                        "tob_index": obj,
                        "Type": "Signal",
                        "signal": 1,
                        "truth_pt": 30.0 + obj * 20.0,
                        "tob_pt": 20.0 + obj * 10.0,
                    }
                )
                scores.append(score)
        frame = pd.DataFrame(rows)
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "target_event_fpr": 0.25,
                    "objective_regions_gev": [[25, 60]],
                    "objective_region_weights": [1.0],
                    "constraint_regions_gev": [[25, 60]],
                    "allowed_deficits": [0.5],
                    "feasibility_confidence_level": 0.95,
                }
            }
        )
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "nn_only",
                    "target_fpr": 0.25,
                    "trigger_objects": 2,
                }
            }
        )
        folds = build_constraint_crossfit_rows(frame, seed=123)
        metrics = calculate_cross_fitted_hard_metrics(
            frame,
            np.asarray(scores),
            None,
            classifier,
            config,
            folds,
        )
        self.assertTrue(metrics["cross_fitted"])
        self.assertEqual(metrics["background_event_count"], 8)
        self.assertEqual(metrics["region_counts"], [16])
        self.assertEqual(len(metrics["folds"]), 2)
        self.assertIn("feasibility", metrics)
        self.assertIn("paired_region_sufficient_statistics", metrics)
        self.assertEqual(metrics["feasibility"]["mode"], "one_sided_confidence")
        self.assertAlmostEqual(
            metrics["folds"][0]["classifier_calibration"][
                "feasibility_confidence_level"
            ],
            0.975,
        )


class MeasurementBudgetSearchTests(unittest.TestCase):
    @staticmethod
    def _configs(classifier, loss_extra=None):
        loss = {
            "name": "constrained_trigger",
            "primal_objective": "tail_ranking",
            "proxy_threshold_mode": "batch_rank",
            "objective_regions_gev": [[25, 32], [32, 40], [40, 60]],
            "objective_region_weights": [0.35, 0.35, 0.3],
            "constraint_regions_gev": [[25, 32], [32, 40], [40, 60], [60, 120]],
            "allowed_deficits": [0.005] * 4,
            "minimum_region_advantages": [0.0025, 0.0025, 0.0025, 0],
            "reference_model_allowed_deficits": [0.0025, 0.0025, 0.0025, 0.005],
            "validation_crossfit": True,
            "feasibility_confidence_level": 0.95,
            "fpr_feasibility_mode": "point",
            "certified_guards_use_allowed_deficits": True,
        }
        loss.update(loss_extra or {})
        config = {"loss": loss, "classifier": classifier}
        from constrained_objective import parse_constrained_objective
        return parse_classifier(config), parse_constrained_objective(config)

    def test_rank_proxy_or_search_builds_grid_and_nn_only_surrogate(self):
        classifier, objective = self._configs(
            {
                "name": "tob_nn_or",
                "target_fpr": 0.005,
                "trigger_objects": 2,
                "tob_budget": {
                    "mode": "validation_search",
                    "values": [0.0, 0.0005, 0.001, 0.0015, 0.002],
                    "cross_validation_folds": 2,
                },
            }
        )
        candidates, surrogate = resolve_constrained_or_budget(
            classifier, objective
        )
        self.assertEqual(
            [candidate.tob_fpr for candidate in candidates],
            [0.0, 0.0005, 0.001, 0.0015, 0.002],
        )
        self.assertEqual(surrogate.name, "nn_only")
        self.assertIsNone(surrogate.tob_fpr)
        self.assertIsNone(surrogate.tob_budget)

    def test_rank_proxy_or_fixed_budget_becomes_single_candidate(self):
        classifier, objective = self._configs(
            {
                "name": "tob_nn_or",
                "target_fpr": 0.005,
                "trigger_objects": 2,
                "tob_fpr": 0.001,
            }
        )
        candidates, surrogate = resolve_constrained_or_budget(
            classifier, objective
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tob_fpr, 0.001)
        self.assertEqual(surrogate.name, "nn_only")

    def test_legacy_fixed_threshold_or_is_untouched(self):
        classifier, objective = self._configs(
            {
                "name": "tob_nn_or",
                "target_fpr": 0.005,
                "trigger_objects": 2,
                "tob_fpr": 0.001,
            },
            {"primal_objective": "soft_efficiency",
             "proxy_threshold_mode": "fixed"},
        )
        candidates, surrogate = resolve_constrained_or_budget(
            classifier, objective
        )
        self.assertIsNone(candidates)
        self.assertIs(surrogate, classifier)

    def test_budget_search_rejected_without_rank_proxy(self):
        classifier, objective = self._configs(
            {
                "name": "tob_nn_or",
                "target_fpr": 0.005,
                "trigger_objects": 2,
                "tob_budget": {
                    "mode": "validation_search",
                    "values": [0.001],
                    "cross_validation_folds": 2,
                },
            },
            {"primal_objective": "soft_efficiency",
             "proxy_threshold_mode": "fixed"},
        )
        with self.assertRaises(ValueError):
            resolve_constrained_or_budget(classifier, objective)

    def test_budget_search_prefers_feasible_over_higher_objective(self):
        records = {
            0.0: {
                "objective_value": 0.30,
                "constraints_satisfied": False,
                "minimum_certified_margin": -0.01,
                "achieved_fpr": 0.004,
                "feasibility": {"mode": "one_sided_confidence"},
            },
            0.001: {
                "objective_value": 0.25,
                "constraints_satisfied": True,
                "minimum_certified_margin": 0.001,
                "achieved_fpr": 0.004,
                "feasibility": {"mode": "one_sided_confidence"},
            },
            0.002: {
                "objective_value": 0.27,
                "constraints_satisfied": True,
                "minimum_certified_margin": 0.002,
                "achieved_fpr": 0.004,
                "feasibility": {"mode": "one_sided_confidence"},
            },
        }

        class _Candidate:
            def __init__(self, budget):
                self.tob_fpr = budget

        best = budget_searched_metrics(
            lambda candidate: dict(records[candidate.tob_fpr]),
            [_Candidate(budget) for budget in (0.0, 0.001, 0.002)],
        )
        self.assertEqual(best["selected_tob_fpr"], 0.002)
        self.assertEqual(
            best["tob_budget_search"]["selected_tob_fpr"], 0.002
        )
        self.assertEqual(len(best["tob_budget_search"]["candidates"]), 3)


if __name__ == "__main__":
    unittest.main()
