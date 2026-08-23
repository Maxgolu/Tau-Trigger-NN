import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constrained_objective import (
    build_confidence_feasibility,
    calculate_soft_constraint_metrics,
    certified_calibration_target,
    kth_event_score,
    one_sided_binomial_upper_bound,
    paired_difference_interval,
    parse_constrained_objective,
    probability_at_least_k,
    rank_calibrated_threshold,
    soft_object_pass,
    tail_ranking_objective,
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

    def test_confidence_configuration_is_explicit_and_validated(self):
        configured = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "validation_crossfit": True,
                    "feasibility_confidence_level": 0.95,
                }
            }
        )
        self.assertTrue(configured.validation_crossfit)
        self.assertEqual(configured.feasibility_confidence_level, 0.95)
        with self.assertRaises(ValueError):
            parse_constrained_objective(
                {
                    "loss": {
                        "name": "constrained_trigger",
                        "feasibility_confidence_level": 0.5,
                    }
                }
            )

    def test_certified_calibration_target_meets_exact_upper_bound(self):
        event_count = 25_000
        target = certified_calibration_target(event_count, 0.005, 0.95)
        accepted = int(round(target * event_count))
        self.assertLessEqual(
            one_sided_binomial_upper_bound(accepted, event_count, 0.95),
            0.005,
        )
        self.assertGreater(
            one_sided_binomial_upper_bound(accepted + 1, event_count, 0.95),
            0.005,
        )

    def test_paired_lower_bound_rejects_zero_point_margin(self):
        statistics = {
            "cluster_count": 100,
            "object_count": 100,
            "difference_sum": 0.0,
            "difference_square_sum": 20.0,
            "difference_count_product_sum": 0.0,
            "count_square_sum": 100.0,
        }
        interval = paired_difference_interval(statistics, 0.95)
        self.assertEqual(interval["estimate"], 0.0)
        self.assertLess(interval["lower_confidence_bound"], 0.0)

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

    def test_objective_and_constraint_regions_are_independent(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "objective_regions_gev": [[25, 32], [32, 40], [40, 60]],
                    "objective_region_weights": [0.35, 0.35, 0.30],
                    "constraint_regions_gev": [
                        [25, 32], [32, 40], [40, 60], [60, 120]
                    ],
                    "allowed_deficits": [0.005] * 4,
                    "minimum_region_advantages": [0.0025, 0.0025, 0.0025, 0.0],
                }
            }
        )
        self.assertEqual(len(config.objective_regions_gev), 3)
        self.assertEqual(len(config.constraint_regions_gev), 4)
        probabilities = torch.tensor(
            [[0.8, 0.7, 0.6, 0.5]], requires_grad=True
        )
        mask = torch.ones_like(probabilities, dtype=torch.bool)
        metrics = calculate_soft_constraint_metrics(
            probabilities,
            mask,
            mask,
            torch.tensor([True]),
            torch.tensor([[28.0, 35.0, 50.0, 80.0]]),
            torch.zeros_like(mask),
            config,
        )
        self.assertEqual(tuple(metrics.objective_region_deltas.shape), (3,))
        self.assertEqual(tuple(metrics.region_deltas.shape), (4,))
        self.assertEqual(tuple(metrics.violations.shape), (5,))

    def test_temperature_schedule_reaches_both_endpoints(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "temperature_start": 0.1,
                    "temperature_end": 0.02,
                    "temperature_schedule": "linear",
                }
            }
        )
        self.assertAlmostEqual(config.temperature_at(0, 10), 0.1)
        self.assertAlmostEqual(config.temperature_at(9, 10), 0.02)

    def test_rank_threshold_and_tail_loss_ignore_common_logit_shift(self):
        config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    "primal_objective": "tail_ranking",
                    "objective_regions_gev": [[25, 60]],
                    "objective_region_weights": [1.0],
                    "constraint_regions_gev": [[25, 120]],
                    "allowed_deficits": [0.005],
                    "tail_fraction": 0.5,
                    "tail_min_events": 1,
                }
            }
        )
        logits = torch.tensor(
            [[-2.0, -1.0], [-0.5, 0.2], [0.1, 0.8], [1.0, 1.5]],
            requires_grad=True,
        )
        mask = torch.ones_like(logits, dtype=torch.bool)
        background = torch.tensor([True, True, False, False])
        signal = torch.tensor(
            [[False, False], [False, False], [True, True], [True, True]]
        )
        truth_pt = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [30.0, 35.0], [45.0, 50.0]]
        )
        threshold = rank_calibrated_threshold(
            logits, mask, background, 0.5, trigger_objects=2
        )
        shifted_threshold = rank_calibrated_threshold(
            logits + 7.0, mask, background, 0.5, trigger_objects=2
        )
        self.assertAlmostEqual(
            float((shifted_threshold - threshold).detach()), 7.0
        )
        objective, loss, offsets, tail_count = tail_ranking_objective(
            logits,
            mask,
            signal,
            background,
            truth_pt,
            config,
        )
        shifted = tail_ranking_objective(
            logits + 7.0,
            mask,
            signal,
            background,
            truth_pt,
            config,
        )
        self.assertAlmostEqual(
            float(loss.detach()), float(shifted[1].detach()), places=6
        )
        self.assertTrue(torch.allclose(offsets, shifted[2]))
        self.assertEqual(tail_count, 1)
        self.assertAlmostEqual(float(objective.detach()), -float(loss.detach()))

    def test_kth_event_score_uses_second_largest_object(self):
        logits = torch.tensor([[0.1, 0.8, 0.3], [1.0, -1.0, 0.5]])
        mask = torch.tensor([[True, True, True], [True, False, False]])
        result = kth_event_score(logits, mask, 2)
        self.assertAlmostEqual(float(result[0]), 0.3)
        self.assertTrue(torch.isneginf(result[1]))

    @staticmethod
    def _feasibility_config(**extra_loss_fields):
        loss = {
            "name": "constrained_trigger",
            "constraint_regions_gev": [[60, 120]],
            "objective_regions_gev": [[60, 120]],
            "objective_region_weights": [1.0],
            "allowed_deficits": [0.005],
            "minimum_region_advantages": [0.0],
            "reference_model_allowed_deficits": [0.005],
            "feasibility_confidence_level": 0.95,
        }
        loss.update(extra_loss_fields)
        return parse_constrained_objective({"loss": loss})

    @staticmethod
    def _saturation_region_statistics():
        # About +0.004 estimated delta with a paired standard error near
        # 0.0046: certified-positive fails, tolerance-adjusted passes.
        return {
            "cluster_count": 1600,
            "object_count": 1939,
            "difference_sum": 8.0,
            "difference_square_sum": 80.0,
            "difference_count_product_sum": 9.7,
            "count_square_sum": 2351.0,
        }

    @staticmethod
    def _zero_difference_statistics():
        return {
            "cluster_count": 1600,
            "object_count": 1939,
            "difference_sum": 0.0,
            "difference_square_sum": 0.0,
            "difference_count_product_sum": 0.0,
            "count_square_sum": 2351.0,
        }

    def test_certified_fpr_veto_remains_the_legacy_default(self):
        config = self._feasibility_config()
        self.assertEqual(config.fpr_feasibility_mode, "certified")
        self.assertFalse(config.certified_guards_use_allowed_deficits)
        # 119 / 24835 = 0.00479 measured, but its 95% upper bound tops 0.005.
        feasibility = build_confidence_feasibility(
            config,
            119,
            24_835,
            [self._zero_difference_statistics()],
            [self._zero_difference_statistics()],
        )
        self.assertFalse(feasibility["fpr"]["satisfied"])
        self.assertLess(feasibility["fpr"]["certified_margin"], 0.0)
        self.assertGreater(feasibility["fpr"]["point_margin"], 0.0)

    def test_point_fpr_mode_accepts_measured_rate_below_target(self):
        config = self._feasibility_config(fpr_feasibility_mode="point")
        feasibility = build_confidence_feasibility(
            config,
            119,
            24_835,
            [self._zero_difference_statistics()],
            [self._zero_difference_statistics()],
        )
        self.assertTrue(feasibility["fpr"]["satisfied"])
        self.assertEqual(feasibility["fpr"]["feasibility_mode"], "point")
        # The certified bound stays available as a diagnostic.
        self.assertLess(feasibility["fpr"]["certified_margin"], 0.0)
        self.assertEqual(
            feasibility["fpr"]["binding_margin"],
            feasibility["fpr"]["point_margin"],
        )
        self.assertTrue(feasibility["constraints_satisfied"])

    def test_certified_guards_apply_the_configured_tolerance(self):
        statistics = self._saturation_region_statistics()
        strict = self._feasibility_config(fpr_feasibility_mode="point")
        strict_result = build_confidence_feasibility(
            strict,
            80,
            24_835,
            [statistics],
            [self._zero_difference_statistics()],
        )
        # Without the tolerance the certified lower bound must clear zero,
        # which a saturated region cannot do at this sample size.
        self.assertFalse(strict_result["regions"][0]["satisfied"])
        tolerant = self._feasibility_config(
            fpr_feasibility_mode="point",
            certified_guards_use_allowed_deficits=True,
        )
        tolerant_result = build_confidence_feasibility(
            tolerant,
            80,
            24_835,
            [statistics],
            [self._zero_difference_statistics()],
        )
        record = tolerant_result["regions"][0]["candidate_minus_baseline"]
        self.assertAlmostEqual(record["required_minimum"], -0.005)
        self.assertTrue(tolerant_result["regions"][0]["satisfied"])
        self.assertTrue(tolerant_result["constraints_satisfied"])

    def test_invalid_fpr_feasibility_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._feasibility_config(fpr_feasibility_mode="lenient")


if __name__ == "__main__":
    unittest.main()
