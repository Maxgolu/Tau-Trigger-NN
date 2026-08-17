import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classifier_selection import (
    _paired_event_standard_error,
    _window_metrics,
    build_validation_folds,
    is_better_budget_candidate,
    search_validation_tob_budget,
)
from classifiers import parse_classifier


def make_validation_frame(background_events=1000):
    background = pd.DataFrame(
        {
            "eventNumber": np.repeat(np.arange(background_events), 2),
            "Type": "BKG",
            "signal": 0,
            "truth_pt": 0.0,
            "tob_pt": np.column_stack(
                [
                    np.linspace(100.0, 5.0, background_events),
                    np.linspace(80.0, 4.0, background_events),
                ]
            ).reshape(-1),
        }
    )
    centers = np.arange(27.5, 120.0, 5.0)
    signal_events = 20 * len(centers)
    signal = pd.DataFrame(
        {
            "eventNumber": np.arange(signal_events),
            "Type": "Signal",
            "signal": 1,
            "truth_pt": np.tile(centers, 20) * 1000.0,
            "tob_pt": np.tile(np.linspace(15.0, 120.0, len(centers)), 20),
        }
    )
    return pd.concat([background, signal], ignore_index=True)


class ValidationFoldTests(unittest.TestCase):
    @staticmethod
    def _objective(mode):
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "tob_budget": {
                        "mode": "validation_search",
                        "values": [0.0],
                        "objective": {"noninferiority_mode": mode},
                    },
                }
            }
        )
        return classifier.tob_budget.objective

    def test_complete_composite_events_remain_in_one_fold(self):
        frame = make_validation_frame()
        folds, audit = build_validation_folds(frame, seed=42)
        keyed = frame.assign(fold=folds).groupby(["Type", "eventNumber"])["fold"]

        self.assertTrue((keyed.nunique() == 1).all())
        self.assertEqual(audit["folds"], 2)
        self.assertGreater(audit["event_counts"]["0"]["background"], 0)
        self.assertGreater(audit["event_counts"]["1"]["signal"], 0)

    def test_budget_order_prioritizes_feasibility_then_worst_window(self):
        best = {
            "noninferiority_satisfied": False,
            "objective_value": 0.3,
            "minimum_delta": -0.2,
            "tob_fpr": 0.004,
        }
        candidate = {
            "noninferiority_satisfied": True,
            "objective_value": 0.01,
            "minimum_delta": -0.004,
            "tob_fpr": 0.001,
        }

        self.assertTrue(is_better_budget_candidate(candidate, best, 0.002))

    def test_feasible_objective_tie_uses_worst_window_then_larger_budget(self):
        best = {
            "noninferiority_satisfied": True,
            "objective_value": 0.0200,
            "minimum_delta": -0.003,
            "tob_fpr": 0.002,
        }
        larger_budget = {
            "noninferiority_satisfied": True,
            "objective_value": 0.0210,
            "minimum_delta": -0.003,
            "tob_fpr": 0.003,
        }

        self.assertTrue(
            is_better_budget_candidate(larger_budget, best, 0.002)
        )

    def test_budget_order_uses_uncertainty_guard_margin_when_available(self):
        best = {
            "noninferiority_satisfied": False,
            "objective_value": 0.02,
            "minimum_delta": 0.01,
            "minimum_guard_margin": -0.03,
            "tob_fpr": 0.002,
        }
        candidate = {
            "noninferiority_satisfied": False,
            "objective_value": 0.01,
            "minimum_delta": 0.00,
            "minimum_guard_margin": -0.01,
            "tob_fpr": 0.001,
        }

        self.assertTrue(is_better_budget_candidate(candidate, best, 0.002))

    def test_paired_uncertainty_is_clustered_by_event(self):
        region = pd.DataFrame(
            {
                "eventNumber": np.repeat([1, 2, 3, 4], 2),
                "or_pass": [True, True, False, False, True, True, True, True],
                "baseline_pass": [
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        object_differences = (
            region["or_pass"].to_numpy(dtype=float)
            - region["baseline_pass"].to_numpy(dtype=float)
        )
        independent_object_error = np.std(object_differences, ddof=1) / np.sqrt(
            len(object_differences)
        )

        clustered_error = _paired_event_standard_error(region)

        self.assertGreater(clustered_error, independent_object_error)

    def test_sparse_saturation_fluctuation_is_pooled(self):
        rows = []
        for low in np.arange(25.0, 120.0, 5.0):
            count = 20
            or_pass = np.ones(count, dtype=bool)
            baseline_pass = np.ones(count, dtype=bool)
            if low == 110.0:
                or_pass[0] = False
            for index in range(count):
                rows.append(
                    {
                        "truth_pt_gev": low + 2.5,
                        "or_pass": bool(or_pass[index]),
                        "baseline_pass": bool(baseline_pass[index]),
                    }
                )
        signal = pd.DataFrame(rows)

        (
            _,
            pooled_regions,
            _,
            pooled_minimum,
            _,
            pooled_complete,
        ) = _window_metrics(
            [signal], self._objective("pooled_saturation")
        )
        _, fine_regions, _, fine_minimum, _, fine_complete = _window_metrics(
            [signal], self._objective("per_window")
        )

        self.assertTrue(pooled_complete)
        self.assertTrue(fine_complete)
        self.assertAlmostEqual(fine_minimum, -1.0 / 20.0)
        self.assertAlmostEqual(pooled_minimum, -1.0 / (12.0 * 20.0))
        self.assertEqual(pooled_regions[-1]["low_gev"], 60.0)
        self.assertEqual(pooled_regions[-1]["high_gev"], 120.0)
        self.assertTrue(pooled_regions[-1]["pooled"])

    def test_multiscale_saturation_uses_uncertainty_aware_overlapping_guards(self):
        rows = []
        event_number = 0
        for low in np.arange(25.0, 120.0, 5.0):
            for index in range(20):
                or_pass = True
                baseline_pass = True
                if low == 60.0 and index < 5:
                    or_pass = False
                elif low == 60.0 and index < 10:
                    baseline_pass = False
                rows.append(
                    {
                        "eventNumber": event_number,
                        "truth_pt_gev": low + 2.5,
                        "or_pass": or_pass,
                        "baseline_pass": baseline_pass,
                    }
                )
                event_number += 1
        signal = pd.DataFrame(rows)
        objective = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "tob_budget": {
                        "mode": "validation_search",
                        "values": [0.0],
                        "objective": {
                            "noninferiority_mode": "multiscale_saturation",
                            "confidence_z": 1.0,
                            "allowed_physical_deficit": 0.0,
                        },
                    },
                }
            }
        ).tob_budget.objective

        _, regions, _, minimum_delta, guard_margin, complete = _window_metrics(
            [signal], objective
        )
        rolling = [
            region
            for region in regions
            if region["region_type"] == "rolling_saturation"
        ]
        full_pool = [
            region
            for region in regions
            if region["region_type"] == "full_saturation_pool"
        ]

        self.assertTrue(complete)
        self.assertEqual(
            [(region["low_gev"], region["high_gev"]) for region in rolling],
            [(60.0, 90.0), (70.0, 100.0), (80.0, 110.0), (90.0, 120.0)],
        )
        self.assertEqual(len(full_pool), 1)
        self.assertAlmostEqual(rolling[0]["delta"], 0.0)
        self.assertGreater(rolling[0]["standard_error"], 0.0)
        self.assertLess(rolling[0]["lower_confidence_bound"], 0.0)
        self.assertFalse(rolling[0]["guard_satisfied"])
        self.assertAlmostEqual(minimum_delta, 0.0)
        self.assertLess(guard_margin, 0.0)

    def test_cross_fitted_search_returns_audited_candidate(self):
        frame = make_validation_frame()
        background_scores = np.column_stack(
            [np.linspace(0.99, 0.01, 1000), np.linspace(0.98, 0.0, 1000)]
        ).reshape(-1)
        signal_scores = np.linspace(0.4, 0.95, len(frame) - len(background_scores))
        scores = np.concatenate([background_scores, signal_scores])
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "target_fpr": 0.005,
                    "tob_budget": {
                        "mode": "validation_search",
                        "values": [0.0, 0.002, 0.004],
                        "cross_validation_folds": 2,
                    },
                }
            }
        )
        folds, audit = build_validation_folds(frame, seed=42)

        result = search_validation_tob_budget(
            frame, scores, classifier, folds, audit
        )

        self.assertIn(result["selected_tob_fpr"], (0.0, 0.002, 0.004))
        self.assertEqual(len(result["tob_budget_search"]["candidates"]), 3)
        self.assertEqual(
            result["tob_budget_search"]["fold_audit"]["seed"], 42
        )


if __name__ == "__main__":
    unittest.main()
