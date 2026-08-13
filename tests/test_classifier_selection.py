import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classifier_selection import (
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
