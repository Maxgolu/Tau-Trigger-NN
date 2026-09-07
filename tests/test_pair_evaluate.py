import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_evaluate import (
    assign_validation_folds,
    build_selection_report,
    crossfit_checkpoint,
    evaluate_prediction_table,
    select_checkpoint,
)


class PairEvaluationTests(unittest.TestCase):
    def test_validation_folds_are_deterministic_and_balanced_by_stratum(self):
        event_ids = np.arange(15)
        signal = np.array([0] * 6 + [1] * 9, dtype=bool)
        observable = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        eligible = np.array([0] * 6 + [1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        first = assign_validation_folds(event_ids, signal, observable, eligible)
        second = assign_validation_folds(event_ids, signal, observable, eligible)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), {"A", "B"})

    def _events(self):
        rows = []
        event_id = 0
        for fold in ("A", "B"):
            for index in range(200):
                rows.append({
                    "global_event_id": event_id,
                    "sample": "background",
                    "fold": fold,
                    "pair_eligible": False,
                    "event_score": float(index),
                })
                event_id += 1
            for index in range(20):
                rows.append({
                    "global_event_id": event_id,
                    "sample": "signal",
                    "fold": fold,
                    "pair_eligible": True,
                    "event_score": 180.0 + index,
                })
                event_id += 1
        return pd.DataFrame(rows)

    def test_crossfit_calibrates_on_opposite_folds_and_pools_counts(self):
        result = crossfit_checkpoint(self._events(), target_fpr=0.005)
        self.assertEqual(len(result.legs), 2)
        self.assertEqual(result.background_events, 400)
        self.assertEqual(result.signal_events, 40)
        self.assertEqual(result.signal_accepted, 2)
        self.assertEqual(result.background_accepted, 2)

    def test_no_pair_negative_infinity_remains_in_background_denominator(self):
        events = self._events()
        extra = pd.DataFrame([
            {"global_event_id": 9999, "sample": "background", "fold": "A",
             "pair_eligible": False, "event_score": -np.inf},
            {"global_event_id": 10000, "sample": "background", "fold": "B",
             "pair_eligible": False, "event_score": -np.inf},
        ])
        result = crossfit_checkpoint(pd.concat([events, extra], ignore_index=True))
        self.assertEqual(result.background_events, 402)

    def test_checkpoint_ties_use_background_bce_then_earlier_epoch(self):
        records = [
            {"signal_accepted": 10, "background_accepted": 2, "validation_bce": 0.2, "epoch": 3},
            {"signal_accepted": 10, "background_accepted": 1, "validation_bce": 0.3, "epoch": 4},
            {"signal_accepted": 10, "background_accepted": 1, "validation_bce": 0.2, "epoch": 5},
        ]
        self.assertEqual(select_checkpoint(records)["epoch"], 5)

    def test_checkpoint_bce_values_within_tolerance_choose_earlier_epoch(self):
        records = [
            {"signal_accepted": 10, "background_accepted": 1,
             "validation_bce": 0.2 + 5e-13, "epoch": 2},
            {"signal_accepted": 10, "background_accepted": 1,
             "validation_bce": 0.2, "epoch": 5},
        ]
        self.assertEqual(select_checkpoint(records)["epoch"], 2)

    def test_selection_report_selects_and_recalibrates_each_seed(self):
        rows = []
        for seed in (42, 123):
            for epoch in (1, 2):
                for row in self._events().to_dict(orient="records"):
                    value = dict(row)
                    value.update(
                        {
                            "model_seed": seed,
                            "checkpoint_id": f"checkpoint_epoch_{epoch}",
                            "validation_bce": 0.3 - 0.1 * epoch,
                            "baseline_event_score": row["event_score"],
                            "event_score": row["event_score"]
                            + (20.0 if epoch == 2 and row["sample"] == "signal" else 0.0),
                        }
                    )
                    rows.append(value)
        predictions = pd.DataFrame(rows)
        summary = evaluate_prediction_table(predictions)
        report = build_selection_report(predictions, summary)
        self.assertEqual(set(report["seeds"]), {"42", "123"})
        self.assertEqual(report["seeds"]["42"]["epoch"], 2)
        self.assertIn("full_validation", report["baseline"])
        self.assertIn("full_validation", report["seeds"]["123"])


if __name__ == "__main__":
    unittest.main()
