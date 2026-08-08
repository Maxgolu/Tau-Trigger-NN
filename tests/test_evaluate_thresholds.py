import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import CalcThresh, build_event_trigger_scores, select_fpr_threshold


class ThresholdCalibrationTests(unittest.TestCase):
    def test_continuous_scores_use_largest_allowed_event_budget(self):
        frame = pd.DataFrame(
            {
                "eventNumber": np.repeat(np.arange(1000), 2),
                "score": np.column_stack(
                    [np.linspace(1.0, 0.001, 1000), np.linspace(0.999, 0.0, 1000)]
                ).reshape(-1),
            }
        )

        threshold, achieved = CalcThresh(frame, "score", 0.005, objects=2)

        self.assertLessEqual(achieved, 0.005)
        self.assertEqual(achieved, 0.005)
        event_scores, event_count = build_event_trigger_scores(frame, "score", objects=2)
        measured = np.count_nonzero(event_scores >= threshold) / event_count
        self.assertEqual(measured, achieved)

    def test_discrete_tie_is_not_split_or_mislabeled(self):
        frame = pd.DataFrame(
            {
                "eventNumber": np.repeat(np.arange(1000), 2),
                "score": np.where(np.repeat(np.arange(1000), 2) < 900, 1.0, 0.0),
            }
        )

        threshold, achieved = CalcThresh(frame, "score", 0.005, objects=2)

        self.assertGreater(threshold, 1.0)
        self.assertEqual(achieved, 0.0)

    def test_threshold_and_measurement_share_greater_equal_semantics(self):
        event_scores = np.array([0.9, 0.8, 0.8, 0.1])
        threshold, achieved = select_fpr_threshold(event_scores, 4, 0.75)

        self.assertEqual(threshold, 0.8)
        self.assertEqual(achieved, 0.75)


if __name__ == "__main__":
    unittest.main()
