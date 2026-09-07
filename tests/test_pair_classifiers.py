import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_classifiers import calibrate_joint_or, joint_or_decisions


class PairClassifierTests(unittest.TestCase):
    def test_joint_or_uses_one_background_budget(self):
        pt = np.arange(200, dtype=np.float64)
        neural = np.arange(200, dtype=np.float64)[::-1]
        thresholds = calibrate_joint_or(
            pt,
            neural,
            pt_budget=0.0025,
            target_fpr=0.005,
        )
        decisions = joint_or_decisions(pt, neural, thresholds)
        self.assertLessEqual(int(decisions.sum()), 1)
        self.assertEqual(int(decisions.sum()), thresholds.union_accepted)

    def test_overlap_counts_once(self):
        pt = np.arange(400, dtype=np.float64)
        neural = pt.copy()
        thresholds = calibrate_joint_or(
            pt,
            neural,
            pt_budget=0.005,
            target_fpr=0.005,
        )
        self.assertEqual(thresholds.union_accepted, 2)
        self.assertEqual(thresholds.overlap_accepted, 2)

    def test_no_pair_scores_remain_in_denominator_and_fail(self):
        pt = np.full(200, -np.inf)
        neural = np.full(200, -np.inf)
        thresholds = calibrate_joint_or(
            pt,
            neural,
            pt_budget=0.0025,
            target_fpr=0.005,
        )
        self.assertTrue(np.isposinf(thresholds.pt_threshold))
        self.assertTrue(np.isposinf(thresholds.neural_threshold))
        self.assertEqual(thresholds.union_accepted, 0)

    def test_tied_scores_stay_together(self):
        pt = np.zeros(200, dtype=np.float64)
        neural = np.zeros(200, dtype=np.float64)
        thresholds = calibrate_joint_or(
            pt,
            neural,
            pt_budget=0.005,
            target_fpr=0.005,
        )
        self.assertEqual(thresholds.union_accepted, 0)
        self.assertGreater(thresholds.pt_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
