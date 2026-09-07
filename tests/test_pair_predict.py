import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_predict import (
    maximum_event_scores,
    pair_scores,
    validation_bce,
    validation_objective,
)


class PairPredictionTests(unittest.TestCase):
    def test_maximum_pair_score_and_no_pair_behavior(self):
        result = maximum_event_scores(
            np.array([1, 1, 2]),
            np.array([-0.2, 0.8, 0.3]),
            np.array([1, 2, 3]),
        )
        np.testing.assert_array_equal(result[:2], [0.8, 0.3])
        self.assertTrue(np.isneginf(result[2]))

    def test_unknown_events_and_nonfinite_pair_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            maximum_event_scores([2], [0.3], [1])
        with self.assertRaises(ValueError):
            maximum_event_scores([1], [np.nan], [1])
        with self.assertRaises(ValueError):
            maximum_event_scores([], [], [1, 1])

    def test_validation_bce_uses_background_and_eligible_signal_pairs(self):
        logits = np.array([0.0, 2.0, -3.0, 4.0])
        labels = np.array([0, 1, 0, 1])
        value = validation_bce(
            logits,
            labels,
            np.array([10, 11, 12, 13]),
            np.array([10, 11, 12, 13]),
            np.array([0, 1, 1, 1]),
            np.array([0, 1, 0, 1]),
        )
        expected = np.mean(
            np.logaddexp(0.0, logits[[0, 1, 3]])
            - labels[[0, 1, 3]] * logits[[0, 1, 3]]
        )
        self.assertAlmostEqual(value, expected)
        with self.assertRaisesRegex(ValueError, "unknown event"):
            validation_bce([0.0], [0], [99], [1], [0], [0])

    def test_three_class_pair_score_is_class_two_probability(self):
        logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.log(2.0)]])
        np.testing.assert_allclose(pair_scores(logits), [1.0 / 3.0, 0.5])

    def test_validation_cross_entropy_uses_the_frozen_population(self):
        logits = np.array(
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]]
        )
        labels = np.array([0, 1, 2])
        result = validation_objective(
            logits,
            labels,
            np.array([10, 11, 12]),
            np.array([10, 11, 12]),
            np.array([0, 1, 1]),
            np.array([0, 1, 0]),
            loss_name="cross_entropy",
        )
        selected = logits[:2]
        expected = np.mean(
            np.log(np.exp(selected).sum(axis=1)) - selected[[0, 1], [0, 1]]
        )
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
