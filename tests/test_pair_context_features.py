import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pair_context_features import build_pair_context_features
from pair_data import build_pair_dataset


class PairContextFeatureTests(unittest.TestCase):
    def setUp(self):
        self.objects = pd.DataFrame({
            "eventNumber": [7, 7, 7, 7, 7],
            "tob_index": [0, 1, 2, 3, 4],
            "tob_pt": [40.0, 30.0, 20.0, 10.0, 5.0],
            "signal": [1, 1, 0, 0, 0],
        })
        self.dataset = build_pair_dataset(self.objects)
        self.mapped = build_pair_context_features(
            self.objects,
            self.dataset.pairs,
        )

    def test_exact_event_and_outside_pair_features(self):
        row = self.mapped.iloc[0]
        self.assertEqual(row["event_sum_pt"], 105.0)
        self.assertEqual(row["outside_pair_max_pt"], 20.0)
        self.assertEqual(row["outside_pair_sum_pt"], 35.0)
        self.assertAlmostEqual(row["pair_fraction_event_pt"], 70.0 / 105.0)

    def test_top_four_concentration_and_entropy(self):
        fractions = np.array([0.4, 0.3, 0.2, 0.1])
        self.assertAlmostEqual(
            self.mapped.iloc[0]["top4_pt_concentration"],
            float(np.square(fractions).sum()),
        )
        self.assertAlmostEqual(
            self.mapped.iloc[0]["top4_pt_entropy"],
            float(-(fractions * np.log(fractions)).sum()),
        )

    def test_pair_sum_and_balance(self):
        row = self.mapped.iloc[0]
        self.assertEqual(row["pair_sum_pt"], 70.0)
        self.assertAlmostEqual(row["pair_balance"], 10.0 / 70.0)

    def test_reversed_or_incomplete_pairs_are_rejected(self):
        reversed_pair = self.dataset.pairs.copy()
        reversed_pair.loc[0, ["tob_index_a", "tob_index_b"]] = [1, 0]
        reversed_pair.loc[0, ["tob_pt_a", "tob_pt_b"]] = [30.0, 40.0]
        with self.assertRaisesRegex(ValueError, "canonical"):
            build_pair_context_features(self.objects, reversed_pair)
        with self.assertRaisesRegex(ValueError, "selected top four"):
            build_pair_context_features(self.objects, self.dataset.pairs.iloc[:-1])

    def test_safe_zero_ratios(self):
        objects = pd.DataFrame({
            "eventNumber": [2, 2],
            "tob_index": [0, 1],
            "tob_pt": [0.0, 0.0],
            "signal": [0, 0],
        })
        pairs = build_pair_dataset(objects).pairs
        mapped = build_pair_context_features(objects, pairs)
        self.assertEqual(mapped.loc[0, "pair_fraction_event_pt"], 0.0)
        self.assertEqual(mapped.loc[0, "pair_balance"], 0.0)
        self.assertEqual(mapped.loc[0, "top4_pt_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
