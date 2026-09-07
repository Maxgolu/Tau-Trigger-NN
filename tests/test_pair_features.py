import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_REGISTRY
from src.pair_features import (
    em2_best_3x3_fraction,
    member_summary_features,
    raw_cell_pair_features,
    raw_cell_pt_pair_features,
    summary_pair_features,
    summary_pt_pair_features,
)


class PairFeatureTests(unittest.TestCase):
    def test_raw_cells_preserve_member_local_order(self):
        left = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
        right = (100 + np.arange(45, dtype=np.float32)).reshape(5, 3, 3)

        result = raw_cell_pair_features(left, right)

        self.assertEqual(result.shape, (90,))
        np.testing.assert_array_equal(result[:45], left.reshape(-1))
        np.testing.assert_array_equal(result[45:], right.reshape(-1))

    def test_raw_cells_with_pt_use_positions_45_and_91(self):
        left = np.arange(45, dtype=np.float32).reshape(5, 3, 3)
        right = (100 + np.arange(45, dtype=np.float32)).reshape(5, 3, 3)

        result = raw_cell_pt_pair_features(left, 25_000.0, right, 14_000.0)

        self.assertEqual(result.shape, (92,))
        np.testing.assert_array_equal(result[:45], left.reshape(-1))
        self.assertEqual(result[45], 25_000.0)
        np.testing.assert_array_equal(result[46:91], right.reshape(-1))
        self.assertEqual(result[91], 14_000.0)

    def test_summary_formula_golden_values(self):
        core = np.zeros((5, 3, 3), dtype=np.float32)
        core[0, 0, 0] = 10.0
        core[0, 0, 1] = 1.0
        em2 = np.zeros((12, 12), dtype=np.float32)
        em2[:3, :3] = 1.0
        em2[11, 11] = 1.0

        result = member_summary_features(core, em2)

        self.assertEqual(result.shape, (16,))
        self.assertEqual(result[0], 1.0)  # strict > 10% maximum
        self.assertEqual(result[5], 11.0)
        self.assertEqual(result[10], 9.0)
        self.assertAlmostEqual(float(result[15]), 0.9, places=6)

    def test_all_zero_inputs_have_zero_summaries(self):
        result = member_summary_features(
            np.zeros((5, 3, 3), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
        )
        np.testing.assert_array_equal(result, np.zeros(16, dtype=np.float32))

    def test_best_window_checks_all_100_positions(self):
        image = np.zeros((12, 12), dtype=np.float32)
        image[9:12, 9:12] = 2.0
        image[0, 0] = 1.0
        self.assertAlmostEqual(
            float(em2_best_3x3_fraction(image)), 18.0 / 19.0, places=6
        )

    def test_summary_pair_layout_and_pt_positions(self):
        left_core = np.ones((5, 3, 3), dtype=np.float32)
        right_core = np.full((5, 3, 3), 2.0, dtype=np.float32)
        left_em2 = np.ones((12, 12), dtype=np.float32)
        right_em2 = np.full((12, 12), 2.0, dtype=np.float32)
        summaries = summary_pair_features(
            left_core, left_em2, right_core, right_em2
        )
        result = summary_pt_pair_features(
            left_core, left_em2, 30_000.0,
            right_core, right_em2, 20_000.0,
        )

        self.assertEqual(summaries.shape, (32,))
        self.assertEqual(result.shape, (34,))
        np.testing.assert_array_equal(result[:16], summaries[:16])
        self.assertEqual(result[16], 30_000.0)
        np.testing.assert_array_equal(result[17:33], summaries[16:])
        self.assertEqual(result[33], 20_000.0)

    def test_batch_mapping(self):
        left = np.zeros((3, 5, 3, 3), dtype=np.float32)
        right = np.ones((3, 5, 3, 3), dtype=np.float32)
        left_em2 = np.zeros((3, 12, 12), dtype=np.float32)
        right_em2 = np.ones((3, 12, 12), dtype=np.float32)

        self.assertEqual(raw_cell_pair_features(left, right).shape, (3, 90))
        self.assertEqual(
            summary_pair_features(left, left_em2, right, right_em2).shape,
            (3, 32),
        )

    def test_summaries_match_registered_single_object_features(self):
        generator = np.random.default_rng(42)
        core = generator.uniform(0.0, 20.0, size=(4, 5, 3, 3)).astype(
            np.float32
        )
        em2 = generator.uniform(0.0, 20.0, size=(4, 12, 12)).astype(
            np.float32
        )
        frame = pd.DataFrame(
            np.column_stack((core.reshape(4, 45), em2.reshape(4, 144))),
            columns=(
                [f"tensor_{index}" for index in range(45)]
                + [f"em2_cell_{index}" for index in range(144)]
            ),
        )
        names = (
            [f"{layer}_sparsity" for layer in ("em0", "em1", "em2_3x3", "em3", "had")]
            + [f"{layer}_sum" for layer in ("em0", "em1", "em2_3x3", "em3", "had")]
            + [
                f"{layer}_dominance"
                for layer in ("em0", "em1", "em2_3x3", "em3", "had")
            ]
        )
        expected = np.column_stack(
            [FEATURE_REGISTRY[name](frame).reshape(-1) for name in names]
        )
        result = member_summary_features(core, em2)

        np.testing.assert_allclose(
            result[:, :15],
            expected,
            rtol=3e-7,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            result[:, 15],
            FEATURE_REGISTRY["em2_best_3x3_fraction"](frame).reshape(-1),
            rtol=0.0,
            atol=1e-7,
        )

    def test_non_finite_inputs_are_rejected(self):
        core = np.zeros((5, 3, 3), dtype=np.float32)
        core[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            raw_cell_pair_features(core, np.zeros_like(core))


if __name__ == "__main__":
    unittest.main()
