import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    get_object_partner_context,
    get_object_reference_tob_dr2,
)


class ObjectReferenceDistanceTests(unittest.TestCase):
    def test_reference_selection_and_wrapped_delta_phi(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1, 2],
                "tob_index": [0, 1, 2, 0],
                "tob_pt": [30.0, 20.0, 10.0, 8.0],
                "tob_eta": [0.0, 1.0, 2.0, 0.0],
                "tob_phi": [
                    np.pi - 0.1,
                    -np.pi + 0.1,
                    np.pi - 0.2,
                    0.0,
                ],
            }
        )

        result = get_object_reference_tob_dr2(frame).reshape(-1)

        np.testing.assert_allclose(
            result,
            [1.0 + 0.2**2, 1.0 + 0.2**2, 2.0**2 + 0.1**2, 0.0],
            rtol=0.0,
            atol=1e-6,
        )

    def test_equal_pt_uses_lowest_tob_index_as_leader(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1],
                "tob_index": [2, 0, 1],
                "tob_pt": [10.0, 10.0, 5.0],
                "tob_eta": [2.0, 0.0, 1.0],
                "tob_phi": [0.0, 0.0, 0.0],
            }
        )

        result = get_object_reference_tob_dr2(frame).reshape(-1)

        # tob_index 0 is the leader. It uses tob_index 2 as its reference;
        # every other object uses tob_index 0 as its reference.
        np.testing.assert_allclose(result, [4.0, 4.0, 1.0])


class ObjectPartnerContextTests(unittest.TestCase):
    def test_partner_selection_phi_wrapping_and_output_columns(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1, 2],
                "tob_index": [0, 1, 2, 0],
                "tob_pt": [30.0, 20.0, 10.0, 8.0],
                "tob_eta": [0.0, 1.0, 2.0, 0.0],
                "tob_phi": [
                    np.pi - 0.1,
                    -np.pi + 0.1,
                    np.pi - 0.2,
                    0.0,
                ],
            }
        )

        result = get_object_partner_context(frame)
        expected = np.array(
            [
                [np.log1p(30.0), np.log1p(20.0), np.pi - 0.2, 1.0],
                [np.log1p(20.0), np.log1p(30.0), np.pi - 0.2, 1.0],
                [np.log1p(10.0), np.log1p(30.0), np.pi - 0.1, 2.0],
                [np.log1p(8.0), 0.0, np.pi, 0.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(result.shape, (4, 4))
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-6)

    def test_equal_pt_uses_deterministic_partner_order(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1],
                "tob_index": [2, 0, 1],
                "tob_pt": [10.0, 10.0, 5.0],
                "tob_eta": [2.0, 0.0, 1.0],
                "tob_phi": [0.0, 0.0, 0.0],
            }
        )

        result = get_object_partner_context(frame)

        # tob_index 0 is the leader and uses tob_index 2 as its partner.
        # Both remaining objects use tob_index 0 as their partner.
        np.testing.assert_allclose(
            result[:, 1],
            [np.log1p(10.0), np.log1p(10.0), np.log1p(10.0)],
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(result[:, 3], [2.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
