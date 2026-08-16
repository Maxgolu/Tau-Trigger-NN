import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from losses import (
    build_loss,
    calculate_sample_weights,
    fit_loss_weighting,
    parse_loss,
)


class EnergyWeightedBCETests(unittest.TestCase):
    def test_alpha_profile_is_normalized_on_training_signal(self):
        labels = np.asarray([1, 1, 1, 1, 1, 1, 0])
        metadata = pd.DataFrame(
            {"truth_pt": [10_000, 30_000, 35_000, 50_000, 90_000, 110_000, 0]}
        )
        loss = parse_loss(
            {
                "loss": {
                    "name": "energy_weighted_bce",
                    "weighting": {"profile": "alpha", "alpha": 2},
                }
            }
        )

        fitted = fit_loss_weighting(loss, metadata, labels)
        weights = calculate_sample_weights(fitted, metadata, labels)

        self.assertAlmostEqual(float(weights[labels == 1].mean()), 1.0, places=6)
        self.assertEqual(float(weights[-1]), 1.0)
        self.assertGreater(weights[1], weights[3])
        self.assertGreater(weights[3], weights[0])

    def test_inverse_frequency_is_fitted_once_on_training_data(self):
        train_labels = np.ones(5, dtype=np.int64)
        train = pd.DataFrame(
            {"truth_pt": [10_000, 26_000, 27_000, 31_000, 36_000]}
        )
        loss = parse_loss(
            {
                "loss": {
                    "name": "energy_weighted_bce",
                    "weighting": {
                        "profile": "inverse_frequency",
                        "pt_min_gev": 25,
                        "pt_max_gev": 40,
                        "bin_width_gev": 5,
                        "min_weight": 0.2,
                        "max_weight": 5,
                    },
                }
            }
        )

        fitted = fit_loss_weighting(loss, train, train_labels)
        train_weights = calculate_sample_weights(fitted, train, train_labels)
        validation = pd.DataFrame({"truth_pt": [26_000, 31_000, 36_000, 0]})
        validation_labels = np.asarray([1, 1, 1, 0])
        validation_weights = calculate_sample_weights(
            fitted,
            validation,
            validation_labels,
        )

        self.assertAlmostEqual(float(train_weights.mean()), 1.0, places=6)
        self.assertLess(validation_weights[0], validation_weights[1])
        self.assertAlmostEqual(validation_weights[1], validation_weights[2], places=6)
        self.assertEqual(float(validation_weights[-1]), 1.0)

    def test_weighted_bce_matches_explicit_per_object_calculation(self):
        loss = parse_loss(
            {
                "loss": {
                    "name": "energy_weighted_bce",
                    "weighting": {"profile": "alpha", "alpha": 1},
                }
            }
        )
        criterion = build_loss(loss)
        predictions = torch.tensor([[0.8], [0.3]], dtype=torch.float32)
        targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        weights = torch.tensor([[2.0], [1.0]], dtype=torch.float32)

        measured = criterion(predictions, targets, weights)
        expected = (
            torch.nn.functional.binary_cross_entropy(
                predictions,
                targets,
                reduction="none",
            )
            * weights
        ).mean()
        self.assertTrue(torch.allclose(measured, expected))

    def test_invalid_weight_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_loss(
                {
                    "loss": {
                        "name": "energy_weighted_bce",
                        "weighting": {"profile": "unknown"},
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
