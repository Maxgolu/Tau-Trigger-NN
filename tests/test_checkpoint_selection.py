import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from checkpoint_selection import (
    calculate_validation_operating_point,
    is_better_checkpoint,
    parse_checkpoint_selection,
)


class CheckpointSelectionTests(unittest.TestCase):
    def test_missing_config_preserves_validation_bce_default(self):
        selection = parse_checkpoint_selection({})

        self.assertEqual(selection.methods, ("validation_bce",))
        self.assertEqual(selection.primary_method, "validation_bce")
        self.assertEqual(selection.target_fpr, 0.005)

    def test_both_methods_allow_target_fpr_primary(self):
        selection = parse_checkpoint_selection(
            {
                "checkpoint_selection": {
                    "methods": ["validation_bce", "target_fpr"],
                    "primary_method": "target_fpr",
                    "target_fpr": 0.01,
                }
            }
        )

        self.assertEqual(
            selection.methods, ("validation_bce", "target_fpr")
        )
        self.assertEqual(selection.primary_method, "target_fpr")
        self.assertEqual(selection.target_fpr, 0.01)

    def test_target_fpr_uses_event_threshold_and_truth_tau_labels(self):
        background_events = 1000
        background = pd.DataFrame(
            {
                "eventNumber": np.repeat(np.arange(background_events), 2),
                "Type": "BKG",
                "signal": 0,
                "truth_pt": 0.0,
            }
        )
        signal = pd.DataFrame(
            {
                "eventNumber": np.arange(2000, 2004),
                "Type": ["Signal"] * 4,
                "signal": [1, 1, 0, 1],
                "truth_pt": [15_000.0, 30_000.0, 50_000.0, 90_000.0],
            }
        )
        frame = pd.concat([background, signal], ignore_index=True)
        event_scores = np.linspace(1.0, 0.001, background_events)
        scores = np.concatenate(
            [np.repeat(event_scores, 2), [1.1, 1.05, 1.2, 0.2]]
        )

        result = calculate_validation_operating_point(frame, scores)

        self.assertEqual(result["achieved_fpr"], 0.005)
        self.assertEqual(result["signal_object_count"], 3)
        self.assertEqual(result["energy_band_efficiencies"]["10-20"], 1.0)
        self.assertEqual(result["energy_band_efficiencies"]["80-120"], 0.0)

    def test_each_method_uses_its_own_objective(self):
        older = {
            "validation_bce": 0.10,
            "signal_efficiency": 0.70,
            "achieved_fpr": 0.005,
        }
        lower_bce_lower_efficiency = {
            "validation_bce": 0.09,
            "signal_efficiency": 0.65,
            "achieved_fpr": 0.005,
        }

        self.assertTrue(
            is_better_checkpoint(
                "validation_bce", lower_bce_lower_efficiency, older
            )
        )
        self.assertFalse(
            is_better_checkpoint(
                "target_fpr", lower_bce_lower_efficiency, older
            )
        )


if __name__ == "__main__":
    unittest.main()
