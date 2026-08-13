import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classifiers import (
    calibrate_classifier,
    classifier_event_pass_mask,
    classifier_object_pass_mask,
    parse_classifier,
)
from losses import build_loss, parse_loss


def make_background(event_count=1000):
    event_ids = np.repeat(np.arange(event_count), 2)
    leading = np.linspace(100.0, 1.0, event_count)
    subleading = np.linspace(80.0, 0.5, event_count)
    return pd.DataFrame(
        {
            "eventNumber": event_ids,
            "tob_pt": np.column_stack([leading, subleading]).reshape(-1),
            "nn_score": np.column_stack(
                [
                    np.linspace(0.99, 0.01, event_count),
                    np.linspace(0.98, 0.0, event_count),
                ]
            ).reshape(-1),
        }
    )


class ClassifierConfigTests(unittest.TestCase):
    def test_missing_configs_preserve_nn_only_and_bce_defaults(self):
        classifier = parse_classifier({})
        loss = parse_loss({})

        self.assertEqual(classifier.name, "nn_only")
        self.assertEqual(classifier.target_fpr, 0.005)
        self.assertIsNone(classifier.tob_fpr)
        self.assertEqual(loss.name, "bce")
        self.assertEqual(build_loss(loss).__class__.__name__, "BCELoss")

    def test_or_config_validates_branch_budget(self):
        with self.assertRaises(ValueError):
            parse_classifier(
                {
                    "classifier": {
                        "name": "tob_nn_or",
                        "target_fpr": 0.005,
                        "tob_fpr": 0.006,
                    }
                }
            )

    def test_validation_search_is_parsed_without_changing_fixed_mode(self):
        fixed = parse_classifier(
            {"classifier": {"name": "tob_nn_or", "tob_fpr": 0.004}}
        )
        searched = parse_classifier(
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

        self.assertEqual(fixed.tob_fpr, 0.004)
        self.assertIsNone(fixed.tob_budget)
        self.assertIsNone(searched.tob_fpr)
        self.assertEqual(searched.tob_budget.values, (0.0, 0.002, 0.004))
        self.assertEqual(searched.with_tob_fpr(0.002).tob_fpr, 0.002)


class OrClassifierTests(unittest.TestCase):
    def test_object_or_accepts_either_branch(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 2],
                "tob_pt": [25.0, 5.0, 5.0],
                "nn_score": [0.1, 0.9, 0.1],
            }
        )
        calibration = {
            "name": "tob_nn_or",
            "trigger_objects": 2,
            "tob_threshold_gev": 20.0,
            "nn_threshold": 0.8,
        }

        np.testing.assert_array_equal(
            classifier_object_pass_mask(frame, calibration),
            [True, True, False],
        )
        self.assertTrue(classifier_event_pass_mask(frame, calibration).loc[1])
        self.assertFalse(classifier_event_pass_mask(frame, calibration).loc[2])

    def test_mixed_events_are_counted_during_calibration(self):
        background = pd.DataFrame(
            {
                "eventNumber": np.repeat(np.arange(1000), 2),
                "tob_pt": np.tile([20.0, 1.0], 1000),
                "nn_score": np.column_stack(
                    [np.zeros(1000), np.linspace(1.0, 0.001, 1000)]
                ).reshape(-1),
            }
        )
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "target_fpr": 0.005,
                    "tob_fpr": 0.0,
                }
            }
        )

        calibration = calibrate_classifier(
            background, background["nn_score"], classifier
        )
        measured = classifier_event_pass_mask(background, calibration).mean()

        self.assertEqual(measured, 0.005)
        self.assertEqual(calibration["achieved_fpr"], measured)
        self.assertEqual(
            calibration["diagnostics"]["mixed_only_event_fraction"],
            measured,
        )

    def test_calibration_never_exceeds_total_budget(self):
        background = make_background()
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "target_fpr": 0.005,
                    "tob_fpr": 0.004,
                }
            }
        )

        calibration = calibrate_classifier(
            background, background["nn_score"], classifier
        )
        measured = classifier_event_pass_mask(background, calibration).mean()

        self.assertLessEqual(measured, 0.005)
        self.assertEqual(calibration["achieved_fpr"], measured)

    def test_score_ties_are_not_split(self):
        background = pd.DataFrame(
            {
                "eventNumber": np.repeat(np.arange(1000), 2),
                "tob_pt": 0.0,
                "nn_score": 1.0,
            }
        )
        classifier = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "target_fpr": 0.005,
                    "tob_fpr": 0.0,
                }
            }
        )

        calibration = calibrate_classifier(
            background, background["nn_score"], classifier
        )

        self.assertEqual(calibration["achieved_fpr"], 0.0)
        self.assertGreater(calibration["nn_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
