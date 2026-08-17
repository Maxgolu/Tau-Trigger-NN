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
        self.assertEqual(
            searched.tob_budget.objective.noninferiority_mode,
            "pooled_saturation",
        )
        self.assertEqual(
            searched.tob_budget.objective.saturation_start_truth_pt_gev,
            60.0,
        )
        self.assertEqual(
            searched.tob_budget.objective.protected_max_truth_pt_gev,
            120.0,
        )
        self.assertEqual(
            searched.tob_budget.objective.saturation_window_width_gev,
            30.0,
        )
        self.assertEqual(
            searched.tob_budget.objective.saturation_window_stride_gev,
            10.0,
        )
        self.assertEqual(
            searched.tob_budget.objective.uncertainty_mode,
            "paired_standard_error",
        )
        self.assertEqual(searched.with_tob_fpr(0.002).tob_fpr, 0.002)

    def test_multiscale_saturation_settings_are_configurable(self):
        searched = parse_classifier(
            {
                "classifier": {
                    "name": "tob_nn_or",
                    "tob_budget": {
                        "mode": "validation_search",
                        "values": [0.0],
                        "objective": {
                            "noninferiority_mode": "multiscale_saturation",
                            "saturation_window_width_gev": 20.0,
                            "saturation_window_stride_gev": 5.0,
                            "include_full_saturation_pool": False,
                            "uncertainty_mode": "none",
                            "confidence_z": 1.5,
                            "allowed_physical_deficit": 0.002,
                        },
                    },
                }
            }
        )
        objective = searched.tob_budget.objective

        self.assertEqual(objective.noninferiority_mode, "multiscale_saturation")
        self.assertEqual(objective.saturation_window_width_gev, 20.0)
        self.assertEqual(objective.saturation_window_stride_gev, 5.0)
        self.assertFalse(objective.include_full_saturation_pool)
        self.assertEqual(objective.uncertainty_mode, "none")
        self.assertEqual(objective.confidence_z, 1.5)
        self.assertEqual(objective.allowed_physical_deficit, 0.002)

    def test_invalid_saturation_region_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_classifier(
                {
                    "classifier": {
                        "name": "tob_nn_or",
                        "tob_budget": {
                            "mode": "validation_search",
                            "values": [0.0],
                            "objective": {
                                "saturation_start_truth_pt_gev": 125.0,
                                "protected_max_truth_pt_gev": 120.0,
                            },
                        },
                    }
                }
            )

    def test_invalid_multiscale_stride_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_classifier(
                {
                    "classifier": {
                        "name": "tob_nn_or",
                        "tob_budget": {
                            "mode": "validation_search",
                            "values": [0.0],
                            "objective": {
                                "saturation_window_width_gev": 20.0,
                                "saturation_window_stride_gev": 25.0,
                            },
                        },
                    }
                }
            )


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
