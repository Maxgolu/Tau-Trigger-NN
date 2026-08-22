import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (
    CalcThresh,
    build_event_trigger_scores,
    calculate_binned_efficiencies,
    generate_averaged_metrics,
    score_pass_mask,
    select_fpr_threshold,
    select_truth_tau_objects,
    discover_checkpoint_variants,
    evaluate_experiment,
)


class ThresholdCalibrationTests(unittest.TestCase):
    def test_primary_dynamic_or_and_baseline_share_test_recalibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            background_events = np.repeat(np.arange(1000), 2)
            background_rank = np.repeat(np.arange(1000), 2)
            background = pd.DataFrame(
                {
                    "eventNumber": background_events,
                    "Type": "BKG",
                    "signal": 0,
                    "tob_pt": 1_000_000.0 - 100.0 * background_rank,
                    "truth_pt": 0.0,
                    "nn_score": 1.0 - background_rank / 1000.0,
                }
            )
            signal_count = 100
            signal = pd.DataFrame(
                {
                    "eventNumber": np.arange(2000, 2000 + signal_count),
                    "Type": "Signal",
                    "signal": 1,
                    "tob_pt": np.linspace(10_000.0, 120_000.0, signal_count),
                    "truth_pt": np.linspace(10_000.0, 120_000.0, signal_count),
                    "nn_score": np.linspace(0.0, 1.0, signal_count),
                }
            )
            pd.concat([background, signal], ignore_index=True).to_csv(
                folder / "predictions.csv", index=False
            )
            config = {
                "experiment_name": "test_dynamic_or",
                "classifier": {
                    "name": "tob_nn_or",
                    "target_fpr": 0.005,
                    "trigger_objects": 2,
                    "tob_budget": {
                        "mode": "validation_search",
                        "values": [0.0, 0.002, 0.005],
                    },
                },
            }
            with open(folder / "config.json", "w") as output:
                json.dump(config, output)
            validation_calibration = {
                "name": "tob_nn_or",
                "target_fpr": 0.005,
                "tob_fpr_budget": 0.002,
                "trigger_objects": 2,
                "tob_threshold_gev": float("inf"),
                "nn_threshold": float("inf"),
                "achieved_fpr": 0.0,
            }
            manifest = {
                "artifacts": {
                    "target_fpr": {
                        "best_validation_record": {
                            "selected_tob_fpr": 0.002,
                            "classifier_calibration": validation_calibration,
                        }
                    }
                }
            }
            with open(folder / "checkpoint_selection.json", "w") as output:
                json.dump(manifest, output)

            with patch("evaluate.save_turn_on_plot"):
                evaluate_experiment(
                    str(folder),
                    recalc=True,
                    num_bins=11,
                    pt_min=10.0,
                    pt_max=120.0,
                    bin_var="truth_pt",
                    checkpoint_method="target_fpr",
                )

            with open(folder / "metrics.json") as source:
                metrics = json.load(source)

            self.assertEqual(metrics["calibration_split"], "test_recalibrated")
            self.assertEqual(
                metrics["calibration_source_by_fake_rate"]["0.5%"],
                "test_recalibrated",
            )
            self.assertAlmostEqual(
                metrics["achieved_fake_rates"]["0.5%"], 0.005
            )
            self.assertAlmostEqual(
                metrics["baseline_tob_pt"]["achieved_fake_rate"], 0.005
            )
            self.assertNotEqual(
                metrics["classifier_calibrations"]["0.5%"]["nn_threshold"],
                validation_calibration["nn_threshold"],
            )
            self.assertEqual(
                metrics["validation_calibrated_operating_point"][
                    "test_achieved_fake_rate"
                ],
                0.0,
            )

    def test_checkpoint_manifest_exposes_primary_and_secondary_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = {
                "methods": ["validation_bce", "target_fpr"],
                "artifacts": {
                    "validation_bce": {
                        "role": "secondary",
                        "predictions": "predictions_validation_bce.parquet",
                    },
                    "target_fpr": {
                        "role": "primary",
                        "predictions": "predictions.parquet",
                    },
                },
            }
            with open(
                Path(temp_dir) / "checkpoint_selection.json", "w"
            ) as output:
                json.dump(manifest, output)

            variants = discover_checkpoint_variants(temp_dir)

            self.assertEqual(
                variants,
                [
                    (
                        "validation_bce",
                        "validation_bce",
                        "predictions_validation_bce",
                    ),
                    (None, "target_fpr", "predictions"),
                ],
            )

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

    def test_float32_tie_stays_below_next_float64_threshold(self):
        max_score = np.float32(0.08594151586294174)
        threshold = np.nextafter(np.float64(max_score), np.inf)
        frame = pd.DataFrame(
            {"score": np.array([max_score, max_score], dtype=np.float32)}
        )

        passed = score_pass_mask(frame, "score", threshold)

        np.testing.assert_array_equal(passed, [False, False])

    def test_signal_sample_noise_is_not_counted_as_truth_tau(self):
        frame = pd.DataFrame(
            {
                "Type": ["Signal", "Signal", "BKG"],
                "signal": [1, 0, 0],
                "nn_score": [0.8, 0.9, 0.1],
            }
        )

        selected = select_truth_tau_objects(frame)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["nn_score"], 0.8)

    def test_binned_baseline_uses_shared_threshold_semantics(self):
        frame = pd.DataFrame(
            {
                "truth_pt": [15.0, 15.0, 25.0, 25.0],
                "tob_pt": np.array([20.0, 10.0, 30.0, 5.0], dtype=np.float32),
            }
        )

        efficiencies, errors = calculate_binned_efficiencies(
            frame,
            "tob_pt",
            np.float64(20.0),
            np.array([10.0, 20.0, 30.0]),
            "truth_pt",
        )

        np.testing.assert_allclose(efficiencies, [0.5, 0.5])
        self.assertTrue(all(error > 0.0 for error in errors))

    def test_averaged_metrics_include_matching_baseline(self):
        bins = np.linspace(10.0, 70.0, 7).tolist()
        nn_efficiencies = [0.05, 0.2, 0.5, 0.75, 0.9, 0.95]
        errors = [0.02] * 6

        with tempfile.TemporaryDirectory() as temp_dir:
            folders = []
            for seed, baseline_threshold, baseline_fpr in (
                (42, 20.0, 0.0048),
                (123, 22.0, 0.0050),
            ):
                folder = Path(temp_dir) / f"seed_{seed}"
                folder.mkdir()
                folders.append(str(folder))
                metrics = {
                    "thresholds_by_fake_rate": {"0.5%": 0.3},
                    "achieved_fake_rates": {"0.5%": 0.0049},
                    "global_efficiency": {"0.5%": 0.6},
                    "turn_on_curve": {
                        "binning_variable": "truth_pt",
                        "bins": bins,
                        "binned_efficiencies": nn_efficiencies,
                        "binned_efficiencies_err": errors,
                        "target_fake_rate_used": "0.5%",
                    },
                    "baseline_tob_pt": {
                        "threshold": baseline_threshold,
                        "achieved_fake_rate": baseline_fpr,
                        "binned_efficiencies": [
                            value + (seed == 123) * 0.02
                            for value in nn_efficiencies
                        ],
                        "binned_efficiencies_err": errors,
                    },
                }
                with open(folder / "metrics.json", "w") as output:
                    json.dump(metrics, output)

            generate_averaged_metrics("test", folders)

            with open(Path(folders[0]) / "metrics_averaged.json") as source:
                averaged = json.load(source)

            baseline = averaged["baseline_tob_pt"]
            self.assertEqual(baseline["seeds_averaged"], 2)
            self.assertAlmostEqual(baseline["threshold"], 21.0)
            self.assertAlmostEqual(baseline["achieved_fake_rate"], 0.0049)
            np.testing.assert_allclose(
                baseline["binned_efficiencies"],
                np.asarray(nn_efficiencies) + 0.01,
            )


if __name__ == "__main__":
    unittest.main()
