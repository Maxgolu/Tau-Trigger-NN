from __future__ import annotations

import csv
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "representation_reproduction"


def _read(name: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RepresentationReproductionResultTests(unittest.TestCase):
    def test_seed42_screen_has_expected_closure_partition(self) -> None:
        rows = _read("seed42_scalar_coarse_screen.csv")
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({row["candidate"] for row in rows}), 22)
        self.assertTrue(all(int(row["inclusive_denominator"]) == 7307 for row in rows))
        reproduced = [
            row
            for row in rows
            if row["screen_decision"] == "reproduced_across_three_seeds"
        ]
        closed = [
            row
            for row in rows
            if row["screen_decision"] == "closed_after_seed42_screen"
        ]
        self.assertEqual(len(reproduced), 3)
        self.assertEqual(len(closed), 19)
        self.assertEqual(
            {row["candidate"] for row in reproduced},
            {
                "raw_coarse_cells_without_member_pt",
                "raw_coarse_cells_with_member_pt",
                "compact_em2_measurements_with_member_pt",
            },
        )
        for row in rows:
            self.assertAlmostEqual(
                int(row["inclusive_accepted"]) / int(row["inclusive_denominator"]),
                float(row["inclusive_signal_efficiency"]),
            )
            self.assertEqual(row["evidence_status"], "validation_only_unconfirmed")

    def test_global_rows_reconcile_with_frozen_values(self) -> None:
        rows = _read("global_validation.csv")
        expected_parameters = {
            "raw_coarse_cells_without_member_pt": 3601,
            "raw_coarse_cells_with_member_pt": 3633,
            "compact_em2_measurements_with_member_pt": 2705,
        }
        self.assertEqual(len(rows), 9)
        self.assertEqual({int(row["seed"]) for row in rows}, {42, 123, 456})
        self.assertEqual({row["candidate"] for row in rows}, set(expected_parameters))
        for row in rows:
            self.assertEqual(
                int(row["parameter_count"]), expected_parameters[row["candidate"]]
            )
            efficiency = float(row["inclusive_signal_efficiency"])
            baseline = float(row["baseline_efficiency"])
            self.assertAlmostEqual(baseline, 0.34391679211714793)
            self.assertAlmostEqual(
                float(row["candidate_minus_baseline"]), efficiency - baseline
            )
            self.assertLessEqual(float(row["background_fpr"]), 0.005)
            self.assertEqual(row["evidence_status"], "validation_only_unconfirmed")

    def test_summary_is_the_exact_three_seed_mean(self) -> None:
        rows = _read("global_validation.csv")
        summaries = _read("global_summary.csv")
        self.assertEqual(len(summaries), 3)
        for summary in summaries:
            selected = [
                row for row in rows if row["candidate"] == summary["candidate"]
            ]
            self.assertEqual(len(selected), 3)
            mean_efficiency = sum(
                float(row["inclusive_signal_efficiency"]) for row in selected
            ) / 3
            mean_gain = sum(
                float(row["candidate_minus_baseline"]) for row in selected
            ) / 3
            self.assertAlmostEqual(
                float(summary["mean_inclusive_signal_efficiency"]), mean_efficiency
            )
            self.assertAlmostEqual(
                float(summary["mean_candidate_minus_baseline"]), mean_gain
            )
            self.assertEqual(int(summary["seed_count"]), 3)
            self.assertEqual(
                summary["evidence_status"], "validation_only_unconfirmed"
            )

    def test_regional_means_and_direction_reconcile(self) -> None:
        rows = _read("energy_region_deltas.csv")
        expected_denominators = {"10-25": 822, "25-60": 1489, "60+": 1312}
        candidates = {row["candidate"] for row in rows}
        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(rows), 36)
        for candidate in candidates:
            for region, denominator in expected_denominators.items():
                selected = [
                    row
                    for row in rows
                    if row["candidate"] == candidate and row["region_gev"] == region
                ]
                seeds = [row for row in selected if row["seed"] != "mean"]
                mean_row = next(row for row in selected if row["seed"] == "mean")
                self.assertEqual({int(row["seed"]) for row in seeds}, {42, 123, 456})
                self.assertTrue(
                    all(int(row["denominator"]) == denominator for row in selected)
                )
                mean = sum(float(row["candidate_minus_baseline"]) for row in seeds) / 3
                self.assertAlmostEqual(
                    float(mean_row["candidate_minus_baseline"]), mean
                )
                if region == "60+":
                    self.assertTrue(
                        all(float(row["candidate_minus_baseline"]) < 0 for row in seeds)
                    )
                else:
                    self.assertTrue(
                        all(float(row["candidate_minus_baseline"]) > 0 for row in seeds)
                    )
                self.assertTrue(
                    all(
                        row["evidence_status"] == "validation_only_unconfirmed"
                        for row in selected
                    )
                )


if __name__ == "__main__":
    unittest.main()
