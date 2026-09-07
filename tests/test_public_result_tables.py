import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(relative_path):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class PublicResultTableTests(unittest.TestCase):
    def assert_operating_point(self, row):
        background_count = int(row["background_count"])
        background_accepted = int(row["background_accepted"])
        self.assertEqual(background_count, 24835)
        self.assertLessEqual(background_accepted / background_count, 0.005)
        self.assertAlmostEqual(
            float(row["background_fpr"]),
            background_accepted / background_count,
        )
        for population in ("pair_observable", "pair_eligible", "inclusive"):
            accepted = int(row[f"{population}_signal_accepted"])
            count = int(row[f"{population}_signal_count"])
            self.assertAlmostEqual(
                float(row[f"{population}_signal_efficiency"]),
                accepted / count,
            )

    def test_representation_table_has_all_seed_level_evidence(self):
        rows = _rows(
            "results/representation_study/representation_validation_by_seed.csv"
        )
        self.assertEqual(len(rows), 13)
        baseline = [row for row in rows if row["candidate"] == "deterministic_pt_baseline"]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(int(baseline[0]["background_accepted"]), 123)
        candidates = {
            row["candidate"]
            for row in rows
            if row["candidate"] != "deterministic_pt_baseline"
        }
        self.assertEqual(
            candidates,
            {
                "pt_only_pair_control",
                "raw_coarse_cells_plus_member_pt",
                "normalized_em2_summaries_plus_member_pt_and_event_context",
                "raw_em2_summaries_plus_member_pt_and_event_context",
            },
        )
        for candidate in candidates:
            seeds = {
                int(row["seed"])
                for row in rows
                if row["candidate"] == candidate
            }
            self.assertEqual(seeds, {42, 123, 456})
        for row in rows:
            self.assert_operating_point(row)

    def test_high_resolution_thresholds_reconcile(self):
        rows = _rows(
            "results/high_resolution/high_resolution_operating_points.csv"
        )
        self.assertEqual({int(row["seed"]) for row in rows}, {42, 123, 456})
        self.assertEqual({int(row["background_accepted"]) for row in rows}, {124})
        self.assertEqual({int(row["selected_epoch"]) for row in rows}, {4, 5, 10})
        self.assertEqual(len({float(row["learned_threshold"]) for row in rows}), 3)
        for row in rows:
            self.assert_operating_point(row)

    def test_recovery_candidates_include_global_operating_points(self):
        rows = _rows(
            "results/high_energy_recovery/recovery_operating_points.csv"
        )
        candidates = {
            "shared_high_resolution",
            "inverse_frequency_weighting",
            "power_law_p_minus_1",
            "raw_cell_plus_pt_or",
        }
        self.assertEqual({row["candidate"] for row in rows}, candidates)
        self.assertEqual(len(rows), 12)
        for candidate in candidates:
            seeds = {
                int(row["seed"])
                for row in rows
                if row["candidate"] == candidate
            }
            self.assertEqual(seeds, {42, 123, 456})
        for row in rows:
            self.assert_operating_point(row)
            if row["candidate"] == "raw_cell_plus_pt_or":
                self.assertEqual(float(row["pt_threshold_mev"]), 22300.0)
            else:
                self.assertEqual(row["pt_threshold_mev"], "")

    def test_pt_context_results_reconcile_and_retain_reference(self):
        rows = _rows("results/pt_context/validation_by_seed.csv")
        learned = {
            row["candidate"]
            for row in rows
            if row["candidate"] != "deterministic_second_pt_baseline"
        }
        self.assertEqual(len(learned), 8)
        for candidate in learned:
            selected = [row for row in rows if row["candidate"] == candidate]
            self.assertEqual({int(row["seed"]) for row in selected}, {42, 123, 456})
            for row in selected:
                self.assertEqual(int(row["background_count"]), 24835)
                self.assertLessEqual(float(row["background_fpr"]), 0.005)
                self.assertAlmostEqual(
                    float(row["inclusive_signal_efficiency"]),
                    int(row["inclusive_signal_accepted"])
                    / int(row["inclusive_signal_count"]),
                )

        summary = _rows("results/pt_context/summary.csv")
        means = {
            row["candidate"]: float(row["mean_inclusive_signal_efficiency"])
            for row in summary
        }
        reference = means["current_member_pt_and_event_second_pt"]
        baseline = means["deterministic_second_pt_baseline"]
        alternatives = {
            candidate: value
            for candidate, value in means.items()
            if candidate not in {
                "current_member_pt_and_event_second_pt",
                "deterministic_second_pt_baseline",
            }
        }
        self.assertTrue(all(value > baseline for value in alternatives.values()))
        self.assertTrue(all(value < reference for value in alternatives.values()))
        self.assertEqual(max(alternatives, key=alternatives.get), "pair_sum_and_balance")
        for candidate in learned:
            expected = sum(
                float(row["inclusive_signal_efficiency"])
                for row in rows
                if row["candidate"] == candidate
            ) / 3
            self.assertAlmostEqual(means[candidate], expected)

    def test_pt_context_regional_claim_has_per_seed_evidence(self):
        rows = _rows("results/pt_context/validation_regions_by_seed.csv")
        self.assertEqual(len(rows), 28)
        self.assertEqual(
            {row["population"] for row in rows},
            {"inclusive_signal", "low_10_25", "medium_25_60", "high_60_plus"},
        )
        self.assertEqual(len({row["candidate_id"] for row in rows}), 7)
        for row in rows:
            values = [float(row[f"seed_{seed}"]) for seed in (42, 123, 456)]
            self.assertTrue(all(value > 0 for value in values))
            self.assertEqual(row["all_three_strictly_positive"], "True")
            self.assertAlmostEqual(float(row["mean"]), sum(values) / 3)
            self.assertAlmostEqual(float(row["minimum"]), min(values))
            self.assertAlmostEqual(float(row["maximum"]), max(values))


if __name__ == "__main__":
    unittest.main()
