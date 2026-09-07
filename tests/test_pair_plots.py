from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pair_plots


class PairResultBundleTests(unittest.TestCase):
    def test_pt_control_summary_preserves_operating_point_counts(self) -> None:
        path = REPO_ROOT / "results" / "pt_control" / "validation_summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        baseline = rows[0]
        self.assertEqual(baseline["method"], "deterministic_second_pt_baseline")
        self.assertEqual(int(baseline["background_accepted"]), 123)
        self.assertEqual(int(baseline["background_count"]), 24835)
        self.assertAlmostEqual(float(baseline["threshold"]), 19400.0)
        for row in rows[1:]:
            self.assertEqual(int(row["background_accepted"]), 124)
            self.assertLessEqual(float(row["background_fpr"]), 0.005)

    def test_one_checkpoint_is_selected_per_seed(self) -> None:
        path = REPO_ROOT / "results" / "pt_control" / "checkpoint_selection.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        selected = {
            int(row["seed"]): int(row["epoch"])
            for row in rows
            if row["selected"].lower() == "true"
        }
        self.assertEqual(selected, {42: 4, 123: 5, 456: 18})

    def test_feature_bundle_distinguishes_exact_and_near_duplicates(self) -> None:
        path = (
            REPO_ROOT
            / "results"
            / "training_feature_analysis"
            / "redundant_features.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        exact = [row for row in rows if row["relationship"] == "exact duplicate"]
        near = [row for row in rows if row["relationship"] == "near duplicate"]
        self.assertTrue(all(abs(float(row["spearman"])) == 1.0 for row in exact))
        self.assertEqual(len(near), 1)
        self.assertLess(abs(float(near[0]["spearman"])), 1.0)
        self.assertGreater(abs(float(near[0]["spearman"])), 0.999)

    def test_representation_summary_preserves_baseline_and_best_result(self) -> None:
        path = (
            REPO_ROOT
            / "results"
            / "representation_study"
            / "representation_validation.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        representation = [row for row in rows if row["study"] == "representation"]
        baseline = next(
            row for row in representation
            if row["candidate"] == "deterministic_pt_baseline"
        )
        best = max(representation, key=lambda row: float(row["mean_inclusive_efficiency"]))
        self.assertAlmostEqual(float(baseline["mean_inclusive_efficiency"]), 0.34391679211714793)
        self.assertEqual(
            best["candidate"],
            "raw_em2_summaries_plus_member_pt_and_event_context",
        )
        self.assertGreater(float(best["mean_delta_vs_baseline"]), 0.08)

    def test_or_summary_keeps_event_level_background_budget(self) -> None:
        path = REPO_ROOT / "results" / "pair_or" / "pair_or_validation.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({int(row["seed"]) for row in rows}, {42, 123, 456})
        for row in rows:
            self.assertEqual(int(row["union_accepted"]), 124)
            self.assertEqual(int(row["background_count"]), 24835)
            self.assertLessEqual(float(row["union_fpr"]), 0.005)
            self.assertEqual(row["status"], "pending_protected_confirmation")

    def test_plot_csv_schema_failure_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("present\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                pair_plots._read_rows(path, ("missing",))

    def test_plot_generation_when_matplotlib_is_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed in this test interpreter")
        with tempfile.TemporaryDirectory() as directory:
            outputs = pair_plots.generate_plots(
                REPO_ROOT / "results", Path(directory), pair_plots.PLOTS
            )
            self.assertEqual(len(outputs), len(pair_plots.PLOTS))
            for output in outputs:
                self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
