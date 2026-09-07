from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "current_configurations.csv"
SEEDS = {42, 123, 456}
BASELINE_EFFICIENCY = 0.34391679211714793


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CurrentConfigurationSummaryTests(unittest.TestCase):
    def test_summary_points_to_three_complete_seed_families(self) -> None:
        rows = _read_csv(SUMMARY_PATH)
        self.assertEqual(len(rows), 3)
        for row in rows:
            config_dir = REPO_ROOT / row["config_directory"]
            self.assertTrue(config_dir.is_dir())
            configs = sorted(config_dir.glob("*.json"))
            self.assertEqual(len(configs), 3)
            seeds = {
                int(json.loads(path.read_text(encoding="utf-8"))["seed"])
                for path in configs
            }
            self.assertEqual(seeds, SEEDS)
            self.assertEqual(int(row["seed_count"]), 3)
            self.assertEqual(
                row["evidence_status"],
                "adaptive_validation_confirmation_pending",
            )

    def test_summary_efficiencies_match_result_sources(self) -> None:
        rows = {row["configuration"]: row for row in _read_csv(SUMMARY_PATH)}

        hybrid = _read_csv(
            REPO_ROOT / "results/high_resolution_pt_or/validation_summary.csv"
        )
        neural = [
            row
            for row in _read_csv(
                REPO_ROOT / "results/high_resolution/high_resolution_validation.csv"
            )
            if row["method"] == "shared_high_resolution_pair_model"
        ]
        compact = next(
            row
            for row in _read_csv(
                REPO_ROOT / "results/representation_study/representation_validation.csv"
            )
            if row["candidate"]
            == "raw_em2_summaries_plus_member_pt_and_event_context"
        )

        expected = {
            "high_resolution_pair_with_pt_branch": sum(
                float(row["inclusive_signal_efficiency"]) for row in hybrid
            )
            / len(hybrid),
            "high_resolution_pair_neural_only": sum(
                float(row["inclusive_signal_efficiency"]) for row in neural
            )
            / len(neural),
            "compact_shower_pair_neural_only": float(
                compact["mean_inclusive_efficiency"]
            ),
        }

        for name, mean_efficiency in expected.items():
            row = rows[name]
            self.assertAlmostEqual(
                float(row["mean_inclusive_signal_efficiency"]),
                mean_efficiency,
            )
            self.assertAlmostEqual(
                float(row["baseline_efficiency"]), BASELINE_EFFICIENCY
            )
            self.assertAlmostEqual(
                float(row["mean_delta_vs_baseline"]),
                mean_efficiency - BASELINE_EFFICIENCY,
            )


if __name__ == "__main__":
    unittest.main()
