import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pair_models import build_pair_model, count_parameters
from prepare_pair_data import preparation_contract

FACTORIAL = {
    "pair_highres_member_pt_event_pt": (True, True, 47, 1, 4_321),
    "pair_highres_member_pt_no_event_pt": (True, False, 47, 0, 4_289),
    "pair_highres_no_member_pt_no_event_pt": (False, False, 46, 0, 4_273),
    "pair_highres_no_member_pt_event_pt": (False, True, 46, 1, 4_305),
}


class PublicPtFactorialTests(unittest.TestCase):
    def _configs(self, family):
        return sorted((ROOT / "configs" / family).glob("*.json"))

    def test_four_factorial_cells_are_executable_for_all_seeds(self):
        for family, expected in FACTORIAL.items():
            member_pt, event_pt, scalar_width, context_width, parameters = expected
            paths = self._configs(family)
            self.assertEqual(len(paths), 3)
            configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            self.assertEqual({config["seed"] for config in configs}, {42, 123, 456})
            for config in configs:
                self.assertEqual(config["include_member_pt"], member_pt)
                self.assertEqual(config["include_event_pt"], event_pt)
                contract = preparation_contract(config)
                self.assertEqual(contract[3:5], (member_pt, event_pt))
                model = build_pair_model(config)
                self.assertEqual(model.member_scalar_width, scalar_width)
                self.assertEqual(model.context_width, context_width)
                self.assertEqual(count_parameters(model), parameters)

    def test_high_resolution_pt_or_contract_uses_one_union_budget(self):
        paths = self._configs("pair_highres_powerlaw_pt_or")
        self.assertEqual(len(paths), 3)
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            classifier = config["event_classifier"]
            self.assertEqual(classifier["name"], "neural_or_pair_min_pt")
            self.assertEqual(classifier["total_background_event_fpr"], 0.005)
            self.assertEqual(classifier["pt_branch_budget"], 0.001)
            self.assertEqual(classifier["overlap_counting"], "union")
            self.assertEqual(count_parameters(build_pair_model(config)), 4_321)

    def test_factorial_result_table_matches_the_presented_validation_claims(self):
        path = ROOT / "results" / "pt_factorial" / "validation_metrics.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual({int(row["seed"]) for row in rows}, {42, 123, 456})
        self.assertEqual({row["candidate"] for row in rows}, {
            "member_pt_and_event_pt",
            "member_pt_only",
            "no_member_or_event_pt",
            "event_pt_only",
        })
        self.assertTrue(all(int(row["background_accepted"]) == 124 for row in rows))
        self.assertTrue(all(int(row["background_count"]) == 24_835 for row in rows))
        self.assertTrue(all(float(row["background_fpr"]) <= 0.005 for row in rows))

    def test_factorial_summary_preserves_the_four_reported_neural_gains(self):
        path = ROOT / "results" / "pt_factorial" / "summary.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        neural = {
            row["candidate"]: float(row["mean_delta_vs_baseline_percentage_points"])
            for row in rows if row["classifier"] == "nn_only"
        }
        self.assertAlmostEqual(neural["no_member_or_event_pt"], 7.3810501345741555)
        self.assertAlmostEqual(neural["member_pt_only"], 7.435792162766298)
        self.assertAlmostEqual(neural["event_pt_only"], 9.538798412481183)
        self.assertAlmostEqual(neural["member_pt_and_event_pt"], 10.419232699238176)

    def test_high_resolution_pt_or_result_is_jointly_calibrated(self):
        path = ROOT / "results" / "high_resolution_pt_or" / "operating_points.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        full = [row for row in rows if row["scope"] == "full_validation"]
        self.assertEqual(len(full), 3)
        self.assertTrue(all(int(row["calibration_union_accepted"]) == 124 for row in full))
        self.assertTrue(all(int(row["calibration_background_count"]) == 24_835 for row in full))
        self.assertTrue(all(float(row["calibration_union_fpr"]) <= 0.005 for row in full))

        summary_path = (
            ROOT / "results" / "high_resolution_pt_or" / "validation_summary.csv"
        )
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual([int(row["selected_epoch"]) for row in summary], [17, 20, 8])
        self.assertTrue(all(float(row["delta_vs_baseline"]) > 0.0 for row in summary))

    def test_public_evidence_contains_no_private_paths_or_internal_cell_ids(self):
        roots = [ROOT / "results" / "pt_factorial", ROOT / "results" / "high_resolution_pt_or"]
        text = "\n".join(path.read_text(encoding="utf-8") for root in roots for path in root.glob("*.csv"))
        self.assertNotIn("/Users/", text)
        self.assertNotIn("phase8", text)
        for cell in ("A00", "A01", "A02", "A03"):
            self.assertNotIn(cell, text)


if __name__ == "__main__":
    unittest.main()
