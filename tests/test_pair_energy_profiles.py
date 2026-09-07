import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_energy_profiles import (
    CONDITIONAL_MATCHED_MEMBER_EVENT_COLUMNS,
    CONDITIONAL_MATCHED_MEMBER_LIMITATIONS,
    CONDITIONAL_MATCHED_MEMBER_TERMINOLOGY,
    build_conditional_matched_member_events,
    paired_count_summary,
    summarize_conditional_matched_members,
)


class PairEnergyProfileTests(unittest.TestCase):
    @staticmethod
    def _builder_inputs():
        pairs = pd.DataFrame(
            {
                "global_event_id": [10, 10, 10, 20],
                "pair_index": [0, 1, 2, 0],
                "tob_index_a": [0, 0, 1, 0],
                "tob_index_b": [1, 2, 2, 1],
                "selected_rank_a": [0, 0, 1, 0],
                "selected_rank_b": [1, 2, 2, 1],
                "three_class_label": [2, 2, 2, 0],
                "split": ["validation"] * 4,
            }
        )
        objects = pd.DataFrame(
            {
                "global_event_id": [10, 10, 10, 20, 20],
                "tob_index": [0, 1, 2, 0, 1],
                "signal": [1, 1, 1, 0, 0],
                # Equal values are intentionally allowed: this diagnostic does
                # not interpret equality as a distinct-particle identity.
                "truth_pt": [30_000.0, 20_000.0, 20_000.0, 90_000.0, 80_000.0],
                "split": ["validation"] * 5,
            }
        )
        decisions = pd.DataFrame(
            {
                "global_event_id": [10, 20, 10, 20],
                "model_seed": [42, 42, 123, 123],
                "model_pass": [True, False, False, False],
                "baseline_pass": [False, False, False, False],
                "split": ["validation"] * 4,
            }
        )
        return pairs, objects, decisions

    def test_builder_reconstructs_operational_label2_event_table(self):
        pairs, objects, decisions = self._builder_inputs()
        result = build_conditional_matched_member_events(
            pairs, objects, decisions
        )

        self.assertEqual(
            tuple(result.columns), CONDITIONAL_MATCHED_MEMBER_EVENT_COLUMNS
        )
        self.assertEqual(result["global_event_id"].tolist(), [10, 10])
        self.assertEqual(result["model_seed"].tolist(), [42, 123])
        self.assertEqual(
            result["member_l_associated_truth_pt_gev"].tolist(), [30.0, 30.0]
        )
        self.assertEqual(
            result["member_s_associated_truth_pt_gev"].tolist(), [20.0, 20.0]
        )
        self.assertEqual(result["model_pass"].tolist(), [True, False])
        self.assertEqual(result["baseline_pass"].tolist(), [False, False])
        self.assertTrue(result["has_operational_label2_pair"].all())

        profile = summarize_conditional_matched_members(result)
        low = profile.regions.set_index(["model_seed", "region"])
        self.assertEqual(low.loc[(42, "low_10_25"), "model_accepted"], 1)
        self.assertEqual(low.loc[(123, "low_10_25"), "baseline_accepted"], 0)

    def test_builder_uses_lowest_pair_index_not_truth_values(self):
        pairs, objects, decisions = self._builder_inputs()
        event_10 = pairs["global_event_id"].eq(10)
        pairs.loc[event_10, "pair_index"] = [7, 3, 9]
        objects.loc[
            objects["global_event_id"].eq(10) & objects["tob_index"].eq(2),
            "truth_pt",
        ] = 10_000.0
        result = build_conditional_matched_member_events(
            pairs, objects, decisions
        )

        # Pair index 3 is (TOB 0, TOB 2), so the S-associated value is 10 GeV.
        # Selection depends only on the operational labels and pair index.
        self.assertEqual(
            result["member_s_associated_truth_pt_gev"].unique().tolist(), [10.0]
        )

    def test_builder_accepts_explicit_gev_without_rescaling(self):
        pairs, objects, decisions = self._builder_inputs()
        objects["truth_pt"] /= 1000.0
        result = build_conditional_matched_member_events(
            pairs, objects, decisions, truth_pt_unit="GeV"
        )
        self.assertEqual(
            result["member_l_associated_truth_pt_gev"].unique().tolist(), [30.0]
        )

    def test_builder_rejects_stored_pair_label_disagreement(self):
        pairs, objects, decisions = self._builder_inputs()
        pairs.loc[0, "three_class_label"] = 1
        with self.assertRaisesRegex(ValueError, "stored pair labels disagree"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_builder_rejects_noncanonical_or_unresolved_members(self):
        pairs, objects, decisions = self._builder_inputs()
        pairs.loc[0, "selected_rank_a"] = 1
        pairs.loc[0, "selected_rank_b"] = 0
        with self.assertRaisesRegex(ValueError, "canonical L/S"):
            build_conditional_matched_member_events(pairs, objects, decisions)

        pairs, objects, decisions = self._builder_inputs()
        objects = objects.loc[~objects["tob_index"].eq(2)].copy()
        with self.assertRaisesRegex(ValueError, "does not resolve"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_builder_requires_integer_pair_indices_and_selected_ranks(self):
        pairs, objects, decisions = self._builder_inputs()
        pairs["pair_index"] = pairs["pair_index"].astype(float)
        pairs.loc[0, "pair_index"] = 0.5
        with self.assertRaisesRegex(ValueError, "pair_index"):
            build_conditional_matched_member_events(pairs, objects, decisions)

        pairs, objects, decisions = self._builder_inputs()
        pairs["selected_rank_a"] = pairs["selected_rank_a"].astype(float)
        pairs.loc[0, "selected_rank_a"] = 0.5
        with self.assertRaisesRegex(ValueError, "canonical L/S"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_builder_rejects_nonfinite_truth_and_mixed_splits(self):
        pairs, objects, decisions = self._builder_inputs()
        objects.loc[
            objects["global_event_id"].eq(10) & objects["tob_index"].eq(1),
            "truth_pt",
        ] = np.nan
        with self.assertRaisesRegex(ValueError, "two finite associated truth_pt"):
            build_conditional_matched_member_events(pairs, objects, decisions)

        pairs, objects, decisions = self._builder_inputs()
        decisions.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation rows only"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_builder_requires_each_group_to_cover_conditional_events(self):
        pairs, objects, decisions = self._builder_inputs()
        decisions = decisions.loc[
            ~(
                decisions["model_seed"].eq(123)
                & decisions["global_event_id"].eq(10)
            )
        ]
        with self.assertRaisesRegex(ValueError, "missing conditional event"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_builder_is_invariant_to_input_row_order(self):
        pairs, objects, decisions = self._builder_inputs()
        expected = build_conditional_matched_member_events(
            pairs, objects, decisions
        )
        actual = build_conditional_matched_member_events(
            pairs.sample(frac=1.0, random_state=1).reset_index(drop=True),
            objects.sample(frac=1.0, random_state=2).reset_index(drop=True),
            decisions.sample(frac=1.0, random_state=3).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_builder_rejects_seed_dependent_baseline_decisions(self):
        pairs, objects, decisions = self._builder_inputs()
        decisions.loc[
            decisions["model_seed"].eq(123)
            & decisions["global_event_id"].eq(10),
            "baseline_pass",
        ] = True
        with self.assertRaisesRegex(ValueError, "identical across model groups"):
            build_conditional_matched_member_events(pairs, objects, decisions)

    def test_paired_count_uncertainty_uses_event_differences(self):
        model = np.array([1, 1, 0, 0, 1], dtype=bool)
        baseline = np.array([1, 0, 1, 0, 0], dtype=bool)
        result = paired_count_summary(model, baseline)

        differences = model.astype(float) - baseline.astype(float)
        expected_se = differences.std(ddof=1) / math.sqrt(len(differences))
        self.assertEqual(result["denominator"], 5)
        self.assertEqual(result["both_accepted"], 1)
        self.assertEqual(result["model_only_accepted"], 2)
        self.assertEqual(result["baseline_only_accepted"], 1)
        self.assertEqual(result["neither_accepted"], 1)
        self.assertAlmostEqual(result["paired_difference"], 0.2)
        self.assertAlmostEqual(result["paired_standard_error"], expected_se)

    def test_identical_decisions_have_zero_difference_and_uncertainty(self):
        result = paired_count_summary([True, False], [True, False])
        self.assertEqual(result["paired_difference"], 0.0)
        self.assertEqual(result["paired_standard_error"], 0.0)
        self.assertEqual(result["point_status"], "ties")
        self.assertEqual(result["evidence_status"], "uncertain")

    def test_empty_summary_is_explicit(self):
        result = paired_count_summary([], [])
        self.assertEqual(result["denominator"], 0)
        self.assertIsNone(result["paired_difference"])
        self.assertEqual(result["evidence_status"], "empty")

    def test_conditional_profile_uses_lower_associated_member_value(self):
        events = pd.DataFrame(
            {
                "global_event_id": [1, 2, 3, 4],
                "model_seed": [42, 42, 42, 42],
                "member_l_associated_truth_pt_gev": [30.0, 80.0, 12.0, 8.0],
                "member_s_associated_truth_pt_gev": [20.0, 40.0, 70.0, 100.0],
                "model_pass": [True, True, True, True],
                "baseline_pass": [False, True, False, False],
                "has_operational_label2_pair": [True, True, True, True],
            }
        )
        profile = summarize_conditional_matched_members(events)
        rows = profile.regions.set_index("region")

        self.assertEqual(rows.loc["low_10_25", "denominator"], 2)
        self.assertEqual(rows.loc["medium_25_60", "denominator"], 1)
        self.assertEqual(rows.loc["high_60_plus", "denominator"], 0)
        self.assertEqual(profile.population.iloc[0]["conditional_event_count"], 4)
        self.assertEqual(profile.population.iloc[0]["outside_reported_regions"], 1)
        self.assertEqual(profile.terminology, CONDITIONAL_MATCHED_MEMBER_TERMINOLOGY)
        self.assertEqual(profile.limitations, CONDITIONAL_MATCHED_MEMBER_LIMITATIONS)

    def test_profiles_are_separate_for_each_seed(self):
        events = pd.DataFrame(
            {
                "global_event_id": [1, 1],
                "model_seed": [42, 123],
                "member_l_associated_truth_pt_gev": [70.0, 70.0],
                "member_s_associated_truth_pt_gev": [65.0, 65.0],
                "model_pass": [True, False],
                "baseline_pass": [False, False],
                "has_operational_label2_pair": [True, True],
            }
        )
        profile = summarize_conditional_matched_members(events)
        high = profile.regions[profile.regions["region"] == "high_60_plus"]
        self.assertEqual(set(high["model_seed"]), {42, 123})
        self.assertEqual(dict(zip(high.model_seed, high.model_accepted)), {42: 1, 123: 0})

    def test_energy_boundaries_are_half_open(self):
        events = pd.DataFrame(
            {
                "global_event_id": [1, 2, 3],
                "model_seed": [42, 42, 42],
                "member_l_associated_truth_pt_gev": [20.0, 30.0, 70.0],
                "member_s_associated_truth_pt_gev": [10.0, 25.0, 60.0],
                "model_pass": [True, True, True],
                "baseline_pass": [False, False, False],
                "has_operational_label2_pair": [True, True, True],
            }
        )
        rows = summarize_conditional_matched_members(events).regions.set_index(
            "region"
        )
        self.assertEqual(rows.loc["low_10_25", "denominator"], 1)
        self.assertEqual(rows.loc["medium_25_60", "denominator"], 1)
        self.assertEqual(rows.loc["high_60_plus", "denominator"], 1)

    def test_duplicate_event_within_seed_is_rejected(self):
        events = pd.DataFrame(
            {
                "global_event_id": [1, 1],
                "model_seed": [42, 42],
                "member_l_associated_truth_pt_gev": [20.0, 20.0],
                "member_s_associated_truth_pt_gev": [30.0, 30.0],
                "model_pass": [True, True],
                "baseline_pass": [False, False],
                "has_operational_label2_pair": [True, True],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            summarize_conditional_matched_members(events)

    def test_non_operational_or_nonfinite_rows_are_rejected(self):
        base = {
            "global_event_id": [1],
            "model_seed": [42],
            "member_l_associated_truth_pt_gev": [20.0],
            "member_s_associated_truth_pt_gev": [30.0],
            "model_pass": [True],
            "baseline_pass": [False],
            "has_operational_label2_pair": [False],
        }
        with self.assertRaisesRegex(ValueError, "operational label-2"):
            summarize_conditional_matched_members(pd.DataFrame(base))
        base["has_operational_label2_pair"] = [True]
        base["member_s_associated_truth_pt_gev"] = [np.nan]
        with self.assertRaisesRegex(ValueError, "must be finite"):
            summarize_conditional_matched_members(pd.DataFrame(base))

    def test_terminology_states_scientific_limit(self):
        self.assertIn("conditional event efficiency", CONDITIONAL_MATCHED_MEMBER_TERMINOLOGY)
        combined = " ".join(CONDITIONAL_MATCHED_MEMBER_LIMITATIONS)
        self.assertIn("not an inclusive", combined)
        self.assertIn("distinct generator-level tau", combined)
        self.assertIn("never an inference input", combined)


if __name__ == "__main__":
    unittest.main()
