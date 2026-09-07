import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_features import member_summary_features
from prepare_pair_data import (
    CORE_COLUMNS,
    EM2_COLUMNS,
    _member_frame,
    _member_min_truth_pt_gev,
    _representation_arrays,
    preparation_contract,
    preparation_contract_for_split,
    prepare_pair_split,
)


class PairPreparationTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for tob_index, pt in ((0, 30.0), (1, 20.0)):
            row = {"eventNumber": 7, "tob_index": tob_index, "tob_pt": pt}
            row.update(
                {
                    name: float(index + 100 * tob_index)
                    for index, name in enumerate(CORE_COLUMNS)
                }
            )
            row.update({name: 0.0 for name in EM2_COLUMNS})
            row["em2_cell_0"] = 10.0 + tob_index
            rows.append(row)
        self.frame = pd.DataFrame(rows)
        self.pairs = pd.DataFrame(
            {
                "eventNumber": [7],
                "pair_index": [0],
                "tob_index_a": [0],
                "tob_index_b": [1],
                "tob_pt_a": [30.0],
                "tob_pt_b": [20.0],
                "tob_pt_max": [30.0],
                "tob_pt_min": [20.0],
                "binary_label": [True],
                "three_class_label": [2],
            }
        )
        self.events = pd.DataFrame(
            {"eventNumber": [7], "baseline_event_score": [20.0]}
        )

    def test_raw_member_join_preserves_pair_order(self):
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        arrays = _representation_arrays("raw_cells", self.pairs, left, right, self.events)
        self.assertEqual(arrays["inputs"].shape, (1, 90))
        self.assertEqual(arrays["inputs"][0, 0], 0.0)
        self.assertEqual(arrays["inputs"][0, 45], 100.0)

    def test_high_resolution_layout_uses_47_scalars_and_one_event_context(self):
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        arrays = _representation_arrays(
            "high_resolution_em2",
            self.pairs,
            left,
            right,
            self.events,
        )
        self.assertEqual(arrays["images"].shape, (1, 2, 12, 12))
        self.assertEqual(arrays["scalars"].shape, (1, 2, 47))
        self.assertEqual(arrays["scalars"][0, 0, 45], 30.0)
        self.assertEqual(arrays["scalars"][0, 1, 45], 20.0)
        self.assertEqual(arrays["scalars"][0, 0, 46], 1.0)
        self.assertEqual(arrays["context"].shape, (1, 1))
        self.assertEqual(arrays["context"][0, 0], 20.0)

    def test_high_resolution_factorial_can_remove_member_and_event_pt(self):
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        arrays = _representation_arrays(
            "high_resolution_em2",
            self.pairs,
            left,
            right,
            self.events,
            include_member_pt=False,
            include_event_pt=False,
        )
        self.assertEqual(arrays["scalars"].shape, (1, 2, 46))
        self.assertEqual(arrays["context"].shape, (1, 0))
        self.assertEqual(arrays["scalars"][0, 0, 45], 1.0)
        self.assertEqual(arrays["scalars"][0, 1, 45], 1.0)

    def test_high_resolution_context_can_replace_member_pt_with_pair_encoding(self):
        source = self.frame.loc[:, ["eventNumber", "tob_index", "tob_pt"]]
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        arrays = _representation_arrays(
            "high_resolution_em2_context",
            self.pairs,
            left,
            right,
            self.events,
            source_frame=source,
            context_features=["pair_sum_pt", "pair_balance"],
            include_member_pt=False,
        )
        self.assertEqual(arrays["scalars"].shape, (1, 2, 46))
        self.assertEqual(arrays["context"].shape, (1, 3))
        np.testing.assert_allclose(arrays["context"], [[20.0, 50.0, 0.2]])

    def test_context_representation_contract_binds_model_widths(self):
        config = {
            "representation": "high_resolution_em2_context",
            "context_features": ["pair_sum_pt", "pair_balance"],
            "include_member_pt": False,
            "model": {
                "name": "shared_member_high_resolution_em2",
                "member_scalar_width": 46,
                "context_width": 3,
            },
        }
        self.assertEqual(
            preparation_contract(config),
            (
                "high_resolution_em2_context",
                None,
                ["pair_sum_pt", "pair_balance"],
                False,
                True,
                False,
            ),
        )
        config["model"]["context_width"] = 2
        with self.assertRaisesRegex(ValueError, "context_width"):
            preparation_contract(config)

    def test_prepared_member_features_follow_declared_order(self):
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        names = [
            "em2_raw_dominance",
            "em2_normalized_dominance",
            "em2_best_3x3_fraction",
            "measured_tob_pt",
            "event_second_highest_tob_pt",
        ]
        arrays = _representation_arrays(
            "prepared_member_features",
            self.pairs,
            left,
            right,
            self.events,
            member_features=names,
        )
        self.assertEqual(arrays["inputs"].shape, (1, 2, 5))
        left_core = left[CORE_COLUMNS].to_numpy(np.float32).reshape(-1, 5, 3, 3)
        left_em2 = left[EM2_COLUMNS].to_numpy(np.float32).reshape(-1, 12, 12)
        summary = member_summary_features(left_core, left_em2)[0]
        expected = [summary[12], summary[12] / summary[7], summary[15], 30.0, 20.0]
        np.testing.assert_allclose(arrays["inputs"][0, 0], expected)
        self.assertEqual(arrays["inputs"][0, 1, 3], 20.0)
        self.assertEqual(arrays["inputs"][0, 1, 4], 20.0)

    def test_prepared_member_normalized_dominance_has_safe_zero_division(self):
        frame = self.frame.copy()
        for name in CORE_COLUMNS[18:27]:
            frame[name] = 0.0
        left = _member_frame(self.pairs, frame, "a")
        right = _member_frame(self.pairs, frame, "b")
        arrays = _representation_arrays(
            "prepared_member_features",
            self.pairs,
            left,
            right,
            self.events,
            member_features=["em2_normalized_dominance"],
        )
        np.testing.assert_array_equal(arrays["inputs"], 0.0)

    def test_prepared_member_feature_contract_rejects_unknown_or_duplicate_names(self):
        left = _member_frame(self.pairs, self.frame, "a")
        right = _member_frame(self.pairs, self.frame, "b")
        for names in (["unknown"], ["measured_tob_pt", "measured_tob_pt"]):
            with self.assertRaises(ValueError):
                _representation_arrays(
                    "prepared_member_features",
                    self.pairs,
                    left,
                    right,
                    self.events,
                    member_features=names,
                )

    def test_config_contract_enables_only_declared_training_truth_weight(self):
        config = {
            "representation": "prepared_member_features",
            "member_features": ["measured_tob_pt", "event_second_highest_tob_pt"],
            "model": {"name": "shared_member_pair_mlp", "member_width": 2},
            "loss": {"name": "bce", "pair_weighting": "equal"},
        }
        self.assertEqual(
            preparation_contract(config),
            (
                "prepared_member_features",
                ["measured_tob_pt", "event_second_highest_tob_pt"],
                None,
                True,
                False,
                False,
            ),
        )
        weighted = {
            "representation": "high_resolution_em2",
            "loss": {
                "name": "bce",
                "pair_weighting": "inverse_frequency_member_min",
                "weight_coordinate": "member_min_truth_pt_gev",
            },
        }
        self.assertEqual(
            preparation_contract(weighted),
            ("high_resolution_em2", None, None, True, True, True),
        )
        self.assertEqual(
            preparation_contract_for_split(weighted, "train"),
            ("high_resolution_em2", None, None, True, True, True),
        )
        self.assertEqual(
            preparation_contract_for_split(weighted, "validation"),
            ("high_resolution_em2", None, None, True, True, False),
        )
        weighted["loss"]["weight_coordinate"] = "unverified"
        with self.assertRaisesRegex(ValueError, "member_min_truth_pt_gev"):
            preparation_contract(weighted)

    def test_truth_analysis_column_is_joined_only_when_requested(self):
        frame = self.frame.assign(truth_pt=[31_000.0, 19_000.0])
        ordinary = _member_frame(self.pairs, frame, "a")
        self.assertNotIn("truth_pt", ordinary)
        analysis = _member_frame(
            self.pairs,
            frame,
            "a",
            extra_columns=("truth_pt",),
        )
        self.assertEqual(analysis.loc[0, "truth_pt"], 31_000.0)
        right = _member_frame(
            self.pairs,
            frame,
            "b",
            extra_columns=("truth_pt",),
        )
        np.testing.assert_array_equal(
            _member_min_truth_pt_gev(analysis, right, np.array([1.0])),
            [19.0],
        )
        right.loc[0, "truth_pt"] = np.nan
        with self.assertRaisesRegex(ValueError, "positive training pair"):
            _member_min_truth_pt_gev(analysis, right, np.array([1.0]))

    def test_truth_weight_coordinate_is_rejected_for_holdout_before_data_access(self):
        with self.assertRaisesRegex(ValueError, "training-only analysis field"):
            prepare_pair_split(
                "/path/that/must/not/be-read",
                split="validation",
                representation="high_resolution_em2",
                include_member_min_truth_pt_gev=True,
            )


if __name__ == "__main__":
    unittest.main()
