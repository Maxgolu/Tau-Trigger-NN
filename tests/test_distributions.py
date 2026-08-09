import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributions.data import add_split_column, split_event_ids
from src.distributions.plotting import grouped_values
from src.distributions.variables import build_event_table, extract_object_variables


class DistributionDataTests(unittest.TestCase):
    def test_event_split_is_deterministic_and_has_no_event_leakage(self):
        event_ids = np.arange(101)
        first = split_event_ids(event_ids, seed=42)
        second = split_event_ids(event_ids, seed=42)

        for split_name in first:
            np.testing.assert_array_equal(first[split_name], second[split_name])
        self.assertEqual(len(first["train"]), 70)
        self.assertEqual(len(first["validation"]), 10)
        self.assertEqual(len(first["test"]), 21)
        combined = np.concatenate(list(first.values()))
        self.assertEqual(len(combined), len(np.unique(combined)))

    def test_all_objects_from_an_event_receive_the_same_split(self):
        frame = pd.DataFrame(
            {
                "event_uid": np.repeat(np.arange(20), 3),
                "label": 0,
            }
        )
        result = add_split_column(frame, seed=123)
        self.assertTrue(result.groupby("event_uid")["split"].nunique().eq(1).all())


class EventVariableTests(unittest.TestCase):
    def _objects(self):
        return pd.DataFrame(
            {
                "event_uid": [1, 1, 1, 2],
                "original_event_number": [1, 1, 1, 2],
                "sample_type": ["signal", "signal", "signal", "background"],
                "tob_index": [0, 1, 2, 0],
                "label": [1, 1, 0, 0],
                "event_tau_count": [2, 2, 2, 0],
                "split": ["train", "train", "train", "train"],
                "tob_pt": [30_000.0, 20_000.0, 10_000.0, 8_000.0],
                "tob_eta": [0.5, -0.5, 1.0, 0.0],
                "tob_phi": [np.pi - 0.1, -np.pi + 0.1, 0.0, 0.0],
            }
        )

    def test_event_aggregates_and_wrapped_delta_phi(self):
        objects = self._objects()
        objects["tob_pt_gev"] = objects["tob_pt"] / 1000.0
        events = build_event_table(objects).set_index("event_uid")
        self.assertEqual(events.loc[1, "second_highest_tob_pt"], 20.0)
        self.assertEqual(events.loc[1, "sum_event_tob_pt"], 60.0)
        self.assertAlmostEqual(events.loc[1, "top2_tob_dr2"], 1.0 + 0.2**2)
        self.assertEqual(events.loc[1, "tau_group"], "2+ tau")
        self.assertTrue(np.isnan(events.loc[2, "second_highest_tob_pt"]))
        self.assertTrue(np.isnan(events.loc[2, "top2_tob_dr2"]))

    def test_object_grouping_uses_event_tau_multiplicity(self):
        objects = self._objects()
        objects["tob_pt_gev"] = objects["tob_pt"] / 1000.0
        groups = grouped_values(objects, "tob_pt_gev", "object_label_and_event_tau_count")
        self.assertEqual(groups["Tau objects | 2+ tau event"].size, 2)
        self.assertEqual(groups["Noise objects | 2+ tau event"].size, 1)
        self.assertEqual(groups["Noise objects | 0 tau event"].size, 1)

    def test_metadata_only_object_variable_extraction(self):
        objects = self._objects()
        result = extract_object_variables(objects, {"tob_pt"})
        np.testing.assert_array_equal(result["tob_pt"], [30.0, 20.0, 10.0, 8.0])


if __name__ == "__main__":
    unittest.main()
