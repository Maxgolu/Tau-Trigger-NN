import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from event_data import EventTensorDataset, collate_events, split_training_events


def _frame(event_count=24):
    rows = []
    for event in range(event_count):
        background = event % 3 == 0
        for object_index in range(2 + event % 2):
            signal = int(not background and object_index == 0)
            rows.append(
                {
                    "eventNumber": event,
                    "tob_index": object_index,
                    "Type": "BKG" if background else "Signal",
                    "signal": signal,
                    "truth_pt": (30_000 + event * 1_000) if signal else 0,
                    "tob_pt": 20_000 + object_index * 5_000,
                }
            )
    return pd.DataFrame(rows)


class EventDataTests(unittest.TestCase):
    def test_inner_split_is_deterministic_and_has_no_event_leakage(self):
        frame = _frame()
        first = split_training_events(frame, seed=123, constraint_fraction=0.3)
        second = split_training_events(frame, seed=123, constraint_fraction=0.3)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        primal_events = set(frame.iloc[first[0]]["eventNumber"])
        constraint_events = set(frame.iloc[first[1]]["eventNumber"])
        self.assertFalse(primal_events.intersection(constraint_events))
        self.assertEqual(primal_events.union(constraint_events), set(frame["eventNumber"]))

    def test_event_dataset_and_collate_preserve_complete_events(self):
        frame = _frame(event_count=4)
        features = np.arange(len(frame) * 2, dtype=np.float32).reshape(-1, 2)
        labels = frame["signal"].to_numpy(dtype=np.float32)
        dataset = EventTensorDataset(features, labels, frame)
        batch = collate_events([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch.features.shape[:2]), (2, 3))
        self.assertEqual(int(batch.object_mask[0].sum()), 2)
        self.assertEqual(int(batch.object_mask[1].sum()), 3)
        self.assertFalse(bool(batch.object_mask[0, 2]))
        self.assertTrue(bool(batch.background_event_mask[0]))
        self.assertFalse(bool(batch.background_event_mask[1]))

    def test_padded_hard_trigger_matches_dataframe_event_counts(self):
        frame = _frame(event_count=8)
        scores = np.linspace(0.1, 0.9, len(frame), dtype=np.float32)
        dataset = EventTensorDataset(
            scores.reshape(-1, 1),
            frame["signal"].to_numpy(dtype=np.float32),
            frame,
        )
        batch = collate_events([dataset[index] for index in range(len(dataset))])
        padded_decision = (
            ((batch.features[..., 0] >= 0.5) & batch.object_mask).sum(dim=1) >= 2
        ).numpy()
        dataframe_decision = (
            pd.Series(scores >= 0.5)
            .groupby(frame["eventNumber"], sort=False)
            .sum()
            .ge(2)
            .to_numpy()
        )
        self.assertTrue(np.array_equal(padded_decision, dataframe_decision))


if __name__ == "__main__":
    unittest.main()
