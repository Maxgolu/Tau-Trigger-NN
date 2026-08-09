import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import training_data
from training_data import DataKey, PreparedDataset, TrainingDataCache


def make_dataset(key=None):
    event_numbers = np.repeat(np.arange(20), 3)
    frame = pd.DataFrame(
        {
            "eventNumber": event_numbers,
            "tob_index": np.tile(np.arange(3), 20),
            "tob_pt": np.tile([30.0, 20.0, 10.0], 20),
            "tob_eta": np.tile([0.2, -0.1, 0.4], 20),
            "tob_phi": np.tile([3.1, -3.1, 0.5], 20),
            "label": np.tile([1.0, 0.0, 0.0], 20),
        }
    )
    if key is None:
        key = DataKey("synthetic", None, None)

    labels = frame["label"].to_numpy(copy=True)
    events = event_numbers.copy()
    labels.flags.writeable = False
    events.flags.writeable = False
    return PreparedDataset(key, frame, events, labels)


class TrainingDataCacheTests(unittest.TestCase):
    def test_split_matches_legacy_algorithm_and_has_no_event_leakage(self):
        dataset = make_dataset()
        split = TrainingDataCache().get_split(dataset, seed=42)

        expected_events = np.unique(dataset.event_numbers).copy()
        np.random.seed(42)
        np.random.shuffle(expected_events)
        expected_train = expected_events[:14]

        np.testing.assert_array_equal(
            split.train,
            np.flatnonzero(np.isin(dataset.event_numbers, expected_train)),
        )
        train_events = set(dataset.event_numbers[split.train])
        validation_events = set(dataset.event_numbers[split.validation])
        test_events = set(dataset.event_numbers[split.test])
        self.assertTrue(train_events.isdisjoint(validation_events))
        self.assertTrue(train_events.isdisjoint(test_events))
        self.assertTrue(validation_events.isdisjoint(test_events))

    def test_split_does_not_advance_global_numpy_randomness(self):
        dataset = make_dataset()
        np.random.seed(1234)
        expected = np.random.random(5)

        np.random.seed(1234)
        TrainingDataCache().get_split(dataset, seed=42)
        actual = np.random.random(5)

        np.testing.assert_array_equal(actual, expected)

    def test_cached_and_original_feature_paths_are_equal(self):
        dataset = make_dataset()
        split = TrainingDataCache().get_split(dataset, seed=123)
        names = ["tob_pt_only", "event_context_core"]

        cached = TrainingDataCache(enabled=True)
        uncached = TrainingDataCache(enabled=False)
        cached_values = cached.assemble_features(dataset, split.train, names)
        uncached_values = uncached.assemble_features(dataset, split.train, names)

        np.testing.assert_array_equal(cached_values, uncached_values)

    def test_feature_and_split_cache_hits_are_reused(self):
        dataset = make_dataset()
        cache = TrainingDataCache(enabled=True)
        first_split = cache.get_split(dataset, seed=42)
        second_split = cache.get_split(dataset, seed=42)
        self.assertIs(first_split, second_split)

        first_feature = cache.get_feature(dataset, "tob_pt_only")
        second_feature = cache.get_feature(dataset, "tob_pt_only")
        self.assertIs(first_feature, second_feature)
        self.assertEqual(cache.stats["split_hits"], 1)
        self.assertEqual(cache.stats["feature_hits"], 1)
        with self.assertRaises(ValueError):
            first_feature[0, 0] = -1.0

    def test_full_data_reuses_alignment_but_sampled_data_uses_seed(self):
        calls = []

        def fake_loader(data_dir, max_events_per_class, sampling_seed):
            calls.append((max_events_per_class, sampling_seed))
            key = training_data._make_data_key(
                data_dir,
                max_events_per_class,
                sampling_seed,
            )
            return make_dataset(key)

        cache = TrainingDataCache(enabled=True)
        with patch.object(training_data, "load_and_align_dataset", fake_loader):
            full_42 = cache.get_dataset("synthetic", None, 42)
            full_123 = cache.get_dataset("synthetic", None, 123)
            smoke_42 = cache.get_dataset("synthetic", 5, 42)
            smoke_123 = cache.get_dataset("synthetic", 5, 123)

        self.assertIs(full_42, full_123)
        self.assertIsNot(smoke_42, smoke_123)
        self.assertEqual(len(calls), 3)

    def test_prediction_copy_does_not_modify_cached_frame(self):
        dataset = make_dataset()
        split = TrainingDataCache().get_split(dataset, seed=42)
        output = dataset.frame.iloc[split.test].copy().reset_index(drop=True)
        output["nn_score"] = 0.75

        self.assertNotIn("nn_score", dataset.frame.columns)


if __name__ == "__main__":
    unittest.main()
