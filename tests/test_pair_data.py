import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pair_data import (
    NON_PASSING_SCORE,
    aggregate_pair_scores,
    build_pair_dataset,
)


class PairConstructionTests(unittest.TestCase):
    def test_interleaved_events_are_ranked_and_paired_separately(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [2, 1, 2, 1, 2, 1, 1, 1],
                "tob_index": [0, 4, 1, 1, 2, 0, 3, 2],
                "tob_pt": [22.0, 5.0, 30.0, 40.0, 12.0, 50.0, 10.0, 20.0],
                "signal": [0, 1, 0, 1, 0, 0, 0, 1],
            }
        )

        result = build_pair_dataset(frame)

        event_one = result.selected_tobs.loc[
            result.selected_tobs["eventNumber"].eq(1)
        ]
        self.assertEqual(event_one["tob_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(len(result.pairs.loc[result.pairs["eventNumber"].eq(1)]), 6)
        for row in result.pairs.itertuples():
            source = result.selected_tobs.loc[
                result.selected_tobs["eventNumber"].eq(row.eventNumber),
                "tob_index",
            ]
            self.assertIn(row.tob_index_a, source.to_list())
            self.assertIn(row.tob_index_b, source.to_list())

    def test_more_than_four_tobs_retains_only_the_highest_four(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1] * 6,
                "tob_index": [0, 1, 2, 3, 4, 5],
                "tob_pt": [10.0, 60.0, 20.0, 50.0, 30.0, 40.0],
                "signal": [0, 1, 0, 1, 0, 1],
            }
        )

        result = build_pair_dataset(frame)

        self.assertEqual(result.selected_tobs["tob_index"].tolist(), [1, 3, 5, 4])
        self.assertEqual(result.events.loc[0, "selected_tob_count"], 4)
        self.assertEqual(result.events.loc[0, "pair_count"], 6)

    def test_exact_ties_use_increasing_tob_index(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1, 1],
                "tob_index": [3, 0, 2, 1],
                "tob_pt": [10.0, 20.0, 10.0, 20.0],
                "signal": [0, 1, 1, 0],
            }
        )

        result = build_pair_dataset(frame)

        self.assertEqual(result.selected_tobs["tob_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            result.pairs[["selected_rank_a", "selected_rank_b"]].values.tolist(),
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        )

    def test_nonfinite_tobs_are_excluded_and_short_events_are_preserved(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 2, 2, 3],
                "tob_index": [0, 1, 0, 1, 0],
                "tob_pt": [20.0, np.nan, np.inf, -np.inf, 9.0],
                "signal": [1, 0, 0, 0, 1],
            }
        )

        result = build_pair_dataset(frame)

        self.assertEqual(len(result.pairs), 0)
        self.assertEqual(result.events["pair_count"].tolist(), [0, 0, 0])
        self.assertTrue(
            np.isneginf(result.events["baseline_event_score"].to_numpy()).all()
        )

    def test_source_event_without_object_rows_is_preserved(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1],
                "tob_index": [0, 1],
                "tob_pt": [20.0, 10.0],
                "signal": [1, 1],
            }
        )
        result = build_pair_dataset(frame, event_ids=[1, 2])
        empty = result.events.loc[result.events["eventNumber"].eq(2)].iloc[0]
        self.assertEqual(empty["valid_tob_count"], 0)
        self.assertEqual(empty["pair_count"], 0)
        self.assertEqual(empty["baseline_event_score"], NON_PASSING_SCORE)

    def test_pair_labels_are_zero_one_or_two(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1, 1],
                "tob_index": [0, 1, 2, 3],
                "tob_pt": [40.0, 30.0, 20.0, 10.0],
                "signal": [1, 1, 0, 0],
            }
        )

        result = build_pair_dataset(frame)

        self.assertEqual(result.pairs["three_class_label"].tolist(), [2, 1, 1, 1, 1, 0])
        self.assertEqual(
            result.pairs["binary_label"].tolist(),
            [True, False, False, False, False, False],
        )

    def test_pair_baseline_equals_second_highest_pt_for_every_event(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 2, 1, 2, 1, 2, 1],
                "tob_index": [0, 0, 1, 1, 2, 2, 3],
                "tob_pt": [50.0, 8.0, 40.0, 7.0, 30.0, 6.0, 20.0],
                "signal": [1, 0, 1, 0, 0, 0, 0],
            }
        )

        result = build_pair_dataset(frame)
        pair_max = result.pairs.groupby("eventNumber")["baseline_pair_score"].max()

        np.testing.assert_array_equal(
            pair_max.reindex(result.events["eventNumber"]).to_numpy(),
            result.events["baseline_event_score"].to_numpy(),
        )
        self.assertEqual(result.events["baseline_event_score"].tolist(), [40.0, 7.0])

    def test_shuffling_rows_does_not_change_the_pair_table(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1, 1, 1, 2, 2],
                "tob_index": [0, 1, 2, 3, 0, 1],
                "tob_pt": [40.0, 30.0, 20.0, 10.0, 8.0, 7.0],
                "signal": [1, 0, 1, 0, 0, 0],
            }
        )

        expected = build_pair_dataset(frame)
        shuffled = build_pair_dataset(frame.sample(frac=1.0, random_state=9))

        pd.testing.assert_frame_equal(expected.selected_tobs, shuffled.selected_tobs)
        pd.testing.assert_frame_equal(expected.pairs, shuffled.pairs)
        pd.testing.assert_frame_equal(expected.events, shuffled.events)

    def test_large_integer_pt_that_float64_cannot_rank_exactly_is_rejected(self):
        frame = pd.DataFrame(
            {
                "eventNumber": [1, 1],
                "tob_index": [0, 1],
                "tob_pt": [2**53, 2**53 + 1],
                "signal": [0, 1],
            }
        )

        with self.assertRaisesRegex(ValueError, "exact float64 range"):
            build_pair_dataset(frame)

    def test_duplicate_keys_and_nonbinary_labels_are_rejected(self):
        duplicate = pd.DataFrame(
            {
                "eventNumber": [1, 1],
                "tob_index": [0, 0],
                "tob_pt": [10.0, 9.0],
                "signal": [0, 1],
            }
        )
        bad_label = duplicate.assign(tob_index=[0, 1], signal=[0, 2])

        with self.assertRaisesRegex(ValueError, "unique within each event"):
            build_pair_dataset(duplicate)
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            build_pair_dataset(bad_label)


class PairAggregationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = build_pair_dataset(
            pd.DataFrame(
                {
                    "eventNumber": [1, 1, 1, 2],
                    "tob_index": [0, 1, 2, 0],
                    "tob_pt": [30.0, 20.0, 10.0, 5.0],
                    "signal": [1, 1, 0, 0],
                }
            )
        )

    def predictions(self):
        return pd.DataFrame(
            {
                "eventNumber": [1, 1, 1],
                "pair_index": [2, 0, 1],
                "pair_score": [-0.4, 0.8, 0.2],
            }
        )

    def test_maximum_pair_score_becomes_the_event_score(self):
        result = aggregate_pair_scores(self.dataset, self.predictions())

        self.assertEqual(result.loc[0, "event_score"], 0.8)
        self.assertEqual(result.loc[1, "event_score"], NON_PASSING_SCORE)

    def test_missing_duplicate_foreign_and_nonfinite_predictions_are_rejected(self):
        predictions = self.predictions()
        cases = [
            predictions.iloc[:-1],
            pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True),
            predictions.assign(eventNumber=[1, 1, 99]),
            predictions.assign(pair_score=[0.1, np.nan, 0.2]),
        ]

        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                aggregate_pair_scores(self.dataset, invalid)

    def test_empty_predictions_are_valid_when_no_pairs_exist(self):
        dataset = build_pair_dataset(
            pd.DataFrame(
                {
                    "eventNumber": [1],
                    "tob_index": [0],
                    "tob_pt": [10.0],
                    "signal": [1],
                }
            )
        )
        predictions = pd.DataFrame(
            columns=["eventNumber", "pair_index", "pair_score"]
        )

        result = aggregate_pair_scores(dataset, predictions)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "event_score"], NON_PASSING_SCORE)


if __name__ == "__main__":
    unittest.main()
