"""Deterministic construction and event aggregation for TOB pairs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

MAX_SELECTED_TOBS = 4
NON_PASSING_SCORE = -np.inf
MAX_EXACT_FLOAT64_INTEGER = 2**53

SELECTED_TOB_COLUMNS = (
    "eventNumber",
    "tob_index",
    "tob_pt",
    "tob_label",
    "selected_rank",
)
PAIR_COLUMNS = (
    "eventNumber",
    "pair_index",
    "tob_index_a",
    "tob_index_b",
    "selected_rank_a",
    "selected_rank_b",
    "tob_label_a",
    "tob_label_b",
    "three_class_label",
    "binary_label",
    "tob_pt_a",
    "tob_pt_b",
    "tob_pt_max",
    "tob_pt_min",
    "baseline_pair_score",
)
EVENT_COLUMNS = (
    "eventNumber",
    "valid_tob_count",
    "selected_tob_count",
    "pair_count",
    "baseline_event_score",
)
PAIR_PREDICTION_COLUMNS = (
    "eventNumber",
    "pair_index",
    "pair_score",
)


@dataclass(frozen=True)
class PairDataset:
    """Selected TOBs, unordered pairs, and one summary row per event."""

    selected_tobs: pd.DataFrame
    pairs: pd.DataFrame
    events: pd.DataFrame


def _numeric_values(series: pd.Series, name: str) -> np.ndarray:
    """Convert a real numeric column to float64 without accepting text."""
    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise ValueError(f"{name} must use a real numeric dtype")
    if pd.api.types.is_integer_dtype(series.dtype):
        unsafe = series.map(
            lambda value: (
                not pd.isna(value)
                and abs(int(value)) > MAX_EXACT_FLOAT64_INTEGER
            )
        )
        if bool(unsafe.any()):
            raise ValueError(
                f"{name} contains integers outside the exact float64 range"
            )
    return series.to_numpy(dtype=np.float64, na_value=np.nan)


def _validate_input(frame: pd.DataFrame) -> None:
    required = {"eventNumber", "tob_index", "tob_pt", "signal"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if frame["eventNumber"].isna().any():
        raise ValueError("eventNumber must be present for every TOB")
    if frame["tob_index"].isna().any():
        raise ValueError("tob_index must be present for every TOB")
    if frame.duplicated(["eventNumber", "tob_index"]).any():
        raise ValueError("tob_index must be unique within each event")

    labels = _numeric_values(frame["signal"], "signal")
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("signal labels must be binary 0/1")


def build_pair_dataset(frame: pd.DataFrame, event_ids=None) -> PairDataset:
    """Select up to four finite-pT TOBs and construct same-event pairs.

    TOBs are ordered within each event by decreasing measured ``tob_pt``.
    Increasing ``tob_index`` breaks exact ties. All available unordered pairs
    are emitted once in selected-rank order. Events with fewer than two valid
    TOBs remain in the event table and receive a non-passing baseline score.
    """
    _validate_input(frame)
    work = frame.loc[:, ["eventNumber", "tob_index", "tob_pt", "signal"]].copy()
    work["_pt"] = _numeric_values(work["tob_pt"], "tob_pt")
    work["_label"] = work["signal"].to_numpy(dtype=np.uint8)

    observed_events = pd.Index(work["eventNumber"].drop_duplicates())
    if event_ids is None:
        event_ids = observed_events
    else:
        event_ids = pd.Index(event_ids)
        if event_ids.has_duplicates:
            raise ValueError("event_ids must be unique")
        missing_events = observed_events.difference(event_ids)
        if len(missing_events):
            raise ValueError("event_ids must include every event present in the TOB table")
    event_ids = pd.Index(sorted(event_ids), name="eventNumber")
    finite = work.loc[np.isfinite(work["_pt"])].copy()
    finite = finite.sort_values(
        ["eventNumber", "_pt", "tob_index"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    finite["selected_rank"] = finite.groupby(
        "eventNumber", sort=False
    ).cumcount()
    selected = finite.loc[finite["selected_rank"] < MAX_SELECTED_TOBS].copy()
    selected = selected.rename(columns={"_pt": "tob_pt_value", "_label": "tob_label"})
    selected["tob_pt"] = selected["tob_pt_value"]
    selected = selected.loc[:, SELECTED_TOB_COLUMNS].reset_index(drop=True)

    pair_rows = []
    for event_number, event_tobs in selected.groupby("eventNumber", sort=False):
        records = event_tobs.to_dict("records")
        for pair_index, (first, second) in enumerate(combinations(records, 2)):
            label = int(first["tob_label"] + second["tob_label"])
            first_pt = float(first["tob_pt"])
            second_pt = float(second["tob_pt"])
            pair_rows.append(
                {
                    "eventNumber": event_number,
                    "pair_index": pair_index,
                    "tob_index_a": first["tob_index"],
                    "tob_index_b": second["tob_index"],
                    "selected_rank_a": int(first["selected_rank"]),
                    "selected_rank_b": int(second["selected_rank"]),
                    "tob_label_a": int(first["tob_label"]),
                    "tob_label_b": int(second["tob_label"]),
                    "three_class_label": label,
                    "binary_label": label == 2,
                    "tob_pt_a": first_pt,
                    "tob_pt_b": second_pt,
                    "tob_pt_max": max(first_pt, second_pt),
                    "tob_pt_min": min(first_pt, second_pt),
                    "baseline_pair_score": min(first_pt, second_pt),
                }
            )
    pairs = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)

    valid_counts = (
        finite.groupby("eventNumber", sort=False)
        .size()
        .reindex(event_ids, fill_value=0)
        .astype(np.int64)
    )
    selected_counts = (
        selected.groupby("eventNumber", sort=False)
        .size()
        .reindex(event_ids, fill_value=0)
        .astype(np.int64)
    )
    pair_counts = selected_counts * (selected_counts - 1) // 2
    direct_scores = (
        selected.loc[selected["selected_rank"].eq(1)]
        .set_index("eventNumber")["tob_pt"]
        .reindex(event_ids)
        .fillna(NON_PASSING_SCORE)
        .astype(np.float64)
    )
    if pairs.empty:
        pair_scores = pd.Series(
            NON_PASSING_SCORE, index=event_ids, dtype=np.float64
        )
    else:
        pair_scores = (
            pairs.groupby("eventNumber", sort=False)["baseline_pair_score"]
            .max()
            .reindex(event_ids)
            .fillna(NON_PASSING_SCORE)
            .astype(np.float64)
        )
    if not np.array_equal(direct_scores.to_numpy(), pair_scores.to_numpy()):
        raise AssertionError(
            "Maximum pair-minimum pT does not equal second-highest TOB pT"
        )

    events = pd.DataFrame(
        {
            "eventNumber": event_ids,
            "valid_tob_count": valid_counts.to_numpy(),
            "selected_tob_count": selected_counts.to_numpy(),
            "pair_count": pair_counts.to_numpy(),
            "baseline_event_score": direct_scores.to_numpy(),
        },
        columns=EVENT_COLUMNS,
    )
    return PairDataset(selected_tobs=selected, pairs=pairs, events=events)


def aggregate_pair_scores(
    dataset: PairDataset,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validate pair predictions and return the maximum score per event."""
    if tuple(predictions.columns) != PAIR_PREDICTION_COLUMNS:
        raise ValueError(
            "Prediction columns must be exactly "
            f"{list(PAIR_PREDICTION_COLUMNS)}"
        )
    key_columns = ["eventNumber", "pair_index"]
    if predictions.duplicated(key_columns).any():
        raise ValueError("Pair prediction identities must be unique")
    scores = pd.to_numeric(predictions["pair_score"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=np.float64)).all():
        raise ValueError("Pair prediction scores must be finite")

    expected_keys = set(
        dataset.pairs.loc[:, key_columns].itertuples(index=False, name=None)
    )
    prediction_keys = set(
        predictions.loc[:, key_columns].itertuples(index=False, name=None)
    )
    if prediction_keys != expected_keys:
        missing = len(expected_keys.difference(prediction_keys))
        foreign = len(prediction_keys.difference(expected_keys))
        raise ValueError(
            f"Pair predictions do not match the dataset: "
            f"{missing} missing, {foreign} foreign"
        )

    event_ids = pd.Index(dataset.events["eventNumber"], name="eventNumber")
    if predictions.empty:
        event_scores = pd.Series(
            NON_PASSING_SCORE, index=event_ids, dtype=np.float64
        )
    else:
        event_scores = (
            predictions.assign(pair_score=scores)
            .groupby("eventNumber", sort=False)["pair_score"]
            .max()
            .reindex(event_ids)
            .fillna(NON_PASSING_SCORE)
            .astype(np.float64)
        )
    result = pd.DataFrame(
        {
            "eventNumber": event_ids,
            "event_score": event_scores.to_numpy(dtype=np.float64),
        }
    )
    no_pair = dataset.events["pair_count"].to_numpy(dtype=np.int64) == 0
    if not np.isneginf(result.loc[no_pair, "event_score"]).all():
        raise AssertionError("No-pair events must receive a non-passing score")
    return result
