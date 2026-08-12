"""Shared event-level operating-point calculations for training and evaluation."""

import numpy as np
import pandas as pd


def build_event_trigger_scores(df, criterion, objects=2):
    """Return the score at which every background event starts to pass."""
    if objects < 1:
        raise ValueError("objects must be at least 1")
    required = {"eventNumber", criterion}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    event_count = int(df["eventNumber"].nunique())
    if event_count == 0:
        raise ValueError("Cannot calibrate a threshold without background events")

    finite_rows = df.loc[
        np.isfinite(df[criterion].to_numpy(dtype=float)),
        ["eventNumber", criterion],
    ]
    ordered = finite_rows.sort_values(
        ["eventNumber", criterion],
        ascending=[True, False],
        kind="mergesort",
    )
    kth_scores = (
        ordered.groupby("eventNumber", sort=False)[criterion]
        .nth(objects - 1)
        .to_numpy(dtype=float)
    )
    return kth_scores, event_count


def select_fpr_threshold(event_trigger_scores, event_count, target_fake_rate):
    """Select the lowest threshold whose empirical event FPR is within target."""
    if not 0.0 <= target_fake_rate <= 1.0:
        raise ValueError("target_fake_rate must be between 0 and 1")
    if event_count <= 0:
        raise ValueError("event_count must be positive")

    scores = np.asarray(event_trigger_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.inf, 0.0

    max_accepted = int(np.floor(target_fake_rate * event_count + 1e-12))
    unique_scores, tied_counts = np.unique(scores, return_counts=True)
    unique_scores = unique_scores[::-1]
    tied_counts = tied_counts[::-1]
    cumulative = np.cumsum(tied_counts)
    feasible = np.flatnonzero(cumulative <= max_accepted)

    if feasible.size == 0:
        threshold = np.nextafter(unique_scores[0], np.inf)
    else:
        threshold = unique_scores[int(feasible[-1])]

    achieved_fpr = float(np.count_nonzero(scores >= threshold) / event_count)
    return float(threshold), achieved_fpr


def CalcThresh(df, criterion, fake_rate, objects=2):
    """Backward-compatible wrapper for exact event-level FPR calibration."""
    event_scores, event_count = build_event_trigger_scores(
        df, criterion, objects
    )
    return select_fpr_threshold(event_scores, event_count, fake_rate)


def select_truth_tau_objects(df):
    """Return only objects that are truth-matched to a tau."""
    required = {"Type", "signal"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df.loc[(df["Type"] == "Signal") & (df["signal"] == 1)]


def select_background_objects(df):
    """Return background-sample objects for either supported type label."""
    if "Type" not in df.columns:
        raise KeyError("Missing required column: Type")
    return df.loc[df["Type"].isin(["BKG", "Background"])]


def score_pass_mask(df, criterion, threshold):
    """Apply a score cut with the same float64 >= rule as calibration."""
    if criterion not in df.columns:
        raise KeyError(f"Missing required column: {criterion}")
    scores = df[criterion].to_numpy(dtype=np.float64)
    return np.isfinite(scores) & (scores >= np.float64(threshold))

