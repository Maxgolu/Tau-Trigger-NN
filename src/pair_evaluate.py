"""Cross-fitted validation and independent pair-model calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .operating_point import select_fpr_threshold
    from .pair_classifiers import pass_scores
except ImportError:
    from operating_point import select_fpr_threshold
    from pair_classifiers import pass_scores


@dataclass(frozen=True)
class CrossFitResult:
    """Pooled held-out counts for one model checkpoint."""

    signal_accepted: int
    signal_events: int
    background_accepted: int
    background_events: int
    legs: tuple[dict, dict]

    @property
    def signal_efficiency(self):
        return self.signal_accepted / self.signal_events

    @property
    def background_fpr(self):
        return self.background_accepted / self.background_events


def assign_validation_folds(
    event_ids,
    signal_sample,
    pair_observable,
    pair_eligible,
    *,
    seed=42,
):
    """Assign deterministic balanced A/B folds within five event strata."""
    event_ids = np.asarray(event_ids, dtype=np.int64)
    signal_sample = np.asarray(signal_sample, dtype=bool)
    pair_observable = np.asarray(pair_observable, dtype=bool)
    pair_eligible = np.asarray(pair_eligible, dtype=bool)
    if not (
        event_ids.shape
        == signal_sample.shape
        == pair_observable.shape
        == pair_eligible.shape
    ):
        raise ValueError("validation event arrays must be aligned")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("validation event IDs must be unique")

    strata = np.select(
        (
            ~signal_sample & pair_observable,
            ~signal_sample & ~pair_observable,
            signal_sample & pair_eligible,
            signal_sample & ~pair_eligible & pair_observable,
            signal_sample & ~pair_observable,
        ),
        (
            "background_paired",
            "background_no_pair",
            "signal_eligible",
            "signal_ineligible_paired",
            "signal_no_pair",
        ),
        default="",
    )
    folds = np.empty(len(event_ids), dtype="<U1")
    version = "pair-validation-sha256-alternating-v1"
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        ordered = sorted(
            positions,
            key=lambda position: (
                hashlib.sha256(
                    f"{version}|{seed}|{stratum}|{int(event_ids[position])}".encode()
                ).hexdigest(),
                int(event_ids[position]),
            ),
        )
        for offset, position in enumerate(ordered):
            folds[position] = "A" if offset % 2 == 0 else "B"
    return folds


def calibrate_event_threshold(scores, target_fpr=0.005):
    """Calibrate one threshold using background-event scores."""
    values = np.asarray(scores, dtype=np.float64)
    threshold, achieved = select_fpr_threshold(values, len(values), target_fpr)
    return float(threshold), float(achieved)


def _validate_events(events, score_column):
    required = {"global_event_id", "sample", "fold", "pair_eligible", score_column}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"missing event columns: {sorted(missing)}")
    if events["global_event_id"].duplicated().any():
        raise ValueError("event table must contain one row per event")
    if not set(events["fold"]).issubset({"A", "B"}):
        raise ValueError("validation fold must be A or B")


def crossfit_checkpoint(events, score_column="event_score", target_fpr=0.005):
    """Calibrate on each validation fold and evaluate on the other fold."""
    _validate_events(events, score_column)
    legs = []
    signal_accepted = signal_events = 0
    background_accepted = background_events = 0

    for calibration_fold, evaluation_fold in (("A", "B"), ("B", "A")):
        calibration_background = events.loc[
            (events["fold"] == calibration_fold)
            & (events["sample"] == "background"),
            score_column,
        ].to_numpy(dtype=np.float64)
        threshold, calibration_fpr = calibrate_event_threshold(
            calibration_background,
            target_fpr,
        )
        heldout = events.loc[events["fold"] == evaluation_fold]
        decisions = pass_scores(heldout[score_column], threshold)
        background_mask = heldout["sample"].eq("background").to_numpy()
        signal_mask = (
            heldout["sample"].eq("signal")
            & heldout["pair_eligible"].astype(bool)
        ).to_numpy()
        leg_background_accepted = int((decisions & background_mask).sum())
        leg_background_events = int(background_mask.sum())
        leg_signal_accepted = int((decisions & signal_mask).sum())
        leg_signal_events = int(signal_mask.sum())
        signal_accepted += leg_signal_accepted
        signal_events += leg_signal_events
        background_accepted += leg_background_accepted
        background_events += leg_background_events
        legs.append(
            {
                "calibration_fold": calibration_fold,
                "evaluation_fold": evaluation_fold,
                "threshold": threshold,
                "calibration_fpr": calibration_fpr,
                "heldout_signal_accepted": leg_signal_accepted,
                "heldout_signal_events": leg_signal_events,
                "heldout_background_accepted": leg_background_accepted,
                "heldout_background_events": leg_background_events,
            }
        )

    if signal_events == 0 or background_events == 0:
        raise ValueError("cross-fitting requires eligible signal and background events")
    return CrossFitResult(
        signal_accepted=signal_accepted,
        signal_events=signal_events,
        background_accepted=background_accepted,
        background_events=background_events,
        legs=tuple(legs),
    )


def select_checkpoint(records):
    """Select one checkpoint using the frozen operating-point tie rules."""
    if not records:
        raise ValueError("checkpoint records cannot be empty")
    best = records[0]
    for candidate in records[1:]:
        candidate_primary = (
            int(candidate["signal_accepted"]),
            -int(candidate["background_accepted"]),
        )
        best_primary = (
            int(best["signal_accepted"]),
            -int(best["background_accepted"]),
        )
        if candidate_primary > best_primary:
            best = candidate
            continue
        if candidate_primary < best_primary:
            continue

        bce_difference = float(candidate["validation_bce"]) - float(
            best["validation_bce"]
        )
        if bce_difference < -1e-12 or abs(bce_difference) <= 1e-12 and int(candidate["epoch"]) < int(
            best["epoch"]
        ):
            best = candidate
    return best


def evaluate_prediction_table(predictions, target_fpr=0.005):
    """Cross-fit every seed/checkpoint in an event-prediction table."""
    required = {"model_seed", "checkpoint_id", "validation_bce"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"missing prediction columns: {sorted(missing)}")
    rows = []
    for (seed, checkpoint), group in predictions.groupby(
        ["model_seed", "checkpoint_id"], sort=True
    ):
        bce_values = group["validation_bce"].to_numpy(dtype=np.float64)
        if not np.isfinite(bce_values).all() or not np.all(bce_values == bce_values[0]):
            raise ValueError("validation BCE must be one finite value per checkpoint")
        result = crossfit_checkpoint(group, target_fpr=target_fpr)
        epoch = int(str(checkpoint).split("_")[-1])
        rows.append(
            {
                "model_seed": int(seed),
                "checkpoint_id": checkpoint,
                "epoch": epoch,
                "signal_accepted": result.signal_accepted,
                "signal_events": result.signal_events,
                "signal_efficiency": result.signal_efficiency,
                "background_accepted": result.background_accepted,
                "background_events": result.background_events,
                "background_fpr": result.background_fpr,
                "validation_bce": float(group["validation_bce"].iloc[0]),
                "crossfit_legs": json.dumps(result.legs),
            }
        )
    return pd.DataFrame(rows)


def _full_validation_result(events, score_column, target_fpr):
    background = events["sample"].eq("background").to_numpy()
    signal = events["sample"].eq("signal").to_numpy()
    eligible = signal & events["pair_eligible"].astype(bool).to_numpy()
    scores = events[score_column].to_numpy(dtype=np.float64)
    threshold, achieved_fpr = calibrate_event_threshold(
        scores[background],
        target_fpr,
    )
    decisions = pass_scores(scores, threshold)

    def counts(mask):
        denominator = int(mask.sum())
        accepted = int((decisions & mask).sum())
        return {
            "accepted": accepted,
            "events": denominator,
            "efficiency": accepted / denominator if denominator else None,
        }

    return {
        "threshold": threshold,
        "background_fpr": achieved_fpr,
        "background": counts(background),
        "inclusive_signal": counts(signal),
        "pair_eligible_signal": counts(eligible),
    }


def build_selection_report(predictions, checkpoint_summary, target_fpr=0.005):
    """Select one checkpoint per seed and perform full-validation calibration."""
    required = {
        "model_seed",
        "checkpoint_id",
        "baseline_event_score",
        "global_event_id",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"missing prediction columns: {sorted(missing)}")
    if checkpoint_summary.empty:
        raise ValueError("checkpoint summary cannot be empty")

    grouped = list(
        predictions.groupby(["model_seed", "checkpoint_id"], sort=True)
    )
    reference = grouped[0][1].sort_values("global_event_id")
    baseline_columns = [
        "global_event_id",
        "sample",
        "fold",
        "pair_eligible",
        "baseline_event_score",
    ]
    baseline = reference.loc[:, baseline_columns].rename(
        columns={"baseline_event_score": "event_score"}
    )
    for _, group in grouped[1:]:
        candidate = group.sort_values("global_event_id")
        for column in (
            "global_event_id",
            "sample",
            "fold",
            "pair_eligible",
            "baseline_event_score",
        ):
            if not np.array_equal(
                reference[column].to_numpy(),
                candidate[column].to_numpy(),
            ):
                raise ValueError(
                    f"baseline event metadata differ across checkpoints: {column}"
                )

    baseline_crossfit = crossfit_checkpoint(
        baseline,
        target_fpr=target_fpr,
    )
    report = {
        "target_event_fpr": target_fpr,
        "baseline": {
            "crossfit": {
                "signal_accepted": baseline_crossfit.signal_accepted,
                "signal_events": baseline_crossfit.signal_events,
                "background_accepted": baseline_crossfit.background_accepted,
                "background_events": baseline_crossfit.background_events,
                "legs": baseline_crossfit.legs,
            },
            "full_validation": _full_validation_result(
                baseline,
                "event_score",
                target_fpr,
            ),
        },
        "seeds": {},
    }
    for seed, seed_summary in checkpoint_summary.groupby("model_seed", sort=True):
        selected = select_checkpoint(seed_summary.to_dict(orient="records"))
        selected_events = predictions.loc[
            predictions["model_seed"].eq(seed)
            & predictions["checkpoint_id"].eq(selected["checkpoint_id"])
        ].copy()
        report["seeds"][str(int(seed))] = {
            "checkpoint_id": selected["checkpoint_id"],
            "epoch": int(selected["epoch"]),
            "crossfit": {
                "signal_accepted": int(selected["signal_accepted"]),
                "signal_events": int(selected["signal_events"]),
                "background_accepted": int(selected["background_accepted"]),
                "background_events": int(selected["background_events"]),
                "validation_bce": float(selected["validation_bce"]),
            },
            "full_validation": _full_validation_result(
                selected_events,
                "event_score",
                target_fpr,
            ),
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pair-model validation predictions")
    parser.add_argument("--predictions", required=True, help="Event prediction CSV")
    parser.add_argument("--output", required=True, help="Checkpoint summary CSV")
    parser.add_argument("--target_fpr", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    summary = evaluate_prediction_table(predictions, args.target_fpr)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    report = build_selection_report(predictions, summary, args.target_fpr)
    report_path = output.with_suffix(".selection.json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
