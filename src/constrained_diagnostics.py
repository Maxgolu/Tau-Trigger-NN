"""Compare smooth trigger surrogates with exact decisions before training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from classifiers import calibrate_classifier, parse_classifier
from constrained_objective import (
    calculate_hard_constraint_metrics,
    calibrate_tob_baseline,
    parse_constrained_objective,
    probability_at_least_k,
    soft_object_pass,
)
from event_data import EventBatch, _gev
from operating_point import select_background_objects


def _load_predictions(run_dir):
    parquet = Path(run_dir) / "predictions.parquet"
    csv = Path(run_dir) / "predictions.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No predictions file in {run_dir}")


def _padded_event_batch(frame):
    """Build one compact padded tensor batch for repeated temperature audits."""
    event_codes, event_numbers = pd.factorize(frame["eventNumber"], sort=False)
    object_positions = frame.groupby("eventNumber", sort=False).cumcount().to_numpy()
    event_count = len(event_numbers)
    max_objects = int(object_positions.max()) + 1
    shape = (event_count, max_objects)

    object_mask = np.zeros(shape, dtype=bool)
    scores = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.float32)
    truth_pt = np.zeros(shape, dtype=np.float32)
    tob_pt = np.zeros(shape, dtype=np.float32)
    signal_mask = np.zeros(shape, dtype=bool)
    target = (event_codes, object_positions)
    object_mask[target] = True
    scores[target] = frame["nn_score"].to_numpy(dtype=np.float32)
    labels[target] = frame["signal"].to_numpy(dtype=np.float32)
    truth_pt[target] = _gev(frame["truth_pt"]).astype(np.float32)
    tob_pt[target] = _gev(frame["tob_pt"]).astype(np.float32)
    signal_mask[target] = (
        (frame["Type"].to_numpy() == "Signal")
        & (frame["signal"].to_numpy(dtype=np.int64) == 1)
    )
    background_rows = frame["Type"].isin(["BKG", "Background"])
    background_event = (
        background_rows.groupby(frame["eventNumber"], sort=False).all().to_numpy()
    )
    return EventBatch(
        features=torch.from_numpy(scores[..., None]),
        labels=torch.from_numpy(labels),
        truth_pt_gev=torch.from_numpy(truth_pt),
        tob_pt_gev=torch.from_numpy(tob_pt),
        object_mask=torch.from_numpy(object_mask),
        signal_object_mask=torch.from_numpy(signal_mask),
        background_event_mask=torch.from_numpy(background_event.copy()),
        event_numbers=torch.from_numpy(np.asarray(event_numbers, dtype=np.int64)),
    )


def _soft_metrics_from_batch(
    batch,
    calibration,
    baseline_threshold_gev,
    classifier_name,
    objective_config,
):
    """Accumulate soft rates over a reusable complete-event tensor."""
    signal_sums = np.zeros(len(objective_config.regions_gev), dtype=np.float64)
    baseline_sums = np.zeros(len(objective_config.regions_gev), dtype=np.float64)
    signal_counts = np.zeros(len(objective_config.regions_gev), dtype=np.int64)
    with torch.no_grad():
        object_scores = batch.features[..., 0]
        probabilities = soft_object_pass(
            object_scores,
            calibration["nn_threshold"],
            objective_config.temperature,
            classifier_name,
            batch.tob_pt_gev,
            calibration.get("tob_threshold_gev"),
        )
        event_probability = probability_at_least_k(
            probabilities,
            batch.object_mask,
            objective_config.trigger_objects,
        )
        background_sum = float(event_probability[batch.background_event_mask].sum())
        background_count = int(batch.background_event_mask.sum())

        baseline_pass = batch.tob_pt_gev >= baseline_threshold_gev
        for index, (low, high) in enumerate(objective_config.regions_gev):
            selected = (
                batch.signal_object_mask
                & batch.object_mask
                & (batch.truth_pt_gev >= low)
                & (batch.truth_pt_gev < high)
            )
            signal_counts[index] = int(selected.sum())
            signal_sums[index] = float((probabilities * selected).sum())
            baseline_sums[index] = float((baseline_pass * selected).sum())

    if background_count == 0:
        raise ValueError("Diagnostic input does not contain background events")
    efficiencies = np.divide(
        signal_sums,
        signal_counts,
        out=np.zeros_like(signal_sums),
        where=signal_counts > 0,
    )
    baseline_efficiencies = np.divide(
        baseline_sums,
        signal_counts,
        out=np.zeros_like(baseline_sums),
        where=signal_counts > 0,
    )
    valid = signal_counts > 0
    weights = np.asarray(objective_config.region_weights) * valid
    if weights.sum():
        weights /= weights.sum()
    deltas = efficiencies - baseline_efficiencies
    return {
        "objective_value": float(np.sum(weights * deltas)),
        "event_fpr": float(background_sum / background_count),
        "region_efficiencies": efficiencies.tolist(),
        "baseline_efficiencies": baseline_efficiencies.tolist(),
        "region_deltas": deltas.tolist(),
        "region_counts": signal_counts.tolist(),
    }


def diagnose_run(run_dir, classifier_config, objective_raw, temperatures):
    """Measure hard/soft agreement for one saved prediction table."""
    frame = _load_predictions(run_dir)
    background = select_background_objects(frame)
    calibration = calibrate_classifier(
        background,
        background["nn_score"].to_numpy(dtype=np.float64),
        classifier_config,
    )
    objective_config = parse_constrained_objective(
        {"loss": {"name": "constrained_trigger", **objective_raw}}
    )
    baseline_threshold, _ = calibrate_tob_baseline(
        background,
        objective_config.target_event_fpr,
        objective_config.trigger_objects,
    )
    hard = calculate_hard_constraint_metrics(
        frame,
        frame["nn_score"].to_numpy(dtype=np.float64),
        classifier_config,
        objective_config,
        calibration=calibration,
        baseline_threshold_gev=baseline_threshold,
    )
    event_batch = _padded_event_batch(frame)
    soft = {}
    for temperature in temperatures:
        temperature_config = parse_constrained_objective(
            {
                "loss": {
                    "name": "constrained_trigger",
                    **objective_raw,
                    "temperature": float(temperature),
                }
            }
        )
        soft[f"{temperature:g}"] = _soft_metrics_from_batch(
            event_batch,
            calibration,
            baseline_threshold,
            classifier_config.name,
            temperature_config,
        )
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "hard": hard,
        "soft_by_temperature": soft,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Run folders or batch folders")
    parser.add_argument("--classifier", choices=("nn_only", "tob_nn_or"), default="nn_only")
    parser.add_argument("--tob-fpr", type=float, default=0.001)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    parser.add_argument("--output", default="constrained_surrogate_diagnostic.json")
    return parser.parse_args()


def _discover_runs(inputs):
    discovered = []
    for raw in inputs:
        path = Path(raw)
        if (path / "config.json").exists():
            discovered.append(path)
        else:
            discovered.extend(sorted(file.parent for file in path.glob("*/config.json")))
    return discovered


def main():
    args = parse_args()
    classifier_raw = {
        "name": args.classifier,
        "target_fpr": 0.005,
        "trigger_objects": 2,
    }
    if args.classifier == "tob_nn_or":
        classifier_raw["tob_fpr"] = args.tob_fpr
    classifier = parse_classifier({"classifier": classifier_raw})
    objective_raw = {
        "target_event_fpr": 0.005,
        "trigger_objects": 2,
        "regions_gev": [[25, 40], [40, 60], [60, 120]],
        "region_weights": [1 / 3, 1 / 3, 1 / 3],
        "allowed_deficits": [0.005, 0.005, 0.005],
    }
    results = [
        diagnose_run(path, classifier, objective_raw, args.temperatures)
        for path in _discover_runs(args.runs)
    ]
    output = {
        "note": (
            "Engineering diagnostic only. Do not choose scientific settings "
            "from test predictions."
        ),
        "classifier": classifier.to_dict(),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved surrogate diagnostic for {len(results)} run(s): {args.output}")


if __name__ == "__main__":
    main()
