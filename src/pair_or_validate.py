"""Cross-fitted validation for a neural-pair plus measured-pT OR classifier."""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .pair_classifiers import (
        calibrate_joint_or,
        joint_or_decisions,
        pass_scores,
    )
    from .pair_evaluate import calibrate_event_threshold
except ImportError:
    from pair_classifiers import calibrate_joint_or, joint_or_decisions, pass_scores
    from pair_evaluate import calibrate_event_threshold


REQUIRED_EVENT_COLUMNS = {
    "global_event_id",
    "sample",
    "pair_observable",
    "pair_eligible",
    "fold",
    "model_seed",
    "checkpoint_id",
    "event_score",
    "baseline_event_score",
    "validation_bce",
}
DEFAULT_PT_BUDGET_GRID = tuple(np.arange(0.0, 0.0051, 0.0005).tolist())
DEFAULT_OBJECTIVE_EDGES_GEV = tuple(np.arange(25.0, 100.1, 5.0).tolist())
DEFAULT_PROTECTED_EDGES_GEV = tuple(np.arange(25.0, 120.1, 5.0).tolist())
DEFAULT_REGIONS_GEV = (
    ("10_to_25", 10.0, 25.0),
    ("25_to_60", 25.0, 60.0),
    ("60_plus", 60.0, math.inf),
)


def _rate(accepted, events):
    return accepted / events if events else None


def _checkpoint_epoch(checkpoint_id):
    match = re.search(r"(\d+)$", str(checkpoint_id))
    if not match:
        raise ValueError(f"checkpoint ID has no terminal epoch: {checkpoint_id}")
    return int(match.group(1))


def _paired_counts(model_pass, baseline_pass):
    model = np.asarray(model_pass, dtype=bool)
    baseline = np.asarray(baseline_pass, dtype=bool)
    if model.shape != baseline.shape or model.ndim != 1:
        raise ValueError("paired decision arrays must be one-dimensional and aligned")
    events = len(model)
    model_only = int((model & ~baseline).sum())
    baseline_only = int((~model & baseline).sum())
    delta = (model_only - baseline_only) / events if events else None
    if events:
        differences = model.astype(np.float64) - baseline.astype(np.float64)
        standard_error = (
            float(differences.std(ddof=1) / math.sqrt(events)) if events > 1 else 0.0
        )
        confidence_interval = [
            float(delta - 1.96 * standard_error),
            float(delta + 1.96 * standard_error),
        ]
    else:
        standard_error = None
        confidence_interval = [None, None]
    return {
        "events": events,
        "model_accepted": int(model.sum()),
        "baseline_accepted": int(baseline.sum()),
        "both_accepted": int((model & baseline).sum()),
        "model_only": model_only,
        "baseline_only": baseline_only,
        "neither": int((~model & ~baseline).sum()),
        "model_efficiency": _rate(int(model.sum()), events),
        "baseline_efficiency": _rate(int(baseline.sum()), events),
        "efficiency_delta": delta,
        "paired_standard_error": standard_error,
        "paired_95pct_interval": confidence_interval,
    }


def _population_masks(events):
    signal = events["sample"].eq("signal").to_numpy()
    background = events["sample"].eq("background").to_numpy()
    observable = events["pair_observable"].astype(bool).to_numpy()
    eligible = events["pair_eligible"].astype(bool).to_numpy()
    return {
        "background": background,
        "inclusive_signal": signal,
        "pair_observable_signal": signal & observable,
        "pair_eligible_signal": signal & eligible,
    }


def _decision_counts(decisions, mask):
    events = int(mask.sum())
    accepted = int((decisions & mask).sum())
    return {"accepted": accepted, "events": events, "rate": _rate(accepted, events)}


def _window_records(energy, model_pass, baseline_pass, edges):
    energy = np.asarray(energy, dtype=np.float64)
    model_pass = np.asarray(model_pass, dtype=bool)
    baseline_pass = np.asarray(baseline_pass, dtype=bool)
    if not (energy.shape == model_pass.shape == baseline_pass.shape):
        raise ValueError("energy and decision arrays must be aligned")
    records = []
    for low, high in pairwise(edges):
        mask = (energy >= low) & (energy < high)
        comparison = _paired_counts(model_pass[mask], baseline_pass[mask])
        comparison.update({"low_gev": float(low), "high_gev": float(high)})
        records.append(comparison)
    return records


def _regional_records(energy, model_pass, baseline_pass, regions):
    records = []
    for name, low, high in regions:
        mask = (energy >= low) & (energy < high)
        comparison = _paired_counts(model_pass[mask], baseline_pass[mask])
        comparison.update(
            {
                "region": name,
                "low_gev": float(low),
                "high_gev": None if math.isinf(high) else float(high),
            }
        )
        records.append(comparison)
    return records


def _validate_prediction_table(predictions):
    missing = REQUIRED_EVENT_COLUMNS.difference(predictions.columns)
    if missing:
        raise KeyError(f"missing event-prediction columns: {sorted(missing)}")
    keys = ["model_seed", "checkpoint_id", "global_event_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("event predictions contain duplicate run/event identities")
    if not set(predictions["sample"]).issubset({"signal", "background"}):
        raise ValueError("sample must be signal or background")
    if set(predictions["fold"]) != {"A", "B"}:
        raise ValueError("validation predictions must cover folds A and B")

    grouped = list(predictions.groupby(["model_seed", "checkpoint_id"], sort=True))
    if not grouped:
        raise ValueError("event-prediction table cannot be empty")
    metadata_columns = [
        "global_event_id",
        "sample",
        "pair_observable",
        "pair_eligible",
        "fold",
        "baseline_event_score",
    ]
    reference = grouped[0][1].sort_values("global_event_id")
    for _, group in grouped:
        candidate = group.sort_values("global_event_id")
        if len(candidate) != len(reference):
            raise ValueError("every checkpoint must cover the same validation events")
        for column in metadata_columns:
            if not np.array_equal(
                reference[column].to_numpy(), candidate[column].to_numpy()
            ):
                raise ValueError(f"event metadata differ across checkpoints: {column}")

        pair_observable = candidate["pair_observable"].astype(bool).to_numpy()
        for column in ("event_score", "baseline_event_score"):
            scores = candidate[column].to_numpy(dtype=np.float64)
            if np.isnan(scores).any() or np.isposinf(scores).any():
                raise ValueError(f"{column} contains NaN or positive infinity")
            if not np.isfinite(scores[pair_observable]).all():
                raise ValueError(f"paired events require finite {column}")
            if not np.isneginf(scores[~pair_observable]).all():
                raise ValueError(f"no-pair events require -infinity {column}")
        bce = candidate["validation_bce"].to_numpy(dtype=np.float64)
        if not np.isfinite(bce).all() or not np.all(bce == bce[0]):
            raise ValueError("validation BCE must be one finite value per checkpoint")


def _validate_energy_table(predictions, energy_table, energy_column):
    required = {"global_event_id", energy_column}
    missing = required.difference(energy_table.columns)
    if missing:
        raise KeyError(f"missing energy-profile columns: {sorted(missing)}")
    if energy_table["global_event_id"].duplicated().any():
        raise ValueError("energy profile must contain one row per event")
    energies = energy_table[energy_column].to_numpy(dtype=np.float64)
    if not np.isfinite(energies).all():
        raise ValueError("energy profile contains non-finite values")

    reference = predictions.drop_duplicates("global_event_id")
    expected = set(
        reference.loc[
            reference["sample"].eq("signal")
            & reference["pair_eligible"].astype(bool),
            "global_event_id",
        ].astype(int)
    )
    observed = set(energy_table["global_event_id"].astype(int))
    if observed != expected:
        raise ValueError(
            "energy profile must cover exactly the pair-eligible signal events"
        )


def _comparison_by_population(events, model_pass, baseline_pass):
    return {
        name: _paired_counts(model_pass[mask], baseline_pass[mask])
        for name, mask in _population_masks(events).items()
    }


def _branch_overlap(events, pt_pass, neural_pass):
    result = {}
    for name, mask in _population_masks(events).items():
        pt = pt_pass[mask]
        neural = neural_pass[mask]
        result[name] = {
            "events": int(mask.sum()),
            "pt_only": int((pt & ~neural).sum()),
            "neural_only": int((~pt & neural).sum()),
            "both": int((pt & neural).sum()),
            "neither": int((~pt & ~neural).sum()),
            "union": int((pt | neural).sum()),
        }
    return result


def _evaluate_candidate(
    events,
    energy_table,
    *,
    pt_budget,
    target_fpr,
    objective_edges,
    protected_edges,
    allowed_protected_deficit,
    energy_column,
    baseline_fold_calibrations,
):
    heldout_frames = []
    leg_reports = []
    for calibration_fold, evaluation_fold in (("A", "B"), ("B", "A")):
        calibration_background = events[
            events["fold"].eq(calibration_fold)
            & events["sample"].eq("background")
        ]
        thresholds = calibrate_joint_or(
            calibration_background["baseline_event_score"],
            calibration_background["event_score"],
            pt_budget=pt_budget,
            target_fpr=target_fpr,
        )
        baseline_calibration = baseline_fold_calibrations[calibration_fold]
        baseline_threshold = baseline_calibration["threshold"]
        baseline_calibration_fpr = baseline_calibration["achieved_fpr"]
        heldout = events.loc[events["fold"].eq(evaluation_fold)].copy()
        heldout["model_pass"] = joint_or_decisions(
            heldout["baseline_event_score"], heldout["event_score"], thresholds
        )
        heldout["baseline_pass"] = pass_scores(
            heldout["baseline_event_score"], baseline_threshold
        )
        heldout_frames.append(heldout)
        masks = _population_masks(heldout)
        leg_reports.append(
            {
                "calibration_fold": calibration_fold,
                "evaluation_fold": evaluation_fold,
                "pt_budget": float(pt_budget),
                "pt_threshold": thresholds.pt_threshold,
                "neural_threshold": thresholds.neural_threshold,
                "joint_calibration_background": {
                    "events": thresholds.background_events,
                    "pt_accepted": thresholds.pt_accepted,
                    "neural_accepted": thresholds.neural_accepted,
                    "overlap_accepted": thresholds.overlap_accepted,
                    "union_accepted": thresholds.union_accepted,
                    "union_fpr": thresholds.union_fpr,
                },
                "baseline_threshold": baseline_threshold,
                "baseline_calibration_fpr": baseline_calibration_fpr,
                "heldout": {
                    name: {
                        "model": _decision_counts(
                            heldout["model_pass"].to_numpy(), mask
                        ),
                        "baseline": _decision_counts(
                            heldout["baseline_pass"].to_numpy(), mask
                        ),
                    }
                    for name, mask in masks.items()
                },
            }
        )

    heldout = pd.concat(heldout_frames, ignore_index=True).sort_values(
        "global_event_id"
    )
    if heldout["global_event_id"].duplicated().any() or len(heldout) != len(events):
        raise RuntimeError("cross-fitting did not cover every validation event once")
    energy_decisions = energy_table.merge(
        heldout[["global_event_id", "model_pass", "baseline_pass"]],
        on="global_event_id",
        how="left",
        validate="one_to_one",
    )
    if energy_decisions[["model_pass", "baseline_pass"]].isna().any().any():
        raise RuntimeError("missing held-out decision for an energy-profile event")
    energy = energy_decisions[energy_column].to_numpy(dtype=np.float64)
    model_pass = energy_decisions["model_pass"].to_numpy(dtype=bool)
    baseline_pass = energy_decisions["baseline_pass"].to_numpy(dtype=bool)
    objective_windows = _window_records(
        energy, model_pass, baseline_pass, objective_edges
    )
    protected_windows = _window_records(
        energy, model_pass, baseline_pass, protected_edges
    )
    if any(record["events"] == 0 for record in objective_windows):
        raise ValueError("every objective energy window must contain an event")
    if any(record["events"] == 0 for record in protected_windows):
        raise ValueError("every protected energy window must contain an event")
    objective = float(
        np.mean([record["efficiency_delta"] for record in objective_windows])
    )
    worst_protected = float(
        min(record["efficiency_delta"] for record in protected_windows)
    )
    comparisons = _comparison_by_population(
        heldout,
        heldout["model_pass"].to_numpy(dtype=bool),
        heldout["baseline_pass"].to_numpy(dtype=bool),
    )
    return {
        "pt_budget": float(pt_budget),
        "crossfit_legs": leg_reports,
        "pooled_comparisons": comparisons,
        "objective_windows": objective_windows,
        "protected_windows": protected_windows,
        "equal_window_objective": objective,
        "worst_protected_window_delta": worst_protected,
        "protection_feasible": worst_protected >= allowed_protected_deficit,
    }


def _select_candidate(records, objective_tolerance):
    if not records:
        raise ValueError("candidate records cannot be empty")
    feasible = [record for record in records if record["protection_feasible"]]
    if not feasible:
        return max(
            records,
            key=lambda record: (
                record["worst_protected_window_delta"],
                record["equal_window_objective"],
                -record["pt_budget"],
                -record["epoch"],
            ),
        ), False
    pool = feasible if feasible else records
    best = pool[0]
    for candidate in pool[1:]:
        difference = (
            candidate["equal_window_objective"] - best["equal_window_objective"]
        )
        if difference > objective_tolerance:
            best = candidate
            continue
        if difference < -objective_tolerance:
            continue

        candidate_ties = (
            candidate["worst_protected_window_delta"],
            candidate["pooled_comparisons"]["inclusive_signal"]["model_accepted"],
            candidate["pooled_comparisons"]["pair_eligible_signal"]["model_accepted"],
            -candidate["pooled_comparisons"]["background"]["model_accepted"],
        )
        best_ties = (
            best["worst_protected_window_delta"],
            best["pooled_comparisons"]["inclusive_signal"]["model_accepted"],
            best["pooled_comparisons"]["pair_eligible_signal"]["model_accepted"],
            -best["pooled_comparisons"]["background"]["model_accepted"],
        )
        if candidate_ties > best_ties:
            best = candidate
            continue
        if candidate_ties < best_ties:
            continue

        bce_difference = candidate["validation_bce"] - best["validation_bce"]
        if bce_difference < -1e-12:
            best = candidate
            continue
        if bce_difference > 1e-12:
            continue
        if (candidate["pt_budget"], candidate["epoch"]) < (
            best["pt_budget"],
            best["epoch"],
        ):
            best = candidate
    return best, True


def _full_validation(
    events,
    energy_table,
    selected,
    *,
    target_fpr,
    energy_column,
    baseline_full_calibration,
):
    background = events["sample"].eq("background").to_numpy()
    thresholds = calibrate_joint_or(
        events.loc[background, "baseline_event_score"],
        events.loc[background, "event_score"],
        pt_budget=selected["pt_budget"],
        target_fpr=target_fpr,
    )
    baseline_threshold = baseline_full_calibration["threshold"]
    baseline_fpr = baseline_full_calibration["achieved_fpr"]
    pt_pass = pass_scores(events["baseline_event_score"], thresholds.pt_threshold)
    neural_pass = pass_scores(events["event_score"], thresholds.neural_threshold)
    model_pass = pt_pass | neural_pass
    baseline_pass = pass_scores(events["baseline_event_score"], baseline_threshold)
    energy_decisions = energy_table.merge(
        pd.DataFrame(
            {
                "global_event_id": events["global_event_id"],
                "model_pass": model_pass,
                "baseline_pass": baseline_pass,
            }
        ),
        on="global_event_id",
        how="left",
        validate="one_to_one",
    )
    energy = energy_decisions[energy_column].to_numpy(dtype=np.float64)
    energy_model = energy_decisions["model_pass"].to_numpy(dtype=bool)
    energy_baseline = energy_decisions["baseline_pass"].to_numpy(dtype=bool)
    comparisons = _comparison_by_population(events, model_pass, baseline_pass)
    regional = _regional_records(
        energy, energy_model, energy_baseline, DEFAULT_REGIONS_GEV
    )
    positive_regions = all(
        record["events"] > 0 and record["efficiency_delta"] > 0
        for record in regional
    )
    return {
        "joint_thresholds": {
            "pt_budget": selected["pt_budget"],
            "pt_threshold": thresholds.pt_threshold,
            "neural_threshold": thresholds.neural_threshold,
            "background_events": thresholds.background_events,
            "pt_accepted": thresholds.pt_accepted,
            "neural_accepted": thresholds.neural_accepted,
            "overlap_accepted": thresholds.overlap_accepted,
            "union_accepted": thresholds.union_accepted,
            "union_fpr": thresholds.union_fpr,
        },
        "baseline_threshold": baseline_threshold,
        "baseline_fpr": baseline_fpr,
        "comparisons": comparisons,
        "branch_overlap": _branch_overlap(events, pt_pass, neural_pass),
        "conditional_energy_regions": regional,
        "adaptive_validation_gate_passed": (
            comparisons["inclusive_signal"]["efficiency_delta"] > 0
            and positive_regions
        ),
        "protected_confirmation_required": True,
        "promotion_allowed_from_this_validation_report": False,
    }


def validate_or_family(
    predictions,
    energy_table,
    *,
    pt_budget_grid=DEFAULT_PT_BUDGET_GRID,
    target_fpr=0.005,
    objective_edges_gev=DEFAULT_OBJECTIVE_EDGES_GEV,
    protected_edges_gev=DEFAULT_PROTECTED_EDGES_GEV,
    allowed_protected_deficit=-0.005,
    objective_tolerance=0.002,
    energy_column="member_min_truth_pt_gev",
):
    """Select and calibrate one raw-cell neural+pT OR model per seed."""
    predictions = predictions.copy()
    energy_table = energy_table.copy()
    _validate_prediction_table(predictions)
    _validate_energy_table(predictions, energy_table, energy_column)
    budgets = tuple(float(value) for value in pt_budget_grid)
    if not budgets or any(not 0 <= value <= target_fpr for value in budgets):
        raise ValueError("pT budgets must lie between zero and target FPR")

    reference = predictions.groupby(
        ["model_seed", "checkpoint_id"], sort=True
    ).first().reset_index()
    reference_seed = int(reference["model_seed"].iloc[0])
    reference_checkpoint = str(reference["checkpoint_id"].iloc[0])
    reference = predictions.loc[
        predictions["model_seed"].eq(reference_seed)
        & predictions["checkpoint_id"].eq(reference_checkpoint)
    ]
    baseline_fold_calibrations = {}
    for calibration_fold in ("A", "B"):
        calibration_scores = reference.loc[
            reference["fold"].eq(calibration_fold)
            & reference["sample"].eq("background"),
            "baseline_event_score",
        ]
        threshold, achieved_fpr = calibrate_event_threshold(
            calibration_scores, target_fpr
        )
        baseline_fold_calibrations[calibration_fold] = {
            "threshold": threshold,
            "achieved_fpr": achieved_fpr,
            "background_events": len(calibration_scores),
        }
    full_background_scores = reference.loc[
        reference["sample"].eq("background"), "baseline_event_score"
    ]
    full_baseline_threshold, full_baseline_fpr = calibrate_event_threshold(
        full_background_scores, target_fpr
    )
    baseline_full_calibration = {
        "threshold": full_baseline_threshold,
        "achieved_fpr": full_baseline_fpr,
        "background_events": len(full_background_scores),
    }

    report = {
        "workflow": "raw-cell neural pair score OR measured-pT pair score",
        "data_scope": "validation predictions only; test outcomes are not inputs",
        "target_event_fpr": float(target_fpr),
        "pt_budget_grid": list(budgets),
        "selection_objective": {
            "energy_column": energy_column,
            "objective_edges_gev": list(objective_edges_gev),
            "protected_edges_gev": list(protected_edges_gev),
            "allowed_protected_deficit": float(allowed_protected_deficit),
            "objective_tolerance": float(objective_tolerance),
        },
        "energy_interpretation": (
            "minimum TOB-associated truth_pt for operational label-2 pairs; "
            "not an inclusive distinct-generator-tau quantity"
        ),
        "baseline_calibration": {
            "folds": baseline_fold_calibrations,
            "complete_validation": baseline_full_calibration,
            "reuse": "calculated once and reused across every candidate and seed",
        },
        "seeds": {},
    }
    for seed, seed_predictions in predictions.groupby("model_seed", sort=True):
        records = []
        for checkpoint_id, events in seed_predictions.groupby(
            "checkpoint_id", sort=True
        ):
            epoch = _checkpoint_epoch(checkpoint_id)
            validation_bce = float(events["validation_bce"].iloc[0])
            for budget in budgets:
                record = _evaluate_candidate(
                    events,
                    energy_table,
                    pt_budget=budget,
                    target_fpr=target_fpr,
                    objective_edges=objective_edges_gev,
                    protected_edges=protected_edges_gev,
                    allowed_protected_deficit=allowed_protected_deficit,
                    energy_column=energy_column,
                    baseline_fold_calibrations=baseline_fold_calibrations,
                )
                record.update(
                    {
                        "model_seed": int(seed),
                        "checkpoint_id": str(checkpoint_id),
                        "epoch": epoch,
                        "validation_bce": validation_bce,
                    }
                )
                records.append(record)
        selected, had_feasible_candidate = _select_candidate(
            records, objective_tolerance
        )
        selected = dict(selected)
        selected["selection_status"] = (
            "feasible_selected"
            if had_feasible_candidate
            else "no_feasible_candidate_descriptive_only"
        )
        selected_events = seed_predictions.loc[
            seed_predictions["checkpoint_id"].eq(selected["checkpoint_id"])
        ].copy()
        report["seeds"][str(int(seed))] = {
            "candidate_count": len(records),
            "had_protection_feasible_candidate": had_feasible_candidate,
            "selection": selected,
            "full_validation": _full_validation(
                selected_events,
                energy_table,
                selected,
                target_fpr=target_fpr,
                energy_column=energy_column,
                baseline_full_calibration=baseline_full_calibration,
            ),
            "candidates": records,
        }
    return report


def load_prediction_directory(path):
    """Load all event-prediction files produced by ``pair_validate.py``."""
    files = sorted(Path(path).rglob("event_predictions.csv.gz"))
    if not files:
        files = sorted(Path(path).rglob("event_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"no event prediction tables found below {path}")
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)


def load_or_configs(path):
    """Load per-seed OR configs and require one identical validation contract."""
    files = sorted(Path(path).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no OR configuration files found in {path}")
    configs = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    seeds = [int(config["seed"]) for config in configs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("OR configuration seeds must be unique")
    reference = dict(configs[0])
    reference.pop("seed")
    for config in configs[1:]:
        candidate = dict(config)
        candidate.pop("seed")
        if candidate != reference:
            raise ValueError("per-seed OR validation settings must be identical")
    return configs, reference["validation"]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isinf(number):
            return "+inf" if number > 0 else "-inf"
        if math.isnan(number):
            raise ValueError("NaN cannot be serialized")
        return number
    return value


def _candidate_table(report):
    rows = []
    for seed, seed_report in report["seeds"].items():
        selected = seed_report["selection"]
        for candidate in seed_report["candidates"]:
            rows.append(
                {
                    "model_seed": int(seed),
                    "checkpoint_id": candidate["checkpoint_id"],
                    "epoch": candidate["epoch"],
                    "pt_budget": candidate["pt_budget"],
                    "equal_window_objective": candidate["equal_window_objective"],
                    "worst_protected_window_delta": candidate[
                        "worst_protected_window_delta"
                    ],
                    "protection_feasible": candidate["protection_feasible"],
                    "inclusive_signal_accepted": candidate["pooled_comparisons"][
                        "inclusive_signal"
                    ]["model_accepted"],
                    "eligible_signal_accepted": candidate["pooled_comparisons"][
                        "pair_eligible_signal"
                    ]["model_accepted"],
                    "background_accepted": candidate["pooled_comparisons"][
                        "background"
                    ]["model_accepted"],
                    "validation_bce": candidate["validation_bce"],
                    "selected": (
                        candidate["checkpoint_id"] == selected["checkpoint_id"]
                        and candidate["pt_budget"] == selected["pt_budget"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the raw-cell neural plus measured-pT OR classifier"
    )
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--energy-profile", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = load_prediction_directory(args.prediction_dir)
    energy_table = pd.read_csv(args.energy_profile)
    configs, settings = load_or_configs(args.config_dir)
    configured_seeds = {int(config["seed"]) for config in configs}
    prediction_seeds = set(predictions["model_seed"].astype(int))
    if configured_seeds != prediction_seeds:
        raise ValueError("prediction seeds do not match the OR configurations")
    report = validate_or_family(
        predictions,
        energy_table,
        pt_budget_grid=settings["pt_budget_grid"],
        target_fpr=settings["target_event_fpr"],
        objective_edges_gev=settings["objective_energy_edges_gev"],
        protected_edges_gev=settings["protected_energy_edges_gev"],
        allowed_protected_deficit=settings["allowed_protected_deficit"],
        objective_tolerance=settings["objective_tolerance"],
        energy_column=settings["energy_column"],
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "or_validation_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _candidate_table(report).to_csv(
        output_dir / "or_candidate_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
