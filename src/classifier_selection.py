"""Leakage-safe validation search for the TOB-NN OR budget."""

import numpy as np
import pandas as pd

from classifiers import (
    calibrate_classifier,
    classifier_event_pass_mask,
    classifier_object_pass_mask,
    tob_pt_gev,
)
from operating_point import (
    build_event_trigger_scores,
    select_fpr_threshold,
)


def _sample_type(values):
    background = pd.Series(values).isin(["BKG", "Background"])
    return np.where(background, "BKG", "Signal")


def _truth_pt_gev(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1000.0:
        values = values / 1000.0
    return values


def _second_highest(values):
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    return float(values[1]) if len(values) > 1 else float(values[0])


def build_validation_folds(validation_frame, seed, folds=2):
    """Split complete events into deterministic, approximately balanced folds."""
    if folds != 2:
        raise ValueError("TOB budget search currently requires exactly 2 folds")

    frame = validation_frame.copy()
    frame["_sample_type"] = _sample_type(frame["Type"])
    frame["_tob_pt_gev"] = tob_pt_gev(frame)
    frame["_truth_pt_gev"] = _truth_pt_gev(frame["truth_pt"])
    event_columns = ["_sample_type", "eventNumber"]

    events = []
    for key, group in frame.groupby(event_columns, sort=True):
        truth_tau = group.loc[group["signal"].to_numpy() == 1]
        events.append(
            {
                "_sample_type": key[0],
                "eventNumber": key[1],
                "second_tob_pt": _second_highest(group["_tob_pt_gev"]),
                "tau_count": int(len(truth_tau)),
                "leading_truth_pt": (
                    float(truth_tau["_truth_pt_gev"].max())
                    if len(truth_tau)
                    else -1.0
                ),
            }
        )
    event_frame = pd.DataFrame(events)

    background = event_frame["_sample_type"] == "BKG"
    ranks = event_frame.loc[background, "second_tob_pt"].rank(
        method="first", pct=True
    )
    quantiles = np.minimum((ranks * 20).astype(int), 19)
    event_frame.loc[background, "_stratum"] = (
        "BKG_q" + quantiles.astype(str)
    )
    signal = ~background
    tau_groups = event_frame.loc[signal, "tau_count"].clip(upper=2).astype(str)
    pt_groups = np.floor(
        event_frame.loc[signal, "leading_truth_pt"].clip(lower=0.0) / 10.0
    ).clip(upper=12).astype(int).astype(str)
    event_frame.loc[signal, "_stratum"] = (
        "Signal_m" + tau_groups + "_p" + pt_groups
    )

    rng = np.random.default_rng(int(seed))
    event_frame["_fold"] = -1
    start_fold = 0
    for _, indices in event_frame.groupby("_stratum", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        indices = rng.permutation(indices)
        assigned = (np.arange(len(indices)) + start_fold) % folds
        event_frame.loc[indices, "_fold"] = assigned
        start_fold = (start_fold + len(indices)) % folds

    fold_map = {
        (row["_sample_type"], row["eventNumber"]): int(row["_fold"])
        for _, row in event_frame.iterrows()
    }
    row_keys = list(zip(frame["_sample_type"], frame["eventNumber"]))
    fold_ids = np.asarray([fold_map[key] for key in row_keys], dtype=np.int8)

    audit = {"seed": int(seed), "folds": folds, "event_counts": {}}
    for fold in range(folds):
        selected = event_frame[event_frame["_fold"] == fold]
        audit["event_counts"][str(fold)] = {
            "total": int(len(selected)),
            "background": int((selected["_sample_type"] == "BKG").sum()),
            "signal": int((selected["_sample_type"] == "Signal").sum()),
            "zero_tau": int((selected["tau_count"] == 0).sum()),
            "one_tau": int((selected["tau_count"] == 1).sum()),
            "two_plus_tau": int((selected["tau_count"] >= 2).sum()),
            "strata": {
                str(name): int(count)
                for name, count in selected["_stratum"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
    return fold_ids, audit


def _baseline_calibration(background, target_fpr, trigger_objects):
    frame = background.copy()
    frame["_tob_pt_gev"] = tob_pt_gev(frame)
    event_scores, event_count = build_event_trigger_scores(
        frame, "_tob_pt_gev", objects=trigger_objects
    )
    threshold, achieved_fpr = select_fpr_threshold(
        event_scores, event_count, target_fpr
    )
    return threshold, achieved_fpr


def _window_metrics(signal_parts, objective):
    signal = pd.concat(signal_parts, ignore_index=True)
    regular_edges = np.arange(
        objective.min_truth_pt_gev,
        objective.protected_max_truth_pt_gev,
        objective.window_width_gev,
    )
    # Explicit boundaries keep configurable objective and saturation regions exact.
    edges = np.unique(
        np.concatenate(
            [
                regular_edges,
                [
                    objective.objective_max_truth_pt_gev,
                    objective.saturation_start_truth_pt_gev,
                    objective.protected_max_truth_pt_gev,
                ],
            ]
        )
    )
    windows = []
    objective_deltas = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (signal["truth_pt_gev"] >= low) & (signal["truth_pt_gev"] < high)
        count = int(selected.sum())
        if count:
            or_eff = float(signal.loc[selected, "or_pass"].mean())
            baseline_eff = float(signal.loc[selected, "baseline_pass"].mean())
            delta = or_eff - baseline_eff
            if low < objective.objective_max_truth_pt_gev:
                objective_deltas.append(delta)
        else:
            or_eff = baseline_eff = delta = None
        windows.append(
            {
                "low_gev": float(low),
                "high_gev": float(high),
                "object_count": count,
                "or_efficiency": or_eff,
                "baseline_efficiency": baseline_eff,
                "delta": delta,
            }
        )

    if objective.noninferiority_mode == "per_window":
        protection_regions = [dict(window, pooled=False) for window in windows]
    else:
        protection_regions = [
            dict(window, pooled=False)
            for window in windows
            if window["high_gev"] <= objective.saturation_start_truth_pt_gev
        ]
        pooled = (
            (signal["truth_pt_gev"] >= objective.saturation_start_truth_pt_gev)
            & (signal["truth_pt_gev"] < objective.protected_max_truth_pt_gev)
        )
        pooled_count = int(pooled.sum())
        if pooled_count:
            pooled_or_eff = float(signal.loc[pooled, "or_pass"].mean())
            pooled_baseline_eff = float(signal.loc[pooled, "baseline_pass"].mean())
            pooled_delta = pooled_or_eff - pooled_baseline_eff
        else:
            pooled_or_eff = pooled_baseline_eff = pooled_delta = None
        protection_regions.append(
            {
                "low_gev": float(objective.saturation_start_truth_pt_gev),
                "high_gev": float(objective.protected_max_truth_pt_gev),
                "object_count": pooled_count,
                "or_efficiency": pooled_or_eff,
                "baseline_efficiency": pooled_baseline_eff,
                "delta": pooled_delta,
                "pooled": True,
            }
        )

    objective_window_count = sum(
        window["low_gev"] < objective.objective_max_truth_pt_gev
        for window in windows
    )
    complete = (
        len(objective_deltas) == objective_window_count
        and all(region["delta"] is not None for region in protection_regions)
    )
    protected_deltas = [
        region["delta"]
        for region in protection_regions
        if region["delta"] is not None
    ]
    objective_value = (
        float(np.mean(objective_deltas)) if objective_deltas else -2.0
    )
    min_delta = float(np.min(protected_deltas)) if protected_deltas else -2.0
    return windows, protection_regions, objective_value, min_delta, complete


def is_better_budget_candidate(candidate, best, tie_tolerance):
    """Apply feasibility, objective, worst-window, then larger-budget ordering."""
    if best is None:
        return True
    if candidate["noninferiority_satisfied"] != best["noninferiority_satisfied"]:
        return candidate["noninferiority_satisfied"]
    if candidate["noninferiority_satisfied"]:
        difference = candidate["objective_value"] - best["objective_value"]
        if abs(difference) > tie_tolerance:
            return difference > 0.0
        if not np.isclose(candidate["minimum_delta"], best["minimum_delta"]):
            return candidate["minimum_delta"] > best["minimum_delta"]
    else:
        if not np.isclose(candidate["minimum_delta"], best["minimum_delta"]):
            return candidate["minimum_delta"] > best["minimum_delta"]
        if not np.isclose(candidate["objective_value"], best["objective_value"]):
            return candidate["objective_value"] > best["objective_value"]
    return candidate["tob_fpr"] > best["tob_fpr"]


def search_validation_tob_budget(
    validation_frame,
    scores,
    classifier_config,
    fold_ids,
    fold_audit,
):
    """Choose a TOB budget from held-out validation folds for one epoch."""
    if classifier_config.tob_budget is None:
        raise ValueError("Classifier does not request validation TOB-budget search")

    frame = validation_frame.copy()
    frame["nn_score"] = np.asarray(scores, dtype=np.float64)
    background_mask = frame["Type"].isin(["BKG", "Background"]).to_numpy()
    truth_tau_mask = (
        (frame["Type"].to_numpy() == "Signal")
        & (frame["signal"].to_numpy() == 1)
    )
    objective = classifier_config.tob_budget.objective
    candidates = []

    for tob_fpr in classifier_config.tob_budget.values:
        signal_parts = []
        background_passes = 0
        background_events = 0
        fold_calibrations = []
        for selection_fold in range(classifier_config.tob_budget.cross_validation_folds):
            calibration_rows = fold_ids != selection_fold
            selection_rows = fold_ids == selection_fold
            calibration_background = frame.loc[calibration_rows & background_mask]
            selected_background = frame.loc[selection_rows & background_mask]
            selected_signal = frame.loc[selection_rows & truth_tau_mask].copy()

            concrete = classifier_config.with_tob_fpr(tob_fpr)
            calibration = calibrate_classifier(
                calibration_background,
                calibration_background["nn_score"].to_numpy(),
                concrete,
            )
            selected_event_pass = classifier_event_pass_mask(
                selected_background, calibration
            )
            background_passes += int(selected_event_pass.sum())
            background_events += int(len(selected_event_pass))

            baseline_threshold, baseline_fpr = _baseline_calibration(
                calibration_background,
                classifier_config.target_fpr,
                classifier_config.trigger_objects,
            )
            selected_signal["truth_pt_gev"] = _truth_pt_gev(
                selected_signal["truth_pt"]
            )
            selected_signal["or_pass"] = classifier_object_pass_mask(
                selected_signal, calibration
            )
            signal_tob = tob_pt_gev(selected_signal)
            selected_signal["baseline_pass"] = (
                np.isfinite(signal_tob) & (signal_tob >= baseline_threshold)
            )
            signal_parts.append(
                selected_signal[
                    ["truth_pt_gev", "or_pass", "baseline_pass"]
                ]
            )
            fold_calibrations.append(
                {
                    "selection_fold": selection_fold,
                    "nn_threshold": float(calibration["nn_threshold"]),
                    "tob_threshold_gev": float(calibration["tob_threshold_gev"]),
                    "calibration_fpr": float(calibration["achieved_fpr"]),
                    "baseline_threshold_gev": float(baseline_threshold),
                    "baseline_calibration_fpr": float(baseline_fpr),
                }
            )

        achieved_fpr = background_passes / background_events
        (
            windows,
            protection_regions,
            objective_value,
            minimum_delta,
            complete,
        ) = _window_metrics(signal_parts, objective)
        signal = pd.concat(signal_parts, ignore_index=True)
        noninferiority = (
            complete
            and achieved_fpr <= classifier_config.target_fpr + 1e-12
            and minimum_delta >= -objective.noninferiority_tolerance
        )
        candidates.append(
            {
                "tob_fpr": float(tob_fpr),
                "achieved_fpr": float(achieved_fpr),
                "signal_efficiency": float(signal["or_pass"].mean()),
                "objective_value": float(objective_value),
                "minimum_delta": float(minimum_delta),
                "noninferiority_satisfied": bool(noninferiority),
                "complete_protected_windows": bool(complete),
                "windows": windows,
                "protection_regions": protection_regions,
                "fold_calibrations": fold_calibrations,
            }
        )

    best = None
    for candidate in candidates:
        if is_better_budget_candidate(
            candidate, best, objective.objective_tie_tolerance
        ):
            best = candidate

    return {
        "threshold": float(
            np.mean(
                [item["nn_threshold"] for item in best["fold_calibrations"]]
            )
        ),
        "target_fpr": float(classifier_config.target_fpr),
        "achieved_fpr": best["achieved_fpr"],
        "signal_efficiency": best["signal_efficiency"],
        "energy_band_efficiencies": {},
        "background_event_count": int(background_events),
        "signal_object_count": int(sum(len(part) for part in signal_parts)),
        "trigger_objects": int(classifier_config.trigger_objects),
        "selected_tob_fpr": best["tob_fpr"],
        "objective_value": best["objective_value"],
        "minimum_delta": best["minimum_delta"],
        "noninferiority_satisfied": best["noninferiority_satisfied"],
        "tob_budget_search": {
            "mode": "validation_search",
            "selected_tob_fpr": best["tob_fpr"],
            "objective_value": best["objective_value"],
            "minimum_delta": best["minimum_delta"],
            "noninferiority_satisfied": best["noninferiority_satisfied"],
            "objective": {
                "min_truth_pt_gev": objective.min_truth_pt_gev,
                "objective_max_truth_pt_gev": objective.objective_max_truth_pt_gev,
                "window_width_gev": objective.window_width_gev,
                "protected_max_truth_pt_gev": objective.protected_max_truth_pt_gev,
                "noninferiority_mode": objective.noninferiority_mode,
                "saturation_start_truth_pt_gev": (
                    objective.saturation_start_truth_pt_gev
                ),
                "noninferiority_tolerance": objective.noninferiority_tolerance,
                "objective_tie_tolerance": objective.objective_tie_tolerance,
            },
            "fold_audit": fold_audit,
            "candidates": candidates,
        },
    }
