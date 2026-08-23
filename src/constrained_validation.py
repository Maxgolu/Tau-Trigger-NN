"""Cross-fitted hard constraints and constrained-loss diagnostics."""

import math

import numpy as np

from classifiers import calibrate_classifier
from constrained_objective import (
    aggregate_paired_sufficient_statistics,
    build_confidence_feasibility,
    calibrate_constraint_classifier,
    calculate_hard_constraint_metrics,
    calibrate_tob_baseline,
    certified_calibration_target,
    one_sided_binomial_upper_bound,
)
from event_data import split_training_events
from operating_point import (
    build_event_trigger_scores,
    select_background_objects,
    select_truth_tau_objects,
)


def build_constraint_crossfit_rows(frame, seed):
    """Create two disjoint, stratified event folds for hard-metric evaluation."""
    first, second = split_training_events(
        frame,
        seed=seed,
        constraint_fraction=0.5,
    )
    return (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64))


def _calibration_for_rows(
    frame,
    scores,
    rows,
    classifier_config,
    objective_config,
    confidence_level,
):
    calibration_frame = frame.iloc[rows].copy().reset_index(drop=True)
    calibration_scores = np.asarray(scores, dtype=np.float64)[rows]
    background_mask = calibration_frame["Type"].isin(
        ["BKG", "Background"]
    ).to_numpy()
    background = select_background_objects(calibration_frame)
    classifier_calibration = calibrate_constraint_classifier(
        background,
        calibration_scores[background_mask],
        classifier_config,
        objective_config,
        confidence_level_override=confidence_level,
    )
    baseline_target_fpr = certified_calibration_target(
        int(background["eventNumber"].nunique()),
        objective_config.target_event_fpr,
        confidence_level,
    )
    baseline_threshold, _ = calibrate_tob_baseline(
        background,
        baseline_target_fpr,
        classifier_config.trigger_objects,
    )
    return classifier_calibration, baseline_threshold


def _safe_efficiency(pass_count, count):
    return None if not count else float(pass_count / count)


def _aggregate_cross_fitted_metrics(folds, objective_config):
    objective_counts = np.sum(
        [fold["objective_region_counts"] for fold in folds], axis=0
    ).astype(int)
    objective_pass = np.sum(
        [fold["objective_region_pass_counts"] for fold in folds], axis=0
    ).astype(int)
    objective_baseline_pass = np.sum(
        [fold["objective_baseline_pass_counts"] for fold in folds], axis=0
    ).astype(int)
    objective_efficiencies = [
        _safe_efficiency(passed, count)
        for passed, count in zip(objective_pass, objective_counts)
    ]
    objective_baselines = [
        _safe_efficiency(passed, count)
        for passed, count in zip(objective_baseline_pass, objective_counts)
    ]
    objective_deltas = [
        None if value is None else value - baseline
        for value, baseline in zip(objective_efficiencies, objective_baselines)
    ]
    valid_objective = np.asarray([value is not None for value in objective_deltas])
    weights = np.asarray(
        objective_config.objective_region_weights, dtype=np.float64
    ) * valid_objective
    weights = weights / weights.sum() if weights.sum() else weights
    objective_value = float(
        np.sum(
            weights
            * np.asarray(
                [0.0 if value is None else value for value in objective_deltas]
            )
        )
    )

    counts = np.sum([fold["region_counts"] for fold in folds], axis=0).astype(int)
    pass_counts = np.sum(
        [fold["region_pass_counts"] for fold in folds], axis=0
    ).astype(int)
    baseline_counts = np.sum(
        [fold["baseline_pass_counts"] for fold in folds], axis=0
    ).astype(int)
    reference_counts = np.sum(
        [fold["reference_pass_counts"] for fold in folds], axis=0
    ).astype(int)
    efficiencies = [
        _safe_efficiency(passed, count)
        for passed, count in zip(pass_counts, counts)
    ]
    baseline_efficiencies = [
        _safe_efficiency(passed, count)
        for passed, count in zip(baseline_counts, counts)
    ]
    reference_efficiencies = [
        _safe_efficiency(passed, count)
        for passed, count in zip(reference_counts, counts)
    ]

    required = []
    deltas = []
    margins = []
    for index, (efficiency, baseline) in enumerate(
        zip(efficiencies, baseline_efficiencies)
    ):
        if efficiency is None:
            required.append(None)
            deltas.append(None)
            margins.append(0.0)
            continue
        floor = baseline + objective_config.minimum_region_advantages[index]
        if objective_config.reference_model_allowed_deficits is not None:
            floor = max(
                floor,
                reference_efficiencies[index]
                - objective_config.reference_model_allowed_deficits[index],
            )
        floor = min(floor, 1.0)
        required.append(floor)
        deltas.append(efficiency - baseline)
        margins.append(efficiency - floor)

    background_count = int(sum(fold["background_event_count"] for fold in folds))
    background_pass = int(
        sum(fold["background_event_pass_count"] for fold in folds)
    )
    achieved_fpr = float(background_pass / background_count)
    baseline_statistics = [
        aggregate_paired_sufficient_statistics(
            [
                fold["paired_region_sufficient_statistics"][
                    "candidate_minus_baseline"
                ][index]
                for fold in folds
            ]
        )
        for index in range(len(objective_config.constraint_regions_gev))
    ]
    reference_statistics = [
        aggregate_paired_sufficient_statistics(
            [
                fold["paired_region_sufficient_statistics"][
                    "candidate_minus_reference"
                ][index]
                for fold in folds
            ]
        )
        for index in range(len(objective_config.constraint_regions_gev))
    ]
    feasibility = build_confidence_feasibility(
        objective_config,
        background_pass,
        background_count,
        baseline_statistics,
        reference_statistics,
        fpr_upper_override=(
            None
            if objective_config.feasibility_confidence_level is None
            else max(
                one_sided_binomial_upper_bound(
                    fold["background_event_pass_count"],
                    fold["background_event_count"],
                    1.0
                    - (1.0 - objective_config.feasibility_confidence_level)
                    / len(folds),
                )
                for fold in folds
            )
        ),
    )
    valid = np.asarray([value is not None for value in deltas])
    margins_array = np.asarray(margins, dtype=np.float64)
    resolutions = [None if not count else 1.0 / count for count in counts]
    return {
        "objective_value": objective_value,
        "achieved_fpr": achieved_fpr,
        "objective_region_efficiencies": objective_efficiencies,
        "objective_baseline_efficiencies": objective_baselines,
        "objective_region_deltas": objective_deltas,
        "objective_region_counts": objective_counts.tolist(),
        "objective_region_pass_counts": objective_pass.tolist(),
        "objective_baseline_pass_counts": objective_baseline_pass.tolist(),
        "region_efficiencies": efficiencies,
        "baseline_efficiencies": baseline_efficiencies,
        "reference_efficiencies": reference_efficiencies,
        "required_efficiencies": required,
        "region_deltas": deltas,
        "region_counts": counts.tolist(),
        "region_pass_counts": pass_counts.tolist(),
        "baseline_pass_counts": baseline_counts.tolist(),
        "reference_pass_counts": reference_counts.tolist(),
        "region_efficiency_resolutions": resolutions,
        "constraint_margins": margins_array.tolist(),
        "constraint_margins_in_objects": [
            None if resolution is None else float(margin / resolution)
            for margin, resolution in zip(margins_array, resolutions)
        ],
        "constraints_satisfied": feasibility["constraints_satisfied"],
        "minimum_margin": (
            float(np.min(margins_array[valid])) if np.any(valid) else None
        ),
        "minimum_certified_margin": feasibility["minimum_certified_margin"],
        "background_event_count": background_count,
        "background_event_pass_count": background_pass,
        "paired_region_sufficient_statistics": {
            "candidate_minus_baseline": baseline_statistics,
            "candidate_minus_reference": reference_statistics,
        },
        "feasibility": feasibility,
        "cross_fitted": True,
        "folds": folds,
    }


def calculate_cross_fitted_hard_metrics(
    frame,
    scores,
    reference_scores,
    classifier_config,
    objective_config,
    fold_rows,
):
    """Calibrate on one fold and measure exact constraints on the other."""
    scores = np.asarray(scores, dtype=np.float64)
    reference_scores = (
        None
        if reference_scores is None
        else np.asarray(reference_scores, dtype=np.float64)
    )
    fold_metrics = []
    fold_confidence_level = objective_config.feasibility_confidence_level
    if fold_confidence_level is not None:
        fold_confidence_level = 1.0 - (1.0 - fold_confidence_level) / 2.0
    for calibration_rows, measurement_rows in (
        (fold_rows[0], fold_rows[1]),
        (fold_rows[1], fold_rows[0]),
    ):
        calibration, baseline_threshold = _calibration_for_rows(
            frame,
            scores,
            calibration_rows,
            classifier_config,
            objective_config,
            fold_confidence_level,
        )
        reference_calibration = None
        if reference_scores is not None:
            reference_calibration, _ = _calibration_for_rows(
                frame,
                reference_scores,
                calibration_rows,
                classifier_config,
                objective_config,
                fold_confidence_level,
            )
        measurement_frame = frame.iloc[measurement_rows].copy().reset_index(drop=True)
        fold = calculate_hard_constraint_metrics(
            measurement_frame,
            scores[measurement_rows],
            classifier_config,
            objective_config,
            calibration=calibration,
            baseline_threshold_gev=baseline_threshold,
            reference_scores=(
                None
                if reference_scores is None
                else reference_scores[measurement_rows]
            ),
            reference_calibration=reference_calibration,
        )
        fold["calibration_row_count"] = int(len(calibration_rows))
        fold["measurement_row_count"] = int(len(measurement_rows))
        fold_metrics.append(fold)
    return _aggregate_cross_fitted_metrics(fold_metrics, objective_config)


def _logit(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(probabilities) - np.log1p(-probabilities)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def regional_gradient_diagnostics(
    frame,
    scores,
    classifier_config,
    objective_config,
    temperature,
    histogram_limit=10.0,
    histogram_bins=40,
    maximum_tail_samples=512,
    maximum_signal_samples=8192,
):
    """Summarize boundary coverage and tail-ranking gradients by energy region."""
    scored = frame.copy()
    scored["nn_score"] = np.asarray(scores, dtype=np.float64)
    background = select_background_objects(scored)
    signal = select_truth_tau_objects(scored)
    calibration = calibrate_classifier(
        background,
        background["nn_score"].to_numpy(dtype=np.float64),
        classifier_config,
    )
    threshold_logit = float(_logit([calibration["nn_threshold"]])[0])
    signal_logits = _logit(signal["nn_score"].to_numpy(dtype=np.float64))
    truth_pt = signal["truth_pt"].to_numpy(dtype=np.float64)
    finite = truth_pt[np.isfinite(truth_pt) & (truth_pt > 0.0)]
    if finite.size and np.median(finite) > 1000.0:
        truth_pt = truth_pt / 1000.0

    background = background.copy()
    background["_logit"] = _logit(background["nn_score"].to_numpy(dtype=np.float64))
    event_scores, background_event_count = build_event_trigger_scores(
        background,
        "_logit",
        objects=objective_config.trigger_objects,
    )
    tail_count = max(
        1,
        int(math.ceil(objective_config.tail_fraction * background_event_count)),
    )
    tail_count = min(tail_count, len(event_scores))
    tail = np.sort(event_scores)[-tail_count:]
    if len(tail) > maximum_tail_samples:
        indices = np.linspace(0, len(tail) - 1, maximum_tail_samples).astype(int)
        tail = tail[indices]

    edges = np.linspace(-histogram_limit, histogram_limit, histogram_bins + 1)
    regions = []
    for low, high in objective_config.constraint_regions_gev:
        mask = (truth_pt >= low) & (truth_pt < high)
        logits = signal_logits[mask]
        normalized = (logits - threshold_logit) / temperature
        histogram, _ = np.histogram(normalized, bins=edges)
        boundary_gradient = _sigmoid(normalized) * (1.0 - _sigmoid(normalized))
        boundary_gradient /= temperature

        tail_gradient_sum = 0.0
        tail_gradient_mean = None
        if len(logits) and len(tail):
            gradient_logits = logits
            if len(gradient_logits) > maximum_signal_samples:
                indices = np.linspace(
                    0, len(gradient_logits) - 1, maximum_signal_samples
                ).astype(int)
                gradient_logits = np.sort(gradient_logits)[indices]
            accumulated = 0.0
            for start in range(0, len(gradient_logits), 2048):
                differences = (
                    tail.reshape(1, -1)
                    - gradient_logits[start : start + 2048].reshape(-1, 1)
                ) / objective_config.tail_temperature
                accumulated += float(
                    (_sigmoid(differences) / objective_config.tail_temperature)
                    .mean(axis=1)
                    .sum()
                )
            tail_gradient_mean = accumulated / len(gradient_logits)
            tail_gradient_sum = tail_gradient_mean * len(logits)

        regions.append(
            {
                "low_gev": low,
                "high_gev": high,
                "signal_count": int(len(logits)),
                "normalized_margin_quantiles": (
                    None
                    if not len(normalized)
                    else {
                        name: float(value)
                        for name, value in zip(
                            ("q01", "q10", "q25", "q50", "q75", "q90", "q99"),
                            np.quantile(
                                normalized,
                                (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99),
                            ),
                        )
                    }
                ),
                "normalized_margin_histogram": {
                    "edges": edges.tolist(),
                    "counts": histogram.tolist(),
                    "below_range": int(np.count_nonzero(normalized < edges[0])),
                    "above_range": int(np.count_nonzero(normalized >= edges[-1])),
                },
                "fraction_within_one_temperature": (
                    None if not len(normalized) else float(np.mean(np.abs(normalized) <= 1.0))
                ),
                "fraction_within_three_temperatures": (
                    None if not len(normalized) else float(np.mean(np.abs(normalized) <= 3.0))
                ),
                "false_negative_fraction_below_minus_three": (
                    None if not len(normalized) else float(np.mean(normalized < -3.0))
                ),
                "boundary_surrogate_gradient_sum": float(boundary_gradient.sum()),
                "tail_ranking_gradient_sum": float(tail_gradient_sum),
                "tail_ranking_gradient_mean": tail_gradient_mean,
                "tail_ranking_gradient_signal_sample": int(
                    min(len(logits), maximum_signal_samples)
                ),
            }
        )
    return {
        "temperature": float(temperature),
        "nn_threshold": float(calibration["nn_threshold"]),
        "nn_threshold_logit": threshold_logit,
        "tail_fraction": objective_config.tail_fraction,
        "tail_event_count_full": int(tail_count),
        "tail_event_count_diagnostic": int(len(tail)),
        "regions": regions,
    }
