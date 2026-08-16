"""Configurable trigger classifiers built from network and TOB decisions."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from operating_point import (
    build_event_trigger_scores,
    score_pass_mask,
    select_fpr_threshold,
)


VALID_CLASSIFIERS = ("nn_only", "tob_nn_or")
VALID_NONINFERIORITY_MODES = ("per_window", "pooled_saturation")


@dataclass(frozen=True)
class TobBudgetObjectiveConfig:
    min_truth_pt_gev: float
    objective_max_truth_pt_gev: float
    window_width_gev: float
    protected_max_truth_pt_gev: float
    noninferiority_mode: str
    saturation_start_truth_pt_gev: float
    noninferiority_tolerance: float
    objective_tie_tolerance: float


@dataclass(frozen=True)
class TobBudgetSearchConfig:
    mode: str
    values: tuple[float, ...]
    cross_validation_folds: int
    objective: TobBudgetObjectiveConfig


@dataclass(frozen=True)
class ClassifierConfig:
    name: str
    target_fpr: float
    trigger_objects: int
    tob_fpr: float | None
    composition: str
    tob_budget: TobBudgetSearchConfig | None = None

    def with_target_fpr(self, target_fpr):
        """Return the same classifier evaluated at another total FPR."""
        return replace(self, target_fpr=float(target_fpr))

    def with_tob_fpr(self, tob_fpr):
        """Return a concrete OR classifier for one TOB budget."""
        return replace(self, tob_fpr=float(tob_fpr))

    def to_dict(self):
        return asdict(self)


def parse_classifier(config):
    """Validate classifier settings while preserving the legacy NN default."""
    raw = config.get("classifier", {})
    name = raw.get("name", "nn_only")
    if name not in VALID_CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {name}")

    target_fpr = float(raw.get("target_fpr", 0.005))
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("classifier.target_fpr must be in the interval (0, 1]")

    trigger_objects = int(raw.get("trigger_objects", 2))
    if trigger_objects < 1:
        raise ValueError("classifier.trigger_objects must be at least 1")

    composition = raw.get("composition", "object_or")
    if composition != "object_or":
        raise ValueError("Only classifier.composition='object_or' is supported")

    tob_fpr = raw.get("tob_fpr")
    tob_budget = None
    if name == "tob_nn_or":
        raw_budget = raw.get("tob_budget")
        if raw_budget and raw_budget.get("mode") == "validation_search":
            values = tuple(float(value) for value in raw_budget.get("values", ()))
            if not values:
                raise ValueError("classifier.tob_budget.values cannot be empty")
            if len(set(values)) != len(values):
                raise ValueError("classifier.tob_budget.values cannot contain duplicates")
            if any(value < 0.0 or value > target_fpr for value in values):
                raise ValueError(
                    "Every TOB budget must be between zero and target_fpr"
                )
            folds = int(raw_budget.get("cross_validation_folds", 2))
            if folds != 2:
                raise ValueError("TOB budget search currently requires exactly 2 folds")
            raw_objective = raw_budget.get("objective", {})
            noninferiority_mode = raw_objective.get(
                "noninferiority_mode", "pooled_saturation"
            )
            if noninferiority_mode not in VALID_NONINFERIORITY_MODES:
                raise ValueError(
                    "Unknown classifier noninferiority mode: "
                    f"{noninferiority_mode}"
                )
            objective = TobBudgetObjectiveConfig(
                min_truth_pt_gev=float(raw_objective.get("min_truth_pt_gev", 25.0)),
                objective_max_truth_pt_gev=float(
                    raw_objective.get("objective_max_truth_pt_gev", 100.0)
                ),
                window_width_gev=float(raw_objective.get("window_width_gev", 5.0)),
                protected_max_truth_pt_gev=float(
                    raw_objective.get("protected_max_truth_pt_gev", 120.0)
                ),
                noninferiority_mode=noninferiority_mode,
                saturation_start_truth_pt_gev=float(
                    raw_objective.get("saturation_start_truth_pt_gev", 60.0)
                ),
                noninferiority_tolerance=float(
                    raw_objective.get("noninferiority_tolerance", 0.005)
                ),
                objective_tie_tolerance=float(
                    raw_objective.get("objective_tie_tolerance", 0.002)
                ),
            )
            if not (
                objective.min_truth_pt_gev < objective.objective_max_truth_pt_gev
                <= objective.protected_max_truth_pt_gev
                and objective.min_truth_pt_gev
                < objective.saturation_start_truth_pt_gev
                <= objective.protected_max_truth_pt_gev
                and objective.window_width_gev > 0.0
                and objective.noninferiority_tolerance >= 0.0
                and objective.objective_tie_tolerance >= 0.0
            ):
                raise ValueError("Invalid classifier.tob_budget.objective settings")
            tob_budget = TobBudgetSearchConfig(
                mode="validation_search",
                values=tuple(sorted(values)),
                cross_validation_folds=folds,
                objective=objective,
            )
            tob_fpr = None
        else:
            if raw_budget and raw_budget.get("mode", "fixed") != "fixed":
                raise ValueError("Unknown classifier.tob_budget.mode")
            tob_fpr = float(0.004 if tob_fpr is None else tob_fpr)
            if not 0.0 <= tob_fpr <= target_fpr:
                raise ValueError(
                    "classifier.tob_fpr must be between zero and target_fpr"
                )
    else:
        tob_fpr = None

    return ClassifierConfig(
        name=name,
        target_fpr=target_fpr,
        trigger_objects=trigger_objects,
        tob_fpr=tob_fpr,
        composition=composition,
        tob_budget=tob_budget,
    )


def tob_pt_gev(frame):
    """Return observable TOB pT in GeV for either supported storage unit."""
    values = frame["tob_pt"].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1000.0:
        values = values / 1000.0
    return values


def _frame_with_scores(frame, scores, score_column):
    result = frame.copy()
    result[score_column] = np.asarray(scores, dtype=np.float64)
    if len(result) != len(scores):
        raise ValueError("Classifier scores must align one-to-one with frame rows")
    return result


def _select_threshold_with_base(
    activation_scores,
    event_count,
    base_pass_count,
    target_fpr,
):
    """Select a tie-safe threshold after fixed branch passes are counted."""
    max_accepted = int(np.floor(target_fpr * event_count + 1e-12))
    remaining = max_accepted - int(base_pass_count)
    if remaining < 0:
        raise ValueError("The fixed TOB branch already exceeds target_fpr")

    scores = np.asarray(activation_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.inf, float(base_pass_count / event_count)

    unique_scores, tied_counts = np.unique(scores, return_counts=True)
    unique_scores = unique_scores[::-1]
    tied_counts = tied_counts[::-1]
    cumulative = np.cumsum(tied_counts)
    feasible = np.flatnonzero(cumulative <= remaining)
    if feasible.size == 0:
        threshold = np.nextafter(unique_scores[0], np.inf)
    else:
        threshold = unique_scores[int(feasible[-1])]

    accepted = base_pass_count + np.count_nonzero(scores >= threshold)
    return float(threshold), float(accepted / event_count)


def build_or_event_activation_scores(
    background,
    score_column,
    tob_threshold,
    trigger_objects=2,
):
    """Return NN thresholds at which remaining background events pass OR."""
    required = {"eventNumber", "tob_pt", score_column}
    missing = required.difference(background.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame = background[["eventNumber", "tob_pt", score_column]].copy()
    frame["_tob_pt_gev"] = tob_pt_gev(frame)
    frame["_tob_pass"] = (
        np.isfinite(frame["_tob_pt_gev"])
        & (frame["_tob_pt_gev"] >= np.float64(tob_threshold))
    )

    event_count = int(frame["eventNumber"].nunique())
    tob_counts = frame.groupby("eventNumber", sort=False)["_tob_pass"].sum()
    base_pass_count = int(np.count_nonzero(tob_counts >= trigger_objects))
    needed = (trigger_objects - tob_counts).clip(lower=1)
    eligible = tob_counts < trigger_objects

    candidates = frame.loc[
        ~frame["_tob_pass"]
        & frame["eventNumber"].map(eligible).fillna(False)
        & np.isfinite(frame[score_column].to_numpy(dtype=np.float64)),
        ["eventNumber", score_column],
    ].sort_values(
        ["eventNumber", score_column],
        ascending=[True, False],
        kind="mergesort",
    )
    candidates["_rank"] = candidates.groupby(
        "eventNumber", sort=False
    ).cumcount()
    candidates["_needed"] = (
        candidates["eventNumber"].map(needed).to_numpy(dtype=int)
    )
    activation = candidates.loc[
        candidates["_rank"] == candidates["_needed"] - 1,
        score_column,
    ].to_numpy(dtype=np.float64)
    return activation, event_count, base_pass_count


def classifier_object_pass_mask(frame, calibration, score_column="nn_score"):
    """Apply the configured object-level decision with shared >= semantics."""
    nn_pass = score_pass_mask(
        frame,
        score_column,
        calibration["nn_threshold"],
    )
    if calibration["name"] == "nn_only":
        return nn_pass
    tob_values = tob_pt_gev(frame)
    tob_pass = np.isfinite(tob_values) & (
        tob_values >= np.float64(calibration["tob_threshold_gev"])
    )
    return tob_pass | nn_pass


def classifier_event_pass_mask(
    frame,
    calibration,
    score_column="nn_score",
):
    """Return one Boolean pass decision per event."""
    passed = classifier_object_pass_mask(frame, calibration, score_column)
    counts = pd.Series(passed, index=frame.index).groupby(
        frame["eventNumber"], sort=False
    ).sum()
    return counts >= int(calibration["trigger_objects"])


def _branch_diagnostics(background, calibration, score_column):
    event_count = int(background["eventNumber"].nunique())
    nn_calibration = {
        "name": "nn_only",
        "nn_threshold": calibration["nn_threshold"],
        "trigger_objects": calibration["trigger_objects"],
    }
    nn_event = classifier_event_pass_mask(
        background, nn_calibration, score_column
    )
    if calibration["name"] == "nn_only":
        return {
            "nn_event_fpr": float(nn_event.mean()),
            "combined_event_fpr": float(nn_event.mean()),
            "background_event_count": event_count,
        }

    tob_values = tob_pt_gev(background)
    tob_pass = np.isfinite(tob_values) & (
        tob_values >= np.float64(calibration["tob_threshold_gev"])
    )
    tob_counts = pd.Series(tob_pass, index=background.index).groupby(
        background["eventNumber"], sort=False
    ).sum()
    tob_event = tob_counts >= int(calibration["trigger_objects"])
    combined_event = classifier_event_pass_mask(
        background, calibration, score_column
    )
    aligned = pd.concat(
        [
            tob_event.rename("tob"),
            nn_event.rename("nn"),
            combined_event.rename("combined"),
        ],
        axis=1,
    ).fillna(False)
    return {
        "tob_event_fpr": float(aligned["tob"].mean()),
        "nn_event_fpr": float(aligned["nn"].mean()),
        "combined_event_fpr": float(aligned["combined"].mean()),
        "event_branch_overlap": float(
            (aligned["tob"] & aligned["nn"]).mean()
        ),
        "mixed_only_event_fraction": float(
            (
                aligned["combined"]
                & ~aligned["tob"]
                & ~aligned["nn"]
            ).mean()
        ),
        "background_event_count": event_count,
    }


def calibrate_classifier(
    background,
    scores,
    classifier,
    score_column="nn_score",
):
    """Calibrate all thresholds on background events without splitting ties."""
    frame = _frame_with_scores(background, scores, score_column)
    if classifier.name == "nn_only":
        event_scores, event_count = build_event_trigger_scores(
            frame,
            score_column,
            objects=classifier.trigger_objects,
        )
        nn_threshold, achieved_fpr = select_fpr_threshold(
            event_scores,
            event_count,
            classifier.target_fpr,
        )
        calibration = {
            "name": classifier.name,
            "target_fpr": classifier.target_fpr,
            "trigger_objects": classifier.trigger_objects,
            "nn_threshold": nn_threshold,
            "achieved_fpr": achieved_fpr,
        }
    else:
        if classifier.tob_fpr is None:
            raise ValueError(
                "A concrete tob_fpr is required before classifier calibration"
            )
        tob_frame = frame.copy()
        tob_frame["_tob_pt_gev"] = tob_pt_gev(tob_frame)
        tob_scores, event_count = build_event_trigger_scores(
            tob_frame,
            "_tob_pt_gev",
            objects=classifier.trigger_objects,
        )
        tob_threshold, tob_achieved_fpr = select_fpr_threshold(
            tob_scores,
            event_count,
            classifier.tob_fpr,
        )
        activation, _, base_pass_count = build_or_event_activation_scores(
            frame,
            score_column,
            tob_threshold,
            trigger_objects=classifier.trigger_objects,
        )
        nn_threshold, achieved_fpr = _select_threshold_with_base(
            activation,
            event_count,
            base_pass_count,
            classifier.target_fpr,
        )
        calibration = {
            "name": classifier.name,
            "composition": classifier.composition,
            "target_fpr": classifier.target_fpr,
            "tob_fpr_budget": classifier.tob_fpr,
            "trigger_objects": classifier.trigger_objects,
            "tob_threshold_gev": tob_threshold,
            "nn_threshold": nn_threshold,
            "tob_achieved_fpr": tob_achieved_fpr,
            "achieved_fpr": achieved_fpr,
        }

    calibration["diagnostics"] = _branch_diagnostics(
        frame, calibration, score_column
    )
    return calibration
