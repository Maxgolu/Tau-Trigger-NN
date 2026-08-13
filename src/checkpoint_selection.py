"""Configurable validation checkpoint selection for TauNet training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from classifiers import (
    classifier_object_pass_mask,
    calibrate_classifier,
    parse_classifier,
)
from operating_point import (
    select_background_objects,
    select_truth_tau_objects,
)


VALID_METHODS = ("validation_bce", "target_fpr")
DEFAULT_ENERGY_BANDS_GEV = ((10.0, 20.0), (20.0, 40.0), (40.0, 80.0), (80.0, 120.0))


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    methods: tuple[str, ...]
    primary_method: str
    target_fpr: float
    trigger_objects: int
    energy_bands_gev: tuple[tuple[float, float], ...]


def parse_checkpoint_selection(config):
    """Validate checkpoint settings while preserving the legacy default."""
    raw = config.get("checkpoint_selection", {})
    methods = tuple(raw.get("methods", ["validation_bce"]))
    if not methods:
        raise ValueError("checkpoint_selection.methods cannot be empty")
    if len(set(methods)) != len(methods):
        raise ValueError("checkpoint_selection.methods cannot contain duplicates")
    unknown = set(methods).difference(VALID_METHODS)
    if unknown:
        raise ValueError(f"Unknown checkpoint method(s): {sorted(unknown)}")

    primary = raw.get("primary_method", methods[0])
    if primary not in methods:
        raise ValueError("primary_method must be included in methods")

    target_fpr = float(raw.get("target_fpr", 0.005))
    if not 0.0 < target_fpr <= 1.0:
        raise ValueError("target_fpr must be in the interval (0, 1]")

    trigger_objects = int(raw.get("trigger_objects", 2))
    if trigger_objects < 1:
        raise ValueError("trigger_objects must be at least 1")

    bands = tuple(
        (float(bounds[0]), float(bounds[1]))
        for bounds in raw.get("energy_bands_gev", DEFAULT_ENERGY_BANDS_GEV)
    )
    if any(low >= high for low, high in bands):
        raise ValueError("Every energy band must satisfy low < high")

    return CheckpointSelectionConfig(
        methods=methods,
        primary_method=primary,
        target_fpr=target_fpr,
        trigger_objects=trigger_objects,
        energy_bands_gev=bands,
    )


def _truth_pt_gev(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1000.0:
        return values / 1000.0
    return values


def calculate_validation_operating_point(
    validation_frame,
    scores,
    target_fpr=0.005,
    trigger_objects=2,
    energy_bands_gev=DEFAULT_ENERGY_BANDS_GEV,
    classifier_config=None,
):
    """Calibrate validation FPR and measure truth-tau efficiency."""
    frame = validation_frame.copy()
    frame["nn_score"] = np.asarray(scores, dtype=np.float64)
    background = select_background_objects(frame)
    signal = select_truth_tau_objects(frame)

    if classifier_config is None:
        classifier_config = parse_classifier(
            {
                "classifier": {
                    "name": "nn_only",
                    "target_fpr": target_fpr,
                    "trigger_objects": trigger_objects,
                }
            }
        )
    else:
        classifier_config = classifier_config.with_target_fpr(target_fpr)

    calibration = calibrate_classifier(
        background,
        background["nn_score"].to_numpy(dtype=np.float64),
        classifier_config,
    )
    passed = classifier_object_pass_mask(signal, calibration)
    global_efficiency = float(passed.mean()) if len(signal) else 0.0

    truth_pt = _truth_pt_gev(signal["truth_pt"].to_numpy())
    energy_efficiencies = {}
    for low, high in energy_bands_gev:
        in_band = (truth_pt >= low) & (truth_pt < high)
        key = f"{low:g}-{high:g}"
        energy_efficiencies[key] = (
            float(passed[in_band].mean()) if np.any(in_band) else None
        )

    return {
        # threshold remains for readers of legacy NN-only manifests.
        "threshold": float(calibration["nn_threshold"]),
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(calibration["achieved_fpr"]),
        "signal_efficiency": global_efficiency,
        "energy_band_efficiencies": energy_efficiencies,
        "background_event_count": int(
            calibration["diagnostics"]["background_event_count"]
        ),
        "signal_object_count": int(len(signal)),
        "trigger_objects": int(classifier_config.trigger_objects),
        "classifier_calibration": calibration,
    }


def is_better_checkpoint(method, candidate, best):
    """Compare checkpoints only within the same validation objective."""
    if best is None:
        return True
    if method == "validation_bce":
        return candidate["validation_bce"] < best["validation_bce"]
    if method == "target_fpr":
        candidate_key = (
            candidate["signal_efficiency"],
            candidate["achieved_fpr"],
            -candidate["validation_bce"],
        )
        best_key = (
            best["signal_efficiency"],
            best["achieved_fpr"],
            -best["validation_bce"],
        )
        return candidate_key > best_key
    raise ValueError(f"Unknown checkpoint method: {method}")
