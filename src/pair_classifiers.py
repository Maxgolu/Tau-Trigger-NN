"""Pair-level trigger decisions and event-level operating-point calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from .operating_point import select_fpr_threshold
except ImportError:
    from operating_point import select_fpr_threshold


@dataclass(frozen=True)
class JointOrThresholds:
    """Thresholds and accounting for the measured-pT plus neural OR."""

    pt_threshold: float
    neural_threshold: float
    background_events: int
    pt_accepted: int
    neural_accepted: int
    overlap_accepted: int
    union_accepted: int

    @property
    def union_fpr(self) -> float:
        return self.union_accepted / self.background_events


def pass_scores(scores, threshold):
    """Apply the finite ``score >= threshold`` rule used by calibration."""
    values = np.asarray(scores, dtype=np.float64)
    return np.isfinite(values) & (values >= np.float64(threshold))


def calibrate_joint_or(
    pt_event_scores,
    neural_event_scores,
    *,
    pt_budget,
    target_fpr=0.005,
):
    """Calibrate two OR branches under one event-level background budget.

    The measured-pT branch receives ``pt_budget``. The neural threshold then
    uses the remaining accepted-event allowance. Events accepted by both
    branches count once.
    """
    pt_scores = np.asarray(pt_event_scores, dtype=np.float64)
    neural_scores = np.asarray(neural_event_scores, dtype=np.float64)
    if pt_scores.shape != neural_scores.shape or pt_scores.ndim != 1:
        raise ValueError("pt and neural event scores must be one-dimensional and aligned")
    if len(pt_scores) == 0:
        raise ValueError("background event scores cannot be empty")
    if not 0.0 <= pt_budget <= target_fpr <= 1.0:
        raise ValueError("expected 0 <= pt_budget <= target_fpr <= 1")

    event_count = len(pt_scores)
    pt_threshold, _ = select_fpr_threshold(
        pt_scores,
        event_count,
        pt_budget,
    )
    pt_pass = pass_scores(pt_scores, pt_threshold)
    max_accepted = math.floor(target_fpr * event_count + 1e-12)
    remaining_allowance = max_accepted - int(pt_pass.sum())
    if remaining_allowance < 0:
        raise RuntimeError("measured-pT branch exceeded the total background budget")

    # Scores from events already accepted by pT cannot increase the union.
    remaining_neural = neural_scores[~pt_pass]
    neural_threshold, _ = select_fpr_threshold(
        remaining_neural,
        event_count,
        remaining_allowance / event_count,
    )
    neural_pass = pass_scores(neural_scores, neural_threshold)
    union_pass = pt_pass | neural_pass
    if int(union_pass.sum()) > max_accepted:
        raise RuntimeError("joint OR calibration exceeded the event-level budget")

    return JointOrThresholds(
        pt_threshold=float(pt_threshold),
        neural_threshold=float(neural_threshold),
        background_events=event_count,
        pt_accepted=int(pt_pass.sum()),
        neural_accepted=int(neural_pass.sum()),
        overlap_accepted=int((pt_pass & neural_pass).sum()),
        union_accepted=int(union_pass.sum()),
    )


def joint_or_decisions(pt_event_scores, neural_event_scores, thresholds):
    """Return the event decision for a calibrated two-branch OR."""
    pt_scores = np.asarray(pt_event_scores, dtype=np.float64)
    neural_scores = np.asarray(neural_event_scores, dtype=np.float64)
    if pt_scores.shape != neural_scores.shape:
        raise ValueError("pt and neural event scores must be aligned")
    return pass_scores(pt_scores, thresholds.pt_threshold) | pass_scores(
        neural_scores,
        thresholds.neural_threshold,
    )
