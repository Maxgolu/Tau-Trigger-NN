from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.features import FEATURE_REGISTRY


Extractor = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class VariableSpec:
    name: str
    level: str
    xlabel: str
    extractor: Extractor | None = None
    requires: frozenset[str] = frozenset()
    discrete: bool = False
    default_bins: int = 50
    default_range: tuple[float, float] | None = None


def _one_dimension(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"A distribution variable must be scalar, got shape {values.shape}")
    return values


def _registry_feature(name: str, scale: float = 1.0) -> Extractor:
    def extract(frame: pd.DataFrame) -> np.ndarray:
        return _one_dimension(FEATURE_REGISTRY[name](frame)).astype(np.float64) * scale

    return extract


def _tob_pt_gev(frame: pd.DataFrame) -> np.ndarray:
    return frame["tob_pt"].to_numpy(dtype=np.float64) / 1000.0


OBJECT_VARIABLES: dict[str, VariableSpec] = {
    "tob_pt": VariableSpec(
        "tob_pt", "object", "TOB $p_T$ [GeV]", _tob_pt_gev,
        default_bins=58, default_range=(5.0, 120.0),
    ),
    "em2_maxdist": VariableSpec(
        "em2_maxdist", "object", "EM2 top-cell squared distance [cell$^2$]",
        _registry_feature("em2_maxdist"), frozenset({"em2"}), discrete=True,
    ),
    "em2_3x3_maxdist": VariableSpec(
        "em2_3x3_maxdist", "object", "EM2 3x3 top-cell squared distance [cell$^2$]",
        _registry_feature("em2_3x3_maxdist"), frozenset({"tensors"}), discrete=True,
    ),
    "em2_3x3_dominance": VariableSpec(
        "em2_3x3_dominance", "object", "EM2 3x3 dominance [GeV]",
        _registry_feature("em2_3x3_dominance", scale=1.0 / 1000.0),
        frozenset({"tensors"}),
    ),
    "em2_3x3_normalized_dominance": VariableSpec(
        "em2_3x3_normalized_dominance", "object", "Normalized EM2 3x3 dominance",
        _registry_feature("em2_3x3_normalized_dominance"),
        frozenset({"tensors"}), default_range=(-1.0, 1.0),
    ),
    "em2_3x3_sum": VariableSpec(
        "em2_3x3_sum", "object", "EM2 3x3 energy sum [GeV]",
        _registry_feature("em2_3x3_sum", scale=1.0 / 1000.0),
        frozenset({"tensors"}),
    ),
    "em2_3x3_sum_over_tob_pt": VariableSpec(
        "em2_3x3_sum_over_tob_pt", "object", "EM2 3x3 energy sum / TOB $p_T$",
        _registry_feature("em2_3x3_sum_over_tob_pt"), frozenset({"tensors"}),
    ),
    "em2_3x3_sparsity": VariableSpec(
        "em2_3x3_sparsity", "object", "EM2 3x3 active-cell count",
        _registry_feature("em2_3x3_sparsity"), frozenset({"tensors"}),
        discrete=True, default_range=(0.0, 9.0),
    ),
    "em2_width": VariableSpec(
        "em2_width", "object", "Raw EM2 energy-weighted width [GeV cell$^2$]",
        _registry_feature("em2_width", scale=1.0 / 1000.0), frozenset({"em2"}),
    ),
    "em2_normalized_width": VariableSpec(
        "em2_normalized_width", "object", "Normalized EM2 width [cell$^2$]",
        _registry_feature("em2_normalized_width"),
        frozenset({"em2"}), default_range=(0.0, 50.0),
    ),
    "em2_best_3x3_fraction": VariableSpec(
        "em2_best_3x3_fraction", "object", "Best EM2 3x3 energy fraction",
        _registry_feature("em2_best_3x3_fraction"), frozenset({"em2"}),
        default_range=(0.0, 1.0),
    ),
}


EVENT_VARIABLES: dict[str, VariableSpec] = {
    "top2_tob_dr2": VariableSpec(
        "top2_tob_dr2", "event", "Top-two TOB $\\Delta R^2$", default_range=(0.0, 35.0)
    ),
    "top2_tob_dr": VariableSpec(
        "top2_tob_dr", "event", "Top-two TOB $\\Delta R$", default_range=(0.0, 6.0)
    ),
    "second_highest_tob_pt": VariableSpec(
        "second_highest_tob_pt", "event", "Second-highest TOB $p_T$ [GeV]",
        default_bins=58, default_range=(5.0, 120.0),
    ),
    "sum_event_tob_pt": VariableSpec(
        "sum_event_tob_pt", "event", "Event sum of TOB $p_T$ [GeV]",
        default_bins=60, default_range=(0.0, 500.0),
    ),
}


ALL_VARIABLES = {**OBJECT_VARIABLES, **EVENT_VARIABLES}


def required_sources(variable_names: set[str]) -> set[str]:
    unknown = variable_names.difference(OBJECT_VARIABLES)
    if unknown:
        raise KeyError(f"Unknown object distribution variables: {sorted(unknown)}")
    required: set[str] = set()
    for name in variable_names:
        required.update(OBJECT_VARIABLES[name].requires)
    return required


def extract_object_variables(frame: pd.DataFrame, variable_names: set[str]) -> pd.DataFrame:
    """Calculate requested scalar variables for one aligned object batch."""
    base_columns = [
        "event_uid",
        "original_event_number",
        "sample_type",
        "tob_index",
        "label",
        "event_tau_count",
        "split",
        "tob_eta",
        "tob_phi",
    ]
    result = frame[base_columns].copy()
    result["tob_pt_raw"] = frame["tob_pt"].to_numpy(dtype=np.float64)
    result["tob_pt_gev"] = frame["tob_pt"].to_numpy(dtype=np.float64) / 1000.0
    for name in sorted(variable_names):
        spec = OBJECT_VARIABLES[name]
        if spec.extractor is None:
            raise RuntimeError(f"Object variable '{name}' has no extractor")
        result[name] = _one_dimension(spec.extractor(frame))
    return result


def build_event_table(objects: pd.DataFrame) -> pd.DataFrame:
    """Aggregate object metadata into event-level physics variables."""
    required = {
        "event_uid", "sample_type", "label", "event_tau_count", "split",
        "tob_pt_gev", "tob_eta", "tob_phi",
    }
    missing = required.difference(objects.columns)
    if missing:
        raise KeyError(f"Cannot build event table; missing columns: {sorted(missing)}")

    ranked = objects.sort_values(
        ["event_uid", "tob_pt_gev", "tob_index"], ascending=[True, False, True]
    ).copy()
    ranked["pt_rank"] = ranked.groupby("event_uid", sort=False).cumcount()

    grouped = ranked.groupby("event_uid", sort=False)
    events = grouped.agg(
        sample_type=("sample_type", "first"),
        split=("split", "first"),
        event_tau_count=("event_tau_count", "first"),
        object_count=("tob_index", "size"),
        sum_event_tob_pt=("tob_pt_gev", "sum"),
    )

    first = ranked[ranked["pt_rank"].eq(0)].set_index("event_uid")
    second = ranked[ranked["pt_rank"].eq(1)].set_index("event_uid")
    events["second_highest_tob_pt"] = second["tob_pt_gev"]

    delta_eta = first["tob_eta"] - second["tob_eta"]
    delta_phi = (first["tob_phi"] - second["tob_phi"] + np.pi) % (2 * np.pi) - np.pi
    events["top2_tob_dr2"] = delta_eta**2 + delta_phi**2
    events["top2_tob_dr"] = np.sqrt(events["top2_tob_dr2"])
    events["tau_group"] = np.select(
        [events["event_tau_count"].eq(0), events["event_tau_count"].eq(1)],
        ["0 tau", "1 tau"],
        default="2+ tau",
    )
    return events.reset_index()
