from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.distributions.variables import VariableSpec


GROUP_COLORS = {
    "Tau objects": "#d62728",
    "Noise objects": "#1f77b4",
    "Tau objects | 1 tau event": "#ff7f0e",
    "Tau objects | 2+ tau event": "#d62728",
    "Noise objects | 0 tau event": "#1f77b4",
    "Noise objects | 1 tau event": "#2ca02c",
    "Noise objects | 2+ tau event": "#9467bd",
    "0 tau": "#1f77b4",
    "1 tau": "#ff7f0e",
    "2+ tau": "#d62728",
}


def _pyplot(show_plots: bool):
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def grouped_values(
    frame: pd.DataFrame, variable: str, group_by: str
) -> dict[str, np.ndarray]:
    """Create the approved physics groups for a configured plot."""
    if group_by == "object_label":
        masks = {
            "Tau objects": frame["label"].eq(1),
            "Noise objects": frame["label"].eq(0),
        }
    elif group_by == "object_label_and_event_tau_count":
        masks = {
            "Tau objects | 1 tau event": frame["label"].eq(1)
            & frame["event_tau_count"].eq(1),
            "Tau objects | 2+ tau event": frame["label"].eq(1)
            & frame["event_tau_count"].ge(2),
            "Noise objects | 0 tau event": frame["label"].eq(0)
            & frame["event_tau_count"].eq(0),
            "Noise objects | 1 tau event": frame["label"].eq(0)
            & frame["event_tau_count"].eq(1),
            "Noise objects | 2+ tau event": frame["label"].eq(0)
            & frame["event_tau_count"].ge(2),
        }
    elif group_by == "event_tau_count":
        masks = {
            "0 tau": frame["event_tau_count"].eq(0),
            "1 tau": frame["event_tau_count"].eq(1),
            "2+ tau": frame["event_tau_count"].ge(2),
        }
    else:
        raise ValueError(f"Unknown grouping '{group_by}'")

    groups: dict[str, np.ndarray] = {}
    for label, mask in masks.items():
        values = frame.loc[mask, variable].to_numpy(dtype=np.float64)
        groups[label] = values[np.isfinite(values)]
    return groups


def _plot_limits(
    groups: dict[str, np.ndarray],
    spec: VariableSpec,
    options: dict,
) -> tuple[float, float]:
    configured_range = options.get("range")
    if configured_range is not None:
        low, high = map(float, configured_range)
    elif spec.default_range is not None:
        low, high = spec.default_range
    else:
        pooled = np.concatenate([values for values in groups.values() if values.size])
        if pooled.size == 0:
            raise ValueError(f"No finite values are available for '{spec.name}'")
        quantiles = options.get("clip_quantiles", [0.001, 0.999])
        low, high = np.quantile(pooled, quantiles)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(f"Invalid plot range ({low}, {high}) for '{spec.name}'")
    return float(low), float(high)


def _bin_edges(
    groups: dict[str, np.ndarray], spec: VariableSpec, options: dict
) -> np.ndarray:
    low, high = _plot_limits(groups, spec, options)
    if spec.discrete:
        return np.arange(math.floor(low) - 0.5, math.ceil(high) + 1.5, 1.0)

    bin_count = int(options.get("bins", spec.default_bins))
    if bin_count <= 0:
        raise ValueError("Histogram bin count must be positive")
    if options.get("log_x", False):
        positive_values = np.concatenate(
            [values[values > 0] for values in groups.values() if values.size]
        )
        if positive_values.size == 0:
            raise ValueError(f"Logarithmic plot '{spec.name}' has no positive values")
        low = max(low, float(positive_values.min()))
        return np.geomspace(low, high, bin_count + 1)
    return np.linspace(low, high, bin_count + 1)


def _draw_histograms(ax, groups, bins, normalization: str, options: dict) -> dict:
    if normalization not in {"probability", "density", "count"}:
        raise ValueError("normalization must be probability, density, or count")

    summary: dict[str, dict] = {}
    for label, all_values in groups.items():
        visible = all_values[(all_values >= bins[0]) & (all_values <= bins[-1])]
        if visible.size == 0:
            summary[label] = {
                "finite_count": int(all_values.size),
                "visible_count": 0,
                "excluded_by_range": int(all_values.size),
            }
            continue

        weights = None
        density = normalization == "density"
        if normalization == "probability":
            weights = np.full(visible.size, 1.0 / visible.size)
        ax.hist(
            visible,
            bins=bins,
            weights=weights,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=f"{label} (N={all_values.size:,})",
            color=GROUP_COLORS.get(label),
        )
        summary[label] = {
            "finite_count": int(all_values.size),
            "visible_count": int(visible.size),
            "excluded_by_range": int(all_values.size - visible.size),
        }
    return summary


def plot_distribution(
    frame: pd.DataFrame,
    spec: VariableSpec,
    group_by: str,
    output_path: str | Path,
    normalization: str,
    show_plots: bool,
    options: dict | None = None,
) -> dict:
    options = options or {}
    groups = grouped_values(frame, spec.name, group_by)
    bins = _bin_edges(groups, spec, options)
    plt = _pyplot(show_plots)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    group_summary = _draw_histograms(ax, groups, bins, normalization, options)

    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(
        {"probability": "Probability per bin", "density": "Probability density", "count": "Count"}[
            normalization
        ]
    )
    ax.set_title(options.get("title", f"Distribution of {spec.name}"))
    if options.get("log_x", False):
        ax.set_xscale("log")
    if options.get("log_y", False):
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)
    return {
        "bin_edges": bins.tolist(),
        "groups": group_summary,
        "output": str(output_path),
    }


def plot_pt_conditioned(
    objects: pd.DataFrame,
    spec: VariableSpec,
    pt_bins_gev: list[float],
    output_path: str | Path,
    normalization: str,
    show_plots: bool,
    options: dict | None = None,
) -> dict:
    """Plot tau/noise shapes within common TOB-pT intervals."""
    options = options or {}
    if len(pt_bins_gev) < 2 or any(
        upper <= lower for lower, upper in zip(pt_bins_gev[:-1], pt_bins_gev[1:])
    ):
        raise ValueError("pt_bins_gev must contain increasing bin boundaries")

    plt = _pyplot(show_plots)
    interval_count = len(pt_bins_gev) - 1
    columns = min(3, interval_count)
    rows = math.ceil(interval_count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.7 * columns, 4.5 * rows), squeeze=False)
    summaries: dict[str, dict] = {}
    common_groups = grouped_values(objects, spec.name, "object_label")
    common_bins = _bin_edges(common_groups, spec, options)

    for index, (low_pt, high_pt) in enumerate(zip(pt_bins_gev[:-1], pt_bins_gev[1:])):
        ax = axes.flat[index]
        subset = objects[
            objects["tob_pt_gev"].ge(low_pt) & objects["tob_pt_gev"].lt(high_pt)
        ]
        groups = grouped_values(subset, spec.name, "object_label")
        if not any(values.size for values in groups.values()):
            ax.set_visible(False)
            continue
        summaries[f"{low_pt:g}-{high_pt:g} GeV"] = {
            "bin_edges": common_bins.tolist(),
            "groups": _draw_histograms(ax, groups, common_bins, normalization, options),
        }
        ax.set_title(f"{low_pt:g} <= TOB $p_T$ < {high_pt:g} GeV")
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel("Probability per bin" if normalization == "probability" else normalization)
        if options.get("log_x", False):
            ax.set_xscale("log")
        if options.get("log_y", False):
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for index in range(interval_count, rows * columns):
        axes.flat[index].set_visible(False)
    fig.suptitle(options.get("conditioned_title", f"{spec.name}, conditioned on TOB $p_T$"))
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(fig)
    return {"pt_intervals": summaries, "output": str(output_path)}
