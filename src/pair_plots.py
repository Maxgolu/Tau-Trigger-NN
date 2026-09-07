"""Regenerate compact figures from the pair-model result summaries.

The plotting code reads only the small CSV summaries under ``results``. It does
not require raw detector data, checkpoints, or saved model predictions.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _read_rows(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    """Read a CSV and fail clearly if a required field is absent."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = set(required) - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Plot generation requires matplotlib; install the project requirements."
        ) from exc
    return plt


def _finish(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    return output


def plot_training_curves(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pt_control" / "training_curves.csv",
        ("seed", "epoch", "training_bce"),
    )
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = [row for row in rows if int(row["seed"]) == seed]
        ax.plot(
            [int(row["epoch"]) for row in selected],
            [float(row["training_bce"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"seed {seed}",
        )
    ax.set(
        title="Training loss decreased for all three pT-only seeds",
        xlabel="Epoch",
        ylabel="Training BCE",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _finish(fig, output_dir / "pt_control_training_curves.png")


def plot_checkpoint_selection(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pt_control" / "checkpoint_selection.csv",
        ("seed", "epoch", "eligible_accepted", "eligible_count", "selected"),
    )
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = [row for row in rows if int(row["seed"]) == seed]
        epochs = [int(row["epoch"]) for row in selected]
        efficiencies = [
            int(row["eligible_accepted"]) / int(row["eligible_count"]) for row in selected
        ]
        line = ax.plot(epochs, efficiencies, linewidth=1.8, label=f"seed {seed}")[0]
        winner = next(row for row in selected if row["selected"].lower() == "true")
        ax.scatter(
            int(winner["epoch"]),
            int(winner["eligible_accepted"]) / int(winner["eligible_count"]),
            s=75,
            color=line.get_color(),
            edgecolor="black",
            zorder=3,
        )
    ax.set(
        title="Validation selected a different checkpoint for each seed",
        xlabel="Checkpoint epoch",
        ylabel="Cross-fitted pair-eligible event efficiency",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _finish(fig, output_dir / "pt_control_checkpoint_selection.png")


def plot_pt_validation(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pt_control" / "validation_summary.csv",
        ("method", "seed", "inclusive_signal_efficiency", "background_fpr"),
    )
    baseline = next(row for row in rows if row["method"].startswith("deterministic"))
    models = [row for row in rows if row["method"] == "pt_only_pair_model"]
    labels = ["Baseline"] + [f"Pair NN\nseed {row['seed']}" for row in models]
    values = [float(baseline["inclusive_signal_efficiency"])] + [
        float(row["inclusive_signal_efficiency"]) for row in models
    ]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#666666"] + ["#2878b5"] * len(models)
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=3)
    ax.set(
        title="The pT-only control did not beat the deterministic baseline",
        ylabel="Inclusive signal-sample event efficiency",
        ylim=(0, max(values) * 1.2),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "pt_control_validation_summary.png")


def plot_feature_ranking(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "training_feature_analysis" / "feature_ranking.csv",
        ("display_label", "oriented_auc"),
    )
    rows.sort(key=lambda row: float(row["oriented_auc"]))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    values = [float(row["oriented_auc"]) for row in rows]
    bars = ax.barh([row["display_label"] for row in rows], values, color="#2878b5")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
    ax.axvline(0.5, color="#777777", linewidth=1, linestyle="--")
    ax.set(
        title="Training data highlighted several useful information sources",
        xlabel="Single-feature oriented AUC (training only)",
        xlim=(0.5, 1.0),
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "training_feature_ranking.png")


def plot_redundancy(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "training_feature_analysis" / "redundant_features.csv",
        ("feature_a", "feature_b", "spearman", "relationship"),
    )
    labels = [f"{row['feature_a']}\n↔ {row['feature_b']}" for row in rows]
    values = [abs(float(row["spearman"])) for row in rows]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.barh(labels, values, color="#ef8a17")
    ax.bar_label(bars, labels=[row["relationship"] for row in rows], padding=4)
    ax.set(
        title="Several candidate inputs carried the same information",
        xlabel="Absolute Spearman correlation (training only)",
        xlim=(0.98, 1.005),
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "training_feature_redundancy.png")


def _candidate_label(name: str) -> str:
    labels = {
        "deterministic_pt_baseline": "Deterministic\nbaseline",
        "pt_only_pair_control": "pT-only\npair control",
        "raw_coarse_cells_plus_member_pt": "Raw coarse cells\n+ member pT",
        "normalized_em2_summaries_plus_member_pt_and_event_context": (
            "Normalized EM2 summaries\n+ pT + event context"
        ),
        "raw_em2_summaries_plus_member_pt_and_event_context": (
            "Raw EM2 summaries\n+ pT + event context"
        ),
        "binary_ordered": "Binary\nordered",
        "three_class_ordered": "Three-class\nordered",
        "three_class_symmetric": "Three-class\nsymmetric",
        "shared_high_resolution": "High-resolution\nmodel",
        "inverse_frequency_weighting": "Inverse-frequency\nweighting",
        "power_law_p_minus_1": "Power-law\nweighting",
        "raw_cell_plus_pt_or": "Pair network\nOR pT",
        "raw_coarse_cells_without_member_pt": "Raw coarse cells\nwithout member pT",
        "raw_coarse_cells_with_member_pt": "Raw coarse cells\nwith member pT",
        "compact_em2_measurements_with_member_pt": (
            "Compact EM2 measurements\nwith member pT"
        ),
    }
    return labels.get(name, name.replace("_", " "))


def plot_representation_validation(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "representation_study" / "representation_validation.csv",
        ("study", "candidate", "mean_inclusive_efficiency"),
    )
    rows = [row for row in rows if row["study"] == "representation"]
    values = [float(row["mean_inclusive_efficiency"]) for row in rows]
    errors = [float(row["sd_inclusive_efficiency"]) for row in rows]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    colors = ["#666666", "#8f9aa6"] + ["#2878b5"] * max(0, len(rows) - 2)
    bars = ax.bar(
        [_candidate_label(row["candidate"]) for row in rows],
        values,
        yerr=errors,
        capsize=4,
        color=colors,
    )
    ax.bar_label(bars, labels=[f"{100 * value:.1f}%" for value in values], padding=5)
    ax.set(
        title="Adding calorimeter information improved validation efficiency",
        ylabel="Inclusive signal-sample event efficiency",
        ylim=(0, max(values) * 1.22),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "representation_validation.png")


def plot_gain_sources(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "representation_study" / "gain_source_effects.csv",
        ("effect", "mean_validation_point_difference"),
    )
    labels = {
        "event_context_given_member_pt": "Add event context\nto member pT",
        "shower_summaries_given_member_pt": "Add shower summaries\nto member pT",
        "additional_combined_difference": "Additional difference\nwhen combined",
    }
    values = [100 * float(row["mean_validation_point_difference"]) for row in rows]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    bars = ax.bar(
        [labels.get(row["effect"], row["effect"]) for row in rows],
        values,
        color=["#4daf4a", "#2878b5", "#8f6bb3"],
    )
    ax.bar_label(bars, labels=[f"+{value:.1f} points" for value in values], padding=4)
    ax.set(
        title="Both shower information and event context contributed",
        ylabel="Mean validation efficiency difference (percentage points)",
        ylim=(0, max(values) * 1.25),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "representation_gain_sources.png")


def plot_output_fusion(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "representation_study" / "representation_validation.csv",
        ("study", "candidate", "mean_inclusive_efficiency", "sd_inclusive_efficiency"),
    )
    rows = [row for row in rows if row["study"] == "output_fusion"]
    values = [float(row["mean_inclusive_efficiency"]) for row in rows]
    errors = [float(row["sd_inclusive_efficiency"]) for row in rows]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    bars = ax.bar(
        [_candidate_label(row["candidate"]) for row in rows],
        values,
        yerr=errors,
        capsize=4,
        color=["#2878b5", "#4daf4a", "#8f6bb3"],
    )
    ax.bar_label(bars, labels=[f"{100 * value:.2f}%" for value in values], padding=5)
    ax.set(
        title="Output target and pair symmetry gave similar validation results",
        ylabel="Inclusive signal-sample event efficiency",
        ylim=(0.39, 0.45),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "output_and_fusion_validation.png")


def plot_high_resolution_regions(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "high_resolution" / "high_resolution_energy_regions.csv",
        ("seed", "region_gev", "paired_delta"),
    )
    regions = list(dict.fromkeys(row["region_gev"] for row in rows))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    means = []
    lows = []
    highs = []
    for region in regions:
        selected = [row for row in rows if row["region_gev"] == region]
        means.append(sum(float(row["paired_delta"]) for row in selected) / len(selected))
        lows.append(min(float(row["ci95_low"]) for row in selected))
        highs.append(max(float(row["ci95_high"]) for row in selected))
    errors = [[mean - low for mean, low in zip(means, lows, strict=True)],
              [high - mean for mean, high in zip(means, highs, strict=True)]]
    ax.errorbar(regions, [100 * value for value in means],
                yerr=[[100 * value for value in errors[0]], [100 * value for value in errors[1]]],
                fmt="o-", linewidth=2.2, markersize=8, capsize=5, color="#2878b5")
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set(
        title="The high-resolution model's gain was concentrated below 60 GeV",
        xlabel="Matched-member pT region (GeV; conditional diagnostic)",
        ylabel="Model minus baseline efficiency (percentage points)",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "high_resolution_energy_regions.png")


def plot_high_energy_recovery(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "high_energy_recovery" / "recovery_candidates.csv",
        ("candidate", "seed", "paired_delta"),
    )
    candidates = list(dict.fromkeys(row["candidate"] for row in rows))
    means = []
    lows = []
    highs = []
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate]
        means.append(sum(float(row["paired_delta"]) for row in selected) / len(selected))
        lows.append(min(float(row["ci95_low"]) for row in selected))
        highs.append(max(float(row["ci95_high"]) for row in selected))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    positions = list(range(len(candidates)))
    ax.errorbar(
        positions,
        [100 * value for value in means],
        yerr=[
            [100 * (mean - low) for mean, low in zip(means, lows, strict=True)],
            [100 * (high - mean) for mean, high in zip(means, highs, strict=True)],
        ],
        fmt="o",
        markersize=8,
        capsize=5,
        color="#2878b5",
    )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(positions, [_candidate_label(candidate) for candidate in candidates])
    ax.set(
        title="Only the OR candidate closed the high-energy point-estimate gap",
        ylabel="Model minus baseline efficiency above 60 GeV (percentage points)",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "high_energy_recovery.png")


def plot_pair_or_validation(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pair_or" / "pair_or_validation.csv",
        ("seed", "inclusive_delta", "union_fpr"),
    )
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    values = [100 * float(row["inclusive_delta"]) for row in rows]
    bars = ax.bar([f"seed {row['seed']}" for row in rows], values, color="#4daf4a")
    ax.bar_label(bars, labels=[f"+{value:.1f} points" for value in values], padding=4)
    ax.set(
        title="The pair-network + pT OR improved the validation point estimate",
        ylabel="Inclusive efficiency difference versus baseline",
        ylim=(0, max(values) * 1.25),
    )
    ax.text(
        0.99,
        0.96,
        "All seeds: 124 / 24,835 background events accepted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#555555",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "pair_or_validation.png")


def plot_pair_or_regions(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pair_or" / "pair_or_energy_regions.csv",
        ("seed", "region_gev", "paired_delta", "ci95_low", "ci95_high"),
    )
    regions = list(dict.fromkeys(row["region_gev"] for row in rows))
    means = []
    lows = []
    highs = []
    for region in regions:
        selected = [row for row in rows if row["region_gev"] == region]
        means.append(sum(float(row["paired_delta"]) for row in selected) / len(selected))
        lows.append(min(float(row["ci95_low"]) for row in selected))
        highs.append(max(float(row["ci95_high"]) for row in selected))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    positions = list(range(len(regions)))
    ax.errorbar(
        positions,
        [100 * value for value in means],
        yerr=[
            [100 * (mean - low) for mean, low in zip(means, lows, strict=True)],
            [100 * (high - mean) for mean, high in zip(means, highs, strict=True)],
        ],
        fmt="o-",
        linewidth=2.2,
        markersize=8,
        capsize=5,
        color="#4daf4a",
    )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(positions, regions)
    ax.set(
        title="The OR candidate had positive point estimates in all three regions",
        xlabel="Matched-member pT region (GeV; conditional diagnostic)",
        ylabel="OR minus baseline efficiency (percentage points)",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "pair_or_energy_regions.png")


def plot_pt_context_ranking(results_dir: Path, output_dir: Path) -> Path:
    rows = _read_rows(
        results_dir / "pt_context" / "summary.csv",
        ("candidate", "display_label", "mean_inclusive_signal_efficiency"),
    )
    rows.sort(key=lambda row: float(row["mean_inclusive_signal_efficiency"]))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    colors = []
    for row in rows:
        if row["candidate"] == "current_member_pt_and_event_second_pt":
            colors.append("#163b65")
        elif row["candidate"] == "pair_sum_and_balance":
            colors.append("#1f9e89")
        elif row["candidate"] == "deterministic_second_pt_baseline":
            colors.append("#8a9199")
        else:
            colors.append("#75aadb")
    values = [100 * float(row["mean_inclusive_signal_efficiency"]) for row in rows]
    bars = ax.barh([row["display_label"] for row in rows], values, color=colors)
    ax.bar_label(bars, labels=[f"{value:.3f}%" for value in values], padding=4)
    ax.set(
        title="Alternative pT context did not improve the current model",
        xlabel="Mean inclusive signal-sample validation efficiency",
        xlim=(30, 47),
    )
    ax.text(0.01, -0.17, "Zoomed efficiency scale", transform=ax.transAxes, color="#555555")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _finish(fig, output_dir / "pt_context_ranking.png")


def plot_representation_reproduction_global(
    results_dir: Path, output_dir: Path
) -> Path:
    rows = _read_rows(
        results_dir / "representation_reproduction" / "global_validation.csv",
        (
            "candidate",
            "display_label",
            "seed",
            "inclusive_signal_efficiency",
            "baseline_efficiency",
        ),
    )
    summary = _read_rows(
        results_dir / "representation_reproduction" / "global_summary.csv",
        ("candidate", "display_label", "mean_inclusive_signal_efficiency"),
    )
    baseline = float(rows[0]["baseline_efficiency"])
    candidates = [row["candidate"] for row in summary]
    labels = ["Deterministic\npT baseline"] + [
        _candidate_label(candidate) for candidate in candidates
    ]
    means = [baseline] + [
        float(row["mean_inclusive_signal_efficiency"]) for row in summary
    ]

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    positions = list(range(len(labels)))
    colors = ["#7b8188", "#2f74b5", "#1f9e89", "#7a63a8"]
    bars = ax.bar(positions, [100 * value for value in means], color=colors, width=0.67)
    ax.bar_label(
        bars,
        labels=[f"{100 * value:.2f}%" for value in means],
        padding=5,
        fontsize=9,
    )

    jitter = {42: -0.10, 123: 0.0, 456: 0.10}
    for index, candidate in enumerate(candidates, start=1):
        selected = [row for row in rows if row["candidate"] == candidate]
        for row in selected:
            seed = int(row["seed"])
            ax.scatter(
                index + jitter[seed],
                100 * float(row["inclusive_signal_efficiency"]),
                s=35,
                facecolor="white",
                edgecolor="#20252a",
                linewidth=1,
                zorder=3,
            )

    ax.set_xticks(positions, labels)
    ax.set(
        title="Three reproduced representations improved the validation point estimate",
        ylabel="Inclusive signal-sample validation efficiency (%)",
        ylim=(32, 43),
    )
    ax.text(
        0.01,
        0.97,
        "Bars: three-seed mean   •   White dots: individual seeds",
        transform=ax.transAxes,
        va="top",
        color="#50565c",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.99,
        0.015,
        "Validation only · protected confirmation pending · test unopened",
        ha="right",
        color="#666666",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return _finish(fig, output_dir / "representation_reproduction_global.png")


def plot_representation_reproduction_regions(
    results_dir: Path, output_dir: Path
) -> Path:
    rows = _read_rows(
        results_dir / "representation_reproduction" / "energy_region_deltas.csv",
        (
            "candidate",
            "seed",
            "region_gev",
            "candidate_minus_baseline",
        ),
    )
    region_order = ["10-25", "25-60", "60+"]
    candidates = list(dict.fromkeys(row["candidate"] for row in rows))
    colors = ["#2f74b5", "#1f9e89", "#7a63a8"]
    offsets = [-0.18, 0.0, 0.18]
    seed_jitter = {"42": -0.035, "123": 0.0, "456": 0.035}

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    for candidate_index, (candidate, color) in enumerate(
        zip(candidates, colors, strict=True)
    ):
        mean_rows = {
            row["region_gev"]: row
            for row in rows
            if row["candidate"] == candidate and row["seed"] == "mean"
        }
        x_values = [index + offsets[candidate_index] for index in range(3)]
        mean_values = [
            100 * float(mean_rows[region]["candidate_minus_baseline"])
            for region in region_order
        ]
        ax.plot(
            x_values,
            mean_values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color=color,
            label=_candidate_label(candidate).replace("\n", " "),
        )
        for region_index, region in enumerate(region_order):
            seed_rows = [
                row
                for row in rows
                if row["candidate"] == candidate
                and row["region_gev"] == region
                and row["seed"] != "mean"
            ]
            for row in seed_rows:
                ax.scatter(
                    region_index
                    + offsets[candidate_index]
                    + seed_jitter[row["seed"]],
                    100 * float(row["candidate_minus_baseline"]),
                    s=27,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1,
                    zorder=3,
                )

    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.set_xticks(range(3), ["10–25", "25–60", "60+"])
    ax.set(
        title="Validation gains were concentrated below 60 GeV",
        xlabel="Matched-member pT region (GeV; conditional diagnostic)",
        ylabel="Candidate minus baseline efficiency (percentage points)",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.text(
        0.99,
        0.015,
        "Lines: three-seed mean   •   White dots: individual seeds\n"
        "Validation only",
        ha="right",
        color="#666666",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return _finish(fig, output_dir / "representation_reproduction_energy_regions.png")


PLOTS = {
    "training-curves": plot_training_curves,
    "checkpoint-selection": plot_checkpoint_selection,
    "pt-validation": plot_pt_validation,
    "feature-ranking": plot_feature_ranking,
    "feature-redundancy": plot_redundancy,
    "representation-validation": plot_representation_validation,
    "gain-sources": plot_gain_sources,
    "output-fusion": plot_output_fusion,
    "high-resolution-regions": plot_high_resolution_regions,
    "high-energy-recovery": plot_high_energy_recovery,
    "pair-or-validation": plot_pair_or_validation,
    "pair-or-regions": plot_pair_or_regions,
    "pt-context-ranking": plot_pt_context_ranking,
    "representation-reproduction-global": plot_representation_reproduction_global,
    "representation-reproduction-regions": plot_representation_reproduction_regions,
}


def generate_plots(results_dir: Path, output_dir: Path, names: Iterable[str]) -> list[Path]:
    return [PLOTS[name](results_dir, output_dir) for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", choices=("all", *PLOTS), default="all")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.results_dir / "plots"
    names = list(PLOTS) if args.plot == "all" else [args.plot]
    for output in generate_plots(args.results_dir, output_dir, names):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
