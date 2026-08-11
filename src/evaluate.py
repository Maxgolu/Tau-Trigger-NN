import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def build_event_trigger_scores(df, criterion, objects=2):
    """Return the score at which every background event starts to pass.

    With the common ``score >= threshold`` rule, an event passes an
    ``at least objects`` trigger exactly when its objects-th highest score is
    at least the threshold. Events with too few finite objects never pass but
    remain part of the FPR denominator.
    """
    if objects < 1:
        raise ValueError("objects must be at least 1")
    required = {"eventNumber", criterion}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    event_count = int(df["eventNumber"].nunique())
    if event_count == 0:
        raise ValueError("Cannot calibrate a threshold without background events")

    finite_rows = df.loc[
        np.isfinite(df[criterion].to_numpy(dtype=float)),
        ["eventNumber", criterion],
    ]
    ordered = finite_rows.sort_values(
        ["eventNumber", criterion], ascending=[True, False], kind="mergesort"
    )
    kth_scores = (
        ordered.groupby("eventNumber", sort=False)[criterion]
        .nth(objects - 1)
        .to_numpy(dtype=float)
    )
    return kth_scores, event_count


def select_fpr_threshold(event_trigger_scores, event_count, target_fake_rate):
    """Select the lowest threshold whose empirical event FPR is within target.

    Tied scores are kept together. Therefore the achieved FPR can be below the
    requested value when no deterministic threshold can attain it exactly.
    """
    if not 0.0 <= target_fake_rate <= 1.0:
        raise ValueError("target_fake_rate must be between 0 and 1")
    if event_count <= 0:
        raise ValueError("event_count must be positive")

    scores = np.asarray(event_trigger_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.inf, 0.0

    # The integer budget guarantees achieved_fpr <= target_fake_rate.
    max_accepted = int(np.floor(target_fake_rate * event_count + 1e-12))
    unique_scores, tied_counts = np.unique(scores, return_counts=True)
    unique_scores = unique_scores[::-1]
    tied_counts = tied_counts[::-1]
    cumulative = np.cumsum(tied_counts)
    feasible = np.flatnonzero(cumulative <= max_accepted)

    if feasible.size == 0:
        # Including even the highest tied score would exceed the target.
        threshold = np.nextafter(unique_scores[0], np.inf)
    else:
        last = int(feasible[-1])
        threshold = unique_scores[last]

    # Recompute with the exact same >= rule used by the final efficiency curve.
    achieved_fpr = float(np.count_nonzero(scores >= threshold) / event_count)
    return float(threshold), achieved_fpr


def CalcThresh(DF, Criteria, FakeRate, objects=2):
    """Backward-compatible wrapper for exact event-level FPR calibration."""
    event_scores, event_count = build_event_trigger_scores(DF, Criteria, objects)
    return select_fpr_threshold(event_scores, event_count, FakeRate)


def select_truth_tau_objects(df):
    """Return only objects that are truth-matched to a tau.

    The Signal sample also contains non-tau objects, so its sample name alone
    is not an object label.
    """
    required = {"Type", "signal"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df.loc[(df["Type"] == "Signal") & (df["signal"] == 1)]


def score_pass_mask(df, criterion, threshold):
    """Apply the final score cut with the calibration numeric precision."""
    if criterion not in df.columns:
        raise KeyError(f"Missing required column: {criterion}")
    scores = df[criterion].to_numpy(dtype=np.float64)
    return np.isfinite(scores) & (scores >= np.float64(threshold))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tau Particle NN Predictions")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to the experiment directory. If omitted, runs on all subfolders.")
    parser.add_argument("--experiments_dir", type=str, default=str(DEFAULT_EXPERIMENTS_DIR),
                        help="Directory containing experiment subfolders.")
    parser.add_argument("--recalc", action="store_true",
                        help="Recalculates even if metrics.json already exists.")

    # Binning Arguments
    parser.add_argument("--num_bins", type=int, default=44,
                        help="Number of bins. Default is 44.")
    parser.add_argument("--pt_min", type=float, default=10.0,
                        help="Minimum value for bins. Default is 10.0.")
    parser.add_argument("--pt_max", type=float, default=120.0,
                        help="Maximum value for bins. Default is 120.0.")
    parser.add_argument("--bin_var", type=str, default="truth_pt",
                        help="Column name to use for binning (e.g., truth_pt, pt, eta). Default is truth_pt.")

    return parser.parse_args()


def fermi_dirac(x, plateau, slope, midpoint):
    return plateau / (1.0 + np.exp(-np.clip(slope * (x - midpoint), -500, 500)))


def calculate_binned_efficiencies(
    signal_df, criterion, threshold, eval_bins, bin_var
):
    """Measure object efficiency in fixed bins using the shared score cut."""
    efficiencies = []
    errors = []
    for b_min, b_max in zip(eval_bins[:-1], eval_bins[1:]):
        bin_signal = signal_df[
            (signal_df[bin_var] >= b_min) & (signal_df[bin_var] < b_max)
        ]
        denominator = len(bin_signal)
        if denominator > 0:
            passed = np.count_nonzero(
                score_pass_mask(bin_signal, criterion, threshold)
            )
            efficiency = passed / denominator
            error = np.sqrt(
                efficiency * (1.0 - efficiency) / denominator
            )
            if error == 0.0:
                error = 1e-5
        else:
            efficiency = 0.0
            error = 1.0

        efficiencies.append(float(efficiency))
        errors.append(float(error))

    return efficiencies, errors


def fit_binned_efficiencies(bin_centers, efficiencies, errors):
    """Fit the standard inverse Fermi-Dirac turn-on model."""
    try:
        p0 = [1.0, 0.1, 40.0]
        bounds = ([0.0, 0.0, 0.0], [1.0, np.inf, 300.0])
        popt, _ = curve_fit(
            fermi_dirac,
            bin_centers,
            efficiencies,
            p0=p0,
            bounds=bounds,
            sigma=errors,
            absolute_sigma=True,
        )
        plateau, slope, midpoint = popt

        observed = np.asarray(efficiencies)
        uncertainty = np.asarray(errors)
        fitted = fermi_dirac(bin_centers, plateau, slope, midpoint)
        valid = uncertainty < 1.0
        degrees_of_freedom = np.sum(valid) - len(popt)
        if degrees_of_freedom > 0:
            chi2_value = np.sum(
                ((observed[valid] - fitted[valid]) / uncertainty[valid]) ** 2
            )
            chi2_reduced = chi2_value / degrees_of_freedom
            p_value = chi2.sf(chi2_value, degrees_of_freedom)
        else:
            chi2_reduced, p_value = 0.0, 0.0

        return {
            "fit_success": True,
            "plateau": float(plateau),
            "slope": float(slope),
            "midpoint": float(midpoint),
            "chi2_red": float(chi2_reduced),
            "p_value": float(p_value),
        }
    except Exception as error:
        return {
            "fit_success": False,
            "plateau": 0.0,
            "slope": 0.0,
            "midpoint": 0.0,
            "chi2_red": 0.0,
            "p_value": 0.0,
            "error": str(error),
        }


def save_turn_on_plot(metrics, exp_dir):
    """Save a basic turn-on plot beside the numerical metrics."""
    curve = metrics["turn_on_curve"]
    bins = np.asarray(curve["bins"], dtype=float)
    centers = (bins[:-1] + bins[1:]) / 2.0
    efficiencies = np.asarray(curve["binned_efficiencies"], dtype=float)
    errors = np.asarray(curve["binned_efficiencies_err"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers, efficiencies, yerr=errors, fmt="o", markersize=4,
                capsize=2, label="Measured efficiency")

    fit = curve["fermi_dirac_fit"]
    if curve.get("fit_success"):
        x_fit = np.linspace(bins[0], bins[-1], 500)
        y_fit = fermi_dirac(x_fit, fit["plateau"], fit["slope"], fit["midpoint"])
        ax.plot(x_fit, y_fit, label="Inverse Fermi-Dirac fit")

    baseline = metrics.get("baseline_tob_pt")
    if baseline:
        baseline_efficiencies = np.asarray(
            baseline["binned_efficiencies"], dtype=float
        )
        baseline_errors = np.asarray(
            baseline["binned_efficiencies_err"], dtype=float
        )
        ax.errorbar(
            centers,
            baseline_efficiencies,
            yerr=baseline_errors,
            fmt="s-",
            color="black",
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label="Baseline tob_pt (data)",
        )
        baseline_fit = baseline["fermi_dirac_fit"]
        if baseline.get("fit_success"):
            x_fit = np.linspace(bins[0], bins[-1], 500)
            y_fit = fermi_dirac(
                x_fit,
                baseline_fit["plateau"],
                baseline_fit["slope"],
                baseline_fit["midpoint"],
            )
            ax.plot(
                x_fit,
                y_fit,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Baseline tob_pt (fit)",
            )

    ax.set_xlabel(f"{curve['binning_variable']} [GeV]")
    ax.set_ylabel("Signal efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(exp_dir, "turn_on_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved turn-on plot to {plot_path}")


def load_predictions(exp_dir):
    """Load predictions from preferred Parquet storage or the CSV fallback."""
    parquet_path = os.path.join(exp_dir, "predictions.parquet")
    csv_path = os.path.join(exp_dir, "predictions.csv")

    if os.path.exists(parquet_path):
        try:
            print(f"Loading predictions from {parquet_path}...")
            return pd.read_parquet(parquet_path)
        except (ImportError, OSError) as error:
            if not os.path.exists(csv_path):
                raise RuntimeError(
                    f"Could not read {parquet_path} and no predictions.csv fallback exists"
                ) from error
            print(f"Parquet unavailable ({error.__class__.__name__}); loading CSV fallback.")

    if os.path.exists(csv_path):
        print(f"Loading predictions from {csv_path}...")
        return pd.read_csv(csv_path)

    return None


def evaluate_experiment(exp_dir, recalc, num_bins, pt_min, pt_max, bin_var):
    print(f"\n{'=' * 40}")
    print(f"Evaluating Experiment: {exp_dir}")
    print(f"{'=' * 40}")

    metrics_path = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_path) and not recalc:
        print(f"  -> metrics.json already exists. Skipping (use --recalc to override).\n")
        return

    df = load_predictions(exp_dir)
    if df is None:
        print(f"  -> Could not find predictions.parquet or predictions.csv in {exp_dir}. Skipping.\n")
        return

    # CRITICAL FIX 1: Convert MeV to GeV for specific kinematic columns
    if 'tob_pt' in df.columns:
        df['tob_pt'] = df['tob_pt'] / 1000.0
    if 'truth_pt' in df.columns:
        df['truth_pt'] = df['truth_pt'] / 1000.0

    # Ensure the requested bin_var actually exists in the dataframe
    if bin_var not in df.columns:
        print(f"  -> Error: Column '{bin_var}' not found in predictions. parquet. Skipping.\n")
        return

    sig_df = select_truth_tau_objects(df)
    bkg_df = df[df['Type'] == 'BKG']

    target_fake_rates = [0.005, 0.010, 0.020]
    thresholds = {}
    achieved_fake_rates = {}
    global_efficiencies = {}

    print("Calculating Thresholds and Global Efficiencies...")
    event_trigger_scores, background_event_count = build_event_trigger_scores(
        bkg_df, 'nn_score', objects=2
    )
    for fr in target_fake_rates:
        threshold, ratio = select_fpr_threshold(
            event_trigger_scores, background_event_count, fr
        )
        sig_passed = np.count_nonzero(
            score_pass_mask(sig_df, 'nn_score', threshold)
        )
        eff = sig_passed / len(sig_df) if len(sig_df) > 0 else 0.0

        fr_str = f"{fr * 100:.1f}%"
        thresholds[fr_str] = float(threshold)
        achieved_fake_rates[fr_str] = float(ratio)
        global_efficiencies[fr_str] = float(eff)
        print(
            f"  Target FPR: {fr_str:>4} | Achieved FPR: {ratio * 100:7.4f}% "
            f"| Cut: {threshold:9.6g} | Signal Eff: {eff:.4f}"
        )
        if fr - ratio > (1.0 / background_event_count) + 1e-12:
            print(
                "    Note: score ties prevent a closer deterministic operating "
                "point without exceeding the target."
            )

    target_fr = "0.5%"
    working_threshold = thresholds[target_fr]

    # Dynamically generate equal-width bins
    eval_bins = np.linspace(pt_min, pt_max, num_bins + 1)
    bin_centers = (eval_bins[:-1] + eval_bins[1:]) / 2.0

    print(
        f"\nCalculating Binned Efficiencies at {target_fr} Fake Rate (using {num_bins} bins from {pt_min} to {pt_max} on '{bin_var}')...")
    binned_effs, binned_effs_err = calculate_binned_efficiencies(
        sig_df, 'nn_score', working_threshold, eval_bins, bin_var
    )

    print("Fitting Fermi-Dirac Function...")
    nn_fit = fit_binned_efficiencies(
        bin_centers, binned_effs, binned_effs_err
    )
    if nn_fit["fit_success"]:
        print(
            f"  Midpoint: {nn_fit['midpoint']:.2f} | "
            f"Slope: {nn_fit['slope']:.4f} | "
            f"Plateau: {nn_fit['plateau']:.4f}"
        )
        print(
            f"  Fit Quality -> Chi2/ndf: {nn_fit['chi2_red']:.3f} | "
            f"p-value: {nn_fit['p_value']:.4e}"
        )
    else:
        print(f"  Curve fitting failed: {nn_fit['error']}")

    baseline_event_scores, baseline_event_count = build_event_trigger_scores(
        bkg_df, 'tob_pt', objects=2
    )
    baseline_threshold, baseline_fpr = select_fpr_threshold(
        baseline_event_scores, baseline_event_count, 0.005
    )
    baseline_effs, baseline_errors = calculate_binned_efficiencies(
        sig_df, 'tob_pt', baseline_threshold, eval_bins, bin_var
    )
    baseline_fit = fit_binned_efficiencies(
        bin_centers, baseline_effs, baseline_errors
    )
    print(
        f"Baseline tob_pt -> Achieved FPR: {baseline_fpr * 100:.4f}% | "
        f"Cut: {baseline_threshold:.6g}"
    )

    metrics = {
        "thresholds_by_fake_rate": thresholds,
        "achieved_fake_rates": achieved_fake_rates,
        "global_efficiency": global_efficiencies,
        "turn_on_curve": {
            "binning_variable": bin_var,
            "bins": eval_bins.tolist(),
            "binned_efficiencies": binned_effs,
            "binned_efficiencies_err": binned_effs_err,
            "target_fake_rate_used": target_fr,
            "fit_success": nn_fit["fit_success"],
            "fermi_dirac_fit": {
                "plateau": nn_fit["plateau"],
                "slope": nn_fit["slope"],
                "midpoint": nn_fit["midpoint"],
                "chi2_red": nn_fit["chi2_red"],
                "p_value": nn_fit["p_value"]
            }
        },
        "baseline_tob_pt": {
            "threshold": float(baseline_threshold),
            "achieved_fake_rate": float(baseline_fpr),
            "binned_efficiencies": baseline_effs,
            "binned_efficiencies_err": baseline_errors,
            "fit_success": baseline_fit["fit_success"],
            "fermi_dirac_fit": {
                "plateau": baseline_fit["plateau"],
                "slope": baseline_fit["slope"],
                "midpoint": baseline_fit["midpoint"],
                "chi2_red": baseline_fit["chi2_red"],
                "p_value": baseline_fit["p_value"],
            },
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    save_turn_on_plot(metrics, exp_dir)

    print(f"\nEvaluation complete! Hard numbers saved to {metrics_path}")


def generate_averaged_metrics(exp_name, folders):
    from scipy.stats import chi2

    all_metrics = []
    for folder in folders:
        with open(os.path.join(folder, "metrics.json"), 'r') as f:
            all_metrics.append(json.load(f))

    base = all_metrics[0]
    bin_var = base["turn_on_curve"]["binning_variable"]
    eval_bins = np.array(base["turn_on_curve"]["bins"])
    bin_centers = (eval_bins[:-1] + eval_bins[1:]) / 2.0
    target_fr = base["turn_on_curve"]["target_fake_rate_used"]

    # Average the thresholds and global efficiencies across all fake rates
    avg_thresholds = {k: np.mean([m["thresholds_by_fake_rate"][k] for m in all_metrics]) for k in
                      base["thresholds_by_fake_rate"]}
    avg_achieved_fake_rates = {
        k: np.mean([m["achieved_fake_rates"][k] for m in all_metrics])
        for k in base["achieved_fake_rates"]
    }
    avg_global_effs = {k: np.mean([m["global_efficiency"][k] for m in all_metrics]) for k in base["global_efficiency"]}

    # Average the bins
    avg_binned_effs = np.mean([m["turn_on_curve"]["binned_efficiencies"] for m in all_metrics], axis=0)
    avg_binned_errs = np.mean([m["turn_on_curve"]["binned_efficiencies_err"] for m in all_metrics], axis=0)

    # Average the independently calibrated tob_pt baseline across the same seeds.
    baseline_entries = [m.get("baseline_tob_pt") for m in all_metrics]
    has_complete_baseline = all(entry is not None for entry in baseline_entries)
    if has_complete_baseline:
        avg_baseline_threshold = np.mean(
            [entry["threshold"] for entry in baseline_entries]
        )
        avg_baseline_fpr = np.mean(
            [entry["achieved_fake_rate"] for entry in baseline_entries]
        )
        avg_baseline_effs = np.mean(
            [entry["binned_efficiencies"] for entry in baseline_entries],
            axis=0,
        )
        avg_baseline_errs = np.mean(
            [entry["binned_efficiencies_err"] for entry in baseline_entries],
            axis=0,
        )
        averaged_baseline_fit = fit_binned_efficiencies(
            bin_centers,
            avg_baseline_effs,
            avg_baseline_errs,
        )

    try:
        p0 = [1.0, 0.1, 40.0]
        bounds = ([0.0, 0.0, 0.0], [1.0, np.inf, 300.0])
        popt, _ = curve_fit(fermi_dirac, bin_centers, avg_binned_effs, p0=p0, bounds=bounds, sigma=avg_binned_errs,
                            absolute_sigma=True)
        fd_plateau, fd_slope, fd_midpoint = popt

        y_obs, y_err = np.array(avg_binned_effs), np.array(avg_binned_errs)
        y_fit = fermi_dirac(bin_centers, fd_plateau, fd_slope, fd_midpoint)

        valid = y_err < 1.0
        dof = np.sum(valid) - len(popt)
        if dof > 0:
            chi2_val = np.sum(((y_obs[valid] - y_fit[valid]) / y_err[valid]) ** 2)
            chi2_red = chi2_val / dof
            p_value = chi2.sf(chi2_val, dof)
        else:
            chi2_red, p_value = 0.0, 0.0

        fit_success = True
        print(
            f"  [{exp_name}] Averaged Midpoint: {fd_midpoint:.2f} | Plateau: {fd_plateau:.4f} | Chi2/ndf: {chi2_red:.3f}")
    except Exception as e:
        print(f"  [{exp_name}] Averaged Curve fitting failed: {e}")
        fd_plateau, fd_slope, fd_midpoint, chi2_red, p_value = 0.0, 0.0, 0.0, 0.0, 0.0
        fit_success = False

    metrics_averaged = {
        "thresholds_by_fake_rate": {k: float(v) for k, v in avg_thresholds.items()},
        "achieved_fake_rates": {k: float(v) for k, v in avg_achieved_fake_rates.items()},
        "global_efficiency": {k: float(v) for k, v in avg_global_effs.items()},
        "turn_on_curve": {
            "binning_variable": bin_var,
            "bins": eval_bins.tolist(),
            "binned_efficiencies": avg_binned_effs.tolist(),
            "binned_efficiencies_err": avg_binned_errs.tolist(),
            "target_fake_rate_used": target_fr,
            "fit_success": fit_success,
            "fermi_dirac_fit": {
                "plateau": float(fd_plateau),
                "slope": float(fd_slope),
                "midpoint": float(fd_midpoint),
                "chi2_red": float(chi2_red),
                "p_value": float(p_value)
            },
            "seeds_averaged": len(all_metrics)
        }
    }

    if has_complete_baseline:
        metrics_averaged["baseline_tob_pt"] = {
            "threshold": float(avg_baseline_threshold),
            "achieved_fake_rate": float(avg_baseline_fpr),
            "binned_efficiencies": avg_baseline_effs.tolist(),
            "binned_efficiencies_err": avg_baseline_errs.tolist(),
            "fit_success": averaged_baseline_fit["fit_success"],
            "fermi_dirac_fit": {
                "plateau": averaged_baseline_fit["plateau"],
                "slope": averaged_baseline_fit["slope"],
                "midpoint": averaged_baseline_fit["midpoint"],
                "chi2_red": averaged_baseline_fit["chi2_red"],
                "p_value": averaged_baseline_fit["p_value"],
            },
            "seeds_averaged": len(all_metrics),
        }

    for folder in folders:
        with open(os.path.join(folder, "metrics_averaged.json"), 'w') as f:
            json.dump(metrics_averaged, f, indent=4)


def main():
    args = parse_args()

    base_dir = os.path.abspath(args.experiments_dir)

    # 1. Standard evaluation sequence
    if args.experiment_dir:
        experiment_dir = os.path.abspath(args.experiment_dir)
        evaluate_experiment(experiment_dir, args.recalc, args.num_bins, args.pt_min, args.pt_max, args.bin_var)
        base_dir = os.path.dirname(experiment_dir)
    else:
        if not os.path.isdir(base_dir):
            print(f"Experiments directory not found: {base_dir}")
            return
        print(f"No --experiment_dir provided. Scanning '{base_dir}' for experiment folders...")
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            has_predictions = (
                os.path.exists(os.path.join(item_path, "predictions.parquet"))
                or os.path.exists(os.path.join(item_path, "predictions.csv"))
            )
            if os.path.isdir(item_path) and has_predictions:
                evaluate_experiment(item_path, args.recalc, args.num_bins, args.pt_min, args.pt_max, args.bin_var)

    # 2. Always group and average the seeds afterward
    print(f"\n{'=' * 40}")
    print("Generating Averaged Metrics for multi-seed experiments...")
    print(f"{'=' * 40}")

    from collections import defaultdict
    experiment_groups = defaultdict(list)

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        config_path = os.path.join(item_path, "config.json")
        metrics_path = os.path.join(item_path, "metrics.json")

        # Only group folders that have successfully generated the standard metrics
        if os.path.isdir(item_path) and os.path.exists(config_path) and os.path.exists(metrics_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if "achieved_fake_rates" not in metrics:
                print(f"  -> Skipping legacy metrics without verified FPR: {item_path}")
                continue
            exp_name = config.get("experiment_name", "unknown")
            experiment_groups[exp_name].append(item_path)

    for exp_name, folders in experiment_groups.items():
        if len(folders) > 1:
            generate_averaged_metrics(exp_name, folders)
        else:
            print(f"  -> Skipping '{exp_name}': Only 1 seed found.")

if __name__ == "__main__":
    main()
