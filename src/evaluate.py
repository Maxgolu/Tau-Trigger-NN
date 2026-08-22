import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from pathlib import Path

from classifiers import (
    calibrate_classifier,
    classifier_event_pass_mask,
    classifier_object_pass_mask,
    parse_classifier,
    tob_pt_gev,
)
from operating_point import (
    CalcThresh,
    build_event_trigger_scores,
    score_pass_mask,
    select_background_objects,
    select_fpr_threshold,
    select_truth_tau_objects,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tau Particle NN Predictions")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to the experiment directory. If omitted, runs on all subfolders.")
    parser.add_argument("--experiments_dir", type=str, default=str(DEFAULT_EXPERIMENTS_DIR),
                        help="Directory containing experiment subfolders.")
    parser.add_argument("--recalc", action="store_true",
                        help="Recalculates even if metrics.json already exists.")
    parser.add_argument(
        "--classifier",
        choices=["nn_only", "tob_nn_or"],
        default=None,
        help=(
            "Optional post-hoc classifier override. Results receive a classifier "
            "suffix and do not overwrite the configured classifier metrics."
        ),
    )
    parser.add_argument(
        "--classifier-tob-fpr",
        type=float,
        default=0.004,
        help="TOB branch event-FPR budget for tob_nn_or (default: 0.004).",
    )

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


def calculate_binned_mask_efficiencies(
    signal_df, passed, eval_bins, bin_var
):
    """Measure binned efficiency from a precomputed classifier decision."""
    passed = np.asarray(passed, dtype=bool)
    if len(passed) != len(signal_df):
        raise ValueError("Pass mask must align one-to-one with signal rows")
    values = signal_df[bin_var].to_numpy(dtype=np.float64)
    efficiencies = []
    errors = []
    for b_min, b_max in zip(eval_bins[:-1], eval_bins[1:]):
        in_bin = (values >= b_min) & (values < b_max)
        denominator = int(np.count_nonzero(in_bin))
        if denominator:
            efficiency = float(np.count_nonzero(passed & in_bin) / denominator)
            error = float(
                np.sqrt(efficiency * (1.0 - efficiency) / denominator)
            )
            if error == 0.0:
                error = 1e-5
        else:
            efficiency = 0.0
            error = 1.0
        efficiencies.append(efficiency)
        errors.append(error)
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


def _add_baseline(ax, metrics, centers, bins):
    """Add the measured and fitted TOB baseline to one axis."""
    baseline = metrics.get("baseline_tob_pt")
    if not baseline:
        return
    ax.errorbar(
        centers,
        np.asarray(baseline["binned_efficiencies"], dtype=float),
        yerr=np.asarray(baseline["binned_efficiencies_err"], dtype=float),
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
        ax.plot(
            x_fit,
            fermi_dirac(
                x_fit,
                baseline_fit["plateau"],
                baseline_fit["slope"],
                baseline_fit["midpoint"],
            ),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Baseline tob_pt (fit)",
        )


def _finish_turn_on_plot(fig, ax, curve, output_path):
    ax.set_xlabel(f"{curve['binning_variable']} [GeV]")
    ax.set_ylabel("Signal efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved turn-on plot to {output_path}")


def save_turn_on_plot(metrics, exp_dir, filename="turn_on_curve.png"):
    """Save the final classifier versus baseline without branch clutter."""
    curve = metrics["turn_on_curve"]
    bins = np.asarray(curve["bins"], dtype=float)
    centers = (bins[:-1] + bins[1:]) / 2.0
    efficiencies = np.asarray(curve["binned_efficiencies"], dtype=float)
    errors = np.asarray(curve["binned_efficiencies_err"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers, efficiencies, yerr=errors, fmt="o", markersize=4,
                capsize=2, label="Configured classifier")

    fit = curve["fermi_dirac_fit"]
    if curve.get("fit_success"):
        x_fit = np.linspace(bins[0], bins[-1], 500)
        y_fit = fermi_dirac(x_fit, fit["plateau"], fit["slope"], fit["midpoint"])
        ax.plot(x_fit, y_fit, label="Inverse Fermi-Dirac fit")

    _add_baseline(ax, metrics, centers, bins)
    plot_path = os.path.join(exp_dir, filename)
    _finish_turn_on_plot(fig, ax, curve, plot_path)

    branches = metrics.get("classifier_branches", {})
    if branches:
        fig, ax = plt.subplots(figsize=(8, 5))
        for branch_name, style in (
            ("nn", {"color": "tab:orange", "linestyle": ":"}),
            ("tob", {"color": "0.45", "linestyle": "-."}),
        ):
            branch = branches.get(branch_name)
            if branch:
                ax.plot(
                    centers,
                    branch["binned_efficiencies"],
                    linewidth=1.6,
                    label=branch["label"],
                    **style,
                )
        _add_baseline(ax, metrics, centers, bins)
        stem, extension = os.path.splitext(filename)
        branch_path = os.path.join(
            exp_dir, f"{stem}_branches{extension or '.png'}"
        )
        _finish_turn_on_plot(fig, ax, curve, branch_path)


def load_predictions(exp_dir, stem="predictions"):
    """Load predictions from preferred Parquet storage or the CSV fallback."""
    parquet_path = os.path.join(exp_dir, f"{stem}.parquet")
    csv_path = os.path.join(exp_dir, f"{stem}.csv")

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


def discover_checkpoint_variants(exp_dir):
    """Return standard and optional secondary checkpoint prediction artifacts."""
    manifest_path = os.path.join(exp_dir, "checkpoint_selection.json")
    if not os.path.exists(manifest_path):
        return [(None, None, "predictions")]

    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)

    variants = []
    for method in manifest.get("methods", []):
        artifact = manifest.get("artifacts", {}).get(method, {})
        prediction_name = artifact.get("predictions", "")
        if not prediction_name:
            continue
        stem = os.path.splitext(prediction_name)[0]
        suffix = None if artifact.get("role") == "primary" else method
        variants.append((suffix, method, stem))
    return variants or [(None, None, "predictions")]


def _validation_record(exp_dir, checkpoint_method):
    if checkpoint_method is None:
        return None
    manifest_path = os.path.join(exp_dir, "checkpoint_selection.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    return (
        manifest.get("artifacts", {})
        .get(checkpoint_method, {})
        .get("best_validation_record")
    )


def evaluate_experiment(
    exp_dir,
    recalc,
    num_bins,
    pt_min,
    pt_max,
    bin_var,
    output_suffix=None,
    checkpoint_method=None,
    prediction_stem="predictions",
    classifier_override=None,
):
    print(f"\n{'=' * 40}")
    print(f"Evaluating Experiment: {exp_dir}")
    print(f"{'=' * 40}")

    if classifier_override is not None:
        classifier_suffix = classifier_override.name
        output_suffix = (
            f"{output_suffix}_{classifier_suffix}"
            if output_suffix
            else classifier_suffix
        )
    filename_suffix = f"_{output_suffix}" if output_suffix else ""
    metrics_path = os.path.join(exp_dir, f"metrics{filename_suffix}.json")

    if os.path.exists(metrics_path) and not recalc:
        print(f"  -> metrics.json already exists. Skipping (use --recalc to override).\n")
        return

    df = load_predictions(exp_dir, stem=prediction_stem)
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
    bkg_df = select_background_objects(df)

    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as source:
            experiment_config = json.load(source)
    else:
        experiment_config = {}
    classifier_config = (
        classifier_override
        if classifier_override is not None
        else parse_classifier(experiment_config)
    )
    validation_record = _validation_record(exp_dir, checkpoint_method)
    if classifier_config.tob_budget is not None:
        if not validation_record or "selected_tob_fpr" not in validation_record:
            raise ValueError(
                "Validation-search classifier is missing its selected TOB budget"
            )
        classifier_config = classifier_config.with_tob_fpr(
            validation_record["selected_tob_fpr"]
        )

    target_fake_rates = [0.005, 0.010, 0.020]
    thresholds = {}
    achieved_fake_rates = {}
    global_efficiencies = {}
    calibrations = {}
    calibration_sources = {}

    print("Calculating Thresholds and Global Efficiencies...")
    for fr in target_fake_rates:
        active_classifier = classifier_config.with_target_fpr(fr)
        # Recalibrate every configured classifier on the same test background
        # used for the TOB baseline, so the primary curves share one FPR policy.
        calibration = calibrate_classifier(
            bkg_df,
            bkg_df["nn_score"].to_numpy(dtype=np.float64),
            active_classifier,
        )
        ratio = calibration["achieved_fpr"]
        passed = classifier_object_pass_mask(sig_df, calibration)
        eff = float(passed.mean()) if len(sig_df) else 0.0
        threshold = calibration["nn_threshold"]

        fr_str = f"{fr * 100:.1f}%"
        calibrations[fr_str] = calibration
        calibration_sources[fr_str] = "test_recalibrated"
        thresholds[fr_str] = float(threshold)
        achieved_fake_rates[fr_str] = float(ratio)
        global_efficiencies[fr_str] = eff
        threshold_text = f"NN cut: {threshold:9.6g}"
        if classifier_config.name == "tob_nn_or":
            threshold_text += (
                f" | TOB cut: {calibration['tob_threshold_gev']:.4g} GeV"
            )
        print(
            f"  Target FPR: {fr_str:>4} | Achieved FPR: {ratio * 100:7.4f}% "
            f"| {threshold_text} | Signal Eff: {eff:.4f}"
        )
        background_event_count = calibration["diagnostics"][
            "background_event_count"
        ]
        if fr - ratio > (1.0 / background_event_count) + 1e-12:
            print(
                "    Note: score ties prevent a closer deterministic operating "
                "point without exceeding the target."
            )

    target_fr = "0.5%"
    working_calibration = calibrations[target_fr]

    # Dynamically generate equal-width bins
    eval_bins = np.linspace(pt_min, pt_max, num_bins + 1)
    bin_centers = (eval_bins[:-1] + eval_bins[1:]) / 2.0

    print(
        f"\nCalculating Binned Efficiencies at {target_fr} Fake Rate (using {num_bins} bins from {pt_min} to {pt_max} on '{bin_var}')...")
    signal_passed = classifier_object_pass_mask(
        sig_df, working_calibration
    )
    binned_effs, binned_effs_err = calculate_binned_mask_efficiencies(
        sig_df, signal_passed, eval_bins, bin_var
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
    baseline_global_efficiency = float(
        score_pass_mask(sig_df, "tob_pt", baseline_threshold).mean()
    ) if len(sig_df) else 0.0
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
        "checkpoint_method": checkpoint_method,
        "classifier": classifier_config.to_dict(),
        "calibration_split": "test_recalibrated",
        "classifier_calibrations": calibrations,
        "calibration_source_by_fake_rate": calibration_sources,
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
            "global_efficiency": baseline_global_efficiency,
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

    if classifier_config.name == "tob_nn_or":
        nn_pass = score_pass_mask(
            sig_df, "nn_score", working_calibration["nn_threshold"]
        )
        tob_values = tob_pt_gev(sig_df)
        tob_pass = np.isfinite(tob_values) & (
            tob_values >= working_calibration["tob_threshold_gev"]
        )
        nn_effs, nn_errs = calculate_binned_mask_efficiencies(
            sig_df, nn_pass, eval_bins, bin_var
        )
        tob_effs, tob_errs = calculate_binned_mask_efficiencies(
            sig_df, tob_pass, eval_bins, bin_var
        )
        metrics["classifier_branches"] = {
            "nn": {
                "label": "NN branch at hybrid cut",
                "binned_efficiencies": nn_effs,
                "binned_efficiencies_err": nn_errs,
            },
            "tob": {
                "label": (
                    f"TOB branch ({working_calibration['tob_fpr_budget'] * 100:.1f}% budget)"
                ),
                "binned_efficiencies": tob_effs,
                "binned_efficiencies_err": tob_errs,
            },
        }

    if validation_record and (
        "classifier_calibration" in validation_record
        or "threshold" in validation_record
    ):
        fixed_calibration = validation_record.get("classifier_calibration")
        if fixed_calibration is None:
            fixed_calibration = {
                "name": "nn_only",
                "nn_threshold": float(validation_record["threshold"]),
                "trigger_objects": int(
                    validation_record.get("trigger_objects", 2)
                ),
            }
        if fixed_calibration["name"] != classifier_config.name:
            fixed_calibration = None

    else:
        fixed_calibration = None

    if fixed_calibration is not None:
        fixed_test_event_pass = classifier_event_pass_mask(
            bkg_df, fixed_calibration
        )
        fixed_test_fpr = float(fixed_test_event_pass.mean())
        fixed_signal_pass = classifier_object_pass_mask(
            sig_df, fixed_calibration
        )
        fixed_signal_efficiency = (
            float(fixed_signal_pass.mean()) if len(sig_df) else 0.0
        )
        fixed_binned_effs, fixed_binned_errs = (
            calculate_binned_mask_efficiencies(
                sig_df,
                fixed_signal_pass,
                eval_bins,
                bin_var,
            )
        )
        metrics["validation_calibrated_operating_point"] = {
            "classifier_calibration": fixed_calibration,
            "validation": validation_record,
            "test_achieved_fake_rate": fixed_test_fpr,
            "test_signal_efficiency": fixed_signal_efficiency,
            "test_binned_efficiencies": fixed_binned_effs,
            "test_binned_efficiencies_err": fixed_binned_errs,
        }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    save_turn_on_plot(
        metrics,
        exp_dir,
        filename=f"turn_on_curve{filename_suffix}.png",
    )

    print(f"\nEvaluation complete! Hard numbers saved to {metrics_path}")


def generate_averaged_metrics(
    exp_name,
    folders,
    metrics_filename="metrics.json",
    output_filename="metrics_averaged.json",
):
    from scipy.stats import chi2

    all_metrics = []
    for folder in folders:
        with open(os.path.join(folder, metrics_filename), 'r') as f:
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
        baseline_global_values = [
            entry.get("global_efficiency") for entry in baseline_entries
        ]
        has_baseline_global = all(
            value is not None for value in baseline_global_values
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
        "checkpoint_method": base.get("checkpoint_method"),
        "classifier": base.get("classifier", {"name": "nn_only"}),
        "calibration_split": base.get("calibration_split"),
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
        if has_baseline_global:
            metrics_averaged["baseline_tob_pt"]["global_efficiency"] = float(
                np.mean(baseline_global_values)
            )

    branch_entries = [m.get("classifier_branches") for m in all_metrics]
    if all(entry is not None for entry in branch_entries):
        metrics_averaged["classifier_branches"] = {}
        for branch_name in branch_entries[0]:
            metrics_averaged["classifier_branches"][branch_name] = {
                "label": branch_entries[0][branch_name]["label"],
                "binned_efficiencies": np.mean(
                    [
                        entry[branch_name]["binned_efficiencies"]
                        for entry in branch_entries
                    ],
                    axis=0,
                ).tolist(),
                "binned_efficiencies_err": np.mean(
                    [
                        entry[branch_name]["binned_efficiencies_err"]
                        for entry in branch_entries
                    ],
                    axis=0,
                ).tolist(),
                "seeds_averaged": len(all_metrics),
            }

    for folder in folders:
        with open(os.path.join(folder, output_filename), 'w') as f:
            json.dump(metrics_averaged, f, indent=4)


def main():
    args = parse_args()

    classifier_override = None
    if args.classifier:
        override_config = {
            "classifier": {
                "name": args.classifier,
                "target_fpr": 0.005,
                "trigger_objects": 2,
            }
        }
        if args.classifier == "tob_nn_or":
            override_config["classifier"]["tob_fpr"] = (
                args.classifier_tob_fpr
            )
        classifier_override = parse_classifier(override_config)

    base_dir = os.path.abspath(args.experiments_dir)

    # 1. Standard evaluation sequence
    if args.experiment_dir:
        experiment_dir = os.path.abspath(args.experiment_dir)
        for suffix, method, stem in discover_checkpoint_variants(experiment_dir):
            evaluate_experiment(
                experiment_dir,
                args.recalc,
                args.num_bins,
                args.pt_min,
                args.pt_max,
                args.bin_var,
                output_suffix=suffix,
                checkpoint_method=method,
                prediction_stem=stem,
                classifier_override=classifier_override,
            )
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
                for suffix, method, stem in discover_checkpoint_variants(item_path):
                    evaluate_experiment(
                        item_path,
                        args.recalc,
                        args.num_bins,
                        args.pt_min,
                        args.pt_max,
                        args.bin_var,
                        output_suffix=suffix,
                        checkpoint_method=method,
                        prediction_stem=stem,
                        classifier_override=classifier_override,
                    )

    # 2. Always group and average the seeds afterward
    print(f"\n{'=' * 40}")
    print("Generating Averaged Metrics for multi-seed experiments...")
    print(f"{'=' * 40}")

    from collections import defaultdict
    experiment_groups = defaultdict(list)

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        config_path = os.path.join(item_path, "config.json")
        if os.path.isdir(item_path) and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            exp_name = config.get("experiment_name", "unknown")
            variants = discover_checkpoint_variants(item_path)
            for suffix, method, _ in variants:
                if classifier_override is not None:
                    suffix = (
                        f"{suffix}_{classifier_override.name}"
                        if suffix
                        else classifier_override.name
                    )
                filename_suffix = f"_{suffix}" if suffix else ""
                metrics_filename = f"metrics{filename_suffix}.json"
                metrics_path = os.path.join(item_path, metrics_filename)
                if not os.path.exists(metrics_path):
                    continue
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                if "achieved_fake_rates" not in metrics:
                    print(
                        "  -> Skipping legacy metrics without verified FPR: "
                        f"{metrics_path}"
                    )
                    continue
                group_key = (exp_name, suffix, method, metrics_filename)
                experiment_groups[group_key].append(item_path)

    for group_key, folders in experiment_groups.items():
        exp_name, suffix, method, metrics_filename = group_key
        display_name = exp_name if method is None else f"{exp_name} [{method}]"
        if len(folders) > 1:
            filename_suffix = f"_{suffix}" if suffix else ""
            generate_averaged_metrics(
                display_name,
                folders,
                metrics_filename=metrics_filename,
                output_filename=f"metrics_averaged{filename_suffix}.json",
            )
        else:
            print(f"  -> Skipping '{display_name}': Only 1 seed found.")

if __name__ == "__main__":
    main()
