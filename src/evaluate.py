import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import chi2

DEFAULT_FOLDER = "experiments\config batch 6 - simple stuff - fixed model weights to have validation"

def CalcThresh(DF, Criteria, FakeRate, objects=2):
    ### This funCalcThreshctions calculates what is the need threshold for "Criteria" columns in the bkg dataframe "DF",
    ###  such that the fakerate "FakeRate" is acquired for the passing of "objects" objects per event with the same id in "event_num" column.
    MaxVal = np.max(DF[Criteria].values)
    DF2 = DF.copy()
    DF[Criteria] /= MaxVal  ### normalize it to 1 to make the binary search quicker
    DF["signal"] = 0  # DROR: "Pretty sure this is useless, it's all BK so it should just be zero alerady"
    powers = 15  ### how many iterations in the binary tree
    threshold = 0.5
    for i in tqdm(range(2, powers)):
        TempDF = DF.copy()
        TempDF.loc[TempDF[Criteria] > threshold, "signal"] = 1
        TempDF = TempDF.groupby("eventNumber").sum().reset_index(drop=True)
        Denom = float(TempDF.shape[0])
        Numer = (TempDF["signal"] >= objects).sum()
        ratio = Numer / Denom
        if ratio > FakeRate:
            threshold += np.power(0.5, i)
        else:
            threshold -= np.power(0.5, i)

    return threshold * MaxVal, ratio

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tau Particle NN Predictions")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="Path to the experiment directory. If omitted, runs on all subfolders.")
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


def evaluate_experiment(exp_dir, recalc, num_bins, pt_min, pt_max, bin_var):
    print(f"\n{'=' * 40}")
    print(f"Evaluating Experiment: {exp_dir}")
    print(f"{'=' * 40}")

    metrics_path = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_path) and not recalc:
        print(f"  -> metrics.json already exists. Skipping (use --recalc to override).\n")
        return

    parquet_path = os.path.join(exp_dir, "predictions.parquet")
    if not os.path.exists(parquet_path):
        print(f"  -> Could not find predictions.parquet in {exp_dir}. Skipping.\n")
        return

    print(f"Loading predictions from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # CRITICAL FIX 1: Convert MeV to GeV for specific kinematic columns
    if 'tob_pt' in df.columns:
        df['tob_pt'] = df['tob_pt'] / 1000.0
    if 'truth_pt' in df.columns:
        df['truth_pt'] = df['truth_pt'] / 1000.0

    # Ensure the requested bin_var actually exists in the dataframe
    if bin_var not in df.columns:
        print(f"  -> Error: Column '{bin_var}' not found in predictions. parquet. Skipping.\n")
        return

    sig_df = df[df['Type'] == 'Signal']
    bkg_df = df[df['Type'] == 'BKG']

    target_fake_rates = [0.005, 0.010, 0.020]
    thresholds = {}
    global_efficiencies = {}

    print("Calculating Thresholds and Global Efficiencies...")
    for fr in target_fake_rates:
        # threshold = np.percentile(bkg_df['nn_score'], (1.0 - fr) * 100) # old STUPID gemini code. Sorry gembros
        threshold, ratio = CalcThresh(bkg_df, 'nn_score', fr)
        sig_passed = len(sig_df[sig_df['nn_score'] >= threshold])
        eff = sig_passed / len(sig_df) if len(sig_df) > 0 else 0.0

        fr_str = f"{fr * 100:.1f}%"
        thresholds[fr_str] = float(threshold)
        global_efficiencies[fr_str] = float(eff)
        print(f"  Fake Rate: {fr_str:>4} -> Cut: {threshold:7.4f} | Signal Eff: {eff:.4f}")

    target_fr = "0.5%"
    working_threshold = thresholds[target_fr]

    # Dynamically generate equal-width bins
    eval_bins = np.linspace(pt_min, pt_max, num_bins + 1)
    bin_centers = (eval_bins[:-1] + eval_bins[1:]) / 2.0

    binned_effs = [] # y value
    binned_effs_err = [] # dy value

    print(
        f"\nCalculating Binned Efficiencies at {target_fr} Fake Rate (using {num_bins} bins from {pt_min} to {pt_max} on '{bin_var}')...")
    for i in range(len(eval_bins) - 1):
        b_min = eval_bins[i]
        b_max = eval_bins[i + 1]

        # Use dynamic variable for slicing
        bin_sig = sig_df[(sig_df[bin_var] >= b_min) & (sig_df[bin_var] < b_max)]
        denom = len(bin_sig) # count of data points in bin

        if denom > 0:
            passed = len(bin_sig[bin_sig['nn_score'] >= working_threshold])
            eff = passed / denom
            err = np.sqrt(eff * (1.0 - eff) / denom) # binomial error, standard for bins
            if err == 0.0: # safeguard against unlucky zero error
                err = 1e-5
        else:
            eff = 0.0
            err = 1 # massive error so fit ignores

        binned_effs.append(float(eff))
        binned_effs_err.append(float(err))

    print("Fitting Fermi-Dirac Function...")
    try:
        p0 = [1.0, 0.1, 40.0]
        bounds = ([0.0, 0.0, 0.0], [1.0, np.inf, 300.0])
        popt, _ = curve_fit(fermi_dirac, bin_centers, binned_effs, p0=p0, bounds=bounds, sigma=binned_effs_err,
                            absolute_sigma=True)

        fd_plateau, fd_slope, fd_midpoint = popt

        # --- STATISTICAL TESTS ---
        y_obs = np.array(binned_effs)
        y_err = np.array(binned_effs_err)
        y_fit = fermi_dirac(bin_centers, fd_plateau, fd_slope, fd_midpoint)

        # Only compute chi-squared on valid bins (ignoring empty bins where we forced err=1)
        valid = y_err < 1.0
        dof = np.sum(valid) - len(popt)

        if dof > 0:
            chi2_val = np.sum(((y_obs[valid] - y_fit[valid]) / y_err[valid]) ** 2)
            chi2_red = chi2_val / dof
            p_value = chi2.sf(chi2_val, dof)
        else:
            chi2_red, p_value = 0.0, 0.0

        fit_success = True
        print(f"  Midpoint: {fd_midpoint:.2f} | Slope: {fd_slope:.4f} | Plateau: {fd_plateau:.4f}")
        print(f"  Fit Quality -> Chi2/ndf: {chi2_red:.3f} | p-value: {p_value:.4e}")

    except Exception as e:
        print(f"  Curve fitting failed: {e}")
        fd_plateau, fd_slope, fd_midpoint, chi2_red, p_value = 0.0, 0.0, 0.0, 0.0, 0.0
        fit_success = False

    metrics = {
        "thresholds_by_fake_rate": thresholds,
        "global_efficiency": global_efficiencies,
        "turn_on_curve": {
            "binning_variable": bin_var,
            "bins": eval_bins.tolist(),
            "binned_efficiencies": binned_effs,
            "binned_efficiencies_err": binned_effs_err,
            "target_fake_rate_used": target_fr,
            "fit_success": fit_success,
            "fermi_dirac_fit": {
                "plateau": float(fd_plateau),
                "slope": float(fd_slope),
                "midpoint": float(fd_midpoint),
                "chi2_red": float(chi2_red),
                "p_value": float(p_value)
            }
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

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
    avg_global_effs = {k: np.mean([m["global_efficiency"][k] for m in all_metrics]) for k in base["global_efficiency"]}

    # Average the bins
    avg_binned_effs = np.mean([m["turn_on_curve"]["binned_efficiencies"] for m in all_metrics], axis=0)
    avg_binned_errs = np.mean([m["turn_on_curve"]["binned_efficiencies_err"] for m in all_metrics], axis=0)

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

    for folder in folders:
        with open(os.path.join(folder, "metrics_averaged.json"), 'w') as f:
            json.dump(metrics_averaged, f, indent=4)


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "..", DEFAULT_FOLDER))

    # 1. Standard evaluation sequence
    if args.experiment_dir:
        evaluate_experiment(args.experiment_dir, args.recalc, args.num_bins, args.pt_min, args.pt_max, args.bin_var)
    else:
        print(f"No --experiment_dir provided. Scanning '{base_dir}' for experiment folders...")
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "predictions.parquet")):
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
                exp_name = config.get("experiment_name", "unknown")
                experiment_groups[exp_name].append(item_path)

    for exp_name, folders in experiment_groups.items():
        if len(folders) > 1:
            generate_averaged_metrics(exp_name, folders)
        else:
            print(f"  -> Skipping '{exp_name}': Only 1 seed found.")

if __name__ == "__main__":
    main()