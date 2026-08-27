"""Predict-only evaluation of a trained run on an external dataset.

Loads a finished run (``config.json`` + ``model_weights.pt``), reproduces the
training preprocessing exactly (branch input transforms, then per-column
z-scoring fitted on the ORIGINAL training split for the run's seed — never on
the external data), predicts on every object of an external ``--data_dir``,
and evaluates with the standard machinery: thresholds are recalibrated on the
external background at the requested fake rates, and the ``tob_pt`` baseline
is recalibrated on the same background for comparison.

Results are written into ``<run_dir>/external_<tag>/`` as a standard
run-shaped directory (``config.json``, ``predictions.parquet``,
``metrics.json``, ``turn_on_curve.png``), so every existing analysis tool
works on it unchanged. The default training path is not affected in any way:
this script only reads existing artifacts.

Example:
    python src/evaluate_external.py \
        --run_dir experiments/cnn_v10_dual_em2_frac/run_v10_dual_em2_frac_s42_20260826_142216 \
        --data_dir data_new
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_experiment  # noqa: E402
from model import build_model  # noqa: E402
from training_data import TrainingDataCache  # noqa: E402

CORE_COLUMNS = [
    "eventNumber", "tob_index", "signal", "Type",
    "truth_pt", "tob_pt", "tob_eta", "tob_phi", "nn_score",
]


def build_feature_layout(data_cache, dataset, indices, feature_names):
    layout = []
    offset = 0
    for name in feature_names:
        width = data_cache.assemble_features(
            dataset, indices[:1], [name]
        ).shape[1]
        layout.append((name, offset, width))
        offset += width
    return layout


def apply_branch_transforms(X, config, feature_layout):
    """Apply per-branch input transforms in place, exactly like train.py."""
    branch_transforms = {}
    model_config = config.get("model")
    if model_config is not None and model_config.get("name") == "tensor_cnn":
        for branch in model_config.get("branches", []):
            branch_transforms[branch["feature"]] = branch.get(
                "transform", "none"
            )
    for name, start, width in feature_layout:
        transform = branch_transforms.get(name, "none")
        if transform in ("none", None):
            continue
        if transform != "log1p":
            raise ValueError(f"Unknown branch transform: {transform}")
        block = X[:, start:start + width]
        X[:, start:start + width] = np.sign(block) * np.log1p(np.abs(block))
    return X


def fit_reference_preprocessing(config, train_data_dir, data_cache):
    """Fit transforms + mean/std on the run's original training split."""
    seed = int(config.get("seed", 42))
    dataset = data_cache.get_dataset(
        data_dir=train_data_dir,
        max_events_per_class=config.get("max_events_per_class"),
        seed=seed,
    )
    split = data_cache.get_split(dataset, seed=seed)
    feature_names = config["features_to_use"]
    layout = build_feature_layout(
        data_cache, dataset, split.train, feature_names
    )
    X_train = data_cache.assemble_features(
        dataset, split.train, feature_names
    ).astype(np.float64, copy=True)
    apply_branch_transforms(X_train, config, layout)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return layout, mean, std


def predict_external(run_dir, data_dir, train_data_dir, weights_name,
                     batch_size=8192):
    run_path = Path(run_dir)
    with open(run_path / "config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    data_cache = TrainingDataCache(enabled=True)

    print("Fitting reference preprocessing on the original training split...")
    layout, mean, std = fit_reference_preprocessing(
        config, train_data_dir, data_cache
    )

    print(f"Loading external dataset: {data_dir}")
    seed = int(config.get("seed", 42))
    external = data_cache.get_dataset(
        data_dir=data_dir, max_events_per_class=None, seed=seed
    )
    indices = np.arange(len(external.frame))
    X = data_cache.assemble_features(
        external, indices, config["features_to_use"]
    ).astype(np.float64, copy=True)
    apply_branch_transforms(X, config, layout)
    X = (X - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, X.shape[1], layout).to(device)
    state = torch.load(run_path / weights_name, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scores = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(
                X[start:start + batch_size], dtype=torch.float32
            ).to(device)
            scores.append(model(batch).cpu().numpy().reshape(-1))
    scores = np.concatenate(scores)

    columns = [c for c in CORE_COLUMNS if c != "nn_score"]
    if "case" in external.frame.columns:
        columns = columns + ["case"]
    df_eval = external.frame[columns].copy().reset_index(drop=True)
    df_eval["nn_score"] = scores
    return config, df_eval


def save_outputs(run_dir, tag, config, df_eval, data_dir, train_data_dir,
                 weights_name):
    out_dir = Path(run_dir) / f"external_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = dict(config)
    annotated["external_evaluation"] = {
        "data_dir": str(data_dir),
        "train_data_dir": str(train_data_dir),
        "source_run": str(run_dir),
        "weights": weights_name,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(annotated, handle, indent=2)
    try:
        df_eval.to_parquet(out_dir / "predictions.parquet", index=False)
    except (ImportError, OSError):
        df_eval.to_csv(out_dir / "predictions.csv", index=False)
    print(f"External predictions saved under: {out_dir}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained run on an external data_dir"
    )
    parser.add_argument("--run_dir", required=True,
                        help="Finished run directory with config + weights")
    parser.add_argument("--data_dir", required=True,
                        help="External data_dir (Signal/ + Background/)")
    parser.add_argument("--train_data_dir", default=".",
                        help="Original data_dir the run was trained on")
    parser.add_argument("--tag", default=None,
                        help="Output suffix (default: external data_dir name)")
    parser.add_argument("--weights", default="model_weights.pt")
    parser.add_argument("--num_bins", type=int, default=44)
    parser.add_argument("--pt_min", type=float, default=10.0)
    parser.add_argument("--pt_max", type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.tag or Path(args.data_dir).name
    config, df_eval = predict_external(
        args.run_dir, args.data_dir, args.train_data_dir, args.weights
    )
    out_dir = save_outputs(
        args.run_dir, tag, config, df_eval,
        args.data_dir, args.train_data_dir, args.weights,
    )
    evaluate_experiment(
        str(out_dir),
        recalc=True,
        num_bins=args.num_bins,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        bin_var="truth_pt",
    )


if __name__ == "__main__":
    main()
