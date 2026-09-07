"""Score all configured checkpoints and run cross-fitted validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .pair_evaluate import build_selection_report, evaluate_prediction_table
    from .pair_predict import run_prediction
except ImportError:
    from pair_evaluate import build_selection_report, evaluate_prediction_table
    from pair_predict import run_prediction


def configured_runs(config_dir, experiments_dir):
    """Resolve one experiment directory for each per-seed config."""
    entries = []
    for config_path in sorted(Path(config_dir).glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "seed" not in config:
            raise KeyError(f"{config_path} does not define one seed")
        run_dir = (
            Path(experiments_dir)
            / config["experiment_name"]
            / f"seed_{int(config['seed'])}"
        )
        entries.append((config_path, config, run_dir))
    if not entries:
        raise ValueError("config directory contains no JSON files")
    return entries


def validate_config_family(
    config_dir,
    pair_data,
    experiments_dir,
    output_dir,
    *,
    target_fpr=0.005,
):
    """Evaluate every checkpoint for every configured seed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    event_tables = []
    for config_path, config, run_dir in configured_runs(
        config_dir,
        experiments_dir,
    ):
        normalizer_path = run_dir / "normalizer.npz"
        checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
        expected = int(config.get("epochs", 20))
        if len(checkpoints) != expected:
            raise ValueError(
                f"{run_dir} contains {len(checkpoints)} checkpoints; expected {expected}"
            )
        seed = int(config["seed"])
        seed_output = output_dir / f"seed_{seed}"
        seed_output.mkdir()
        for checkpoint in checkpoints:
            checkpoint_id = checkpoint.stem
            pairs, events = run_prediction(
                config_path,
                pair_data,
                normalizer_path,
                checkpoint,
            )
            pairs["model_seed"] = seed
            pairs["checkpoint_id"] = checkpoint_id
            events["model_seed"] = seed
            events["checkpoint_id"] = checkpoint_id
            checkpoint_output = seed_output / checkpoint_id
            checkpoint_output.mkdir()
            pairs.to_csv(
                checkpoint_output / "pair_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            events.to_csv(
                checkpoint_output / "event_predictions.csv.gz",
                index=False,
                compression="gzip",
            )
            event_tables.append(events)

    predictions = pd.concat(event_tables, ignore_index=True)
    checkpoint_summary = evaluate_prediction_table(predictions, target_fpr)
    selection_report = build_selection_report(
        predictions,
        checkpoint_summary,
        target_fpr,
    )
    checkpoint_summary.to_csv(output_dir / "checkpoint_summary.csv", index=False)
    (output_dir / "selection_report.json").write_text(
        json.dumps(selection_report, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--pair_data", required=True)
    parser.add_argument("--experiments_dir", default="experiments/pair_models")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_fpr", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_config_family(
        args.config_dir,
        args.pair_data,
        args.experiments_dir,
        args.output_dir,
        target_fpr=args.target_fpr,
    )


if __name__ == "__main__":
    main()
