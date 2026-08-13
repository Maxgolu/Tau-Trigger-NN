import argparse
import json
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TauNet training configurations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory in which config JSON files are created.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate one lightweight CPU configuration instead of the full sweep.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        default=None,
        help=(
            "Comma-separated feature names. Repeat this argument to generate "
            "multiple feature combinations."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds for every requested feature set.",
    )
    parser.add_argument(
        "--checkpoint-method",
        action="append",
        choices=["validation_bce", "target_fpr"],
        default=None,
        help=(
            "Checkpoint selector. Repeat to retain both validation BCE and "
            "target-FPR checkpoints."
        ),
    )
    parser.add_argument(
        "--checkpoint-primary",
        choices=["validation_bce", "target_fpr"],
        default=None,
        help="Primary artifact method when both checkpoint selectors are used.",
    )
    parser.add_argument(
        "--checkpoint-target-fpr",
        type=float,
        default=0.005,
        help="Validation event FPR used by target_fpr selection (default: 0.005).",
    )
    parser.add_argument(
        "--classifier",
        choices=["nn_only", "tob_nn_or"],
        default=None,
        help="Final trigger classifier (default: legacy nn_only).",
    )
    parser.add_argument(
        "--classifier-target-fpr",
        type=float,
        default=0.005,
        help="Total event FPR for the classifier (default: 0.005).",
    )
    parser.add_argument(
        "--classifier-tob-fpr",
        type=float,
        default=0.004,
        help="TOB branch event-FPR budget for tob_nn_or (default: 0.004).",
    )
    parser.add_argument(
        "--classifier-tob-budget-mode",
        choices=["fixed", "validation_search"],
        default="fixed",
        help="Use a fixed TOB budget or select it on validation data.",
    )
    parser.add_argument(
        "--classifier-tob-budget-values",
        nargs="+",
        type=float,
        default=[0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025,
                 0.003, 0.0035, 0.004, 0.0045, 0.005],
        help="Candidate TOB event-FPR budgets for validation search.",
    )
    parser.add_argument(
        "--classifier-tob-budget-folds",
        type=int,
        default=2,
        help="Validation cross-fitting folds for TOB-budget search (default: 2).",
    )
    parser.add_argument(
        "--classifier-objective-min-pt",
        type=float,
        default=25.0,
        help="Lowest protected truth-pT value in GeV (default: 25).",
    )
    parser.add_argument(
        "--classifier-objective-max-pt",
        type=float,
        default=100.0,
        help="Upper truth-pT edge included in the mean objective (default: 100).",
    )
    parser.add_argument(
        "--classifier-objective-window-width",
        type=float,
        default=5.0,
        help="Truth-pT comparison-window width in GeV (default: 5).",
    )
    parser.add_argument(
        "--classifier-noninferiority-tolerance",
        type=float,
        default=0.005,
        help="Allowed efficiency deficit per protected window (default: 0.005).",
    )
    parser.add_argument(
        "--classifier-objective-tie-tolerance",
        type=float,
        default=0.002,
        help="Objective difference treated as a tie (default: 0.002).",
    )
    parser.add_argument(
        "--loss",
        choices=["bce"],
        default=None,
        help="Training loss (default: legacy bce).",
    )
    return parser.parse_args()


def write_config(config, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    return filepath


def generate_smoke_config(output_dir=DEFAULT_CONFIG_DIR):
    """Create one intentionally small end-to-end CPU test configuration."""
    config = {
        "run_id": "smoke_s42",
        "experiment_name": "Smoke_CPU_tob_pt",
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_layers": [16, 8],
        "features_to_use": ["tob_pt_only"],
        "epochs": 2,
        "seed": 42,
        "max_events_per_class": 5000,
    }
    filepath = write_config(config, Path(output_dir), "smoke_s42.json")
    print(f"Generated smoke-test configuration: {filepath}")


def generate_sweep_configs(
    output_dir=DEFAULT_CONFIG_DIR,
    feature_sets=None,
    seeds=None,
    checkpoint_selection=None,
    classifier=None,
    loss=None,
):
    base_config = {"epochs": 20}
    learning_rates = [0.001]
    batch_sizes = [256]
    architectures = [[32, 16]]
    if feature_sets is None:
        feature_sets = [
            ["tob_pt_only"],
            ["em2_3x3_maxdist", "tob_pt_only"],
            ["em2_3x3_dominance", "tob_pt_only"],
            ["em2_3x3_dominance", "em2_3x3_maxdist"],
            ["em2_3x3_dominance", "em2_3x3_maxdist", "tob_pt_only"],
        ]
    if seeds is None:
        seeds = [42, 123, 456]

    output_dir = Path(output_dir)
    count = 0
    combinations = product(learning_rates, batch_sizes, architectures, feature_sets)
    for config_number, (lr, bs, arch, features) in enumerate(combinations, start=1):
        features_str = "_".join(features)
        arch_str = "x".join(map(str, arch))
        experiment_name = f"TauNet_lr{lr}_bs{bs}_arch{arch_str}_{features_str}"

        for seed in seeds:
            config = base_config.copy()
            config.update(
                {
                    # Short IDs are used only for paths. Full metadata remains below.
                    "run_id": f"c{config_number:03d}_s{seed}",
                    "experiment_name": experiment_name,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "hidden_layers": arch,
                    "features_to_use": features,
                    "seed": seed,
                }
            )
            if checkpoint_selection is not None:
                config["checkpoint_selection"] = checkpoint_selection
            if classifier is not None:
                config["classifier"] = classifier
            if loss is not None:
                config["loss"] = loss
            filename = f"c{config_number:03d}_s{seed}.json"
            write_config(config, output_dir, filename)
            count += 1

    print(f"Successfully generated {count} configuration files in '{output_dir}'")


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        generate_smoke_config(args.output_dir)
    else:
        requested_feature_sets = None
        if args.feature_set:
            requested_feature_sets = [
                [name.strip() for name in feature_set.split(",") if name.strip()]
                for feature_set in args.feature_set
            ]
            if any(not feature_set for feature_set in requested_feature_sets):
                raise ValueError("Each --feature-set must contain at least one feature name.")

        checkpoint_selection = None
        if args.checkpoint_method:
            primary = args.checkpoint_primary or args.checkpoint_method[0]
            if primary not in args.checkpoint_method:
                raise ValueError(
                    "--checkpoint-primary must also be supplied through "
                    "--checkpoint-method"
                )
            checkpoint_selection = {
                "methods": args.checkpoint_method,
                "primary_method": primary,
                "target_fpr": args.checkpoint_target_fpr,
            }

        classifier = None
        if args.classifier:
            classifier = {
                "name": args.classifier,
                "target_fpr": args.classifier_target_fpr,
                "trigger_objects": 2,
            }
            if args.classifier == "tob_nn_or":
                if args.classifier_tob_budget_mode == "fixed":
                    classifier["tob_fpr"] = args.classifier_tob_fpr
                else:
                    classifier["tob_budget"] = {
                        "mode": "validation_search",
                        "values": args.classifier_tob_budget_values,
                        "cross_validation_folds": args.classifier_tob_budget_folds,
                        "objective": {
                            "min_truth_pt_gev": args.classifier_objective_min_pt,
                            "objective_max_truth_pt_gev": args.classifier_objective_max_pt,
                            "window_width_gev": args.classifier_objective_window_width,
                            "noninferiority_tolerance": args.classifier_noninferiority_tolerance,
                            "objective_tie_tolerance": args.classifier_objective_tie_tolerance,
                        },
                    }
            elif args.classifier_tob_budget_mode != "fixed":
                raise ValueError(
                    "TOB-budget search requires --classifier tob_nn_or"
                )

        loss = {"name": args.loss} if args.loss else None

        generate_sweep_configs(
            output_dir=args.output_dir,
            feature_sets=requested_feature_sets,
            seeds=args.seeds,
            checkpoint_selection=checkpoint_selection,
            classifier=classifier,
            loss=loss,
        )
