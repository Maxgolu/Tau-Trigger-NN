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


def generate_sweep_configs(output_dir=DEFAULT_CONFIG_DIR):
    base_config = {"epochs": 20}
    learning_rates = [0.001]
    batch_sizes = [256]
    architectures = [[32, 16]]
    feature_sets = [
        ["tob_pt_only"],
        ["em2_3x3_maxdist", "tob_pt_only"],
        ["em2_3x3_dominance", "tob_pt_only"],
        ["em2_3x3_dominance", "em2_3x3_maxdist"],
        ["em2_3x3_dominance", "em2_3x3_maxdist", "tob_pt_only"],
    ]
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
            filename = f"c{config_number:03d}_s{seed}.json"
            write_config(config, output_dir, filename)
            count += 1

    print(f"Successfully generated {count} configuration files in '{output_dir}'")


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        generate_smoke_config(args.output_dir)
    else:
        generate_sweep_configs(args.output_dir)
