import os
import json
from itertools import product


def generate_sweep_configs():
    # 1. Define your base static architecture
    base_config = {
        "epochs": 20,
        # hidden_layers removed from here to be dynamically generated
    }

    # 2. Define the hyperparameter grids you want to test
    learning_rates = [0.001]
    batch_sizes = [256]

    # ARCHITECTURE: Enforced to only test the base (32x16) model
    architectures = [
        [32, 16]
    ]

    #########################################################
    #       DYNAMIC FEATURE COMBINATION GENERATION          #
    #########################################################

    feature_sets = [
        # 1. tob_pt by itself
        ["tob_pt_only"],

        # 2. 3x3 maxdist + tob_pt
        ["em2_3x3_maxdist", "tob_pt_only"],

        # 3. EM2 dominance + tob_pt
        ["em2_3x3_dominance", "tob_pt_only"],

        # 4. EM2 dominance + 3x3 maxdist
        ["em2_3x3_dominance", "em2_3x3_maxdist"],

        # 5. EM2 dominance + 3x3 maxdist + tob_pt
        ["em2_3x3_dominance", "em2_3x3_maxdist", "tob_pt_only"]
    ]

    #########################################################
    ##################### STOP EDITING ######################
    #########################################################

    # 3. Define the seeds for statistical averaging
    seeds = [42, 123, 456]

    # Create output directory
    output_dir = "configs/"
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    # 4. Loop through every combination, now including architectures
    for lr, bs, arch, features, seed in product(learning_rates, batch_sizes, architectures, feature_sets, seeds):
        # Join all feature names together with an underscore for the filename
        features_str = "_".join(features)

        # Create a string representation of the architecture for the filename
        # e.g., [32, 16] -> "32x16"
        arch_str = "x".join(map(str, arch))

        # Build a descriptive experiment name (OMIT THE SEED HERE)
        exp_name = f"TauNet_lr{lr}_bs{bs}_arch{arch_str}_{features_str}"

        # Assemble the final dictionary
        config = base_config.copy()
        config["experiment_name"] = exp_name
        config["learning_rate"] = lr
        config["batch_size"] = bs
        config["hidden_layers"] = arch  # Dynamically set the architecture
        config["features_to_use"] = features
        config["seed"] = seed

        # Save to a uniquely named file so we don't overwrite the different seeds
        filename = f"{exp_name}_seed{seed}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

        count += 1

    print(f"Successfully generated {count} configuration files in '{output_dir}/'")


if __name__ == "__main__":
    generate_sweep_configs()