import argparse
import json
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import copy
from pathlib import Path

from model import DynamicMLP
from tracker import ExperimentTracker
from training_data import TrainingDataCache


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tau Particle NN")
    # Make --config optional
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a specific config JSON file")
    # Add a default directory to scan
    parser.add_argument("--configs_dir", type=str, default=str(DEFAULT_CONFIGS_DIR),
                        help="Directory containing config files to run if --config is not set")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Project directory containing Signal/ and Background/")
    parser.add_argument("--experiments_dir", type=str, default=str(DEFAULT_EXPERIMENTS_DIR),
                        help="Directory in which experiment outputs are stored")
    parser.add_argument("--max_events_per_class", type=int, default=None,
                        help="Optional CPU smoke-test limit for Signal and Background events")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Optional command-line override for the configured epoch count")
    parser.add_argument("--disable_data_cache", action="store_true",
                        help="Disable aligned-data, split, and raw-feature reuse between configs")
    parser.add_argument("--feature_cache_mb", type=int, default=512,
                        help="Maximum memory used by cached raw features (default: 512 MB)")
    # Add the force flag
    parser.add_argument("--force_redo", action="store_true",
                        help="Force training even if experiment with same name and seed exists")
    return parser.parse_args()


def get_completed_runs(base_dir="experiments"):
    """
    Scans the experiments directory and returns a set of tuples:
    (experiment_name, seed) for all runs that have already been executed.
    """
    completed = set()
    if not os.path.exists(base_dir):
        return completed

    for folder in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, folder)
        config_path = os.path.join(run_dir, "config.json")
        has_predictions = (
            os.path.exists(os.path.join(run_dir, "predictions.parquet"))
            or os.path.exists(os.path.join(run_dir, "predictions.csv"))
        )
        if os.path.exists(config_path) and has_predictions:
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    # We need both name and seed to uniquely identify a completed run
                    exp_name = cfg.get("experiment_name", "")
                    seed = cfg.get("seed", 42)
                    completed.add((exp_name, seed))
            except Exception as e:
                print(f"Warning: Could not read {config_path} - {e}")

    return completed


def set_random_seeds(seed):
    """Reset model and DataLoader randomness before every experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training_pipeline(config_path, data_dir=DEFAULT_DATA_DIR,
                          experiments_dir=DEFAULT_EXPERIMENTS_DIR,
                          max_events_per_class=None, epochs_override=None,
                          data_cache=None):
    """
    This function wraps all data loading, model init, and training loops.
    """

    # Load Configuration & Tracker
    with open(config_path, 'r') as f:
        config = json.load(f)

    if max_events_per_class is not None:
        config["max_events_per_class"] = max_events_per_class
    if epochs_override is not None:
        config["epochs"] = epochs_override

    # The shared cache is supplied by main() when a config directory is used.
    if data_cache is None:
        data_cache = TrainingDataCache(enabled=False)

    seed = config.get("seed", 42)
    print(f"\n[{config.get('experiment_name')}] Locking random seed to: {seed}")

    # =====================================================================
    # 2. PHYSICS DATA PREPARATION
    # =====================================================================
    max_events = config.get("max_events_per_class")
    dataset = data_cache.get_dataset(
        data_dir=data_dir,
        max_events_per_class=max_events,
        seed=seed,
    )
    split = data_cache.get_split(dataset, seed=seed)

    # Create the output folder only after data preparation succeeds.
    tracker = ExperimentTracker(
        config,
        base_dir=str(Path(experiments_dir).resolve()),
    )
    print(f"Started experiment: {tracker.experiment_dir}")
    print(
        f"Data split ready! Train: {len(split.train)} | "
        f"Val: {len(split.validation)} | Test: {len(split.test)}"
    )

    # =====================================================================
    # 4. FEATURE ASSEMBLY & SCALING
    # =====================================================================
    print(f"Assembling features: {config['features_to_use']}")
    feature_names = config["features_to_use"]
    X_train_raw = data_cache.assemble_features(
        dataset, split.train, feature_names
    )
    X_val_raw = data_cache.assemble_features(
        dataset, split.validation, feature_names
    )
    X_test_raw = data_cache.assemble_features(
        dataset, split.test, feature_names
    )

    # EXACT Normalization from your notebook
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    X_train_np = (X_train_raw - mean) / std
    X_val_np = (X_val_raw - mean) / std
    X_test_np = (X_test_raw - mean) / std

    y_train_np = dataset.labels[split.train]
    y_val_np = dataset.labels[split.validation]
    y_test_np = dataset.labels[split.test]

    # Experiment outputs must never mutate the shared cached DataFrame.
    df_test_meta = (
        dataset.frame.iloc[split.test].copy().reset_index(drop=True)
    )

    # =====================================================================
    # 5. PYTORCH TRAINING
    # =====================================================================
    set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Input Dimension: {X_train_np.shape[1]}")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    batch_size = config.get("batch_size", 1024)  # updated to your notebook's default 1024
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = DynamicMLP(
        input_dim=X_train_np.shape[1],
        hidden_layers=config.get("hidden_layers", [32, 16])  # Matches your MLP logic
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    epochs = config.get("epochs", 20)
    print(f"Starting training for {epochs} epochs...")

    # 1. Initialize the tracker variable before the loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # -- TRAINING PHASE --
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # -- VALIDATION PHASE --
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss = criterion(outputs, y_batch)
                running_val_loss += val_loss.item() * X_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Print both metrics to monitor for overfitting
        print(f"Epoch {epoch + 1:02d}/{epochs} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            print(f"  --> Validation loss decreased ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving model!")
            best_val_loss = epoch_val_loss

            # Save the PyTorch weights ONLY on the best epoch
            tracker.save_weights(model)
            best_weights = copy.deepcopy(model.state_dict())

    # =====================================================================
    # 6. EVALUATION & TRACKING
    # =====================================================================
    model.load_state_dict(best_weights)
    model.eval()
    scores = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            scores.extend(predictions.cpu().numpy().flatten())

    df_test_meta["nn_score"] = scores

    # Grab all columns required for both object-level kinematics AND event-level grouping
    core_columns = [
        'eventNumber', 'tob_index', 'signal', 'Type',
        'truth_pt', 'tob_pt', 'tob_eta', 'tob_phi', 'nn_score'
    ]
    df_eval = df_test_meta[core_columns].copy()

    # Save everything to the fast Parquet format
    predictions_path = tracker.save_predictions(df_eval)
    print(f"--> Saved rich physics predictions to: {predictions_path}")

    # =====================================================================
    # OPTIONAL: BACKWARD COMPATIBILITY EXPORT (For Old TOC_1D Notebook)
    # Uncomment the block below to save a heavy CSV formatted exactly
    # for your original MakeTOC and CalcThresh functions.
    # =====================================================================

    # Grab the exact columns your old notebook expects
    legacy_columns = [
        'eventNumber', 'tob_index', 'signal', 'Type',
        'truth_pt', 'tob_pt', 'tob_eta', 'tob_phi', 'nn_score'
    ]
    if config.get("save_legacy_csv", False):
        df_legacy = df_test_meta[legacy_columns].copy()
        df_legacy['Type'] = df_legacy['Type'].replace('Background', 'BKG')
        csv_path = os.path.join(tracker.experiment_dir, "legacy_results_with_scores.csv")
        df_legacy.to_csv(csv_path, index=False)
        print(f"--> Saved backward-compatible CSV to: {csv_path}")

    # =====================================================================

    print(f"Experiment '{config['experiment_name']}' complete! All files saved to {tracker.experiment_dir}")


def main():
    args = parse_args()

    # One shared cache serves every configuration in this Python process.
    data_cache = TrainingDataCache(
        enabled=not args.disable_data_cache,
        feature_cache_mb=args.feature_cache_mb,
    )

    # 1. Determine which configs to run
    configs_to_run = []
    if args.config:
        # User specified a single file
        configs_to_run.append(args.config)
    else:
        # User ran without args, scan the configs directory
        if not os.path.exists(args.configs_dir):
            print(f"Directory '{args.configs_dir}' not found. Create it or use --configs_dir")
            return

        configs_to_run = glob.glob(os.path.join(args.configs_dir, "*.json"))
        if not configs_to_run:
            print(f"No JSON files found in '{args.configs_dir}'.")
            return

    # 2. Map what has already been done
    completed_runs = get_completed_runs(args.experiments_dir) if not args.force_redo else set()

    # 3. Execute the loop
    for config_path in sorted(configs_to_run):
        # We briefly open the config just to check its name and seed
        with open(config_path, 'r') as f:
            config = json.load(f)

        exp_name = config.get("experiment_name")
        seed = config.get("seed", 42)

        # Check if this exact architecture AND seed has already been run
        if not args.force_redo and (exp_name, seed) in completed_runs:
            print(f"Skipping {os.path.basename(config_path)}: Run '{exp_name}' with seed {seed} already exists.")
            continue

        print(f"\n{'=' * 60}\nExecuting {config_path}\n{'=' * 60}")
        run_training_pipeline(
            config_path,
            data_dir=args.data_dir,
            experiments_dir=args.experiments_dir,
            max_events_per_class=args.max_events_per_class,
            epochs_override=args.epochs,
            data_cache=data_cache,
        )

    print(f"\n{data_cache.summary()}")


if __name__ == "__main__":
    main()
