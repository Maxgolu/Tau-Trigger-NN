import argparse
import json
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import glob
import copy

from features import FEATURE_REGISTRY
from model import DynamicMLP
from tracker import ExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tau Particle NN")
    # Make --config optional
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a specific config JSON file")
    # Add a default directory to scan
    parser.add_argument("--configs_dir", type=str, default="configs",
                        help="Directory containing config files to run if --config is not set")
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
        config_path = os.path.join(base_dir, folder, "config.json")
        if os.path.exists(config_path):
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


def run_training_pipeline(config_path):
    """
    This function wraps all data loading, model init, and training loops.
    """

    # Load Configuration & Tracker
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Lock the seed
    seed = config.get("seed", 42)
    print(f"\n[{config.get('experiment_name')}] Locking random seed to: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 2. Init tracker
    tracker = ExperimentTracker(config)
    print(f"Started experiment: {tracker.experiment_dir}")

    # =====================================================================
    # 2. EXACT PHYSICS DATA LOADING & ALIGNMENT (From NN_Example.ipynb)
    # =====================================================================
    print("Loading NPZ and CSV data...")
    sig_csv_path = "Signal/signal_combined.csv"
    sig_npz_path = "Signal/signal_combined.npz"
    bkg_csv_path = "Background/bkg_combined.csv"
    bkg_npz_path = "Background/bkg_combined.npz"

    # Load arrays and dataframes
    npz_sig = np.load(sig_npz_path)
    npz_bkg = np.load(bkg_npz_path)
    df_sig = pd.read_csv(sig_csv_path)
    df_bkg = pd.read_csv(bkg_csv_path)

    # Offset background events to avoid overlapping
    ev_nums_sig = npz_sig["event_nums"]
    ev_nums_bkg = npz_bkg["event_nums"].copy()
    offset = 2 * ev_nums_sig.max()
    ev_nums_bkg += offset
    df_bkg["eventNumber"] = df_bkg["eventNumber"] + offset

    # Combine tracking lists and data matrices
    event_nums_all = np.concatenate([ev_nums_sig, ev_nums_bkg], axis=0)
    X_tensors_all = np.concatenate([npz_sig["X_tensors"], npz_bkg["X_tensors"]], axis=0)
    X_em2tensors_all = np.concatenate([npz_sig["X_em2_tensors"], npz_bkg["X_em2_tensors"]], axis=0)
    X_feats_all = np.concatenate([npz_sig["X_feats"], npz_bkg["X_feats"]], axis=0)
    df_all = pd.concat([df_sig, df_bkg], ignore_index=True)

    print("Computing Relative Eta/Phi...")
    pt = X_feats_all[:, :, 1]
    max_idx = np.argmax(pt, axis=1)

    eta_ref = X_feats_all[np.arange(len(X_feats_all)), max_idx][:, None, 2]
    phi_ref = X_feats_all[np.arange(len(X_feats_all)), max_idx][:, None, 3]

    X_feats_rel = X_feats_all.copy()
    X_feats_rel[:, :, 2] -= eta_ref
    X_feats_rel[:, :, 3] = (X_feats_rel[:, :, 3] - phi_ref + np.pi) % (2 * np.pi) - np.pi

    print("Flattening and Aligning objects with CSV...")
    # Flatten arrays
    X_tens_flat = X_tensors_all.reshape(-1, 45)
    X_feat_flat = X_feats_rel.reshape(-1, 4)

    # 1. Dynamically flatten the entire EM2 3D tensor (e.g., 12x12 = 144 cells)
    em2_spatial_size = X_em2tensors_all.shape[2] * X_em2tensors_all.shape[3]
    X_em2_flat = X_em2tensors_all.reshape(-1, em2_spatial_size)

    # Generate tracking indexes mapping back to parent events
    groups_flat = np.repeat(event_nums_all, 6)
    tob_index_flat = np.tile(np.arange(6), len(event_nums_all))

    # Filter out invalid objects
    csv_lookup_keys = df_all["eventNumber"].astype(str) + "_" + df_all["tob_index"].astype(str)
    npz_lookup_keys = pd.Series(groups_flat).astype(str) + "_" + pd.Series(tob_index_flat).astype(str)

    indexer = pd.Series(np.arange(len(groups_flat)), index=npz_lookup_keys)
    valid_indices = indexer.loc[csv_lookup_keys].values

    # Apply valid alignments
    groups_aligned = groups_flat[valid_indices]
    X_tens_aligned = X_tens_flat[valid_indices]
    X_feat_aligned = X_feat_flat[valid_indices]
    X_em2_aligned = X_em2_flat[valid_indices]

    # **CRITICAL BRIDGE**: Map the numpy arrays directly into the master DataFrame
    # This allows features.py to fetch them easily by column name
    # Create list of column names dynamically
    tensor_cols = [f'tensor_{i}' for i in range(45)]
    feat_cols = [f'feat_{i}' for i in range(4)]
    em2_cols = [f'em2_cell_{i}' for i in range(em2_spatial_size)]

    # Create DataFrames directly from the 2D arrays, keeping the index safe!
    df_tensors = pd.DataFrame(X_tens_aligned, columns=tensor_cols, index=df_all.index)
    df_feats = pd.DataFrame(X_feat_aligned, columns=feat_cols, index=df_all.index)
    df_em2 = pd.DataFrame(X_em2_aligned, columns=em2_cols, index=df_all.index)

    # Concatenate everything at once
    df_all = pd.concat([df_all, df_tensors, df_feats, df_em2], axis=1)

    # Ensure standard evaluation labels exist
    df_all['label'] = df_all['signal'].values.astype(np.float32)

    # 3. Train/Val/Test Split by Unique Event IDs
    unique_evs = np.unique(groups_aligned)
    np.random.shuffle(unique_evs)

    # 70% Train, 10% Val, 20% Test
    train_idx = int(len(unique_evs) * 0.70)
    val_idx = int(len(unique_evs) * 0.80)

    train_evs = unique_evs[:train_idx]
    val_evs = unique_evs[train_idx:val_idx]
    test_evs = unique_evs[val_idx:]

    train_mask = np.isin(groups_aligned, train_evs)
    val_mask = np.isin(groups_aligned, val_evs)
    test_mask = np.isin(groups_aligned, test_evs)

    df_train = df_all[train_mask].copy().reset_index(drop=True)
    df_val = df_all[val_mask].copy().reset_index(drop=True)
    df_test_meta = df_all[test_mask].copy().reset_index(drop=True)

    print(f"Data split ready! Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test_meta)}")

    # =====================================================================
    # 4. FEATURE ASSEMBLY & SCALING
    # =====================================================================
    print(f"Assembling features: {config['features_to_use']}")
    train_features, val_features, test_features = [], [], []

    for feature_name in config["features_to_use"]:
        train_features.append(FEATURE_REGISTRY[feature_name](df_train))
        val_features.append(FEATURE_REGISTRY[feature_name](df_val))
        test_features.append(FEATURE_REGISTRY[feature_name](df_test_meta))

    X_train_raw = np.hstack(train_features)
    X_val_raw = np.hstack(val_features)
    X_test_raw = np.hstack(test_features)

    # EXACT Normalization from your notebook
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    X_train_np = (X_train_raw - mean) / std
    X_val_np = (X_val_raw - mean) / std
    X_test_np = (X_test_raw - mean) / std

    y_train_np = df_train['label'].values
    y_val_np = df_val['label'].values
    y_test_np = df_test_meta['label'].values

    # =====================================================================
    # 5. PYTORCH TRAINING
    # =====================================================================
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
    tracker.save_predictions(df_eval)

    print(f"--> Saved rich physics predictions to: predictions.parquet")

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
    df_legacy = df_test_meta[legacy_columns].copy()

    # Translate to the old naming conventions
    # df_legacy.rename(columns={'eventNumber': 'event_num'}, inplace=True)
    df_legacy['Type'] = df_legacy['Type'].replace('Background', 'BKG')

    # Save directly into the current experiment's timestamped folder
    import os
    csv_path = os.path.join(tracker.experiment_dir, "legacy_results_with_scores.csv")
    df_legacy.to_csv(csv_path, index=False)
    print(f"--> Saved backward-compatible CSV to: {csv_path}")

    # =====================================================================

    print(f"Experiment '{config['experiment_name']}' complete! All files saved to {tracker.experiment_dir}")


def main():
    args = parse_args()

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
    completed_runs = get_completed_runs() if not args.force_redo else set()

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
        run_training_pipeline(config_path)


if __name__ == "__main__":
    main()
