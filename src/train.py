import argparse
import json
import torch
import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import glob
import copy
from pathlib import Path

from model import DynamicMLP
from classifiers import parse_classifier
from classifier_selection import (
    build_validation_folds,
    search_validation_tob_budget,
)
from losses import (
    build_loss,
    calculate_sample_weights,
    fit_loss_weighting,
    parse_loss,
)
from tracker import ExperimentTracker
from training_data import TrainingDataCache
from checkpoint_selection import (
    calculate_validation_operating_point,
    is_better_checkpoint,
    parse_checkpoint_selection,
)
from constrained_training import run_constrained_training_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = PROJECT_ROOT
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def predict_scores(model, data_loader, device):
    """Return model scores in deterministic DataLoader order."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in data_loader:
            X_batch = batch[0]
            predictions = model(X_batch.to(device))
            scores.extend(predictions.cpu().numpy().reshape(-1))
    return np.asarray(scores, dtype=np.float64)


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

    # The direct trigger objective needs complete events and a separate dual
    # update, so config selection dispatches it before the object-level path.
    if config.get("loss", {}).get("name") == "constrained_trigger":
        return run_constrained_training_pipeline(
            config_path=config_path,
            data_dir=data_dir,
            experiments_dir=experiments_dir,
            project_root=PROJECT_ROOT,
            max_events_per_class=max_events_per_class,
            epochs_override=epochs_override,
            data_cache=data_cache,
        )

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

    df_train_meta = (
        dataset.frame.iloc[split.train][["truth_pt"]]
        .copy()
        .reset_index(drop=True)
    )
    df_val_meta = (
        dataset.frame.iloc[split.validation].copy().reset_index(drop=True)
    )

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

    loss_config = parse_loss(config)
    fitted_loss_weighting = fit_loss_weighting(
        loss_config,
        df_train_meta,
        y_train_np,
    )
    uses_sample_weights = fitted_loss_weighting is not None

    batch_size = config.get("batch_size", 1024)  # updated to your notebook's default 1024
    if uses_sample_weights:
        train_weights = torch.tensor(
            calculate_sample_weights(
                fitted_loss_weighting,
                df_train_meta,
                y_train_np,
            ),
            dtype=torch.float32,
        ).unsqueeze(1)
        val_weights = torch.tensor(
            calculate_sample_weights(
                fitted_loss_weighting,
                df_val_meta,
                y_val_np,
            ),
            dtype=torch.float32,
        ).unsqueeze(1)
        train_dataset = TensorDataset(X_train, y_train, train_weights)
        val_dataset = TensorDataset(X_val, y_val, val_weights)
        print(
            "Energy weights fitted on training signal: "
            f"profile={fitted_loss_weighting.profile} | "
            f"signal mean={train_weights[y_train.reshape(-1) == 1].mean():.4f}"
        )
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = DynamicMLP(
        input_dim=X_train_np.shape[1],
        hidden_layers=config.get("hidden_layers", [32, 16])  # Matches your MLP logic
    ).to(device)

    training_criterion = build_loss(loss_config)
    bce_monitor = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    epochs = config.get("epochs", 20)
    selection = parse_checkpoint_selection(config)
    if "classifier" not in config:
        classifier_config = parse_classifier(
            {
                "classifier": {
                    "name": "nn_only",
                    "target_fpr": selection.target_fpr,
                    "trigger_objects": selection.trigger_objects,
                }
            }
        )
    else:
        classifier_config = parse_classifier(config)
        if classifier_config.trigger_objects != selection.trigger_objects:
            raise ValueError(
                "classifier.trigger_objects and "
                "checkpoint_selection.trigger_objects must match"
            )
        if not np.isclose(classifier_config.target_fpr, selection.target_fpr):
            raise ValueError(
                "classifier.target_fpr and checkpoint_selection.target_fpr "
                "must match"
            )
    needs_operating_point = (
        "target_fpr" in selection.methods
        or classifier_config.name != "nn_only"
    )
    validation_fold_ids = None
    validation_fold_audit = None
    if classifier_config.tob_budget is not None:
        validation_fold_ids, validation_fold_audit = build_validation_folds(
            df_val_meta,
            seed=seed,
            folds=classifier_config.tob_budget.cross_validation_folds,
        )
        print(
            "TOB budget: validation search with "
            f"{len(classifier_config.tob_budget.values)} candidates"
        )
    print(f"Starting training for {epochs} epochs...")
    print(
        f"Classifier: {classifier_config.name} | Loss: {loss_config.name}"
    )
    print(
        "Checkpoint selection: "
        f"{list(selection.methods)} | primary={selection.primary_method}"
    )

    best_records = {method: None for method in selection.methods}
    best_weights = {}
    checkpoint_history = []

    for epoch in range(epochs):
        # -- TRAINING PHASE --
        model.train()
        running_train_loss = 0.0
        running_train_bce = 0.0

        for batch in train_loader:
            X_batch, y_batch = batch[:2]
            weight_batch = batch[2] if uses_sample_weights else None
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if weight_batch is not None:
                weight_batch = weight_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            if weight_batch is None:
                loss = training_criterion(outputs, y_batch)
            else:
                loss = training_criterion(outputs, y_batch, weight_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)
            running_train_bce += (
                bce_monitor(outputs.detach(), y_batch).item()
                * X_batch.size(0)
            )

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_bce = running_train_bce / len(train_loader.dataset)

        # -- VALIDATION PHASE --
        model.eval()
        running_val_loss = 0.0
        running_val_bce = 0.0
        validation_scores = []

        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = batch[:2]
                weight_batch = batch[2] if uses_sample_weights else None
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if weight_batch is not None:
                    weight_batch = weight_batch.to(device)
                outputs = model(X_batch)
                if weight_batch is None:
                    val_loss = training_criterion(outputs, y_batch)
                else:
                    val_loss = training_criterion(outputs, y_batch, weight_batch)
                running_val_loss += val_loss.item() * X_batch.size(0)
                running_val_bce += (
                    bce_monitor(outputs, y_batch).item() * X_batch.size(0)
                )
                validation_scores.extend(outputs.cpu().numpy().reshape(-1))

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_bce = running_val_bce / len(val_loader.dataset)
        epoch_record = {
            "epoch": epoch + 1,
            "training_loss": float(epoch_train_loss),
            "validation_loss": float(epoch_val_loss),
            "training_bce": float(epoch_train_bce),
            "validation_bce": float(epoch_val_bce),
        }

        if needs_operating_point:
            if classifier_config.tob_budget is not None:
                operating_point = search_validation_tob_budget(
                    df_val_meta,
                    validation_scores,
                    classifier_config,
                    validation_fold_ids,
                    validation_fold_audit,
                )
            else:
                operating_point = calculate_validation_operating_point(
                    df_val_meta,
                    validation_scores,
                    target_fpr=selection.target_fpr,
                    trigger_objects=selection.trigger_objects,
                    energy_bands_gev=selection.energy_bands_gev,
                    classifier_config=classifier_config,
                )
            epoch_record.update(operating_point)

        message = (
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Train {loss_config.name}: {epoch_train_loss:.4f} | "
            f"Val {loss_config.name}: {epoch_val_loss:.4f} | "
            f"Val BCE: {epoch_val_bce:.4f}"
        )
        if needs_operating_point:
            message += (
                f" | Val Eff@{selection.target_fpr * 100:.1f}% FPR: "
                f"{epoch_record['signal_efficiency']:.4f} "
                f"(achieved {epoch_record['achieved_fpr'] * 100:.4f}%)"
            )
            if classifier_config.tob_budget is not None:
                message += (
                    f" | b={epoch_record['selected_tob_fpr']:.4f} "
                    f"| J={epoch_record['objective_value']:.4f} "
                    f"| feasible={epoch_record['noninferiority_satisfied']}"
                )
        print(message)

        for method in selection.methods:
            if is_better_checkpoint(method, epoch_record, best_records[method]):
                best_records[method] = copy.deepcopy(epoch_record)
                best_weights[method] = copy.deepcopy(model.state_dict())
                if method == "validation_bce":
                    print("  --> New best validation-BCE checkpoint.")
                else:
                    print("  --> New best target-FPR checkpoint.")

        checkpoint_history.append(epoch_record)

    # =====================================================================
    # 6. EVALUATION & TRACKING
    # =====================================================================
    # Columns required for object kinematics and event-level threshold calibration.
    core_columns = [
        'eventNumber', 'tob_index', 'signal', 'Type',
        'truth_pt', 'tob_pt', 'tob_eta', 'tob_phi', 'nn_score'
    ]
    artifacts = {}
    for method in selection.methods:
        model.load_state_dict(best_weights[method])
        if classifier_config.tob_budget is not None:
            # Cross-fitting selects the epoch and budget. This final pass only
            # calibrates deployable thresholds on the complete validation set.
            validation_scores = predict_scores(model, val_loader, device)
            selected_tob_fpr = best_records[method]["selected_tob_fpr"]
            concrete_classifier = classifier_config.with_tob_fpr(
                selected_tob_fpr
            )
            final_validation = calculate_validation_operating_point(
                df_val_meta,
                validation_scores,
                target_fpr=selection.target_fpr,
                trigger_objects=selection.trigger_objects,
                energy_bands_gev=selection.energy_bands_gev,
                classifier_config=concrete_classifier,
            )
            cross_fitted = {
                key: copy.deepcopy(best_records[method][key])
                for key in (
                    "achieved_fpr",
                    "signal_efficiency",
                    "objective_value",
                    "minimum_delta",
                    "minimum_guard_margin",
                    "noninferiority_satisfied",
                    "tob_budget_search",
                )
            }
            best_records[method].update(final_validation)
            best_records[method]["selected_tob_fpr"] = selected_tob_fpr
            best_records[method]["cross_fitted_selection"] = cross_fitted
            best_records[method]["final_validation_calibration"] = {
                "selected_tob_fpr": selected_tob_fpr,
                "thresholds": final_validation["classifier_calibration"],
            }
        scores = predict_scores(model, test_loader, device)
        method_frame = df_test_meta.copy()
        method_frame["nn_score"] = scores
        df_eval = method_frame[core_columns].copy()

        is_primary = method == selection.primary_method
        weights_filename = (
            "model_weights.pt" if is_primary
            else f"model_weights_{method}.pt"
        )
        prediction_stem = (
            "predictions" if is_primary else f"predictions_{method}"
        )
        tracker.save_weights(model, filename=weights_filename)
        predictions_path = tracker.save_predictions(
            df_eval,
            stem=prediction_stem,
        )
        artifacts[method] = {
            "role": "primary" if is_primary else "secondary",
            "weights": weights_filename,
            "predictions": os.path.basename(predictions_path),
            "best_validation_record": best_records[method],
        }
        print(
            f"--> Saved {method} test predictions to: {predictions_path}"
        )

    selection_manifest = {
        "methods": list(selection.methods),
        "primary_method": selection.primary_method,
        "target_fpr": selection.target_fpr,
        "trigger_objects": selection.trigger_objects,
        "classifier": classifier_config.to_dict(),
        "loss": loss_config.to_dict(),
        "fitted_loss_weighting": (
            fitted_loss_weighting.to_dict()
            if fitted_loss_weighting is not None
            else None
        ),
        "energy_bands_gev": [list(band) for band in selection.energy_bands_gev],
        "artifacts": artifacts,
        "epoch_history": checkpoint_history,
    }
    tracker.save_json(selection_manifest, "checkpoint_selection.json")

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
        model.load_state_dict(best_weights[selection.primary_method])
        primary_scores = predict_scores(model, test_loader, device)
        primary_frame = df_test_meta.copy()
        primary_frame["nn_score"] = primary_scores
        df_legacy = primary_frame[legacy_columns].copy()
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
