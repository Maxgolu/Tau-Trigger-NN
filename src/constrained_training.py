"""Fine-tune TauNet with a direct event-level constrained objective."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from classifiers import calibrate_classifier, parse_classifier
from constrained_objective import (
    calculate_hard_constraint_metrics,
    calculate_soft_constraint_metrics,
    calibrate_tob_baseline,
    parse_constrained_objective,
    soft_object_pass,
)
from event_data import (
    EventTensorDataset,
    collate_events,
    split_training_events,
)
from model import DynamicMLP
from operating_point import select_background_objects
from tracker import ExperimentTracker


@dataclass
class DualState:
    """Non-negative prices for the FPR and every energy constraint."""

    multipliers: torch.Tensor

    def to_dict(self):
        return {"multipliers": self.multipliers.detach().cpu().tolist()}


def constrained_primal_loss(metrics, dual_state):
    """Minimize negative physics gain plus detached constraint prices."""
    return -metrics.objective + torch.sum(
        dual_state.multipliers.detach() * metrics.violations
    )


@torch.no_grad()
def update_dual_state(dual_state, hard_violations, learning_rate, maximum):
    """Ascend on measured violations and project prices to a finite interval."""
    dual_state.multipliers.add_(learning_rate * hard_violations)
    dual_state.multipliers.clamp_(min=0.0, max=maximum)


def _predict_numpy(model, features, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.tensor(features, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            scores.extend(
                model(batch.to(device)).detach().cpu().numpy().reshape(-1)
            )
    return np.asarray(scores, dtype=np.float64)


def _resolve_initial_weights(config, project_root):
    initialization = config.get("initialization", {})
    if initialization.get("mode") != "pretrained":
        raise ValueError("Constrained training currently requires pretrained initialization")
    raw_path = initialization.get("weights_path")
    if not raw_path:
        raise ValueError("initialization.weights_path is required")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    path = path.resolve()
    source_config_path = path.parent / "config.json"
    if source_config_path.exists():
        with source_config_path.open("r", encoding="utf-8") as handle:
            source_config = json.load(handle)
        checks = (
            ("features_to_use", config.get("features_to_use")),
            ("hidden_layers", config.get("hidden_layers", [32, 16])),
            ("seed", int(config.get("seed", 42))),
        )
        for key, expected in checks:
            if source_config.get(key) != expected:
                raise ValueError(
                    f"Pretrained {key} does not match constrained config; "
                    "this would invalidate the saved normalization or architecture"
                )
    return path


def _hard_violations(hard_metrics, objective_config, device):
    deltas = hard_metrics["region_deltas"]
    region_violations = [
        0.0 if delta is None else -deficit - delta
        for deficit, delta in zip(objective_config.allowed_deficits, deltas)
    ]
    return torch.tensor(
        [
            hard_metrics["achieved_fpr"] - objective_config.target_event_fpr,
            *region_violations,
        ],
        dtype=torch.float32,
        device=device,
    )


def _is_better_hard_candidate(candidate, best):
    if best is None:
        return True
    if candidate["constraints_satisfied"] != best["constraints_satisfied"]:
        return candidate["constraints_satisfied"]
    if not np.isclose(candidate["objective_value"], best["objective_value"]):
        return candidate["objective_value"] > best["objective_value"]
    return candidate["minimum_margin"] > best["minimum_margin"]


def _event_model_scores(model, batch):
    scores = batch.features.new_zeros(batch.object_mask.shape)
    scores[batch.object_mask] = model(
        batch.features[batch.object_mask]
    ).reshape(-1)
    return scores


def run_constrained_training_pipeline(
    config_path,
    data_dir,
    experiments_dir,
    project_root,
    max_events_per_class=None,
    epochs_override=None,
    data_cache=None,
):
    """Run the constrained path while leaving standard BCE training untouched."""
    with open(config_path, "r") as handle:
        config = json.load(handle)
    if max_events_per_class is not None:
        config["max_events_per_class"] = max_events_per_class
    if epochs_override is not None:
        config["epochs"] = epochs_override

    objective_config = parse_constrained_objective(config)
    classifier_config = parse_classifier(config)
    if classifier_config.target_fpr != objective_config.target_event_fpr:
        raise ValueError("Classifier and constrained target FPR must match")
    if classifier_config.trigger_objects != objective_config.trigger_objects:
        raise ValueError("Classifier and constrained trigger object counts must match")
    if classifier_config.tob_budget is not None:
        raise ValueError("First constrained OR experiments require a fixed TOB budget")

    seed = int(config.get("seed", 42))
    dataset = data_cache.get_dataset(
        data_dir=data_dir,
        max_events_per_class=config.get("max_events_per_class"),
        seed=seed,
    )
    split = data_cache.get_split(dataset, seed=seed)
    tracker = ExperimentTracker(config, base_dir=str(Path(experiments_dir).resolve()))
    print(f"Started constrained experiment: {tracker.experiment_dir}")

    feature_names = config["features_to_use"]
    X_train_raw = data_cache.assemble_features(dataset, split.train, feature_names)
    X_val_raw = data_cache.assemble_features(dataset, split.validation, feature_names)
    X_test_raw = data_cache.assemble_features(dataset, split.test, feature_names)
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0
    X_train = ((X_train_raw - mean) / std).astype(np.float32)
    X_val = ((X_val_raw - mean) / std).astype(np.float32)
    X_test = ((X_test_raw - mean) / std).astype(np.float32)
    y_train = dataset.labels[split.train].astype(np.float32)

    train_frame = dataset.frame.iloc[split.train].copy().reset_index(drop=True)
    validation_frame = dataset.frame.iloc[split.validation].copy().reset_index(drop=True)
    test_frame = dataset.frame.iloc[split.test].copy().reset_index(drop=True)
    primal_rows, constraint_rows = split_training_events(
        train_frame,
        seed=seed + 10_000,
        constraint_fraction=objective_config.constraint_fraction,
    )
    print(
        "Inner event split ready: "
        f"primal objects={len(primal_rows)} | constraint objects={len(constraint_rows)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicMLP(
        input_dim=X_train.shape[1],
        hidden_layers=config.get("hidden_layers", [32, 16]),
    ).to(device)
    weights_path = _resolve_initial_weights(config, project_root)
    try:
        initial_state = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Older PyTorch releases do not expose the safe weights_only argument.
        initial_state = torch.load(weights_path, map_location=device)
    model.load_state_dict(initial_state)
    print(f"Loaded constrained initialization: {weights_path}")

    object_batch_size = int(config.get("batch_size", 256))
    constraint_scores = _predict_numpy(
        model,
        X_train[constraint_rows],
        object_batch_size,
        device,
    )
    constraint_frame = train_frame.iloc[constraint_rows].copy().reset_index(drop=True)
    background = select_background_objects(constraint_frame)
    background_positions = constraint_frame.index[
        constraint_frame["Type"].isin(["BKG", "Background"])
    ].to_numpy()
    calibration = calibrate_classifier(
        background,
        constraint_scores[background_positions],
        classifier_config,
    )
    baseline_threshold_gev, baseline_fpr = calibrate_tob_baseline(
        background,
        objective_config.target_event_fpr,
        objective_config.trigger_objects,
    )
    fixed_nn_threshold = float(calibration["nn_threshold"])
    fixed_tob_threshold = calibration.get("tob_threshold_gev")
    print(
        "Fixed training calibration: "
        f"NN={fixed_nn_threshold:.6f} | "
        f"TOB={fixed_tob_threshold} | baseline FPR={baseline_fpr:.6f}"
    )

    primal_dataset = EventTensorDataset(
        X_train,
        y_train,
        train_frame,
        row_indices=primal_rows,
    )
    primal_loader = DataLoader(
        primal_dataset,
        batch_size=objective_config.event_batch_size,
        shuffle=True,
        collate_fn=collate_events,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0001)),
    )
    dual_state = DualState(
        multipliers=torch.tensor(
            [
                objective_config.initial_fpr_multiplier,
                *(
                    [objective_config.initial_region_multiplier]
                    * len(objective_config.regions_gev)
                ),
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    bce_monitor = nn.BCELoss()
    epochs = int(config.get("epochs", 5))
    history = []
    initial_constraint = calculate_hard_constraint_metrics(
        constraint_frame,
        constraint_scores,
        classifier_config,
        objective_config,
        calibration=calibration,
        baseline_threshold_gev=baseline_threshold_gev,
    )
    initial_validation_scores = _predict_numpy(
        model,
        X_val,
        object_batch_size,
        device,
    )
    initial_validation = calculate_hard_constraint_metrics(
        validation_frame,
        initial_validation_scores,
        classifier_config,
        objective_config,
    )
    history.append(
        {
            "epoch": 0,
            "training_loss": None,
            "training_bce": None,
            "dual_state": dual_state.to_dict(),
            "constraint_training": initial_constraint,
            "validation": initial_validation,
        }
    )
    best_record = copy.deepcopy(initial_validation)
    best_record["epoch"] = 0
    best_weights = copy.deepcopy(model.state_dict())
    best_dual = copy.deepcopy(dual_state.to_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    print(
        "Epoch 00 (pretrained) - "
        f"Val J: {initial_validation['objective_value']:.5f} | "
        f"feasible={initial_validation['constraints_satisfied']}"
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bce = 0.0
        batch_count = 0
        for batch in primal_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            scores = _event_model_scores(model, batch)
            pass_probabilities = soft_object_pass(
                scores,
                fixed_nn_threshold,
                objective_config.temperature,
                classifier_config.name,
                batch.tob_pt_gev,
                fixed_tob_threshold,
            )
            baseline_pass = batch.tob_pt_gev >= baseline_threshold_gev
            metrics = calculate_soft_constraint_metrics(
                pass_probabilities,
                batch.object_mask,
                batch.signal_object_mask,
                batch.background_event_mask,
                batch.truth_pt_gev,
                baseline_pass,
                objective_config,
            )
            loss = constrained_primal_loss(metrics, dual_state)
            loss.backward()
            optimizer.step()

            valid_scores = scores[batch.object_mask]
            valid_labels = batch.labels[batch.object_mask].reshape(-1)
            epoch_bce += bce_monitor(valid_scores, valid_labels).item()
            epoch_loss += loss.item()
            batch_count += 1

        constraint_scores = _predict_numpy(
            model,
            X_train[constraint_rows],
            object_batch_size,
            device,
        )
        hard_constraint = calculate_hard_constraint_metrics(
            constraint_frame,
            constraint_scores,
            classifier_config,
            objective_config,
            calibration=calibration,
            baseline_threshold_gev=baseline_threshold_gev,
        )
        hard_violations = _hard_violations(
            hard_constraint,
            objective_config,
            device,
        )
        if (
            epoch + 1 > objective_config.dual_warmup_epochs
            and (epoch + 1) % objective_config.dual_update_frequency == 0
        ):
            update_dual_state(
                dual_state,
                hard_violations,
                objective_config.dual_learning_rate,
                objective_config.max_multiplier,
            )

        validation_scores = _predict_numpy(
            model,
            X_val,
            object_batch_size,
            device,
        )
        hard_validation = calculate_hard_constraint_metrics(
            validation_frame,
            validation_scores,
            classifier_config,
            objective_config,
        )
        record = {
            "epoch": epoch + 1,
            "training_loss": float(epoch_loss / max(batch_count, 1)),
            "training_bce": float(epoch_bce / max(batch_count, 1)),
            "dual_state": dual_state.to_dict(),
            "constraint_training": hard_constraint,
            "validation": hard_validation,
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Constrained loss: {record['training_loss']:.5f} | "
            f"Val J: {hard_validation['objective_value']:.5f} | "
            f"feasible={hard_validation['constraints_satisfied']} | "
            f"lambdas={dual_state.to_dict()['multipliers']}"
        )
        if _is_better_hard_candidate(hard_validation, best_record):
            best_record = copy.deepcopy(hard_validation)
            best_record["epoch"] = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            best_dual = copy.deepcopy(dual_state.to_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())

    model.load_state_dict(best_weights)
    test_scores = _predict_numpy(model, X_test, object_batch_size, device)
    output_frame = test_frame.copy()
    output_frame["nn_score"] = test_scores
    core_columns = [
        "eventNumber", "tob_index", "signal", "Type", "truth_pt",
        "tob_pt", "tob_eta", "tob_phi", "nn_score",
    ]
    tracker.save_weights(model, filename="model_weights.pt")
    predictions_path = tracker.save_predictions(
        output_frame[core_columns].copy(),
        stem="predictions",
    )
    checkpoint_path = Path(tracker.experiment_dir) / "constrained_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_weights,
            "optimizer_state_dict": best_optimizer,
            "dual_state": best_dual,
            "fixed_nn_threshold": fixed_nn_threshold,
            "fixed_tob_threshold_gev": fixed_tob_threshold,
            "baseline_threshold_gev": baseline_threshold_gev,
            "epoch": best_record["epoch"],
        },
        checkpoint_path,
    )
    tracker.save_json(
        {
            "objective": objective_config.to_dict(),
            "classifier": classifier_config.to_dict(),
            "initial_weights": str(weights_path),
            "fixed_training_calibration": calibration,
            "baseline_threshold_gev": baseline_threshold_gev,
            "best_validation_record": best_record,
            "history": history,
            "artifacts": {
                "weights": "model_weights.pt",
                "checkpoint": checkpoint_path.name,
                "predictions": Path(predictions_path).name,
            },
        },
        "constrained_training.json",
    )
    tracker.save_json(
        {
            "methods": ["target_fpr"],
            "primary_method": "target_fpr",
            "target_fpr": objective_config.target_event_fpr,
            "trigger_objects": objective_config.trigger_objects,
            "classifier": classifier_config.to_dict(),
            "loss": {"name": "constrained_trigger", **objective_config.to_dict()},
            "artifacts": {
                "target_fpr": {
                    "role": "primary",
                    "weights": "model_weights.pt",
                    "predictions": Path(predictions_path).name,
                    "best_validation_record": best_record,
                }
            },
            "epoch_history": history,
        },
        "checkpoint_selection.json",
    )
    print(f"Constrained experiment complete: {tracker.experiment_dir}")
    return tracker.experiment_dir
