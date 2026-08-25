"""Fine-tune TauNet with a direct event-level constrained objective."""

import copy
import json
from dataclasses import dataclass, replace
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
    rank_calibrated_threshold,
    parse_constrained_objective,
    soft_object_pass,
    tail_ranking_objective,
)
from constrained_validation import (
    build_constraint_crossfit_rows,
    calculate_cross_fitted_hard_metrics,
    regional_gradient_diagnostics,
)
from event_data import (
    EventTensorDataset,
    collate_events,
    split_training_events,
)
from model import DynamicMLP, build_model
from operating_point import select_background_objects
from tracker import ExperimentTracker


def _build_constrained_model(config, input_dim, feature_layout, device):
    """Build the configured model (legacy MLP or tensor_cnn) for fine-tuning.

    The constrained pipeline standardizes features with plain per-column
    z-scoring and applies no branch input transforms, so a tensor_cnn config
    is only accepted when every branch declares transform "none" (or omits
    it as documentation). Otherwise the pretrained weights would have been
    fitted on differently preprocessed inputs.
    """
    model_config = config.get("model")
    if model_config is not None and model_config.get("name") == "tensor_cnn":
        for branch in model_config.get("branches", []):
            transform = branch.get("transform", "none")
            if transform not in ("none", None):
                raise ValueError(
                    "Constrained fine-tuning supports only transform 'none' "
                    f"branches; branch '{branch.get('feature')}' declares "
                    f"'{transform}'"
                )
    return build_model(config, input_dim, feature_layout).to(device)


@dataclass
class DualState:
    """Non-negative prices for the FPR and every energy constraint."""

    multipliers: torch.Tensor

    def to_dict(self):
        return {"multipliers": self.multipliers.detach().cpu().tolist()}


class HardNegativeMemoryBank:
    """Keep rank-selected, median-centered background event offsets."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self._values = None

    @property
    def values(self):
        return self._values

    @torch.no_grad()
    def update(self, values):
        if self.capacity <= 0 or values is None or not values.numel():
            return
        values = values.detach()
        combined = values if self._values is None else torch.cat(
            (self._values.to(values.device), values), dim=0
        )
        keep = min(self.capacity, combined.numel())
        self._values = torch.topk(combined, k=keep).values.detach()

    def __len__(self):
        return 0 if self._values is None else int(self._values.numel())


def constrained_primal_loss(metrics, dual_state):
    """Minimize negative physics gain plus detached constraint prices."""
    return -metrics.objective + torch.sum(
        dual_state.multipliers.detach() * metrics.violations
    )


def parameter_gradient_norm(value, parameters, retain_graph=True):
    """Measure one objective's gradient scale without changing model gradients."""
    gradients = torch.autograd.grad(
        value,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared_norm = value.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared_norm = squared_norm + torch.sum(gradient.detach() ** 2)
    return torch.sqrt(squared_norm)


def parameter_gradient_pair_statistics(
    first_value,
    second_value,
    parameters,
    retain_graph=True,
):
    """Measure two gradient scales and their cosine without populating .grad."""
    parameters = tuple(parameters)
    first_gradients = torch.autograd.grad(
        first_value,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second_gradients = torch.autograd.grad(
        second_value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    first_squared = first_value.new_zeros(())
    second_squared = first_value.new_zeros(())
    dot_product = first_value.new_zeros(())
    for first_gradient, second_gradient in zip(
        first_gradients,
        second_gradients,
    ):
        if first_gradient is not None:
            first_squared = first_squared + torch.sum(first_gradient.detach() ** 2)
        if second_gradient is not None:
            second_squared = second_squared + torch.sum(second_gradient.detach() ** 2)
        if first_gradient is not None and second_gradient is not None:
            dot_product = dot_product + torch.sum(
                first_gradient.detach() * second_gradient.detach()
            )
    first_norm = torch.sqrt(first_squared)
    second_norm = torch.sqrt(second_squared)
    denominator = first_norm * second_norm
    cosine = torch.where(
        denominator > 0.0,
        dot_product / denominator,
        dot_product.new_zeros(()),
    )
    return first_norm, second_norm, cosine


def _soft_batch_metrics(
    model,
    batch,
    classifier_config,
    objective_config,
    fixed_nn_threshold,
    fixed_tob_threshold,
    baseline_threshold_gev,
    reference_model=None,
    temperature=None,
    tail_memory_bank=None,
    update_memory=False,
):
    """Build the differentiable objective for one complete-event batch."""
    if temperature is None:
        temperature = objective_config.temperature
    scores = _event_model_scores(model, batch)
    logits = None
    rank_proxy = (
        objective_config.primal_objective == "tail_ranking"
        or objective_config.proxy_threshold_mode == "batch_rank"
    )
    if rank_proxy:
        if classifier_config.name != "nn_only":
            raise ValueError("Rank-calibrated proxies currently require nn_only")
        logits = _event_model_logits(model, batch)
        nn_threshold = rank_calibrated_threshold(
            logits,
            batch.object_mask,
            batch.background_event_mask,
            objective_config.target_event_fpr,
            objective_config.trigger_objects,
            temperature,
        )
        pass_probabilities = soft_object_pass(
            logits,
            nn_threshold,
            temperature,
            classifier_config.name,
        )
    else:
        pass_probabilities = soft_object_pass(
            scores,
            fixed_nn_threshold,
            temperature,
            classifier_config.name,
            batch.tob_pt_gev,
            fixed_tob_threshold,
        )
    baseline_pass = batch.tob_pt_gev >= baseline_threshold_gev
    reference_pass_probabilities = None
    if reference_model is not None:
        with torch.no_grad():
            reference_scores = _event_model_scores(reference_model, batch)
            reference_threshold = fixed_nn_threshold
            reference_input = reference_scores
            if rank_proxy:
                reference_logits = _event_model_logits(reference_model, batch)
                reference_threshold = rank_calibrated_threshold(
                    reference_logits,
                    batch.object_mask,
                    batch.background_event_mask,
                    objective_config.target_event_fpr,
                    objective_config.trigger_objects,
                    temperature,
                )
                reference_input = reference_logits
            reference_pass_probabilities = soft_object_pass(
                reference_input,
                reference_threshold,
                temperature,
                classifier_config.name,
                batch.tob_pt_gev,
                fixed_tob_threshold,
            )
    objective_override = None
    ranking_loss = None
    current_tail_offsets = None
    tail_event_count = None
    if objective_config.primal_objective == "tail_ranking":
        memory_values = None if tail_memory_bank is None else tail_memory_bank.values
        (
            objective_override,
            ranking_loss,
            current_tail_offsets,
            tail_event_count,
        ) = tail_ranking_objective(
            logits,
            batch.object_mask,
            batch.signal_object_mask,
            batch.background_event_mask,
            batch.truth_pt_gev,
            objective_config,
            memory_tail_offsets=memory_values,
        )
    metrics = calculate_soft_constraint_metrics(
        pass_probabilities,
        batch.object_mask,
        batch.signal_object_mask,
        batch.background_event_mask,
        batch.truth_pt_gev,
        baseline_pass,
        objective_config,
        reference_object_pass_probabilities=reference_pass_probabilities,
        objective_override=objective_override,
        ranking_loss=ranking_loss,
        tail_event_count=tail_event_count,
        current_tail_offsets=current_tail_offsets,
    )
    if update_memory and tail_memory_bank is not None:
        tail_memory_bank.update(current_tail_offsets)
    return scores, metrics


def initialize_fpr_multiplier_from_gradients(
    model,
    batches,
    classifier_config,
    objective_config,
    fixed_nn_threshold,
    fixed_tob_threshold,
    baseline_threshold_gev,
    reference_model=None,
    temperature=None,
):
    """Balance initial objective and FPR gradient scales on training batches."""
    if objective_config.initial_fpr_multiplier_mode == "fixed":
        return objective_config.initial_fpr_multiplier, {
            "mode": "fixed",
            "fixed_fallback": objective_config.initial_fpr_multiplier,
            "recommended_unclipped": None,
            "selected": objective_config.initial_fpr_multiplier,
            "batches_measured": 0,
            "measurements": [],
            "reason": "Gradient balancing is disabled by configuration.",
        }
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    measurements = []
    model.train()
    for batch_index, batch in enumerate(batches):
        if batch_index >= objective_config.gradient_balance_batches:
            break
        _, metrics = _soft_batch_metrics(
            model,
            batch,
            classifier_config,
            objective_config,
            fixed_nn_threshold,
            fixed_tob_threshold,
            baseline_threshold_gev,
            reference_model=reference_model,
            temperature=temperature,
        )
        objective_norm, fpr_norm, cosine = parameter_gradient_pair_statistics(
            metrics.objective,
            metrics.violations[0],
            parameters,
            retain_graph=False,
        )
        objective_value = float(objective_norm.detach().cpu())
        fpr_value = float(fpr_norm.detach().cpu())
        if np.isfinite(objective_value) and np.isfinite(fpr_value):
            ratio = objective_value / (
                fpr_value + objective_config.gradient_balance_epsilon
            )
            measurements.append(
                {
                    "soft_objective": float(metrics.objective.detach().cpu()),
                    "soft_event_fpr": float(metrics.event_fpr.detach().cpu()),
                    "scaled_fpr_violation": float(
                        metrics.violations[0].detach().cpu()
                    ),
                    "objective_gradient_norm": objective_value,
                    "fpr_gradient_norm": fpr_value,
                    "gradient_cosine_similarity": float(cosine.detach().cpu()),
                    "ratio": float(ratio),
                }
            )
    if not measurements:
        raise RuntimeError("Could not measure constrained gradient scales")

    ratios = np.asarray([item["ratio"] for item in measurements], dtype=np.float64)
    recommended = float(np.median(ratios))
    selected = objective_config.initial_fpr_multiplier
    if objective_config.initial_fpr_multiplier_mode == "gradient_balance":
        selected = float(np.clip(recommended, 0.0, objective_config.max_multiplier))
    return selected, {
        "mode": objective_config.initial_fpr_multiplier_mode,
        "fixed_fallback": objective_config.initial_fpr_multiplier,
        "recommended_unclipped": recommended,
        "selected": selected,
        "batches_measured": len(measurements),
        "measurements": measurements,
    }


def _score_quantiles(frame, scores):
    """Summarize score motion separately for truth taus and background objects."""
    scores = np.asarray(scores, dtype=np.float64)
    signal_mask = (
        (frame["Type"].to_numpy() == "Signal")
        & (frame["signal"].to_numpy(dtype=np.int64) == 1)
    )
    background_mask = frame["Type"].isin(["BKG", "Background"]).to_numpy()

    def summarize(mask):
        values = scores[mask]
        if not len(values):
            return None
        probabilities = [0.01, 0.1, 0.5, 0.9, 0.99]
        quantiles = np.quantile(values, probabilities)
        return {
            f"q{int(probability * 100):02d}": float(value)
            for probability, value in zip(probabilities, quantiles)
        }

    return {
        "signal": summarize(signal_mask),
        "background": summarize(background_mask),
    }


@torch.no_grad()
def update_dual_state(
    dual_state,
    hard_violations,
    learning_rate,
    maximum,
    region_learning_rate=None,
):
    """Ascend with separate FPR and energy-region response rates."""
    if region_learning_rate is None:
        region_learning_rate = learning_rate
    rates = torch.full_like(hard_violations, float(region_learning_rate))
    rates[0] = float(learning_rate)
    dual_state.multipliers.add_(rates * hard_violations)
    dual_state.multipliers.clamp_(min=0.0, max=maximum)


def constraint_resolution_warnings(metrics, regions):
    """Describe initial guards whose slack is below one observed object."""
    messages = []
    margins = metrics.get("constraint_margins", [])
    resolutions = metrics.get("region_efficiency_resolutions", [])
    for index, ((low, high), margin, resolution) in enumerate(
        zip(regions, margins, resolutions)
    ):
        if margin is None or resolution is None:
            continue
        if margin < 0.0:
            messages.append(
                f"region {index} [{low:g}, {high:g}) GeV starts infeasible "
                f"by {-margin:.6f}"
            )
        elif margin < resolution:
            messages.append(
                f"region {index} [{low:g}, {high:g}) GeV has initial slack "
                f"{margin:.6f}, below one-object resolution {resolution:.6f}"
            )
    return messages


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
            ("seed", int(config.get("seed", 42))),
        )
        for key, expected in checks:
            if source_config.get(key) != expected:
                raise ValueError(
                    f"Pretrained {key} does not match constrained config; "
                    "this would invalidate the saved normalization or architecture"
                )
        # Architecture compatibility. tensor_cnn configs describe the
        # architecture in a "model" block; legacy MLP configs use
        # "hidden_layers". Compare whichever representation applies, so a
        # tensor_cnn checkpoint is accepted by an identical tensor_cnn
        # config and never silently mixed with an MLP (or a different CNN).
        if (config.get("model") is not None
                or source_config.get("model") is not None):
            if source_config.get("model") != config.get("model"):
                raise ValueError(
                    "Pretrained model block does not match constrained "
                    "config; this would invalidate the saved architecture"
                )
        elif (source_config.get("hidden_layers")
                != config.get("hidden_layers", [32, 16])):
            raise ValueError(
                "Pretrained hidden_layers does not match constrained config; "
                "this would invalidate the saved normalization or architecture"
            )
    return path


def _hard_violations(hard_metrics, objective_config, device):
    feasibility = hard_metrics.get("feasibility")
    if (
        objective_config.feasibility_confidence_level is not None
        and feasibility is not None
    ):
        fpr_value = feasibility["fpr"]["upper_confidence_bound"]
        region_violations = [
            0.0
            if region["certified_margin"] is None
            else -region["certified_margin"]
            for region in feasibility["regions"]
        ]
    else:
        fpr_value = hard_metrics["achieved_fpr"]
        region_violations = [
            0.0 if margin is None else -margin
            for margin in hard_metrics["constraint_margins"]
        ]
    return torch.tensor(
        [
            (
                fpr_value - objective_config.target_event_fpr
            )
            * objective_config.fpr_violation_scale,
            *region_violations,
        ],
        dtype=torch.float32,
        device=device,
    )


def resolve_constrained_or_budget(classifier_config, objective_config):
    """Return (budget candidates, surrogate classifier) for the OR path.

    Under rank-calibrated objectives the TOB branch never receives gradients,
    so the budget only affects hard measurements and may be selected there.
    The differentiable surrogate stays NN-only in that case. Legacy paths
    (fixed-threshold surrogates) keep their previous behavior exactly.
    """
    rank_proxy = (
        objective_config.primal_objective == "tail_ranking"
        or objective_config.proxy_threshold_mode == "batch_rank"
    )
    if classifier_config.name == "nn_only":
        if classifier_config.tob_budget is not None:
            raise ValueError("nn_only cannot carry a TOB budget")
        return None, classifier_config
    if not rank_proxy:
        if classifier_config.tob_budget is not None:
            raise ValueError(
                "Dynamic TOB budgets in constrained training require a "
                "rank-calibrated objective, whose gradients never touch "
                "the budget"
            )
        # Legacy fixed-threshold OR training (Stage E) is unchanged.
        return None, classifier_config
    if classifier_config.tob_budget is not None:
        budget_values = classifier_config.tob_budget.values
    elif classifier_config.tob_fpr is not None:
        budget_values = (classifier_config.tob_fpr,)
    else:
        raise ValueError("tob_nn_or requires tob_fpr or tob_budget.values")
    candidates = tuple(
        classifier_config.with_tob_fpr(value) for value in budget_values
    )
    surrogate = replace(
        classifier_config,
        name="nn_only",
        tob_fpr=None,
        tob_budget=None,
    )
    return candidates, surrogate


def _budget_candidate_summary(record, budget):
    return {
        "tob_fpr": budget,
        "objective_value": record.get("objective_value"),
        "constraints_satisfied": record.get("constraints_satisfied"),
        "minimum_certified_margin": record.get("minimum_certified_margin"),
        "achieved_fpr": record.get("achieved_fpr"),
    }


def budget_searched_metrics(measure, budget_candidates):
    """Measure every candidate budget and keep the feasibility-first best."""
    best = None
    summaries = []
    for candidate in budget_candidates:
        record = measure(candidate)
        record["selected_tob_fpr"] = candidate.tob_fpr
        summaries.append(_budget_candidate_summary(record, candidate.tob_fpr))
        if _is_better_hard_candidate(record, best):
            best = record
    best["tob_budget_search"] = {
        "mode": "validation_search",
        "selected_tob_fpr": best["selected_tob_fpr"],
        "candidates": summaries,
    }
    return best


def _is_better_hard_candidate(candidate, best):
    if best is None:
        return True
    if candidate["constraints_satisfied"] != best["constraints_satisfied"]:
        return candidate["constraints_satisfied"]
    confidence_mode = (
        candidate.get("feasibility", {}).get("mode") == "one_sided_confidence"
        and best.get("feasibility", {}).get("mode") == "one_sided_confidence"
    )
    if not candidate["constraints_satisfied"] and confidence_mode:
        candidate_margin = candidate.get(
            "minimum_certified_margin", candidate.get("minimum_margin")
        )
        best_margin = best.get(
            "minimum_certified_margin", best.get("minimum_margin")
        )
        if candidate_margin is not None and best_margin is not None and not np.isclose(
            candidate_margin, best_margin
        ):
            return candidate_margin > best_margin
    if not np.isclose(candidate["objective_value"], best["objective_value"]):
        return candidate["objective_value"] > best["objective_value"]
    margin_key = "minimum_certified_margin" if confidence_mode else "minimum_margin"
    candidate_margin = candidate.get(margin_key)
    best_margin = best.get(margin_key)
    if candidate_margin is None or best_margin is None:
        return False
    return candidate_margin > best_margin


def _calculate_checkpoint_validation(
    frame,
    scores,
    reference_scores,
    classifier_config,
    objective_config,
    fold_rows,
):
    """Measure checkpoint fitness with held-out validation calibration folds."""
    if fold_rows is None:
        return calculate_hard_constraint_metrics(
            frame,
            scores,
            classifier_config,
            objective_config,
            reference_scores=reference_scores,
        )
    return calculate_cross_fitted_hard_metrics(
        frame,
        scores,
        reference_scores,
        classifier_config,
        objective_config,
        fold_rows,
    )


def _event_model_scores(model, batch):
    scores = batch.features.new_zeros(batch.object_mask.shape)
    scores[batch.object_mask] = model(
        batch.features[batch.object_mask]
    ).reshape(-1)
    return scores


def _event_model_logits(model, batch):
    """Evaluate pre-sigmoid logits while preserving padded event structure."""
    logits = batch.features.new_zeros(batch.object_mask.shape)
    logits[batch.object_mask] = model.forward_logits(
        batch.features[batch.object_mask]
    ).reshape(-1)
    return logits


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
    budget_candidates, surrogate_classifier = resolve_constrained_or_budget(
        classifier_config,
        objective_config,
    )
    if budget_candidates is not None:
        print(
            "Measurement-level TOB budget search enabled: "
            f"{[candidate.tob_fpr for candidate in budget_candidates]} | "
            "surrogate gradients remain NN-only"
        )

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
    validation_crossfit_rows = None
    if objective_config.validation_crossfit:
        validation_crossfit_rows = build_constraint_crossfit_rows(
            validation_frame,
            seed=seed + 40_000,
        )
        print(
            "Validation checkpoint selection: two-fold event cross-fitting "
            "with disjoint calibration and measurement folds."
        )
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
    # Column layout for the model factory, matching the assemble_features
    # hstack order (same construction as train.py). Configs without a model
    # block still build the legacy DynamicMLP through the factory.
    feature_layout = []
    offset = 0
    for name in feature_names:
        width = data_cache.assemble_features(
            dataset, split.train[:1], [name]
        ).shape[1]
        feature_layout.append((name, offset, width))
        offset += width
    model = _build_constrained_model(
        config, X_train.shape[1], feature_layout, device
    )
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
    reference_model = None
    if objective_config.reference_model_allowed_deficits is not None:
        reference_model = copy.deepcopy(model).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    print(f"Loaded constrained initialization: {weights_path}")

    object_batch_size = int(config.get("batch_size", 256))
    constraint_scores = _predict_numpy(
        model,
        X_train[constraint_rows],
        object_batch_size,
        device,
    )
    reference_constraint_scores = constraint_scores.copy()
    constraint_frame = train_frame.iloc[constraint_rows].copy().reset_index(drop=True)
    constraint_crossfit_rows = build_constraint_crossfit_rows(
        constraint_frame,
        seed=seed + 30_000,
    )
    background = select_background_objects(constraint_frame)
    background_positions = constraint_frame.index[
        constraint_frame["Type"].isin(["BKG", "Background"])
    ].to_numpy()
    calibration = calibrate_classifier(
        background,
        constraint_scores[background_positions],
        surrogate_classifier,
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
    balance_generator = torch.Generator().manual_seed(seed + 20_000)
    balance_loader = DataLoader(
        primal_dataset,
        batch_size=objective_config.event_batch_size,
        shuffle=True,
        generator=balance_generator,
        collate_fn=collate_events,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0001)),
    )
    first_temperature = objective_config.temperature_at(0, int(config.get("epochs", 5)))
    initial_fpr_multiplier, gradient_initialization = (
        initialize_fpr_multiplier_from_gradients(
            model,
            (batch.to(device) for batch in balance_loader),
            surrogate_classifier,
            objective_config,
            fixed_nn_threshold,
            fixed_tob_threshold,
            baseline_threshold_gev,
            reference_model=reference_model,
            temperature=first_temperature,
        )
    )
    print(
        "FPR multiplier initialization: "
        f"mode={gradient_initialization['mode']} | "
        f"selected={initial_fpr_multiplier:.6f} | "
        f"gradient ratio={gradient_initialization['recommended_unclipped']}"
    )
    dual_state = DualState(
        multipliers=torch.tensor(
            [
                initial_fpr_multiplier,
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
    tail_memory_bank = HardNegativeMemoryBank(
        objective_config.tail_memory_bank_size
    )
    history = []

    def _measure_constraint_split(scores_array):
        def measure(candidate):
            return calculate_cross_fitted_hard_metrics(
                constraint_frame,
                scores_array,
                reference_constraint_scores if reference_model is not None else None,
                candidate,
                objective_config,
                constraint_crossfit_rows,
            )
        if budget_candidates is None:
            return measure(classifier_config)
        return budget_searched_metrics(measure, budget_candidates)

    def _measure_validation(scores_array, reference_scores_array):
        def measure(candidate):
            return _calculate_checkpoint_validation(
                validation_frame,
                scores_array,
                reference_scores_array,
                candidate,
                objective_config,
                validation_crossfit_rows,
            )
        if budget_candidates is None:
            return measure(classifier_config)
        return budget_searched_metrics(measure, budget_candidates)

    initial_constraint = _measure_constraint_split(constraint_scores)
    initial_validation_scores = _predict_numpy(
        model,
        X_val,
        object_batch_size,
        device,
    )
    initial_validation = _measure_validation(
        initial_validation_scores,
        initial_validation_scores,
    )
    initial_resolution_warnings = {
        "constraint_training": constraint_resolution_warnings(
            initial_constraint,
            objective_config.regions_gev,
        ),
        "validation": constraint_resolution_warnings(
            initial_validation,
            objective_config.regions_gev,
        ),
    }
    for split_name, messages in initial_resolution_warnings.items():
        for message in messages:
            print(f"Constraint resolution warning ({split_name}): {message}")
    history.append(
        {
            "epoch": 0,
            "training_loss": None,
            "training_bce": None,
            "dual_state": dual_state.to_dict(),
            "constraint_training": initial_constraint,
            "validation": initial_validation,
            "diagnostics": {
                "gradient_initialization": gradient_initialization,
                "constraint_resolution_warnings": initial_resolution_warnings,
                "score_quantiles": _score_quantiles(
                    constraint_frame,
                    constraint_scores,
                ),
                "regional_gradient_coverage": regional_gradient_diagnostics(
                    constraint_frame,
                    constraint_scores,
                    surrogate_classifier,
                    objective_config,
                    first_temperature,
                ),
            },
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
        temperature = objective_config.temperature_at(epoch, epochs)
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_soft_objective = 0.0
        epoch_soft_fpr = 0.0
        epoch_ranking_loss = 0.0
        epoch_tail_events = 0.0
        epoch_gradient_norms = None
        batch_count = 0
        for batch in primal_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            scores, metrics = _soft_batch_metrics(
                model,
                batch,
                surrogate_classifier,
                objective_config,
                fixed_nn_threshold,
                fixed_tob_threshold,
                baseline_threshold_gev,
                reference_model=reference_model,
                temperature=temperature,
                tail_memory_bank=tail_memory_bank,
                update_memory=True,
            )
            if epoch_gradient_norms is None:
                parameters = tuple(
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                objective_norm, fpr_norm, gradient_cosine = (
                    parameter_gradient_pair_statistics(
                        metrics.objective,
                        metrics.violations[0],
                        parameters,
                        retain_graph=True,
                    )
                )
                epoch_gradient_norms = {
                    "objective": float(objective_norm.detach().cpu()),
                    "event_fpr": float(fpr_norm.detach().cpu()),
                    "fpr_violation_scale": objective_config.fpr_violation_scale,
                    "weighted_event_fpr": float(
                        dual_state.multipliers[0].detach().cpu()
                        * fpr_norm.detach().cpu()
                    ),
                    "cosine_similarity": float(
                        gradient_cosine.detach().cpu()
                    ),
                }
            loss = constrained_primal_loss(metrics, dual_state)
            loss.backward()
            optimizer.step()

            valid_scores = scores[batch.object_mask]
            valid_labels = batch.labels[batch.object_mask].reshape(-1)
            epoch_bce += bce_monitor(valid_scores, valid_labels).item()
            epoch_loss += loss.item()
            epoch_soft_objective += float(metrics.objective.detach().cpu())
            epoch_soft_fpr += float(metrics.event_fpr.detach().cpu())
            if metrics.ranking_loss is not None:
                epoch_ranking_loss += float(metrics.ranking_loss.detach().cpu())
                epoch_tail_events += float(metrics.tail_event_count)
            batch_count += 1

        constraint_scores = _predict_numpy(
            model,
            X_train[constraint_rows],
            object_batch_size,
            device,
        )
        hard_constraint = _measure_constraint_split(constraint_scores)
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
                objective_config.fpr_dual_learning_rate,
                objective_config.max_multiplier,
                region_learning_rate=(
                    objective_config.region_dual_learning_rate
                ),
            )

        validation_scores = _predict_numpy(
            model,
            X_val,
            object_batch_size,
            device,
        )
        hard_validation = _measure_validation(
            validation_scores,
            initial_validation_scores,
        )
        record = {
            "epoch": epoch + 1,
            "training_loss": float(epoch_loss / max(batch_count, 1)),
            "training_bce": float(epoch_bce / max(batch_count, 1)),
            "dual_state": dual_state.to_dict(),
            "constraint_training": hard_constraint,
            "validation": hard_validation,
            "diagnostics": {
                "soft_training": {
                    "objective_mean": float(
                        epoch_soft_objective / max(batch_count, 1)
                    ),
                    "event_fpr_mean": float(
                        epoch_soft_fpr / max(batch_count, 1)
                    ),
                    "ranking_loss_mean": (
                        None
                        if objective_config.primal_objective != "tail_ranking"
                        else float(epoch_ranking_loss / max(batch_count, 1))
                    ),
                    "tail_event_count_mean": (
                        None
                        if objective_config.primal_objective != "tail_ranking"
                        else float(epoch_tail_events / max(batch_count, 1))
                    ),
                    "temperature": float(temperature),
                    "memory_bank_size": len(tail_memory_bank),
                },
                "gradient_norms_first_batch": epoch_gradient_norms,
                "score_quantiles": _score_quantiles(
                    constraint_frame,
                    constraint_scores,
                ),
                "regional_gradient_coverage": regional_gradient_diagnostics(
                    constraint_frame,
                    constraint_scores,
                    surrogate_classifier,
                    objective_config,
                    temperature,
                ),
            },
        }
        history.append(record)
        budget_note = (
            ""
            if "selected_tob_fpr" not in hard_validation
            else f" | b*={hard_validation['selected_tob_fpr']}"
        )
        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Constrained loss: {record['training_loss']:.5f} | "
            f"Val J: {hard_validation['objective_value']:.5f} | "
            f"feasible={hard_validation['constraints_satisfied']}"
            f"{budget_note} | "
            f"lambdas={dual_state.to_dict()['multipliers']}"
        )
        if _is_better_hard_candidate(hard_validation, best_record):
            best_record = copy.deepcopy(hard_validation)
            best_record["epoch"] = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            best_dual = copy.deepcopy(dual_state.to_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())

    last_epoch_weights = copy.deepcopy(model.state_dict())
    last_weights_path = Path(tracker.experiment_dir) / "last_epoch_weights.pt"
    torch.save(last_epoch_weights, last_weights_path)
    model.load_state_dict(best_weights)
    if validation_crossfit_rows is not None:
        cross_fitted_selection = copy.deepcopy(best_record)
        selected_validation_scores = _predict_numpy(
            model,
            X_val,
            object_batch_size,
            device,
        )
        def _final_measure(candidate):
            return calculate_hard_constraint_metrics(
                validation_frame,
                selected_validation_scores,
                candidate,
                objective_config,
                reference_scores=initial_validation_scores,
            )
        if budget_candidates is None:
            final_validation = _final_measure(classifier_config)
        else:
            final_validation = budget_searched_metrics(
                _final_measure,
                budget_candidates,
            )
        final_validation["epoch"] = int(cross_fitted_selection["epoch"])
        final_validation["cross_fitted_selection"] = cross_fitted_selection
        final_validation["selection_protocol"] = {
            "validation_crossfit": True,
            "folds": int(objective_config.crossfit_folds),
            "final_calibration_split": "complete_validation_after_selection",
        }
        best_record = final_validation
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
            "gradient_initialization": gradient_initialization,
            "initial_constraint_resolution_warnings": (
                initial_resolution_warnings
            ),
            "best_validation_record": best_record,
            "history": history,
            "artifacts": {
                "weights": "model_weights.pt",
                "last_epoch_weights": last_weights_path.name,
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
                    "last_epoch_weights": last_weights_path.name,
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
