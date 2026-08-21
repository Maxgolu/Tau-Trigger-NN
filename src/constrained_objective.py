"""Differentiable trigger objectives shared by NN-only and OR training."""

from dataclasses import dataclass

import numpy as np
import torch

from classifiers import (
    calibrate_classifier,
    classifier_event_pass_mask,
    classifier_object_pass_mask,
    tob_pt_gev,
)
from operating_point import (
    build_event_trigger_scores,
    select_background_objects,
    select_fpr_threshold,
    select_truth_tau_objects,
)


@dataclass(frozen=True)
class ConstrainedObjectiveConfig:
    """Physics targets and optimizer settings for constrained fine-tuning."""

    temperature: float
    target_event_fpr: float
    trigger_objects: int
    regions_gev: tuple[tuple[float, float], ...]
    region_weights: tuple[float, ...]
    allowed_deficits: tuple[float, ...]
    minimum_region_advantages: tuple[float, ...]
    reference_model_allowed_deficits: tuple[float, ...] | None
    constraint_fraction: float
    dual_learning_rate: float
    dual_update_frequency: int
    dual_warmup_epochs: int
    initial_fpr_multiplier_mode: str
    initial_fpr_multiplier: float
    initial_region_multiplier: float
    max_multiplier: float
    event_batch_size: int
    gradient_balance_batches: int
    gradient_balance_epsilon: float

    def to_dict(self):
        return {
            "temperature": self.temperature,
            "target_event_fpr": self.target_event_fpr,
            "trigger_objects": self.trigger_objects,
            "regions_gev": [list(region) for region in self.regions_gev],
            "region_weights": list(self.region_weights),
            "allowed_deficits": list(self.allowed_deficits),
            "minimum_region_advantages": list(self.minimum_region_advantages),
            "reference_model_allowed_deficits": (
                None
                if self.reference_model_allowed_deficits is None
                else list(self.reference_model_allowed_deficits)
            ),
            "constraint_fraction": self.constraint_fraction,
            "dual_learning_rate": self.dual_learning_rate,
            "dual_update_frequency": self.dual_update_frequency,
            "dual_warmup_epochs": self.dual_warmup_epochs,
            "initial_fpr_multiplier_mode": self.initial_fpr_multiplier_mode,
            "initial_fpr_multiplier": self.initial_fpr_multiplier,
            "initial_region_multiplier": self.initial_region_multiplier,
            "max_multiplier": self.max_multiplier,
            "event_batch_size": self.event_batch_size,
            "gradient_balance_batches": self.gradient_balance_batches,
            "gradient_balance_epsilon": self.gradient_balance_epsilon,
        }


@dataclass
class SoftConstraintMetrics:
    """Differentiable objective values for one event batch."""

    objective: torch.Tensor
    event_fpr: torch.Tensor
    region_efficiencies: torch.Tensor
    baseline_efficiencies: torch.Tensor
    region_deltas: torch.Tensor
    violations: torch.Tensor
    valid_regions: torch.Tensor
    reference_efficiencies: torch.Tensor | None = None
    required_efficiencies: torch.Tensor | None = None
    region_margins: torch.Tensor | None = None


def parse_constrained_objective(config):
    """Validate constrained-loss settings without changing legacy defaults."""
    raw = config.get("loss", {})
    if raw.get("name") != "constrained_trigger":
        raise ValueError("Expected loss.name='constrained_trigger'")

    regions = tuple(
        (float(region[0]), float(region[1]))
        for region in raw.get(
            "regions_gev",
            ((25.0, 40.0), (40.0, 60.0), (60.0, 120.0)),
        )
    )
    if not regions or any(low >= high for low, high in regions):
        raise ValueError("Every constrained region must satisfy low < high")

    weights = tuple(
        float(value)
        for value in raw.get(
            "region_weights",
            [1.0 / len(regions)] * len(regions),
        )
    )
    deficits = tuple(
        float(value)
        for value in raw.get("allowed_deficits", [0.005] * len(regions))
    )
    raw_advantages = raw.get("minimum_region_advantages")
    advantages = (
        tuple(-value for value in deficits)
        if raw_advantages is None
        else tuple(float(value) for value in raw_advantages)
    )
    raw_reference_deficits = raw.get("reference_model_allowed_deficits")
    reference_deficits = (
        None
        if raw_reference_deficits is None
        else tuple(float(value) for value in raw_reference_deficits)
    )
    if (
        len(weights) != len(regions)
        or len(deficits) != len(regions)
        or len(advantages) != len(regions)
        or (
            reference_deficits is not None
            and len(reference_deficits) != len(regions)
        )
    ):
        raise ValueError("Regions, weights, and deficits must have equal lengths")
    if any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("Constrained region weights must be non-negative and sum to one")
    if any(value < 0.0 for value in deficits):
        raise ValueError("Constrained allowed deficits must be non-negative")
    if reference_deficits is not None and any(
        value < 0.0 for value in reference_deficits
    ):
        raise ValueError("Reference-model allowed deficits must be non-negative")

    result = ConstrainedObjectiveConfig(
        temperature=float(raw.get("temperature", 0.02)),
        target_event_fpr=float(raw.get("target_event_fpr", 0.005)),
        trigger_objects=int(raw.get("trigger_objects", 2)),
        regions_gev=regions,
        region_weights=weights,
        allowed_deficits=deficits,
        minimum_region_advantages=advantages,
        reference_model_allowed_deficits=reference_deficits,
        constraint_fraction=float(raw.get("constraint_fraction", 0.3)),
        dual_learning_rate=float(raw.get("dual_learning_rate", 1.0)),
        dual_update_frequency=int(raw.get("dual_update_frequency", 1)),
        dual_warmup_epochs=int(raw.get("dual_warmup_epochs", 0)),
        initial_fpr_multiplier_mode=str(
            raw.get("initial_fpr_multiplier_mode", "fixed")
        ),
        initial_fpr_multiplier=float(raw.get("initial_fpr_multiplier", 1.0)),
        initial_region_multiplier=float(raw.get("initial_region_multiplier", 0.0)),
        max_multiplier=float(raw.get("max_multiplier", 10.0)),
        event_batch_size=int(raw.get("event_batch_size", 512)),
        gradient_balance_batches=int(raw.get("gradient_balance_batches", 8)),
        gradient_balance_epsilon=float(raw.get("gradient_balance_epsilon", 1e-12)),
    )
    if result.temperature <= 0.0:
        raise ValueError("Constrained temperature must be positive")
    if not 0.0 < result.target_event_fpr <= 1.0:
        raise ValueError("Constrained target_event_fpr must be in (0, 1]")
    if result.trigger_objects < 1:
        raise ValueError("Constrained trigger_objects must be positive")
    if not 0.0 < result.constraint_fraction < 1.0:
        raise ValueError("constraint_fraction must be in (0, 1)")
    if result.dual_learning_rate <= 0.0:
        raise ValueError("dual_learning_rate must be positive")
    if result.dual_update_frequency < 1 or result.dual_warmup_epochs < 0:
        raise ValueError("Invalid constrained dual update schedule")
    if result.initial_fpr_multiplier_mode not in {"fixed", "gradient_balance"}:
        raise ValueError(
            "initial_fpr_multiplier_mode must be 'fixed' or 'gradient_balance'"
        )
    if result.initial_fpr_multiplier < 0.0 or result.initial_region_multiplier < 0.0:
        raise ValueError("Initial constrained multipliers must be non-negative")
    if result.max_multiplier <= 0.0 or result.event_batch_size < 1:
        raise ValueError("Invalid constrained multiplier cap or event batch size")
    if result.gradient_balance_batches < 1 or result.gradient_balance_epsilon <= 0.0:
        raise ValueError("Invalid constrained gradient-balance settings")
    return result


def soft_nn_pass(scores, threshold, temperature):
    """Approximate a score cut while retaining useful boundary gradients."""
    return torch.sigmoid((scores - threshold) / temperature)


def soft_object_pass(
    scores,
    threshold,
    temperature,
    classifier_name,
    tob_pt_gev_values=None,
    tob_threshold_gev=None,
):
    """Return soft object decisions for either supported classifier."""
    nn_pass = soft_nn_pass(scores, threshold, temperature)
    if classifier_name == "nn_only":
        return nn_pass
    if classifier_name != "tob_nn_or":
        raise ValueError(f"Unsupported constrained classifier: {classifier_name}")
    if tob_pt_gev_values is None or tob_threshold_gev is None:
        raise ValueError("OR constrained training requires TOB values and threshold")

    # The first controlled OR experiment keeps the comparator branch hard.
    tob_pass = (tob_pt_gev_values >= tob_threshold_gev).to(nn_pass.dtype)
    return tob_pass + (1.0 - tob_pass) * nn_pass


def probability_at_least_k(object_probabilities, object_mask, k=2):
    """Compute a differentiable Poisson-binomial tail for padded events."""
    if object_probabilities.shape != object_mask.shape:
        raise ValueError("Object probabilities and mask must have equal shapes")
    if object_probabilities.ndim != 2 or k < 1:
        raise ValueError("Expected [events, objects] probabilities and positive k")

    event_count = object_probabilities.shape[0]
    probabilities_below_k = object_probabilities.new_zeros((event_count, k))
    probabilities_below_k[:, 0] = 1.0

    for index in range(object_probabilities.shape[1]):
        probability = object_probabilities[:, index]
        valid = object_mask[:, index]
        updated = probabilities_below_k.clone()
        updated[:, 0] = probabilities_below_k[:, 0] * (1.0 - probability)
        for count in range(1, k):
            updated[:, count] = (
                probabilities_below_k[:, count] * (1.0 - probability)
                + probabilities_below_k[:, count - 1] * probability
            )
        probabilities_below_k = torch.where(
            valid.unsqueeze(1),
            updated,
            probabilities_below_k,
        )

    return (1.0 - probabilities_below_k.sum(dim=1)).clamp(0.0, 1.0)


def calculate_soft_constraint_metrics(
    object_pass_probabilities,
    object_mask,
    signal_object_mask,
    background_event_mask,
    truth_pt_gev,
    baseline_object_pass,
    objective_config,
    reference_object_pass_probabilities=None,
):
    """Calculate the differentiable objective and constraint violations."""
    event_pass = probability_at_least_k(
        object_pass_probabilities,
        object_mask,
        k=objective_config.trigger_objects,
    )
    if not torch.any(background_event_mask):
        raise ValueError("Every constrained batch must contain background events")
    event_fpr = event_pass[background_event_mask].mean()

    efficiencies = []
    baseline_efficiencies = []
    reference_efficiencies = []
    valid_regions = []
    for low, high in objective_config.regions_gev:
        in_region = (
            signal_object_mask
            & object_mask
            & (truth_pt_gev >= low)
            & (truth_pt_gev < high)
        )
        count = in_region.sum()
        valid = count > 0
        denominator = count.clamp(min=1).to(object_pass_probabilities.dtype)
        efficiencies.append(
            (object_pass_probabilities * in_region).sum() / denominator
        )
        baseline_efficiencies.append(
            (baseline_object_pass.to(object_pass_probabilities.dtype) * in_region).sum()
            / denominator
        )
        if reference_object_pass_probabilities is not None:
            reference_efficiencies.append(
                (reference_object_pass_probabilities * in_region).sum()
                / denominator
            )
        valid_regions.append(valid)

    efficiencies = torch.stack(efficiencies)
    baseline_efficiencies = torch.stack(baseline_efficiencies)
    reference_efficiencies = (
        None
        if reference_object_pass_probabilities is None
        else torch.stack(reference_efficiencies)
    )
    valid_regions = torch.stack(valid_regions)
    deltas = efficiencies - baseline_efficiencies
    weights = object_pass_probabilities.new_tensor(objective_config.region_weights)
    active_weights = weights * valid_regions.to(weights.dtype)
    active_weights = active_weights / active_weights.sum().clamp(min=1e-12)
    objective = torch.sum(active_weights * deltas)

    minimum_advantages = object_pass_probabilities.new_tensor(
        objective_config.minimum_region_advantages
    )
    required_efficiencies = baseline_efficiencies + minimum_advantages
    if objective_config.reference_model_allowed_deficits is not None:
        if reference_efficiencies is None:
            raise ValueError("Reference-model guards require reference probabilities")
        reference_deficits = object_pass_probabilities.new_tensor(
            objective_config.reference_model_allowed_deficits
        )
        required_efficiencies = torch.maximum(
            required_efficiencies,
            reference_efficiencies - reference_deficits,
        )
    # No classifier can exceed unit efficiency in a saturated region.
    required_efficiencies = required_efficiencies.clamp(max=1.0)
    region_margins = efficiencies - required_efficiencies
    violations = torch.cat(
        (
            (event_fpr - objective_config.target_event_fpr).reshape(1),
            -region_margins,
        )
    )
    return SoftConstraintMetrics(
        objective=objective,
        event_fpr=event_fpr,
        region_efficiencies=efficiencies,
        baseline_efficiencies=baseline_efficiencies,
        region_deltas=deltas,
        violations=violations,
        valid_regions=valid_regions,
        reference_efficiencies=reference_efficiencies,
        required_efficiencies=required_efficiencies,
        region_margins=region_margins,
    )


def calibrate_tob_baseline(background, target_fpr, trigger_objects=2):
    """Calibrate the comparison TOB threshold on background events only."""
    frame = background.copy()
    frame["_tob_pt_gev"] = tob_pt_gev(frame)
    scores, event_count = build_event_trigger_scores(
        frame,
        "_tob_pt_gev",
        objects=trigger_objects,
    )
    return select_fpr_threshold(scores, event_count, target_fpr)


def calculate_hard_constraint_metrics(
    frame,
    scores,
    classifier_config,
    objective_config,
    calibration=None,
    baseline_threshold_gev=None,
    reference_scores=None,
    reference_calibration=None,
):
    """Measure the exact deployable objective for validation and audits."""
    scored = frame.copy()
    scored["nn_score"] = np.asarray(scores, dtype=np.float64)
    background = select_background_objects(scored)
    signal = select_truth_tau_objects(scored)
    if calibration is None:
        calibration = calibrate_classifier(
            background,
            background["nn_score"].to_numpy(dtype=np.float64),
            classifier_config,
        )
    if baseline_threshold_gev is None:
        baseline_threshold_gev, _ = calibrate_tob_baseline(
            background,
            objective_config.target_event_fpr,
            objective_config.trigger_objects,
        )

    signal_pass = classifier_object_pass_mask(signal, calibration)
    baseline_pass = tob_pt_gev(signal) >= baseline_threshold_gev
    reference_pass = None
    if objective_config.reference_model_allowed_deficits is not None:
        if reference_scores is None:
            raise ValueError("Reference-model guards require reference scores")
        reference_scored = frame.copy()
        reference_scored["nn_score"] = np.asarray(reference_scores, dtype=np.float64)
        reference_background = select_background_objects(reference_scored)
        reference_signal = select_truth_tau_objects(reference_scored)
        if reference_calibration is None:
            reference_calibration = calibrate_classifier(
                reference_background,
                reference_background["nn_score"].to_numpy(dtype=np.float64),
                classifier_config,
            )
        reference_pass = classifier_object_pass_mask(
            reference_signal,
            reference_calibration,
        )
    truth_pt = signal["truth_pt"].to_numpy(dtype=np.float64)
    finite = truth_pt[np.isfinite(truth_pt) & (truth_pt > 0.0)]
    if finite.size and np.median(finite) > 1000.0:
        truth_pt = truth_pt / 1000.0

    efficiencies = []
    baseline_efficiencies = []
    reference_efficiencies = []
    required_efficiencies = []
    deltas = []
    counts = []
    for low, high in objective_config.regions_gev:
        in_region = (truth_pt >= low) & (truth_pt < high)
        counts.append(int(np.count_nonzero(in_region)))
        if not np.any(in_region):
            efficiencies.append(None)
            baseline_efficiencies.append(None)
            reference_efficiencies.append(None)
            required_efficiencies.append(None)
            deltas.append(None)
            continue
        efficiency = float(signal_pass[in_region].mean())
        baseline_efficiency = float(baseline_pass[in_region].mean())
        efficiencies.append(efficiency)
        baseline_efficiencies.append(baseline_efficiency)
        reference_efficiency = (
            None if reference_pass is None else float(reference_pass[in_region].mean())
        )
        reference_efficiencies.append(reference_efficiency)
        required = baseline_efficiency + objective_config.minimum_region_advantages[
            len(efficiencies) - 1
        ]
        if reference_efficiency is not None:
            required = max(
                required,
                reference_efficiency
                - objective_config.reference_model_allowed_deficits[
                    len(efficiencies) - 1
                ],
            )
        required = min(required, 1.0)
        required_efficiencies.append(required)
        deltas.append(efficiency - baseline_efficiency)

    valid = np.asarray([value is not None for value in deltas])
    delta_values = np.asarray(
        [0.0 if value is None else value for value in deltas],
        dtype=np.float64,
    )
    weights = np.asarray(objective_config.region_weights, dtype=np.float64)
    weights = weights * valid
    weights = weights / weights.sum() if weights.sum() else weights
    objective = float(np.sum(weights * delta_values))
    margins = np.asarray(
        [
            0.0 if required is None else efficiency - required
            for efficiency, required in zip(efficiencies, required_efficiencies)
        ],
        dtype=np.float64,
    )

    event_pass = classifier_event_pass_mask(background, calibration)
    achieved_fpr = float(event_pass.mean())
    return {
        "objective_value": objective,
        "achieved_fpr": achieved_fpr,
        "region_efficiencies": efficiencies,
        "baseline_efficiencies": baseline_efficiencies,
        "reference_efficiencies": reference_efficiencies,
        "required_efficiencies": required_efficiencies,
        "region_deltas": deltas,
        "region_counts": counts,
        "constraint_margins": margins.tolist(),
        "constraints_satisfied": bool(
            achieved_fpr <= objective_config.target_event_fpr + 1e-12
            and np.all(margins[valid] >= 0.0)
        ),
        "minimum_margin": float(np.min(margins[valid])) if np.any(valid) else None,
        "classifier_calibration": calibration,
        "baseline_threshold_gev": float(baseline_threshold_gev),
    }
