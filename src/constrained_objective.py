"""Differentiable trigger objectives shared by NN-only and OR training."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta, norm

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

    temperature_start: float
    temperature_end: float
    temperature_schedule: str
    target_event_fpr: float
    trigger_objects: int
    primal_objective: str
    proxy_threshold_mode: str
    objective_regions_gev: tuple[tuple[float, float], ...]
    objective_region_weights: tuple[float, ...]
    constraint_regions_gev: tuple[tuple[float, float], ...]
    allowed_deficits: tuple[float, ...]
    minimum_region_advantages: tuple[float, ...]
    reference_model_allowed_deficits: tuple[float, ...] | None
    tail_fraction: float
    tail_temperature: float
    tail_min_events: int
    tail_memory_bank_size: int
    constraint_fraction: float
    crossfit_folds: int
    validation_crossfit: bool
    feasibility_confidence_level: float | None
    fpr_feasibility_mode: str
    certified_guards_use_allowed_deficits: bool
    fpr_violation_scale: float
    fpr_dual_learning_rate: float
    region_dual_learning_rate: float
    dual_update_frequency: int
    dual_warmup_epochs: int
    initial_fpr_multiplier_mode: str
    initial_fpr_multiplier: float
    initial_region_multiplier: float
    max_multiplier: float
    event_batch_size: int
    gradient_balance_batches: int
    gradient_balance_epsilon: float

    @property
    def temperature(self):
        """Backward-compatible final surrogate temperature."""
        return self.temperature_end

    @property
    def regions_gev(self):
        """Backward-compatible name for protected regions."""
        return self.constraint_regions_gev

    @property
    def region_weights(self):
        """Backward-compatible name for objective-only weights."""
        return self.objective_region_weights

    def temperature_at(self, epoch_index, epoch_count):
        """Return the configured continuation temperature for one epoch."""
        if epoch_count <= 1 or self.temperature_schedule == "constant":
            return self.temperature_end
        fraction = min(max(float(epoch_index) / float(epoch_count - 1), 0.0), 1.0)
        if self.temperature_schedule == "cosine":
            fraction = 0.5 * (1.0 - math.cos(math.pi * fraction))
        return self.temperature_start + fraction * (
            self.temperature_end - self.temperature_start
        )

    def to_dict(self):
        return {
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
            "temperature_schedule": self.temperature_schedule,
            "target_event_fpr": self.target_event_fpr,
            "trigger_objects": self.trigger_objects,
            "primal_objective": self.primal_objective,
            "proxy_threshold_mode": self.proxy_threshold_mode,
            "objective_regions_gev": [
                list(region) for region in self.objective_regions_gev
            ],
            "objective_region_weights": list(self.objective_region_weights),
            "constraint_regions_gev": [
                list(region) for region in self.constraint_regions_gev
            ],
            "allowed_deficits": list(self.allowed_deficits),
            "minimum_region_advantages": list(self.minimum_region_advantages),
            "reference_model_allowed_deficits": (
                None
                if self.reference_model_allowed_deficits is None
                else list(self.reference_model_allowed_deficits)
            ),
            "tail_fraction": self.tail_fraction,
            "tail_temperature": self.tail_temperature,
            "tail_min_events": self.tail_min_events,
            "tail_memory_bank_size": self.tail_memory_bank_size,
            "constraint_fraction": self.constraint_fraction,
            "crossfit_folds": self.crossfit_folds,
            "validation_crossfit": self.validation_crossfit,
            "feasibility_confidence_level": self.feasibility_confidence_level,
            "fpr_feasibility_mode": self.fpr_feasibility_mode,
            "certified_guards_use_allowed_deficits": (
                self.certified_guards_use_allowed_deficits
            ),
            "fpr_violation_scale": self.fpr_violation_scale,
            "fpr_dual_learning_rate": self.fpr_dual_learning_rate,
            "region_dual_learning_rate": self.region_dual_learning_rate,
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
    objective_region_efficiencies: torch.Tensor | None = None
    objective_baseline_efficiencies: torch.Tensor | None = None
    objective_region_deltas: torch.Tensor | None = None
    ranking_loss: torch.Tensor | None = None
    tail_event_count: int | None = None
    current_tail_offsets: torch.Tensor | None = None


def _parse_regions(values, option):
    regions = tuple((float(item[0]), float(item[1])) for item in values)
    if not regions or any(low >= high for low, high in regions):
        raise ValueError(f"Every {option} region must satisfy low < high")
    return regions


def parse_constrained_objective(config):
    """Validate constrained-loss settings without changing legacy defaults."""
    raw = config.get("loss", {})
    if raw.get("name") != "constrained_trigger":
        raise ValueError("Expected loss.name='constrained_trigger'")

    legacy_regions = raw.get(
        "regions_gev",
        ((25.0, 40.0), (40.0, 60.0), (60.0, 120.0)),
    )
    objective_regions = _parse_regions(
        raw.get("objective_regions_gev", legacy_regions), "objective"
    )
    constraint_regions = _parse_regions(
        raw.get("constraint_regions_gev", legacy_regions), "constraint"
    )
    weights = tuple(
        float(value)
        for value in raw.get(
            "objective_region_weights",
            raw.get(
                "region_weights",
                [1.0 / len(objective_regions)] * len(objective_regions),
            ),
        )
    )
    deficits = tuple(
        float(value)
        for value in raw.get("allowed_deficits", [0.005] * len(constraint_regions))
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
    if len(weights) != len(objective_regions):
        raise ValueError("Objective regions and weights must have equal lengths")
    if (
        len(deficits) != len(constraint_regions)
        or len(advantages) != len(constraint_regions)
        or (
            reference_deficits is not None
            and len(reference_deficits) != len(constraint_regions)
        )
    ):
        raise ValueError("Constraint regions, advantages, and deficits must align")
    if any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("Objective region weights must be non-negative and sum to one")
    if any(value < 0.0 for value in deficits):
        raise ValueError("Constrained allowed deficits must be non-negative")
    if reference_deficits is not None and any(
        value < 0.0 for value in reference_deficits
    ):
        raise ValueError("Reference-model allowed deficits must be non-negative")

    temperature = float(raw.get("temperature", 0.02))
    legacy_dual_learning_rate = float(raw.get("dual_learning_rate", 1.0))
    result = ConstrainedObjectiveConfig(
        temperature_start=float(raw.get("temperature_start", temperature)),
        temperature_end=float(raw.get("temperature_end", temperature)),
        temperature_schedule=str(raw.get("temperature_schedule", "constant")),
        target_event_fpr=float(raw.get("target_event_fpr", 0.005)),
        trigger_objects=int(raw.get("trigger_objects", 2)),
        primal_objective=str(raw.get("primal_objective", "soft_efficiency")),
        proxy_threshold_mode=str(raw.get("proxy_threshold_mode", "fixed")),
        objective_regions_gev=objective_regions,
        objective_region_weights=weights,
        constraint_regions_gev=constraint_regions,
        allowed_deficits=deficits,
        minimum_region_advantages=advantages,
        reference_model_allowed_deficits=reference_deficits,
        tail_fraction=float(raw.get("tail_fraction", 0.05)),
        tail_temperature=float(raw.get("tail_temperature", 0.2)),
        tail_min_events=int(raw.get("tail_min_events", 16)),
        tail_memory_bank_size=int(raw.get("tail_memory_bank_size", 0)),
        constraint_fraction=float(raw.get("constraint_fraction", 0.3)),
        crossfit_folds=int(raw.get("crossfit_folds", 2)),
        validation_crossfit=bool(raw.get("validation_crossfit", False)),
        feasibility_confidence_level=(
            None
            if raw.get("feasibility_confidence_level") is None
            else float(raw["feasibility_confidence_level"])
        ),
        fpr_feasibility_mode=str(raw.get("fpr_feasibility_mode", "certified")),
        certified_guards_use_allowed_deficits=bool(
            raw.get("certified_guards_use_allowed_deficits", False)
        ),
        fpr_violation_scale=float(raw.get("fpr_violation_scale", 1.0)),
        fpr_dual_learning_rate=float(
            raw.get("fpr_dual_learning_rate", legacy_dual_learning_rate)
        ),
        region_dual_learning_rate=float(
            raw.get("region_dual_learning_rate", legacy_dual_learning_rate)
        ),
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
    if result.temperature_start <= 0.0 or result.temperature_end <= 0.0:
        raise ValueError("Constrained temperatures must be positive")
    if result.temperature_schedule not in {"constant", "linear", "cosine"}:
        raise ValueError("temperature_schedule must be constant, linear, or cosine")
    if result.primal_objective not in {"soft_efficiency", "tail_ranking"}:
        raise ValueError("primal_objective must be soft_efficiency or tail_ranking")
    if result.proxy_threshold_mode not in {"fixed", "batch_rank"}:
        raise ValueError("proxy_threshold_mode must be fixed or batch_rank")
    if not 0.0 < result.target_event_fpr <= 1.0:
        raise ValueError("Constrained target_event_fpr must be in (0, 1]")
    if result.trigger_objects < 1:
        raise ValueError("Constrained trigger_objects must be positive")
    if not 0.0 < result.tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if result.tail_temperature <= 0.0 or result.tail_min_events < 1:
        raise ValueError("Tail temperature and minimum event count must be positive")
    if result.tail_memory_bank_size < 0:
        raise ValueError("tail_memory_bank_size cannot be negative")
    if not 0.0 < result.constraint_fraction < 1.0:
        raise ValueError("constraint_fraction must be in (0, 1)")
    if result.crossfit_folds != 2:
        raise ValueError("Hard constraint cross-fitting currently requires two folds")
    if (
        result.feasibility_confidence_level is not None
        and not 0.5 < result.feasibility_confidence_level < 1.0
    ):
        raise ValueError("feasibility_confidence_level must be in (0.5, 1)")
    if result.fpr_feasibility_mode not in {"certified", "point"}:
        raise ValueError("fpr_feasibility_mode must be 'certified' or 'point'")
    if result.fpr_violation_scale <= 0.0:
        raise ValueError("fpr_violation_scale must be positive")
    if result.fpr_dual_learning_rate <= 0.0 or result.region_dual_learning_rate <= 0.0:
        raise ValueError("Constrained dual learning rates must be positive")
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
    tob_pass = (tob_pt_gev_values >= tob_threshold_gev).to(nn_pass.dtype)
    return tob_pass + (1.0 - tob_pass) * nn_pass


def kth_event_score(object_scores, object_mask, k=2):
    """Return the k-th largest valid object score for every complete event."""
    if object_scores.shape != object_mask.shape or object_scores.ndim != 2:
        raise ValueError("Expected aligned [events, objects] scores and mask")
    if k < 1 or object_scores.shape[1] < k:
        raise ValueError("The requested event rank is unavailable")
    masked = object_scores.masked_fill(~object_mask, -torch.inf)
    result = torch.topk(masked, k=k, dim=1).values[:, -1]
    # Events with fewer than k objects can never fire and therefore receive
    # negative infinity rather than aborting an otherwise valid batch.
    return result.masked_fill(object_mask.sum(dim=1) < k, -torch.inf)


def rank_calibrated_threshold(
    object_logits,
    object_mask,
    background_event_mask,
    target_fpr,
    trigger_objects=2,
    temperature=0.02,
):
    """Select a batch threshold by rank, preserving additive-shift invariance."""
    event_scores = kth_event_score(object_logits, object_mask, trigger_objects)
    background_scores = event_scores[background_event_mask]
    background_event_count = int(background_scores.numel())
    background_scores = background_scores[torch.isfinite(background_scores)]
    if not background_scores.numel():
        raise ValueError("Rank calibration requires background events")
    accepted = int(math.floor(target_fpr * background_event_count))
    if accepted < 1:
        return background_scores.max() + 8.0 * float(temperature)
    return torch.topk(background_scores, k=accepted).values[-1]


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
            valid.unsqueeze(1), updated, probabilities_below_k
        )
    return (1.0 - probabilities_below_k.sum(dim=1)).clamp(0.0, 1.0)


def _soft_region_values(
    object_pass_probabilities,
    object_mask,
    signal_object_mask,
    truth_pt_gev,
    comparison_object_pass,
    regions,
):
    efficiencies = []
    comparisons = []
    valid_regions = []
    for low, high in regions:
        in_region = (
            signal_object_mask
            & object_mask
            & (truth_pt_gev >= low)
            & (truth_pt_gev < high)
        )
        count = in_region.sum()
        denominator = count.clamp(min=1).to(object_pass_probabilities.dtype)
        efficiencies.append((object_pass_probabilities * in_region).sum() / denominator)
        comparisons.append(
            (comparison_object_pass.to(object_pass_probabilities.dtype) * in_region).sum()
            / denominator
        )
        valid_regions.append(count > 0)
    return torch.stack(efficiencies), torch.stack(comparisons), torch.stack(valid_regions)


def tail_ranking_objective(
    object_logits,
    object_mask,
    signal_object_mask,
    background_event_mask,
    truth_pt_gev,
    objective_config,
    memory_tail_offsets=None,
):
    """Compare truth taus with a rank-selected high-background event tail."""
    background_event_scores = kth_event_score(
        object_logits, object_mask, objective_config.trigger_objects
    )[background_event_mask]
    background_event_count = int(background_event_scores.numel())
    background_event_scores = background_event_scores[
        torch.isfinite(background_event_scores)
    ]
    if not background_event_scores.numel():
        raise ValueError("Tail ranking requires background events in every batch")

    # Rank-defined median centering keeps a hard-negative memory bank invariant
    # to a common additive shift of all current logits.
    center = torch.median(background_event_scores)
    requested = int(math.ceil(
        objective_config.tail_fraction * background_event_count
    ))
    tail_count = min(
        background_event_scores.numel(),
        max(objective_config.tail_min_events, requested),
    )
    current_tail_offsets = torch.topk(
        background_event_scores, k=tail_count
    ).values - center
    tail_offsets = current_tail_offsets
    if memory_tail_offsets is not None and memory_tail_offsets.numel():
        tail_offsets = torch.cat(
            (tail_offsets, memory_tail_offsets.to(tail_offsets.device)), dim=0
        )

    losses = []
    valid = []
    for low, high in objective_config.objective_regions_gev:
        in_region = (
            signal_object_mask
            & object_mask
            & (truth_pt_gev >= low)
            & (truth_pt_gev < high)
        )
        signal_offsets = object_logits[in_region] - center
        if signal_offsets.numel():
            pairwise = (
                tail_offsets.unsqueeze(0) - signal_offsets.unsqueeze(1)
            ) / objective_config.tail_temperature
            losses.append(F.softplus(pairwise).mean())
            valid.append(True)
        else:
            losses.append(object_logits.new_zeros(()))
            valid.append(False)
    losses = torch.stack(losses)
    valid = torch.tensor(valid, dtype=torch.bool, device=losses.device)
    weights = losses.new_tensor(objective_config.objective_region_weights)
    active = weights * valid.to(weights.dtype)
    active = active / active.sum().clamp(min=1e-12)
    ranking_loss = torch.sum(active * losses)
    return -ranking_loss, ranking_loss, current_tail_offsets.detach(), tail_offsets.numel()


def calculate_soft_constraint_metrics(
    object_pass_probabilities,
    object_mask,
    signal_object_mask,
    background_event_mask,
    truth_pt_gev,
    baseline_object_pass,
    objective_config,
    reference_object_pass_probabilities=None,
    objective_override=None,
    ranking_loss=None,
    tail_event_count=None,
    current_tail_offsets=None,
):
    """Calculate the differentiable objective and proxy violations."""
    event_pass = probability_at_least_k(
        object_pass_probabilities, object_mask, k=objective_config.trigger_objects
    )
    if not torch.any(background_event_mask):
        raise ValueError("Every constrained batch must contain background events")
    event_fpr = event_pass[background_event_mask].mean()

    objective_eff, objective_base, objective_valid = _soft_region_values(
        object_pass_probabilities,
        object_mask,
        signal_object_mask,
        truth_pt_gev,
        baseline_object_pass,
        objective_config.objective_regions_gev,
    )
    objective_deltas = objective_eff - objective_base
    weights = object_pass_probabilities.new_tensor(
        objective_config.objective_region_weights
    )
    active_weights = weights * objective_valid.to(weights.dtype)
    active_weights = active_weights / active_weights.sum().clamp(min=1e-12)
    objective = torch.sum(active_weights * objective_deltas)
    if objective_override is not None:
        objective = objective_override

    efficiencies, baseline_efficiencies, valid_regions = _soft_region_values(
        object_pass_probabilities,
        object_mask,
        signal_object_mask,
        truth_pt_gev,
        baseline_object_pass,
        objective_config.constraint_regions_gev,
    )
    deltas = efficiencies - baseline_efficiencies
    reference_efficiencies = None
    if reference_object_pass_probabilities is not None:
        reference_efficiencies, _, _ = _soft_region_values(
            reference_object_pass_probabilities,
            object_mask,
            signal_object_mask,
            truth_pt_gev,
            baseline_object_pass,
            objective_config.constraint_regions_gev,
        )

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
            required_efficiencies, reference_efficiencies - reference_deficits
        )
    required_efficiencies = required_efficiencies.clamp(max=1.0)
    region_margins = efficiencies - required_efficiencies
    fpr_violation = (
        event_fpr - objective_config.target_event_fpr
    ) * objective_config.fpr_violation_scale
    violations = torch.cat((fpr_violation.reshape(1), -region_margins))
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
        objective_region_efficiencies=objective_eff,
        objective_baseline_efficiencies=objective_base,
        objective_region_deltas=objective_deltas,
        ranking_loss=ranking_loss,
        tail_event_count=tail_event_count,
        current_tail_offsets=current_tail_offsets,
    )


def calibrate_tob_baseline(background, target_fpr, trigger_objects=2):
    """Calibrate the comparison TOB threshold on background events only."""
    frame = background.copy()
    frame["_tob_pt_gev"] = tob_pt_gev(frame)
    scores, event_count = build_event_trigger_scores(
        frame, "_tob_pt_gev", objects=trigger_objects
    )
    return select_fpr_threshold(scores, event_count, target_fpr)


def _truth_pt_gev(signal):
    values = signal["truth_pt"].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size and np.median(finite) > 1000.0:
        values = values / 1000.0
    return values


def _hard_region_values(truth_pt, passed, comparison, regions):
    efficiencies = []
    comparisons = []
    counts = []
    pass_counts = []
    comparison_counts = []
    for low, high in regions:
        in_region = (truth_pt >= low) & (truth_pt < high)
        count = int(np.count_nonzero(in_region))
        counts.append(count)
        if not count:
            efficiencies.append(None)
            comparisons.append(None)
            pass_counts.append(0)
            comparison_counts.append(0)
            continue
        pass_count = int(np.count_nonzero(passed[in_region]))
        comparison_count = int(np.count_nonzero(comparison[in_region]))
        pass_counts.append(pass_count)
        comparison_counts.append(comparison_count)
        efficiencies.append(pass_count / count)
        comparisons.append(comparison_count / count)
    return efficiencies, comparisons, counts, pass_counts, comparison_counts


def one_sided_binomial_upper_bound(pass_count, event_count, confidence_level):
    """Return an exact Clopper-Pearson upper bound for a binomial rate."""
    pass_count = int(pass_count)
    event_count = int(event_count)
    if event_count < 1 or not 0 <= pass_count <= event_count:
        raise ValueError("Binomial counts must satisfy 0 <= pass_count <= event_count")
    if confidence_level is None:
        return float(pass_count / event_count)
    if pass_count == event_count:
        return 1.0
    return float(beta.ppf(confidence_level, pass_count + 1, event_count - pass_count))


def certified_calibration_target(event_count, target_fpr, confidence_level):
    """Largest empirical rate whose one-sided upper bound meets target_fpr."""
    if confidence_level is None:
        return float(target_fpr)
    maximum = int(math.floor(float(target_fpr) * int(event_count) + 1e-12))
    accepted = 0
    for pass_count in range(maximum + 1):
        if one_sided_binomial_upper_bound(
            pass_count, event_count, confidence_level
        ) <= target_fpr + 1e-15:
            accepted = pass_count
        else:
            break
    return float(accepted / event_count)


def calibrate_constraint_classifier(
    background,
    background_scores,
    classifier_config,
    objective_config,
    confidence_level_override=None,
):
    """Calibrate at a confidence-safe empirical rate when requested."""
    event_count = int(background["eventNumber"].nunique())
    confidence_level = (
        objective_config.feasibility_confidence_level
        if confidence_level_override is None
        else float(confidence_level_override)
    )
    empirical_target = certified_calibration_target(
        event_count,
        objective_config.target_event_fpr,
        confidence_level,
    )
    calibration = calibrate_classifier(
        background,
        background_scores,
        classifier_config.with_target_fpr(empirical_target),
    )
    calibration["nominal_target_fpr"] = float(objective_config.target_event_fpr)
    calibration["calibration_empirical_target_fpr"] = empirical_target
    calibration["feasibility_confidence_level"] = confidence_level
    calibration_pass_count = int(round(calibration["achieved_fpr"] * event_count))
    calibration_upper = one_sided_binomial_upper_bound(
        calibration_pass_count,
        event_count,
        confidence_level,
    )
    calibration["calibration_upper_confidence_bound"] = calibration_upper
    calibration["calibration_certified"] = bool(
        calibration_upper <= objective_config.target_event_fpr + 1e-12
    )
    return calibration


def _paired_sufficient_statistics(event_numbers, candidate_pass, comparison_pass, mask):
    """Compress event-clustered paired decisions into mergeable moments."""
    mask = np.asarray(mask, dtype=bool)
    event_numbers = np.asarray(event_numbers)[mask]
    differences = (
        np.asarray(candidate_pass, dtype=np.int8)[mask]
        - np.asarray(comparison_pass, dtype=np.int8)[mask]
    ).astype(np.float64)
    if not len(differences):
        return None
    _, inverse = np.unique(event_numbers, return_inverse=True)
    difference_by_event = np.bincount(inverse, weights=differences)
    count_by_event = np.bincount(inverse).astype(np.float64)
    return {
        "cluster_count": int(len(difference_by_event)),
        "object_count": int(len(differences)),
        "difference_sum": float(difference_by_event.sum()),
        "difference_square_sum": float(np.square(difference_by_event).sum()),
        "difference_count_product_sum": float(
            np.sum(difference_by_event * count_by_event)
        ),
        "count_square_sum": float(np.square(count_by_event).sum()),
    }


def aggregate_paired_sufficient_statistics(statistics):
    """Merge paired sufficient statistics from disjoint measurement folds."""
    populated = [item for item in statistics if item is not None]
    if not populated:
        return None
    keys = (
        "cluster_count",
        "object_count",
        "difference_sum",
        "difference_square_sum",
        "difference_count_product_sum",
        "count_square_sum",
    )
    result = {key: sum(item[key] for item in populated) for key in keys}
    result["cluster_count"] = int(result["cluster_count"])
    result["object_count"] = int(result["object_count"])
    return result


def paired_difference_interval(statistics, confidence_level):
    """Estimate a paired efficiency difference with event-clustered uncertainty."""
    if statistics is None or statistics["object_count"] < 1:
        return None
    count = float(statistics["object_count"])
    clusters = int(statistics["cluster_count"])
    estimate = float(statistics["difference_sum"] / count)
    if clusters <= 1:
        standard_error = None
        lower = estimate if confidence_level is None else None
    else:
        centered_square_sum = (
            statistics["difference_square_sum"]
            - 2.0
            * estimate
            * statistics["difference_count_product_sum"]
            + estimate * estimate * statistics["count_square_sum"]
        )
        variance = (
            clusters
            / (clusters - 1.0)
            * max(float(centered_square_sum), 0.0)
            / (count * count)
        )
        standard_error = float(math.sqrt(variance))
        lower = estimate
        if confidence_level is not None:
            lower -= float(norm.ppf(confidence_level)) * standard_error
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "lower_confidence_bound": None if lower is None else float(lower),
        "cluster_count": clusters,
        "object_count": int(count),
    }


def build_confidence_feasibility(
    objective_config,
    background_pass_count,
    background_event_count,
    baseline_statistics,
    reference_statistics,
    fpr_upper_override=None,
):
    """Evaluate FPR and paired regional guards at the configured confidence."""
    confidence_level = objective_config.feasibility_confidence_level
    fpr_estimate = float(background_pass_count / background_event_count)
    fpr_upper = (
        one_sided_binomial_upper_bound(
            background_pass_count,
            background_event_count,
            confidence_level,
        )
        if fpr_upper_override is None
        else float(fpr_upper_override)
    )
    fpr_certified_margin = float(objective_config.target_event_fpr - fpr_upper)
    fpr_point_margin = float(objective_config.target_event_fpr - fpr_estimate)
    # In point mode the certified bound stays recorded as a diagnostic, while
    # the binding FPR test uses the measured rate. The FPR budget is instead
    # protected by the confidence-safe calibration target.
    if objective_config.fpr_feasibility_mode == "point":
        fpr_margin = fpr_point_margin
    else:
        fpr_margin = fpr_certified_margin
    region_records = []
    certified_margins = []
    for index, region in enumerate(objective_config.constraint_regions_gev):
        baseline_interval = paired_difference_interval(
            baseline_statistics[index], confidence_level
        )
        reference_interval = paired_difference_interval(
            reference_statistics[index], confidence_level
        )
        guards = []
        if baseline_interval is not None:
            required = float(objective_config.minimum_region_advantages[index])
            if objective_config.certified_guards_use_allowed_deficits:
                # Certified non-inferiority keeps the configured tolerance:
                # the lower bound must clear advantage minus allowed deficit,
                # not the raw advantage, so saturated regions stay satisfiable.
                required -= float(objective_config.allowed_deficits[index])
            lower = baseline_interval["lower_confidence_bound"]
            margin = None if lower is None else float(lower - required)
            baseline_interval.update(
                {
                    "required_minimum": required,
                    "certified_margin": margin,
                    "satisfied": bool(margin is not None and margin >= -1e-12),
                }
            )
            guards.append(margin)
        if reference_interval is not None:
            required = -float(
                objective_config.reference_model_allowed_deficits[index]
            )
            lower = reference_interval["lower_confidence_bound"]
            margin = None if lower is None else float(lower - required)
            reference_interval.update(
                {
                    "required_minimum": required,
                    "certified_margin": margin,
                    "satisfied": bool(margin is not None and margin >= -1e-12),
                }
            )
            guards.append(margin)
        populated_guards = [value for value in guards if value is not None]
        region_margin = min(populated_guards) if populated_guards else None
        if region_margin is not None:
            certified_margins.append(region_margin)
        region_records.append(
            {
                "region_gev": list(region),
                "candidate_minus_baseline": baseline_interval,
                "candidate_minus_reference": reference_interval,
                "certified_margin": region_margin,
                "satisfied": bool(region_margin is not None and region_margin >= -1e-12),
            }
        )
    all_margins = [fpr_margin, *certified_margins]
    overall_margin = min(all_margins) if all_margins else None
    return {
        "mode": (
            "point" if confidence_level is None else "one_sided_confidence"
        ),
        "confidence_level": confidence_level,
        "fpr": {
            "estimate": fpr_estimate,
            "upper_confidence_bound": fpr_upper,
            "target": float(objective_config.target_event_fpr),
            "feasibility_mode": objective_config.fpr_feasibility_mode,
            "certified_margin": fpr_certified_margin,
            "point_margin": fpr_point_margin,
            "binding_margin": fpr_margin,
            "satisfied": bool(fpr_margin >= -1e-12),
        },
        "regions": region_records,
        "constraints_satisfied": bool(
            fpr_margin >= -1e-12
            and len(region_records) == len(certified_margins)
            and all(record["satisfied"] for record in region_records)
        ),
        "minimum_certified_margin": (
            None if overall_margin is None else float(overall_margin)
        ),
    }


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
        calibration = calibrate_constraint_classifier(
            background,
            background["nn_score"].to_numpy(dtype=np.float64),
            classifier_config,
            objective_config,
        )
    if baseline_threshold_gev is None:
        baseline_target_fpr = certified_calibration_target(
            int(background["eventNumber"].nunique()),
            objective_config.target_event_fpr,
            objective_config.feasibility_confidence_level,
        )
        baseline_threshold_gev, _ = calibrate_tob_baseline(
            background,
            baseline_target_fpr,
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
            reference_calibration = calibrate_constraint_classifier(
                reference_background,
                reference_background["nn_score"].to_numpy(dtype=np.float64),
                classifier_config,
                objective_config,
            )
        reference_pass = classifier_object_pass_mask(
            reference_signal, reference_calibration
        )
    truth_pt = _truth_pt_gev(signal)

    (
        objective_efficiencies,
        objective_baselines,
        objective_counts,
        objective_pass_counts,
        objective_baseline_counts,
    ) = _hard_region_values(
        truth_pt,
        signal_pass,
        baseline_pass,
        objective_config.objective_regions_gev,
    )
    objective_deltas = [
        None if value is None else value - base
        for value, base in zip(objective_efficiencies, objective_baselines)
    ]
    valid_objective = np.asarray([value is not None for value in objective_deltas])
    delta_values = np.asarray(
        [0.0 if value is None else value for value in objective_deltas],
        dtype=np.float64,
    )
    weights = np.asarray(
        objective_config.objective_region_weights, dtype=np.float64
    ) * valid_objective
    weights = weights / weights.sum() if weights.sum() else weights
    objective = float(np.sum(weights * delta_values))

    efficiencies, baseline_efficiencies, counts, pass_counts, baseline_counts = (
        _hard_region_values(
            truth_pt,
            signal_pass,
            baseline_pass,
            objective_config.constraint_regions_gev,
        )
    )
    reference_efficiencies = [None] * len(counts)
    reference_counts = [0] * len(counts)
    if reference_pass is not None:
        reference_efficiencies, _, _, reference_counts, _ = _hard_region_values(
            truth_pt,
            reference_pass,
            baseline_pass,
            objective_config.constraint_regions_gev,
        )

    required_efficiencies = []
    deltas = []
    margins = []
    for index, (efficiency, baseline_efficiency) in enumerate(
        zip(efficiencies, baseline_efficiencies)
    ):
        if efficiency is None:
            required_efficiencies.append(None)
            deltas.append(None)
            margins.append(0.0)
            continue
        required = baseline_efficiency + objective_config.minimum_region_advantages[index]
        if reference_efficiencies[index] is not None:
            required = max(
                required,
                reference_efficiencies[index]
                - objective_config.reference_model_allowed_deficits[index],
            )
        required = min(required, 1.0)
        required_efficiencies.append(required)
        deltas.append(efficiency - baseline_efficiency)
        margins.append(efficiency - required)

    valid = np.asarray([value is not None for value in deltas])
    margins_array = np.asarray(margins, dtype=np.float64)
    resolutions = [None if count < 1 else 1.0 / count for count in counts]
    margin_in_objects = [
        None if resolution is None else float(margin / resolution)
        for margin, resolution in zip(margins_array, resolutions)
    ]
    event_pass = classifier_event_pass_mask(background, calibration)
    background_event_count = int(background["eventNumber"].nunique())
    background_event_pass_count = int(np.count_nonzero(event_pass))
    achieved_fpr = float(background_event_pass_count / background_event_count)
    event_numbers = signal["eventNumber"].to_numpy()
    baseline_statistics = []
    reference_statistics = []
    for low, high in objective_config.constraint_regions_gev:
        in_region = (truth_pt >= low) & (truth_pt < high)
        baseline_statistics.append(
            _paired_sufficient_statistics(
                event_numbers,
                signal_pass,
                baseline_pass,
                in_region,
            )
        )
        reference_statistics.append(
            None
            if reference_pass is None
            else _paired_sufficient_statistics(
                event_numbers,
                signal_pass,
                reference_pass,
                in_region,
            )
        )
    feasibility = build_confidence_feasibility(
        objective_config,
        background_event_pass_count,
        background_event_count,
        baseline_statistics,
        reference_statistics,
    )
    return {
        "objective_value": objective,
        "achieved_fpr": achieved_fpr,
        "objective_region_efficiencies": objective_efficiencies,
        "objective_baseline_efficiencies": objective_baselines,
        "objective_region_deltas": objective_deltas,
        "objective_region_counts": objective_counts,
        "objective_region_pass_counts": objective_pass_counts,
        "objective_baseline_pass_counts": objective_baseline_counts,
        "region_efficiencies": efficiencies,
        "baseline_efficiencies": baseline_efficiencies,
        "reference_efficiencies": reference_efficiencies,
        "required_efficiencies": required_efficiencies,
        "region_deltas": deltas,
        "region_counts": counts,
        "region_pass_counts": pass_counts,
        "baseline_pass_counts": baseline_counts,
        "reference_pass_counts": reference_counts,
        "region_efficiency_resolutions": resolutions,
        "constraint_margins": margins_array.tolist(),
        "constraint_margins_in_objects": margin_in_objects,
        "constraints_satisfied": feasibility["constraints_satisfied"],
        "minimum_margin": (
            float(np.min(margins_array[valid])) if np.any(valid) else None
        ),
        "minimum_certified_margin": feasibility["minimum_certified_margin"],
        "background_event_count": background_event_count,
        "background_event_pass_count": background_event_pass_count,
        "classifier_calibration": calibration,
        "baseline_threshold_gev": float(baseline_threshold_gev),
        "paired_region_sufficient_statistics": {
            "candidate_minus_baseline": baseline_statistics,
            "candidate_minus_reference": reference_statistics,
        },
        "feasibility": feasibility,
    }
