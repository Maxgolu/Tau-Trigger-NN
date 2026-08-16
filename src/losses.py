"""Configurable training losses for TauNet experiments."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_LOSSES = ("bce", "energy_weighted_bce")
VALID_WEIGHT_PROFILES = ("alpha", "inverse_frequency")


@dataclass(frozen=True)
class EnergyWeightingConfig:
    profile: str = "alpha"
    alpha: float = 0.0
    pt_min_gev: float = 25.0
    pt_max_gev: float = 100.0
    bin_width_gev: float = 5.0
    min_weight: float = 0.2
    max_weight: float = 5.0

    def to_dict(self):
        result = {
            "profile": self.profile,
            "normalization": "training_signal_mean",
        }
        if self.profile == "alpha":
            result["alpha"] = self.alpha
            result["bands_gev"] = [25.0, 40.0, 80.0]
        else:
            result.update(
                {
                    "pt_min_gev": self.pt_min_gev,
                    "pt_max_gev": self.pt_max_gev,
                    "bin_width_gev": self.bin_width_gev,
                    "min_weight": self.min_weight,
                    "max_weight": self.max_weight,
                }
            )
        return result


@dataclass(frozen=True)
class LossConfig:
    name: str
    weighting: EnergyWeightingConfig | None = None

    def to_dict(self):
        result = {"name": self.name}
        if self.weighting is not None:
            result["weighting"] = self.weighting.to_dict()
        return result


@dataclass(frozen=True)
class FittedEnergyWeighting:
    """Signal-energy multipliers fitted only from the training partition."""

    profile: str
    edges_gev: tuple
    raw_multipliers: tuple
    multipliers: tuple
    signal_counts: tuple
    normalization_factor: float
    training_signal_count: int

    def to_dict(self):
        def finite_or_none(value):
            return float(value) if np.isfinite(value) else None

        return {
            "profile": self.profile,
            "normalization": "training_signal_mean",
            "edges_gev": [finite_or_none(value) for value in self.edges_gev],
            "raw_multipliers": [float(value) for value in self.raw_multipliers],
            "multipliers": [float(value) for value in self.multipliers],
            "signal_counts": [int(value) for value in self.signal_counts],
            "normalization_factor": float(self.normalization_factor),
            "training_signal_count": int(self.training_signal_count),
        }


class EnergyWeightedBCELoss(nn.Module):
    """Apply per-object weights while keeping ordinary BCE as the base loss."""

    def forward(self, predictions, targets, sample_weights):
        per_object = F.binary_cross_entropy(
            predictions,
            targets,
            reduction="none",
        )
        return (per_object * sample_weights).mean()


def parse_loss(config):
    """Validate loss settings while preserving the legacy BCE default."""
    raw = config.get("loss", {})
    name = raw.get("name", "bce")
    if name not in VALID_LOSSES:
        raise ValueError(f"Unknown loss: {name}")
    if name == "bce":
        return LossConfig(name=name)

    weighting_raw = raw.get("weighting", {})
    profile = weighting_raw.get("profile", "alpha")
    if profile not in VALID_WEIGHT_PROFILES:
        raise ValueError(f"Unknown energy-weight profile: {profile}")

    alpha = float(weighting_raw.get("alpha", 0.0))
    pt_min = float(weighting_raw.get("pt_min_gev", 25.0))
    pt_max = float(weighting_raw.get("pt_max_gev", 100.0))
    bin_width = float(weighting_raw.get("bin_width_gev", 5.0))
    min_weight = float(weighting_raw.get("min_weight", 0.2))
    max_weight = float(weighting_raw.get("max_weight", 5.0))

    if alpha < 0:
        raise ValueError("loss.weighting.alpha must be non-negative")
    if not 0 <= pt_min < pt_max:
        raise ValueError("inverse-frequency pT limits must satisfy 0 <= min < max")
    if bin_width <= 0:
        raise ValueError("loss.weighting.bin_width_gev must be positive")
    if not 0 < min_weight <= max_weight:
        raise ValueError("inverse-frequency weight limits must satisfy 0 < min <= max")

    weighting = EnergyWeightingConfig(
        profile=profile,
        alpha=alpha,
        pt_min_gev=pt_min,
        pt_max_gev=pt_max,
        bin_width_gev=bin_width,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    return LossConfig(name=name, weighting=weighting)


def build_loss(loss_config):
    """Build the configured PyTorch loss."""
    if loss_config.name == "bce":
        return nn.BCELoss()
    if loss_config.name == "energy_weighted_bce":
        return EnergyWeightedBCELoss()
    raise ValueError(f"Unknown loss: {loss_config.name}")


def _truth_pt_gev(metadata, labels):
    if "truth_pt" not in metadata.columns:
        raise ValueError("energy-weighted BCE requires a truth_pt metadata column")
    values = metadata["truth_pt"].to_numpy(dtype=np.float64, copy=True)
    signal_values = values[(labels == 1) & np.isfinite(values) & (values > 0)]
    if signal_values.size == 0:
        raise ValueError("energy-weighted BCE requires positive signal truth_pt values")

    # The stored samples normally use MeV, while configs and reports use GeV.
    if np.median(signal_values) > 1000.0:
        values /= 1000.0
    return values


def _alpha_regions(alpha):
    edges = np.asarray([-np.inf, 25.0, 40.0, 80.0, np.inf])
    multipliers = np.asarray(
        [1.0, 1.0 + alpha, 1.0 + alpha / 2.0, 1.0],
        dtype=np.float64,
    )
    return edges, multipliers


def _inverse_frequency_regions(weighting, truth_pt_gev, signal_mask):
    protected_edges = np.arange(
        weighting.pt_min_gev,
        weighting.pt_max_gev + weighting.bin_width_gev * 0.5,
        weighting.bin_width_gev,
        dtype=np.float64,
    )
    if protected_edges[-1] < weighting.pt_max_gev:
        protected_edges = np.append(protected_edges, weighting.pt_max_gev)
    protected_edges[-1] = weighting.pt_max_gev
    edges = np.concatenate(([-np.inf], protected_edges, [np.inf]))
    region_ids = np.digitize(truth_pt_gev, edges[1:-1], right=False)
    counts = np.bincount(region_ids[signal_mask], minlength=len(edges) - 1)

    raw = np.ones(len(edges) - 1, dtype=np.float64)
    protected_counts = counts[1:-1]
    populated = protected_counts[protected_counts > 0]
    if populated.size == 0:
        raise ValueError("No training signal falls inside the inverse-frequency range")
    reference_count = float(populated.mean())
    raw[1:-1] = np.divide(
        reference_count,
        protected_counts,
        out=np.full(protected_counts.shape, weighting.max_weight, dtype=np.float64),
        where=protected_counts > 0,
    )
    raw = np.clip(raw, weighting.min_weight, weighting.max_weight)
    return edges, raw


def fit_loss_weighting(loss_config, training_metadata, training_labels):
    """Fit energy weights on training data; return None for ordinary BCE."""
    if loss_config.name == "bce":
        return None

    labels = np.asarray(training_labels).reshape(-1)
    signal_mask = labels == 1
    truth_pt_gev = _truth_pt_gev(training_metadata, labels)
    weighting = loss_config.weighting
    if weighting.profile == "alpha":
        edges, raw = _alpha_regions(weighting.alpha)
    else:
        edges, raw = _inverse_frequency_regions(
            weighting,
            truth_pt_gev,
            signal_mask,
        )

    region_ids = np.digitize(truth_pt_gev, edges[1:-1], right=False)
    counts = np.bincount(region_ids[signal_mask], minlength=len(raw))
    signal_raw_weights = raw[region_ids[signal_mask]]
    normalization = float(signal_raw_weights.mean())
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Could not normalize energy weights")

    return FittedEnergyWeighting(
        profile=weighting.profile,
        edges_gev=tuple(edges.tolist()),
        raw_multipliers=tuple(raw.tolist()),
        multipliers=tuple((raw / normalization).tolist()),
        signal_counts=tuple(counts.tolist()),
        normalization_factor=normalization,
        training_signal_count=int(signal_mask.sum()),
    )


def calculate_sample_weights(fitted_weighting, metadata, labels):
    """Apply training-fitted signal weights without changing background weights."""
    labels = np.asarray(labels).reshape(-1)
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if fitted_weighting is None:
        return weights

    signal_mask = labels == 1
    truth_pt_gev = _truth_pt_gev(metadata, labels)
    edges = np.asarray(fitted_weighting.edges_gev, dtype=np.float64)
    multipliers = np.asarray(fitted_weighting.multipliers, dtype=np.float64)
    region_ids = np.digitize(truth_pt_gev, edges[1:-1], right=False)
    weights[signal_mask] = multipliers[region_ids[signal_mask]].astype(np.float32)
    return weights
