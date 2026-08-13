"""Configurable training losses for TauNet experiments."""

from dataclasses import dataclass

import torch.nn as nn


VALID_LOSSES = ("bce",)


@dataclass(frozen=True)
class LossConfig:
    name: str


def parse_loss(config):
    """Validate loss settings while preserving the legacy BCE default."""
    raw = config.get("loss", {})
    name = raw.get("name", "bce")
    if name not in VALID_LOSSES:
        raise ValueError(f"Unknown loss: {name}")
    return LossConfig(name=name)


def build_loss(loss_config):
    """Build the configured PyTorch loss."""
    if loss_config.name == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unknown loss: {loss_config.name}")
