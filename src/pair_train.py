"""Train configured pair models on a prepared training-only pair dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .pair_models import build_pair_model
except ImportError:
    from pair_models import build_pair_model


def set_random_seed(seed):
    """Configure reproducible CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_zscore(values, *, reject_zero_scale=False):
    """Fit a position-wise z-score with population standard deviation."""
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0, dtype=np.float64)
    scale = array.std(axis=0, dtype=np.float64, ddof=0)
    if reject_zero_scale and (scale <= 1e-12).any():
        raise ValueError("inputs contain a zero-scale training feature")
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def transform_zscore(values, mean, scale):
    """Apply a frozen z-score and return float32 model inputs."""
    transformed = (
        np.asarray(values, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    ) / np.asarray(scale, dtype=np.float64)
    return transformed.astype(np.float32)


def prepare_training_arrays(
    path,
    *,
    target="binary",
    member_width=None,
    high_resolution_widths=None,
):
    """Load pair inputs and fit normalization using this training file only."""
    with np.load(path, allow_pickle=False) as source:
        if target == "binary":
            if "labels" not in source:
                raise KeyError("prepared pair data must contain labels")
            labels = np.asarray(source["labels"], dtype=np.float32)
            if labels.ndim != 1 or not np.isin(labels, [0.0, 1.0]).all():
                raise ValueError("labels must be a one-dimensional binary array")
        elif target == "three_class":
            if "three_class_labels" not in source:
                raise KeyError("prepared pair data must contain three_class_labels")
            labels = np.asarray(source["three_class_labels"], dtype=np.int64)
            if labels.ndim != 1 or not np.isin(labels, [0, 1, 2]).all():
                raise ValueError("three_class_labels must contain only 0, 1, or 2")
        else:
            raise ValueError("target must be binary or three_class")

        if "inputs" in source:
            raw = np.asarray(source["inputs"], dtype=np.float64)
            if member_width is not None and raw.ndim == 2:
                expected = 2 * int(member_width)
                if raw.shape[1] != expected:
                    raise ValueError(
                        f"shared-member flat input must contain {expected} values"
                    )
                raw = raw.reshape(-1, 2, int(member_width))
            if raw.ndim == 3 and raw.shape[1] == 2:
                mean, scale = fit_zscore(
                    raw.reshape(-1, raw.shape[2]),
                    reject_zero_scale=True,
                )
            else:
                mean, scale = fit_zscore(raw)
            arrays = (transform_zscore(raw, mean, scale),)
            normalizer = {"inputs_mean": mean, "inputs_scale": scale}
        else:
            required = {"images", "scalars", "context"}
            missing = required.difference(source.files)
            if missing:
                raise KeyError(f"prepared high-resolution data is missing {sorted(missing)}")
            images = np.asarray(source["images"], dtype=np.float64)
            scalars = np.asarray(source["scalars"], dtype=np.float64)
            context = np.asarray(source["context"], dtype=np.float64)
            if images.ndim != 4 or images.shape[1:] != (2, 12, 12):
                raise ValueError("images must have shape (pairs, 2, 12, 12)")
            scalar_width, context_width = high_resolution_widths or (47, 1)
            if scalars.ndim != 3 or scalars.shape[1:] != (2, scalar_width):
                raise ValueError(
                    f"scalars must have shape (pairs, 2, {scalar_width})"
                )
            if context.ndim != 2 or context.shape[1] != context_width:
                raise ValueError(
                    f"context must have shape (pairs, {context_width})"
                )
            image_mean, image_scale = fit_zscore(
                images.reshape(-1, 12, 12),
                reject_zero_scale=True,
            )
            scalar_mean, scalar_scale = fit_zscore(
                scalars.reshape(-1, scalar_width),
                reject_zero_scale=True,
            )
            context_mean, context_scale = fit_zscore(
                context,
                reject_zero_scale=True,
            )
            arrays = (
                transform_zscore(images, image_mean, image_scale),
                transform_zscore(scalars, scalar_mean, scalar_scale),
                transform_zscore(context, context_mean, context_scale),
            )
            normalizer = {
                "images_mean": image_mean,
                "images_scale": image_scale,
                "scalars_mean": scalar_mean,
                "scalars_scale": scalar_scale,
                "context_mean": context_mean,
                "context_scale": context_scale,
            }

    if any(len(values) != len(labels) for values in arrays):
        raise ValueError("all model inputs must align with labels")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("model inputs must be finite")
    return arrays, labels, normalizer


def _positive_mask(labels, target):
    return labels.astype(np.int64) == (1 if target == "binary" else 2)


def _member_min_truth(source, labels, target):
    key = "member_min_truth_pt_gev"
    if key not in source:
        raise KeyError(f"weighted training requires {key}")
    values = np.asarray(source[key], dtype=np.float64)
    if values.shape != labels.shape:
        raise ValueError(f"{key} must contain one value per pair")
    positive = _positive_mask(labels, target)
    if not np.isfinite(values[positive]).all():
        raise ValueError(f"{key} must be finite for every positive pair")
    return values, positive


def training_weights(path, labels, *, target, weighting="equal"):
    """Build the exact supported equal or matched-member energy weights."""
    if weighting in {None, "equal", "unweighted"}:
        return np.ones(len(labels), dtype=np.float32)
    with np.load(path, allow_pickle=False) as source:
        energy, positive = _member_min_truth(source, labels, target)
    positive_energy = energy[positive]
    positive_weights = np.ones(len(positive_energy), dtype=np.float64)
    if weighting == "inverse_frequency_member_min":
        edges = np.arange(25.0, 105.0, 5.0)
        inside = (positive_energy >= 25.0) & (positive_energy < 100.0)
        indices = np.searchsorted(edges, positive_energy[inside], side="right") - 1
        counts = np.bincount(indices, minlength=15)
        occupied = counts[counts > 0]
        if not len(occupied):
            raise ValueError("inverse-frequency weighting has no occupied bins")
        mean_count = occupied.mean(dtype=np.float64)
        positive_weights[inside] = np.clip(
            mean_count / counts[indices],
            0.2,
            5.0,
        )
    elif weighting == "power_law_pminus1_member_min":
        positive_weights = np.clip(positive_energy, 10.0, 200.0)
    else:
        raise ValueError(f"unsupported pair weighting: {weighting}")
    positive_weights /= positive_weights.mean(dtype=np.float64)
    weights = np.ones(len(labels), dtype=np.float64)
    weights[positive] = positive_weights
    return weights.astype(np.float32)


def _logits(model, batch, input_count):
    inputs = batch[:input_count]
    return model(*inputs) if len(inputs) > 1 else model(inputs[0])


def _target_and_loss(config):
    loss_config = config.get("loss", {})
    name = loss_config.get("name", "bce")
    if name == "bce":
        target = "binary"
        declared_target = "both_members_tau_labelled"
    elif name == "cross_entropy":
        target = "three_class"
        declared_target = "zero_one_or_two_tau_labelled_members"
    else:
        raise ValueError("loss name must be bce or cross_entropy")
    if "target" in config and config["target"] != declared_target:
        raise ValueError(
            f"target must be {declared_target!r} when loss is {name!r}"
        )
    return target, name


def train_seed(config, pair_data_path, output_dir, seed):
    """Train one seed and save one checkpoint per configured epoch."""
    set_random_seed(seed)
    model_config = config.get("model", config.get("neural_model"))
    if model_config is None:
        raise KeyError("config requires model or neural_model")
    member_width = (
        model_config.get("member_width")
        if model_config.get("name") == "shared_member_pair_mlp"
        else None
    )
    high_resolution_widths = (
        (
            int(model_config.get("member_scalar_width", 47)),
            int(model_config.get("context_width", 1)),
        )
        if model_config.get("name") == "shared_member_high_resolution_em2"
        else None
    )
    target, loss_name = _target_and_loss(config)
    arrays, labels, normalizer = prepare_training_arrays(
        pair_data_path,
        target=target,
        member_width=member_width,
        high_resolution_widths=high_resolution_widths,
    )
    weighting = config.get("loss", {}).get("pair_weighting", "equal")
    weights = training_weights(
        pair_data_path,
        labels,
        target=target,
        weighting=weighting,
    )
    model = build_pair_model(model_config)

    tensors = [torch.from_numpy(values) for values in arrays]
    tensors.append(torch.from_numpy(labels))
    tensors.append(torch.from_numpy(weights))
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(424200 + int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer_config = config.get("optimizer", {})
    if optimizer_config.get("name", "adam").lower() != "adam":
        raise ValueError("pair training currently supports only the Adam optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 0.001)),
    )
    run_dir = Path(output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez(run_dir / "normalizer.npz", **normalizer)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    records = []
    epochs = int(config.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        unweighted_loss = 0.0
        total_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, batch, len(arrays))
            targets = batch[-2]
            batch_weights = batch[-1]
            if loss_name == "bce":
                logits = logits.reshape(-1)
                targets = targets.reshape(-1)
                per_pair = F.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
            else:
                if logits.ndim != 2 or logits.shape[1] != 3:
                    raise ValueError("three-class training requires three output logits")
                per_pair = F.cross_entropy(logits, targets.reshape(-1), reduction="none")
            loss = (per_pair * batch_weights.reshape(-1)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            total_loss += float((per_pair * batch_weights).detach().sum())
            unweighted_loss += float(per_pair.detach().sum())
            total_count += len(targets)

        mean_loss = total_loss / total_count
        records.append(
            {
                "epoch": epoch,
                "loss_name": loss_name,
                "pair_weighting": weighting,
                "training_objective": mean_loss,
                "training_unweighted_loss": unweighted_loss / total_count,
                "training_bce": mean_loss if loss_name == "bce" else "",
                "training_cross_entropy": (
                    mean_loss if loss_name == "cross_entropy" else ""
                ),
            }
        )
        if config.get("checkpoint_each_epoch", True):
            torch.save(model.state_dict(), run_dir / f"checkpoint_epoch_{epoch:02d}.pt")

    with (run_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a configured pair model")
    parser.add_argument("--config", required=True, help="Pair experiment JSON")
    parser.add_argument("--pair_data", required=True, help="Training-only pair NPZ")
    parser.add_argument("--experiments_dir", default="experiments/pair_models")
    parser.add_argument("--seed", type=int, default=None, help="Run one configured seed")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        seeds = [args.seed]
    elif "seed" in config:
        seeds = [config["seed"]]
    else:
        seeds = config.get("seeds", [42])
    experiment_dir = Path(args.experiments_dir) / config["experiment_name"]
    for seed in seeds:
        train_seed(config, args.pair_data, experiment_dir, int(seed))


if __name__ == "__main__":
    main()
