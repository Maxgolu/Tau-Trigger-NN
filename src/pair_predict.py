"""Generate pair and event predictions from a saved pair-model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .pair_evaluate import assign_validation_folds
    from .pair_models import build_pair_model
    from .pair_train import transform_zscore
except ImportError:
    from pair_evaluate import assign_validation_folds
    from pair_models import build_pair_model
    from pair_train import transform_zscore


def _normalized_inputs(source, normalizer, *, member_width=None):
    if "inputs" in source:
        inputs = np.asarray(source["inputs"])
        if member_width is not None and inputs.ndim == 2:
            expected = 2 * int(member_width)
            if inputs.shape[1] != expected:
                raise ValueError(
                    f"shared-member flat input must contain {expected} values"
                )
            inputs = inputs.reshape(-1, 2, int(member_width))
        return (
            transform_zscore(
                inputs,
                normalizer["inputs_mean"],
                normalizer["inputs_scale"],
            ),
        )
    return tuple(
        transform_zscore(
            source[name],
            normalizer[f"{name}_mean"],
            normalizer[f"{name}_scale"],
        )
        for name in ("images", "scalars", "context")
    )


def predict_outputs(model, arrays, batch_size=4096):
    """Return raw model outputs in deterministic pair order."""
    loader = DataLoader(
        TensorDataset(*(torch.from_numpy(values) for values in arrays)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            logits = model(*batch) if len(batch) > 1 else model(batch[0])
            scores.append(logits.cpu().numpy())
    result = np.concatenate(scores).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("model produced a non-finite output")
    return result


def pair_scores(outputs):
    """Convert binary logits or three-class logits into one pair score."""
    values = np.asarray(outputs, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError("model outputs must be one- or two-dimensional")
    if values.shape[1] == 1:
        return values[:, 0]
    if values.shape[1] == 3:
        shifted = values - values.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 2]
    raise ValueError("model must produce one binary logit or three class logits")


def predict_pairs(model, arrays, batch_size=4096):
    """Return the event-aggregation score for every pair."""
    return pair_scores(predict_outputs(model, arrays, batch_size=batch_size))


def maximum_event_scores(pair_event_ids, pair_scores, event_ids):
    """Aggregate finite pair scores and retain no-pair events as -infinity."""
    pair_event_ids = np.asarray(pair_event_ids, dtype=np.int64)
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if len(pair_event_ids) != len(pair_scores):
        raise ValueError("pair event IDs and scores must be aligned")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("event IDs must be unique")
    if not np.isfinite(pair_scores).all():
        raise ValueError("pair scores must be finite")
    positions = {int(event_id): index for index, event_id in enumerate(event_ids)}
    result = np.full(len(event_ids), -np.inf, dtype=np.float64)
    for event_id, score in zip(pair_event_ids, pair_scores, strict=True):
        if int(event_id) not in positions:
            raise ValueError("pair belongs to an unknown event")
        position = positions[int(event_id)]
        result[position] = max(result[position], float(score))
    return result


def validation_bce(logits, labels, pair_event_ids, event_ids, signal, eligible):
    """Calculate BCE on the predefined pair-level validation population."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pair_event_ids = np.asarray(pair_event_ids, dtype=np.int64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    signal = np.asarray(signal, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if logits.shape != labels.shape or logits.shape != pair_event_ids.shape:
        raise ValueError("validation logits, labels, and pair identities must align")
    if event_ids.shape != signal.shape or event_ids.shape != eligible.shape:
        raise ValueError("validation event metadata must align")
    event_population = {
        int(event_id): (not is_signal) or is_eligible
        for event_id, is_signal, is_eligible in zip(
            event_ids, signal, eligible, strict=True
        )
    }
    unknown_events = set(map(int, pair_event_ids)).difference(event_population)
    if unknown_events:
        raise ValueError("validation pairs contain an unknown event")
    mask = np.asarray(
        [event_population.get(int(event_id), False) for event_id in pair_event_ids],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("validation BCE population is empty")
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    losses = np.logaddexp(0.0, selected_logits) - selected_labels * selected_logits
    return float(losses.sum(dtype=np.float64) / len(losses))


def validation_objective(
    outputs,
    labels,
    pair_event_ids,
    event_ids,
    signal,
    eligible,
    *,
    loss_name,
):
    """Calculate loss on the predefined pair-level validation population."""
    outputs = np.asarray(outputs, dtype=np.float64)
    labels = np.asarray(labels)
    pair_event_ids = np.asarray(pair_event_ids, dtype=np.int64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    signal = np.asarray(signal, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    event_population = {
        int(event_id): (not is_signal) or is_eligible
        for event_id, is_signal, is_eligible in zip(
            event_ids, signal, eligible, strict=True
        )
    }
    if any(int(event_id) not in event_population for event_id in pair_event_ids):
        raise ValueError("validation pair belongs to an unknown event")
    mask = np.asarray(
        [event_population[int(event_id)] for event_id in pair_event_ids],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("validation loss population is empty")
    if loss_name == "bce":
        if outputs.ndim == 2 and outputs.shape[1] != 1:
            raise ValueError("BCE validation requires one binary logit")
        return validation_bce(
            pair_scores(outputs),
            labels,
            pair_event_ids,
            event_ids,
            signal,
            eligible,
        )
    if loss_name != "cross_entropy":
        raise ValueError("loss name must be bce or cross_entropy")
    if outputs.ndim != 2 or outputs.shape[1] != 3:
        raise ValueError("cross-entropy validation requires three class logits")
    if labels.shape != (len(outputs),) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("cross-entropy labels must contain only 0, 1, or 2")
    selected = outputs[mask]
    selected_labels = labels[mask].astype(np.int64)
    maxima = selected.max(axis=1)
    log_denominator = maxima + np.log(
        np.exp(selected - maxima[:, None]).sum(axis=1)
    )
    losses = log_denominator - selected[np.arange(len(selected)), selected_labels]
    return float(losses.sum(dtype=np.float64) / len(losses))


def run_prediction(config_path, pair_data_path, normalizer_path, checkpoint_path):
    """Load saved artifacts and return pair and complete-event tables."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_config = config.get("model", config.get("neural_model"))
    model = build_pair_model(model_config)
    member_width = (
        model_config.get("member_width")
        if model_config.get("name") == "shared_member_pair_mlp"
        else None
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    with np.load(normalizer_path, allow_pickle=False) as normalizer_source:
        normalizer = {name: normalizer_source[name] for name in normalizer_source.files}
    with np.load(pair_data_path, allow_pickle=False) as source:
        arrays = _normalized_inputs(
            source,
            normalizer,
            member_width=member_width,
        )
        outputs = predict_outputs(model, arrays)
        scores = pair_scores(outputs)
        pair_event_ids = np.asarray(source["pair_event_ids"], dtype=np.int64)
        pair_indices = np.asarray(source["pair_indices"], dtype=np.int64)
        pair_tob_indices = np.asarray(source["pair_tob_indices"], dtype=np.int64)
        pair_selected_ranks = np.asarray(
            source["pair_selected_ranks"],
            dtype=np.int64,
        )
        loss_name = config.get("loss", {}).get("name", "bce")
        label_key = "labels" if loss_name == "bce" else "three_class_labels"
        labels = np.asarray(source[label_key])
        event_ids = np.asarray(source["event_ids"], dtype=np.int64)
        event_scores = maximum_event_scores(pair_event_ids, scores, event_ids)
        signal = np.asarray(source["event_samples"], dtype=bool)
        observable = np.asarray(source["event_pair_observable"], dtype=bool)
        eligible = np.asarray(source["event_pair_eligible"], dtype=bool)
        baseline = np.asarray(source["baseline_event_scores"], dtype=np.float64)
    objective = validation_objective(
        outputs,
        labels,
        pair_event_ids,
        event_ids,
        signal,
        eligible,
        loss_name=loss_name,
    )

    pairs = pd.DataFrame(
        {
            "global_event_id": pair_event_ids,
            "pair_index": pair_indices,
            "tob_index_a": pair_tob_indices[:, 0],
            "tob_index_b": pair_tob_indices[:, 1],
            "selected_rank_a": pair_selected_ranks[:, 0],
            "selected_rank_b": pair_selected_ranks[:, 1],
            "pair_score": scores,
            "target_label": labels.astype(np.uint8),
        }
    )
    if loss_name == "bce":
        pairs["binary_label"] = labels.astype(np.uint8)
    else:
        pairs["three_class_label"] = labels.astype(np.uint8)
    events = pd.DataFrame(
        {
            "global_event_id": event_ids,
            "sample": np.where(signal, "signal", "background"),
            "pair_observable": observable,
            "pair_eligible": eligible,
            "event_score": event_scores,
            "baseline_event_score": baseline,
            "fold": assign_validation_folds(event_ids, signal, observable, eligible),
            "validation_bce": objective,
            "validation_loss_name": loss_name,
        }
    )
    return pairs, events


def parse_args():
    parser = argparse.ArgumentParser(description="Score prepared pair-model data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pair_data", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--checkpoint_id", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs, events = run_prediction(
        args.config,
        args.pair_data,
        args.normalizer,
        args.checkpoint,
    )
    pairs["model_seed"] = args.model_seed
    pairs["checkpoint_id"] = args.checkpoint_id
    events["model_seed"] = args.model_seed
    events["checkpoint_id"] = args.checkpoint_id
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "pair_predictions.csv", index=False)
    events.to_csv(output / "event_predictions.csv", index=False)


if __name__ == "__main__":
    main()
