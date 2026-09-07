"""Prepare pair-model arrays from the aligned CSV and NPZ source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .pair_context_features import (
        build_pair_context_features,
        select_context_features,
    )
    from .pair_data import build_pair_dataset
    from .pair_features import (
        compact_em2_member_features,
        em2_best_3x3_fraction,
        member_summary_features,
        raw_cell_pair_features,
        raw_cell_pt_pair_features,
        summary_pair_features,
        summary_pt_pair_features,
    )
    from .training_data import TrainingDataCache
except ImportError:
    from pair_context_features import (
        build_pair_context_features,
        select_context_features,
    )
    from pair_data import build_pair_dataset
    from pair_features import (
        compact_em2_member_features,
        em2_best_3x3_fraction,
        member_summary_features,
        raw_cell_pair_features,
        raw_cell_pt_pair_features,
        summary_pair_features,
        summary_pt_pair_features,
    )
    from training_data import TrainingDataCache


CORE_COLUMNS = [f"tensor_{index}" for index in range(45)]
EM2_COLUMNS = [f"em2_cell_{index}" for index in range(144)]
PREPARED_MEMBER_FEATURES = {
    "em2_raw_dominance",
    "em2_normalized_dominance",
    "em2_best_3x3_fraction",
    "measured_tob_pt",
    "event_second_highest_tob_pt",
}


def _source_event_inventory(data_dir, split_seed, max_events_per_class=None):
    data_dir = Path(data_dir)
    with np.load(data_dir / "Signal" / "signal_combined.npz") as source:
        signal_events = np.asarray(source["event_nums"], dtype=np.int64)
    with np.load(data_dir / "Background" / "bkg_combined.npz") as source:
        background_events = np.asarray(source["event_nums"], dtype=np.int64)

    if max_events_per_class is not None:
        rng = np.random.default_rng(split_seed)
        if max_events_per_class < len(signal_events):
            positions = np.sort(
                rng.choice(len(signal_events), size=max_events_per_class, replace=False)
            )
            signal_events = signal_events[positions]
        if max_events_per_class < len(background_events):
            positions = np.sort(
                rng.choice(len(background_events), size=max_events_per_class, replace=False)
            )
            background_events = background_events[positions]

    offset = 2 * int(signal_events.max())
    background_events = background_events + offset
    event_ids = np.concatenate([signal_events, background_events])
    samples = np.concatenate(
        [
            np.repeat("signal", len(signal_events)),
            np.repeat("background", len(background_events)),
        ]
    )
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("signal and background event identifiers must be collision-safe")

    order = event_ids.copy()
    rng = np.random.RandomState(split_seed)
    rng.shuffle(order)
    train_end = int(0.70 * len(order))
    validation_end = int(0.80 * len(order))
    split_by_event = {
        int(event_id): split
        for split, values in (
            ("train", order[:train_end]),
            ("validation", order[train_end:validation_end]),
            ("test", order[validation_end:]),
        )
        for event_id in values
    }
    return pd.DataFrame(
        {
            "eventNumber": event_ids,
            "sample": samples,
            "split": [split_by_event[int(event_id)] for event_id in event_ids],
        }
    )


def _member_frame(pairs, frame, suffix, *, extra_columns=()):
    index_column = f"tob_index_{suffix}"
    keys = pairs.loc[:, ["eventNumber", index_column]].rename(
        columns={index_column: "tob_index"}
    )
    source_columns = (
        ["eventNumber", "tob_index", "tob_pt"]
        + list(extra_columns)
        + CORE_COLUMNS
        + EM2_COLUMNS
    )
    missing = set(source_columns).difference(frame.columns)
    if missing:
        raise KeyError(f"source data is missing {sorted(missing)}")
    source = frame.loc[:, source_columns]
    source = source.drop_duplicates(["eventNumber", "tob_index"])
    joined = keys.merge(
        source,
        on=["eventNumber", "tob_index"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined[CORE_COLUMNS + EM2_COLUMNS + ["tob_pt"]].isna().any().any():
        raise ValueError(f"pair member {suffix} could not be joined to source inputs")
    return joined


def _safe_ratio(numerator, denominator):
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.abs(denominator) > 1e-12,
    )
    return result


def _prepared_member_feature_arrays(
    pairs,
    left,
    right,
    events,
    member_features,
    left_core,
    right_core,
    left_em2,
    right_em2,
    left_pt,
    right_pt,
):
    """Build the compact, verified member features in declared config order."""
    requested = tuple(member_features or ())
    if not requested:
        raise ValueError("prepared_member_features requires member_features")
    if len(set(requested)) != len(requested):
        raise ValueError("member_features must not contain duplicates")
    unsupported = set(requested).difference(PREPARED_MEMBER_FEATURES)
    if unsupported:
        raise ValueError(f"unsupported prepared member features: {sorted(unsupported)}")

    left_summary = member_summary_features(left_core, left_em2)
    right_summary = member_summary_features(right_core, right_em2)
    context_map = events.set_index("eventNumber")["baseline_event_score"]
    context = pairs["eventNumber"].map(context_map).to_numpy(dtype=np.float32)
    if not np.isfinite(context).all():
        raise ValueError("event_second_highest_tob_pt is unavailable for a pair")

    def available(summary, pt):
        raw_dominance = summary[:, 12]
        return {
            "em2_raw_dominance": raw_dominance,
            "em2_normalized_dominance": _safe_ratio(
                raw_dominance,
                summary[:, 7],
            ),
            "em2_best_3x3_fraction": summary[:, 15],
            "measured_tob_pt": pt,
            "event_second_highest_tob_pt": context,
        }

    left_values = available(left_summary, left_pt)
    right_values = available(right_summary, right_pt)
    inputs = np.stack(
        [
            np.stack([left_values[name] for name in requested], axis=1),
            np.stack([right_values[name] for name in requested], axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError("prepared member features contain a non-finite value")
    return {"inputs": inputs}


def _representation_arrays(
    name,
    pairs,
    left,
    right,
    events,
    *,
    member_features=None,
    source_frame=None,
    context_features=None,
    include_member_pt=True,
    include_event_pt=True,
):
    left_core = left[CORE_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 5, 3, 3)
    right_core = right[CORE_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 5, 3, 3)
    left_em2 = left[EM2_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 12, 12)
    right_em2 = right[EM2_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, 12, 12)
    left_pt = left["tob_pt"].to_numpy(dtype=np.float32)
    right_pt = right["tob_pt"].to_numpy(dtype=np.float32)

    if name == "pair_pt":
        return {"inputs": pairs[["tob_pt_max", "tob_pt_min"]].to_numpy(np.float32)}
    if name == "raw_cells":
        return {"inputs": raw_cell_pair_features(left_core, right_core)}
    if name == "raw_cells_pt":
        return {"inputs": raw_cell_pt_pair_features(left_core, left_pt, right_core, right_pt)}
    if name == "member_coarse_cells":
        return {
            "inputs": np.stack(
                (left_core.reshape(-1, 45), right_core.reshape(-1, 45)),
                axis=1,
            ).astype(np.float32)
        }
    if name == "member_coarse_cells_pt":
        return {
            "inputs": np.stack(
                (
                    np.concatenate((left_core.reshape(-1, 45), left_pt[:, None]), axis=1),
                    np.concatenate((right_core.reshape(-1, 45), right_pt[:, None]), axis=1),
                ),
                axis=1,
            ).astype(np.float32)
        }
    if name == "member_compact_em2_pt":
        return {
            "inputs": compact_em2_member_features(
                np.stack((left_em2, right_em2), axis=1),
                np.stack((left_pt, right_pt), axis=1),
            )
        }
    if name == "summaries":
        return {"inputs": summary_pair_features(left_core, left_em2, right_core, right_em2)}
    if name == "summaries_pt":
        return {
            "inputs": summary_pt_pair_features(
                left_core,
                left_em2,
                left_pt,
                right_core,
                right_em2,
                right_pt,
            )
        }
    if name == "prepared_member_features":
        return _prepared_member_feature_arrays(
            pairs,
            left,
            right,
            events,
            member_features,
            left_core,
            right_core,
            left_em2,
            right_em2,
            left_pt,
            right_pt,
        )
    if name in {"high_resolution_em2", "high_resolution_em2_context"}:
        left_fraction = em2_best_3x3_fraction(left_em2)[:, None]
        right_fraction = em2_best_3x3_fraction(right_em2)[:, None]
        def member_scalars(core, pt, fraction):
            parts = [core.reshape(-1, 45)]
            if include_member_pt:
                parts.append(pt[:, None])
            parts.append(fraction)
            return np.concatenate(parts, axis=1)

        scalars = np.stack(
            [
                member_scalars(left_core, left_pt, left_fraction),
                member_scalars(right_core, right_pt, right_fraction),
            ],
            axis=1,
        )
        context_map = events.set_index("eventNumber")["baseline_event_score"]
        baseline_context = pairs["eventNumber"].map(context_map).to_numpy(np.float32)[:, None]
        if name == "high_resolution_em2":
            context = (
                baseline_context
                if include_event_pt
                else np.empty((len(pairs), 0), dtype=np.float32)
            )
        else:
            if source_frame is None:
                raise ValueError("context representation requires the source TOB table")
            mapped = build_pair_context_features(source_frame, pairs)
            added = select_context_features(mapped, context_features)
            parts = ([baseline_context] if include_event_pt else []) + [added]
            context = np.concatenate(parts, axis=1)
        return {
            "images": np.stack([left_em2, right_em2], axis=1),
            "scalars": scalars.astype(np.float32),
            "context": context,
        }
    raise ValueError(f"unknown pair representation: {name}")


def prepare_pair_split(
    data_dir,
    *,
    split,
    representation,
    member_features=None,
    context_features=None,
    include_member_pt=True,
    include_event_pt=True,
    include_member_min_truth_pt_gev=False,
    split_seed=42,
    max_events_per_class=None,
):
    """Build one prepared split and return portable NumPy arrays."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if include_member_min_truth_pt_gev and split != "train":
        raise ValueError(
            "member_min_truth_pt_gev is a training-only analysis field"
        )
    cache = TrainingDataCache(enabled=True)
    dataset = cache.get_dataset(data_dir, max_events_per_class, split_seed)
    inventory = _source_event_inventory(
        data_dir,
        split_seed,
        max_events_per_class=max_events_per_class,
    )
    split_inventory = inventory.loc[inventory["split"].eq(split)].copy()
    split_event_ids = set(split_inventory["eventNumber"])
    frame = dataset.frame.loc[
        dataset.frame["eventNumber"].isin(split_event_ids)
    ].copy().reset_index(drop=True)
    pairs = build_pair_dataset(frame, event_ids=split_inventory["eventNumber"])

    samples = split_inventory.set_index("eventNumber")["sample"]
    positive_counts = (
        pairs.selected_tobs.groupby("eventNumber", sort=False)["tob_label"].sum()
        .reindex(pairs.events["eventNumber"], fill_value=0)
        .astype(np.int64)
    )
    events = pairs.events.copy()
    events["sample"] = events["eventNumber"].map(samples)
    events["pair_observable"] = events["pair_count"].gt(0)
    events["pair_eligible"] = (
        events["sample"].eq("signal") & (positive_counts.to_numpy() >= 2)
    )
    events["training_eligible"] = np.where(
        events["sample"].eq("background"),
        events["pair_observable"],
        events["pair_eligible"],
    )

    pair_frame = pairs.pairs
    if split == "train":
        eligible_events = set(events.loc[events["training_eligible"], "eventNumber"])
        pair_frame = pair_frame.loc[
            pair_frame["eventNumber"].isin(eligible_events)
        ].reset_index(drop=True)

    extra_columns = ("truth_pt",) if include_member_min_truth_pt_gev else ()
    left = _member_frame(pair_frame, frame, "a", extra_columns=extra_columns)
    right = _member_frame(pair_frame, frame, "b", extra_columns=extra_columns)
    arrays = _representation_arrays(
        representation,
        pair_frame,
        left,
        right,
        events,
        member_features=member_features,
        source_frame=frame,
        context_features=context_features,
        include_member_pt=include_member_pt,
        include_event_pt=include_event_pt,
    )
    arrays.update(
        {
            "labels": pair_frame["binary_label"].to_numpy(dtype=np.float32),
            "three_class_labels": pair_frame["three_class_label"].to_numpy(dtype=np.int64),
            "pair_event_ids": pair_frame["eventNumber"].to_numpy(dtype=np.int64),
            "pair_indices": pair_frame["pair_index"].to_numpy(dtype=np.int64),
            "pair_tob_indices": pair_frame[
                ["tob_index_a", "tob_index_b"]
            ].to_numpy(dtype=np.int64),
            "pair_selected_ranks": pair_frame[
                ["selected_rank_a", "selected_rank_b"]
            ].to_numpy(dtype=np.int64),
            "pair_pt": pair_frame["tob_pt_min"].to_numpy(dtype=np.float64),
            "event_ids": events["eventNumber"].to_numpy(dtype=np.int64),
            "event_samples": events["sample"].eq("signal").to_numpy(dtype=np.uint8),
            "event_pair_observable": events["pair_observable"].to_numpy(dtype=np.uint8),
            "event_pair_eligible": events["pair_eligible"].to_numpy(dtype=np.uint8),
            "baseline_event_scores": events["baseline_event_score"].to_numpy(dtype=np.float64),
        }
    )
    if include_member_min_truth_pt_gev:
        arrays["member_min_truth_pt_gev"] = _member_min_truth_pt_gev(
            left,
            right,
            arrays["labels"],
        )
    return arrays


def preparation_contract(config):
    """Resolve the prepared representation and optional training analysis field."""
    representation = config.get("representation")
    if representation is None:
        raise KeyError("config requires representation")
    member_features = config.get("member_features")
    context_features = config.get("context_features")
    include_member_pt = bool(config.get("include_member_pt", True))
    include_event_pt = bool(
        config.get(
            "include_event_pt",
            representation == "high_resolution_em2" or not include_member_pt,
        )
    )
    if representation == "prepared_member_features":
        if not isinstance(member_features, list) or not member_features:
            raise ValueError("prepared_member_features config requires member_features")
        model = config.get("model", {})
        if model.get("name") != "shared_member_pair_mlp":
            raise ValueError("prepared member features require shared_member_pair_mlp")
        if int(model.get("member_width", -1)) != len(member_features):
            raise ValueError("model member_width must equal the member feature count")
    screen_widths = {
        "member_coarse_cells": 45,
        "member_coarse_cells_pt": 46,
        "member_compact_em2_pt": 17,
    }
    if representation in screen_widths:
        model = config.get("model", {})
        expected_width = screen_widths[representation]
        if model.get("name") != "shared_member_pair_mlp":
            raise ValueError(
                "reproduced member representations require shared_member_pair_mlp"
            )
        if int(model.get("member_width", -1)) != expected_width:
            raise ValueError("model member_width disagrees with the representation")
        if model.get("member_encoder") != [expected_width, 32, 16]:
            raise ValueError("model member_encoder must be [member_width, 32, 16]")
        if model.get("fusion") != "symmetric_sum_absolute_difference":
            raise ValueError("model fusion must be symmetric_sum_absolute_difference")
        if model.get("pair_head") != [32, 32, 16, 1]:
            raise ValueError("model pair_head must be [32, 32, 16, 1]")
        if model.get("activation") != "leaky_relu_0.01":
            raise ValueError("model activation must be leaky_relu_0.01")
        if model.get("initialization") != "pytorch_default":
            raise ValueError("model initialization must be pytorch_default")
    if representation == "high_resolution_em2_context":
        if not isinstance(context_features, list) or not context_features:
            raise ValueError("context representation requires context_features")
        model = config.get("model", {})
        expected_scalar_width = 47 if include_member_pt else 46
        expected_context_width = len(context_features) + int(include_event_pt)
        if model.get("name") != "shared_member_high_resolution_em2":
            raise ValueError("context representation requires the high-resolution model")
        if int(model.get("member_scalar_width", -1)) != expected_scalar_width:
            raise ValueError("model member_scalar_width disagrees with the representation")
        if int(model.get("context_width", -1)) != expected_context_width:
            raise ValueError("model context_width disagrees with the representation")
    weighting = config.get("loss", {}).get("pair_weighting", "equal")
    include_truth = weighting in {
        "inverse_frequency_member_min",
        "power_law_pminus1_member_min",
    }
    if include_truth and config.get("loss", {}).get("weight_coordinate") != (
        "member_min_truth_pt_gev"
    ):
        raise ValueError("energy-weighted config must declare member_min_truth_pt_gev")
    if representation == "high_resolution_em2":
        model = config.get("model", {})
        expected_scalar_width = 47 if include_member_pt else 46
        expected_context_width = int(include_event_pt)
        if model.get("name") == "shared_member_high_resolution_em2":
            if int(model.get("member_scalar_width", expected_scalar_width)) != expected_scalar_width:
                raise ValueError("model member_scalar_width disagrees with the representation")
            if int(model.get("context_width", expected_context_width)) != expected_context_width:
                raise ValueError("model context_width disagrees with the representation")
    return (
        representation,
        member_features,
        context_features,
        include_member_pt,
        include_event_pt,
        include_truth,
    )


def preparation_contract_for_split(config, split):
    """Resolve one config without exposing its training-only weight coordinate."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    contract = preparation_contract(config)
    return (*contract[:-1], contract[-1] and split == "train")


def _member_min_truth_pt_gev(left, right, labels):
    """Return the matched-member minimum truth pT for training weights only.

    This is a TOB-associated simulation analysis coordinate. It does not assert
    that the two positive TOBs correspond to two unique generator particles.
    """
    truth = np.minimum(
        left["truth_pt"].to_numpy(dtype=np.float64),
        right["truth_pt"].to_numpy(dtype=np.float64),
    ) / 1000.0
    positive = np.asarray(labels, dtype=bool)
    if truth.shape != positive.shape:
        raise ValueError("matched-member truth values must align with labels")
    if not np.isfinite(truth[positive]).all():
        raise ValueError(
            "member_min_truth_pt_gev is unavailable for a positive training pair"
        )
    return truth


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pair-model input arrays")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config")
    choice.add_argument(
        "--representation",
        choices=[
            "pair_pt",
            "raw_cells",
            "raw_cells_pt",
            "summaries",
            "summaries_pt",
            "member_coarse_cells",
            "member_coarse_cells_pt",
            "member_compact_em2_pt",
            "high_resolution_em2",
            "high_resolution_em2_context",
        ],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_events_per_class", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    member_features = None
    context_features = None
    include_member_pt = True
    include_event_pt = True
    include_truth = False
    representation = args.representation
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        (
            representation,
            member_features,
            context_features,
            include_member_pt,
            include_event_pt,
            include_truth,
        ) = preparation_contract_for_split(
            config,
            args.split,
        )
    arrays = prepare_pair_split(
        args.data_dir,
        split=args.split,
        representation=representation,
        member_features=member_features,
        context_features=context_features,
        include_member_pt=include_member_pt,
        include_event_pt=include_event_pt,
        include_member_min_truth_pt_gev=include_truth,
        split_seed=args.split_seed,
        max_events_per_class=args.max_events_per_class,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


if __name__ == "__main__":
    main()
