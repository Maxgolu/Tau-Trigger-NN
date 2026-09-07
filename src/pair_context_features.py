"""Observable event and pair pT context for the high-resolution pair model."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_FEATURES = (
    "event_sum_pt",
    "outside_pair_max_pt",
    "outside_pair_sum_pt",
    "pair_fraction_event_pt",
    "top4_pt_concentration",
    "top4_pt_entropy",
    "pair_sum_pt",
    "pair_balance",
)


def _safe_ratio(numerator, denominator):
    return 0.0 if denominator == 0.0 else numerator / denominator


def build_pair_context_features(objects, pairs):
    """Return one row of observable pT summaries for every canonical pair.

    Event sums use all finite TOBs. Concentration and entropy use the selected
    top four. The returned row order is identical to ``pairs``.
    """
    object_columns = {"eventNumber", "tob_index", "tob_pt"}
    pair_columns = {
        "eventNumber", "pair_index", "tob_index_a", "tob_index_b",
        "tob_pt_a", "tob_pt_b",
    }
    if missing := object_columns.difference(objects.columns):
        raise KeyError(f"object data is missing {sorted(missing)}")
    if missing := pair_columns.difference(pairs.columns):
        raise KeyError(f"pair data is missing {sorted(missing)}")
    if objects.duplicated(["eventNumber", "tob_index"]).any():
        raise ValueError("TOB identity must be unique within an event")
    if pairs.duplicated(["eventNumber", "pair_index"]).any():
        raise ValueError("pair identity must be unique within an event")

    object_groups = {
        event: group for event, group in objects.groupby("eventNumber", sort=False)
    }
    rows = []
    for event, group in pairs.groupby("eventNumber", sort=False):
        if event not in object_groups:
            raise ValueError("pair event is absent from the TOB table")
        source = object_groups[event]
        pt = pd.to_numeric(source["tob_pt"], errors="coerce").to_numpy(np.float64)
        indices = source["tob_index"].to_numpy()
        finite = np.isfinite(pt)
        if np.any(pt[finite] < 0.0):
            raise ValueError("finite measured tob_pt must be nonnegative")
        observable = {
            index: float(value)
            for index, value, keep in zip(indices, pt, finite, strict=True)
            if keep
        }
        selected = sorted(observable, key=lambda index: (-observable[index], index))[:4]
        expected = len(selected) * (len(selected) - 1) // 2
        if len(group) != expected:
            raise ValueError("pair inventory does not match the selected top four")

        event_sum = float(sum(observable.values()))
        top4_sum = float(sum(observable[index] for index in selected))
        fractions = (
            np.asarray([observable[index] / top4_sum for index in selected])
            if top4_sum != 0.0 else np.zeros(len(selected), dtype=np.float64)
        )
        positive = fractions[fractions > 0.0]
        concentration = float(np.square(fractions).sum())
        entropy = float(-(positive * np.log(positive)).sum())
        observed = set()

        for pair in group.itertuples(index=False):
            left, right = pair.tob_index_a, pair.tob_index_b
            left_pt, right_pt = float(pair.tob_pt_a), float(pair.tob_pt_b)
            if left not in selected or right not in selected or left == right:
                raise ValueError("pair members must be distinct selected TOBs")
            if left_pt != observable[left] or right_pt != observable[right]:
                raise ValueError("pair pT does not match its TOB identity")
            if left_pt < right_pt or (left_pt == right_pt and left > right):
                raise ValueError("pair members are not in canonical order")
            identity = frozenset((left, right))
            if identity in observed:
                raise ValueError("duplicate unordered pair")
            observed.add(identity)
            outside = [value for index, value in observable.items()
                       if index not in (left, right)]
            pair_sum = left_pt + right_pt
            rows.append({
                "eventNumber": event,
                "pair_index": pair.pair_index,
                "event_sum_pt": event_sum,
                "outside_pair_max_pt": max(outside, default=0.0),
                "outside_pair_sum_pt": float(sum(outside)),
                "pair_fraction_event_pt": _safe_ratio(pair_sum, event_sum),
                "top4_pt_concentration": concentration,
                "top4_pt_entropy": entropy,
                "pair_sum_pt": pair_sum,
                "pair_balance": _safe_ratio(left_pt - right_pt, pair_sum),
            })

    result = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(columns=("eventNumber", "pair_index", *CONTEXT_FEATURES))
    result = result.set_index(["eventNumber", "pair_index"]).loc[
        pd.MultiIndex.from_frame(pairs[["eventNumber", "pair_index"]])
    ].reset_index()
    values = result[list(CONTEXT_FEATURES)].to_numpy(np.float64)
    if values.shape != (len(pairs), len(CONTEXT_FEATURES)) or not np.isfinite(values).all():
        raise ValueError("context feature mapping is incomplete or non-finite")
    return result


def select_context_features(mapped, names):
    """Select declared feature columns in order as a float32 matrix."""
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("context_features must be nonempty and unique")
    if unknown := set(names).difference(CONTEXT_FEATURES):
        raise ValueError(f"unknown context features: {sorted(unknown)}")
    values = mapped.loc[:, names].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("context features contain a non-finite value")
    return values
