from __future__ import annotations

import numpy as np
import pandas as pd

from src.distributions.data import add_split_column


def _ks_distance(first: pd.Series, second: pd.Series) -> float:
    """Exact two-sample Kolmogorov-Smirnov distance without an extra dependency."""
    first_values = np.sort(first.dropna().to_numpy(dtype=np.float64))
    second_values = np.sort(second.dropna().to_numpy(dtype=np.float64))
    if first_values.size == 0 or second_values.size == 0:
        return float("nan")
    combined = np.sort(np.concatenate([first_values, second_values]))
    first_cdf = np.searchsorted(first_values, combined, side="right") / first_values.size
    second_cdf = np.searchsorted(second_values, combined, side="right") / second_values.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def build_split_audit(objects: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Measure whether each legacy 70/10/20 split represents the full dataset."""
    full_tau_pt = objects.loc[objects["label"].eq(1), "tob_pt"]
    full_noise_pt = objects.loc[objects["label"].eq(0), "tob_pt"]
    rows: list[dict] = []

    for seed in seeds:
        split_objects = add_split_column(objects.drop(columns=["split"], errors="ignore"), seed)
        event_table = split_objects.drop_duplicates("event_uid")
        total_events = len(event_table)

        for split_name in ("train", "validation", "test"):
            split_rows = split_objects[split_objects["split"].eq(split_name)]
            split_events = event_table[event_table["split"].eq(split_name)]
            tau_counts = split_events["event_tau_count"]
            rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "event_count": len(split_events),
                    "event_fraction": len(split_events) / total_events,
                    "object_count": len(split_rows),
                    "tau_object_fraction": split_rows["label"].mean(),
                    "zero_tau_event_fraction": tau_counts.eq(0).mean(),
                    "one_tau_event_fraction": tau_counts.eq(1).mean(),
                    "two_plus_tau_event_fraction": tau_counts.ge(2).mean(),
                    "signal_sample_event_fraction": split_events["sample_type"].eq("signal").mean(),
                    "tau_tob_pt_ks_vs_full": _ks_distance(
                        split_rows.loc[split_rows["label"].eq(1), "tob_pt"], full_tau_pt
                    ),
                    "noise_tob_pt_ks_vs_full": _ks_distance(
                        split_rows.loc[split_rows["label"].eq(0), "tob_pt"], full_noise_pt
                    ),
                }
            )
    return pd.DataFrame(rows)
