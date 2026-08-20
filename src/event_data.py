"""Event-preserving datasets used by constrained trigger training."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class EventBatch:
    features: torch.Tensor
    labels: torch.Tensor
    truth_pt_gev: torch.Tensor
    tob_pt_gev: torch.Tensor
    object_mask: torch.Tensor
    signal_object_mask: torch.Tensor
    background_event_mask: torch.Tensor
    event_numbers: torch.Tensor

    def to(self, device):
        return EventBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            truth_pt_gev=self.truth_pt_gev.to(device),
            tob_pt_gev=self.tob_pt_gev.to(device),
            object_mask=self.object_mask.to(device),
            signal_object_mask=self.signal_object_mask.to(device),
            background_event_mask=self.background_event_mask.to(device),
            event_numbers=self.event_numbers.to(device),
        )


def _gev(values):
    values = np.asarray(values, dtype=np.float64).copy()
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size and np.median(finite) > 1000.0:
        values /= 1000.0
    return values


def split_training_events(frame, seed, constraint_fraction=0.3):
    """Split training rows by stratified events without object leakage."""
    if not 0.0 < constraint_fraction < 1.0:
        raise ValueError("constraint_fraction must be in (0, 1)")
    required = {"eventNumber", "Type", "signal", "truth_pt"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing event split columns: {sorted(missing)}")

    work = frame.reset_index(drop=True).copy()
    work["_truth_pt_gev"] = _gev(work["truth_pt"])
    event_rows = []
    for event_number, event in work.groupby("eventNumber", sort=False):
        is_background = bool(event["Type"].isin(["BKG", "Background"]).all())
        tau_rows = event[(event["Type"] == "Signal") & (event["signal"] == 1)]
        tau_count = len(tau_rows)
        tau_group = 0 if tau_count == 0 else (1 if tau_count == 1 else 2)
        max_truth_pt = (
            float(tau_rows["_truth_pt_gev"].max()) if tau_count else -1.0
        )
        if max_truth_pt < 25.0:
            pt_group = 0
        elif max_truth_pt < 40.0:
            pt_group = 1
        elif max_truth_pt < 60.0:
            pt_group = 2
        else:
            pt_group = 3
        event_rows.append(
            {
                "eventNumber": event_number,
                "stratum": (int(is_background), tau_group, pt_group),
            }
        )

    events = pd.DataFrame(event_rows)
    rng = np.random.RandomState(seed)
    primal_events = []
    constraint_events = []
    for _, group in events.groupby("stratum", sort=False):
        numbers = group["eventNumber"].to_numpy(copy=True)
        rng.shuffle(numbers)
        if len(numbers) == 1:
            # Tiny strata remain in primal training; broad strata populate both.
            primal_events.extend(numbers.tolist())
            continue
        constraint_count = int(round(len(numbers) * constraint_fraction))
        constraint_count = min(max(constraint_count, 1), len(numbers) - 1)
        constraint_events.extend(numbers[:constraint_count].tolist())
        primal_events.extend(numbers[constraint_count:].tolist())

    primal_mask = work["eventNumber"].isin(primal_events).to_numpy()
    constraint_mask = work["eventNumber"].isin(constraint_events).to_numpy()
    if np.any(primal_mask & constraint_mask):
        raise AssertionError("Primal and constraint events overlap")
    if not np.all(primal_mask | constraint_mask):
        raise AssertionError("Every training object must belong to one inner split")
    return np.flatnonzero(primal_mask), np.flatnonzero(constraint_mask)


class EventTensorDataset(Dataset):
    """Expose complete variable-length events to a padded collate function."""

    def __init__(self, features, labels, frame, row_indices=None):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        self.frame = frame.reset_index(drop=True).copy()
        if len(self.features) != len(self.frame) or len(self.labels) != len(self.frame):
            raise ValueError("Features, labels, and metadata must align")

        if row_indices is None:
            row_indices = np.arange(len(self.frame), dtype=np.int64)
        row_indices = np.asarray(row_indices, dtype=np.int64)
        selected = self.frame.iloc[row_indices]
        self.event_rows = [
            group.index.to_numpy(dtype=np.int64)
            for _, group in selected.groupby("eventNumber", sort=False)
        ]
        self.truth_pt_gev = _gev(self.frame["truth_pt"])
        self.tob_pt_gev = _gev(self.frame["tob_pt"])

    def __len__(self):
        return len(self.event_rows)

    def __getitem__(self, index):
        rows = self.event_rows[index]
        event = self.frame.iloc[rows]
        background = bool(event["Type"].isin(["BKG", "Background"]).all())
        signal_mask = (
            (event["Type"].to_numpy() == "Signal")
            & (event["signal"].to_numpy(dtype=np.int64) == 1)
        )
        return {
            "features": torch.from_numpy(self.features[rows]),
            "labels": torch.from_numpy(self.labels[rows]),
            "truth_pt_gev": torch.from_numpy(
                self.truth_pt_gev[rows].astype(np.float32)
            ),
            "tob_pt_gev": torch.from_numpy(
                self.tob_pt_gev[rows].astype(np.float32)
            ),
            "signal_object_mask": torch.from_numpy(signal_mask),
            "background_event": background,
            "event_number": int(event["eventNumber"].iloc[0]),
        }


def collate_events(events):
    """Pad complete events while marking every artificial object as invalid."""
    if not events:
        raise ValueError("Cannot collate an empty event batch")
    batch_size = len(events)
    max_objects = max(len(event["labels"]) for event in events)
    feature_count = events[0]["features"].shape[1]

    features = torch.zeros(batch_size, max_objects, feature_count)
    labels = torch.zeros(batch_size, max_objects)
    truth_pt = torch.zeros(batch_size, max_objects)
    tob_pt = torch.zeros(batch_size, max_objects)
    object_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    signal_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)

    for row, event in enumerate(events):
        count = len(event["labels"])
        features[row, :count] = event["features"]
        labels[row, :count] = event["labels"]
        truth_pt[row, :count] = event["truth_pt_gev"]
        tob_pt[row, :count] = event["tob_pt_gev"]
        object_mask[row, :count] = True
        signal_mask[row, :count] = event["signal_object_mask"]

    return EventBatch(
        features=features,
        labels=labels,
        truth_pt_gev=truth_pt,
        tob_pt_gev=tob_pt,
        object_mask=object_mask,
        signal_object_mask=signal_mask,
        background_event_mask=torch.tensor(
            [event["background_event"] for event in events],
            dtype=torch.bool,
        ),
        event_numbers=torch.tensor(
            [event["event_number"] for event in events],
            dtype=torch.int64,
        ),
    )
