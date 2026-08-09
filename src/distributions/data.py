from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "eventNumber",
    "tob_index",
    "signal",
    "truth_pt",
    "prongs",
    "tob_pt",
    "tob_eta",
    "tob_phi",
    "tob_bdt",
    "Type",
]


def _read_sample_metadata(csv_path: Path, sample_type: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, usecols=METADATA_COLUMNS)
    frame = frame.rename(columns={"eventNumber": "original_event_number"})
    frame["sample_type"] = sample_type
    frame["label"] = frame["signal"].astype(np.int8)
    return frame


def load_metadata(data_dir: str | Path) -> pd.DataFrame:
    """Load object metadata and create collision-safe event identifiers.

    The Signal and Background files reuse event numbers.  To match train.py
    exactly, Background event IDs receive the same numeric offset used by the
    training pipeline.  The original event number is retained for inspection.
    """
    data_dir = Path(data_dir).resolve()
    signal_path = data_dir / "Signal" / "signal_combined.csv"
    background_path = data_dir / "Background" / "bkg_combined.csv"
    missing = [str(path) for path in (signal_path, background_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required CSV files:\n" + "\n".join(missing))

    signal = _read_sample_metadata(signal_path, "signal")
    background = _read_sample_metadata(background_path, "background")

    offset = 2 * int(signal["original_event_number"].max())
    signal["event_uid"] = signal["original_event_number"].astype(np.int64)
    background["event_uid"] = (
        background["original_event_number"].astype(np.int64) + offset
    )

    objects = pd.concat([signal, background], ignore_index=True)
    objects["event_tau_count"] = (
        objects.groupby("event_uid", sort=False)["label"].transform("sum").astype(np.int16)
    )
    return objects


def split_event_ids(event_ids: Iterable[int], seed: int) -> dict[str, np.ndarray]:
    """Reproduce train.py's event-level 70/10/20 random split exactly."""
    unique_events = np.unique(np.asarray(list(event_ids), dtype=np.int64))
    random_state = np.random.RandomState(seed)
    random_state.shuffle(unique_events)

    train_end = int(len(unique_events) * 0.70)
    validation_end = int(len(unique_events) * 0.80)
    return {
        "train": unique_events[:train_end],
        "validation": unique_events[train_end:validation_end],
        "test": unique_events[validation_end:],
    }


def add_split_column(objects: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return metadata with a train/validation/test label on every object."""
    result = objects.copy()
    splits = split_event_ids(result["event_uid"].unique(), seed)
    split_by_event = pd.Series(index=result["event_uid"].unique(), dtype="object")
    for split_name, event_ids in splits.items():
        split_by_event.loc[event_ids] = split_name
    result["split"] = result["event_uid"].map(split_by_event)
    if result["split"].isna().any():
        raise RuntimeError("At least one event was not assigned to a data split")
    return result


def select_scope(objects: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Select one configured split, train+validation, or the complete data."""
    aliases = {"val": "validation", "train_validation": "train+validation"}
    scope = aliases.get(scope, scope)
    allowed = {"train", "validation", "test", "all", "train+validation"}
    if scope not in allowed:
        raise ValueError(f"Unknown data scope '{scope}'. Expected one of {sorted(allowed)}")
    if scope == "all":
        return objects.copy()
    if scope == "train+validation":
        return objects[objects["split"].isin(["train", "validation"])].copy()
    return objects[objects["split"].eq(scope)].copy()


def limit_events_per_sample(
    objects: pd.DataFrame, max_events_per_sample: int | None, seed: int
) -> pd.DataFrame:
    """Optionally retain a deterministic small event sample for smoke tests."""
    if max_events_per_sample is None:
        return objects
    if max_events_per_sample <= 0:
        raise ValueError("max_events_per_sample must be positive")

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for _, sample in objects.groupby("sample_type", sort=False):
        event_ids = sample["event_uid"].unique()
        if len(event_ids) > max_events_per_sample:
            event_ids = rng.choice(event_ids, size=max_events_per_sample, replace=False)
        selected.append(np.asarray(event_ids, dtype=np.int64))
    keep = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    return objects[objects["event_uid"].isin(keep)].copy()


def _npz_path(data_dir: Path, sample_type: str) -> Path:
    if sample_type == "signal":
        return data_dir / "Signal" / "signal_combined.npz"
    if sample_type == "background":
        return data_dir / "Background" / "bkg_combined.npz"
    raise ValueError(f"Unknown sample type: {sample_type}")


def iter_aligned_batches(
    objects: pd.DataFrame,
    data_dir: str | Path,
    batch_size: int,
    need_tensors: bool,
    need_em2: bool,
):
    """Yield metadata aligned with only the NPZ tensors required by a plot.

    Alignment uses (original_event_number, tob_index), the same relationship
    used in train.py.  Batching prevents construction of one enormous DataFrame
    containing all 189 tensor columns for every object.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_dir = Path(data_dir).resolve()

    for sample_type, sample_objects in objects.groupby("sample_type", sort=False):
        sample_objects = sample_objects.reset_index(drop=True)
        if not need_tensors and not need_em2:
            for start in range(0, len(sample_objects), batch_size):
                yield sample_objects.iloc[start:start + batch_size].copy()
            continue

        npz_path = _npz_path(data_dir, sample_type)
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing required NPZ file: {npz_path}")

        with np.load(npz_path) as npz:
            event_numbers = np.asarray(npz["event_nums"])
            event_index = pd.Index(event_numbers)
            tensors_flat = (
                np.asarray(npz["X_tensors"]).reshape(-1, 45) if need_tensors else None
            )
            em2_flat = (
                np.asarray(npz["X_em2_tensors"]).reshape(-1, 144) if need_em2 else None
            )

            for start in range(0, len(sample_objects), batch_size):
                batch = sample_objects.iloc[start:start + batch_size].copy()
                event_positions = event_index.get_indexer(batch["original_event_number"])
                if np.any(event_positions < 0):
                    missing_event = batch.loc[event_positions < 0, "original_event_number"].iloc[0]
                    raise KeyError(
                        f"Event {missing_event} exists in CSV but not in {npz_path.name}"
                    )
                tob_indices = batch["tob_index"].to_numpy(dtype=np.int64)
                if np.any((tob_indices < 0) | (tob_indices >= 6)):
                    raise ValueError("tob_index must be between 0 and 5")
                flat_indices = event_positions * 6 + tob_indices

                if tensors_flat is not None:
                    tensor_frame = pd.DataFrame(
                        tensors_flat[flat_indices],
                        columns=[f"tensor_{index}" for index in range(45)],
                        index=batch.index,
                    )
                    batch = pd.concat([batch, tensor_frame], axis=1)
                if em2_flat is not None:
                    em2_frame = pd.DataFrame(
                        em2_flat[flat_indices],
                        columns=[f"em2_cell_{index}" for index in range(144)],
                        index=batch.index,
                    )
                    batch = pd.concat([batch, em2_frame], axis=1)
                yield batch
