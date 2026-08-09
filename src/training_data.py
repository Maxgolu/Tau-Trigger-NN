from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .features import FEATURE_REGISTRY
except ImportError:
    from features import FEATURE_REGISTRY


@dataclass(frozen=True)
class DataKey:
    """Identifies one deterministic aligned dataset."""

    data_dir: str
    max_events_per_class: int | None
    sampling_seed: int | None


@dataclass(frozen=True)
class SplitKey:
    """Identifies one event-level split of an aligned dataset."""

    data_key: DataKey
    seed: int
    train_fraction: float
    validation_fraction: float


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class PreparedDataset:
    """Stores aligned object rows and immutable split inputs."""

    key: DataKey
    frame: pd.DataFrame
    event_numbers: np.ndarray
    labels: np.ndarray


def _load_csv_for_events(csv_path: Path, event_numbers=None) -> pd.DataFrame:
    """Load a full CSV, or stream-filter it for a smoke-test subset."""
    if event_numbers is None:
        return pd.read_csv(csv_path)

    selected_events = set(np.asarray(event_numbers).tolist())
    filtered_chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        selected = chunk[chunk["eventNumber"].isin(selected_events)]
        if not selected.empty:
            filtered_chunks.append(selected)

    if not filtered_chunks:
        raise ValueError(f"No CSV rows matched the selected events in {csv_path}")

    return pd.concat(filtered_chunks, ignore_index=True)


def _load_npz_arrays(npz_path: Path, max_events, rng) -> dict[str, np.ndarray]:
    """Load required arrays, optionally retaining a random event subset."""
    with np.load(npz_path) as data:
        event_count = len(data["event_nums"])
        if max_events is None or max_events >= event_count:
            indices = slice(None)
        else:
            indices = np.sort(
                rng.choice(event_count, size=max_events, replace=False)
            )

        arrays = {
            "event_nums": data["event_nums"][indices].copy(),
            "X_tensors": data["X_tensors"][indices].copy(),
            "X_em2_tensors": data["X_em2_tensors"][indices].copy(),
            "X_feats": data["X_feats"][indices].copy(),
        }

    return arrays


def _make_data_key(
    data_dir: str | Path,
    max_events_per_class: int | None,
    sampling_seed: int,
) -> DataKey:
    """Build a key from every setting that can change the aligned rows."""
    return DataKey(
        data_dir=str(Path(data_dir).resolve()),
        max_events_per_class=max_events_per_class,
        sampling_seed=(
            sampling_seed if max_events_per_class is not None else None
        ),
    )


def load_and_align_dataset(
    data_dir: str | Path,
    max_events_per_class: int | None,
    sampling_seed: int,
) -> PreparedDataset:
    """Load and align CSV metadata with the NPZ object tensors."""
    if max_events_per_class is not None and max_events_per_class <= 0:
        raise ValueError("max_events_per_class must be a positive integer")

    key = _make_data_key(
        data_dir,
        max_events_per_class,
        sampling_seed,
    )
    data_path = Path(key.data_dir)
    sig_csv_path = data_path / "Signal" / "signal_combined.csv"
    sig_npz_path = data_path / "Signal" / "signal_combined.npz"
    bkg_csv_path = data_path / "Background" / "bkg_combined.csv"
    bkg_npz_path = data_path / "Background" / "bkg_combined.npz"

    required_paths = [
        sig_csv_path,
        sig_npz_path,
        bkg_csv_path,
        bkg_npz_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required data files:\n" + "\n".join(missing_paths)
        )

    rng = np.random.default_rng(sampling_seed)
    npz_sig = _load_npz_arrays(sig_npz_path, max_events_per_class, rng)
    npz_bkg = _load_npz_arrays(bkg_npz_path, max_events_per_class, rng)
    df_sig = _load_csv_for_events(
        sig_csv_path,
        npz_sig["event_nums"] if max_events_per_class else None,
    )
    df_bkg = _load_csv_for_events(
        bkg_csv_path,
        npz_bkg["event_nums"] if max_events_per_class else None,
    )

    if max_events_per_class:
        print(
            f"Smoke-test subset: {len(npz_sig['event_nums'])} signal + "
            f"{len(npz_bkg['event_nums'])} background events"
        )

    # Keep signal and background event identifiers separate.
    ev_nums_sig = npz_sig["event_nums"]
    ev_nums_bkg = npz_bkg["event_nums"].copy()
    offset = 2 * ev_nums_sig.max()
    ev_nums_bkg += offset
    df_bkg["eventNumber"] = df_bkg["eventNumber"] + offset

    event_nums_all = np.concatenate([ev_nums_sig, ev_nums_bkg], axis=0)
    X_tensors_all = np.concatenate(
        [npz_sig["X_tensors"], npz_bkg["X_tensors"]], axis=0
    )
    X_em2tensors_all = np.concatenate(
        [npz_sig["X_em2_tensors"], npz_bkg["X_em2_tensors"]], axis=0
    )
    X_feats_all = np.concatenate(
        [npz_sig["X_feats"], npz_bkg["X_feats"]], axis=0
    )
    df_all = pd.concat([df_sig, df_bkg], ignore_index=True)

    print("Computing Relative Eta/Phi...")
    pt = X_feats_all[:, :, 1]
    max_idx = np.argmax(pt, axis=1)
    eta_ref = X_feats_all[np.arange(len(X_feats_all)), max_idx][:, None, 2]
    phi_ref = X_feats_all[np.arange(len(X_feats_all)), max_idx][:, None, 3]

    X_feats_rel = X_feats_all.copy()
    X_feats_rel[:, :, 2] -= eta_ref
    X_feats_rel[:, :, 3] = (
        X_feats_rel[:, :, 3] - phi_ref + np.pi
    ) % (2 * np.pi) - np.pi

    print("Flattening and Aligning objects with CSV...")
    X_tens_flat = X_tensors_all.reshape(-1, 45)
    X_feat_flat = X_feats_rel.reshape(-1, 4)
    em2_spatial_size = (
        X_em2tensors_all.shape[2] * X_em2tensors_all.shape[3]
    )
    X_em2_flat = X_em2tensors_all.reshape(-1, em2_spatial_size)

    groups_flat = np.repeat(event_nums_all, 6)
    tob_index_flat = np.tile(np.arange(6), len(event_nums_all))
    csv_lookup_keys = (
        df_all["eventNumber"].astype(str)
        + "_"
        + df_all["tob_index"].astype(str)
    )
    npz_lookup_keys = (
        pd.Series(groups_flat).astype(str)
        + "_"
        + pd.Series(tob_index_flat).astype(str)
    )

    indexer = pd.Series(np.arange(len(groups_flat)), index=npz_lookup_keys)
    valid_indices = indexer.loc[csv_lookup_keys].values
    groups_aligned = groups_flat[valid_indices]
    X_tens_aligned = X_tens_flat[valid_indices]
    X_feat_aligned = X_feat_flat[valid_indices]
    X_em2_aligned = X_em2_flat[valid_indices]

    tensor_cols = [f"tensor_{i}" for i in range(45)]
    feat_cols = [f"feat_{i}" for i in range(4)]
    em2_cols = [f"em2_cell_{i}" for i in range(em2_spatial_size)]
    df_tensors = pd.DataFrame(
        X_tens_aligned,
        columns=tensor_cols,
        index=df_all.index,
    )
    df_feats = pd.DataFrame(
        X_feat_aligned,
        columns=feat_cols,
        index=df_all.index,
    )
    df_em2 = pd.DataFrame(
        X_em2_aligned,
        columns=em2_cols,
        index=df_all.index,
    )

    frame = pd.concat(
        [df_all, df_tensors, df_feats, df_em2],
        axis=1,
    )
    frame["label"] = frame["signal"].values.astype(np.float32)

    event_numbers = np.asarray(groups_aligned).copy()
    labels = frame["label"].to_numpy(copy=True)
    event_numbers.flags.writeable = False
    labels.flags.writeable = False

    return PreparedDataset(
        key=key,
        frame=frame,
        event_numbers=event_numbers,
        labels=labels,
    )


def _feature_array(values, feature_name: str, expected_rows: int) -> np.ndarray:
    """Normalize a registry result to an immutable two-dimensional array."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"Feature '{feature_name}' returned {values.ndim} dimensions; expected 2."
        )
    if len(values) != expected_rows:
        raise ValueError(
            f"Feature '{feature_name}' returned {len(values)} rows; "
            f"expected {expected_rows}."
        )

    # Detach cached values from mutable DataFrame storage.
    values = np.array(values, copy=True)
    values.flags.writeable = False
    return values


class TrainingDataCache:
    """Reuses deterministic preparation work across configuration sweeps."""

    def __init__(self, enabled: bool = True, feature_cache_mb: int = 512):
        if feature_cache_mb < 0:
            raise ValueError("feature_cache_mb cannot be negative")

        self.enabled = enabled
        self.feature_cache_limit = feature_cache_mb * 1024**2
        self._datasets: dict[DataKey, PreparedDataset] = {}
        self._splits: dict[SplitKey, SplitIndices] = {}
        self._features: OrderedDict[
            tuple[DataKey, str], np.ndarray
        ] = OrderedDict()
        self._feature_cache_bytes = 0
        self.stats = {
            "dataset_hits": 0,
            "dataset_misses": 0,
            "split_hits": 0,
            "split_misses": 0,
            "feature_hits": 0,
            "feature_misses": 0,
        }

    def get_dataset(
        self,
        data_dir: str | Path,
        max_events_per_class: int | None,
        seed: int,
    ) -> PreparedDataset:
        key = _make_data_key(data_dir, max_events_per_class, seed)
        if self.enabled and key in self._datasets:
            self.stats["dataset_hits"] += 1
            print("Data cache hit: reusing aligned dataset")
            return self._datasets[key]

        self.stats["dataset_misses"] += 1
        print("Loading NPZ and CSV data...")
        dataset = load_and_align_dataset(
            data_dir=key.data_dir,
            max_events_per_class=max_events_per_class,
            sampling_seed=seed,
        )
        if self.enabled:
            self._datasets[key] = dataset
        return dataset

    def get_split(
        self,
        dataset: PreparedDataset,
        seed: int,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.10,
    ) -> SplitIndices:
        if train_fraction <= 0 or validation_fraction < 0:
            raise ValueError("Split fractions must be non-negative")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave a test set")

        key = SplitKey(
            data_key=dataset.key,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        if self.enabled and key in self._splits:
            self.stats["split_hits"] += 1
            print(f"Split cache hit: reusing split for seed {seed}")
            return self._splits[key]

        self.stats["split_misses"] += 1
        unique_events = np.unique(dataset.event_numbers).copy()
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_events)

        train_end = int(len(unique_events) * train_fraction)
        validation_end = int(
            len(unique_events) * (train_fraction + validation_fraction)
        )
        train_events = unique_events[:train_end]
        validation_events = unique_events[train_end:validation_end]
        test_events = unique_events[validation_end:]

        split = SplitIndices(
            train=np.flatnonzero(
                np.isin(dataset.event_numbers, train_events)
            ),
            validation=np.flatnonzero(
                np.isin(dataset.event_numbers, validation_events)
            ),
            test=np.flatnonzero(
                np.isin(dataset.event_numbers, test_events)
            ),
        )
        split.train.flags.writeable = False
        split.validation.flags.writeable = False
        split.test.flags.writeable = False

        if self.enabled:
            self._splits[key] = split
        return split

    def get_feature(
        self,
        dataset: PreparedDataset,
        feature_name: str,
    ) -> np.ndarray:
        if feature_name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature: {feature_name}")

        key = (dataset.key, feature_name)
        if self.enabled and key in self._features:
            self.stats["feature_hits"] += 1
            self._features.move_to_end(key)
            print(f"Feature cache hit: {feature_name}")
            return self._features[key]

        self.stats["feature_misses"] += 1
        values = _feature_array(
            FEATURE_REGISTRY[feature_name](dataset.frame),
            feature_name,
            len(dataset.frame),
        )
        if self.enabled:
            self._store_feature(key, values)
        return values

    def assemble_features(
        self,
        dataset: PreparedDataset,
        indices: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        if not feature_names:
            raise ValueError("At least one feature must be requested")

        if self.enabled:
            components = [
                self.get_feature(dataset, name)[indices]
                for name in feature_names
            ]
        else:
            # This path mirrors the original split-by-split feature extraction.
            split_frame = (
                dataset.frame.iloc[indices].copy().reset_index(drop=True)
            )
            components = [
                _feature_array(
                    FEATURE_REGISTRY[name](split_frame),
                    name,
                    len(split_frame),
                )
                for name in feature_names
            ]

        return np.hstack(components)

    def _store_feature(
        self,
        key: tuple[DataKey, str],
        values: np.ndarray,
    ) -> None:
        if self.feature_cache_limit == 0:
            return
        if values.nbytes > self.feature_cache_limit:
            return

        while (
            self._features
            and self._feature_cache_bytes + values.nbytes
            > self.feature_cache_limit
        ):
            _, removed = self._features.popitem(last=False)
            self._feature_cache_bytes -= removed.nbytes

        self._features[key] = values
        self._feature_cache_bytes += values.nbytes

    def summary(self) -> str:
        """Return concise cache statistics for the completed sweep."""
        return (
            "Cache summary: "
            f"data {self.stats['dataset_hits']} hit(s)/"
            f"{self.stats['dataset_misses']} load(s), "
            f"splits {self.stats['split_hits']} hit(s)/"
            f"{self.stats['split_misses']} build(s), "
            f"features {self.stats['feature_hits']} hit(s)/"
            f"{self.stats['feature_misses']} calculation(s)"
        )
