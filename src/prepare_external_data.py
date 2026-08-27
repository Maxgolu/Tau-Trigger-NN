"""Merge several production cases into one external evaluation dataset.

Each case directory holds one ``*_combined.csv`` / ``*_combined.npz`` pair
(Windows duplicate-download names such as ``bkg_combined (1).csv`` are
accepted). Event numbers are only unique inside one case, so every case
receives a disjoint offset before merging, and a ``case`` column records the
provenance of every object so results can be broken down per case offline.

The output directory has the exact layout the training pipeline expects
(``Signal/signal_combined.*`` and ``Background/bkg_combined.*``), so it can
be used directly as an alternative ``--data_dir``. The original data
directories are never touched.

Example:
    python src/prepare_external_data.py \
        --signal_cases signal_new/case_603276 signal_new/case_603422 \
                       signal_new/case_801002 \
        --background_cases bkgr_new/case_801165 bkgr_new/case_801166 \
                           bkgr_new/case_801167 bkgr_new/case_801168 \
        --output_dir data_new
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NPZ_KEYS = ["X_tensors", "X_em2_tensors", "X_feats",
            "y_tob", "y_event", "event_nums"]


def find_combined_pair(case_dir, prefix):
    """Locate the single combined CSV/NPZ pair inside a case directory."""
    case_path = Path(case_dir)
    csvs = sorted(case_path.glob(f"{prefix}_combined*.csv"))
    npzs = sorted(case_path.glob(f"{prefix}_combined*.npz"))
    if len(csvs) != 1 or len(npzs) != 1:
        raise FileNotFoundError(
            f"{case_dir}: expected exactly one {prefix}_combined CSV and NPZ, "
            f"found {len(csvs)} CSV and {len(npzs)} NPZ"
        )
    return csvs[0], npzs[0]


def merge_cases(case_dirs, prefix):
    """Concatenate cases with disjoint event-number ranges.

    Returns the merged CSV frame (with a ``case`` provenance column) and a
    dict of merged NPZ arrays.
    """
    frames = []
    arrays = {key: [] for key in NPZ_KEYS}
    base = 0
    for case_dir in case_dirs:
        csv_path, npz_path = find_combined_pair(case_dir, prefix)
        df = pd.read_csv(csv_path)
        npz = np.load(npz_path)
        events = npz["event_nums"].astype(np.int64)
        if events.min() < 0:
            raise ValueError(f"{npz_path}: negative event numbers")
        # Event numbers repeat between cases (per-campaign counters), so each
        # case is shifted onto its own disjoint range before merging.
        offset = base - int(events.min())
        shifted = events + offset
        df = df.copy()
        df["eventNumber"] = df["eventNumber"].astype(np.int64) + offset
        df["case"] = Path(case_dir).name
        missing = set(df["eventNumber"]) - set(shifted.tolist())
        if missing:
            raise ValueError(
                f"{csv_path}: {len(missing)} CSV events missing from the NPZ"
            )
        frames.append(df)
        for key in NPZ_KEYS:
            arrays[key].append(shifted if key == "event_nums" else npz[key])
        base = int(shifted.max()) + 1
        print(f"  {Path(case_dir).name}: {len(events)} events, "
              f"{len(df)} objects, offset {offset}")
    merged_df = pd.concat(frames, ignore_index=True)
    merged_npz = {key: np.concatenate(arrays[key], axis=0)
                  for key in NPZ_KEYS}
    if len(np.unique(merged_npz["event_nums"])) != len(
            merged_npz["event_nums"]):
        raise ValueError("Merged event numbers are not unique")
    return merged_df, merged_npz


def write_class(output_dir, subdir, prefix, merged_df, merged_npz):
    out = Path(output_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_combined.csv"
    npz_path = out / f"{prefix}_combined.npz"
    merged_df.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **merged_npz)
    print(f"Wrote {csv_path} ({len(merged_df)} objects) and {npz_path} "
          f"({len(merged_npz['event_nums'])} events)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge production cases into one external data_dir"
    )
    parser.add_argument("--signal_cases", nargs="+", required=True)
    parser.add_argument("--background_cases", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Merging signal cases...")
    sig_df, sig_npz = merge_cases(args.signal_cases, "signal")
    print("Merging background cases...")
    bkg_df, bkg_npz = merge_cases(args.background_cases, "bkg")
    write_class(args.output_dir, "Signal", "signal", sig_df, sig_npz)
    write_class(args.output_dir, "Background", "bkg", bkg_df, bkg_npz)


if __name__ == "__main__":
    main()
