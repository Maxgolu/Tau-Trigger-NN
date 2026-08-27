import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_external import apply_branch_transforms
from prepare_external_data import find_combined_pair, merge_cases


def _write_case(case_dir, prefix, event_numbers, duplicate_suffix=""):
    """Create a tiny synthetic combined CSV/NPZ pair for one case."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    n = len(event_numbers)
    rng = np.random.default_rng(int(event_numbers[0]) + n)
    npz = {
        "X_tensors": rng.random((n, 6, 5, 3, 3)).astype(np.float32),
        "X_em2_tensors": rng.random((n, 6, 12, 12)).astype(np.float32),
        "X_feats": rng.random((n, 6, 4)),
        "y_tob": rng.integers(0, 2, (n, 6)),
        "y_event": np.full(n, 2),
        "event_nums": np.asarray(event_numbers, dtype=np.int64),
    }
    rows = []
    for event in event_numbers:
        for tob in range(2):
            rows.append({
                "eventNumber": event, "tob_index": tob, "signal": tob % 2,
                "truth_pt": 30000.0, "prongs": 1, "tob_pt": 20000.0,
                "tob_eta": 0.1, "tob_phi": 0.2, "tob_bdt": 0.0,
                "Type": "Signal" if prefix == "signal" else "BKG",
            })
    df = pd.DataFrame(rows)
    df.to_csv(case_dir / f"{prefix}_combined{duplicate_suffix}.csv",
              index=False)
    np.savez(case_dir / f"{prefix}_combined{duplicate_suffix}.npz", **npz)


class PrepareExternalDataTests(unittest.TestCase):
    def test_merge_offsets_colliding_event_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Both cases reuse event numbers 1..5 - the known collision.
            _write_case(Path(tmp) / "case_a", "signal", [1, 2, 3, 4, 5])
            _write_case(Path(tmp) / "case_b", "signal", [1, 2, 3],
                        duplicate_suffix=" (1)")
            df, npz = merge_cases(
                [Path(tmp) / "case_a", Path(tmp) / "case_b"], "signal"
            )
            events = npz["event_nums"]
            self.assertEqual(len(events), 8)
            self.assertEqual(len(np.unique(events)), 8)
            # CSV event numbers were shifted consistently with the NPZ.
            self.assertTrue(set(df["eventNumber"]).issubset(set(events)))
            # Provenance column present with both case names.
            self.assertEqual(set(df["case"]), {"case_a", "case_b"})
            # Arrays concatenated in order.
            self.assertEqual(npz["X_tensors"].shape, (8, 6, 5, 3, 3))
            self.assertEqual(npz["X_em2_tensors"].shape, (8, 6, 12, 12))

    def test_find_combined_pair_accepts_duplicate_download_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_case(Path(tmp) / "case_c", "bkg", [7, 8],
                        duplicate_suffix=" (2)")
            csv_path, npz_path = find_combined_pair(
                Path(tmp) / "case_c", "bkg"
            )
            self.assertTrue(csv_path.name.endswith(".csv"))
            self.assertTrue(npz_path.name.endswith(".npz"))

    def test_find_combined_pair_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_case(Path(tmp) / "case_d", "bkg", [1, 2])
            _write_case(Path(tmp) / "case_d", "bkg", [3, 4],
                        duplicate_suffix=" (1)")
            with self.assertRaises(FileNotFoundError):
                find_combined_pair(Path(tmp) / "case_d", "bkg")


class ApplyBranchTransformsTests(unittest.TestCase):
    def test_log1p_applied_only_to_declared_branch(self):
        X = np.array([[3.0, -8.0, 5.0]], dtype=np.float64)
        config = {"model": {"name": "tensor_cnn", "branches": [
            {"feature": "a", "transform": "log1p", "layers": []}
        ]}}
        layout = [("a", 0, 2), ("b", 2, 1)]
        out = apply_branch_transforms(X.copy(), config, layout)
        self.assertAlmostEqual(out[0, 0], np.log1p(3.0))
        self.assertAlmostEqual(out[0, 1], -np.log1p(8.0))
        self.assertAlmostEqual(out[0, 2], 5.0)  # scalar untouched

    def test_transform_none_is_identity(self):
        X = np.array([[3.0, -8.0]], dtype=np.float64)
        config = {"model": {"name": "tensor_cnn", "branches": [
            {"feature": "a", "transform": "none", "layers": []}
        ]}}
        out = apply_branch_transforms(X.copy(), config, [("a", 0, 2)])
        self.assertTrue(np.allclose(out, X))


if __name__ == "__main__":
    unittest.main()
