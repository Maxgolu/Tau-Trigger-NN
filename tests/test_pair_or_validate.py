import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_or_validate import (
    _paired_counts,
    _select_candidate,
    load_or_configs,
    validate_or_family,
)


class PairOrValidationTests(unittest.TestCase):
    def _inputs(self):
        rows = []
        energy_rows = []
        event_id = 0
        for checkpoint, bce in (("checkpoint_epoch_1", 0.4), ("checkpoint_epoch_2", 0.3)):
            event_id = 0
            for fold in ("A", "B"):
                for index in range(8):
                    rows.append(
                        {
                            "global_event_id": event_id,
                            "sample": "background",
                            "pair_observable": True,
                            "pair_eligible": False,
                            "fold": fold,
                            "model_seed": 42,
                            "checkpoint_id": checkpoint,
                            "event_score": (
                                float(7 - index) if checkpoint.endswith("_1") else -10.0
                            ),
                            "baseline_event_score": float(index),
                            "validation_bce": bce,
                        }
                    )
                    event_id += 1
                rows.append(
                    {
                        "global_event_id": event_id,
                        "sample": "background",
                        "pair_observable": False,
                        "pair_eligible": False,
                        "fold": fold,
                        "model_seed": 42,
                        "checkpoint_id": checkpoint,
                        "event_score": -np.inf,
                        "baseline_event_score": -np.inf,
                        "validation_bce": bce,
                    }
                )
                event_id += 1
                for index, energy in enumerate((25.0, 35.0, 45.0, 55.0)):
                    rows.append(
                        {
                            "global_event_id": event_id,
                            "sample": "signal",
                            "pair_observable": True,
                            "pair_eligible": True,
                            "fold": fold,
                            "model_seed": 42,
                            "checkpoint_id": checkpoint,
                            "event_score": (
                                1.0 if index % 2 == 0 else 9.0
                            )
                            if checkpoint.endswith("_1")
                            else -10.0,
                            "baseline_event_score": 8.0 if index % 2 == 0 else 1.0,
                            "validation_bce": bce,
                        }
                    )
                    if checkpoint == "checkpoint_epoch_1":
                        energy_rows.append(
                            {
                                "global_event_id": event_id,
                                "member_min_truth_pt_gev": energy,
                            }
                        )
                    event_id += 1
        return pd.DataFrame(rows), pd.DataFrame(energy_rows)

    def test_complete_crossfit_selection_calibration_overlap_and_regions(self):
        predictions, energy = self._inputs()
        report = validate_or_family(
            predictions,
            energy,
            pt_budget_grid=(0.0, 0.125, 0.25),
            target_fpr=0.25,
            objective_edges_gev=(20.0, 40.0, 60.0),
            protected_edges_gev=(20.0, 40.0, 60.0),
            allowed_protected_deficit=0.0,
            objective_tolerance=0.0,
        )
        seed = report["seeds"]["42"]
        self.assertEqual(seed["candidate_count"], 6)
        self.assertEqual(seed["selection"]["checkpoint_id"], "checkpoint_epoch_1")
        # Equal objectives use the frozen lower-budget tie-break.
        self.assertEqual(seed["selection"]["pt_budget"], 0.0)
        full = seed["full_validation"]
        self.assertLessEqual(full["joint_thresholds"]["union_fpr"], 0.25)
        self.assertEqual(full["comparisons"]["background"]["events"], 18)
        self.assertEqual(full["comparisons"]["inclusive_signal"]["events"], 8)
        self.assertEqual(full["branch_overlap"]["background"]["union"], 4)
        self.assertEqual(len(full["conditional_energy_regions"]), 3)
        self.assertFalse(full["promotion_allowed_from_this_validation_report"])

    def test_no_pair_events_are_in_background_denominator_and_never_pass(self):
        predictions, energy = self._inputs()
        report = validate_or_family(
            predictions.loc[predictions["checkpoint_id"].eq("checkpoint_epoch_1")],
            energy,
            pt_budget_grid=(0.125,),
            target_fpr=0.25,
            objective_edges_gev=(20.0, 40.0, 60.0),
            protected_edges_gev=(20.0, 40.0, 60.0),
        )
        full = report["seeds"]["42"]["full_validation"]
        self.assertEqual(full["joint_thresholds"]["background_events"], 18)
        self.assertEqual(full["branch_overlap"]["background"]["neither"], 14)

    def test_energy_profile_must_cover_exactly_eligible_signal_events(self):
        predictions, energy = self._inputs()
        with self.assertRaisesRegex(ValueError, "exactly the pair-eligible"):
            validate_or_family(
                predictions,
                energy.iloc[:-1],
                pt_budget_grid=(0.125,),
                target_fpr=0.25,
                objective_edges_gev=(20.0, 40.0, 60.0),
                protected_edges_gev=(20.0, 40.0, 60.0),
            )

    def test_paired_counts_report_overlap_and_paired_uncertainty(self):
        result = _paired_counts(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(result["both_accepted"], 1)
        self.assertEqual(result["model_only"], 1)
        self.assertEqual(result["baseline_only"], 1)
        self.assertEqual(result["efficiency_delta"], 0.0)
        self.assertGreater(result["paired_standard_error"], 0.0)

    def test_candidate_selection_enforces_protection_then_frozen_ties(self):
        def record(epoch, objective, protected, inclusive, budget=0.001):
            return {
                "epoch": epoch,
                "checkpoint_id": f"checkpoint_epoch_{epoch}",
                "pt_budget": budget,
                "equal_window_objective": objective,
                "worst_protected_window_delta": protected,
                "protection_feasible": protected >= -0.005,
                "validation_bce": 0.2,
                "pooled_comparisons": {
                    "inclusive_signal": {"model_accepted": inclusive},
                    "pair_eligible_signal": {"model_accepted": inclusive},
                    "background": {"model_accepted": 2},
                },
            }

        records = [
            record(1, 0.20, -0.02, 20),
            record(2, 0.10, 0.00, 10),
            record(3, 0.101, 0.00, 11),
        ]
        selected, feasible = _select_candidate(records, objective_tolerance=0.002)
        self.assertTrue(feasible)
        self.assertEqual(selected["epoch"], 3)

    def test_candidate_selection_uses_bce_tolerance_then_earlier_epoch(self):
        def record(epoch, bce):
            return {
                "epoch": epoch,
                "checkpoint_id": f"checkpoint_epoch_{epoch}",
                "pt_budget": 0.001,
                "equal_window_objective": 0.1,
                "worst_protected_window_delta": 0.0,
                "protection_feasible": True,
                "validation_bce": bce,
                "pooled_comparisons": {
                    "inclusive_signal": {"model_accepted": 10},
                    "pair_eligible_signal": {"model_accepted": 10},
                    "background": {"model_accepted": 2},
                },
            }

        selected, _ = _select_candidate(
            [record(2, 0.2 + 5e-13), record(5, 0.2)],
            objective_tolerance=0.002,
        )
        self.assertEqual(selected["epoch"], 2)

    def test_no_feasible_candidate_chooses_least_protected_deficit(self):
        records = []
        for epoch, objective, protected in ((1, 0.2, -0.02), (2, 0.1, -0.01)):
            records.append(
                {
                    "epoch": epoch,
                    "checkpoint_id": f"checkpoint_epoch_{epoch}",
                    "pt_budget": 0.001,
                    "equal_window_objective": objective,
                    "worst_protected_window_delta": protected,
                    "protection_feasible": False,
                    "validation_bce": 0.2,
                    "pooled_comparisons": {
                        "inclusive_signal": {"model_accepted": 10},
                        "pair_eligible_signal": {"model_accepted": 10},
                        "background": {"model_accepted": 2},
                    },
                }
            )
        selected, feasible = _select_candidate(records, objective_tolerance=0.002)
        self.assertFalse(feasible)
        self.assertEqual(selected["epoch"], 2)

    def test_per_seed_configs_share_exact_validation_contract(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "pair_raw_cell_or"
        configs, settings = load_or_configs(config_dir)
        self.assertEqual({config["seed"] for config in configs}, {42, 123, 456})
        self.assertEqual(settings["target_event_fpr"], 0.005)
        self.assertEqual(len(settings["pt_budget_grid"]), 11)

    def test_mismatched_per_seed_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {
                "experiment_name": "example",
                "seed": 42,
                "validation": {"target_event_fpr": 0.005},
            }
            Path(directory, "a.json").write_text(json.dumps(base), encoding="utf-8")
            changed = dict(base)
            changed["seed"] = 123
            changed["validation"] = {"target_event_fpr": 0.01}
            Path(directory, "b.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be identical"):
                load_or_configs(directory)


if __name__ == "__main__":
    unittest.main()
