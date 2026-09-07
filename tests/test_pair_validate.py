import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_train import train_seed
from pair_validate import configured_runs, validate_config_family


class PairValidationRunnerTests(unittest.TestCase):
    def test_per_seed_configs_resolve_to_expected_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            for seed in (42, 123):
                (configs / f"model_s{seed}.json").write_text(
                    json.dumps({"experiment_name": "model", "seed": seed}),
                    encoding="utf-8",
                )
            entries = configured_runs(configs, root / "experiments")
            self.assertEqual([entry[1]["seed"] for entry in entries], [123, 42])
            self.assertEqual(entries[0][2], root / "experiments/model/seed_123")

    def test_empty_or_multiseed_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                configured_runs(root, root / "runs")
            (root / "bad.json").write_text(
                json.dumps({"experiment_name": "bad", "seeds": [42, 123]}),
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                configured_runs(root, root / "runs")

    def test_small_multiseed_validation_workflow(self):
        """Exercise training, prediction, aggregation, and selection together."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            experiment_root = root / "experiments"

            training_inputs = np.asarray(
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                dtype=np.float32,
            )
            training_labels = np.asarray([0, 0, 1, 1], dtype=np.float32)
            training_path = root / "training.npz"
            np.savez(
                training_path,
                inputs=training_inputs,
                labels=training_labels,
            )

            for seed in (42, 123):
                config = {
                    "experiment_name": "integration",
                    "seed": seed,
                    "representation": "pair_pt",
                    "model": {"name": "pair_pt_mlp"},
                    "loss": {"name": "bce", "pair_weighting": "equal"},
                    "optimizer": {"learning_rate": 0.001},
                    "batch_size": 4,
                    "epochs": 1,
                    "checkpoint_each_epoch": True,
                }
                config_path = config_dir / f"integration_s{seed}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                train_seed(
                    config,
                    training_path,
                    experiment_root / "integration",
                    seed,
                )

            event_ids = np.arange(100, 140, dtype=np.int64)
            signal = np.arange(len(event_ids)) % 2 == 0
            inputs = np.column_stack(
                (
                    np.where(signal, 3.0, 0.0),
                    np.where(signal, 2.0, 0.0),
                )
            ).astype(np.float32)
            validation_path = root / "validation.npz"
            np.savez(
                validation_path,
                inputs=inputs,
                labels=signal.astype(np.float32),
                three_class_labels=np.where(signal, 2, 0).astype(np.int64),
                pair_event_ids=event_ids,
                pair_indices=np.zeros(len(event_ids), dtype=np.int64),
                pair_tob_indices=np.tile([0, 1], (len(event_ids), 1)),
                pair_selected_ranks=np.tile([0, 1], (len(event_ids), 1)),
                pair_pt=inputs[:, 1].astype(np.float64),
                event_ids=event_ids,
                event_samples=signal.astype(np.uint8),
                event_pair_observable=np.ones(len(event_ids), dtype=np.uint8),
                event_pair_eligible=signal.astype(np.uint8),
                baseline_event_scores=inputs[:, 1].astype(np.float64),
            )

            output_dir = root / "validation_output"
            report = validate_config_family(
                config_dir,
                validation_path,
                experiment_root,
                output_dir,
            )

            self.assertEqual(set(report["seeds"]), {"42", "123"})
            self.assertTrue((output_dir / "checkpoint_summary.csv").is_file())
            self.assertTrue((output_dir / "selection_report.json").is_file())
            for seed in (42, 123):
                checkpoint_dir = output_dir / f"seed_{seed}" / "checkpoint_epoch_01"
                self.assertTrue((checkpoint_dir / "pair_predictions.csv.gz").is_file())
                self.assertTrue((checkpoint_dir / "event_predictions.csv.gz").is_file())


if __name__ == "__main__":
    unittest.main()
