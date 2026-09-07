import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pair_train import (
    fit_zscore,
    prepare_training_arrays,
    train_seed,
    training_weights,
)


class PairTrainingTests(unittest.TestCase):
    def test_zscore_uses_population_standard_deviation(self):
        values = np.array([[1.0, 10.0], [3.0, 14.0]])
        mean, scale = fit_zscore(values)
        np.testing.assert_array_equal(mean, [2.0, 12.0])
        np.testing.assert_array_equal(scale, [1.0, 2.0])

    def test_preparation_fits_only_supplied_training_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(
                path,
                inputs=np.array([[1.0, 2.0], [3.0, 4.0]]),
                labels=np.array([0, 1]),
            )
            arrays, labels, normalizer = prepare_training_arrays(path)
            np.testing.assert_allclose(arrays[0].mean(axis=0), 0.0)
            np.testing.assert_array_equal(labels, [0.0, 1.0])
            self.assertEqual(set(normalizer), {"inputs_mean", "inputs_scale"})

    def test_shared_member_inputs_use_one_pooled_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            images = np.zeros((2, 2, 12, 12), dtype=np.float32)
            images[:, 1] = 2.0
            scalars = np.zeros((2, 2, 47), dtype=np.float32)
            scalars[:, 1] = 4.0
            context = np.array([[10.0], [14.0]], dtype=np.float32)
            np.savez(
                path,
                images=images,
                scalars=scalars,
                context=context,
                labels=np.array([0, 1]),
            )
            arrays, _, normalizer = prepare_training_arrays(path)
            self.assertEqual(normalizer["images_mean"].shape, (12, 12))
            self.assertEqual(normalizer["scalars_mean"].shape, (47,))
            np.testing.assert_allclose(arrays[0][:, 0], -1.0)
            np.testing.assert_allclose(arrays[0][:, 1], 1.0)
            np.testing.assert_allclose(arrays[1][:, 0], -1.0)
            np.testing.assert_allclose(arrays[1][:, 1], 1.0)

    def test_generic_shared_member_inputs_use_one_pooled_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            inputs = np.zeros((2, 2, 3), dtype=np.float32)
            inputs[:, 1] = 2.0
            np.savez(path, inputs=inputs, labels=np.array([0, 1]))
            arrays, _, normalizer = prepare_training_arrays(path)
            self.assertEqual(normalizer["inputs_mean"].shape, (3,))
            np.testing.assert_allclose(arrays[0][:, 0], -1.0)
            np.testing.assert_allclose(arrays[0][:, 1], 1.0)

    def test_member_local_flat_inputs_are_reshaped_before_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(
                path,
                inputs=np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]),
                labels=np.array([0, 1]),
            )
            arrays, _, normalizer = prepare_training_arrays(
                path,
                member_width=2,
            )
            self.assertEqual(arrays[0].shape, (2, 2, 2))
            self.assertEqual(normalizer["inputs_mean"].shape, (2,))

    def test_supported_energy_weighting_formulas(self):
        labels = np.array([0, 1, 1, 1], dtype=np.float32)
        energies = np.array([np.nan, 25.0, 25.0, 30.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(path, member_min_truth_pt_gev=energies)
            inverse = training_weights(
                path,
                labels,
                target="binary",
                weighting="inverse_frequency_member_min",
            )
            np.testing.assert_allclose(inverse, [1.0, 0.75, 0.75, 1.5])

            power_path = Path(directory) / "power.npz"
            np.savez(
                power_path,
                member_min_truth_pt_gev=np.array([np.nan, 10.0, 20.0]),
            )
            power = training_weights(
                power_path,
                np.array([0, 1, 1]),
                target="binary",
                weighting="power_law_pminus1_member_min",
            )
            np.testing.assert_allclose(power, [1.0, 2.0 / 3.0, 4.0 / 3.0])

    def test_three_class_training_uses_cross_entropy(self):
        config = {
            "experiment_name": "three_class_smoke",
            "model": {
                "name": "shared_member_pair_mlp",
                "member_width": 2,
                "fusion": "symmetric",
                "output_classes": 3,
            },
            "loss": {"name": "cross_entropy", "pair_weighting": "equal"},
            "batch_size": 8,
            "epochs": 1,
        }
        rng = np.random.default_rng(9)
        inputs = rng.normal(size=(24, 2, 2)).astype(np.float32)
        targets = rng.integers(0, 3, size=24, dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pairs.npz"
            np.savez(path, inputs=inputs, three_class_labels=targets)
            run_dir = train_seed(config, path, root / "runs", 42)
            metrics = (run_dir / "training_metrics.csv").read_text()
            self.assertIn("training_cross_entropy", metrics)

    def test_training_rejects_inconsistent_target_or_optimizer(self):
        base = {
            "experiment_name": "bad_contract",
            "model": {"name": "pair_pt_mlp"},
            "loss": {"name": "bce"},
            "epochs": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_data = root / "pairs.npz"
            np.savez(
                pair_data,
                inputs=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
                labels=np.asarray([0.0, 1.0]),
            )
            wrong_target = dict(base, target="zero_one_or_two_tau_labelled_members")
            with self.assertRaisesRegex(ValueError, "target must be"):
                train_seed(wrong_target, pair_data, root / "target", 42)

            wrong_optimizer = dict(base, optimizer={"name": "sgd"})
            with self.assertRaisesRegex(ValueError, "only the Adam"):
                train_seed(wrong_optimizer, pair_data, root / "optimizer", 42)

    def test_high_resolution_zero_scale_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(
                path,
                images=np.zeros((2, 2, 12, 12), dtype=np.float32),
                scalars=np.zeros((2, 2, 47), dtype=np.float32),
                context=np.array([[1.0], [2.0]], dtype=np.float32),
                labels=np.array([0, 1]),
            )
            with self.assertRaisesRegex(ValueError, "zero-scale"):
                prepare_training_arrays(path)

    def test_high_resolution_factorial_accepts_zero_width_context(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez(
                path,
                images=rng.normal(size=(4, 2, 12, 12)).astype(np.float32),
                scalars=rng.normal(size=(4, 2, 46)).astype(np.float32),
                context=np.empty((4, 0), dtype=np.float32),
                labels=np.array([0, 1, 0, 1]),
            )
            arrays, labels, normalizer = prepare_training_arrays(
                path,
                high_resolution_widths=(46, 0),
            )
            self.assertEqual(arrays[2].shape, (4, 0))
            self.assertEqual(normalizer["context_mean"].shape, (0,))
            np.testing.assert_array_equal(labels, [0.0, 1.0, 0.0, 1.0])

    def test_small_training_run_writes_metrics_and_checkpoints(self):
        config = {
            "experiment_name": "smoke",
            "model": {"name": "pair_pt_mlp", "layers": [2, 8, 1]},
            "optimizer": {"learning_rate": 0.001},
            "batch_size": 16,
            "epochs": 2,
            "checkpoint_each_epoch": True,
        }
        rng = np.random.default_rng(42)
        inputs = rng.normal(size=(64, 2))
        labels = (inputs[:, 0] + inputs[:, 1] > 0).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_data = root / "pairs.npz"
            np.savez(pair_data, inputs=inputs, labels=labels)
            run_dir = train_seed(config, pair_data, root / "runs", 42)
            self.assertTrue((run_dir / "checkpoint_epoch_01.pt").is_file())
            self.assertTrue((run_dir / "checkpoint_epoch_02.pt").is_file())
            self.assertEqual(len((run_dir / "training_metrics.csv").read_text().splitlines()), 3)
            self.assertEqual(json.loads((run_dir / "config.json").read_text()), config)


if __name__ == "__main__":
    unittest.main()
