import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_training import _build_constrained_model
from losses import parse_loss
from model import TensorCNN, build_model

LAYOUT = [("core_tensors", 0, 45), ("em2_best_3x3_fraction", 45, 1)]
DIM = 46
ARCHS = ("mix1x1", "conv2x2", "serial", "dual")
SEEDS = (42, 123, 456)


class CnnV8ConfigTests(unittest.TestCase):
    def test_or_sweep_configs_present_and_valid(self):
        found = 0
        for arch in ARCHS:
            for lname in ("bce", "invfreq"):
                d = PROJECT_ROOT / "configs" / f"cnn_v8_or_{arch}_{lname}"
                for seed in SEEDS:
                    cfg = json.loads(
                        (d / f"v8_or_{arch}_{lname}_s{seed}.json").read_text()
                    )
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    self.assertEqual(cfg["classifier"]["name"], "tob_nn_or")
                    self.assertEqual(
                        cfg["classifier"]["tob_budget"]["mode"],
                        "validation_search",
                    )
                    expected = ("bce" if lname == "bce"
                                else "energy_weighted_bce")
                    self.assertEqual(cfg["loss"]["name"], expected)
                    model = build_model(cfg, DIM, LAYOUT)
                    self.assertIsInstance(model, TensorCNN)
                    out = model(torch.randn(4, DIM))
                    self.assertEqual(tuple(out.shape), (4, 1))
                    found += 1
        self.assertEqual(found, 24)

    def test_finetune_configs_load_pretrained_dual_frac_weights(self):
        d = PROJECT_ROOT / "configs" / "cnn_v8_dualfrac_ft"
        for seed in SEEDS:
            cfg = json.loads(
                (d / f"v8_dualfrac_ft_s{seed}.json").read_text()
            )
            self.assertEqual(cfg["loss"]["name"], "constrained_trigger")
            self.assertEqual(
                cfg["initialization"]["mode"], "pretrained"
            )
            weights = PROJECT_ROOT / cfg["initialization"]["weights_path"]
            self.assertTrue(weights.is_file(), weights)
            # The factory model must accept the pretrained state dict exactly.
            model = _build_constrained_model(
                cfg, DIM, LAYOUT, torch.device("cpu")
            )
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)

    def test_constrained_model_rejects_transformed_branches(self):
        cfg = {
            "model": {
                "name": "tensor_cnn",
                "branches": [
                    {"feature": "core_tensors", "shape": [5, 3, 3],
                     "transform": "log1p",
                     "layers": [{"type": "conv", "kernel": 2,
                                 "out_channels": 8}]}
                ],
                "head": [16],
            }
        }
        with self.assertRaises(ValueError):
            _build_constrained_model(cfg, 45, [("core_tensors", 0, 45)],
                                     torch.device("cpu"))

    def test_constrained_model_defaults_to_legacy_mlp(self):
        # A config without a model block must still produce the legacy MLP,
        # so every earlier constrained configuration reproduces exactly.
        from model import DynamicMLP
        model = _build_constrained_model(
            {"hidden_layers": [32, 16]}, 46, LAYOUT, torch.device("cpu")
        )
        self.assertIsInstance(model, DynamicMLP)


if __name__ == "__main__":
    unittest.main()
