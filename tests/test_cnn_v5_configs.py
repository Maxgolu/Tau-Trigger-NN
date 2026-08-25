import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from losses import parse_loss
from model import TensorCNN, build_model


class CnnV5ConfigTests(unittest.TestCase):
    def test_c3_rawinput_configs_present_and_valid(self):
        # c3_rawinput isolates the input transform: identical to the v3 C3
        # inverse-frequency runs except transform "none", so the conv branch
        # consumes raw z-scored cells exactly like the flat MLP (M0).
        for seed in (42, 123, 456):
            d = PROJECT_ROOT / "configs" / f"cnn_v5_c3_rawinput_s{seed}"
            cfg = json.loads(
                (d / f"v5_c3_rawinput_s{seed}.json").read_text()
            )
            parse_classifier(cfg)
            parse_loss(cfg)
            branch = cfg["model"]["branches"][0]
            self.assertEqual(branch["transform"], "none")
            self.assertEqual(cfg["model"]["activation"], "leaky_relu")
            self.assertTrue(cfg["model"]["batchnorm"])
            model = build_model(cfg, 45, [("core_tensors", 0, 45)])
            self.assertIsInstance(model, TensorCNN)
            # Same C3 parameter count as the v3 sweep.
            self.assertEqual(
                sum(p.numel() for p in model.parameters()), 1285
            )
            out = model(torch.randn(5, 45))
            self.assertEqual(tuple(out.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
