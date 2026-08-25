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

# expected first-head-linear input width per experiment
EXPECTED_HEAD_IN = {
    "mix1x1_raw": 12 * 3 * 3,             # 108
    "conv2x2_raw": 12 * 2 * 2,            # 48
    "dualbranch_raw": 12 * 3 * 3 + 12 * 2 * 2,  # 156, parallel branches
    "serial_raw": 12 * 2 * 2,             # 48, 1x1 stacked into 2x2
}


class CnnV6ConfigTests(unittest.TestCase):
    def test_twelve_configs_present_and_valid(self):
        found = 0
        for exp, head_in in EXPECTED_HEAD_IN.items():
            for seed in (42, 123, 456):
                d = PROJECT_ROOT / "configs" / f"cnn_v6_{exp}_s{seed}"
                cfg = json.loads(
                    (d / f"v6_{exp}_s{seed}.json").read_text()
                )
                parse_classifier(cfg)
                parse_loss(cfg)
                for branch in cfg["model"]["branches"]:
                    # the whole v6 sweep runs on raw z-scored cells
                    self.assertEqual(branch["transform"], "none")
                self.assertEqual(cfg["model"]["head"], [32, 16])
                self.assertEqual(cfg["model"]["activation"], "leaky_relu")
                self.assertTrue(cfg["model"]["batchnorm"])
                self.assertEqual(
                    cfg["loss"]["weighting"]["profile"], "inverse_frequency"
                )
                model = build_model(cfg, 45, [("core_tensors", 0, 45)])
                self.assertIsInstance(model, TensorCNN)
                first_linear = next(
                    m for m in model.head if isinstance(m, torch.nn.Linear)
                )
                self.assertEqual(first_linear.in_features, head_in)
                out = model(torch.randn(5, 45))
                self.assertEqual(tuple(out.shape), (5, 1))
                found += 1
        self.assertEqual(found, 12)


if __name__ == "__main__":
    unittest.main()
