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

FEATURE_WIDTHS = {"core_tensors": 45, "em2_all_cells": 144,
                  "em2_best_3x3_fraction": 1}
EM2 = ("em2single", "em2ladder", "em2stride")
CT = ("mix1x1", "conv2x2", "dual")
SEEDS = (42, 123, 456)


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV13ConfigTests(unittest.TestCase):
    def test_twentyseven_or_bce_configs_valid(self):
        found = 0
        for seed in SEEDS:
            d = PROJECT_ROOT / "configs" / f"cnn_v13_or_bce_s{seed}"
            for em2 in EM2:
                for ct in CT:
                    cfg = json.loads(
                        (d / f"v13_or_{em2}_{ct}_s{seed}.json").read_text()
                    )
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    self.assertEqual(cfg["loss"], {"name": "bce"})
                    self.assertEqual(cfg["classifier"]["name"], "tob_nn_or")
                    self.assertEqual(
                        cfg["classifier"]["tob_budget"]["mode"],
                        "validation_search",
                    )
                    self.assertIn("em2_best_3x3_fraction",
                                  cfg["features_to_use"])
                    layout, dim = _layout_for(cfg["features_to_use"])
                    model = build_model(cfg, dim, layout)
                    self.assertIsInstance(model, TensorCNN)
                    out = model(torch.randn(4, dim))
                    self.assertEqual(tuple(out.shape), (4, 1))
                    found += 1
        self.assertEqual(found, 27)


if __name__ == "__main__":
    unittest.main()
