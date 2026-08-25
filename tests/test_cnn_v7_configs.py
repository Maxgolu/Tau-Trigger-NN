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

FEATURE_WIDTHS = {"core_tensors": 45, "core_physics": 4,
                  "em2_best_3x3_fraction": 1}
for _layer in ("em0", "em1", "em2_3x3", "em3", "had"):
    for _kind in ("sparsity", "sum", "dominance"):
        FEATURE_WIDTHS[f"{_layer}_{_kind}"] = 1

ARCHS = ("mix1x1", "conv2x2", "serial", "dual")
FEATS = ("ctdirect", "cp", "sum", "frac")
SEEDS = (42, 123, 456)


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV7ConfigTests(unittest.TestCase):
    def test_fortyeight_configs_present_and_valid(self):
        by_id = {}
        for d in sorted((PROJECT_ROOT / "configs").glob("cnn_v7_gpu*")):
            for p in d.glob("*.json"):
                cfg = json.loads(p.read_text())
                by_id[cfg["run_id"]] = cfg
        # every (arch, feat, seed) combination exists exactly once
        for arch in ARCHS:
            for feat in FEATS:
                for seed in SEEDS:
                    run_id = f"v7_{arch}_{feat}_s{seed}"
                    self.assertIn(run_id, by_id, run_id)
                    cfg = by_id[run_id]
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    for branch in cfg["model"]["branches"]:
                        self.assertEqual(branch["transform"], "none")
                    if feat == "ctdirect":
                        self.assertTrue(
                            cfg["model"]["branches"][0].get("include_raw")
                        )
                        self.assertEqual(
                            cfg["features_to_use"], ["core_tensors"]
                        )
                    layout, dim = _layout_for(cfg["features_to_use"])
                    model = build_model(cfg, dim, layout)
                    self.assertIsInstance(model, TensorCNN)
                    out = model(torch.randn(4, dim))
                    self.assertEqual(tuple(out.shape), (4, 1))
        self.assertEqual(len(by_id), 48)


if __name__ == "__main__":
    unittest.main()
