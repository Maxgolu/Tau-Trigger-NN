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
for _layer in ("em0", "em1", "em2_3x3", "em3", "had"):
    for _kind in ("sparsity", "sum", "dominance"):
        FEATURE_WIDTHS[f"{_layer}_{_kind}"] = 1

EM2_FLATTEN = {"em2single": 100, "em2ladder": 72, "em2stride": 72}
SCALARS = {"ctflat": 45 + 1, "sum": 15 + 1}  # passthrough scalars + fraction
SEEDS = (42, 123, 456)


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV14ConfigTests(unittest.TestCase):
    def test_eighteen_em2_only_configs_valid(self):
        found = 0
        for em2, em2_dim in EM2_FLATTEN.items():
            for fname, scalar_dim in SCALARS.items():
                d = PROJECT_ROOT / "configs" / f"cnn_v14_{em2}_{fname}"
                for seed in SEEDS:
                    cfg = json.loads(
                        (d / f"v14_{em2}_{fname}_s{seed}.json").read_text()
                    )
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    # exactly one conv branch, and it is the EM2 image
                    branches = cfg["model"]["branches"]
                    self.assertEqual(len(branches), 1)
                    self.assertEqual(branches[0]["feature"], "em2_all_cells")
                    layout, dim = _layout_for(cfg["features_to_use"])
                    model = build_model(cfg, dim, layout)
                    self.assertIsInstance(model, TensorCNN)
                    first_linear = next(
                        m for m in model.head
                        if isinstance(m, torch.nn.Linear)
                    )
                    # EM2 flatten + passthrough scalars (CT cells or
                    # summaries, plus the fraction)
                    self.assertEqual(
                        first_linear.in_features, em2_dim + scalar_dim
                    )
                    out = model(torch.randn(4, dim))
                    self.assertEqual(tuple(out.shape), (4, 1))
                    found += 1
        self.assertEqual(found, 18)


if __name__ == "__main__":
    unittest.main()
