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

# Column widths of every feature used by the v4 sweep, matching the
# feature registry output shapes.
FEATURE_WIDTHS = {
    "core_tensors": 45,
    "core_physics": 4,
    "em2_all_cells": 144,
}
# The fifteen layer summaries are scalar features.
for _layer in ("em0", "em1", "em2_3x3", "em3", "had"):
    for _kind in ("sparsity", "sum", "dominance"):
        FEATURE_WIDTHS[f"{_layer}_{_kind}"] = 1

EXPECTED_EXPERIMENTS = [
    "mlplog", "c3h", "c3cp", "c3sum",
    "m11", "m11cp", "m11sum", "m11deep", "c3em2",
]


def _layout_for(features):
    layout = []
    offset = 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV4ConfigTests(unittest.TestCase):
    def test_twenty_seven_configs_present_and_valid(self):
        found = 0
        for exp in EXPECTED_EXPERIMENTS:
            d = PROJECT_ROOT / "configs" / f"cnn_v4_{exp}"
            self.assertTrue(d.is_dir(), d)
            files = sorted(p.name for p in d.glob("*.json"))
            self.assertEqual(
                files,
                sorted(f"v4_{exp}_s{seed}.json" for seed in (42, 123, 456)),
            )
            for name in files:
                cfg = json.loads((d / name).read_text())
                parse_classifier(cfg)
                parse_loss(cfg)
                self.assertEqual(cfg["classifier"]["name"], "nn_only")
                self.assertEqual(
                    cfg["loss"]["weighting"]["profile"], "inverse_frequency"
                )
                layout, dim = _layout_for(cfg["features_to_use"])
                model = build_model(cfg, dim, layout)
                self.assertIsInstance(model, TensorCNN)
                out = model(torch.randn(5, dim))
                self.assertEqual(tuple(out.shape), (5, 1))
                logits = model.forward_logits(torch.randn(5, dim))
                self.assertTrue(torch.isfinite(logits).all())
                found += 1
        self.assertEqual(found, 27)

    def test_mlplog_control_matches_flat_mlp_architecture(self):
        # The control must reproduce the M0 architecture (45 -> 32 -> 16 -> 1,
        # ReLU, no BatchNorm) so that the only variable against M0 is the
        # log1p input transform.
        cfg = json.loads(
            (PROJECT_ROOT / "configs" / "cnn_v4_mlplog"
             / "v4_mlplog_s42.json").read_text()
        )
        self.assertEqual(cfg["model"]["activation"], "relu")
        self.assertFalse(cfg["model"]["batchnorm"])
        self.assertEqual(cfg["model"]["branches"][0]["layers"], [])
        self.assertEqual(cfg["model"]["head"], [32, 16])
        layout, dim = _layout_for(cfg["features_to_use"])
        model = build_model(cfg, dim, layout)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2017)

    def test_hybrid_head_sees_conv_output_plus_raw_input(self):
        cfg = json.loads(
            (PROJECT_ROOT / "configs" / "cnn_v4_c3h"
             / "v4_c3h_s42.json").read_text()
        )
        self.assertTrue(cfg["model"]["branches"][0]["include_raw"])
        layout, dim = _layout_for(cfg["features_to_use"])
        model = build_model(cfg, dim, layout)
        first_linear = next(
            m for m in model.head if isinstance(m, torch.nn.Linear)
        )
        # C3 conv flatten (12*2*2=48) + raw core_tensors (45)
        self.assertEqual(first_linear.in_features, 48 + 45)


if __name__ == "__main__":
    unittest.main()
