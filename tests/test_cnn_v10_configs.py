import json
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_training import (
    _build_constrained_model,
    _resolve_initial_weights,
)
from losses import parse_loss
from model import TensorCNN, build_model

FEATURE_WIDTHS = {"core_tensors": 45, "em2_best_3x3_fraction": 1,
                  "em2_all_cells": 144}
FT_EXPERIMENTS = (
    "ft_mix1x1_frac", "ft_conv2x2_frac", "ft_dual_frac",
    "ft_em2single_mix1x1", "ft_em2single_conv2x2", "ft_em2single_dual",
)
SEEDS = (42, 123, 456)


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV10ConfigTests(unittest.TestCase):
    def test_finetune_configs_resolve_and_load(self):
        found = 0
        for exp in FT_EXPERIMENTS:
            d = PROJECT_ROOT / "configs" / f"cnn_v10_{exp}"
            for seed in SEEDS:
                cfg = json.loads(
                    (d / f"v10_{exp}_s{seed}.json").read_text()
                )
                parse_classifier(cfg)
                loss = cfg["loss"]
                self.assertEqual(loss["name"], "constrained_trigger")
                # The adapted loss: split high-energy constraints and a
                # widened objective. Parallel lists must stay consistent.
                self.assertEqual(
                    loss["constraint_regions_gev"],
                    [[25, 32], [32, 40], [40, 60], [60, 100], [100, 120]],
                )
                for key in ("allowed_deficits",
                            "minimum_region_advantages",
                            "reference_model_allowed_deficits"):
                    self.assertEqual(len(loss[key]),
                                     len(loss["constraint_regions_gev"]))
                self.assertEqual(len(loss["objective_region_weights"]),
                                 len(loss["objective_regions_gev"]))
                self.assertAlmostEqual(
                    sum(loss["objective_region_weights"]), 1.0
                )
                # Resolver accepts the config (model block equality with the
                # source run) and the state dict loads into the factory model.
                weights = PROJECT_ROOT / cfg["initialization"]["weights_path"]
                self.assertTrue(weights.is_file(), weights)
                resolved = _resolve_initial_weights(cfg, PROJECT_ROOT)
                self.assertEqual(resolved, weights.resolve())
                layout, dim = _layout_for(cfg["features_to_use"])
                model = _build_constrained_model(
                    cfg, dim, layout, torch.device("cpu")
                )
                state = torch.load(weights, map_location="cpu")
                model.load_state_dict(state)
                found += 1
        self.assertEqual(found, 18)

    def test_dual_em2_frac_configs_valid(self):
        d = PROJECT_ROOT / "configs" / "cnn_v10_dual_em2_frac"
        for seed in SEEDS:
            cfg = json.loads(
                (d / f"v10_dual_em2_frac_s{seed}.json").read_text()
            )
            parse_classifier(cfg)
            parse_loss(cfg)
            self.assertEqual(
                cfg["features_to_use"],
                ["core_tensors", "em2_all_cells", "em2_best_3x3_fraction"],
            )
            layout, dim = _layout_for(cfg["features_to_use"])
            model = build_model(cfg, dim, layout)
            self.assertIsInstance(model, TensorCNN)
            first_linear = next(
                m for m in model.head if isinstance(m, torch.nn.Linear)
            )
            # dual CT (108 + 48) + em2single (100) + fraction scalar (1)
            self.assertEqual(first_linear.in_features, 108 + 48 + 100 + 1)
            out = model(torch.randn(4, dim))
            self.assertEqual(tuple(out.shape), (4, 1))


if __name__ == "__main__":
    unittest.main()
