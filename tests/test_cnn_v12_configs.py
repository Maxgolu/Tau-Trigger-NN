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

FEATURE_WIDTHS = {"core_tensors": 45, "em2_all_cells": 144,
                  "em2_best_3x3_fraction": 1}
EXPERIMENTS = (
    "ft_em2single_mix1x1_frac", "ft_em2single_conv2x2_frac",
    "ft_em2ladder_mix1x1_frac", "ft_em2ladder_conv2x2_frac",
    "ft_em2ladder_dual_frac",
    "ft_em2stride_mix1x1_frac", "ft_em2stride_conv2x2_frac",
    "ft_em2stride_dual_frac",
    "ft_em2single_dual_fracbce",
)
SEEDS = (42, 123, 456)


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class CnnV12ConfigTests(unittest.TestCase):
    def test_twentyseven_finetune_configs_resolve_and_load(self):
        found = 0
        for exp in EXPERIMENTS:
            d = PROJECT_ROOT / "configs" / f"cnn_v12_{exp}"
            for seed in SEEDS:
                cfg = json.loads(
                    (d / f"v12_{exp}_s{seed}.json").read_text()
                )
                parse_classifier(cfg)
                loss = cfg["loss"]
                self.assertEqual(loss["name"], "constrained_trigger")
                self.assertEqual(
                    loss["constraint_regions_gev"],
                    [[25, 32], [32, 40], [40, 60], [60, 100], [100, 120]],
                )
                weights = (PROJECT_ROOT
                           / cfg["initialization"]["weights_path"])
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
        self.assertEqual(found, 27)


if __name__ == "__main__":
    unittest.main()
