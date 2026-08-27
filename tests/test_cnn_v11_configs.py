import json
import sys
import unittest
from collections import defaultdict
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

FEATURE_WIDTHS = {"core_tensors": 45, "em2_all_cells": 144,
                  "em2_best_3x3_fraction": 1}
SEEDS = (42, 123, 456)

FRAC_EXPERIMENTS = [
    f"{e}_{c}_frac"
    for e in ("em2single", "em2ladder", "em2stride")
    for c in ("mix1x1", "conv2x2", "dual")
    if not (e == "em2single" and c == "dual")  # already run as cnn_v10
]
FT_EXPERIMENTS = [
    f"ft_{e}_{c}"
    for e in ("em2ladder", "em2stride")
    for c in ("mix1x1", "conv2x2", "dual")
] + ["ft_em2single_dual_frac"]
BCE_EXPERIMENTS = ["em2single_dual_frac_bce"]


def _layout_for(features):
    layout, offset = [], 0
    for name in features:
        width = FEATURE_WIDTHS[name]
        layout.append((name, offset, width))
        offset += width
    return layout, offset


def _collect():
    by_id = {}
    for d in sorted((PROJECT_ROOT / "configs").glob("cnn_v11_gpu*")):
        for p in d.glob("*.json"):
            cfg = json.loads(p.read_text())
            by_id[cfg["run_id"]] = cfg
    return by_id


class CnnV11ConfigTests(unittest.TestCase):
    def test_fortyeight_configs_present(self):
        by_id = _collect()
        expected = FRAC_EXPERIMENTS + FT_EXPERIMENTS + BCE_EXPERIMENTS
        for exp in expected:
            for seed in SEEDS:
                self.assertIn(f"v11_{exp}_s{seed}", by_id)
        self.assertEqual(len(by_id), 48)

    def test_training_configs_valid(self):
        by_id = _collect()
        for exp in FRAC_EXPERIMENTS + BCE_EXPERIMENTS:
            for seed in SEEDS:
                cfg = by_id[f"v11_{exp}_s{seed}"]
                parse_classifier(cfg)
                parse_loss(cfg)
                expected_loss = ("bce" if exp.endswith("_bce")
                                 else "energy_weighted_bce")
                self.assertEqual(cfg["loss"]["name"], expected_loss)
                layout, dim = _layout_for(cfg["features_to_use"])
                model = build_model(cfg, dim, layout)
                self.assertIsInstance(model, TensorCNN)
                out = model(torch.randn(4, dim))
                self.assertEqual(tuple(out.shape), (4, 1))

    def test_finetune_configs_resolve_and_load(self):
        by_id = _collect()
        for exp in FT_EXPERIMENTS:
            for seed in SEEDS:
                cfg = by_id[f"v11_{exp}_s{seed}"]
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

    def test_gpu_dirs_are_balanced(self):
        # ft runs cost roughly twice a fresh training run; the greedy split
        # must keep per-directory load within one unit of the mean.
        loads = defaultdict(float)
        for d in sorted((PROJECT_ROOT / "configs").glob("cnn_v11_gpu*")):
            for p in d.glob("*.json"):
                cfg = json.loads(p.read_text())
                cost = 2.0 if cfg["loss"]["name"] == "constrained_trigger" \
                    else 1.0
                loads[d.name] += cost
        values = list(loads.values())
        self.assertEqual(len(values), 12)
        self.assertLessEqual(max(values) - min(values), 1.0)


if __name__ == "__main__":
    unittest.main()
