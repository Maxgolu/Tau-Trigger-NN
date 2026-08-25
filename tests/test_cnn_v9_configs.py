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

CT_FLATTEN = {"mix1x1": 108, "conv2x2": 48, "serial": 48, "dual": 156}
EM2_FLATTEN = {"em2single": 100, "em2ladder": 72, "em2stride": 72}
LAYOUT_EM2 = [("core_tensors", 0, 45), ("em2_all_cells", 45, 144)]
DIM_EM2 = 189
LAYOUT_FT = [("core_tensors", 0, 45), ("em2_best_3x3_fraction", 45, 1)]
DIM_FT = 46
SEEDS = (42, 123, 456)


class CnnV9ConfigTests(unittest.TestCase):
    def test_twelve_em2_combos_present_and_valid(self):
        found = 0
        for em2, em2_dim in EM2_FLATTEN.items():
            for ct, ct_dim in CT_FLATTEN.items():
                d = PROJECT_ROOT / "configs" / f"cnn_v9_{em2}_{ct}"
                for seed in SEEDS:
                    cfg = json.loads(
                        (d / f"v9_{em2}_{ct}_s{seed}.json").read_text()
                    )
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    self.assertEqual(cfg["classifier"]["name"], "nn_only")
                    self.assertEqual(
                        cfg["loss"]["weighting"]["profile"],
                        "inverse_frequency",
                    )
                    for branch in cfg["model"]["branches"]:
                        self.assertEqual(branch["transform"], "none")
                    model = build_model(cfg, DIM_EM2, LAYOUT_EM2)
                    self.assertIsInstance(model, TensorCNN)
                    first_linear = next(
                        m for m in model.head
                        if isinstance(m, torch.nn.Linear)
                    )
                    self.assertEqual(
                        first_linear.in_features, ct_dim + em2_dim
                    )
                    out = model(torch.randn(4, DIM_EM2))
                    self.assertEqual(tuple(out.shape), (4, 1))
                    found += 1
        self.assertEqual(found, 36)

    def test_finetune_configs_load_pretrained_dual_frac_weights(self):
        d = PROJECT_ROOT / "configs" / "cnn_v9_dualfrac_ft2"
        for seed in SEEDS:
            cfg = json.loads(
                (d / f"v9_dualfrac_ft2_s{seed}.json").read_text()
            )
            self.assertEqual(cfg["loss"]["name"], "constrained_trigger")
            weights = PROJECT_ROOT / cfg["initialization"]["weights_path"]
            self.assertTrue(weights.is_file(), weights)
            # The resolver performs the source-config compatibility checks
            # (features, seed, model block); it must accept these configs.
            resolved = _resolve_initial_weights(cfg, PROJECT_ROOT)
            self.assertEqual(resolved, weights.resolve())
            # The factory model must accept the pretrained state dict exactly.
            model = _build_constrained_model(
                cfg, DIM_FT, LAYOUT_FT, torch.device("cpu")
            )
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)

    def test_resolver_rejects_mismatched_model_block(self):
        # A fine-tune config whose model block differs from the pretrained
        # run's must be refused, so a checkpoint is never loaded into a
        # different architecture.
        d = PROJECT_ROOT / "configs" / "cnn_v9_dualfrac_ft2"
        cfg = json.loads((d / "v9_dualfrac_ft2_s42.json").read_text())
        cfg["model"]["branches"][0]["layers"][0]["out_channels"] = 99
        with self.assertRaises(ValueError):
            _resolve_initial_weights(cfg, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
