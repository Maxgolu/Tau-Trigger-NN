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


EXPECTED_PARAMS = {"c1": 713, "c2": 1053, "c3": 1245}


class CnnArchitectureConfigTests(unittest.TestCase):
    def test_eighteen_configs_present_and_valid(self):
        found = 0
        for arch in ("c1", "c2", "c3"):
            for seed in (42, 123, 456):
                d = PROJECT_ROOT / "configs" / f"cnn_arch_{arch}_s{seed}"
                self.assertTrue(d.is_dir(), d)
                files = sorted(p.name for p in d.glob("*.json"))
                self.assertEqual(
                    files,
                    [f"{arch}_invfreq_s{seed}.json", f"{arch}_plaw_s{seed}.json"],
                )
                for name in files:
                    cfg = json.loads((d / name).read_text())
                    # Parsers accept the config unchanged.
                    parse_classifier(cfg)
                    parse_loss(cfg)
                    self.assertEqual(cfg["seed"], seed)
                    self.assertEqual(cfg["features_to_use"], ["core_tensors"])
                    self.assertEqual(cfg["classifier"]["name"], "nn_only")
                    branch = cfg["model"]["branches"][0]
                    self.assertEqual(branch["feature"], "core_tensors")
                    self.assertEqual(branch["shape"], [5, 3, 3])
                    self.assertEqual(branch["transform"], "log1p")
                    # Build the model from the config and check the param count.
                    model = build_model(cfg, 45, [("core_tensors", 0, 45)])
                    self.assertIsInstance(model, TensorCNN)
                    self.assertEqual(
                        sum(p.numel() for p in model.parameters()),
                        EXPECTED_PARAMS[arch],
                    )
                    out = model(torch.randn(5, 45))
                    self.assertEqual(tuple(out.shape), (5, 1))
                    found += 1
        self.assertEqual(found, 18)

    def test_loss_profiles_are_the_two_requested(self):
        c1 = PROJECT_ROOT / "configs" / "cnn_arch_c1_s42"
        inv = json.loads((c1 / "c1_invfreq_s42.json").read_text())["loss"]
        plaw = json.loads((c1 / "c1_plaw_s42.json").read_text())["loss"]
        self.assertEqual(inv["weighting"]["profile"], "inverse_frequency")
        self.assertEqual(plaw["weighting"]["profile"], "power_law")
        self.assertEqual(plaw["weighting"]["p"], -1.0)


if __name__ == "__main__":
    unittest.main()
