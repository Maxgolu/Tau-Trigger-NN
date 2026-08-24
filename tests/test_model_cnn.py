import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import DynamicMLP, TensorCNN, build_model


def _layout(*name_width_pairs):
    layout = []
    offset = 0
    for name, width in name_width_pairs:
        layout.append((name, offset, width))
        offset += width
    return layout, offset


class TensorCNNTests(unittest.TestCase):
    def test_reshape_preserves_c_order_layer_row_col(self):
        # The registry uses reshape(-1, 5, 3, 3) in C-order, so
        # tensor_(9L + 3r + c) must land at [., L, r, c]. This is the single
        # most important contract: if it breaks, the CNN sees a scrambled image.
        folded = torch.arange(45).reshape(1, 5, 3, 3)
        self.assertEqual(int(folded[0, 2, 1, 1]), 9 * 2 + 3 * 1 + 1)
        self.assertEqual(int(folded[0, 0, 0, 0]), 0)
        self.assertEqual(int(folded[0, 4, 2, 2]), 44)
        self.assertEqual(folded[0, 2].reshape(-1).tolist(), list(range(18, 27)))

    def test_default_config_returns_mlp(self):
        layout, dim = _layout(("core_tensors", 45))
        model = build_model({}, dim, layout)
        self.assertIsInstance(model, DynamicMLP)
        model = build_model({"model": {"name": "mlp"}}, dim, layout)
        self.assertIsInstance(model, DynamicMLP)

    def _cnn(self, layers, extra_features=()):
        pairs = [("core_tensors", 45)] + list(extra_features)
        layout, dim = _layout(*pairs)
        config = {
            "model": {
                "name": "tensor_cnn",
                "branches": [
                    {
                        "feature": "core_tensors",
                        "shape": [5, 3, 3],
                        "transform": "log1p",
                        "layers": layers,
                    }
                ],
                "head": [16],
            }
        }
        return build_model(config, dim, layout), dim

    @staticmethod
    def _count(model):
        return sum(p.numel() for p in model.parameters())

    def test_c1_c2_c3_shapes_and_param_counts(self):
        c1, dim = self._cnn([{"type": "conv", "kernel": 2, "out_channels": 8}])
        c2, _ = self._cnn([{"type": "conv", "kernel": 2, "out_channels": 12}])
        c3, _ = self._cnn([
            {"type": "conv", "kernel": 1, "out_channels": 8},
            {"type": "conv", "kernel": 2, "out_channels": 12},
        ])
        self.assertEqual(self._count(c1), 713)
        self.assertEqual(self._count(c2), 1053)
        self.assertEqual(self._count(c3), 1245)
        x = torch.randn(7, dim)
        for model in (c1, c2, c3):
            out = model(x)
            self.assertEqual(tuple(out.shape), (7, 1))
            logits = model.forward_logits(x)
            self.assertEqual(tuple(logits.shape), (7, 1))
            self.assertTrue(torch.allclose(torch.sigmoid(logits), out, atol=1e-6))

    def test_scalar_features_are_concatenated_at_the_head(self):
        # A feature not consumed by a branch must reach the dense head.
        model, dim = self._cnn(
            [{"type": "conv", "kernel": 2, "out_channels": 8}],
            extra_features=[("em2_best_3x3_fraction", 1),
                            ("core_physics", 4)],
        )
        # conv flatten (32) + scalars (1 + 4) -> first head Linear in_features
        first_linear = next(
            m for m in model.head if isinstance(m, torch.nn.Linear)
        )
        self.assertEqual(first_linear.in_features, 32 + 5)
        out = model(torch.randn(3, dim))
        self.assertEqual(tuple(out.shape), (3, 1))

    def test_model_expects_preprocessed_inputs(self):
        # The input transform (log1p) and standardization happen in train.py
        # preprocessing, not in the model. The model consumes standardized
        # inputs, exactly like DynamicMLP. A standardized batch must produce
        # finite, non-degenerate outputs.
        model, dim = self._cnn([{"type": "conv", "kernel": 2, "out_channels": 8}])
        x = torch.randn(64, dim)  # already-standardized scale
        out = model(x)
        self.assertTrue(torch.isfinite(out).all())
        # Outputs must not be collapsed to a single constant value.
        self.assertGreater(out.std().item(), 0.0)

    def test_shape_mismatch_is_rejected(self):
        layout, dim = _layout(("core_tensors", 45))
        bad = {
            "model": {
                "name": "tensor_cnn",
                "branches": [
                    {"feature": "core_tensors", "shape": [5, 4, 3],
                     "layers": [{"type": "conv", "kernel": 2, "out_channels": 8}]}
                ],
            }
        }
        with self.assertRaises(ValueError):
            build_model(bad, dim, layout)

    def test_kernel_swap_is_config_only(self):
        # C1 with a 1x1 kernel instead of 2x2 must build without code changes.
        model, dim = self._cnn([{"type": "conv", "kernel": 1, "out_channels": 8}])
        out = model(torch.randn(4, dim))
        self.assertEqual(tuple(out.shape), (4, 1))

    def test_cnn_can_learn_on_standardized_inputs(self):
        # Training-stability guard. The original failure was a dead ReLU: with
        # unstandardized all-positive inputs the network collapsed to a constant
        # 0.5 output (BCE = ln 2 = 0.6931) and never recovered. On properly
        # standardized inputs a few Adam steps must drive BCE well below ln 2.
        torch.manual_seed(0)
        model, dim = self._cnn([{"type": "conv", "kernel": 2, "out_channels": 8}])
        n = 512
        x = torch.randn(n, dim)
        # A simple separable signal in the folded image's central cell (EM2
        # centre = flat index 9*2 + 3*1 + 1 = 22).
        y = (x[:, 22] > 0).float().unsqueeze(1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        bce = torch.nn.BCELoss()
        first = None
        for _ in range(150):
            opt.zero_grad()
            loss = bce(model(x), y)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        self.assertLess(loss.item(), 0.60)  # below ln 2, i.e. it actually learned
        self.assertLess(loss.item(), first)


if __name__ == "__main__":
    unittest.main()
