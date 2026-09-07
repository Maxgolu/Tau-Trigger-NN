import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pair_models import (
    FlatRawPairMLP,
    PairPtMLP,
    SharedHighResolutionPairModel,
    SharedMemberPairMLP,
    build_pair_model,
    count_parameters,
)


class PairModelTests(unittest.TestCase):
    def test_pt_control_shape_and_parameter_count(self):
        model = PairPtMLP()
        inputs = torch.randn(7, 2)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (7, 1))
        self.assertEqual(count_parameters(model), 33)
        self.assertTrue(
            torch.allclose(model.predict_proba(inputs), torch.sigmoid(logits))
        )

    def test_flat_raw_shape_and_parameter_count(self):
        model = FlatRawPairMLP()
        inputs = torch.randn(7, 90)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (7, 1))
        self.assertEqual(count_parameters(model), 3_457)
        self.assertTrue(
            torch.allclose(model.predict_proba(inputs), torch.sigmoid(logits))
        )

    def test_high_resolution_shape_and_parameter_count(self):
        model = SharedHighResolutionPairModel()
        images = torch.randn(6, 2, 12, 12)
        scalars = torch.randn(6, 2, 47)
        event_second_pt = torch.randn(6, 1)

        logits = model(images, scalars, event_second_pt)

        self.assertEqual(tuple(logits.shape), (6, 1))
        self.assertEqual(count_parameters(model), 4_321)
        self.assertTrue(
            torch.allclose(
                model.predict_proba(images, scalars, event_second_pt),
                torch.sigmoid(logits),
            )
        )

    def test_high_resolution_uses_one_shared_member_encoder(self):
        model = SharedHighResolutionPairModel()

        self.assertEqual(
            len([module for module in model.modules()
                 if isinstance(module, torch.nn.Conv2d)]),
            1,
        )
        self.assertEqual(model.pair_head[0].in_features, 33)

    def test_high_resolution_supports_pair_sum_balance_context(self):
        model = SharedHighResolutionPairModel(
            member_scalar_width=46,
            context_width=3,
        )
        images = torch.randn(5, 2, 12, 12)
        scalars = torch.randn(5, 2, 46)
        context = torch.randn(5, 3)
        self.assertEqual(tuple(model(images, scalars, context).shape), (5, 1))
        self.assertEqual(model.pair_head[0].in_features, 35)
        self.assertEqual(count_parameters(model), 4_369)

    def test_high_resolution_supports_factorial_absent_context(self):
        model = SharedHighResolutionPairModel(
            member_scalar_width=46,
            context_width=0,
        )
        images = torch.randn(5, 2, 12, 12)
        scalars = torch.randn(5, 2, 46)
        context = torch.empty(5, 0)
        self.assertEqual(tuple(model(images, scalars, context).shape), (5, 1))
        self.assertEqual(model.pair_head[0].in_features, 32)
        self.assertEqual(count_parameters(model), 4_273)

    def test_high_resolution_rejects_wrong_shapes(self):
        model = SharedHighResolutionPairModel()
        with self.assertRaises(ValueError):
            model(
                torch.randn(2, 2, 10, 10),
                torch.randn(2, 2, 47),
                torch.randn(2, 1),
            )

    def test_shared_member_model_supports_binary_and_three_class_outputs(self):
        inputs = torch.randn(5, 2, 4)
        binary = SharedMemberPairMLP(4, fusion="ordered", output_classes=1)
        multiclass = SharedMemberPairMLP(
            4,
            fusion="symmetric",
            output_classes=3,
        )
        self.assertEqual(tuple(binary(inputs).shape), (5, 1))
        self.assertEqual(tuple(multiclass(inputs).shape), (5, 3))
        self.assertEqual(tuple(multiclass.predict_proba(inputs).shape), (5, 1))

    def test_symmetric_shared_member_model_is_swap_invariant(self):
        model = SharedMemberPairMLP(3, fusion="symmetric", output_classes=3)
        inputs = torch.randn(7, 2, 3)
        self.assertTrue(torch.allclose(model(inputs), model(inputs.flip(1))))

    def test_reproduced_shared_member_screen_topology(self):
        expected_parameters = {45: 3_601, 46: 3_633, 17: 2_705}
        for member_width, parameter_count in expected_parameters.items():
            with self.subTest(member_width=member_width):
                model = SharedMemberPairMLP(
                    member_width,
                    member_encoder=[member_width, 32, 16],
                    fusion="symmetric_sum_absolute_difference",
                    activation="leaky_relu_0.01",
                    initialization="pytorch_default",
                )
                inputs = torch.randn(5, 2, member_width)
                self.assertEqual(tuple(model(inputs).shape), (5, 1))
                self.assertTrue(
                    torch.allclose(model(inputs), model(inputs.flip(1)))
                )
                self.assertEqual(count_parameters(model), parameter_count)

    def test_shared_member_model_rejects_invalid_contract(self):
        with self.assertRaises(ValueError):
            SharedMemberPairMLP(0)
        with self.assertRaises(ValueError):
            SharedMemberPairMLP(2, fusion="unknown")
        with self.assertRaises(ValueError):
            SharedMemberPairMLP(2, output_classes=2)
        with self.assertRaises(ValueError):
            SharedMemberPairMLP(2, initialization="unknown")

    def test_model_factory(self):
        self.assertIsInstance(
            build_pair_model({"model": {"name": "pair_pt_mlp"}}),
            PairPtMLP,
        )
        self.assertIsInstance(
            build_pair_model({"name": "flat_pair_mlp"}),
            FlatRawPairMLP,
        )
        self.assertIsInstance(
            build_pair_model(
                {"model": {"name": "shared_member_high_resolution_em2"}}
            ),
            SharedHighResolutionPairModel,
        )
        self.assertIsInstance(
            build_pair_model(
                {
                    "model": {
                        "name": "shared_member_pair_mlp",
                        "member_width": 4,
                        "fusion": "ordered",
                        "output_classes": 1,
                    }
                }
            ),
            SharedMemberPairMLP,
        )
        with self.assertRaises(ValueError):
            build_pair_model({"name": "unknown"})

    def test_model_factory_rejects_conflicting_architecture_declarations(self):
        conflicts = (
            {"name": "pair_pt_mlp", "layers": [2, 16, 1]},
            {"name": "flat_pair_mlp", "activation": "gelu"},
            {
                "name": "shared_member_pair_mlp",
                "member_width": 4,
                "member_embedding_width": 8,
            },
            {
                "name": "shared_member_pair_mlp",
                "member_width": 4,
                "pair_head": [32, 16, 1],
            },
            {
                "name": "shared_member_high_resolution_em2",
                "trainable_parameters": 1,
            },
        )
        for config in conflicts:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_pair_model(config)

    def test_initialization_is_explicit_and_reproducible(self):
        for constructor in (PairPtMLP, FlatRawPairMLP, SharedHighResolutionPairModel):
            torch.manual_seed(123)
            first = constructor()
            torch.manual_seed(123)
            second = constructor()
            for first_parameter, second_parameter in zip(
                first.parameters(), second.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(first_parameter, second_parameter))
            for module in first.modules():
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_shared_member_default_initialization_can_be_reproduced(self):
        torch.manual_seed(123)
        first = SharedMemberPairMLP(4, initialization="pytorch_default")
        torch.manual_seed(123)
        second = SharedMemberPairMLP(4, initialization="pytorch_default")
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(first_parameter, second_parameter))


if __name__ == "__main__":
    unittest.main()
