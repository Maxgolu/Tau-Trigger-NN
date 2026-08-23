import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_objective import parse_constrained_objective


FAMILIES = {
    "ct": {
        "features": ["core_tensors"],
        "pretrain_prefix": "run_c002_s",
    },
    "cpct": {
        "features": ["core_physics", "core_tensors"],
        "pretrain_prefix": "run_c005_s",
    },
    "sumf": {
        "features": [
            "em0_sparsity", "em1_sparsity", "em2_3x3_sparsity",
            "em3_sparsity", "had_sparsity", "em0_sum", "em1_sum",
            "em2_3x3_sum", "em3_sum", "had_sum", "em0_dominance",
            "em1_dominance", "em2_3x3_dominance", "em3_dominance",
            "had_dominance", "em2_best_3x3_fraction",
        ],
        "pretrain_prefix": "run_c004_s",
    },
}


class StageIGeneralizationConfigTests(unittest.TestCase):
    def test_nine_configs_reuse_h11_loss_and_swap_only_inputs(self):
        h11_path = (
            PROJECT_ROOT
            / "configs"
            / "constrained_stage_h11_20_s42"
            / "h101_s42.json"
        )
        with h11_path.open("r") as handle:
            reference = json.load(handle)

        for family, spec in FAMILIES.items():
            for seed in (42, 123, 456):
                with self.subTest(family=family, seed=seed):
                    path = (
                        PROJECT_ROOT
                        / "configs"
                        / f"constrained_stage_i_{family}_s{seed}"
                        / f"i001_s{seed}.json"
                    )
                    with path.open("r") as handle:
                        config = json.load(handle)

                    objective = parse_constrained_objective(config)
                    classifier = parse_classifier(config)

                    # The loss, classifier, and optimizer settings must be
                    # byte-identical to the B1 (Stage H1.1) recipe.
                    self.assertEqual(config["loss"], reference["loss"])
                    self.assertEqual(config["classifier"], reference["classifier"])
                    for key in ("epochs", "learning_rate", "batch_size",
                                "hidden_layers"):
                        self.assertEqual(config[key], reference[key])

                    self.assertEqual(config["seed"], seed)
                    self.assertEqual(config["features_to_use"], spec["features"])
                    weights_path = config["initialization"]["weights_path"]
                    self.assertIn(
                        f"nn_only_benchmark_s{seed}/"
                        f"{spec['pretrain_prefix']}{seed}_",
                        weights_path,
                    )
                    self.assertTrue(weights_path.endswith("model_weights.pt"))

                    self.assertEqual(objective.fpr_feasibility_mode, "point")
                    self.assertTrue(
                        objective.certified_guards_use_allowed_deficits
                    )
                    self.assertTrue(objective.validation_crossfit)
                    self.assertEqual(classifier.name, "nn_only")


if __name__ == "__main__":
    unittest.main()
