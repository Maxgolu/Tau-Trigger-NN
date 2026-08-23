import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_objective import parse_constrained_objective


class StageH1TwentyEpochConfigTests(unittest.TestCase):
    def test_three_seed_configs_keep_h1_and_enable_validation_safety(self):
        expected_weights = {
            42: "experiments/nn_only_core_fraction_s42/run_c001_s42_20260819_232615/model_weights.pt",
            123: "experiments/nn_only_core_fraction_s123/run_c001_s123_20260819_232615/model_weights.pt",
            456: "experiments/nn_only_core_fraction_s456/run_c001_s456_20260819_232614/model_weights.pt",
        }
        for seed, weights_path in expected_weights.items():
            with self.subTest(seed=seed):
                path = (
                    PROJECT_ROOT
                    / "configs"
                    / f"constrained_stage_h1_20_s{seed}"
                    / f"h001_s{seed}.json"
                )
                with path.open("r") as handle:
                    config = json.load(handle)
                objective = parse_constrained_objective(config)
                classifier = parse_classifier(config)
                self.assertEqual(config["epochs"], 20)
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["initialization"]["weights_path"], weights_path)
                self.assertEqual(objective.primal_objective, "soft_efficiency")
                self.assertEqual(objective.proxy_threshold_mode, "batch_rank")
                self.assertEqual(objective.tail_memory_bank_size, 0)
                self.assertTrue(objective.validation_crossfit)
                self.assertEqual(objective.feasibility_confidence_level, 0.95)
                self.assertEqual(classifier.name, "nn_only")
                self.assertEqual(classifier.target_fpr, 0.005)


if __name__ == "__main__":
    unittest.main()
