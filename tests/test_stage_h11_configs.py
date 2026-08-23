import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_objective import parse_constrained_objective


class StageH11TolerantCertificationConfigTests(unittest.TestCase):
    def test_three_seed_configs_match_h1_except_certification_fixes(self):
        for seed in (42, 123, 456):
            with self.subTest(seed=seed):
                h1_path = (
                    PROJECT_ROOT
                    / "configs"
                    / f"constrained_stage_h1_20_s{seed}"
                    / f"h001_s{seed}.json"
                )
                h11_path = (
                    PROJECT_ROOT
                    / "configs"
                    / f"constrained_stage_h11_20_s{seed}"
                    / f"h101_s{seed}.json"
                )
                with h1_path.open("r") as handle:
                    h1 = json.load(handle)
                with h11_path.open("r") as handle:
                    h11 = json.load(handle)

                objective = parse_constrained_objective(h11)
                classifier = parse_classifier(h11)
                self.assertEqual(objective.fpr_feasibility_mode, "point")
                self.assertTrue(objective.certified_guards_use_allowed_deficits)
                self.assertEqual(objective.feasibility_confidence_level, 0.95)
                self.assertTrue(objective.validation_crossfit)
                self.assertEqual(classifier.name, "nn_only")

                # Everything except the two certification switches and the
                # run identifiers must be identical to Stage H1-20.
                h11_loss = dict(h11["loss"])
                self.assertEqual(h11_loss.pop("fpr_feasibility_mode"), "point")
                self.assertTrue(
                    h11_loss.pop("certified_guards_use_allowed_deficits")
                )
                self.assertEqual(h11_loss, h1["loss"])
                for key in ("epochs", "learning_rate", "batch_size",
                            "hidden_layers", "features_to_use", "seed",
                            "initialization", "classifier"):
                    self.assertEqual(h11[key], h1[key])
                self.assertNotEqual(h11["experiment_name"], h1["experiment_name"])


if __name__ == "__main__":
    unittest.main()
