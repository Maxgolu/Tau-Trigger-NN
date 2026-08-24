import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifiers import parse_classifier
from constrained_objective import parse_constrained_objective
from constrained_training import resolve_constrained_or_budget


class StageJRankingOrConfigTests(unittest.TestCase):
    def test_three_seed_configs_add_ranking_and_budget_search_only(self):
        for seed in (42, 123, 456):
            with self.subTest(seed=seed):
                h11_path = (
                    PROJECT_ROOT
                    / "configs"
                    / f"constrained_stage_h11_20_s{seed}"
                    / f"h101_s{seed}.json"
                )
                j_path = (
                    PROJECT_ROOT
                    / "configs"
                    / f"constrained_stage_j_rankor_s{seed}"
                    / f"j001_s{seed}.json"
                )
                with h11_path.open("r") as handle:
                    h11 = json.load(handle)
                with j_path.open("r") as handle:
                    stage_j = json.load(handle)

                classifier = parse_classifier(stage_j)
                objective = parse_constrained_objective(stage_j)
                candidates, surrogate = resolve_constrained_or_budget(
                    classifier, objective
                )

                # Training loss: identical to B1 except the ranking primal.
                j_loss = dict(stage_j["loss"])
                self.assertEqual(j_loss.pop("primal_objective"), "tail_ranking")
                h11_loss = dict(h11["loss"])
                h11_loss.pop("primal_objective")
                self.assertEqual(j_loss, h11_loss)
                for key in ("epochs", "learning_rate", "batch_size",
                            "hidden_layers", "features_to_use", "seed",
                            "initialization"):
                    self.assertEqual(stage_j[key], h11[key])

                # Classifier: OR with a measurement-level budget search.
                self.assertEqual(classifier.name, "tob_nn_or")
                self.assertEqual(
                    [candidate.tob_fpr for candidate in candidates],
                    [0.0, 0.0005, 0.001, 0.0015, 0.002],
                )
                self.assertEqual(surrogate.name, "nn_only")
                self.assertEqual(objective.tail_memory_bank_size, 0)
                self.assertTrue(objective.validation_crossfit)
                self.assertEqual(objective.fpr_feasibility_mode, "point")
                self.assertTrue(
                    objective.certified_guards_use_allowed_deficits
                )


if __name__ == "__main__":
    unittest.main()
