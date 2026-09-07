import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pair_models import build_pair_model
from prepare_pair_data import preparation_contract


class PairConfigTests(unittest.TestCase):
    def test_every_pair_config_is_single_seed_and_runnable(self):
        allowed_representations = {
            "pair_pt",
            "raw_cells",
            "raw_cells_pt",
            "summaries",
            "summaries_pt",
            "high_resolution_em2",
            "high_resolution_em2_context",
            "prepared_member_features",
            "member_coarse_cells",
            "member_coarse_cells_pt",
            "member_compact_em2_pt",
        }
        paths = sorted((ROOT / "configs").glob("pair_*/*.json"))
        self.assertEqual(len(paths), 87)
        seen = {}
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(config["seed"], {42, 123, 456})
            self.assertNotIn("seeds", config)
            self.assertEqual(path.parent.name, config["experiment_name"])
            self.assertTrue(path.stem.endswith(f"_s{config['seed']}"))
            self.assertIn(config["representation"], allowed_representations)
            preparation_contract(config)
            model = build_pair_model(config)
            self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)
            self.assertEqual(config["optimizer"]["name"], "adam")
            self.assertTrue(config["checkpoint_each_epoch"])
            self.assertEqual(config["event_aggregation"], "maximum_pair_score")
            self.assertEqual(config["target_event_fpr"], 0.005)
            loss = config.get("loss", {})
            loss_name = loss.get("name", "bce")
            self.assertIn(loss_name, {"bce", "cross_entropy"})
            if loss_name == "cross_entropy":
                self.assertEqual(config["model"]["output_classes"], 3)
            seen.setdefault(config["experiment_name"], set()).add(config["seed"])
        self.assertEqual(len(seen), 29)
        self.assertTrue(all(seeds == {42, 123, 456} for seeds in seen.values()))


if __name__ == "__main__":
    unittest.main()
