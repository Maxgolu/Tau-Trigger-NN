import os
import json
import datetime
import torch
import pandas as pd


class ExperimentTracker:
    def __init__(self, config: dict, base_dir: str = "experiments"):
        """
        Manages file paths, folder structures, and artifact logging for each unique run.

        Args:
            config (dict): The configuration dictionary for the current experiment.
            base_dir (str): The master directory where all runs are cataloged.
        """
        self.config = config
        self.base_dir = base_dir
        self.experiment_name = config.get("experiment_name", "unnamed_run")

        # 1. Generate unique timestamped folder name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"run_{self.experiment_name}_{timestamp}"
        self.experiment_dir = os.path.join(self.base_dir, self.folder_name)

        # 2. Ensure directories exist safely
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 3. Immediately archive a copy of the configuration for long-term record keeping
        self._archive_config()

    def _archive_config(self):
        """Saves a permanent record of the exact config parameters used for this run."""
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_weights(self, model: torch.nn.Module):
        """
        Saves the PyTorch model state dictionary.

        Args:
            model (torch.nn.Module): The trained neural network.
        """
        weights_path = os.path.join(self.experiment_dir, "model_weights.pt")
        # Ensure model weights are moved to CPU before saving to avoid GPU pinning issues on load
        torch.save(model.state_dict(), weights_path)
        print(f"--> Saved model weights to: {weights_path}")

    def save_predictions(self, df_eval: pd.DataFrame):
        """
        Saves the resulting test-set dataframe into an optimized, fast Parquet file.

        Args:
            df_eval (pd.DataFrame): Dataframe containing 'eventNumber', 'tob_index', 'signal', 'Type', 'truth_pt', 'tob_pt', 'tob_eta', 'tob_phi', 'nn_score'.
        """
        parquet_path = os.path.join(self.experiment_dir, "predictions.parquet")

        # Using the standard pyarrow or fastparquet engine under the hood
        df_eval.to_parquet(parquet_path, index=False, engine='auto')
        print(f"--> Saved test predictions to: {parquet_path}")