import os
import json
import datetime
import hashlib
import re
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

        # Folder names deliberately use a short ID. The full descriptive name and
        # all hyperparameters remain archived in config.json.
        configured_run_id = str(config.get("run_id", "")).strip()
        if configured_run_id:
            run_id = re.sub(r"[^A-Za-z0-9_-]", "-", configured_run_id)[:40]
        else:
            fingerprint = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
            run_id = f"cfg_{fingerprint}"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"run_{run_id}_{timestamp}"
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
        try:
            # Prefer Parquet because it is compact and fast when an engine is available.
            df_eval.to_parquet(parquet_path, index=False, engine='auto')
            output_path = parquet_path
        except (ImportError, OSError) as error:
            # Some Windows Application Control policies block PyArrow's native DLL.
            # CSV preserves the same table and keeps the pipeline operational.
            output_path = os.path.join(self.experiment_dir, "predictions.csv")
            df_eval.to_csv(output_path, index=False)
            print(f"--> Parquet unavailable ({error.__class__.__name__}); using CSV fallback.")

        print(f"--> Saved test predictions to: {output_path}")
        return output_path
