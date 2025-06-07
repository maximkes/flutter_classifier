from mlflow_serve import mlflow_model
import mlflow
import re

import hydra
from omegaconf import DictConfig
from pathlib import Path

def find_best_checkpoint(checkpoints_dirpath):
    """Find the best checkpoint by lowest validation loss from filename"""
    checkpoint_path = Path(checkpoints_dirpath)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    checkpoint_files = list(checkpoint_path.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

    def extract_val_loss(ckpt):
        """Extract validation loss from checkpoint filename"""
        # Fixed regex: match digits, optional decimal point, optional more digits
        match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", ckpt.name)
        if match:
            val_loss_str = match.group(1)
            try:
                return float(val_loss_str)
            except ValueError:
                return float("inf")  # Return high value if conversion fails
        else:
            return float("inf")  # Return high value if no val_loss found

    best_checkpoint = min(checkpoint_files, key=extract_val_loss)
    return best_checkpoint


def get_checkpoint_path(config):
    """Get checkpoint path from config, with fallback to best checkpoint"""
    checkpoint_file = config["Infer"].get("checkpoint_file")

    if checkpoint_file:
        # Use specified checkpoint file
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Specified checkpoint file not found: {checkpoint_path}"
            )
        return checkpoint_path
    else:
        # Find best checkpoint from directory
        checkpoints_dirpath = config["Train"]["checkpoints_dirpath"]
        return find_best_checkpoint(checkpoints_dirpath)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    checkpoint_path = get_checkpoint_path(config)
    
    mlflow.pyfunc.save_model(
        path= config["Infer"]["mlflow_path"],
        python_model=mlflow_model.flutter_classifier(),
        artifacts={"model_weights": str(checkpoint_path), 'config': config},
    )

if __name__ == "__main__":
    main()