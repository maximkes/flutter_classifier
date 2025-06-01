import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


class DataDownloader:
    def __init__(self, dgl_id: str, dvc_url: str, config: dict):
        self.dgl_id = dgl_id
        self.dvc_url = dvc_url
        self.config = config

    def run_command(self, command, capture_output=False):
        """Run a shell command and return its output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=capture_output,
            )
            return result.stdout if capture_output else None
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(f"Error details: {e.stderr}")
            sys.exit(1)

    def install_gdown(self):
        """Install gdown."""
        print("⏳ Installing gdown...")
        self.run_command(f"{sys.executable} -m pip install gdown")

    def load_dataset_from_google_drive(self):
        """Load dataset from Google Drive folder."""
        dataset_path = Path(self.config["Data"]["dataset_path"])
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
            print('Loading data from Google Drive')

            # Extract folder ID from the URL
            folder_id = "11Aa1cIKzlRNRl0x1FTSUTzBvjiQqJ2_q"

            import gdown
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{folder_id}",
                output=str(dataset_path),
                quiet=False,
                use_cookies=False
            )

            print(f"Dataset loaded to {dataset_path} from Google Drive")
        else:
            print(f'{dataset_path} found')

    def download_data(self) -> None:
        """Download data using gdown."""
        self.install_gdown()

        # Load the dataset from Google Drive folder
        self.load_dataset_from_google_drive()


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(config: DictConfig):
    # Configuration for the dataset

    downloader = DataDownloader(
        "152ftcEEKftLs3WqUGKvaPo8H_gbneO53",
        "https://drive.google.com/drive/folders/11Aa1cIKzlRNRl0x1FTSUTzBvjiQqJ2_q?usp=share_link",
        config
    )
    downloader.download_data()


if __name__ == "__main__":
    main()
