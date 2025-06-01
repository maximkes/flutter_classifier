import os
import random
import subprocess
import sys
import time

import gdown
import hydra
from omegaconf import DictConfig


class DataDownloader:
    def __init__(self, dvc_url: str, config: dict):
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
        """Install gdown and modify file limit."""
        print("⏳ Installing gdown...")

        # Find gdown installation path
        gdown_path = os.path.dirname(gdown.__file__)
        download_folder_file = os.path.join(gdown_path, "download_folder.py")

        # Read and modify the file
        with open(download_folder_file, "r") as f:
            content = f.read()

        # Change the limit from 50 to a higher number
        modified_content = content.replace(
            "MAX_NUMBER_FILES = 50", "MAX_NUMBER_FILES = 10000"
        )

        with open(download_folder_file, "w") as f:
            f.write(modified_content)

        print("✅ Modified gdown file limit to 10000")

    def download_data(self) -> None:
        self.install_gdown()
        import gdown

        max_retries = 3
        base_delay = 60  # Start with 1 minute delay

        for attempt in range(max_retries):
            try:
                print(f"📥 Download attempt {attempt + 1}/{max_retries}")
                gdown.download_folder(self.dvc_url, quiet=False, remaining_ok=True)
                print("✅ Download completed successfully")
                return

            except gdown.exceptions.FileURLRetrievalError as e:
                if "many accesses" in str(e) and attempt < max_retries - 1:
                    # Rate limiting detected
                    delay = base_delay * (2**attempt) + random.randint(0, 30)
                    print(f"⚠️ Rate limited. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    print(f"❌ Permission or access error: {e}")
                    raise

            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(30)
                else:
                    raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def load_dataset(config: DictConfig):
    # Configuration for the dataset
    downloader = DataDownloader(config["Data"]["google_drive_folder"], config)
    downloader.download_data()


if __name__ == "__main__":
    load_dataset()
