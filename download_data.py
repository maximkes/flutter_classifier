import io
import zipfile
from urllib.parse import urlencode

import hydra
import requests
from omegaconf import DictConfig
from tqdm import tqdm


def download_with_progress(url):
    """Download file with progress bar using tqdm."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    data = bytearray()
    for data_block in response.iter_content(block_size):
        t.update(len(data_block))
        data.extend(data_block)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
    return data


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    url = config["DataLoader"]["yandex_link"]
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    final_url = base_url + urlencode(dict(public_key=url))

    print(f"Installing from {final_url}")
    response = requests.get(final_url)
    download_url = response.json()["href"]

    print(f"Downloading from {download_url}...")
    download_response_content = download_with_progress(download_url)

    print("Installed successfully")
    zip_file = zipfile.ZipFile(io.BytesIO(download_response_content))
    zip_file.extractall(config["Data"]["dataset_path"])
    print(f"Dataset installed to {config['Data']['dataset_path']}")


if __name__ == "__main__":
    main()
