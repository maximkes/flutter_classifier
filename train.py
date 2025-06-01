import hydra
from omegaconf import DictConfig

from data_processing.load_data import (create_data_loaders,
                                       load_dataset_from_kagle)
from flutter_model.train_model import train_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    load_dataset_from_kagle(config)
    train_loader, validation_loader, test_loader = create_data_loaders(config)
    train_model(config, train_loader, validation_loader, test_loader)


if __name__ == "__main__":
    main()
