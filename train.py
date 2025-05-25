import yaml

from data_processing.load_data import (load_data_from_folders,
                                       load_dataset_from_kagle)
from flutter_model.train_model import train_model


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    load_dataset_from_kagle(config)
    (
        train,
        train_loader,
        validation,
        validation_loader,
        test,
        test_loader,
    ) = load_data_from_folders(config)
    train_model(config, train_loader, validation_loader, test_loader)


if __name__ == "__main__":
    main()
