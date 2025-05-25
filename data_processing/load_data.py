import os
import shutil

import kagglehub
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_dataset_from_kagle(config):
    dataset_path = config["Data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Download latest version
    cache_path = kagglehub.dataset_download("gpiosenka/butterfly-images40-species")

    shutil.copytree(cache_path, dataset_path, dirs_exist_ok=True)
    print(f"Dataset loaded to {dataset_path} from Kaggle")


def load_data_from_folders(config):
    # defining transforms
    image_size = 224
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformations = {
        "train": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_stats),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_stats),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*imagenet_stats),
            ]
        ),
    }

    dataset_path = config["Data"]["dataset_path"]

    train_dir = dataset_path + config["Data"]["train_folder"]
    valid_dir = dataset_path + config["Data"]["val_folder"]
    test_dir = dataset_path + config["Data"]["test_folder"]

    # load train data
    train = torchvision.datasets.ImageFolder(
        train_dir, transform=transformations["train"]
    )
    train_loader = DataLoader(
        train, batch_size=config["Train"]["batch_size"], shuffle=True
    )

    # load validation data
    validation = torchvision.datasets.ImageFolder(
        valid_dir, transform=transformations["validation"]
    )
    validation_loader = DataLoader(
        validation, batch_size=config["Train"]["batch_size"], shuffle=False
    )

    # load test data
    test = torchvision.datasets.ImageFolder(test_dir, transform=transformations["test"])
    test_loader = DataLoader(
        test, batch_size=config["Train"]["batch_size"], shuffle=False
    )

    num_classes = len(train.classes)
    print(f"Number of classes: {num_classes}")

    return train, train_loader, validation, validation_loader, test, test_loader
