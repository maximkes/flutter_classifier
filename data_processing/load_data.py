import ast
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_data_loaders(config):
    # Calculate optimal workers based on CPU count
    num_workers = (
        min(os.cpu_count(), config["data_loader"]["num_workers"])
        if os.cpu_count()
        else 0
    )  # Cap at 8 workers

    # Define transforms
    image_size = config["data"]["image_size"]
    imagenet_stats = ast.literal_eval(
        config["data_loader"]["imagenet_stats"]
    )  # ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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

    dataset_path = Path(config["data"]["dataset_path"])
    batch_size = config["train"]["batch_size"]

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Configure DataLoader parameters
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    # Initialize datasets
    train = torchvision.datasets.ImageFolder(
        dataset_path / config["data"]["train_folder"],
        transform=transformations["train"],
        allow_empty=True,
    )
    validation = torchvision.datasets.ImageFolder(
        dataset_path / config["data"]["val_folder"],
        transform=transformations["validation"],
        allow_empty=True,
    )
    test = torchvision.datasets.ImageFolder(
        dataset_path / config["data"]["test_folder"],
        transform=transformations["test"],
        allow_empty=True,
    )

    # Create data loaders with optimized settings
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation, shuffle=False, **loader_args)
    test_loader = DataLoader(test, shuffle=False, **loader_args)

    print(f"Using {num_workers} workers for data loading")
    print(f"Number of classes: {len(train.classes)}")

    return train_loader, validation_loader, test_loader
