import ast
import re
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

# Import your actual model class
from flutter_model.model import LitModel


def get_class_names_from_dataset(dataset_path="dataset/TRAIN"):
    """Extract class names from dataset folder structure using pathlib"""
    dataset_dir = Path(dataset_path)

    if dataset_dir.exists() and dataset_dir.is_dir():
        class_names = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        return class_names
    else:
        print(f"Dataset path {dataset_path} does not exist!")
        return []


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


def load_and_preprocess_image(image_path, config):
    """Load and preprocess a single image for inference"""
    # Convert to Path object
    image_path = Path(image_path)

    # Check if file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get image size and normalization stats from config
    image_size = config["Data"]["image_size"]
    imagenet_stats = ast.literal_eval(config["DataLoader"]["imagenet_stats"])

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*imagenet_stats),
        ]
    )

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Load class names using pathlib
    class_names = get_class_names_from_dataset("dataset/TRAIN")

    if not class_names:
        print("No class names found! Please check your dataset path.")
        return

    print(f"Found {len(class_names)} classes")

    # Get checkpoint path from config
    try:
        checkpoint_path = get_checkpoint_path(config)
        print(f"Loading checkpoint: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load the model from checkpoint
    model = LitModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Get image path from config
    image_path = Path(config["Infer"]["infer_image"])

    try:
        input_tensor = load_and_preprocess_image(image_path, config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Move to GPU if available
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    print(f"Processing image: {image_path}")
    print(f"Using device: {device}")

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

        # Get predictions
        probabilities = F.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()

        # Get the class name
        predicted_class_name = class_names[predicted_class_idx]

        print(f"\nMost likely class: {predicted_class_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Class index: {predicted_class_idx}")

        # Show top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
        print("\nTop 5 predictions:")
        for i in range(min(5, len(class_names))):
            idx = top5_idx[0][i].item()
            prob = top5_prob[0][i].item()
            print(f"   {i + 1}. {class_names[idx]}: {prob:.2%}")


if __name__ == "__main__":
    main()
