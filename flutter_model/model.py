import lightning.pytorch as pl
import torch
import torchvision.models as models
from torch import nn
from torchmetrics import Accuracy


class LitModel(pl.LightningModule):
    def __init__(self, config, num_classes=100):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["train"]["learning_rate"]
        self.model_choice = config["train"]["model_choice"]

        if self.model_choice == "own":
            self._build_original_model(num_classes)

        elif self.model_choice == "dp":
            # ResNet with frozen weights and 3 FC layers
            base_model_name = config["train"]["base_model"]
            self._build_resnet_model(base_model_name, num_classes)

        elif self.model_choice == "own_light":
            self._build_lightweight_model(num_classes)

        else:
            raise ValueError(
                f"Invalid model_choice: {self.model_choice}. Choose 1, 2, or 3."
            )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _build_original_model(self, num_classes):
        """Build the original convolutional model"""
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def _build_resnet_model(self, base_model_name, num_classes):
        """Build ResNet model with frozen weights and 3 FC layers"""
        # Load pretrained ResNet based on configuration
        if base_model_name.lower() == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
            num_features = 512
        elif base_model_name.lower() == "resnet34":
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1
            )
            num_features = 512
        elif base_model_name.lower() == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            num_features = 2048
        elif base_model_name.lower() == "resnet101":
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V1
            )
            num_features = 2048
        elif base_model_name.lower() == "resnet152":
            self.backbone = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V1
            )
            num_features = 2048
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        # Rest of your code remains the same...

        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Remove the original FC layer and replace with Identity
        self.backbone.fc = nn.Identity()

        # Add 3 fully connected layers as specified
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _build_lightweight_model(self, num_classes):
        """Build lightweight own_light model for efficient deployment"""
        self.backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Get number of features from the existing classifier
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier with simplified version for efficiency
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        if self.model_choice == "own":
            # Original model forward pass
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
        elif self.model_choice == "dp":
            # ResNet forward pass with frozen backbone
            x = self.backbone(x)  # Features extracted, fc is Identity
            x = self.fc_layers(x)
        elif self.model_choice == "own_light":
            x = self.backbone(x)

        return x

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.train_acc(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.val_acc(preds, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.test_acc(preds, targets)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Production-ready prediction step with batch processing
        Handles both single instances and batched inputs
        Returns: Tuple of (logits, probabilities, class_labels)
        """
        # Handle different input formats
        if isinstance(batch, (list, tuple)):
            x = batch[0]  # Assume first element is input tensor
        else:
            x = batch

        # Convert single instance to batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Forward pass
        logits = self(x)

        # Post-processing
        probabilities = torch.softmax(logits, dim=1)
        class_labels = torch.argmax(logits, dim=1)

        return (logits, probabilities, class_labels)
