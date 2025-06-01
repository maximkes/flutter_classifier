import lightning.pytorch as pl  # For Lightning >=2.0
import torch
from torch import nn
from torchmetrics import Accuracy


class LitModel(pl.LightningModule):
    def __init__(self, num_classes=100, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Convolutional layers
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
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            # nn.Dropout(0.3),  # Uncomment for dropout
            nn.Linear(1024, num_classes)
        )

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.val_acc(preds, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.test_acc(preds, targets)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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
