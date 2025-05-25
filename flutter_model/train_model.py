import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

from .model import Model


def train_model(config, train_loader, validation_loader, test_loader):
    Mymodel = Model()
    print("Model created")
    # Move model to GPU if available
    device = torch.device("mps")  # if torch.cuda.is_available() else "cpu")
    print(device)
    Mymodel.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(Mymodel.parameters(), lr=0.001)

    (
        train_loss_history1,
        train_acc_history1,
        val_loss_history1,
        val_acc_history1,
    ) = _train_cycle(
        Mymodel,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        device,
        epochs=config["Train"]["num_epochs"],
    )

    # Evaluate the trained model
    _evaluate_model(Mymodel, test_loader, device)


def _train_cycle(
    model, train_loader, validation_loader, criterion, optimizer, device, epochs=5
):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in trange(epochs):
        # Training phase
        model.train()
        correct, total = 0, 0
        running_train_loss = 0.0  # Track total training loss

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()  # Accumulate loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Compute average training loss & accuracy
        train_loss = running_train_loss / len(train_loader)
        train_acc = correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation phase
        model.eval()
        correct, total = 0, 0
        running_val_loss = 0.0  # Track total validation loss

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()  # Accumulate loss

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Compute average validation loss & accuracy
        val_loss = running_val_loss / len(validation_loader)
        val_acc = correct / total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch + 1} / {epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def _evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
