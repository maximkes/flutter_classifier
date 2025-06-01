import lightning as L
from lightning.pytorch.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         LearningRateMonitor, ModelCheckpoint,
                                         RichModelSummary)
from lightning.pytorch.loggers import MLFlowLogger

from .model import LitModel


def train_model(config, train_loader, validation_loader, test_loader):
    """
    Advanced training with additional Lightning features
    """

    # Initialize the Lightning model
    model = LitModel(num_classes=100, learning_rate=0.001)

    # Configure callbacks
    loggers = [
        MLFlowLogger(
            experiment_name="flutter_classifier",
            run_name="conv_classifier",
            save_dir="./mlflow_logs",  # More explicit directory
            tracking_uri="http://127.0.0.1:8080"  # HTTP instead of HTTPS
        )
    ]

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename="{epoch: 02d}-{val_loss: .4f}",
        save_top_k=5,
        mode='min',
        every_n_epochs=1,
        every_n_train_steps=None
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        mode='min'
    )

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval='step'),
        DeviceStatsMonitor(),
        RichModelSummary(max_depth=2)
    ]
    # Configure trainer with advanced features
    trainer = L.Trainer(
        max_epochs=config["Train"]["num_epochs"],
        accelerator="auto",  # Automatically detect best accelerator
        devices="auto",      # Automatically detect available devices
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,  # Gradient accumulation
        fast_dev_run=False,   # Set to True for debugging
        # val_check_interval=1.0
    )

    # Train the model
    trainer.fit(model, train_loader, validation_loader)

    # Load best checkpoint and test
    trainer.test(ckpt_path="best", dataloaders=test_loader)

    return model, trainer
