from pathlib import Path

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
    model = LitModel(config=config)
    # Configure callbacks
    loggers = []
    if config["train"]["use_MLFlow"]:
        mlflow_logger = MLFlowLogger(
            experiment_name=config["train"]["MLFlow_experiment_name"],
            run_name=config["train"]["MLFlow_run_name"],
            save_dir=config["train"]["MLFlow_save_dir"],
            tracking_uri=config["train"][
                "MLFlow_tracking_uri"
            ],  # HTTP instead of HTTPS
        )
        mlflow_logger.log_hyperparams(config)

        loggers.append(mlflow_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["train"]["checkpoints_dirpath"],
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=5,
        mode="min",
        every_n_epochs=1,
        every_n_train_steps=None,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config["train"]["earlystop_min_delta"],
        patience=config["train"]["earlystop_patiance"],
        mode="min",
    )
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval="step"),
        DeviceStatsMonitor(),
        RichModelSummary(max_depth=2),
    ]

    trainer = L.Trainer(
        max_epochs=config["train"]["num_epochs"],
        accelerator="auto",  # Automatically detect best accelerator
        devices="auto",  # Automatically detect available devices
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=1,  # Gradient accumulation
        fast_dev_run=False,  # Set to True for debugging
        val_check_interval=1.0,
    )

    trainer.fit(model, train_loader, validation_loader)

    trainer.test(ckpt_path="best", dataloaders=test_loader)

    if config.get("Export", {}).get("enable_onnx_export", True):
        onnx_path = config["train"]["onnx_path"]
        sample_batch = next(iter(test_loader))
        sample_input = sample_batch[0]
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

        model.to_onnx(onnx_path, sample_input, export_params=True)
        print(f"Model exported to ONNX and saved at {onnx_path}")
    return model, trainer
