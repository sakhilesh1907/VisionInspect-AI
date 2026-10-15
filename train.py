"""
train.py — VisionInspect AI Training Script

Supports:
  - Unsupervised fitting (PaDiM, PatchCore)
  - Gradient-based training (CAE, StudentTeacher)
  - MLflow experiment tracking
  - Configurable via YAML or CLI flags

Usage:
    python train.py --model patchcore --category bottle
    python train.py --model autoencoder --category tile --epochs 50 --lr 1e-3
    python train.py --model padim --category hazelnut --config experiments/padim_mvtec.yaml
"""

import os
import sys
import time
import click
import yaml
import torch
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import MVTecDataset, get_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from models import build_model
from evaluation.metrics import full_evaluation_report
from utils.logging import get_logger

logger = get_logger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model",    "-m", default="patchcore",   show_default=True, help="Model name: autoencoder | padim | patchcore | student_teacher")
@click.option("--category", "-c", default="bottle",      show_default=True, help="MVTec AD category")
@click.option("--data-root", default="data/raw/mvtec",   show_default=True, help="Path to MVTec AD root")
@click.option("--config",   default=None,                                   help="Path to YAML config file")
@click.option("--epochs",   default=50,                  show_default=True, help="Training epochs (gradient models)")
@click.option("--lr",       default=1e-3,                show_default=True, help="Learning rate")
@click.option("--batch",    default=16,                  show_default=True, help="Batch size")
@click.option("--workers",  default=4,                   show_default=True, help="DataLoader workers")
@click.option("--output",   default="results/models",    show_default=True, help="Model output directory")
@click.option("--device",   default="cuda",              show_default=True, help="cuda | cpu")
@click.option("--few-shot-normal", default=None, type=int, help="Limit normal training images")
@click.option("--experiment-name", default="visioninspect", show_default=True)
@click.option("--use-mlflow", is_flag=True, default=False, help="Enable MLflow experiment tracking")
def main(model, category, data_root, config, epochs, lr, batch, workers, output, device, few_shot_normal, experiment_name, use_mlflow):
    """Train a VisionInspect AI anomaly detection model."""

    # ---- Load YAML config (overrides CLI defaults) ----
    cfg = {}
    if config and Path(config).exists():
        with open(config) as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {config}")

    model_name   = cfg.get("model", model)
    category     = cfg.get("category", category)
    data_root    = cfg.get("data_root", data_root)
    epochs       = cfg.get("epochs", epochs)
    lr           = cfg.get("lr", lr)
    batch        = cfg.get("batch_size", batch)
    device_str   = cfg.get("device", device)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_name} | Category: {category}")

    # ---- Datasets ----
    train_ds = MVTecDataset(
        root=data_root,
        category=category,
        split="train",
        transform=get_train_transforms(),
        few_shot_normal=few_shot_normal,
    )
    test_ds = MVTecDataset(
        root=data_root,
        category=category,
        split="test",
        transform=get_val_transforms(),
    )

    train_dl = get_dataloader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    test_dl  = get_dataloader(test_ds,  batch_size=batch, shuffle=False, num_workers=workers)

    logger.info(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    # ---- Build Model ----
    model_kwargs = cfg.get("model_kwargs", {})
    model_kwargs["device"] = device_str
    anomaly_model = build_model(model_name, **model_kwargs)

    # ---- MLflow Tracking ----
    tracking_active = False
    if HAS_MLFLOW and use_mlflow:
        try:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=f"{model_name}_{category}")
            mlflow.log_params({
                "model": model_name,
                "category": category,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch,
                "device": str(device),
            })
            tracking_active = True
        except Exception as e:
            logger.warning(f"MLflow initialization skipped: {e}")

    # ---- Training ----
    output_dir = Path(output) / model_name / category
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(anomaly_model, "fit"):
        # Unsupervised: PaDiM, PatchCore
        logger.info(f"Fitting {model_name} on normal training images...")
        anomaly_model.fit(train_dl, verbose=True)
        save_path = str(output_dir / "model.pt")
        anomaly_model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        if tracking_active:
            try:
                mlflow.log_artifact(save_path)
            except Exception:
                pass

    else:
        # Gradient-based: CAE, StudentTeacher
        logger.info(f"Training {model_name} for {epochs} epochs...")
        if hasattr(anomaly_model, "get_trainable_params"):
            params = anomaly_model.get_trainable_params()
        else:
            params = anomaly_model.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        anomaly_model.to(device)

        for epoch in range(1, epochs + 1):
            anomaly_model.train()
            epoch_loss = 0.0
            for batch_data in train_dl:
                imgs = batch_data["image"].to(device)
                optimizer.zero_grad()

                if model_name == "student_teacher":
                    tf, sf = anomaly_model(imgs)
                    loss = anomaly_model.loss(tf, sf)
                else:
                    recon, _ = anomaly_model(imgs)
                    loss = anomaly_model.loss(imgs, recon)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_dl)

            if epoch % 10 == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}")
                if tracking_active:
                    try:
                        mlflow.log_metric("train_loss", avg_loss, step=epoch)
                    except Exception:
                        pass

        save_path = str(output_dir / "model.pt")
        torch.save(anomaly_model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
        if tracking_active:
            try:
                mlflow.log_artifact(save_path)
            except Exception:
                pass

    if tracking_active:
        try:
            mlflow.end_run()
        except Exception:
            pass

    logger.info(f"Training complete. Model at: {save_path}")


if __name__ == "__main__":
    main()
