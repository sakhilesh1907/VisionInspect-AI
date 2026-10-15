"""
scripts/train_all.py

Automates training on ALL available product categories on CPU.
Discovers every category in data/raw/mvtec/ and trains the selected model.

Usage:
    python scripts/train_all.py --model autoencoder --epochs 10 --device cpu
    python scripts/train_all.py --model patchcore --device cpu
"""

import sys
import time
from pathlib import Path
import click
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.dataset import MVTecDataset, get_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from models import build_model
from utils.logging import get_logger

logger = get_logger("train_all")


@click.command()
@click.option("--model", "-m", default="autoencoder", help="Model: autoencoder | patchcore | padim")
@click.option("--data-root", default="data/raw/mvtec", help="Directory where category folders live")
@click.option("--epochs", default=10, help="Training epochs for gradient models")
@click.option("--batch", default=4, help="Batch size (recommended 4 for CPU)")
@click.option("--output", default="results/models", help="Output directory")
@click.option("--device", default="cpu", help="Device: cpu or cuda")
def main(model, data_root, epochs, batch, output, device):
    root_path = Path(data_root)
    if not root_path.exists():
        logger.error(f"Data root {data_root} does not exist.")
        return

    # Find all categories with a train/good folder
    categories = []
    for cat_dir in sorted(root_path.iterdir()):
        if cat_dir.is_dir() and (cat_dir / "train" / "good").exists():
            categories.append(cat_dir.name)

    if not categories:
        logger.warning(f"No valid category datasets found in {data_root}.")
        return

    logger.info("=" * 65)
    logger.info(f" 🚀 BATCH TRAINING ON ALL CATEGORIES ({len(categories)} found)")
    logger.info(f" Categories: {categories}")
    logger.info(f" Model:      {model}")
    logger.info(f" Device:     {device}")
    logger.info(f" Epochs:     {epochs}")
    logger.info("=" * 65)

    start_all = time.time()
    successful = []
    failed = []

    dev = torch.device(device)

    for idx, cat in enumerate(categories, 1):
        logger.info(f"\n[{idx}/{len(categories)}] ---> Training on '{cat}'...")
        cat_start = time.time()

        try:
            train_ds = MVTecDataset(root=str(root_path), category=cat, split="train", transform=get_train_transforms())
            train_dl = get_dataloader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
            logger.info(f"Loaded {len(train_ds)} normal training images.")

            out_dir = Path(output) / model / cat
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / "model.pt"

            anomaly_model = build_model(model, device=device)

            if hasattr(anomaly_model, "fit"):
                # Unsupervised memory bank / distribution models (PatchCore, PaDiM)
                logger.info(f"Fitting {model} memory bank...")
                anomaly_model.fit(train_dl, verbose=True)
                anomaly_model.save(str(save_path))
            else:
                # Gradient models (CAE, StudentTeacher)
                anomaly_model.to(dev).train()
                optimizer = torch.optim.Adam(anomaly_model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                for ep in range(1, epochs + 1):
                    loss_sum = 0.0
                    for b in train_dl:
                        imgs = b["image"].to(dev)
                        optimizer.zero_grad()
                        recon, _ = anomaly_model(imgs)
                        loss = anomaly_model.loss(imgs, recon)
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item()
                    scheduler.step()

                    if ep % 5 == 0 or ep == epochs:
                        logger.info(f"  Epoch {ep:2d}/{epochs} - Loss: {loss_sum / len(train_dl):.4f}")

                torch.save(anomaly_model.state_dict(), str(save_path))

            elapsed = time.time() - cat_start
            logger.info(f"✅ Finished '{cat}' in {elapsed:.1f}s -> Saved to {save_path}")
            successful.append((cat, elapsed))

        except Exception as e:
            logger.error(f"❌ Error training on '{cat}': {e}")
            failed.append((cat, str(e)))

    total_time = time.time() - start_all
    logger.info("\n" + "=" * 65)
    logger.info(f"🎉 BATCH TRAINING COMPLETE in {total_time:.1f}s")
    logger.info(f" Successfully trained: {[c[0] for c in successful]}")
    if failed:
        logger.info(f" Failed: {failed}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
