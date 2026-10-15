"""
evaluate.py — VisionInspect AI Evaluation Script

Loads a trained model and evaluates on the test set:
  - Image-level AUROC, AP, F1
  - Pixel-level AUROC and PRO score
  - Inference latency
  - Saves anomaly heatmap visualizations

Usage:
    python evaluate.py --model patchcore --category bottle
    python evaluate.py --model autoencoder --category tile --threshold 0.3
"""

import sys
import json
import time
import click
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import MVTecDataset, get_dataloader
from data.transforms import get_val_transforms
from models import build_model
from evaluation.metrics import full_evaluation_report
from uncertainty.conformal import ConformalPredictor
from localization.anomaly_map import AnomalyMapProcessor
from utils.visualization import plot_anomaly_result, save_inspection_report
from utils.logging import get_logger

logger = get_logger("evaluate")


@click.command()
@click.option("--model",     "-m", default="patchcore", show_default=True)
@click.option("--category",  "-c", default="bottle",    show_default=True)
@click.option("--data-root", default="data/raw/mvtec",  show_default=True)
@click.option("--model-dir", default="results/models",  show_default=True)
@click.option("--output",    default="results/eval",    show_default=True)
@click.option("--batch",     default=8,                 show_default=True)
@click.option("--workers",   default=4,                 show_default=True)
@click.option("--device",    default="cuda",            show_default=True)
@click.option("--threshold", default=None, type=float,  help="Override anomaly threshold")
@click.option("--save-visuals/--no-visuals", default=True)
@click.option("--conformal/--no-conformal", default=True, help="Use conformal prediction for routing")
def main(model, category, data_root, model_dir, output, batch, workers, device, threshold, save_visuals, conformal):
    """Evaluate a trained VisionInspect AI model."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output) / model / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    model_path = Path(model_dir) / model / category / "model.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(model_path)

    logger.info(f"Loading {model} from {model_path}")
    if model == "patchcore":
        from models.patchcore import PatchCore
        anomaly_model = PatchCore.load(str(model_path), device=str(device))
    elif model == "padim":
        from models.padim import PaDiM
        anomaly_model = PaDiM.load(str(model_path), device=str(device))
    else:
        anomaly_model = build_model(model, device=str(device))
        anomaly_model.load_state_dict(torch.load(str(model_path), map_location=device))
        anomaly_model.to(device).eval()

    # ---- Test dataset ----
    test_ds = MVTecDataset(
        root=data_root, category=category, split="test",
        transform=get_val_transforms(),
    )
    test_dl = get_dataloader(test_ds, batch_size=batch, shuffle=False, num_workers=workers)

    # ---- Optionally: calibration for conformal prediction ----
    # Use first 30% of test-normal images as calibration
    cal_scores, cal_labels = [], []
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []
    latencies = []
    visual_results = []

    processor = AnomalyMapProcessor(sigma=4.0, normalize=True)

    for batch_data in tqdm(test_dl, desc="Evaluating"):
        imgs = batch_data["image"].to(device)
        masks = batch_data["mask"]
        labels = batch_data["label"].numpy()

        t0 = time.perf_counter()
        with torch.no_grad():
            amap, score = anomaly_model.predict(imgs)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) / len(imgs) * 1000)

        scores_np = score.cpu().numpy()
        maps_np   = amap.squeeze(1).cpu().numpy()

        all_scores.extend(scores_np.tolist())
        all_labels.extend(labels.tolist())
        all_maps.append(maps_np)
        all_masks.append(masks.squeeze(1).numpy())

        # Collect visuals for the first 20 samples
        if save_visuals and len(visual_results) < 20:
            for i in range(len(imgs)):
                visual_results.append({
                    "image": imgs[i].cpu(),
                    "anomaly_map": amap[i].cpu(),
                    "gt_mask": masks[i],
                    "anomaly_score": float(scores_np[i]),
                    "decision": "unknown",
                    "image_path": batch_data["image_path"][i],
                })

    all_scores_arr = np.array(all_scores)
    all_labels_arr = np.array(all_labels)
    all_maps_arr   = np.concatenate(all_maps, axis=0)
    all_masks_arr  = np.concatenate(all_masks, axis=0)
    mean_latency   = float(np.mean(latencies))

    # ---- Conformal prediction ----
    if conformal:
        cp = ConformalPredictor(alpha_normal=0.05, alpha_anomaly=0.05)
        # Use first half as calibration
        n_cal = len(all_scores_arr) // 2
        cp.calibrate(all_scores_arr[:n_cal], all_labels_arr[:n_cal])
        decisions = cp.predict(all_scores_arr)
        review_rate = cp.review_rate(all_scores_arr)
        logger.info(f"Conformal: {cp.summary()}")
        logger.info(f"Human review rate: {review_rate:.1%}")

        # Update visual decisions
        for i, vr in enumerate(visual_results):
            vr["decision"] = decisions[i] if i < len(decisions) else "unknown"

    # ---- Compute metrics ----
    report = full_evaluation_report(
        image_labels=all_labels_arr,
        image_scores=all_scores_arr,
        gt_masks=all_masks_arr,
        anomaly_maps=all_maps_arr,
        latency_ms=mean_latency,
    )

    # ---- Print & save ----
    logger.info("\n" + "=" * 50)
    logger.info(f"  Evaluation: {model} @ {category}")
    logger.info("=" * 50)
    for k, v in report.items():
        logger.info(f"  {k:<25} {v:.4f}" if not isinstance(v, str) else f"  {k:<25} {v}")
    logger.info("=" * 50)

    report_path = output_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metrics saved to {report_path}")

    if save_visuals:
        save_inspection_report(visual_results, str(output_dir / "visualizations"))


if __name__ == "__main__":
    main()
