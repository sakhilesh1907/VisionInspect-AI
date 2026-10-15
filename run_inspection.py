"""
inspect.py — VisionInspect AI Single-Image Inference Script

Runs inference on a single image or directory of images and outputs:
  - Anomaly score
  - Decision: normal / anomaly / review
  - Heatmap image saved to disk

Usage:
    python inspect.py --image path/to/product.jpg --model patchcore --category bottle
    python inspect.py --dir path/to/images/ --model padim --category tile --output results/inspection
"""

import sys
import json
import click
import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.transforms import get_val_transforms, denormalize
from models import build_model
from localization.anomaly_map import AnomalyMapProcessor
from utils.visualization import plot_anomaly_result
from utils.logging import get_logger

logger = get_logger("inspect")


def load_image(image_path: str, transform) -> torch.Tensor:
    """Load a single image and apply transforms. Returns (1, 3, H, W)."""
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


@click.command()
@click.option("--image",     default=None,              help="Path to single image")
@click.option("--dir",       "image_dir", default=None, help="Path to directory of images")
@click.option("--model",     "-m", default="patchcore", show_default=True)
@click.option("--category",  "-c", default="bottle",    show_default=True)
@click.option("--model-dir", default="results/models",  show_default=True)
@click.option("--output",    default="results/inspection", show_default=True)
@click.option("--threshold", default=0.5,               show_default=True, help="Anomaly score threshold")
@click.option("--device",    default="cuda",            show_default=True)
@click.option("--save-heatmap/--no-heatmap", default=True)
def main(image, image_dir, model, category, model_dir, output, threshold, device, save_heatmap):
    """Run VisionInspect AI inference on one or more images."""

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transforms()
    processor = AnomalyMapProcessor(sigma=4.0, normalize=True)

    # ---- Load model ----
    model_path = Path(model_dir) / model / category / "model.pt"
    logger.info(f"Loading {model} from {model_path}")

    if model == "patchcore":
        from models.patchcore import PatchCore
        anomaly_model = PatchCore.load(str(model_path), device=str(device))
    elif model == "padim":
        from models.padim import PaDiM
        anomaly_model = PaDiM.load(str(model_path), device=str(device))
    else:
        anomaly_model = build_model(model, device=str(device))
        state = torch.load(str(model_path), map_location=device)
        anomaly_model.load_state_dict(state)
        anomaly_model.to(device).eval()

    # ---- Collect image paths ----
    image_paths = []
    if image:
        image_paths.append(Path(image))
    elif image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts]
    else:
        logger.error("Provide --image or --dir")
        return

    logger.info(f"Inspecting {len(image_paths)} image(s)...")
    results = []

    for img_path in image_paths:
        try:
            img_tensor = load_image(str(img_path), transform).to(device)

            with torch.no_grad():
                amap, score = anomaly_model.predict(img_tensor)

            score_val = float(score[0].cpu())
            decision = "anomaly" if score_val >= threshold else "normal"

            result = {
                "image_path": str(img_path),
                "anomaly_score": round(score_val, 6),
                "decision": decision,
                "threshold": threshold,
            }
            results.append(result)

            logger.info(f"  [{decision.upper():7s}] {img_path.name}  score={score_val:.4f}")

            # Save heatmap
            if save_heatmap:
                fig = plot_anomaly_result(
                    image=img_tensor[0].cpu(),
                    anomaly_map=amap[0].cpu(),
                    anomaly_score=score_val,
                    decision=decision,
                    save_path=str(output_dir / f"{img_path.stem}_heatmap.png"),
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")
            results.append({"image_path": str(img_path), "error": str(e)})

    # Save JSON report
    report_path = output_dir / "inspection_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {report_path}")

    # Summary
    n_anomaly = sum(1 for r in results if r.get("decision") == "anomaly")
    n_normal  = sum(1 for r in results if r.get("decision") == "normal")
    logger.info(f"Summary: {n_anomaly} anomaly | {n_normal} normal | total {len(results)}")


if __name__ == "__main__":
    main()
