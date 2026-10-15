"""
src/utils/visualization.py

Visualisation utilities for anomaly detection results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import torch

try:
    from data.transforms import denormalize
except ImportError:
    from ..data.transforms import denormalize


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) normalised tensor to (H, W, 3) uint8 numpy array."""
    img = denormalize(t).permute(1, 2, 0).cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def plot_anomaly_result(
    image: torch.Tensor,
    anomaly_map: torch.Tensor,
    gt_mask: Optional[torch.Tensor] = None,
    anomaly_score: Optional[float] = None,
    decision: str = "unknown",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a 4-panel visualisation:
      [Original] [Anomaly Heatmap] [Overlay] [GT Mask (if available)]

    Args:
        image: (3, H, W) normalised input tensor.
        anomaly_map: (1, H, W) or (H, W) anomaly map tensor.
        gt_mask: (1, H, W) or (H, W) binary ground-truth mask (optional).
        anomaly_score: Scalar anomaly score.
        decision: "normal" | "anomaly" | "review".
        save_path: If set, save figure to this path.
        show: If True, call plt.show().

    Returns:
        matplotlib Figure.
    """
    img_np = tensor_to_numpy(image)
    H, W = img_np.shape[:2]

    amap = anomaly_map.squeeze().cpu().numpy()
    amap = cv2.resize(amap, (W, H))
    # Normalise map
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

    # Heatmap
    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = ((0.6 * img_np) + (0.4 * heatmap)).clip(0, 255).astype(np.uint8)

    # Colour coding for decision
    colour_map = {"normal": "green", "anomaly": "red", "review": "orange", "unknown": "gray"}
    title_colour = colour_map.get(decision, "gray")

    ncols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4))

    axes[0].imshow(img_np); axes[0].set_title("Input Image"); axes[0].axis("off")
    axes[1].imshow(amap_norm, cmap="jet"); axes[1].set_title("Anomaly Map"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].axis("off")
    axes[2].set_title(
        f"Overlay\nScore: {anomaly_score:.4f}" if anomaly_score is not None else "Overlay"
    )

    if gt_mask is not None:
        mask_np = gt_mask.squeeze().cpu().numpy()
        mask_np = cv2.resize(mask_np, (W, H))
        axes[3].imshow(mask_np, cmap="gray"); axes[3].set_title("GT Mask"); axes[3].axis("off")

    fig.suptitle(
        f"Decision: {decision.upper()}",
        fontsize=14,
        fontweight="bold",
        color=title_colour,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def save_inspection_report(
    results: List[dict],
    output_dir: str,
    max_examples: int = 20,
) -> None:
    """
    Save a batch inspection report: individual plots + summary grid.

    Args:
        results: List of dicts with keys: image, anomaly_map, gt_mask,
                 anomaly_score, decision, image_path.
        output_dir: Where to save reports.
        max_examples: Maximum number of individual images to save.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results[:max_examples]):
        stem = Path(r.get("image_path", f"sample_{i}")).stem
        save_path = str(out / f"{stem}_{r.get('decision', 'unk')}.png")
        plot_anomaly_result(
            image=r["image"],
            anomaly_map=r["anomaly_map"],
            gt_mask=r.get("gt_mask"),
            anomaly_score=r.get("anomaly_score"),
            decision=r.get("decision", "unknown"),
            save_path=save_path,
        )
        plt.close("all")

    # Summary bar chart
    decisions = [r.get("decision", "unknown") for r in results]
    labels, counts = np.unique(decisions, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        labels, counts,
        color=["green" if l == "normal" else "red" if l == "anomaly" else "orange" for l in labels]
    )
    ax.bar_label(bars)
    ax.set_title("Inspection Summary")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(str(out / "summary.png"), dpi=150)
    plt.close(fig)
    print(f"[VisionInspect] Report saved to {out}")
