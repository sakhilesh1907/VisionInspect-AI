"""
src/evaluation/metrics.py

Evaluation metrics for anomaly detection and localization:

  Image-level:
    - AUROC         — Area Under the ROC Curve
    - Average Precision (AP)
    - F1 at optimal threshold

  Pixel-level:
    - Pixel AUROC   — per-pixel anomaly map quality
    - PRO Score     — Per-Region Overlap (localisation quality)
    - Pixel AP

  Efficiency:
    - Inference latency (ms/image)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------

def compute_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Image-level AUROC.

    Args:
        labels: Binary ground-truth labels (0=normal, 1=anomaly). Shape (N,)
        scores: Anomaly scores (higher = more anomalous). Shape (N,)

    Returns:
        AUROC float in [0, 1].
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_ap(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Average precision (area under PR curve)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_f1(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[float, float]:
    """
    F1 score at optimal threshold (maximises F1).

    Returns:
        (f1, threshold)
    """
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        with np.errstate(invalid="ignore"):
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        idx = np.nanargmax(f1s)
        threshold = thresholds[min(idx, len(thresholds) - 1)]

    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0)), float(threshold)


def fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """False positive rate at a given true positive rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_target)
    return float(fpr[min(idx, len(fpr) - 1)])


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def compute_pixel_auroc(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> float:
    """
    Pixel-level AUROC.

    Args:
        gt_masks: Binary pixel masks. Shape (N, H, W).
        anomaly_maps: Per-pixel anomaly scores. Shape (N, H, W).

    Returns:
        Pixel-level AUROC.
    """
    flat_gt = gt_masks.ravel().astype(int)
    flat_scores = anomaly_maps.ravel()
    if len(np.unique(flat_gt)) < 2:
        return float("nan")
    return float(roc_auc_score(flat_gt, flat_scores))


def compute_pro_score(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    num_thresh: int = 100,
    max_fpr: float = 0.3,
) -> float:
    """
    Per-Region Overlap (PRO) score.

    Computes the average per-connected-component overlap between predicted
    anomaly maps (thresholded) and ground-truth regions, integrated over
    false positive rates up to max_fpr.

    Args:
        gt_masks: (N, H, W) binary masks.
        anomaly_maps: (N, H, W) float anomaly maps.
        num_thresh: Number of threshold steps.
        max_fpr: Maximum acceptable false positive rate for integration.

    Returns:
        PRO score in [0, 1].
    """
    from scipy import ndimage

    thresholds = np.linspace(anomaly_maps.min(), anomaly_maps.max(), num_thresh)
    pro_values: List[float] = []
    fpr_values: List[float] = []

    n_normal_pixels = (gt_masks == 0).sum()

    for thresh in thresholds:
        pred_masks = (anomaly_maps >= thresh).astype(np.uint8)
        pro_list: List[float] = []

        for gt, pred in zip(gt_masks, pred_masks):
            if gt.max() == 0:
                continue
            labeled, n_comp = ndimage.label(gt)
            for comp_id in range(1, n_comp + 1):
                region = labeled == comp_id
                overlap = (pred[region]).sum() / region.sum()
                pro_list.append(float(overlap))

        fpr = ((pred_masks == 1) & (gt_masks == 0)).sum() / (n_normal_pixels + 1e-8)
        pro = float(np.mean(pro_list)) if pro_list else 0.0
        pro_values.append(pro)
        fpr_values.append(float(fpr))

    fpr_arr = np.array(fpr_values)
    pro_arr = np.array(pro_values)

    # Keep only points within acceptable FPR range
    mask = fpr_arr <= max_fpr
    if mask.sum() < 2:
        return float("nan")

    # Normalised area under PRO-FPR curve
    norm_fpr = fpr_arr[mask] / max_fpr
    pro_score = float(np.trapz(pro_arr[mask], norm_fpr))
    return pro_score


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    gt_masks: Optional[np.ndarray] = None,
    anomaly_maps: Optional[np.ndarray] = None,
    latency_ms: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all relevant metrics and return as a dict.

    Args:
        image_labels: (N,) binary labels.
        image_scores: (N,) image anomaly scores.
        gt_masks: (N, H, W) pixel-level GT masks (optional).
        anomaly_maps: (N, H, W) pixel-level anomaly maps (optional).
        latency_ms: Average inference latency per image in milliseconds.

    Returns:
        dict with keys: auroc, ap, f1, fpr95, pixel_auroc, pro, latency_ms
    """
    report: Dict[str, float] = {}

    # Image-level
    report["image_auroc"] = compute_auroc(image_labels, image_scores)
    report["image_ap"] = compute_ap(image_labels, image_scores)
    f1, thresh = compute_f1(image_labels, image_scores)
    report["image_f1"] = f1
    report["optimal_threshold"] = thresh
    report["fpr_at_95tpr"] = fpr_at_tpr(image_labels, image_scores, 0.95)

    # Pixel-level (if available)
    if gt_masks is not None and anomaly_maps is not None:
        report["pixel_auroc"] = compute_pixel_auroc(gt_masks, anomaly_maps)
        report["pro_score"] = compute_pro_score(gt_masks, anomaly_maps)
    else:
        report["pixel_auroc"] = float("nan")
        report["pro_score"] = float("nan")

    # Latency
    if latency_ms is not None:
        report["latency_ms"] = latency_ms

    return report
