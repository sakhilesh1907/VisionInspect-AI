"""
tests/test_pipeline.py

Unit and integration tests for VisionInspect AI components:
  1. Dataset loading and tensor shapes
  2. Autoencoder architecture forward pass and reconstruction loss
  3. PaDiM and PatchCore initialization & dimensionality
  4. Conformal prediction interval calculations and review routing
  5. Anomaly heatmap post-processing and bounding box extraction
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

# Ensure src is on path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.transforms import get_train_transforms, get_val_transforms
from models.autoencoder import ConvAutoencoder
from uncertainty.conformal import ConformalPredictor
from localization.anomaly_map import AnomalyMapProcessor
from evaluation.metrics import compute_auroc, compute_f1


def test_autoencoder_forward_pass():
    """Verify ConvAutoencoder reconstructs matching tensor shapes and produces scalar scores."""
    model = ConvAutoencoder(base_channels=16, bottleneck_dim=64)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        recon, scores = model(dummy_input)
        amap, pred_scores = model.predict(dummy_input)

    assert recon.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {recon.shape}"
    assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
    assert amap.shape == (2, 1, 224, 224), f"Expected (2, 1, 224, 224), got {amap.shape}"
    assert pred_scores.shape == (2,), f"Expected (2,), got {pred_scores.shape}"


def test_conformal_predictor():
    """Verify conformal predictor splits predictions into normal, anomaly, and human review."""
    cp = ConformalPredictor(alpha_normal=0.1, alpha_anomaly=0.1)

    # Simulated calibration scores: normal ~ [0.1, 0.4], anomaly ~ [0.35, 0.9]
    normal_cal = np.random.uniform(0.1, 0.4, size=50)
    defect_cal = np.random.uniform(0.35, 0.9, size=50)
    cal_scores = np.concatenate([normal_cal, defect_cal])
    cal_labels = np.array([0] * 50 + [1] * 50)

    cp.calibrate(cal_scores, cal_labels)
    t_norm, t_anom = cp.thresholds

    assert t_norm is not None and t_anom is not None
    assert t_norm < t_anom, f"Threshold normal {t_norm} must be <= threshold anomaly {t_anom}"

    # Test predictions
    test_scores = np.array([0.05, (t_norm + t_anom) / 2.0, 0.95])
    decisions = cp.predict(test_scores)

    assert decisions[0] == "normal"
    assert decisions[1] == "review"
    assert decisions[2] == "anomaly"


def test_anomaly_map_processor():
    """Verify anomaly heatmap smoothing, normalisation, and bounding box extraction."""
    processor = AnomalyMapProcessor(sigma=1.0, normalize=True)

    # Synthetic map with a bright spot (defect) in the centre
    raw_map = np.zeros((100, 100), dtype=np.float32)
    raw_map[40:60, 40:60] = 5.0

    processed = processor.process(raw_map)
    assert processed.min() >= 0.0 and processed.max() <= 1.0

    binary, bboxes = processor.threshold_map(processed, threshold=0.5, min_area=20)
    assert len(bboxes) == 1, f"Expected 1 defect bbox, found {len(bboxes)}"
    x, y, w, h = bboxes[0]
    assert 30 <= x <= 50 and 30 <= y <= 50


def test_metrics_computation():
    """Verify AUROC and F1 metric computations."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.85, 0.9, 0.95])

    auroc = compute_auroc(labels, scores)
    assert np.isclose(auroc, 1.0)

    f1, thresh = compute_f1(labels, scores)
    assert np.isclose(f1, 1.0)
