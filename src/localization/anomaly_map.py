"""
src/localization/anomaly_map.py

Post-processing utilities for anomaly maps:
  - Gaussian smoothing
  - Normalisation to [0, 1]
  - Overlay visualisation on original image
  - Binary mask generation with connected-component analysis
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class AnomalyMapProcessor:
    """
    Post-process raw anomaly maps from any detection model.

    Args:
        sigma (float): Gaussian smoothing sigma. 0 = no smoothing.
        normalize (bool): Normalise map to [0, 1] per image.
    """

    def __init__(self, sigma: float = 4.0, normalize: bool = True):
        self.sigma = sigma
        self.normalize = normalize

    # ------------------------------------------------------------------
    def process(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        Smooth and normalise a single anomaly map.

        Args:
            anomaly_map: (H, W) float array.

        Returns:
            Processed map (H, W) in [0, 1].
        """
        if self.sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=self.sigma)

        if self.normalize:
            vmin, vmax = anomaly_map.min(), anomaly_map.max()
            if vmax > vmin:
                anomaly_map = (anomaly_map - vmin) / (vmax - vmin)
            else:
                anomaly_map = np.zeros_like(anomaly_map)

        return anomaly_map

    # ------------------------------------------------------------------
    def overlay(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the anomaly heatmap on the original image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            anomaly_map: (H, W) float map, values in [0, 1].
            alpha: Heatmap blending weight.
            colormap: OpenCV colourmap (default: JET).

        Returns:
            (H, W, 3) uint8 blended image.
        """
        heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = ((1 - alpha) * image + alpha * heatmap).clip(0, 255).astype(np.uint8)
        return overlay

    # ------------------------------------------------------------------
    def threshold_map(
        self,
        anomaly_map: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 50,
    ) -> Tuple[np.ndarray, list]:
        """
        Binarise the anomaly map and extract connected components.

        Args:
            anomaly_map: (H, W) float map.
            threshold: Binarisation threshold.
            min_area: Minimum defect region area in pixels.

        Returns:
            binary_mask: (H, W) uint8 binary mask.
            bboxes: List of (x, y, w, h) bounding boxes for detected regions.
        """
        binary = (anomaly_map >= threshold).astype(np.uint8)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Connected components
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        bboxes = []
        for i in range(1, n_comp):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                bboxes.append((x, y, w, h))

        # Keep only large-enough regions
        if not bboxes:
            binary = np.zeros_like(binary)

        return binary, bboxes

    # ------------------------------------------------------------------
    def draw_bboxes(
        self,
        image: np.ndarray,
        bboxes: list,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes on an image."""
        img_out = image.copy()
        for x, y, w, h in bboxes:
            cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
        return img_out
