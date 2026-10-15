"""
src/uncertainty/conformal.py

Conformal Prediction for Anomaly Detection.

Conformal prediction provides guaranteed coverage: with a calibration set,
we compute a threshold T such that P(score >= T for anomalies) >= 1 - alpha.

This gives statistically valid prediction sets:
  - Images with score > T_high   → classified as anomaly
  - Images with score < T_low    → classified as normal
  - Images in the interval       → uncertain → human review queue

Reference:
  Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification", 2021. arXiv:2107.07511
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch


class ConformalPredictor:
    """
    Split conformal predictor for binary anomaly detection.

    Usage::
        cp = ConformalPredictor(alpha_normal=0.05, alpha_anomaly=0.05)
        cp.calibrate(cal_scores, cal_labels)
        decisions = cp.predict(test_scores)  # "normal", "anomaly", or "review"
    """

    def __init__(
        self,
        alpha_normal: float = 0.05,
        alpha_anomaly: float = 0.05,
    ):
        """
        Args:
            alpha_normal: Allowable false-negative rate (missed defects).
            alpha_anomaly: Allowable false-positive rate (false alarms).
        """
        self.alpha_normal = alpha_normal
        self.alpha_anomaly = alpha_anomaly

        self._threshold_normal: Optional[float] = None   # below → certainly normal
        self._threshold_anomaly: Optional[float] = None  # above → certainly anomaly

    # ------------------------------------------------------------------
    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None:
        """
        Compute calibration thresholds from a held-out calibration set.

        Args:
            cal_scores: Anomaly scores (N,).
            cal_labels: Binary labels — 0=normal, 1=anomaly (N,).
        """
        normal_scores = cal_scores[cal_labels == 0]
        anomaly_scores = cal_scores[cal_labels == 1]

        # Threshold below which normal is guaranteed with probability 1-alpha_normal
        # (high quantile of normal scores)
        if len(normal_scores) > 0:
            self._threshold_normal = float(
                np.quantile(normal_scores, 1.0 - self.alpha_normal)
            )
        else:
            self._threshold_normal = 0.0

        # Threshold above which anomaly is guaranteed with probability 1-alpha_anomaly
        # (low quantile of anomaly scores)
        if len(anomaly_scores) > 0:
            self._threshold_anomaly = float(
                np.quantile(anomaly_scores, self.alpha_anomaly)
            )
        else:
            self._threshold_anomaly = float("inf")

        # Ensure ordering: if thresholds overlap, widen the review band
        if self._threshold_anomaly <= self._threshold_normal:
            mid = (self._threshold_normal + self._threshold_anomaly) / 2
            self._threshold_normal = mid * 0.95
            self._threshold_anomaly = mid * 1.05

    # ------------------------------------------------------------------
    def predict(
        self, scores: np.ndarray
    ) -> np.ndarray:
        """
        Classify each score as 'normal', 'anomaly', or 'review'.

        Returns:
            String array of shape (N,).
        """
        assert self._threshold_normal is not None, "Call calibrate() first"
        decisions = np.empty(len(scores), dtype=object)
        decisions[scores < self._threshold_normal] = "normal"
        decisions[scores >= self._threshold_anomaly] = "anomaly"
        decisions[
            (scores >= self._threshold_normal) & (scores < self._threshold_anomaly)
        ] = "review"
        return decisions

    # ------------------------------------------------------------------
    def review_rate(self, scores: np.ndarray) -> float:
        """Fraction of samples that would be routed to human review."""
        decisions = self.predict(scores)
        return float((decisions == "review").mean())

    # ------------------------------------------------------------------
    @property
    def thresholds(self) -> Tuple[Optional[float], Optional[float]]:
        """(threshold_normal, threshold_anomaly)"""
        return self._threshold_normal, self._threshold_anomaly

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lo, hi = self.thresholds
        return (
            f"ConformalPredictor("
            f"alpha_n={self.alpha_normal}, alpha_a={self.alpha_anomaly}, "
            f"T_normal={lo:.4f}, T_anomaly={hi:.4f})"
        )
