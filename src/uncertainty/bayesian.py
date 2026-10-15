"""
src/uncertainty/bayesian.py

MC-Dropout wrapper for uncertainty estimation in any PyTorch model.

Principle:
  - Enable dropout layers at test time (Monte Carlo Dropout).
  - Run T stochastic forward passes.
  - Uncertainty = variance of predicted anomaly scores across passes.

High-variance predictions are flagged for human review.
"""

from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn


def _enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers for MC inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()


class MCDropoutWrapper:
    """
    Wraps any anomaly detection model to add MC-Dropout uncertainty estimation.

    Args:
        model: An anomaly detection model with a `predict()` method.
        n_passes (int): Number of stochastic forward passes.
        dropout_p (float): Dropout probability injected if model has no Dropout.
    """

    def __init__(
        self,
        model: nn.Module,
        n_passes: int = 20,
        dropout_p: float = 0.1,
    ):
        self.model = model
        self.n_passes = n_passes
        self.dropout_p = dropout_p

        # Inject Dropout into model if none exist
        has_dropout = any(
            isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules()
        )
        if not has_dropout:
            self._inject_dropout(dropout_p)

    # ------------------------------------------------------------------
    def _inject_dropout(self, p: float) -> None:
        """Add dropout after each BatchNorm layer."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                parent = self.model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                original = getattr(parent, parts[-1])
                setattr(
                    parent,
                    parts[-1],
                    nn.Sequential(original, nn.Dropout2d(p=p)),
                )

    # ------------------------------------------------------------------
    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run T stochastic forward passes and compute mean and uncertainty.

        Returns:
            mean_map:    (B, 1, H, W) mean anomaly map
            mean_score:  (B,) mean image-level anomaly score
            uncertainty: (B,) uncertainty (std of scores across passes)
        """
        self.model.eval()
        _enable_dropout(self.model)

        maps_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.n_passes):
                amap, score = self.model.predict(x)
                maps_list.append(amap)
                scores_list.append(score)

        maps_tensor = torch.stack(maps_list, dim=0)    # (T, B, 1, H, W)
        scores_tensor = torch.stack(scores_list, dim=0)  # (T, B)

        mean_map = maps_tensor.mean(dim=0)
        mean_score = scores_tensor.mean(dim=0)
        uncertainty = scores_tensor.std(dim=0)

        return mean_map, mean_score, uncertainty

    # ------------------------------------------------------------------
    def should_review(
        self,
        uncertainty: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Returns a boolean mask indicating which images should be routed
        to human review based on uncertainty level.

        Args:
            uncertainty: (B,) uncertainty scores.
            threshold: Uncertainty threshold above which human review is triggered.

        Returns:
            (B,) BoolTensor.
        """
        return uncertainty > threshold
