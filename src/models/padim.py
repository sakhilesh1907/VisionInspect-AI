"""
src/models/padim.py

PaDiM — Patch Distribution Modeling for Anomaly Detection & Localization.

Reference:
  Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
  Detection and Localization", ICPR 2021.  arXiv:2011.08785

Principle:
  1. Extract multi-scale patch embeddings from a pretrained CNN at every spatial position.
  2. Fit a multivariate Gaussian at each spatial position using only NORMAL training images.
  3. At test time, compute the Mahalanobis distance at each position as the anomaly score.

Key properties:
  - No training of neural-network weights (feature extractor is frozen).
  - Excellent localization via pixel-level Mahalanobis maps.
  - Scales to large backbones (WideResNet, EfficientNet, etc.).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    Extracts hierarchical feature maps from a pretrained backbone at
    specified layer indices and concatenates them along the channel axis.

    Supported backbones: any timm model.
    Default layers: layer1, layer2, layer3 of a ResNet-like architecture.
    """

    LAYER_NAMES = {
        "resnet18": ["layer1", "layer2", "layer3"],
        "wide_resnet50_2": ["layer1", "layer2", "layer3"],
        "efficientnet_b4": ["blocks.1", "blocks.2", "blocks.4"],
    }

    def __init__(self, backbone: str = "resnet18", layer_names: Optional[List[str]] = None, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone
        self.layer_names = layer_names or self.LAYER_NAMES.get(backbone, ["layer1", "layer2", "layer3"])

        try:
            self.model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        except Exception:
            self.model = timm.create_model(backbone, pretrained=False, features_only=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps at the specified layer indices."""
        all_features = self.model(x)
        # timm features_only returns features at multiple scales
        # We use the first len(layer_names) scales
        n = min(len(self.layer_names), len(all_features))
        return list(all_features[:n])


# ---------------------------------------------------------------------------
# PaDiM Model
# ---------------------------------------------------------------------------

class PaDiM(nn.Module):
    """
    PaDiM: Patch Distribution Modeling.

    Call `fit(dataloader)` on normal training images before inference.
    Call `predict(x)` to get anomaly maps and scores.

    Args:
        backbone (str): timm backbone identifier.
        d_reduced (int): Random projection dimensionality (-1 = no reduction).
        device (str): "cuda" or "cpu".
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        d_reduced: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.d_reduced = d_reduced

        self.extractor = FeatureExtractor(backbone).to(self.device)

        # These are set during fit()
        self._mean: Optional[torch.Tensor] = None    # (H*W, d)
        self._cov_inv: Optional[torch.Tensor] = None  # (H*W, d, d)
        self._feat_dim: Optional[int] = None
        self._spatial_shape: Optional[Tuple[int, int]] = None
        self._rnd_idx: Optional[torch.Tensor] = None  # random projection indices

    # ------------------------------------------------------------------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and concatenate multi-scale features, upsampled to the
        resolution of the first (largest) feature map.

        Returns: (B, D, H_f, W_f) where D is the total channel count.
        """
        feats = self.extractor(x)
        target_h, target_w = feats[0].shape[-2], feats[0].shape[-1]

        aligned = [feats[0]]
        for f in feats[1:]:
            aligned.append(
                F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            )
        return torch.cat(aligned, dim=1)  # (B, D, H_f, W_f)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, dataloader, verbose: bool = True) -> None:
        """
        Compute patch-level multivariate Gaussian parameters from normal images.

        Args:
            dataloader: Yields batches with 'image' key (B, 3, H, W).
            verbose: Show progress bar.
        """
        all_feats: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="PaDiM fitting", disable=not verbose):
            images = batch["image"].to(self.device)
            feat = self._extract_features(images)  # (B, D, H_f, W_f)
            all_feats.append(feat.cpu())

        feats = torch.cat(all_feats, dim=0)  # (N, D, H_f, W_f)
        N, D, H_f, W_f = feats.shape

        # Optional: random dimensionality reduction
        if self.d_reduced > 0 and self.d_reduced < D:
            torch.manual_seed(42)
            self._rnd_idx = torch.randperm(D)[: self.d_reduced]
            feats = feats[:, self._rnd_idx]
            D = self.d_reduced
        else:
            self._rnd_idx = None

        self._feat_dim = D
        self._spatial_shape = (H_f, W_f)
        n_patches = H_f * W_f

        # Reshape → (N, D, n_patches)
        feats = feats.view(N, D, n_patches)  # (N, D, P)

        # Mean per patch position: (P, D)
        mean = feats.mean(dim=0).T  # (P, D)

        # Covariance per patch position: (P, D, D) memory-efficient chunked computation
        feats_c = feats - feats.mean(dim=0, keepdim=True)  # centred
        cov = torch.empty((n_patches, D, D), dtype=torch.float32)
        reg = 0.01 * torch.eye(D)
        for p in range(n_patches):
            fp = feats_c[:, :, p]  # (N, D)
            cov[p] = (fp.T @ fp) / max(1, N - 1) + reg

        # Invert covariance matrices in batch
        cov_inv = torch.linalg.inv(cov)  # (P, D, D)

        self._mean = mean       # (P, D)
        self._cov_inv = cov_inv  # (P, D, D)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience forward pass that delegates to predict(x)."""
        return self.predict(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly maps and image-level anomaly scores.

        Args:
            x: (B, 3, H, W) input images (normalised).

        Returns:
            anomaly_map: (B, 1, H, W) upsampled Mahalanobis distance map.
            anomaly_score: (B,) image-level anomaly score (max of map).
        """
        assert self._mean is not None, "Call fit() before predict()"
        B = x.shape[0]
        H_in, W_in = x.shape[-2], x.shape[-1]
        H_f, W_f = self._spatial_shape
        P = H_f * W_f

        feat = self._extract_features(x)      # (B, D, H_f, W_f)
        if self._rnd_idx is not None:
            feat = feat[:, self._rnd_idx]

        feat = feat.view(B, self._feat_dim, P).permute(0, 2, 1)  # (B, P, D)

        mean = self._mean.to(x.device)       # (P, D)
        cov_inv = self._cov_inv.to(x.device) # (P, D, D)

        # Mahalanobis distance at every patch position
        diff = feat - mean.unsqueeze(0)  # (B, P, D)

        # dist² = diff @ cov_inv @ diff^T  (per patch, per image)
        left = torch.einsum("bpd,pde->bpe", diff, cov_inv)   # (B, P, D)
        dist_sq = (left * diff).sum(dim=-1)                   # (B, P)
        dist = dist_sq.clamp(min=0).sqrt()                    # (B, P)

        # Reshape to spatial map and upsample
        dist_map = dist.view(B, 1, H_f, W_f)
        anomaly_map = F.interpolate(
            dist_map, size=(H_in, W_in), mode="bilinear", align_corners=False
        )

        anomaly_score = anomaly_map.view(B, -1).max(dim=1).values
        return anomaly_map, anomaly_score

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist fitted Gaussian parameters."""
        torch.save(
            {
                "mean": self._mean,
                "cov_inv": self._cov_inv,
                "rnd_idx": self._rnd_idx,
                "feat_dim": self._feat_dim,
                "spatial_shape": self._spatial_shape,
                "backbone": self.extractor.backbone_name,
                "d_reduced": self.d_reduced,
                "extractor_state": self.extractor.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PaDiM":
        """Load a fitted PaDiM model from disk."""
        state = torch.load(path, map_location="cpu")
        model = cls(backbone=state["backbone"], d_reduced=state["d_reduced"], device=device)
        model._mean = state["mean"]
        model._cov_inv = state["cov_inv"]
        model._rnd_idx = state["rnd_idx"]
        model._feat_dim = state["feat_dim"]
        model._spatial_shape = state["spatial_shape"]
        if "extractor_state" in state:
            model.extractor.load_state_dict(state["extractor_state"])
        return model
