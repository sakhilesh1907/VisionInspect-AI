"""
src/models/patchcore.py

PatchCore — Memory Bank Anomaly Detection.

Reference:
  Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
  CVPR 2022.  arXiv:2106.08265

Principle:
  1. Extract patch-level features from a pretrained backbone (WideResNet50).
  2. Build a coreset memory bank from normal training patches
     (greedy subsampling to fill a budget).
  3. At test time, compute k-NN distance to the memory bank as the anomaly score.
  4. Localise defects by reshaping patch distances to a spatial heatmap.

Key properties:
  - State-of-the-art image-level AUROC on MVTec AD.
  - No gradient computation or model fine-tuning.
  - Memory bank can be compressed with faiss for efficient retrieval.
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
# Adaptive Average Pooling for patch aggregation
# ---------------------------------------------------------------------------

class PatchAggregator(nn.Module):
    """
    Aggregate spatial features into non-overlapping neighbourhood patches.
    Mimics the AvgPool used in the original PatchCore paper.
    """

    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


# ---------------------------------------------------------------------------
# Feature Extractor (same as PaDiM but with patch aggregation)
# ---------------------------------------------------------------------------

class PatchCoreExtractor(nn.Module):
    """
    Extracts multi-layer features from a pretrained backbone and
    applies local average pooling for patch neighbourhood context.
    """

    def __init__(self, backbone: str = "resnet18", layers: List[int] = [2, 3], pretrained: bool = False):
        super().__init__()
        try:
            self.model = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        except Exception:
            self.model = timm.create_model(backbone, pretrained=False, features_only=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.layers = layers
        self.aggregator = PatchAggregator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, D, H_f, W_f) concatenated patch embeddings."""
        all_feats = self.model(x)
        selected = [all_feats[i] for i in self.layers]

        # Upsample everything to the resolution of the first selected layer
        target_h, target_w = selected[0].shape[-2], selected[0].shape[-1]
        aligned = []
        for f in selected:
            if f.shape[-2] != target_h or f.shape[-1] != target_w:
                f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(self.aggregator(f))

        return torch.cat(aligned, dim=1)  # (B, D, H_f, W_f)


# ---------------------------------------------------------------------------
# Greedy Coreset Subsampling
# ---------------------------------------------------------------------------

def greedy_coreset(
    embeddings: torch.Tensor,
    coreset_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Greedy approximation to the minimax facility-location coreset.
    Iteratively selects the point with the maximum min-distance to
    the already-selected set.

    Args:
        embeddings: (N, D) normalised patch embeddings.
        coreset_size: Target number of coreset points.
        seed: Random seed for initial point selection.

    Returns:
        indices: (coreset_size,) LongTensor of selected indices.
    """
    N = embeddings.shape[0]
    coreset_size = min(coreset_size, N)

    torch.manual_seed(seed)
    start = torch.randint(0, N, (1,)).item()
    selected = [start]
    min_dists = torch.full((N,), fill_value=float("inf"), device=embeddings.device)

    for _ in tqdm(range(coreset_size - 1), desc="Coreset subsampling", leave=False):
        last = embeddings[selected[-1]].unsqueeze(0)  # (1, D)
        dists = torch.cdist(embeddings, last).squeeze(1)  # (N,)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(min_dists.argmax().item())

    return torch.tensor(selected, dtype=torch.long)


# ---------------------------------------------------------------------------
# PatchCore
# ---------------------------------------------------------------------------

class PatchCore(nn.Module):
    """
    PatchCore anomaly detection model.

    Usage::
        model = PatchCore(backbone="wide_resnet50_2", coreset_ratio=0.1)
        model.fit(train_dataloader)
        anomaly_map, score = model.predict(test_image_batch)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: List[int] = [2, 3],
        coreset_ratio: float = 0.1,
        k_neighbours: int = 9,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.coreset_ratio = coreset_ratio
        self.k = k_neighbours

        self.extractor = PatchCoreExtractor(backbone, layers).to(self.device)

        self._memory_bank: Optional[torch.Tensor] = None  # (M, D)
        self._spatial_shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, dataloader, verbose: bool = True) -> None:
        """
        Build memory bank from all normal training patches.

        Args:
            dataloader: yields batches with 'image' key (B, 3, H, W).
        """
        all_patches: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc="PatchCore: extracting features", disable=not verbose):
            imgs = batch["image"].to(self.device)
            feats = self.extractor(imgs)           # (B, D, H_f, W_f)
            B, D, H_f, W_f = feats.shape

            if self._spatial_shape is None:
                self._spatial_shape = (H_f, W_f)

            patches = feats.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H_f*W_f, D)
            all_patches.append(patches.cpu())

        bank = torch.cat(all_patches, dim=0)  # (N, D)

        # L2-normalise
        bank = F.normalize(bank, dim=1)

        # Greedy coreset subsampling
        target_size = max(1, int(len(bank) * self.coreset_ratio))
        idx = greedy_coreset(bank, target_size)
        self._memory_bank = bank[idx]  # (M, D)

        if verbose:
            print(f"[PatchCore] Memory bank size: {len(self._memory_bank):,} / {len(bank):,}")

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
        Args:
            x: (B, 3, H, W) normalised input images.

        Returns:
            anomaly_map: (B, 1, H, W) heatmap.
            anomaly_score: (B,) image-level score.
        """
        assert self._memory_bank is not None, "Call fit() first"

        B = x.shape[0]
        H_in, W_in = x.shape[-2], x.shape[-1]
        H_f, W_f = self._spatial_shape

        feats = self.extractor(x)               # (B, D, H_f, W_f)
        D = feats.shape[1]
        patches = feats.permute(0, 2, 3, 1).reshape(B, H_f * W_f, D)  # (B, P, D)
        patches = F.normalize(patches, dim=-1)

        bank = self._memory_bank.to(x.device)   # (M, D)

        # KNN distances: (B, P, M)
        # Process in chunks to manage memory
        scores_list = []
        for i in range(B):
            dists = torch.cdist(patches[i], bank, p=2)   # (P, M)
            knn_dists, _ = dists.topk(self.k, dim=1, largest=False)  # (P, k)
            patch_score = knn_dists[:, 0]                 # nearest neighbour distance (P,)
            scores_list.append(patch_score)

        patch_scores = torch.stack(scores_list, dim=0)   # (B, P)

        dist_map = patch_scores.view(B, 1, H_f, W_f)
        anomaly_map = F.interpolate(
            dist_map, size=(H_in, W_in), mode="bilinear", align_corners=False
        )

        anomaly_score = anomaly_map.view(B, -1).max(dim=1).values
        return anomaly_map, anomaly_score

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "memory_bank": self._memory_bank,
                "spatial_shape": self._spatial_shape,
                "coreset_ratio": self.coreset_ratio,
                "k": self.k,
                "extractor_state": self.extractor.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PatchCore":
        state = torch.load(path, map_location="cpu")
        model = cls(coreset_ratio=state["coreset_ratio"], k_neighbours=state["k"], device=device)
        model._memory_bank = state["memory_bank"]
        model._spatial_shape = state["spatial_shape"]
        if "extractor_state" in state:
            model.extractor.load_state_dict(state["extractor_state"])
        return model

