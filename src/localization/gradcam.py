"""
src/localization/gradcam.py

Gradient-weighted Class Activation Mapping (Grad-CAM) localiser.

Grad-CAM can be applied to the reconstruction-based Autoencoder to highlight
which spatial regions contribute most to the anomaly score.

For embedding-based models (PaDiM, PatchCore), use anomaly_map.py instead.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMLocalizer:
    """
    Grad-CAM localisation for PyTorch models.

    Registers forward and backward hooks on a target layer to capture
    feature maps and gradients, then computes a weighted combination
    as the class-discriminative localisation map.

    Args:
        model: PyTorch model with a `predict()` or custom forward pass.
        target_layer: The nn.Module layer to hook (e.g. model.encoder.net[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    def generate(
        self,
        image: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            image: (1, 3, H, W) input tensor (requires_grad not needed).
            target_size: Output (H, W). Defaults to input spatial size.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward pass — use reconstruction error as pseudo-class score
        if hasattr(self.model, "forward"):
            try:
                recon, score = self.model(image)
                loss = score.sum()
            except Exception:
                output = self.model(image)
                loss = output.sum()
        else:
            raise RuntimeError("Model must have a forward() method")

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Grad-CAM weights
        gradients = self._gradients   # (1, C, H_f, W_f)
        activations = self._activations  # (1, C, H_f, W_f)

        if gradients is None or activations is None:
            raise RuntimeError("Hooks did not capture data — check target layer")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H_f, W_f)
        cam = F.relu(cam)

        # Upsample to target size
        if target_size is None:
            target_size = image.shape[-2:]
        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().detach().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    # ------------------------------------------------------------------
    def remove_hooks(self) -> None:
        """Deregister hooks when no longer needed."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
