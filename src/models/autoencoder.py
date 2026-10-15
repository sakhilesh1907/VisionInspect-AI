"""
src/models/autoencoder.py

Convolutional Autoencoder (CAE) baseline for unsupervised anomaly detection.

Principle:
  - Train exclusively on normal images.
  - Anomaly score = pixel-wise reconstruction error (MSE / SSIM).
  - Anomaly map  = per-pixel squared error (upsampled to input resolution).

Architecture:
  Encoder: Conv2d → BN → ReLU blocks, stride-2 downsampling
  Bottleneck: Compact latent representation
  Decoder: ConvTranspose2d blocks, bilinear upsampling
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvTBNReLU(nn.Sequential):
    """ConvTranspose2d → BatchNorm2d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 4, stride: int = 2, padding: int = 1):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Progressively downsamples the input by a factor of 32.
    Input:  (B, 3, H, W)
    Output: (B, bottleneck_dim, H//32, W//32)
    """

    def __init__(self, base_channels: int = 64, bottleneck_dim: int = 256):
        super().__init__()
        bc = base_channels
        self.net = nn.Sequential(
            ConvBNReLU(3, bc, stride=2),        # /2
            ConvBNReLU(bc, bc),
            ConvBNReLU(bc, bc * 2, stride=2),   # /4
            ConvBNReLU(bc * 2, bc * 2),
            ConvBNReLU(bc * 2, bc * 4, stride=2),   # /8
            ConvBNReLU(bc * 4, bc * 4),
            ConvBNReLU(bc * 4, bc * 8, stride=2),   # /16
            ConvBNReLU(bc * 8, bc * 8),
            ConvBNReLU(bc * 8, bottleneck_dim, stride=2),  # /32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Progressively upsamples the bottleneck by a factor of 32.
    Input:  (B, bottleneck_dim, H//32, W//32)
    Output: (B, 3, H, W)
    """

    def __init__(self, base_channels: int = 64, bottleneck_dim: int = 256):
        super().__init__()
        bc = base_channels
        self.net = nn.Sequential(
            ConvTBNReLU(bottleneck_dim, bc * 8),     # *2
            ConvBNReLU(bc * 8, bc * 8),
            ConvTBNReLU(bc * 8, bc * 4),             # *4
            ConvBNReLU(bc * 4, bc * 4),
            ConvTBNReLU(bc * 4, bc * 2),             # *8
            ConvBNReLU(bc * 2, bc * 2),
            ConvTBNReLU(bc * 2, bc),                 # *16
            ConvBNReLU(bc, bc),
            ConvTBNReLU(bc, 3),                      # *32
            nn.Sigmoid(),                            # output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Full Autoencoder
# ---------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for unsupervised anomaly detection.

    Usage:
        model = ConvAutoencoder()
        # Training phase:
        recon, score = model(normal_batch)
        loss = model.loss(original_batch, recon)
        # Inference phase:
        anomaly_map, anomaly_score = model.predict(image_batch)
    """

    def __init__(
        self,
        base_channels: int = 64,
        bottleneck_dim: int = 256,
        ssim_weight: float = 0.5,
    ):
        super().__init__()
        self.encoder = Encoder(base_channels, bottleneck_dim)
        self.decoder = Decoder(base_channels, bottleneck_dim)
        self.ssim_weight = ssim_weight

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon: Reconstructed image (B, 3, H, W)
            score: Image-level anomaly scores (B,)
        """
        z = self.encoder(x)
        recon = self.decoder(z)

        # Upsample reconstruction to match input if necessary
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Per-image mean squared error as anomaly score
        score = ((x - recon) ** 2).mean(dim=(1, 2, 3))
        return recon, score

    # ------------------------------------------------------------------
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            anomaly_map: Per-pixel anomaly map (B, 1, H, W)
            anomaly_score: Image-level score (B,)
        """
        with torch.no_grad():
            recon, score = self.forward(x)
        pixel_error = ((x - recon) ** 2).mean(dim=1, keepdim=True)  # (B,1,H,W)
        return pixel_error, score

    # ------------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, recon: torch.Tensor
    ) -> torch.Tensor:
        """
        Combined MSE + SSIM loss.
        Higher ssim_weight emphasises structural similarity.
        """
        mse_loss = F.mse_loss(recon, x)
        if self.ssim_weight > 0:
            ssim_val = self._ssim(recon, x)
            return (1 - self.ssim_weight) * mse_loss + self.ssim_weight * (1 - ssim_val)
        return mse_loss

    # ------------------------------------------------------------------
    @staticmethod
    def _ssim(
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ) -> torch.Tensor:
        """Differentiable SSIM approximation."""
        mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
        mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_x_sq
        sigma_y = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y_sq
        sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        return (num / (den + 1e-8)).mean()

    # ------------------------------------------------------------------
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the bottleneck representation."""
        with torch.no_grad():
            return self.encoder(x)

    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True):
        """Backwards-compatible loader for checkpoints saved with or without inner .net."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.") and not k.startswith("encoder.net."):
                k = "encoder.net." + k[len("encoder."):]
            elif k.startswith("decoder.") and not k.startswith("decoder.net."):
                k = "decoder.net." + k[len("decoder."):]
            remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

