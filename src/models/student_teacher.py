"""
src/models/student_teacher.py

Student-Teacher Anomaly Detection via Knowledge Distillation.

Reference:
  Bergmann et al., "Uninformed Students: Student–Teacher Anomaly
  Detection with Discriminative Latent Embeddings", CVPR 2020.

Principle:
  - A pretrained Teacher encoder maps normal patches to descriptor vectors.
  - A Student network is trained to replicate the Teacher's outputs on normal data.
  - For anomalous patches, the Student fails to reproduce the Teacher → high error.
  - Anomaly score = mean-squared difference between Teacher and Student outputs.

Advantages:
  - Lightweight: Student can be much smaller than Teacher.
  - Calibratable uncertainty: ensemble of Students gives variance estimates.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Teacher (Frozen Pretrained Encoder)
# ---------------------------------------------------------------------------

class TeacherEncoder(nn.Module):
    """
    Frozen pretrained backbone used as the Teacher network.
    Outputs a D-dimensional descriptor vector per spatial position.
    """

    def __init__(self, backbone: str = "wide_resnet50_2", layer_idx: int = 2):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=True, features_only=True)
        self.model.eval()
        self.layer_idx = layer_idx
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns feature map at the selected layer: (B, D, H_f, W_f)."""
        feats = self.model(x)
        return feats[self.layer_idx]


# ---------------------------------------------------------------------------
# Student Network
# ---------------------------------------------------------------------------

class StudentDecoder(nn.Module):
    """
    Lightweight convolutional student that learns to replicate the teacher.

    Architecture: a series of bottleneck-style blocks that map the same
    input → same feature space as the teacher.  We deliberately make the
    student shallower / narrower to speed up training and increase sensitivity
    to anomalous inputs that the student hasn't seen.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 512, base_ch: int = 64):
        super().__init__()
        bc = base_ch
        self.net = nn.Sequential(
            # Encoder path (same stride as teacher up to layer 2)
            nn.Conv2d(in_channels, bc, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
            nn.Conv2d(bc, bc * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bc * 2), nn.ReLU(inplace=True),
            nn.Conv2d(bc * 2, bc * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bc * 4), nn.ReLU(inplace=True),
            # Output projection
            nn.Conv2d(bc * 4, out_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Student-Teacher Ensemble
# ---------------------------------------------------------------------------

class StudentTeacher(nn.Module):
    """
    Student-Teacher anomaly detector with optional Student ensemble
    for uncertainty estimation.

    Args:
        backbone (str): Teacher backbone.
        teacher_layer (int): Which feature layer to distil.
        n_students (int): Number of Student networks in the ensemble.
        base_ch (int): Student base channel width.
        device (str): Compute device.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        teacher_layer: int = 2,
        n_students: int = 3,
        base_ch: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_students = n_students

        # Teacher
        self.teacher = TeacherEncoder(backbone, teacher_layer).to(self.device)

        # Probe teacher output channels
        dummy = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            teacher_out = self.teacher(dummy)
        self.out_channels = teacher_out.shape[1]

        # Student ensemble
        self.students = nn.ModuleList([
            StudentDecoder(3, self.out_channels, base_ch) for _ in range(n_students)
        ]).to(self.device)

        self._is_fitted = False

    # ------------------------------------------------------------------
    def _teacher_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns:
            teacher_feats: (B, D, H_f, W_f)  target
            student_feats_list: List[(B, D, H_f, W_f)] per student
        """
        teacher_feats = self._teacher_features(x)
        student_feats = [s(x) for s in self.students]
        return teacher_feats, student_feats

    # ------------------------------------------------------------------
    def loss(
        self,
        teacher_feats: torch.Tensor,
        student_feats_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Mean squared error between teacher and each student, averaged over ensemble."""
        losses = [F.mse_loss(sf, teacher_feats) for sf in student_feats_list]
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly maps and scores.

        Anomaly map = mean(teacher − student)² over the ensemble.
        Uncertainty map = variance of student predictions (for human review routing).

        Returns:
            anomaly_map: (B, 1, H, W)
            anomaly_score: (B,)
        """
        B = x.shape[0]
        H_in, W_in = x.shape[-2], x.shape[-1]

        teacher_feats = self._teacher_features(x)        # (B, D, H_f, W_f)
        H_f, W_f = teacher_feats.shape[-2], teacher_feats.shape[-1]

        student_preds = torch.stack([s(x) for s in self.students], dim=0)  # (S, B, D, H_f, W_f)

        # Per-student error maps
        errors = ((student_preds - teacher_feats.unsqueeze(0)) ** 2).mean(dim=2)  # (S, B, H_f, W_f)

        # Mean error across students
        mean_error = errors.mean(dim=0, keepdim=True).squeeze(0)  # (B, H_f, W_f)
        dist_map = mean_error.unsqueeze(1)                         # (B, 1, H_f, W_f)

        anomaly_map = F.interpolate(dist_map, size=(H_in, W_in), mode="bilinear", align_corners=False)
        anomaly_score = anomaly_map.view(B, -1).max(dim=1).values

        return anomaly_map, anomaly_score

    # ------------------------------------------------------------------
    @torch.no_grad()
    def uncertainty_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pixel-wise variance across the student ensemble.
        High variance → high uncertainty → route to human review.

        Returns: (B, 1, H, W)
        """
        H_in, W_in = x.shape[-2], x.shape[-1]
        student_preds = torch.stack([s(x) for s in self.students], dim=0)  # (S, B, D, H_f, W_f)
        var_map = student_preds.var(dim=0).mean(dim=1, keepdim=True)        # (B, 1, H_f, W_f)
        return F.interpolate(var_map, size=(H_in, W_in), mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return only student parameters for optimisation."""
        return list(self.students.parameters())

    def save(self, path: str) -> None:
        torch.save(
            {
                "student_states": [s.state_dict() for s in self.students],
                "n_students": self.n_students,
                "out_channels": self.out_channels,
            },
            path,
        )

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        for s, sd in zip(self.students, state["student_states"]):
            s.load_state_dict(sd)
