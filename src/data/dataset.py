"""
src/data/dataset.py

Dataset loaders for:
  - MVTec Anomaly Detection (MVTec AD) benchmark
  - Custom industrial inspection images
  - Few-shot split generation for limited-data experiments
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# MVTec AD Dataset
# ---------------------------------------------------------------------------

# MVTec AD categories
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for the MVTec Anomaly Detection benchmark.

    Directory layout expected:
        root/
          <category>/
            train/
              good/
                *.png
            test/
              good/
                *.png
              <defect_type>/
                *.png
            ground_truth/
              <defect_type>/
                *_mask.png

    Args:
        root (str): Path to the MVTec AD root directory.
        category (str): Product category (e.g. "bottle").
        split (str): One of "train", "val", "test".
        transform: Optional torchvision / albumentations transform.
        gt_transform: Optional transform for ground-truth masks.
        few_shot_normal (int | None): If set, limit normal training images.
        few_shot_defect (int | None): If set, limit defect examples per class.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform=None,
        gt_transform=None,
        few_shot_normal: Optional[int] = None,
        few_shot_defect: Optional[int] = None,
        seed: int = 42,
    ):
        assert category in MVTEC_CATEGORIES, f"Unknown category: {category}"
        assert split in ("train", "val", "test")

        self.root = Path(root) / category
        self.category = category
        self.split = split
        self.transform = transform or get_val_transforms()
        self.gt_transform = gt_transform
        self.seed = seed

        self.samples: List[Dict[str, Any]] = []
        self._build_samples(few_shot_normal, few_shot_defect)

    # ------------------------------------------------------------------
    def _build_samples(
        self,
        few_shot_normal: Optional[int],
        few_shot_defect: Optional[int],
    ) -> None:
        rng = random.Random(self.seed)

        if self.split == "train":
            good_dir = self.root / "train" / "good"
            paths = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
            if few_shot_normal is not None:
                paths = rng.sample(paths, min(few_shot_normal, len(paths)))
            for p in paths:
                self.samples.append(
                    {"image_path": p, "mask_path": None, "label": 0, "defect_type": "good"}
                )

        else:  # val / test
            test_dir = self.root / "test"
            gt_dir = self.root / "ground_truth"

            for defect_dir in sorted(test_dir.iterdir()):
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1

                img_paths = sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg"))

                if label == 1 and few_shot_defect is not None:
                    img_paths = rng.sample(img_paths, min(few_shot_defect, len(img_paths)))

                for img_path in img_paths:
                    mask_path = None
                    if label == 1:
                        mask_candidate = gt_dir / defect_type / (img_path.stem + "_mask.png")
                        if mask_candidate.exists():
                            mask_path = mask_candidate
                    self.samples.append(
                        {
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "label": label,
                            "defect_type": defect_type,
                        }
                    )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        mask = torch.zeros(1, image.shape[1], image.shape[2])
        if sample["mask_path"] is not None:
            raw_mask = Image.open(sample["mask_path"]).convert("L")
            if self.gt_transform:
                raw_mask = self.gt_transform(raw_mask)
            else:
                from torchvision import transforms as T
                raw_mask = T.Compose([
                    T.Resize(image.shape[1]),
                    T.ToTensor(),
                ])(raw_mask)
            mask = (raw_mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }


# ---------------------------------------------------------------------------
# Custom Inspection Dataset
# ---------------------------------------------------------------------------

class CustomInspectionDataset(Dataset):
    """
    Dataset for custom industrial images.

    Expects a manifest JSON file with entries:
        [{"image_path": "...", "label": 0/1, "defect_type": "...", "mask_path": null}, ...]

    Args:
        manifest_path (str): Path to manifest JSON.
        root (str): Optional base directory prepended to relative paths.
        transform: Image transform.
    """

    def __init__(
        self,
        manifest_path: str,
        root: str = "",
        transform=None,
    ):
        with open(manifest_path) as f:
            self.samples: List[Dict[str, Any]] = json.load(f)
        self.root = Path(root)
        self.transform = transform or get_val_transforms()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = self.root / sample["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        mask = torch.zeros(1, image.shape[1], image.shape[2])
        if sample.get("mask_path"):
            mask_path = self.root / sample["mask_path"]
            if mask_path.exists():
                raw_mask = Image.open(mask_path).convert("L")
                from torchvision import transforms as T
                raw_mask = T.Compose([T.Resize(image.shape[1]), T.ToTensor()])(raw_mask)
                mask = (raw_mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(sample.get("label", -1), dtype=torch.long),
            "defect_type": sample.get("defect_type", "unknown"),
            "image_path": str(img_path),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Return a configured DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Few-shot split utilities
# ---------------------------------------------------------------------------

def create_few_shot_splits(
    dataset: MVTecDataset,
    k_values: List[int] = [1, 5, 10, 50],
    seed: int = 42,
    output_dir: str = "data/splits",
) -> Dict[int, List[int]]:
    """
    Create index lists for few-shot experiments with k defect samples each.

    Returns a dict mapping k -> list of sample indices.
    """
    rng = random.Random(seed)
    defect_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 1]
    normal_indices = [i for i, s in enumerate(dataset.samples) if s["label"] == 0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits: Dict[int, List[int]] = {}

    for k in k_values:
        chosen_defect = rng.sample(defect_indices, min(k, len(defect_indices)))
        split_indices = normal_indices + chosen_defect
        splits[k] = split_indices
        split_path = Path(output_dir) / f"few_shot_k{k}.json"
        with open(split_path, "w") as f:
            json.dump({"k": k, "indices": split_indices}, f, indent=2)

    return splits
