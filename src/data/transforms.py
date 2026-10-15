"""
src/data/transforms.py

Image augmentation pipelines for training and validation.
Supports albumentations and torchvision.
"""

from torchvision import transforms as T


# Standard ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default image size for most MVTec experiments
DEFAULT_SIZE = 256
DEFAULT_CROP = 224


def get_train_transforms(
    image_size: int = DEFAULT_SIZE,
    crop_size: int = DEFAULT_CROP,
    use_augmentation: bool = True,
) -> T.Compose:
    """
    Training transforms.

    When use_augmentation=True, applies:
      - Random horizontal / vertical flips
      - Small rotations (±10°)
      - Colour jitter (brightness, contrast, saturation)
      - Random gaussian blur
      - Random erasing (simulates occlusions)
    """
    base = [
        T.Resize(image_size),
        T.CenterCrop(crop_size),
    ]

    if use_augmentation:
        augment = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    else:
        augment = []

    normalize = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if use_augmentation:
        normalize.append(T.RandomErasing(p=0.1, scale=(0.02, 0.1)))

    return T.Compose(base + augment + normalize)


def get_val_transforms(
    image_size: int = DEFAULT_SIZE,
    crop_size: int = DEFAULT_CROP,
) -> T.Compose:
    """
    Validation / test transforms (deterministic, no augmentation).
    """
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalisation for visualisation.
    tensor: (C, H, W) or (B, C, H, W)
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0.0, 1.0)
