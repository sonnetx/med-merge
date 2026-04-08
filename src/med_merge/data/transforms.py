"""Image preprocessing and augmentation transforms.

Supports CLIP and ImageNet normalization (ImageNet stats are also used
by ViT, DINOv2, and DINOv3).
"""

from __future__ import annotations

from torchvision import transforms

# -- Normalization constants -------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORMALIZATION_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "clip": (CLIP_MEAN, CLIP_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "vit": (IMAGENET_MEAN, IMAGENET_STD),
    "dino": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov3": (IMAGENET_MEAN, IMAGENET_STD),
}


def _get_norm(norm_key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean, std = NORMALIZATION_STATS.get(norm_key, (CLIP_MEAN, CLIP_STD))
    return mean, std


# -- Transform factories ----------------------------------------------------


def get_train_transform(
    image_size: int = 224,
    norm_key: str = "clip",
) -> transforms.Compose:
    """Training transforms with augmentation."""
    mean, std = _get_norm(norm_key)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transform(
    image_size: int = 224,
    norm_key: str = "clip",
) -> transforms.Compose:
    """Evaluation transforms: resize + center crop + normalize."""
    mean, std = _get_norm(norm_key)
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_aggressive_train_transform(
    image_size: int = 224,
    norm_key: str = "clip",
) -> transforms.Compose:
    """Aggressive augmentation for small / imbalanced datasets."""
    mean, std = _get_norm(norm_key)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2),
    ])


def norm_key_for_backbone(backbone: str) -> str:
    """Infer the normalization key from a backbone model identifier."""
    lower = backbone.lower()
    if "clip" in lower:
        return "clip"
    # ViT, DINOv2, DINOv3 all use ImageNet stats
    return "imagenet"
