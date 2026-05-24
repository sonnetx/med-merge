"""Model creation from configuration."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import DatasetConfig, ModelConfig
from med_merge.models.backbones import (
    BackboneEncoder,
    BEiTEncoder,
    CLIPVisionEncoder,
    DINOv2Encoder,
    DINOv3Encoder,
    MAEEncoder,
    MedCLIPEncoder,
    RADDINOEncoder,
    ViTEncoder,
)
from med_merge.models.classifier import VisionClassifier
from med_merge.models.heads import create_head


# Maps backbone identifier (or prefix) to (encoder_class, default_hidden_size, default_num_layers)
BACKBONE_REGISTRY: dict[str, tuple[type[BackboneEncoder], int, int]] = {
    # Round-1 backbones
    "openai/clip-vit-base-patch16": (CLIPVisionEncoder, 768, 12),
    "google/vit-base-patch16-224": (ViTEncoder, 768, 12),
    "facebook/dinov3-vits16-pretrain-lvd1689m": (DINOv3Encoder, 384, 12),
    "microsoft/rad-dino": (RADDINOEncoder, 768, 12),
    # Phase B additions (workshop-version backbone zoo)
    "facebook/dinov2-base": (DINOv2Encoder, 768, 12),
    "facebook/vit-mae-base": (MAEEncoder, 768, 12),
    "microsoft/beit-base-patch16-224-pt22k-ft22k": (BEiTEncoder, 768, 12),
    "flaviagiammarino/medclip-vit": (MedCLIPEncoder, 768, 12),
}


# Canonical short alias used in output directory paths (matches the SLURM
# scripts' $BACKBONE variable). Falls back to the HF id's last segment.
BACKBONE_ALIAS: dict[str, str] = {
    "openai/clip-vit-base-patch16": "clip",
    "google/vit-base-patch16-224": "vit",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3",
    "microsoft/rad-dino": "rad_dino",
    "facebook/dinov2-base": "dinov2",
    "facebook/vit-mae-base": "mae",
    "microsoft/beit-base-patch16-224-pt22k-ft22k": "beit",
    "flaviagiammarino/medclip-vit": "medclip",
}


def alias_for(backbone: str) -> str:
    """Return the canonical short alias for a backbone HF id."""
    if backbone in BACKBONE_ALIAS:
        return BACKBONE_ALIAS[backbone]
    lower = backbone.lower()
    if "medclip" in lower:
        return "medclip"
    if "rad-dino" in lower or "rad_dino" in lower:
        return "rad_dino"
    if "clip" in lower:
        return "clip"
    if "dinov3" in lower or "dino-v3" in lower:
        return "dinov3"
    if "dinov2" in lower:
        return "dinov2"
    if "mae" in lower:
        return "mae"
    if "beit" in lower:
        return "beit"
    if "vit" in lower:
        return "vit"
    return backbone.split("/")[-1]


def _resolve_encoder_class(backbone: str) -> type[BackboneEncoder]:
    """Look up encoder class, falling back to keyword heuristics."""
    if backbone in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[backbone][0]
    lower = backbone.lower()
    if "medclip" in lower:
        return MedCLIPEncoder
    if "rad-dino" in lower or "rad_dino" in lower:
        return RADDINOEncoder
    if "clip" in lower:
        return CLIPVisionEncoder
    if "dinov3" in lower or "dino-v3" in lower:
        return DINOv3Encoder
    if "dinov2" in lower or "dino-v2" in lower:
        return DINOv2Encoder
    if "mae" in lower:
        return MAEEncoder
    if "beit" in lower:
        return BEiTEncoder
    if "vit" in lower:
        return ViTEncoder
    raise ValueError(
        f"Cannot resolve backbone {backbone!r}. "
        f"Known backbones: {list(BACKBONE_REGISTRY)}"
    )


def create_encoder(model_config: ModelConfig) -> BackboneEncoder:
    """Instantiate a BackboneEncoder from ModelConfig."""
    cls = _resolve_encoder_class(model_config.backbone)
    # All encoders that accept a hidden_size override (AutoModel-based ones)
    if cls in (DINOv3Encoder, RADDINOEncoder, DINOv2Encoder, MAEEncoder, BEiTEncoder, MedCLIPEncoder):
        return cls(model_config.backbone, hidden_size=model_config.hidden_size)
    return cls(model_config.backbone)


def create_model(
    dataset_config: DatasetConfig,
    model_config: Optional[ModelConfig] = None,
    training_config=None,
) -> VisionClassifier:
    """Build a VisionClassifier from configs."""
    if model_config is None:
        model_config = ModelConfig()

    encoder = create_encoder(model_config)
    head = create_head(
        model_config.head_type,
        encoder.hidden_size,
        dataset_config.num_classes,
    )
    model = VisionClassifier(encoder, head)

    if model_config.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model


def load_pretrained_encoder(model_config: Optional[ModelConfig] = None) -> dict[str, torch.Tensor]:
    """Load pretrained encoder state dict (the merging anchor).

    Builds a fresh encoder from config, returns its state dict prefixed
    with ``encoder.`` (matching ``VisionClassifier.get_encoder_state_dict``).
    """
    if model_config is None:
        model_config = ModelConfig()

    encoder = create_encoder(model_config)
    return {
        f"encoder.{k}": v.clone()
        for k, v in encoder.state_dict().items()
    }
