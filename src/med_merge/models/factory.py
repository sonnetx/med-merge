"""Model creation from configuration."""

from __future__ import annotations

from typing import Optional

import torch

from med_merge.config.schema import DatasetConfig, ModelConfig
from med_merge.models.backbones import (
    BackboneEncoder,
    CLIPVisionEncoder,
    DINOv3Encoder,
    ViTEncoder,
)
from med_merge.models.classifier import VisionClassifier
from med_merge.models.heads import create_head


# Maps backbone identifier (or prefix) to (encoder_class, default_hidden_size, default_num_layers)
BACKBONE_REGISTRY: dict[str, tuple[type[BackboneEncoder], int, int]] = {
    "openai/clip-vit-base-patch16": (CLIPVisionEncoder, 768, 12),
    "google/vit-base-patch16-224": (ViTEncoder, 768, 12),
    "facebook/dinov3-vits16-pretrain-lvd1689m": (DINOv3Encoder, 384, 12),
}


def _resolve_encoder_class(backbone: str) -> type[BackboneEncoder]:
    """Look up encoder class, falling back to keyword heuristics."""
    if backbone in BACKBONE_REGISTRY:
        return BACKBONE_REGISTRY[backbone][0]
    lower = backbone.lower()
    if "clip" in lower:
        return CLIPVisionEncoder
    if "dinov3" in lower or "dino-v3" in lower:
        return DINOv3Encoder
    if "vit" in lower:
        return ViTEncoder
    raise ValueError(
        f"Cannot resolve backbone {backbone!r}. "
        f"Known backbones: {list(BACKBONE_REGISTRY)}"
    )


def create_encoder(model_config: ModelConfig) -> BackboneEncoder:
    """Instantiate a BackboneEncoder from ModelConfig."""
    cls = _resolve_encoder_class(model_config.backbone)
    if cls is DINOv3Encoder:
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
