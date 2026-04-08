"""Vision encoder + classification head for model merging.

The ``VisionClassifier`` wraps any ``BackboneEncoder`` and a classification
head.  Only encoder weights participate in task-vector arithmetic; the head
is dataset-specific and kept separate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from med_merge.models.backbones import BackboneEncoder
from med_merge.models.heads import create_head


class VisionClassifier(nn.Module):
    """Generic encoder + head classifier.

    For model merging, ``get_encoder_state_dict`` / ``load_encoder_state_dict``
    isolate the encoder weights so task vectors only contain encoder deltas.
    """

    def __init__(
        self,
        encoder: BackboneEncoder,
        head: nn.Module,
    ):
        super().__init__()
        self.encoder: nn.Module = encoder  # type: ignore[assignment]
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.encoder(pixel_values)
        return self.head(features)

    # -- state-dict helpers for task-vector computation -----------------------

    def get_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.clone()
            for k, v in self.state_dict().items()
            if k.startswith("encoder.")
        }

    def get_head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.clone()
            for k, v in self.state_dict().items()
            if k.startswith("head.")
        }

    def load_encoder_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current)

    def load_head_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current)


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

class CLIPClassifier(VisionClassifier):
    """Legacy wrapper: builds a VisionClassifier with CLIP ViT-B/16.

    Kept for backward compatibility with existing checkpoints and code
    that imports ``CLIPClassifier`` directly.
    """

    PRETRAINED = "openai/clip-vit-base-patch16"

    def __init__(
        self,
        num_classes: int,
        task_type: str = "multiclass",
        pretrained: str = PRETRAINED,
        freeze_encoder: bool = False,
    ):
        from med_merge.models.backbones import CLIPVisionEncoder

        encoder = CLIPVisionEncoder(pretrained)
        head = create_head("linear", encoder.hidden_size, num_classes)
        super().__init__(encoder, head)
        self.task_type = task_type

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @staticmethod
    def get_pretrained_encoder_state_dict(
        pretrained: str = "openai/clip-vit-base-patch16",
    ) -> dict[str, torch.Tensor]:
        """Load pretrained CLIP encoder state dict (the merging anchor)."""
        from transformers import CLIPVisionModel

        model = CLIPVisionModel.from_pretrained(pretrained)
        prefix = "encoder."
        return {
            f"{prefix}{k}": v.clone()
            for k, v in model.state_dict().items()
        }
