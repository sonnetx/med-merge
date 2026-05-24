"""Vision encoder backends for model merging.

Each encoder wraps a pretrained vision transformer and exposes a uniform
``forward(pixel_values) -> features`` interface.  Only the encoder weights
participate in task-vector arithmetic — the classification head is kept
separate.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BackboneEncoder(ABC):
    """Abstract vision encoder that produces a feature vector."""

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return a [B, hidden_size] feature tensor."""
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer blocks (needed by LiNeS)."""
        ...

    @property
    @abstractmethod
    def layer_key_pattern(self) -> str:
        r"""Regex with a single capture group for the layer index.

        Example: ``r"layers\.(\d+)\."``
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class CLIPVisionEncoder(nn.Module, BackboneEncoder):
    """CLIP ViT-B/16 vision encoder (HuggingFace ``CLIPVisionModel``)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(model_name)
        cfg = self.model.config
        self._hidden_size = cfg.hidden_size
        self._num_layers = cfg.num_hidden_layers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def layer_key_pattern(self) -> str:
        # Keys: encoder.vision_model.encoder.layers.{i}.xxx
        return r"layers\.(\d+)\."


class ViTEncoder(nn.Module, BackboneEncoder):
    """Plain ViT encoder (HuggingFace ``ViTModel``).

    Uses CLS token (``last_hidden_state[:, 0]``) as the feature vector.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(model_name)
        cfg = self.model.config
        self._hidden_size = cfg.hidden_size
        self._num_layers = cfg.num_hidden_layers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]  # CLS token

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def layer_key_pattern(self) -> str:
        # Keys: encoder.layer.{i}.xxx
        return r"layer\.(\d+)\."


class DINOv3Encoder(nn.Module, BackboneEncoder):
    """DINOv3 encoder (gated HuggingFace model via ``AutoModel``).

    Returns ``pooler_output`` (384-d for ViT-S/16).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        hidden_size: int = 384,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self._hidden_size = hidden_size
        # DINOv3 ViT-S/16 has 12 layers
        self._num_layers = getattr(
            getattr(self.model, "config", None), "num_hidden_layers", 12
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).pooler_output

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def layer_key_pattern(self) -> str:
        # DINOv3 keys follow HF ViT conventions
        return r"layer\.(\d+)\."


class RADDINOEncoder(nn.Module, BackboneEncoder):
    """RAD-DINO encoder (``microsoft/rad-dino``, ViT-B/16, 768-d).

    Chest X-ray pretrained DINOv2-style model. Returns ``pooler_output``.
    """

    def __init__(
        self,
        model_name: str = "microsoft/rad-dino",
        hidden_size: int = 768,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        cfg = getattr(self.model, "config", None)
        self._hidden_size = getattr(cfg, "hidden_size", hidden_size)
        self._num_layers = getattr(cfg, "num_hidden_layers", 12)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def layer_key_pattern(self) -> str:
        return r"layer\.(\d+)\."



class _AutoModelEncoder(nn.Module, BackboneEncoder):
    """Shared scaffold for HF AutoModel-based encoders.
    """

    MODEL_DEFAULT: str = ""

    def __init__(self, model_name: Optional[str] = None, hidden_size: int = 768):
        super().__init__()
        from transformers import AutoModel

        name = model_name or self.MODEL_DEFAULT
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
        cfg = getattr(self.model, "config", None)
        self._hidden_size = getattr(cfg, "hidden_size", hidden_size)
        self._num_layers = getattr(cfg, "num_hidden_layers", 12)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def layer_key_pattern(self) -> str:
        return r"layer\.(\d+)\."


class DINOv2Encoder(_AutoModelEncoder):
    """DINOv2 ViT-B/14 (``facebook/dinov2-base``, 768-d)."""

    MODEL_DEFAULT = "facebook/dinov2-base"


class MAEEncoder(_AutoModelEncoder):
    """ViT-MAE-Base (``facebook/vit-mae-base``, 768-d).

    Uses CLS-token feature (last_hidden_state[:, 0]); ViTMAE does not expose
    a pooler_output.
    """

    MODEL_DEFAULT = "facebook/vit-mae-base"

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]


class BEiTEncoder(_AutoModelEncoder):
    """BEiT base (``microsoft/beit-base-patch16-224-pt22k-ft22k``, 768-d)."""

    MODEL_DEFAULT = "microsoft/beit-base-patch16-224-pt22k-ft22k"


class MedCLIPEncoder(_AutoModelEncoder):
    """MedCLIP vision encoder (``flaviagiammarino/medclip-vit``, 768-d).
    """

    MODEL_DEFAULT = "flaviagiammarino/medclip-vit"

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        m = self.model
        sub = getattr(m, "vision_model", None) or m
        outputs = sub(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    @property
    def layer_key_pattern(self) -> str:
         return r"layers\.(\d+)\."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_layer_depth(key: str, pattern: str) -> Optional[int]:
    """Extract transformer layer index from a parameter key."""
    m = re.search(pattern, key)
    return int(m.group(1)) if m else None
