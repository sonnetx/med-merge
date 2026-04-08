"""Backward-compatibility shim — imports from classifier.py."""

from med_merge.models.classifier import CLIPClassifier, VisionClassifier

# Re-export constants that existing code references
CLIP_PRETRAINED = CLIPClassifier.PRETRAINED
CLIP_HIDDEN_SIZE = 768

__all__ = ["CLIPClassifier", "VisionClassifier", "CLIP_PRETRAINED", "CLIP_HIDDEN_SIZE"]
