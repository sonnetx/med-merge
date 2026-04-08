"""Task vector: the difference between fine-tuned and pretrained model weights."""

from __future__ import annotations

from pathlib import Path

import torch


class TaskVector:
    """Represents the difference between fine-tuned and pretrained encoder weights.

    task_vector = fine_tuned_weights - pretrained_weights

    Supports arithmetic: addition, negation, scaling.
    """

    def __init__(
        self,
        pretrained_state_dict: dict[str, torch.Tensor] | None = None,
        finetuned_state_dict: dict[str, torch.Tensor] | None = None,
        vector: dict[str, torch.Tensor] | None = None,
    ):
        if vector is not None:
            self.vector = vector
        elif pretrained_state_dict is not None and finetuned_state_dict is not None:
            self.vector = {}
            for key in pretrained_state_dict:
                if key not in finetuned_state_dict:
                    continue
                if pretrained_state_dict[key].dtype.is_floating_point:
                    self.vector[key] = (
                        finetuned_state_dict[key].cpu() - pretrained_state_dict[key].cpu()
                    )
        else:
            raise ValueError("Provide either (pretrained + finetuned) or vector")

    def __add__(self, other: TaskVector) -> TaskVector:
        new_vector = {}
        for key in self.vector:
            if key in other.vector:
                new_vector[key] = self.vector[key] + other.vector[key]
            else:
                new_vector[key] = self.vector[key].clone()
        for key in other.vector:
            if key not in self.vector:
                new_vector[key] = other.vector[key].clone()
        return TaskVector(vector=new_vector)

    def __neg__(self) -> TaskVector:
        return TaskVector(vector={k: -v for k, v in self.vector.items()})

    def __mul__(self, scalar: float) -> TaskVector:
        return TaskVector(vector={k: v * scalar for k, v in self.vector.items()})

    def __rmul__(self, scalar: float) -> TaskVector:
        return self.__mul__(scalar)

    def apply_to(
        self,
        pretrained_state_dict: dict[str, torch.Tensor],
        scaling_coef: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Return pretrained + scaling_coef * task_vector."""
        result = {}
        for key in pretrained_state_dict:
            if key in self.vector:
                result[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
            else:
                result[key] = pretrained_state_dict[key].clone()
        return result

    def norm(self) -> float:
        """L2 norm of the task vector."""
        total = sum(v.float().norm().item() ** 2 for v in self.vector.values())
        return total ** 0.5

    def save(self, path: Path) -> None:
        """Save task vector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.vector, path)

    @classmethod
    def load(cls, path: Path) -> TaskVector:
        """Load task vector from disk."""
        vector = torch.load(path, map_location="cpu", weights_only=True)
        return cls(vector=vector)
