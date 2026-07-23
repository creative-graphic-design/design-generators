"""Training datasets for CGB-DM."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from cgb_dm.data import CGBDMOriginalDataset


class CGBDMSyntheticDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny deterministic dataset used by tests and smoke configs."""

    def __init__(
        self,
        *,
        length: int = 4,
        max_seq_length: int = 4,
        seq_dim: int = 8,
        image_size: tuple[int, int] = (32, 32),
    ) -> None:
        """Initialize synthetic tensor shapes."""
        self.length = length
        self.max_seq_length = max_seq_length
        self.seq_dim = seq_dim
        self.image_size = image_size

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one deterministic row."""
        generator = torch.Generator().manual_seed(index)
        labels = torch.zeros(self.max_seq_length, self.seq_dim - 4)
        labels[:, 0] = 1
        bbox = torch.rand(self.max_seq_length, 4, generator=generator) * 2 - 1
        return {
            "pixel_values": torch.rand(4, *self.image_size, generator=generator) * 2
            - 1,
            "layout": torch.cat((labels, bbox), dim=-1),
            "saliency_box": torch.zeros(1, 4),
        }


__all__ = ["CGBDMOriginalDataset", "CGBDMSyntheticDataset"]
