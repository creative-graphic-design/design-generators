"""Small DLT datasets used by smoke training configs and tests."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Dataset


class SyntheticDLTDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic synthetic DLT batches that never download data."""

    def __init__(
        self,
        *,
        length: int = 8,
        max_num_comp: int = 4,
        categories_num: int = 7,
        seed: int = 0,
    ) -> None:
        """Initialize a synthetic dataset."""
        self.length = length
        self.max_num_comp = max_num_comp
        self.categories_num = categories_num
        self.seed = seed

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one deterministic synthetic DLT sample."""
        generator = torch.Generator().manual_seed(self.seed + index)
        box = torch.rand(self.max_num_comp, 4, generator=generator) * 4.0 - 2.0
        cat = torch.randint(
            1, self.categories_num - 1, (self.max_num_comp,), generator=generator
        )
        mask = torch.ones(self.max_num_comp, dtype=torch.bool)
        mask_box = torch.ones(self.max_num_comp, 4, dtype=torch.long)
        mask_cat = torch.ones(self.max_num_comp, dtype=torch.long)
        return {
            "box": box.float(),
            "box_cond": box.float(),
            "cat": cat.long(),
            "mask": mask,
            "mask_box": mask_box,
            "mask_cat": mask_cat,
        }


def collate_dlt_batch(
    examples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack DLT examples into one batch."""
    keys: Iterator[str] = iter(examples[0])
    return {key: torch.stack([example[key] for example in examples]) for key in keys}
