"""PyTorch Lightning data module for DLT smoke training."""

from __future__ import annotations

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import SyntheticDLTDataset, collate_dlt_batch


class DLTDataModule(LightningDataModule):
    """DLT data module with synthetic smoke data by default."""

    def __init__(
        self,
        *,
        batch_size: int = 2,
        num_workers: int = 0,
        length: int = 8,
        max_num_comp: int = 4,
        categories_num: int = 7,
        seed: int = 0,
    ) -> None:
        """Initialize data-module parameters."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length
        self.max_num_comp = max_num_comp
        self.categories_num = categories_num
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        """Create synthetic train/validation datasets."""
        del stage
        self.train_dataset = SyntheticDLTDataset(
            length=self.length,
            max_num_comp=self.max_num_comp,
            categories_num=self.categories_num,
            seed=self.seed,
        )
        self.val_dataset = SyntheticDLTDataset(
            length=max(2, self.length // 2),
            max_num_comp=self.max_num_comp,
            categories_num=self.categories_num,
            seed=self.seed + 10_000,
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return the train dataloader."""
        if not hasattr(self, "train_dataset"):
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_dlt_batch,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Return the validation dataloader."""
        if not hasattr(self, "val_dataset"):
            self.setup("validate")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_dlt_batch,
        )
