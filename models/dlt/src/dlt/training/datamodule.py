"""PyTorch Lightning data module for DLT smoke training."""

from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import H5DLTDataset, SyntheticDLTDataset, collate_dlt_batch


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
        data_path: str | None = None,
        train_file: str = "publaynet_train.h5",
        val_file: str = "publaynet_val.h5",
        shuffle_train: bool = False,
    ) -> None:
        """Initialize data-module parameters."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length
        self.max_num_comp = max_num_comp
        self.categories_num = categories_num
        self.seed = seed
        self.data_path = Path(data_path) if data_path is not None else None
        self.train_file = train_file
        self.val_file = val_file
        self.shuffle_train = shuffle_train

    def setup(self, stage: str | None = None) -> None:
        """Create train/validation datasets."""
        del stage
        if self.data_path is not None:
            self.train_dataset = H5DLTDataset(
                self.data_path / self.train_file,
                max_num_comp=self.max_num_comp,
            )
            self.val_dataset = H5DLTDataset(
                self.data_path / self.val_file,
                max_num_comp=self.max_num_comp,
            )
            return
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
            shuffle=self.shuffle_train,
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
