"""Lightning data module for processed LACE layouts."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Shaped
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .config import LaceTrainingDatasetName, LaceTrainingSplit
from .dataset import LaceProcessedDataset, collate_lace_batch


class LaceDataModule(LightningDataModule):
    """Load processed train, validation, and test streams for LACE."""

    def __init__(
        self,
        *,
        processed_data_dir: str | Path = ".cache/lace/processed",
        dataset_name: LaceTrainingDatasetName = "publaynet",
        batch_size: int = 256,
        max_seq_length: int = 25,
        num_workers: int = 4,
        pin_memory: bool = True,
        loader_seed: int | None = None,
    ) -> None:
        """Initialize loader settings matching the original entry point."""
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_seed = loader_seed
        self.train_dataset: LaceProcessedDataset | None = None
        self.val_dataset: LaceProcessedDataset | None = None
        self.test_dataset: LaceProcessedDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open requested processed splits."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("validation")

        if stage in {None, "test"}:
            self.test_dataset = self._dataset("test")

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]:
        """Return the shuffled training stream."""
        return self._loader(self._ensure_dataset("train"), shuffle=True)

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]:
        """Return the deterministic validation stream."""
        return self._loader(self._ensure_dataset("validation"), shuffle=False)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]:
        """Return the deterministic test stream."""
        return self._loader(self._ensure_dataset("test"), shuffle=False)

    def _dataset(self, split: LaceTrainingSplit) -> LaceProcessedDataset:
        return LaceProcessedDataset(
            processed_data_dir=self.processed_data_dir,
            dataset_name=self.dataset_name,
            split=split,
            max_seq_length=self.max_seq_length,
        )

    def _ensure_dataset(self, split: LaceTrainingSplit) -> LaceProcessedDataset:
        dataset = {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset,
        }[split]
        if dataset is None:
            self.setup("test" if split == "test" else "fit")
            dataset = {
                "train": self.train_dataset,
                "validation": self.val_dataset,
                "test": self.test_dataset,
            }[split]

        if dataset is None:
            raise RuntimeError(f"Dataset {split} has not been initialized")

        return dataset

    def _loader(
        self,
        dataset: LaceProcessedDataset | None,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | list[str]]]:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized")

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_lace_batch, max_seq_length=self.max_seq_length),
            generator=(
                torch.Generator().manual_seed(self.loader_seed)
                if self.loader_seed is not None
                else None
            ),
        )
        return cast(
            DataLoader[dict[str, Shaped[torch.Tensor, "..."] | list[str]]], loader
        )
