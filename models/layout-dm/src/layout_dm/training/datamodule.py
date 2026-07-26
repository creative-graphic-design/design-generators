"""LightningDataModule for LayoutDM training."""

from __future__ import annotations

import torch
from jaxtyping import Shaped
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from laygen.common.bbox import BoxFormat

from ..configuration_layout_dm import LayoutDMConfig
from ..tokenization_layout_dm import LayoutDMTokenizer
from .config import (
    LayoutDMTrainingDatasetName,
    LayoutDMTrainingDatasetSource,
    LayoutDMTrainingSplit,
)
from .dataset import LayoutDMDataset, LayoutDMProcessedDataset, LayoutDMSyntheticDataset


class LayoutDMDataModule(LightningDataModule):
    """Package-local LightningDataModule for LayoutDM training data."""

    def __init__(
        self,
        *,
        dataset_name: LayoutDMTrainingDatasetName = "publaynet",
        config: LayoutDMConfig,
        batch_size: int = 256,
        max_seq_length: int | None = None,
        num_workers: int = 4,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        synthetic_size: int | None = None,
        dataset_source: LayoutDMTrainingDatasetSource = "hf",
        processed_data_dir: str | None = None,
    ) -> None:
        """Initialize datamodule settings."""
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length or self.config.max_seq_length
        self.num_workers = num_workers
        self.box_format = box_format
        self.normalized = normalized
        self.synthetic_size = synthetic_size
        self.dataset_source = dataset_source
        self.processed_data_dir = processed_data_dir
        self.tokenizer = LayoutDMTokenizer(self.config)
        self.train_dataset: (
            Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]] | None
        ) = None
        self.val_dataset: (
            Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]] | None
        ) = None
        self.test_dataset: (
            Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]] | None
        ) = None

    def setup(self, stage: str | None = None) -> None:
        """Open datasets for the requested stage."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("validation")
        if stage in {None, "test"}:
            self.test_dataset = self._dataset("test")

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=self.synthetic_size is None)

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the test dataloader."""
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(self.test_dataset, shuffle=False)

    def _dataset(
        self, split: LayoutDMTrainingSplit
    ) -> Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        if self.synthetic_size is not None:
            return LayoutDMSyntheticDataset(
                config=self.config,
                size=self.synthetic_size,
                elements=min(3, self.max_seq_length),
            )
        if self.dataset_source == "processed":
            if self.processed_data_dir is None:
                raise ValueError(
                    "processed_data_dir is required when dataset_source='processed'"
                )
            return LayoutDMProcessedDataset(
                dataset_name=self.dataset_name,
                split=split,
                config=self.config,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                processed_data_dir=self.processed_data_dir,
            )
        return LayoutDMDataset(
            dataset_name=self.dataset_name,
            split=split,
            config=self.config,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            box_format=self.box_format,
            normalized=self.normalized,
        )

    def _loader(
        self,
        dataset: Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]] | None,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
