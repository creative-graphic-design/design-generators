"""Processed LayoutDM streams used by Layout-Corrector training."""

from __future__ import annotations

from pathlib import Path
import torch
from jaxtyping import Shaped
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.training.dataset import LayoutDMProcessedDataset

from .config import (
    LayoutCorrectorTrainingDatasetName,
    LayoutCorrectorTrainingSplit,
)


def _preserve_torch_worker_seed(worker_id: int) -> None:
    """Stop Lightning from auto-adding its worker_init_fn."""
    del worker_id


class LayoutCorrectorDataModule(LightningDataModule):
    """Construct train, validation, and test loaders from processed LayoutDM data."""

    def __init__(
        self,
        *,
        dataset_name: LayoutCorrectorTrainingDatasetName,
        config: LayoutDMConfig,
        processed_data_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 16,
        max_seq_length: int = 25,
        random_order: bool = True,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a processed LayoutDM data-module configuration."""
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.random_order = random_order
        self.pin_memory = pin_memory
        self._datasets: dict[
            LayoutCorrectorTrainingSplit,
            Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]],
        ] = {}

    def setup(self, stage: str | None = None) -> None:
        """Open the requested processed splits."""
        splits: tuple[LayoutCorrectorTrainingSplit, ...]
        if stage == "fit":
            splits = ("train", "validation")
        elif stage == "test":
            splits = ("test",)
        else:
            splits = ("train", "validation", "test")

        for split in splits:
            self._datasets[split] = LayoutDMProcessedDataset(
                dataset_name=self.dataset_name,
                split=split,
                config=self.config,
                max_seq_length=self.max_seq_length,
                processed_data_dir=self.processed_data_dir,
                random_order=self.random_order,
            )

    def _loader(
        self,
        split: LayoutCorrectorTrainingSplit,
        *,
        shuffle: bool,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        if split not in self._datasets:
            self.setup("fit" if split in {"train", "validation"} else "test")

        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=_preserve_torch_worker_seed,
        )

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the LayoutDM-shaped 16-worker training stream."""
        return self._loader("train", shuffle=True)

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the validation stream."""
        return self._loader("validation", shuffle=False)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]]:
        """Return the test stream."""
        return self._loader("test", shuffle=False)


def first_sample_ids(
    loader: DataLoader[dict[str, Shaped[torch.Tensor, "..."] | str]], count: int = 8
) -> list[str]:
    """Return the first sample identifiers without consuming a second stream."""
    ids: list[str] = []
    for batch in loader:
        sample_ids = batch.get("id")
        if isinstance(sample_ids, list):
            ids.extend(str(value) for value in sample_ids)
        elif isinstance(sample_ids, tuple):
            ids.extend(str(value) for value in sample_ids)
        elif sample_ids is not None:
            ids.append(str(sample_ids))

        if len(ids) >= count:
            break

    return ids[:count]
