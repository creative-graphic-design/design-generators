"""LightningDataModule for RADM training."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import cast

from torch.utils.data import DataLoader

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_MAX_TEXT_NUM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_TEXT_FEATURE_DIM,
    RADMTextFeaturePolicy,
    RADMTrainingSplit,
)
from .dataset import (
    CGLV2ParquetDataset,
    RADMTrainingBatch,
    collate_radm_training_batch,
)

try:
    from lightning.pytorch import LightningDataModule as _LightningDataModule
except ModuleNotFoundError:  # pragma: no cover - exercised without training extra

    class _LightningDataModule:
        """Import fallback used when the training extra is not installed."""

        pass


class RADMDataModule(_LightningDataModule):
    """Minimal Lightning-compatible CGL-v2 datamodule."""

    def __init__(
        self,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        train_split: RADMTrainingSplit = "train",
        val_split: RADMTrainingSplit = "validation",
        batch_size: int = DEFAULT_BATCH_SIZE,
        image_size: int = 800,
        max_elements: int = 100,
        max_text_num: int = DEFAULT_MAX_TEXT_NUM,
        text_feature_dim: int = DEFAULT_TEXT_FEATURE_DIM,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = DEFAULT_PREFETCH_FACTOR,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        text_feature_policy: RADMTextFeaturePolicy | str = RADMTextFeaturePolicy.hf,
    ) -> None:
        """Initialize datamodule settings."""
        super().__init__()
        self.data_root = Path(data_root)
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.max_elements = int(max_elements)
        self.max_text_num = int(max_text_num)
        self.text_feature_dim = int(text_feature_dim)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.text_feature_policy = text_feature_policy
        self.train_dataset: CGLV2ParquetDataset | None = None
        self.val_dataset: CGLV2ParquetDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open datasets for fit/validate stages."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset(
                self.train_split, max_samples=self.max_train_samples
            )
            self.val_dataset = self._dataset(
                self.val_split, max_samples=self.max_val_samples
            )
        if stage == "validate":
            self.val_dataset = self._dataset(
                self.val_split, max_samples=self.max_val_samples
            )

    def train_dataloader(self) -> DataLoader[RADMTrainingBatch]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[RADMTrainingBatch]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")
        return self._loader(self.val_dataset, shuffle=False)

    def _dataset(
        self, split: RADMTrainingSplit, *, max_samples: int | None
    ) -> CGLV2ParquetDataset:
        return CGLV2ParquetDataset(
            data_root=self.data_root,
            split=split,
            image_size=self.image_size,
            max_text_num=self.max_text_num,
            text_feature_dim=self.text_feature_dim,
            max_samples=max_samples,
            text_feature_policy=self.text_feature_policy,
        )

    def _loader(
        self, dataset: CGLV2ParquetDataset | None, *, shuffle: bool
    ) -> DataLoader[RADMTrainingBatch]:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized")
        persistent_workers = self.persistent_workers and self.num_workers > 0
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return cast(
            DataLoader[RADMTrainingBatch],
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=partial(
                    collate_radm_training_batch,
                    max_elements=self.max_elements,
                ),
            ),
        )
