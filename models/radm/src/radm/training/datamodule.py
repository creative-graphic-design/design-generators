"""Lightning data module for local RADM training inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from laygen.common.randomness import randperm

import torch
from torch.utils.data import BatchSampler, DataLoader, Sampler

from .config import RADMEffectiveConfig
from .dataset import RADMCOCODataset, RADMDataCollator, RADMTrainingExample

from lightning.pytorch import LightningDataModule


class RADMTrainingSampler(Sampler[int]):
    """Yield an infinite seeded stream of shuffled dataset indices."""

    def __init__(self, size: int, *, seed: int) -> None:
        """Initialize the sampler with a fixed dataset size and RNG seed."""
        if size <= 0:
            raise ValueError("RADMTrainingSampler requires a non-empty dataset")

        self.size = size
        self.seed = int(seed)

    def __iter__(self) -> Iterator[int]:
        """Yield successive seeded permutations forever."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        while True:
            yield from randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        """Return the nominal dataset size used by the loader."""
        return self.size


class RADMAspectRatioBatchSampler(BatchSampler):
    """Group the infinite sampler stream into full same-orientation batches."""

    def __init__(
        self,
        dataset: RADMCOCODataset,
        *,
        batch_size: int,
        seed: int,
        max_batches: int,
    ) -> None:
        """Initialize grouped sampling for a bounded number of full batches."""
        sampler = RADMTrainingSampler(len(dataset), seed=seed)
        super().__init__(sampler, batch_size=batch_size, drop_last=True)
        self.dataset = dataset
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield full batches from the two aspect-ratio buckets."""
        buckets: list[list[int]] = [[], []]
        batches = 0
        for index in self.sampler:
            record = self.dataset.images[index]
            bucket_index = 0 if record["width"] > record["height"] else 1
            bucket = buckets[bucket_index]
            bucket.append(index)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                bucket.clear()
                batches += 1
                if batches == self.max_batches:
                    return

    def __len__(self) -> int:
        """Return the configured number of batches."""
        return self.max_batches


class RADMDataModule(LightningDataModule):
    """Load explicit local train/validation COCO and feature paths."""

    def __init__(
        self,
        *,
        train_annotations: str | Path,
        train_image_root: str | Path,
        train_text_feature_root: str | Path,
        val_annotations: str | Path | None = None,
        val_image_root: str | Path | None = None,
        val_text_feature_root: str | Path | None = None,
        test_annotations: str | Path | None = None,
        test_image_root: str | Path | None = None,
        test_text_feature_root: str | Path | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        sampler_seed: int | None = None,
        allow_missing_text_features: bool = False,
        test_read_text_features: bool = True,
        effective: RADMEffectiveConfig,
    ) -> None:
        """Initialize explicit local train and validation data paths."""
        super().__init__()
        self.train_annotations = Path(train_annotations)
        self.train_image_root = Path(train_image_root)
        self.train_text_feature_root = Path(train_text_feature_root)
        self.val_annotations = Path(val_annotations) if val_annotations else None
        self.val_image_root = Path(val_image_root) if val_image_root else None
        self.val_text_feature_root = (
            Path(val_text_feature_root) if val_text_feature_root else None
        )
        self.test_annotations = (
            Path(test_annotations) if test_annotations else self.val_annotations
        )
        self.test_image_root = (
            Path(test_image_root) if test_image_root else self.val_image_root
        )
        self.test_text_feature_root = (
            Path(test_text_feature_root)
            if test_text_feature_root
            else self.val_text_feature_root
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.effective = effective
        self.sampler_seed = (
            self.effective.seed if sampler_seed is None else int(sampler_seed)
        )
        self.allow_missing_text_features = allow_missing_text_features
        self.test_read_text_features = test_read_text_features
        self.train_dataset: RADMCOCODataset | None = None
        self.val_dataset: RADMCOCODataset | None = None
        self.test_dataset: RADMCOCODataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested Lightning stage."""
        if stage in {None, "fit"}:
            self.train_dataset = self._dataset(
                self.train_annotations,
                self.train_image_root,
                self.train_text_feature_root,
                train=True,
            )

            if (
                self.val_annotations is not None
                and self.val_image_root is not None
                and self.val_text_feature_root is not None
            ):
                self.val_dataset = self._dataset(
                    self.val_annotations,
                    self.val_image_root,
                    self.val_text_feature_root,
                    train=False,
                )

        if stage in {None, "test"}:
            if (
                self.test_annotations is not None
                and self.test_image_root is not None
                and self.test_text_feature_root is not None
            ):
                self.test_dataset = self._dataset(
                    self.test_annotations,
                    self.test_image_root,
                    self.test_text_feature_root,
                    train=False,
                    read_text_features=self.test_read_text_features,
                )

    def train_dataloader(self) -> DataLoader[RADMTrainingExample]:
        """Return the seeded grouped training loader."""
        if self.train_dataset is None:
            self.setup("fit")

        if self.train_dataset is None:
            raise RuntimeError("training dataset was not initialized")

        batch_sampler = RADMAspectRatioBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            seed=self.sampler_seed,
            max_batches=self.effective.max_iter,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=RADMDataCollator(effective=self.effective),
        )

    def val_dataloader(self) -> DataLoader[RADMTrainingExample] | None:
        """Return the validation loader when explicit validation paths exist."""
        if self.val_dataset is None:
            self.setup("fit")

        return (
            None
            if self.val_dataset is None
            else self._loader(self.val_dataset, shuffle=False)
        )

    def test_dataloader(self) -> DataLoader[RADMTrainingExample] | None:
        """Return the approved CGL test loader without reshuffling its stream."""
        if self.test_dataset is None:
            self.setup("test")

        return (
            None
            if self.test_dataset is None
            else self._loader(self.test_dataset, shuffle=False)
        )

    def _dataset(
        self,
        annotations: Path,
        image_root: Path,
        text_root: Path,
        *,
        train: bool,
        read_text_features: bool = True,
    ) -> RADMCOCODataset:
        return RADMCOCODataset(
            annotation_path=annotations,
            image_root=image_root,
            text_feature_root=text_root,
            effective=self.effective,
            allow_missing_text_features=self.allow_missing_text_features,
            train=train,
            read_text_features=read_text_features,
        )

    def _loader(
        self, dataset: RADMCOCODataset, *, shuffle: bool
    ) -> DataLoader[RADMTrainingExample]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=RADMDataCollator(effective=self.effective),
        )
