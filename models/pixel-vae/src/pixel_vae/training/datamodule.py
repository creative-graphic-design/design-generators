"""Lightning data module for the original filtered Crello image records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import TypeAlias, cast

import lightning as L
from torch.utils.data import DataLoader, IterableDataset
from transformers.image_processing_utils import BatchFeature
from tfrecord.iterator_utils import shuffle_iterator
from tfrecord.reader import tfrecord_loader

from ..image_processing_pixel_vae import PixelVAEImageProcessor

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class _PixelVAERecordDataset(IterableDataset[bytes]):
    """Read a split and wrap its final partial batch like the source pipeline."""

    def __init__(
        self,
        record_paths: Sequence[Path],
        record_count: int,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        """Store the split files and the source record count."""
        super().__init__()
        self.record_paths = tuple(record_paths)
        self.record_count = record_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._stream: Iterator[bytes] | None = None

    def __len__(self) -> int:
        """Return the number of full image slots consumed in one epoch."""
        return math.ceil(self.record_count / self.batch_size) * self.batch_size

    def __iter__(self) -> Iterator[bytes]:
        """Continue the ordered or shuffled source stream for one epoch."""
        if self._stream is None:
            self._stream = self._repeat_records()

        return islice(self._stream, len(self))

    def _repeat_records(self) -> Iterator[bytes]:
        while True:
            records = self._checked_pass(self._records())
            if self.shuffle:
                records = shuffle_iterator(records, self.record_count)

            yield from records

    def _checked_pass(self, records: Iterator[Mapping[str, bytes]]) -> Iterator[bytes]:
        seen: set[bytes] = set()
        for record in records:
            image = record.get("image")
            if not isinstance(image, bytes):
                raise ValueError("PixelVAE TFRecord image feature must contain bytes")

            image_hash = sha256(image).digest()
            if image_hash in seen:
                continue

            seen.add(image_hash)
            if len(seen) > self.record_count:
                raise ValueError(
                    "count.json record count is smaller than unique TFRecord contents"
                )

            yield image

        if len(seen) < self.record_count:
            raise ValueError("count.json record count exceeds unique TFRecord contents")

    def _records(self) -> Iterator[Mapping[str, bytes]]:
        for path in self.record_paths:
            records = cast(
                Iterator[Mapping[str, bytes]],
                tfrecord_loader(
                    str(path), index_path=None, description={"image": "byte"}
                ),
            )
            yield from records


class PixelVAEDataModule(L.LightningDataModule):
    """Load original image-only Crello records for PixelVAE training.

    ``data_dir`` contains ``count.json`` and ``train-*.tfrecord`` plus
    ``val-*.tfrecord`` files produced by the original image-record transform.
    That transform keeps ``imageElement``, ``maskElement``, and ``svgElement``
    images and deduplicates identical PNG bytes within each split.

    Args:
        data_dir: Directory containing filtered image records and split counts.
        batch_size: Number of images per original training and validation batch.
    """

    def __init__(
        self,
        data_dir: str = ".cache/pixel-vae/data/crello-image",
        batch_size: int = 64,
    ) -> None:
        """Store the prepared dataset location and batch size."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.processor = PixelVAEImageProcessor()
        self.datasets: dict[str, _PixelVAERecordDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        """Validate the prepared image-only records for fitting."""
        del stage
        if self.datasets:
            return

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        count_path = self.data_dir / "count.json"
        try:
            raw_counts = cast(
                JsonValue, json.loads(count_path.read_text(encoding="utf-8"))
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"missing PixelVAE split counts: {count_path}"
            ) from error

        if not isinstance(raw_counts, dict):
            raise ValueError("count.json must contain a split-to-count object")

        datasets: dict[str, _PixelVAERecordDataset] = {}
        for split in ("train", "val"):
            count = raw_counts.get(split)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"count.json must contain a positive {split} count")

            paths = sorted(self.data_dir.glob(f"{split}-*.tfrecord"))
            if not paths:
                raise FileNotFoundError(
                    f"no {split}-*.tfrecord files in {self.data_dir}"
                )

            datasets[split] = _PixelVAERecordDataset(
                paths,
                record_count=count,
                batch_size=self.batch_size,
                shuffle=split == "train",
            )

        self.datasets = datasets

    def _loader(self, split: str) -> DataLoader[BatchFeature]:
        if split not in self.datasets:
            raise RuntimeError("call setup() before requesting data loaders")

        return cast(
            DataLoader[BatchFeature],
            DataLoader(
                self.datasets[split],
                batch_size=self.batch_size,
                collate_fn=self.processor,
            ),
        )

    def train_dataloader(self) -> DataLoader[BatchFeature]:
        """Return shuffled full batches from the training split."""
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[BatchFeature]:
        """Return ordered full batches from the repeated validation split."""
        return self._loader("val")


__all__ = ["PixelVAEDataModule"]
