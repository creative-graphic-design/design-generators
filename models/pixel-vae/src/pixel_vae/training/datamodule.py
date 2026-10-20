"""Lightning data module for the original filtered Crello image records."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import BinaryIO, NamedTuple, TypeAlias, TypedDict, cast

import lightning as L
import numpy as np
import torch
from jaxtyping import UInt32
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers.image_processing_utils import BatchFeature
from tfrecord import example_pb2
from tfrecord.iterator_utils import shuffle_iterator
from tfrecord.reader import extract_feature_dict, tfrecord_iterator

from ..image_processing_pixel_vae import PixelVAEImageProcessor

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class _RecordLocation:
    """Locate one image payload inside a TFRecord shard."""

    file_index: int
    payload_offset: int
    payload_size: int


class _NumpyRandomState(NamedTuple):
    name: str
    keys: UInt32[np.ndarray, "state"]
    position: int
    has_gauss: int
    cached_gaussian: float


class _RecordDatasetState(TypedDict):
    record_paths: tuple[Path, ...]
    locations: tuple[_RecordLocation, ...]
    _file_handles: dict[int, BinaryIO]
    _owner_pid: int


def _copy_random_state(state: _NumpyRandomState) -> _NumpyRandomState:
    name, keys, position, has_gauss, cached_gaussian = state
    return _NumpyRandomState(
        name,
        keys.copy(),
        position,
        has_gauss,
        cached_gaussian,
    )


def _image_from_payload(payload: bytes | memoryview) -> bytes:
    example = example_pb2.Example()
    example.ParseFromString(payload)
    image = extract_feature_dict(
        example.features, {"image": "byte"}, {"byte": "bytes_list"}
    )["image"]
    if not isinstance(image, bytes):
        raise ValueError("PixelVAE TFRecord image feature must contain bytes")

    return image


class _PixelVAERecordDataset(Dataset[bytes]):
    """Read indexed image records while preserving shard and record positions."""

    def __init__(
        self,
        record_paths: Sequence[Path],
        locations: Sequence[_RecordLocation],
    ) -> None:
        """Store the split files and unique record locations."""
        self.record_paths = tuple(record_paths)
        self.locations = tuple(locations)
        self._file_handles: dict[int, BinaryIO] = {}
        self._owner_pid = os.getpid()

    def __len__(self) -> int:
        """Return the number of unique images in this split."""
        return len(self.locations)

    def __getitem__(self, index: int) -> bytes:
        """Read one image payload by its precomputed TFRecord position."""
        if index < 0 or index >= len(self.locations):
            raise IndexError(index)

        if self._owner_pid != os.getpid():
            self._file_handles = {}
            self._owner_pid = os.getpid()

        location = self.locations[index]
        handle = self._file_handles.get(location.file_index)
        if handle is None:
            handle = self.record_paths[location.file_index].open("rb")
            self._file_handles[location.file_index] = handle

        handle.seek(location.payload_offset)
        payload = handle.read(location.payload_size)
        if len(payload) != location.payload_size:
            raise RuntimeError("Failed to read the indexed PixelVAE TFRecord image")

        return _image_from_payload(payload)

    def __getstate__(self) -> _RecordDatasetState:
        """Drop process-local file handles before a worker dataset is pickled."""
        return {
            "record_paths": self.record_paths,
            "locations": self.locations,
            "_file_handles": {},
            "_owner_pid": os.getpid(),
        }


class _PixelVAERecordSampler(Sampler[int]):
    """Repeat the source shuffle stream and make each epoch resumable by index."""

    def __init__(
        self,
        dataset: _PixelVAERecordDataset,
        batch_size: int,
        shuffle: bool,
        initial_random_state: _NumpyRandomState,
    ) -> None:
        """Store split ordering and the seeded NumPy state for epoch replay."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initial_random_state = initial_random_state
        self.epoch = 0
        self._epoch_was_set = False
        self._next_epoch = 0
        self._stream: Iterator[int] | None = None

    def __len__(self) -> int:
        """Return the original pipeline's number of full-batch image slots."""
        return math.ceil(len(self.dataset) / self.batch_size) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch so a fresh process can reconstruct its source order."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        self.epoch = epoch
        self._epoch_was_set = True

    def __iter__(self) -> Iterator[int]:
        """Yield the original shuffled stream, including repeated tail slots."""
        epoch = self.epoch if self._epoch_was_set else self._next_epoch
        if self._stream is None or epoch != self._next_epoch:
            np.random.set_state(self.initial_random_state)
            self._stream = self._repeat_indices()
            for _ in islice(self._stream, epoch * len(self)):
                pass

        indices = list(islice(self._stream, len(self)))
        self._next_epoch = epoch + 1

        return iter(indices)

    def _repeat_indices(self) -> Iterator[int]:
        while True:
            records = iter(range(len(self.dataset)))
            if self.shuffle:
                yield from shuffle_iterator(records, len(self.dataset))
            else:
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
        num_workers: Number of PyTorch workers that load and decode batches.
        pin_memory: Whether to pin batches for asynchronous CUDA transfers.
    """

    def __init__(
        self,
        data_dir: str = ".cache/pixel-vae/data/crello-image",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Store the prepared dataset location and loader parallelism."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.worker_generator = torch.Generator(device="cpu").manual_seed(0)
        self.processor = PixelVAEImageProcessor()
        self.datasets: dict[str, _PixelVAERecordDataset] = {}
        self.samplers: dict[str, _PixelVAERecordSampler] = {}

    def setup(self, stage: str | None = None) -> None:
        """Index and validate the prepared image-only records for fitting."""
        del stage
        if self.datasets:
            return

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

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

            locations = self._index_unique_records(paths, count)
            datasets[split] = _PixelVAERecordDataset(paths, locations)

        initial_random_state = _copy_random_state(
            cast(_NumpyRandomState, np.random.get_state())
        )
        self.datasets = datasets
        self.samplers = {
            split: _PixelVAERecordSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=split == "train",
                initial_random_state=initial_random_state,
            )
            for split, dataset in datasets.items()
        }

    def _index_unique_records(
        self, record_paths: Sequence[Path], record_count: int
    ) -> tuple[_RecordLocation, ...]:
        seen: set[bytes] = set()
        locations: list[_RecordLocation] = []
        for file_index, path in enumerate(record_paths):
            payload_offset = 0
            for payload in tfrecord_iterator(str(path), index_path=None):
                image = _image_from_payload(payload)
                image_hash = sha256(image).digest()
                if image_hash not in seen:
                    seen.add(image_hash)
                    if len(seen) > record_count:
                        raise ValueError(
                            "count.json record count is smaller than unique TFRecord contents"
                        )

                    locations.append(
                        _RecordLocation(file_index, payload_offset + 12, len(payload))
                    )

                payload_offset += len(payload) + 16

        if len(seen) < record_count:
            raise ValueError("count.json record count exceeds unique TFRecord contents")

        return tuple(locations)

    def _loader(self, split: str) -> DataLoader[BatchFeature]:
        if split not in self.datasets:
            raise RuntimeError("call setup() before requesting data loaders")

        if self.trainer is not None:
            self.samplers[split].set_epoch(self.trainer.current_epoch)

        return cast(
            DataLoader[BatchFeature],
            DataLoader(
                self.datasets[split],
                batch_size=self.batch_size,
                sampler=self.samplers[split],
                collate_fn=self.processor,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
                generator=self.worker_generator,
            ),
        )

    def train_dataloader(self) -> DataLoader[BatchFeature]:
        """Return shuffled full batches from the training split."""
        return self._loader("train")

    def val_dataloader(self) -> DataLoader[BatchFeature]:
        """Return ordered full batches from the repeated validation split."""
        return self._loader("val")


__all__ = ["PixelVAEDataModule"]
