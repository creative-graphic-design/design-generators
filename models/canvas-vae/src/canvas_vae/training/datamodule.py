"""Lightning data module for CanvasVAE on prepared RICO splits."""

from __future__ import annotations

from collections.abc import Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from ..processing_canvas_vae import (
    CanvasVAEProcessor,
    RicoDocument,
    RicoElement,
    RicoSplit,
    load_rico_split,
    load_rico_vocabularies,
)
from .sampling import CrossEpochBatchSampler, sequential_batches, wrapping_batches


class RicoLayoutDataset(Dataset[list[RicoElement]]):
    """Element lists of prepared RICO documents.

    Args:
        documents: Prepared documents.
    """

    def __init__(self, documents: Sequence[RicoDocument]) -> None:
        """Store the documents."""
        self.documents = documents

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self.documents)

    def __getitem__(self, index: int) -> list[RicoElement]:
        """Return the elements of one document."""
        return self.documents[index]["elements"]


class CanvasVAEDataModule(L.LightningDataModule):
    """Serve RICO splits prepared by ``scripts/prepare_rico.py``.

    Training batches come from one endless stream of shuffled passes with
    ``ceil(train_size / batch_size)`` full batches per epoch. Validation
    scores the same number of full batches over a repeated ordered pass, and
    testing scores one ordered pass. Every batch is padded to its longest
    layout.

    Args:
        data_dir: Prepared split directory.
        batch_size: Batch size.
        num_workers: Data-loader worker processes.
        seed: Seed of the training stream. Defaults to the global Torch seed.
    """

    def __init__(
        self,
        data_dir: str = ".cache/canvas-vae/data/rico",
        batch_size: int = 1024,
        num_workers: int = 0,
        seed: int | None = None,
    ) -> None:
        """Store loader settings."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits: dict[RicoSplit, RicoLayoutDataset] = {}
        self.processor: CanvasVAEProcessor | None = None
        self.train_sampler: CrossEpochBatchSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load splits and vocabularies once.

        Args:
            stage: Lightning stage name.
        """
        del stage
        if self.processor is not None:
            return

        self.processor = CanvasVAEProcessor(load_rico_vocabularies(self.data_dir))
        self.splits = {
            split: RicoLayoutDataset(load_rico_split(self.data_dir, split))
            for split in RicoSplit
        }
        seed = torch.initial_seed() if self.seed is None else self.seed
        self.train_sampler = CrossEpochBatchSampler(
            len(self.splits[RicoSplit.train]),
            self.batch_size,
            torch.Generator().manual_seed(seed),
        )

    def _loader(
        self, split: RicoSplit, batches: CrossEpochBatchSampler | list[list[int]] | None
    ) -> DataLoader[list[RicoElement]]:
        if self.processor is None or batches is None:
            raise RuntimeError("call setup() before requesting data loaders")

        return DataLoader(
            self.splits[split],
            batch_sampler=batches,
            collate_fn=self.processor,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader[list[RicoElement]]:
        """Return the cross-epoch training stream."""
        return self._loader(RicoSplit.train, self.train_sampler)

    def val_dataloader(self) -> DataLoader[list[RicoElement]]:
        """Return full batches over a repeated ordered validation pass."""
        size = len(self.splits.get(RicoSplit.val, ()))
        return self._loader(RicoSplit.val, wrapping_batches(size, self.batch_size))

    def test_dataloader(self) -> DataLoader[list[RicoElement]]:
        """Return one ordered test pass."""
        size = len(self.splits.get(RicoSplit.test, ()))
        return self._loader(RicoSplit.test, sequential_batches(size, self.batch_size))


__all__ = ["CanvasVAEDataModule", "RicoLayoutDataset"]
