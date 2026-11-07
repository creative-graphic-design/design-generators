"""Lightning data modules for prepared CanvasVAE RICO and Crello splits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import (
    CrelloBatch,
    CrelloDocument,
    CrelloProcessor,
    CrelloSplit,
    fixture_image_ids,
    load_crello_split,
    load_crello_vocabularies,
    load_embedding_fixture,
)
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


class CrelloLayoutDataset(Dataset[CrelloDocument]):
    """Prepared Crello documents in the canonical split order."""

    def __init__(self, documents: Sequence[CrelloDocument]) -> None:
        """Store the documents."""
        self.documents = documents

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self.documents)

    def __getitem__(self, index: int) -> CrelloDocument:
        """Return one prepared document."""
        return self.documents[index]


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


class CanvasVAECrelloDataModule(L.LightningDataModule):
    """Serve prepared Crello documents with their verified embedding fixture.

    Training batches use shuffled cross-epoch passes. Validation wraps the
    canonical validation order to full batches, and testing uses one ordered
    pass. The fixture loader verifies every row against the canonical image
    IDs before the processor is constructed.

    Args:
        data_dir: Directory containing prepared JSONL splits and vocabulary.
        fixture_dir: Directory containing the shared posterior-mean fixture.
        batch_size: Number of documents per batch.
        num_workers: Data-loader worker processes.
        seed: Seed of the training stream; defaults to the global Torch seed.
    """

    def __init__(
        self,
        data_dir: str = ".cache/canvas-vae/crello/package-run-1",
        fixture_dir: str = ".cache/canvas-vae/crello/fixture",
        batch_size: int = 1024,
        num_workers: int = 0,
        seed: int | None = None,
    ) -> None:
        """Store loader settings."""
        super().__init__()
        self.data_dir = data_dir
        self.fixture_dir = fixture_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits: dict[CrelloSplit, CrelloLayoutDataset] = {}
        self.processor: CrelloProcessor | None = None
        self.train_sampler: CrossEpochBatchSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load prepared documents and verify the shared embedding fixture.

        Args:
            stage: Lightning stage name.
        """
        del stage
        if self.processor is not None:
            return

        documents = {
            split: load_crello_split(self.data_dir, split) for split in CrelloSplit
        }
        ordered_documents = [row for split in CrelloSplit for row in documents[split]]
        embeddings = load_embedding_fixture(
            self.fixture_dir, fixture_image_ids(ordered_documents)
        )
        self.processor = CrelloProcessor(
            load_crello_vocabularies(self.data_dir), embeddings
        )
        self.splits = {
            split: CrelloLayoutDataset(documents[split]) for split in CrelloSplit
        }
        seed = torch.initial_seed() if self.seed is None else self.seed
        self.train_sampler = CrossEpochBatchSampler(
            len(self.splits[CrelloSplit.train]),
            self.batch_size,
            torch.Generator().manual_seed(seed),
        )

    def _loader(
        self,
        split: CrelloSplit,
        batches: CrossEpochBatchSampler | list[list[int]] | None,
    ) -> DataLoader[CrelloBatch]:
        if self.processor is None or batches is None:
            raise RuntimeError("call setup() before requesting data loaders")

        return cast(
            DataLoader[CrelloBatch],
            DataLoader(
                self.splits[split],
                batch_sampler=batches,
                collate_fn=cast(
                    Callable[[Sequence[CrelloDocument]], CrelloBatch],
                    self.processor,
                ),
                num_workers=self.num_workers,
            ),
        )

    def train_dataloader(self) -> DataLoader[CrelloBatch]:
        """Return the shuffled cross-epoch training stream."""
        return self._loader(CrelloSplit.train, self.train_sampler)

    def val_dataloader(self) -> DataLoader[CrelloBatch]:
        """Return full batches over a repeated ordered validation pass."""
        if not self.splits:
            raise RuntimeError("call setup() before requesting data loaders")

        size = len(self.splits[CrelloSplit.val])
        return self._loader(CrelloSplit.val, wrapping_batches(size, self.batch_size))

    def test_dataloader(self) -> DataLoader[CrelloBatch]:
        """Return one ordered test pass."""
        if not self.splits:
            raise RuntimeError("call setup() before requesting data loaders")

        size = len(self.splits[CrelloSplit.test])
        return self._loader(CrelloSplit.test, sequential_batches(size, self.batch_size))


__all__ = [
    "CanvasVAECrelloDataModule",
    "CanvasVAEDataModule",
    "CrelloLayoutDataset",
    "RicoLayoutDataset",
]
