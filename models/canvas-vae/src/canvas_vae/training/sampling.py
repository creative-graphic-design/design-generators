"""Batch index streams for CanvasVAE training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from laygen.common.randomness import randperm
from torch.utils.data import Sampler


class CrossEpochBatchSampler(Sampler[list[int]]):
    """Yield fixed-size batches from one endless stream of shuffled passes.

    The stream concatenates independent permutations of the records, and each
    epoch takes the next ``ceil(num_records / batch_size)`` batches from it, so
    batches straddle pass boundaries and every batch is full.

    Args:
        num_records: Number of training records.
        batch_size: Batch size.
        generator: Generator for the permutations.

    Examples:
        >>> sampler = CrossEpochBatchSampler(5, 2, torch.Generator().manual_seed(0))
        >>> [len(batch) for batch in sampler], len(sampler)
        ([2, 2, 2], 3)
    """

    def __init__(
        self, num_records: int, batch_size: int, generator: torch.Generator
    ) -> None:
        """Store the stream configuration."""
        self.num_records = num_records
        self.batch_size = batch_size
        self.generator = generator
        self.buffer: list[int] = []

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return math.ceil(self.num_records / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield the next epoch of batches from the stream."""
        for _ in range(len(self)):
            while len(self.buffer) < self.batch_size:
                self.buffer.extend(
                    randperm(self.num_records, generator=self.generator).tolist()
                )

            batch, self.buffer = (
                self.buffer[: self.batch_size],
                self.buffer[self.batch_size :],
            )
            yield batch


def wrapping_batches(num_records: int, batch_size: int) -> list[list[int]]:
    """Return ``ceil(num_records / batch_size)`` full batches of a repeated pass.

    Records are taken in order and the last batch wraps to the first records,
    so some records are scored twice.

    Args:
        num_records: Number of records.
        batch_size: Batch size.

    Returns:
        Batches of record indices.

    Examples:
        >>> wrapping_batches(3, 2)
        [[0, 1], [2, 0]]
    """
    steps = math.ceil(num_records / batch_size)
    indices = [index % num_records for index in range(steps * batch_size)]
    return [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]


def sequential_batches(num_records: int, batch_size: int) -> list[list[int]]:
    """Return one ordered pass whose last batch may be smaller.

    Args:
        num_records: Number of records.
        batch_size: Batch size.

    Returns:
        Batches of record indices.

    Examples:
        >>> sequential_batches(3, 2)
        [[0, 1], [2]]
    """
    return [
        list(range(start, min(start + batch_size, num_records)))
        for start in range(0, num_records, batch_size)
    ]


__all__ = ["CrossEpochBatchSampler", "sequential_batches", "wrapping_batches"]
