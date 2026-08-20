"""LightningDataModule for LayoutDiffusion training."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence, Sized
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Shaped
from laygen.common.bbox import BoxFormat
from lightning.pytorch import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from ..configuration_layoutdiffusion import LayoutDiffusionConfig
from ..labels import default_id2label
from .config import (
    LayoutDiffusionTrainingDatasetName,
    LayoutDiffusionTrainingDatasetSource,
    LayoutDiffusionTrainingSplit,
    LayoutDiffusionTrainingTransform,
)
from .dataset import (
    LayoutDiffusionDataset,
    LayoutDiffusionProcessedDataset,
    LayoutDiffusionSyntheticDataset,
)
from .vocab import build_training_tokenizer


class _PreconsumeBatchSampler(Sampler[list[int]]):
    def __init__(self, sampler: BatchSampler[int], *, batches: int) -> None:
        self.sampler = sampler
        self.batches = batches

    def __iter__(self) -> Iterator[list[int]]:
        iterator = iter(self.sampler)
        for _ in range(self.batches):
            next(iterator, None)
        yield from iterator

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.batches)


class LayoutDiffusionDataModule(LightningDataModule):
    """Package-local LightningDataModule for LayoutDiffusion data."""

    def __init__(
        self,
        *,
        dataset_name: LayoutDiffusionTrainingDatasetName = "publaynet",
        config: LayoutDiffusionConfig,
        batch_size: int = 64,
        max_num_elements: int | None = None,
        num_workers: int = 4,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        synthetic_size: int | None = None,
        dataset_source: LayoutDiffusionTrainingDatasetSource = "hf",
        processed_data_dir: str | None = None,
        vocab_file: str | None = None,
        preconsume_train_batches: int = 0,
        processed_stream_rng_warmup: bool = False,
        train_transforms: Sequence[LayoutDiffusionTrainingTransform] | None = (
            "LexicographicOrder",
        ),
    ) -> None:
        """Initialize datamodule settings."""
        super().__init__()
        if preconsume_train_batches < 0:
            raise ValueError("preconsume_train_batches must be non-negative")

        if dataset_source == "hf" and (
            vocab_file is not None or config.id2label != default_id2label(dataset_name)
        ):
            raise ValueError(
                "hf source with non-default id2label is not supported: dataset "
                "numeric labels would be interpreted under a different id2label; "
                "use dataset_source='processed'"
            )

        if vocab_file is not None:
            _validate_vocab_label_count(vocab_file, dataset_name)

        self.dataset_name: LayoutDiffusionTrainingDatasetName = dataset_name
        self.config = config
        self.batch_size = batch_size
        self.max_num_elements = max_num_elements or self.config.max_num_elements

        self.num_workers = num_workers
        self.box_format = box_format
        self.normalized = normalized
        self.synthetic_size = synthetic_size
        self.dataset_source = dataset_source
        self.processed_data_dir = processed_data_dir

        self.vocab_file = vocab_file
        self.preconsume_train_batches = preconsume_train_batches
        self.processed_stream_rng_warmup = processed_stream_rng_warmup
        self.train_transforms = tuple(train_transforms or ())

        unsupported = set(self.train_transforms) - {"LexicographicOrder"}
        if unsupported:
            raise ValueError(
                f"Unsupported LayoutDiffusion train transforms: {unsupported}"
            )

        self.tokenizer = build_training_tokenizer(self.config, vocab_file=vocab_file)
        self.train_dataset: (
            Dataset[dict[str, Shaped[torch.Tensor, ...] | str]] | None
        ) = None
        self.val_dataset: Dataset[dict[str, Shaped[torch.Tensor, ...] | str]] | None = (
            None
        )
        self.test_dataset: (
            Dataset[dict[str, Shaped[torch.Tensor, ...] | str]] | None
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
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | str]]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(
            self.train_dataset,
            shuffle=self.synthetic_size is None,
            preconsume_batches=self.preconsume_train_batches,
            rng_warmup=self.processed_stream_rng_warmup,
            drop_last=self.dataset_source == "processed",
        )

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | str]]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            self.setup("fit")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | str]]:
        """Return the test dataloader."""
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(self.test_dataset, shuffle=False)

    def _dataset(
        self, split: LayoutDiffusionTrainingSplit
    ) -> Dataset[dict[str, Shaped[torch.Tensor, ...] | str]]:
        if self.synthetic_size is not None:
            return LayoutDiffusionSyntheticDataset(
                config=self.config,
                size=self.synthetic_size,
                elements=min(3, self.max_num_elements),
            )
        if self.dataset_source == "processed":
            if self.processed_data_dir is None:
                raise ValueError(
                    "processed_data_dir is required when dataset_source='processed'"
                )

            return LayoutDiffusionProcessedDataset(
                dataset_name=self.dataset_name,
                split=split,
                config=self.config,
                tokenizer=self.tokenizer,
                processed_data_dir=self.processed_data_dir,
            )
        return LayoutDiffusionDataset(
            dataset_name=self.dataset_name,
            split=split,
            config=self.config,
            tokenizer=self.tokenizer,
            max_num_elements=self.max_num_elements,
            box_format=self.box_format,
            normalized=self.normalized,
            lexicographic_order=self._uses_lexicographic_order(split),
        )

    def _uses_lexicographic_order(self, split: LayoutDiffusionTrainingSplit) -> bool:
        del split
        return "LexicographicOrder" in self.train_transforms

    def _loader(
        self,
        dataset: Dataset[dict[str, Shaped[torch.Tensor, ...] | str]] | None,
        *,
        shuffle: bool,
        preconsume_batches: int = 0,
        rng_warmup: bool = False,
        drop_last: bool = False,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | str]]:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialized")

        if rng_warmup:
            _warmup_processed_stream_rng(self.config.vocab_size)
        if preconsume_batches == 0:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.num_workers,
            )
        sampler: Sampler[int]
        sized_dataset = cast(Sized, dataset)
        if shuffle:
            sampler = RandomSampler(sized_dataset)
        else:
            sampler = SequentialSampler(sized_dataset)
        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=drop_last)
        return DataLoader(
            dataset,
            batch_sampler=_PreconsumeBatchSampler(
                batch_sampler, batches=preconsume_batches
            ),
            num_workers=self.num_workers,
        )


def _warmup_processed_stream_rng(vocab_size: int) -> None:
    random_embedding = torch.nn.Embedding(vocab_size - 1, 8)
    torch.nn.init.normal_(random_embedding.weight)


def _validate_vocab_label_count(
    vocab_file: str, dataset_name: LayoutDiffusionTrainingDatasetName
) -> None:
    raw_vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
    labels = [
        str(token)
        for token in raw_vocab
        if str(token) not in {"START", "END", "UNK", "PAD", "|", "MASK"}
        and not str(token).isdigit()
    ]
    expected = len(default_id2label(dataset_name))
    actual = len(labels)
    if actual != expected:
        raise ValueError(
            "LayoutDiffusion vocab label count mismatch: "
            f"dataset={dataset_name} expects {expected} labels but "
            f"vocab_file={vocab_file} has {actual} labels"
        )
