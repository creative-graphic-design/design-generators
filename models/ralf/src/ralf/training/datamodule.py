"""Package-local RALF training datasets and LightningDataModule."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from ..configuration_ralf import RalfConfig
from ..datasets import normalize_org_sample
from ..retrieval import (
    RalfRetrievalTable,
    RalfRetrievedBatch,
)
from ..tokenization_ralf import RalfLayoutTokenizer

RalfSampleScalar: TypeAlias = str | int | float | bool | bytes | Image.Image
RalfSampleValue: TypeAlias = (
    RalfSampleScalar
    | Sequence["RalfSampleValue"]
    | Mapping[str, "RalfSampleValue"]
    | None
)


def _as_image(
    value: RalfSampleValue | Shaped[torch.Tensor, ...], *, channels: int
) -> Float[torch.Tensor, "channels height width"]:
    if isinstance(value, Tensor):
        image = value.float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(
                f"expected image tensor with 2 or 3 dimensions, got {image.shape}"
            )
        if image.numel() and image.max().item() > 1.0:
            image = image / 255.0
    elif isinstance(value, Image.Image):
        image = to_tensor(value)
    else:
        image = torch.zeros(channels, 64, 64)
    if image.size(0) != channels:
        if channels == 1:
            image = image.mean(dim=0, keepdim=True)
        elif image.size(0) == 1:
            image = image.expand(channels, -1, -1)
        else:
            image = image[:channels]
    return image


def _label_mapping(config: RalfConfig) -> dict[str, int]:
    return {str(label): int(index) for index, label in config.id2label.items()}


def _normalize_for_config(
    sample: Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]],
    config: RalfConfig,
) -> dict[str, RalfSampleValue | Shaped[torch.Tensor, ...]]:
    normalized = normalize_org_sample(sample, config.dataset_name)
    raw_labels = sample.get("label")
    if (
        isinstance(raw_labels, Sequence)
        and not isinstance(raw_labels, (str, bytes))
        and raw_labels
        and isinstance(raw_labels[0], str)
    ):
        mapping = _label_mapping(config)
        normalized["labels"] = torch.tensor(
            [mapping[str(label)] for label in raw_labels], dtype=torch.long
        )
    return normalized


def _sorted_layout(
    sample: Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]],
    config: RalfConfig,
) -> dict[str, Shaped[torch.Tensor, ...]]:
    normalized = _normalize_for_config(sample, config)
    labels = cast(Int[torch.Tensor, "elements"], normalized["labels"]).long()
    bbox = cast(Float[torch.Tensor, "elements 4"], normalized["bbox"]).float()
    mask = cast(Bool[torch.Tensor, "elements"], normalized["mask"]).bool()
    valid = [idx for idx, flag in enumerate(mask.tolist()) if flag]
    label_order = sorted(
        valid,
        key=lambda idx: int(labels[idx].item()),
    )
    labels = labels[label_order]
    bbox = bbox[label_order]
    source_geometry = [
        sample.get(key) for key in ("center_x", "center_y", "width", "height")
    ]
    if all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in source_geometry
    ):
        source_center_x, source_center_y, source_width, source_height = (
            cast(Sequence[int | float], value) for value in source_geometry
        )
        source_left = [
            float(source_center_x[idx]) - float(source_width[idx]) / 2.0
            for idx in label_order
        ]
        source_top = [
            float(source_center_y[idx]) - float(source_height[idx]) / 2.0
            for idx in label_order
        ]
    else:
        source_left = [
            float((bbox[idx, 0] - bbox[idx, 2] / 2).item())
            for idx in range(len(label_order))
        ]
        source_top = [
            float((bbox[idx, 1] - bbox[idx, 3] / 2).item())
            for idx in range(len(label_order))
        ]
    lexicographic_order = sorted(
        range(len(label_order)),
        key=lambda idx: (source_top[idx], source_left[idx]),
    )
    if not lexicographic_order:
        labels = torch.tensor([0], dtype=torch.long)
        bbox = torch.tensor([[0.5, 0.5, 0.05, 0.05]], dtype=torch.float32)
    else:
        labels = labels[lexicographic_order]
        bbox = bbox[lexicographic_order]
    padded_labels = torch.zeros(config.max_seq_length, dtype=torch.long)
    padded_bbox = torch.zeros(config.max_seq_length, 4, dtype=torch.float32)
    padded_mask = torch.zeros(config.max_seq_length, dtype=torch.bool)
    length = min(config.max_seq_length, labels.numel())
    padded_labels[:length] = labels[:length]
    padded_bbox[:length] = bbox[:length]
    padded_mask[:length] = True
    return {"labels": padded_labels, "bbox": padded_bbox, "mask": padded_mask}


def _retrieved_from_samples(
    *,
    indexes: Sequence[int],
    samples: Sequence[Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]]],
    config: RalfConfig,
) -> RalfRetrievedBatch:
    if len(indexes) != config.top_k:
        raise ValueError(
            f"retrieval row has {len(indexes)} candidates, expected {config.top_k}"
        )
    images: list[Float[torch.Tensor, "channels height width"]] = []
    saliency: list[Float[torch.Tensor, "1 height width"]] = []
    layouts: list[dict[str, Shaped[torch.Tensor, ...]]] = []
    for index in indexes:
        if index < 0 or index >= len(samples):
            raise ValueError(f"retrieval index {index} is outside the training dataset")
        row = samples[int(index)]
        layouts.append(_sorted_layout(row, config))
        images.append(_as_image(row.get("image"), channels=3))
        saliency.append(_as_image(row.get("saliency"), channels=1))
    bbox = torch.stack([cast(Tensor, item["bbox"]) for item in layouts])
    labels = torch.stack([cast(Tensor, item["labels"]) for item in layouts])
    mask = torch.stack([cast(Tensor, item["mask"]) for item in layouts])
    return RalfRetrievedBatch(
        image=torch.stack(images).unsqueeze(0),
        saliency=torch.stack(saliency).unsqueeze(0),
        bbox=bbox.unsqueeze(0),
        labels=labels.unsqueeze(0),
        mask=mask.unsqueeze(0),
        indexes=torch.tensor([list(indexes)], dtype=torch.long),
    )


def encode_training_sample(
    sample: Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]],
    *,
    config: RalfConfig,
    retrieval_indexes: Sequence[int],
    retrieval_samples: Sequence[
        Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]]
    ],
) -> dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]:
    """Encode one sample with the package tokenizer and explicit retrieval."""
    layout = _sorted_layout(sample, config)
    labels = cast(Tensor, layout["labels"]).unsqueeze(0)
    bbox = cast(Tensor, layout["bbox"]).unsqueeze(0)
    mask = cast(Tensor, layout["mask"]).unsqueeze(0)
    encoded = RalfLayoutTokenizer(config).encode_layout(
        labels=labels,
        bbox=bbox,
        mask=mask,
    )
    sequence = cast(Tensor, encoded["input_ids"])
    sequence_mask = cast(Tensor, encoded["attention_mask"])
    retrieved = _retrieved_from_samples(
        indexes=retrieval_indexes,
        samples=retrieval_samples,
        config=config,
    )
    return {
        "input_ids": sequence[0, :-1].long(),
        "labels": sequence[0, 1:].long(),
        "attention_mask": sequence_mask[0, :-1].bool(),
        "pixel_values": _as_image(sample.get("image"), channels=3),
        "saliency": _as_image(sample.get("saliency"), channels=1),
        "layout_labels": labels[0].long(),
        "layout_bbox": bbox[0].float(),
        "layout_mask": mask[0].bool(),
        "retrieved": retrieved,
    }


class RalfTrainingDataset(
    Dataset[dict[str, Shaped[torch.Tensor, "..."] | RalfRetrievedBatch]]
):
    """Indexable deterministic dataset for package-local RALF training."""

    def __init__(
        self,
        *,
        samples: Sequence[Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]]],
        config: RalfConfig,
        retrieval_table: RalfRetrievalTable | Mapping[int | str, Sequence[int]],
        retrieval_samples: Sequence[
            Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]]
        ],
    ) -> None:
        """Initialize the dataset with samples and retrieval rows."""
        if not retrieval_samples:
            raise ValueError("retrieval samples are required for RALF training")
        self.samples = samples
        self.config = config
        self.retrieval_samples = retrieval_samples
        self.retrieval_table = (
            retrieval_table
            if isinstance(retrieval_table, RalfRetrievalTable)
            else RalfRetrievalTable(retrieval_table, top_k=config.top_k)
        )
        if not self.retrieval_table.table:
            raise ValueError("retrieval table is required for RALF training")

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]:
        """Encode one indexed sample for package training."""
        sample = self.samples[index]
        sample_id = sample.get("id", index)
        retrieval_indexes = self.retrieval_table.lookup([cast(str | int, sample_id)])[
            0
        ].tolist()
        if any(item < 0 for item in retrieval_indexes):
            raise ValueError(
                f"retrieval table has no complete row for sample {sample_id!r}"
            )
        return encode_training_sample(
            sample,
            config=self.config,
            retrieval_indexes=retrieval_indexes,
            retrieval_samples=self.retrieval_samples,
        )


def collate_training_batch(
    batch: Sequence[dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]],
) -> dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]:
    """Collate package samples without substituting retrieval data."""
    output: dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch] = {}
    for key in (
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "saliency",
        "layout_labels",
        "layout_bbox",
        "layout_mask",
    ):
        output[key] = torch.stack([cast(Tensor, item[key]) for item in batch])
    retrieved = [cast(RalfRetrievedBatch, item["retrieved"]) for item in batch]
    output["retrieved"] = RalfRetrievedBatch(
        image=torch.cat([item.image for item in retrieved]),
        saliency=torch.cat([item.saliency for item in retrieved]),
        bbox=torch.cat([item.bbox for item in retrieved]),
        labels=torch.cat([item.labels for item in retrieved]),
        mask=torch.cat([item.mask for item in retrieved]),
        indexes=torch.cat([cast(Tensor, item.indexes) for item in retrieved]),
    )
    return output


class RalfDataModule(LightningDataModule):
    """LightningDataModule backed by the RALF dataset and retrieval cache."""

    def __init__(
        self,
        *,
        config: RalfConfig,
        data_root: str | None = None,
        retrieval_index_path: str | None = None,
        validation_retrieval_index_path: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_split: str = "train",
        validation_split: str = "val",
    ) -> None:
        """Initialize the package training data module."""
        super().__init__()
        self.config = config
        self.data_root = data_root or os.environ.get("RALF_DATA_ROOT")
        self.retrieval_index_path = retrieval_index_path
        self.validation_retrieval_index_path = validation_retrieval_index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.validation_split = validation_split
        self.train_dataset: RalfTrainingDataset | None = None
        self.validation_dataset: RalfTrainingDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load package samples and their exact retrieval rows."""
        if stage not in {None, "fit"}:
            return
        train_samples = self._load_split(self.train_split)
        validation_samples = self._load_split(self.validation_split)
        self.train_dataset = RalfTrainingDataset(
            samples=train_samples,
            config=self.config,
            retrieval_table=self._load_retrieval_table(self.train_split),
            retrieval_samples=train_samples,
        )
        self.validation_dataset = RalfTrainingDataset(
            samples=validation_samples,
            config=self.config,
            retrieval_table=self._load_retrieval_table(
                self.validation_split, validation=True
            ),
            retrieval_samples=train_samples,
        )

    def train_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]]:
        """Return the package training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")
        if self.train_dataset is None:
            raise RuntimeError("training dataset was not initialized")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_training_batch,
            drop_last=False,
        )

    def val_dataloader(
        self,
    ) -> DataLoader[dict[str, Shaped[torch.Tensor, ...] | RalfRetrievedBatch]]:
        """Return the package validation dataloader."""
        if self.validation_dataset is None:
            self.setup("fit")
        if self.validation_dataset is None:
            raise RuntimeError("validation dataset was not initialized")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_training_batch,
            drop_last=False,
        )

    def _load_split(
        self, split: str
    ) -> Sequence[Mapping[str, RalfSampleValue | Shaped[torch.Tensor, ...]]]:
        if self.data_root is None:
            raise ValueError("RALF_DATA_ROOT or data_root is required for training")
        from datasets import load_dataset

        dataset_dir = Path(self.data_root) / (
            "cgl" if self.config.dataset_name.startswith("cgl") else "pku"
        )
        paths = sorted(dataset_dir.glob(f"{split}-*.parquet"))
        if not paths and split == "val":
            paths = sorted(dataset_dir.glob("validation-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no {split} parquet files under {dataset_dir}")
        return cast(
            Sequence[Mapping[str, RalfSampleValue | Shaped[torch.Tensor, "..."]]],
            load_dataset(
                "parquet",
                data_files={split: [str(path) for path in paths]},
                split=split,
            ),
        )

    def _load_retrieval_table(
        self, split: str, *, validation: bool = False
    ) -> RalfRetrievalTable:
        path = (
            self.validation_retrieval_index_path
            if validation and self.validation_retrieval_index_path is not None
            else self.retrieval_index_path
        )
        if path is None:
            raise ValueError("retrieval_index_path is required for RALF training")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"retrieval index at {path} is not a mapping")
        _ = split
        return RalfRetrievalTable(
            cast(Mapping[int | str, Sequence[int]], payload), self.config.top_k
        )
