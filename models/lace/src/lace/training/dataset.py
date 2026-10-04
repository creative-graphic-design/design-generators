"""Processed LayoutDM-style datasets used by the LACE training loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from torch.utils.data import Dataset

from .config import LaceTrainingDatasetName, LaceTrainingSplit


class _ProcessedData(Protocol):
    x: Float[torch.Tensor, "rows 4"]
    y: Int[torch.Tensor, "rows"]
    attr: Mapping[str, object]


_PROCESSED_SPLITS: dict[LaceTrainingSplit, str] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


class LaceProcessedDataset(Dataset[dict[str, Shaped[torch.Tensor, "..."] | str]]):
    """Read one processed LACE split without reprocessing source annotations.

    The file layout and row slicing match the original InMemoryDataset.  The
    dataset returns variable-length rows; the package datamodule pads them in
    the same batch-local order as the original sparse-to-dense loader.
    """

    def __init__(
        self,
        *,
        processed_data_dir: str | Path,
        dataset_name: LaceTrainingDatasetName,
        split: LaceTrainingSplit,
        max_seq_length: int = 25,
    ) -> None:
        """Open a processed split from ``<dataset>-max<S>/processed``."""
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        path = (
            Path(processed_data_dir)
            / f"{dataset_name}-max{max_seq_length}"
            / "processed"
            / f"{_PROCESSED_SPLITS[split]}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(path)

        loaded = torch.load(path, map_location="cpu", weights_only=False)

        data, slices = loaded
        self.data = cast(_ProcessedData, data)
        self.slices = cast(Mapping[str, Int[torch.Tensor, "rows"]], slices)
        x_slices = self.slices.get("x")
        if not isinstance(x_slices, torch.Tensor):
            raise TypeError("processed LACE split must contain x slices")

        self.length = int(x_slices.numel() - 1)

    def __len__(self) -> int:
        """Return the number of layouts in the split."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one raw layout row with its source identifier."""
        x_slices = self.slices["x"]
        y_slices = self.slices["y"]
        x_start, x_end = int(x_slices[index]), int(x_slices[index + 1])
        y_start, y_end = int(y_slices[index]), int(y_slices[index + 1])
        bbox = self.data.x[x_start:x_end].float()
        labels = self.data.y[y_start:y_end].long()
        sample_id = self._sample_id(index)
        result: dict[str, Shaped[torch.Tensor, "..."] | str] = {
            "bbox": bbox,
            "labels": labels,
            "mask": torch.ones(labels.shape, dtype=torch.bool),
        }
        if sample_id is not None:
            result["id"] = sample_id

        return result

    def _sample_id(self, index: int) -> str | None:
        attr = getattr(self.data, "attr", None)
        if not isinstance(attr, Mapping):
            return None

        names = attr.get("name")
        if isinstance(names, (list, tuple)) and index < len(names):
            return str(names[index])

        if isinstance(names, str) and index == 0:
            return names

        return None


def collate_lace_batch(
    batch: list[dict[str, Shaped[torch.Tensor, "..."] | str]],
    *,
    max_seq_length: int = 25,
) -> dict[str, Shaped[torch.Tensor, "..."] | list[str]]:
    """Pad a batch to the reference fixed maximum sequence length."""
    if not batch:
        raise ValueError("LACE batches cannot be empty")

    target = max_seq_length
    bboxes: list[Float[torch.Tensor, "elements 4"]] = []
    labels: list[Int[torch.Tensor, "elements"]] = []
    masks: list[Bool[torch.Tensor, "elements"]] = []
    ids: list[str] = []
    for item in batch:
        bbox = cast(Float[torch.Tensor, "elements 4"], item["bbox"])
        label = cast(Int[torch.Tensor, "elements"], item["labels"])
        length = min(int(label.shape[0]), target)

        bbox_pad = torch.zeros(target, 4, dtype=torch.float32)
        label_pad = torch.zeros(target, dtype=torch.long)
        mask_pad = torch.zeros(target, dtype=torch.bool)

        bbox_pad[:length] = bbox[:length]
        label_pad[:length] = label[:length]
        mask_pad[:length] = True

        bboxes.append(bbox_pad)
        labels.append(label_pad)
        masks.append(mask_pad)
        if isinstance(item.get("id"), str):
            ids.append(cast(str, item["id"]))

    result: dict[str, Shaped[torch.Tensor, "..."] | list[str]] = {
        "bbox": torch.stack(bboxes),
        "labels": torch.stack(labels),
        "mask": torch.stack(masks),
    }
    if len(ids) == len(batch):
        result["id"] = ids

    return result
