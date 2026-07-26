"""CGL-Dataset-v2 adapter for RADM training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np
from jaxtyping import Bool, Float, Int
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from laygen.common.bbox import ltwh_to_xywh, xywh_to_ltrb

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MAX_TEXT_NUM,
    DEFAULT_TEXT_FEATURE_DIM,
    RADMTextFeaturePolicy,
    RADMTrainingSplit,
)


@dataclass(frozen=True)
class RADMTrainingExample:
    """One RADM training example before collation."""

    image: Float[torch.Tensor, "3 height width"]
    boxes_xyxy: Float[torch.Tensor, "elements 4"]
    labels: Int[torch.Tensor, "elements"]
    text_features: Float[torch.Tensor, "text text_dim"]
    text_mask: Bool[torch.Tensor, "text 1"]
    canvas_size: Int[torch.Tensor, "2"]


@dataclass
class RADMTrainingBatch:
    """Batched tensors consumed by ``RADMTrainingModule``."""

    images: Float[torch.Tensor, "batch 3 height width"]
    boxes_xyxy: Float[torch.Tensor, "batch elements 4"]
    labels: Int[torch.Tensor, "batch elements"]
    mask: Bool[torch.Tensor, "batch elements"]
    text_features: Float[torch.Tensor, "batch text text_dim"]
    text_mask: Bool[torch.Tensor, "batch text 1"]
    canvas_size: Int[torch.Tensor, "batch 2"]


class CGLV2ParquetDataset(Dataset[RADMTrainingExample]):
    """Read staged CGL-Dataset-v2 Parquet shards from the overlay cache."""

    def __init__(
        self,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split: RADMTrainingSplit = "train",
        image_size: int = 800,
        max_text_num: int = DEFAULT_MAX_TEXT_NUM,
        text_feature_dim: int = DEFAULT_TEXT_FEATURE_DIM,
        max_samples: int | None = None,
        text_feature_policy: RADMTextFeaturePolicy | str = RADMTextFeaturePolicy.hf,
    ) -> None:
        """Initialize the staged CGL-v2 dataset.

        Args:
            data_root: Directory containing the staged HF dataset snapshot.
            split: Dataset split name.
            image_size: Square image side used by this training slice.
            max_text_num: Maximum text rows per image.
            text_feature_dim: Feature dimension expected by RADM VTRAM.
            max_samples: Optional cap for smoke and first-iteration runs.
            text_feature_policy: ``"hf"`` uses stored text features;
                ``"zeros"`` pads all text features.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.max_text_num = int(max_text_num)
        self.text_feature_dim = int(text_feature_dim)
        self.max_samples = max_samples
        self.text_feature_policy = RADMTextFeaturePolicy(text_feature_policy)
        self._files = self._discover_files()
        self._lengths = self._read_lengths()
        self._table_file: Path | None = None
        self._table_rows: list[Mapping[str, object]] | None = None

    def __len__(self) -> int:
        """Return number of available examples."""
        total = sum(self._lengths)
        if self.max_samples is None:
            return total
        return min(total, int(self.max_samples))

    def __getitem__(self, index: int) -> RADMTrainingExample:
        """Return one model-ready example."""
        file_path, local_index = self._resolve_index(index)
        row = self._row(file_path, local_index)
        image = self._image_from_row(row)
        width, height = image.size
        resized = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        image_tensor = tvf.pil_to_tensor(resized).float()
        boxes_xyxy, labels = self._annotations_from_row(row, width=width, height=height)
        text_features, text_mask = self._text_from_row(row)
        return RADMTrainingExample(
            image=image_tensor,
            boxes_xyxy=boxes_xyxy,
            labels=labels,
            text_features=text_features,
            text_mask=text_mask,
            canvas_size=torch.tensor([width, height], dtype=torch.long),
        )

    def _discover_files(self) -> list[Path]:
        split_dir = self.data_root / "ralf-style"
        files = sorted(split_dir.glob(f"{self.split}-*.parquet"))
        if files:
            return files
        files = sorted(self.data_root.glob(f"ralf-style/{self.split}-*.parquet"))
        if files:
            return files
        raise FileNotFoundError(
            f"No staged CGL-Dataset-v2 ralf-style {self.split!r} parquet files below "
            f"{self.data_root}. Set RADM_TRAINING_DATA_ROOT or pass data_root."
        )

    def _read_lengths(self) -> list[int]:
        import pyarrow.parquet as pq

        lengths: list[int] = []
        for path in self._files:
            lengths.append(pq.ParquetFile(path).metadata.num_rows)
        return lengths

    def _resolve_index(self, index: int) -> tuple[Path, int]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset = index
        for path, length in zip(self._files, self._lengths, strict=True):
            if offset < length:
                return path, offset
            offset -= length
        raise IndexError(index)

    def _row(self, path: Path, index: int) -> Mapping[str, object]:
        if self._table_file != path:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            rows = cast(list[Mapping[str, object]], table.to_pylist())
            self._table_file = path
            self._table_rows = rows
        if self._table_rows is None:
            raise RuntimeError("Parquet table was not loaded")
        return self._table_rows[index]

    def _image_from_row(self, row: Mapping[str, object]) -> Image.Image:
        for key in ("inpainted_poster", "image", "canvas", "original_poster"):
            value = row.get(key)
            if value is not None:
                return _decode_image(value).convert("RGB")
        raise KeyError("CGL-v2 row does not contain an image column")

    def _annotations_from_row(
        self, row: Mapping[str, object], *, width: int, height: int
    ) -> tuple[Float[torch.Tensor, "elements 4"], Int[torch.Tensor, "elements"]]:
        raw_annotations = row.get("annotations")
        rows = _sequence_rows(raw_annotations)
        boxes_ltwh: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        for item in rows:
            if bool(item.get("iscrowd", False)):
                continue
            bbox_value = item.get("bbox")
            if not isinstance(bbox_value, Sequence):
                continue
            numeric_bbox = cast(Sequence[SupportsFloat], bbox_value)
            left, top, box_width, box_height = (float(value) for value in numeric_bbox)
            if box_width <= 0 or box_height <= 0:
                continue
            boxes_ltwh.append(
                (
                    left / width,
                    top / height,
                    box_width / width,
                    box_height / height,
                )
            )
            labels.append(int(cast(int, item.get("category_id", 0))))
        if not boxes_ltwh:
            boxes_ltwh.append((0.0, 0.0, 1.0, 1.0))
            labels.append(0)
        boxes_xywh = ltwh_to_xywh(torch.tensor(boxes_ltwh, dtype=torch.float32))
        boxes_xyxy = xywh_to_ltrb(boxes_xywh).clamp(0.0, 1.0)
        return boxes_xyxy, torch.tensor(labels, dtype=torch.long)

    def _text_from_row(
        self, row: Mapping[str, object]
    ) -> tuple[Float[torch.Tensor, "text text_dim"], Bool[torch.Tensor, "text 1"]]:
        values = torch.zeros(
            self.max_text_num, self.text_feature_dim, dtype=torch.float32
        )
        mask = torch.zeros(self.max_text_num, 1, dtype=torch.bool)
        if self.text_feature_policy is RADMTextFeaturePolicy.zeros:
            return values, mask
        raw = row.get("text_features")
        if isinstance(raw, Mapping):
            feats = raw.get("feats")
            if isinstance(feats, Sequence):
                count = min(len(feats), self.max_text_num)
                for idx in range(count):
                    flattened = np.asarray(feats[idx], dtype=np.float32).reshape(-1)
                    width = min(flattened.shape[0], self.text_feature_dim)
                    values[idx, :width] = torch.from_numpy(flattened[:width])
                    mask[idx, 0] = True
        return values, mask


def collate_radm_training_batch(
    examples: Sequence[RADMTrainingExample],
    *,
    max_elements: int = 100,
) -> RADMTrainingBatch:
    """Collate RADM examples into padded tensors."""
    images = torch.stack([example.image for example in examples])
    text_features = torch.stack([example.text_features for example in examples])
    text_mask = torch.stack([example.text_mask for example in examples])
    canvas_size = torch.stack([example.canvas_size for example in examples])
    boxes = torch.zeros(len(examples), max_elements, 4, dtype=torch.float32)
    labels = torch.full((len(examples), max_elements), -1, dtype=torch.long)
    mask = torch.zeros(len(examples), max_elements, dtype=torch.bool)
    for batch_index, example in enumerate(examples):
        count = min(example.boxes_xyxy.shape[0], max_elements)
        boxes[batch_index, :count] = example.boxes_xyxy[:count]
        labels[batch_index, :count] = example.labels[:count]
        mask[batch_index, :count] = True
    return RADMTrainingBatch(
        images=images,
        boxes_xyxy=boxes,
        labels=labels,
        mask=mask,
        text_features=text_features,
        text_mask=text_mask,
        canvas_size=canvas_size,
    )


def _decode_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, bytes):
            return Image.open(BytesIO(raw_bytes))
        path = value.get("path")
        if isinstance(path, str):
            return Image.open(path)
    if isinstance(value, bytes):
        return Image.open(BytesIO(value))
    if isinstance(value, str):
        return Image.open(value)
    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def _sequence_rows(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            cast(Mapping[str, object], item)
            for item in value
            if isinstance(item, Mapping)
        ]
    if isinstance(value, Mapping):
        lengths = [
            len(cast(Sequence[object], column))
            for column in value.values()
            if isinstance(column, Sequence) and not isinstance(column, (str, bytes))
        ]
        if not lengths:
            return []
        rows: list[Mapping[str, object]] = []
        for index in range(min(lengths)):
            row: dict[str, object] = {}
            for key, column in value.items():
                if isinstance(column, Sequence) and not isinstance(
                    column, (str, bytes)
                ):
                    row[str(key)] = column[index]
                else:
                    row[str(key)] = column
            rows.append(row)
        return rows
    return []
