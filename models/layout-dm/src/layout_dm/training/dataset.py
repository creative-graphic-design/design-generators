"""Dataset and collation helpers for LayoutDM training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import torch
from jaxtyping import Float, Int, Shaped
from torch.utils.data import Dataset as TorchDataset

from laygen.common.bbox import BoxFormat
from laygen.common.labels import label2id_for_dataset

from ..configuration_layout_dm import LayoutDMConfig
from ..processing_layout_dm import LayoutDMProcessor
from ..tokenization_layout_dm import LayoutDMTokenizer
from .config import LayoutDMTrainingDatasetName, LayoutDMTrainingSplit

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

LayoutDMScalar: TypeAlias = str | int | float | bool
LayoutDMAnnotation: TypeAlias = Mapping[str, LayoutDMScalar | Sequence[float]]
LayoutDMValue: TypeAlias = (
    LayoutDMScalar
    | Sequence[LayoutDMScalar]
    | Sequence[Sequence[float]]
    | Sequence[LayoutDMAnnotation]
    | None
)

_RICO_CONFIG: Final[str] = "ui-screenshots-and-hierarchies-with-semantic-annotations"
_DATASET_IDS: Final[dict[str, tuple[str, str | None]]] = {
    "rico25": ("creative-graphic-design/Rico", _RICO_CONFIG),
    "publaynet": ("creative-graphic-design/PubLayNet", None),
}
_BOX_KEYS: Final[tuple[str, ...]] = ("bbox", "bboxes", "boxes")
_LABEL_KEYS: Final[tuple[str, ...]] = (
    "labels",
    "label",
    "category",
    "categories",
    "type",
    "class_id",
)
_ANNOTATION_KEYS: Final[tuple[str, ...]] = (
    "annotations",
    "objects",
    "elements",
    "children",
)


class LayoutDMDataset(TorchDataset[dict[str, Shaped[torch.Tensor, "..."] | str]]):
    """HF datasets-backed LayoutDM training dataset."""

    def __init__(
        self,
        *,
        dataset_name: LayoutDMTrainingDatasetName,
        config: LayoutDMConfig,
        split: LayoutDMTrainingSplit = "train",
        tokenizer: LayoutDMTokenizer | None = None,
        max_seq_length: int | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
    ) -> None:
        """Load a LayoutDM training split from the approved HF dataset source.

        Args:
            dataset_name: ``rico25`` or ``publaynet``.
            split: Dataset split to load.
            config: Optional LayoutDM configuration.
            tokenizer: Optional tokenizer. Built from ``config`` otherwise.
            max_seq_length: Optional element cap before tokenization.
            box_format: Source box format.
            normalized: Whether source boxes are already normalized.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.tokenizer = tokenizer or LayoutDMTokenizer(self.config)
        self.processor = LayoutDMProcessor(self.tokenizer)
        self.max_seq_length = max_seq_length or self.config.max_seq_length
        self.box_format = box_format
        self.normalized = normalized
        self.label2id = _casefold_mapping(label2id_for_dataset(dataset_name))

        import datasets

        path, name = _DATASET_IDS[dataset_name]
        self.dataset: HFDataset = datasets.load_dataset(
            path, name=name, split=split, streaming=False
        )

    def __len__(self) -> int:
        """Return dataset size when the underlying split exposes it."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one tokenized training example."""
        sample = cast(Mapping[str, LayoutDMValue], self.dataset[index])
        return self._encode_sample(sample)

    def _encode_sample(
        self, sample: Mapping[str, LayoutDMValue]
    ) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        bbox, labels, canvas_size = _extract_layout(sample, self.label2id)
        bbox = bbox[: self.max_seq_length]
        labels = labels[: self.max_seq_length]
        mask = torch.ones(labels.shape, dtype=torch.bool)
        encoded = self.processor(
            bbox=bbox.unsqueeze(0),
            labels=labels.unsqueeze(0),
            mask=mask.unsqueeze(0),
            box_format=self.box_format,
            normalized=self.normalized,
            canvas_size=canvas_size,
        )
        output: dict[str, Shaped[torch.Tensor, "..."] | str] = {
            key: value.squeeze(0) for key, value in encoded.items()
        }
        sample_id = sample.get("id") or sample.get("image_id") or sample.get("doc_id")
        if sample_id is not None:
            output["id"] = str(sample_id)
        return output


class LayoutDMSyntheticDataset(
    TorchDataset[dict[str, Shaped[torch.Tensor, "..."] | str]]
):
    """Small deterministic dataset for local CLI smoke tests."""

    def __init__(
        self,
        *,
        config: LayoutDMConfig,
        size: int = 8,
        elements: int = 3,
    ) -> None:
        """Initialize synthetic examples from a LayoutDM config."""
        super().__init__()
        self.config = config
        self.size = size
        self.elements = min(elements, config.max_seq_length)
        self.tokenizer = LayoutDMTokenizer(config)
        self.processor = LayoutDMProcessor(self.tokenizer)

    def __len__(self) -> int:
        """Return synthetic dataset size."""
        return self.size

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one deterministic synthetic tokenized layout."""
        generator = torch.Generator().manual_seed(index)
        bbox = torch.rand(self.elements, 4, generator=generator)
        bbox[:, 2:] = bbox[:, 2:].mul(0.35).add(0.05)
        labels = (
            torch.arange(self.elements, dtype=torch.long) % self.config.num_categories
        )
        encoded = self.processor(
            bbox=bbox.unsqueeze(0),
            labels=labels.unsqueeze(0),
            mask=torch.ones(1, self.elements, dtype=torch.bool),
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def _extract_layout(
    sample: Mapping[str, LayoutDMValue], label2id: Mapping[str, int]
) -> tuple[
    Float[torch.Tensor, "elements 4"],
    Int[torch.Tensor, "elements"],
    tuple[int, int] | None,
]:
    flat = _flatten_annotations(sample)
    bbox_obj = _first_present(flat, _BOX_KEYS)
    label_obj = _first_present(flat, _LABEL_KEYS)
    if bbox_obj is None or label_obj is None:
        raise KeyError("sample must contain bbox/labels or annotation objects")
    bbox = torch.as_tensor(
        cast(Sequence[Sequence[float]], bbox_obj),
        dtype=torch.float32,
    ).reshape(-1, 4)
    labels = _labels_tensor(label_obj, label2id).reshape(-1)
    total = min(bbox.shape[0], labels.shape[0])
    canvas_size = _canvas_size(sample)
    return bbox[:total], labels[:total], canvas_size


def _flatten_annotations(
    sample: Mapping[str, LayoutDMValue],
) -> dict[str, LayoutDMValue]:
    for key in _ANNOTATION_KEYS:
        value = sample.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            rows: list[LayoutDMAnnotation] = [
                cast(LayoutDMAnnotation, row)
                for row in value
                if isinstance(row, Mapping)
            ]
            if rows:
                out: dict[str, LayoutDMValue] = dict(sample)
                for box_key in _BOX_KEYS:
                    values = [_first_present(row, _BOX_KEYS) for row in rows]
                    if all(item is not None for item in values):
                        out[box_key] = cast(LayoutDMValue, values)
                        break
                for label_key in _LABEL_KEYS:
                    values = [_first_present(row, _LABEL_KEYS) for row in rows]
                    if all(item is not None for item in values):
                        out[label_key] = cast(LayoutDMValue, values)
                        break
                return out
    return dict(sample)


def _first_present(
    sample: Mapping[str, LayoutDMValue], keys: Sequence[str]
) -> LayoutDMValue:
    for key in keys:
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _labels_tensor(
    value: LayoutDMValue, label2id: Mapping[str, int]
) -> Int[torch.Tensor, "elements"]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if all(isinstance(item, str) for item in value):
            return torch.tensor(
                [_label_id(cast(str, item), label2id) for item in value],
                dtype=torch.long,
            )
    if isinstance(value, str):
        return torch.tensor([_label_id(value, label2id)], dtype=torch.long)
    return torch.as_tensor(cast(Sequence[int], value), dtype=torch.long)


def _label_id(value: str, label2id: Mapping[str, int]) -> int:
    key = value.casefold()
    if key not in label2id:
        raise KeyError(f"Unknown dataset label: {value}")
    return label2id[key]


def _casefold_mapping(mapping: Mapping[str, int]) -> dict[str, int]:
    return {str(key).casefold(): int(value) for key, value in mapping.items()}


def _canvas_size(sample: Mapping[str, LayoutDMValue]) -> tuple[int, int] | None:
    width = (
        sample.get("width") or sample.get("image_width") or sample.get("canvas_width")
    )
    height = (
        sample.get("height")
        or sample.get("image_height")
        or sample.get("canvas_height")
    )
    if isinstance(width, int | float) and isinstance(height, int | float):
        return int(width), int(height)
    size = sample.get("canvas_size") or sample.get("image_size") or sample.get("size")
    if isinstance(size, Sequence) and len(size) >= 2:
        return _to_int(cast(LayoutDMValue, size[0])), _to_int(
            cast(LayoutDMValue, size[1])
        )
    return None


def _to_int(value: LayoutDMValue) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"Expected numeric canvas dimension, got {type(value).__name__}")
