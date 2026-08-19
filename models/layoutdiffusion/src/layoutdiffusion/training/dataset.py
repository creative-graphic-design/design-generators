"""Dataset helpers for LayoutDiffusion training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import torch
from jaxtyping import Float, Int, Shaped
from laygen.common.bbox import BoxFormat, ltwh_to_xywh, xywh_to_ltrb
from laygen.common.layout_keys import (
    LAYOUT_ANNOTATION_KEYS,
    LAYOUT_BBOX_KEYS,
    LAYOUT_LABEL_KEYS,
)
from torch.utils.data import Dataset as TorchDataset

from ..configuration_layoutdiffusion import LayoutDiffusionConfig
from ..labels import default_id2label, normalize_layoutdiffusion_label
from ..tokenization_layoutdiffusion import LayoutDiffusionTokenizer
from .config import LayoutDiffusionTrainingDatasetName, LayoutDiffusionTrainingSplit

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset

LayoutSampleScalar: TypeAlias = str | int | float | bool | None
LayoutSampleValue: TypeAlias = (
    LayoutSampleScalar
    | Sequence["LayoutSampleValue"]
    | Mapping[str, "LayoutSampleValue"]
)
LayoutSample: TypeAlias = Mapping[str, LayoutSampleValue]
_NumericSequence: TypeAlias = Sequence[int | float | Sequence[int | float]]

_RICO_CONFIG: Final[str] = "ui-screenshots-and-hierarchies-with-semantic-annotations"
_DATASET_IDS: Final[dict[str, tuple[str, str | None]]] = {
    "rico25": ("creative-graphic-design/Rico", _RICO_CONFIG),
    "publaynet": ("creative-graphic-design/PubLayNet", None),
}
_PROCESSED_SPLITS: Final[dict[str, str]] = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
_PROCESSED_STREAM_NAMES: Final[dict[str, str]] = {
    "rico25": "RICO_ltrb_lex",
    "publaynet": "PublayNet_ltrb_lex",
}
_BOX_KEYS: Final[tuple[str, ...]] = LAYOUT_BBOX_KEYS
_LABEL_KEYS: Final[tuple[str, ...]] = LAYOUT_LABEL_KEYS
_ANNOTATION_KEYS: Final[tuple[str, ...]] = LAYOUT_ANNOTATION_KEYS


class LayoutDiffusionDataset(
    TorchDataset[dict[str, Shaped[torch.Tensor, "..."] | str]]
):
    """HF datasets-backed LayoutDiffusion training dataset."""

    def __init__(
        self,
        *,
        dataset_name: LayoutDiffusionTrainingDatasetName,
        config: LayoutDiffusionConfig,
        split: LayoutDiffusionTrainingSplit = "train",
        tokenizer: LayoutDiffusionTokenizer | None = None,
        max_num_elements: int | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        lexicographic_order: bool = True,
    ) -> None:
        """Load a LayoutDiffusion training split from approved HF sources."""
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.tokenizer = tokenizer or LayoutDiffusionTokenizer(self.config)
        self.max_num_elements = max_num_elements or self.config.max_num_elements
        self.box_format = box_format
        self.normalized = normalized
        self.lexicographic_order = lexicographic_order
        self.label2id = _layoutdiffusion_label2id(self.config)
        self.public_id2label = default_id2label(self.config.dataset_name)

        import datasets

        path, name = _DATASET_IDS[dataset_name]
        self.dataset: HFDataset = datasets.load_dataset(
            path, name=name, split=split, streaming=False
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one tokenized training example."""
        sample = cast(LayoutSample, self.dataset[index])
        encoded = self._encode_sample(sample)
        output: dict[str, Shaped[torch.Tensor, "..."] | str] = dict(encoded)
        sample_id = sample.get("id") or sample.get("image_id") or sample.get("doc_id")
        if sample_id is not None:
            output["id"] = str(sample_id)
        return output

    def _encode_sample(
        self, sample: LayoutSample
    ) -> dict[str, Shaped[torch.Tensor, "..."]]:
        bbox, labels, canvas_size = _extract_layout(
            sample, self.label2id, self.public_id2label
        )
        bbox = bbox[: self.max_num_elements]
        labels = labels[: self.max_num_elements]
        if self.lexicographic_order:
            bbox, labels = _lexicographic_order(bbox, labels, self.box_format)
        mask = torch.ones(labels.shape, dtype=torch.bool)
        encoded = self.tokenizer(
            bbox=bbox.unsqueeze(0),
            labels=labels.unsqueeze(0),
            mask=mask.unsqueeze(0),
            box_format=self.box_format,
            normalized=self.normalized,
            canvas_size=canvas_size,
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class LayoutDiffusionProcessedDataset(
    TorchDataset[dict[str, Shaped[torch.Tensor, "..."] | str]]
):
    """Processed LayoutDiffusion token stream used for parity reruns."""

    def __init__(
        self,
        *,
        dataset_name: LayoutDiffusionTrainingDatasetName,
        config: LayoutDiffusionConfig,
        processed_data_dir: str | Path,
        split: LayoutDiffusionTrainingSplit = "train",
        tokenizer: LayoutDiffusionTokenizer | None = None,
    ) -> None:
        """Load processed token ids or text lines from a local directory."""
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.tokenizer = tokenizer or LayoutDiffusionTokenizer(self.config)
        self.path = _processed_path(Path(processed_data_dir), dataset_name, split)
        if self.path.suffix == ".pt":
            self.rows = _load_processed_tensor_rows(
                self.path, self.config.max_token_length
            )
        else:
            self.rows = self.tokenizer.text_to_token_ids(
                self.path.read_text(encoding="utf-8").splitlines()
            )

    def __len__(self) -> int:
        """Return the number of processed rows."""
        return int(self.rows.shape[0])

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one processed token row."""
        input_ids = self.rows[index].long()
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.config.pad_token_id),
            "mask": input_ids.ne(self.config.pad_token_id),
            "id": str(index),
        }


class LayoutDiffusionSyntheticDataset(
    TorchDataset[dict[str, Shaped[torch.Tensor, "..."] | str]]
):
    """Small deterministic dataset for local LightningCLI smoke tests."""

    def __init__(
        self,
        *,
        config: LayoutDiffusionConfig,
        size: int = 8,
        elements: int = 3,
    ) -> None:
        """Initialize deterministic synthetic layout examples."""
        super().__init__()
        self.config = config
        self.size = size
        self.elements = min(elements, config.max_num_elements)
        self.tokenizer = LayoutDiffusionTokenizer(config)

    def __len__(self) -> int:
        """Return synthetic dataset size."""
        return self.size

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."] | str]:
        """Return one deterministic synthetic tokenized layout."""
        generator = torch.Generator().manual_seed(index)
        bbox = torch.rand(self.elements, 4, generator=generator)
        bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:].mul(0.35).add(0.05)
        bbox = bbox.clamp(0.0, 1.0)
        labels = torch.arange(self.elements, dtype=torch.long) % self.config.num_labels
        encoded = self.tokenizer(
            bbox=bbox.unsqueeze(0),
            labels=labels.unsqueeze(0),
            mask=torch.ones(1, self.elements, dtype=torch.bool),
            box_format=BoxFormat.ltrb,
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def _extract_layout(
    sample: LayoutSample,
    label2id: Mapping[str, int],
    public_id2label: Mapping[int, str],
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

    bbox = _bbox_tensor(bbox_obj)
    label_strings = _label_strings(label_obj, public_id2label)
    labels = torch.tensor(
        [
            label2id[normalize_layoutdiffusion_label(label).casefold()]
            for label in label_strings
        ],
        dtype=torch.long,
    )
    total = min(bbox.shape[0], labels.shape[0])
    canvas = _canvas_size(sample)
    return bbox[:total], labels[:total], canvas


def _flatten_annotations(sample: LayoutSample) -> LayoutSample:
    for key in _ANNOTATION_KEYS:
        annotations = sample.get(key)
        if isinstance(annotations, Sequence) and not isinstance(annotations, str):
            rows = [row for row in annotations if isinstance(row, Mapping)]
            if rows:
                out: dict[str, LayoutSampleValue] = dict(sample)
                keys = set().union(*(row.keys() for row in rows))
                for ann_key in keys:
                    out[str(ann_key)] = [row.get(str(ann_key)) for row in rows]
                return out
    return sample


def _first_present(
    sample: LayoutSample, keys: Sequence[str]
) -> LayoutSampleValue | None:
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return value
    return None


def _bbox_tensor(value: LayoutSampleValue) -> Float[torch.Tensor, "elements 4"]:
    if isinstance(value, torch.Tensor):
        return value.float().reshape(-1, 4)
    if isinstance(value, Sequence) and not isinstance(value, str):
        rows = cast(_NumericSequence, value)
        return torch.as_tensor(rows, dtype=torch.float32).reshape(-1, 4)
    raise TypeError("bbox value must be a tensor or numeric sequence")


def _label_strings(
    value: LayoutSampleValue, public_id2label: Mapping[int, str]
) -> list[str]:
    if isinstance(value, torch.Tensor):
        return [
            public_id2label[int(item)] for item in value.long().reshape(-1).tolist()
        ]
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = list(value)
        if values and isinstance(values[0], str):
            return [str(item) for item in values]
        label_values = cast(Sequence[int | float], values)
        return [public_id2label[int(item)] for item in label_values]
    if isinstance(value, str):
        return [value]
    if isinstance(value, int | float | bool):
        return [public_id2label[int(value)]]
    raise TypeError("label value must be a tensor, scalar, or sequence")


def _canvas_size(sample: LayoutSample) -> tuple[int, int] | None:
    width = (
        sample.get("width") or sample.get("image_width") or sample.get("canvas_width")
    )
    height = (
        sample.get("height")
        or sample.get("image_height")
        or sample.get("canvas_height")
    )
    if width is None or height is None:
        size = sample.get("canvas_size") or sample.get("size")
        if isinstance(size, Sequence) and not isinstance(size, str) and len(size) >= 2:
            width, height = size[0], size[1]
    if not isinstance(width, str | int | float | bool):
        return None
    if not isinstance(height, str | int | float | bool):
        return None
    return int(width), int(height)


def _lexicographic_order(
    bbox: Float[torch.Tensor, "elements 4"],
    labels: Int[torch.Tensor, "elements"],
    box_format: BoxFormat | str,
) -> tuple[Float[torch.Tensor, "elements 4"], Int[torch.Tensor, "elements"]]:
    fmt = BoxFormat(box_format)
    if fmt is BoxFormat.xywh:
        ltrb = xywh_to_ltrb(bbox)
    elif fmt is BoxFormat.ltwh:
        ltrb = xywh_to_ltrb(ltwh_to_xywh(bbox))
    else:
        ltrb = bbox
    keys = (
        ltrb[:, 0] * 10_000_000 + ltrb[:, 1] * 100_000 + ltrb[:, 2] * 1_000 + ltrb[:, 3]
    )
    order = keys.argsort(stable=True)
    return bbox[order], labels[order]


def _processed_path(
    root: Path,
    dataset_name: LayoutDiffusionTrainingDatasetName,
    split: LayoutDiffusionTrainingSplit,
) -> Path:
    split_name = _PROCESSED_SPLITS[split]
    stream_name = _PROCESSED_STREAM_NAMES[dataset_name]
    candidates = [
        root / stream_name / f"{split_name}.txt",
        root / stream_name / f"{split_name}.pt",
        root / f"{stream_name}_{split_name}.txt",
        root / f"{stream_name}_{split_name}.pt",
        root / f"{dataset_name}_{split_name}.txt",
        root / f"{dataset_name}_{split_name}.pt",
        root / f"{split_name}.txt",
        root / f"{split_name}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _load_processed_tensor_rows(
    path: Path, max_token_length: int
) -> Int[torch.Tensor, "rows tokens"]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        rows = data.long()
    elif isinstance(data, Mapping) and "input_ids" in data:
        rows = torch.as_tensor(data["input_ids"], dtype=torch.long)
    else:
        rows = torch.as_tensor(data, dtype=torch.long)
    if rows.ndim == 1:
        rows = rows.unsqueeze(0)
    if rows.shape[1] < max_token_length:
        pad = torch.full(
            (rows.shape[0], max_token_length - rows.shape[1]), 3, dtype=torch.long
        )
        rows = torch.cat([rows, pad], dim=1)
    return rows[:, :max_token_length]


def _layoutdiffusion_label2id(config: LayoutDiffusionConfig) -> dict[str, int]:
    return {
        normalize_layoutdiffusion_label(label).casefold(): int(label_id)
        for label_id, label in config.id2label.items()
    }
