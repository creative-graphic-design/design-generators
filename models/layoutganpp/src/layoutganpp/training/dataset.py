"""Training datasets and deterministic layout collation for LayoutGAN++."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import torch
from jaxtyping import Bool, Float, Int
from torch.utils.data import Dataset

from laygen.common.randomness import rand, randint, randperm

from layoutganpp.datasets import (
    MAGAZINE_LABELS,
    PUBLAYNET_LABELS,
    RICO_LABELS,
    normalize_dataset_name,
)


@dataclass(frozen=True)
class LayoutRow:
    """One normalized layout row before padding."""

    name: str
    bbox: Float[torch.Tensor, "elements 4"]
    labels: Int[torch.Tensor, "elements"]
    width: float
    height: float


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


def _as_float(value: JsonValue) -> float:
    return float(cast(str | int | float, value))


def _as_int(value: JsonValue) -> int:
    return int(cast(str | int | float, value))


def _as_str(value: JsonValue) -> str:
    return str(value)


def _as_float_list(value: JsonValue) -> list[float]:
    return [_as_float(item) for item in cast(list[JsonValue], value)]


def _sort_row(row: LayoutRow) -> LayoutRow:
    """Apply the reference top-then-left lexicographic ordering."""
    left = row.bbox[:, 0] - row.bbox[:, 2] / 2
    top = row.bbox[:, 1] - row.bbox[:, 3] / 2
    order = sorted(
        range(len(row.labels)),
        key=lambda index: (float(top[index]), float(left[index])),
    )
    indices = torch.tensor(order, dtype=torch.long)
    return LayoutRow(
        name=row.name,
        bbox=row.bbox[indices],
        labels=row.labels[indices],
        width=row.width,
        height=row.height,
    )


def _split_indices(
    length: int, dataset_name: str, split: str
) -> Int[torch.Tensor, "indices"]:
    generator = torch.Generator().manual_seed(0)
    shuffled = randperm(length, generator=generator)
    if dataset_name == "publaynet":
        train_end = int(length * 0.95)
        if split == "train":
            return shuffled[:train_end]

        if split == "val":
            return shuffled[train_end:]

        if split == "test":
            return torch.arange(0)

    first = int(length * 0.85)
    second = int(length * 0.90)
    if split == "train":
        return shuffled[:first]

    if split == "val":
        return shuffled[first:second]

    if split == "test":
        return shuffled[second:]

    raise ValueError(f"Unsupported split: {split}")


def _magazine_rows(root: Path) -> list[LayoutRow]:
    rows: list[LayoutRow] = []
    for xml_path in sorted((root / "layoutdata" / "annotations").glob("*.xml")):
        tree = ET.parse(xml_path)
        annotation = tree.getroot()

        size = annotation.find("size")
        width_text = size.findtext("width") if size is not None else None
        height_text = size.findtext("height") if size is not None else None
        if width_text is None or height_text is None:
            raise ValueError(f"Magazine annotation has no size: {xml_path}")

        width = float(width_text)
        height = float(height_text)
        boxes: list[list[float]] = []
        labels: list[int] = []
        label2id = {str(label): index for index, label in enumerate(MAGAZINE_LABELS)}
        for element in annotation.findall("layout/element"):
            polygon_x = [
                float(value) for value in str(element.get("polygon_x")).split()
            ]
            polygon_y = [
                float(value) for value in str(element.get("polygon_y")).split()
            ]
            x1, x2 = min(polygon_x), max(polygon_x)
            y1, y2 = min(polygon_y), max(polygon_y)
            boxes.append(
                [
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
            )
            labels.append(label2id[str(element.get("label"))])

        rows.append(
            LayoutRow(
                name=str(annotation.findtext("filename")),
                bbox=torch.tensor(boxes, dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.long),
                width=width,
                height=height,
            )
        )

    return rows


def _rico_rows(root: Path) -> list[LayoutRow]:
    rows: list[LayoutRow] = []
    label2id = {str(label): index for index, label in enumerate(RICO_LABELS)}
    for json_path in sorted((root / "semantic_annotations").glob("*.json")):
        annotation = cast(JsonObject, json.loads(json_path.read_text()))
        bounds = _as_float_list(annotation["bounds"])
        width, height = bounds[2], bounds[3]
        if bounds[0] != 0 or bounds[1] != 0 or height < width:
            continue

        elements = _rico_elements(annotation)
        valid = []
        for element in elements:
            component_label = _as_str(element["componentLabel"])
            if component_label not in label2id:
                continue

            x1, y1, x2, y2 = _as_float_list(element["bounds"])
            if x1 < 0 or y1 < 0 or width < x2 or height < y2 or x2 <= x1 or y2 <= y1:
                continue

            valid.append(element)

        if not 0 < len(valid) <= 9:
            continue

        rows.append(
            LayoutRow(
                name=json_path.name,
                bbox=torch.tensor(
                    [
                        [
                            (
                                _as_float_list(element["bounds"])[0]
                                + _as_float_list(element["bounds"])[2]
                            )
                            / (2 * width),
                            (
                                _as_float_list(element["bounds"])[1]
                                + _as_float_list(element["bounds"])[3]
                            )
                            / (2 * height),
                            (
                                _as_float_list(element["bounds"])[2]
                                - _as_float_list(element["bounds"])[0]
                            )
                            / width,
                            (
                                _as_float_list(element["bounds"])[3]
                                - _as_float_list(element["bounds"])[1]
                            )
                            / height,
                        ]
                        for element in valid
                    ],
                    dtype=torch.float32,
                ),
                labels=torch.tensor(
                    [label2id[_as_str(element["componentLabel"])] for element in valid],
                    dtype=torch.long,
                ),
                width=width,
                height=height,
            )
        )

    return rows


def _rico_elements(
    element: JsonObject, output: list[JsonObject] | None = None
) -> list[JsonObject]:
    output = [] if output is None else output
    for child in cast(list[JsonObject], element.get("children", [])):
        output.append(child)
        _rico_elements(child, output)

    return output


def _publaynet_rows(root: Path) -> tuple[list[LayoutRow], list[LayoutRow]]:
    label2id = {str(label): index for index, label in enumerate(PUBLAYNET_LABELS)}
    rows_by_split: list[list[LayoutRow]] = []
    for split in ("train", "val"):
        payload = cast(
            JsonObject, json.loads((root / "publaynet" / f"{split}.json").read_text())
        )
        categories = {
            _as_int(item["id"]): _as_str(item["name"])
            for item in cast(list[JsonObject], payload["categories"])
        }
        annotations_by_image: dict[int, list[JsonObject]] = {}
        for annotation in cast(list[JsonObject], payload["annotations"]):
            annotations_by_image.setdefault(_as_int(annotation["image_id"]), []).append(
                annotation
            )

        rows: list[LayoutRow] = []
        for image in sorted(
            cast(list[JsonObject], payload["images"]),
            key=lambda item: _as_int(item["id"]),
        ):
            width, height = _as_float(image["width"]), _as_float(image["height"])
            if height < width:
                continue

            valid = []
            for element in annotations_by_image.get(_as_int(image["id"]), []):
                x1, y1, box_width, box_height = _as_float_list(element["bbox"])
                x2, y2 = x1 + box_width, y1 + box_height
                if (
                    x1 < 0
                    or y1 < 0
                    or width < x2
                    or height < y2
                    or x2 <= x1
                    or y2 <= y1
                ):
                    continue

                valid.append(
                    (x1, y1, x2, y2, categories[_as_int(element["category_id"])])
                )

            if not 0 < len(valid) <= 9:
                continue

            if any(label not in label2id for *_, label in valid):
                continue

            rows.append(
                LayoutRow(
                    name=_as_str(image["file_name"]),
                    bbox=torch.tensor(
                        [
                            [
                                (x1 + x2) / (2 * width),
                                (y1 + y2) / (2 * height),
                                (x2 - x1) / width,
                                (y2 - y1) / height,
                            ]
                            for x1, y1, x2, y2, _ in valid
                        ],
                        dtype=torch.float32,
                    ),
                    labels=torch.tensor(
                        [label2id[label] for *_, label in valid], dtype=torch.long
                    ),
                    width=width,
                    height=height,
                )
            )

        rows_by_split.append(rows)

    return rows_by_split[0], rows_by_split[1]


def load_rows(dataset_name: str, data_root: str | Path, split: str) -> list[LayoutRow]:
    """Load rows with the original split and preprocessing rules."""
    canonical = str(normalize_dataset_name(dataset_name))
    root = Path(data_root)
    if canonical == "magazine":
        rows = _magazine_rows(root)
        return (
            [
                _sort_row(rows[index])
                for index in _split_indices(len(rows), canonical, split)
            ]
            if split == "train"
            else [rows[index] for index in _split_indices(len(rows), canonical, split)]
        )

    if canonical == "rico13":
        rows = _rico_rows(root)
        return (
            [
                _sort_row(rows[index])
                for index in _split_indices(len(rows), canonical, split)
            ]
            if split == "train"
            else [rows[index] for index in _split_indices(len(rows), canonical, split)]
        )

    if canonical == "publaynet":
        train_rows, test_rows = _publaynet_rows(root)
        if split == "test":
            return test_rows

        return [
            _sort_row(train_rows[index])
            for index in _split_indices(len(train_rows), canonical, split)
        ]

    raise ValueError(f"Unsupported LayoutGAN++ dataset: {dataset_name}")


def synthetic_rows(dataset_name: str, size: int, seed: int) -> list[LayoutRow]:
    """Build deterministic rows for network-free wiring checks."""
    canonical = str(normalize_dataset_name(dataset_name))
    num_labels = {"rico13": 13, "publaynet": 5, "magazine": 5}[canonical]
    max_elements = {"rico13": 9, "publaynet": 9, "magazine": 33}[canonical]
    rows = []
    for index in range(size):
        generator = torch.Generator().manual_seed(seed + index)
        count = 2 + index % max(1, min(max_elements - 1, 6))
        bbox = rand((count, 4), generator=generator)
        bbox[:, 2:] = bbox[:, 2:] * 0.4 + 0.1
        labels = randint(0, num_labels, (count,), generator=generator)
        rows.append(LayoutRow(f"synthetic-{index}", bbox, labels, 100.0, 200.0))

    return rows


class LayoutGANPPDataset(Dataset[LayoutRow]):
    """Dataset with reference-compatible sorting and split behavior."""

    def __init__(
        self,
        *,
        dataset_name: str,
        split: str,
        data_root: str | Path | None,
        synthetic_size: int,
        seed: int,
    ) -> None:
        """Initialize a dataset from a local root or synthetic rows."""
        canonical = str(normalize_dataset_name(dataset_name))
        if data_root is None:
            rows = synthetic_rows(canonical, synthetic_size, seed)
        else:
            rows = load_rows(canonical, data_root, split)

        self.rows = rows

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> LayoutRow:
        """Return one normalized layout row."""
        return self.rows[index]


def collate_layoutganpp(
    rows: list[LayoutRow],
) -> dict[
    str,
    Float[torch.Tensor, "batch elements 4"]
    | Int[torch.Tensor, "batch elements"]
    | Bool[torch.Tensor, "batch elements"]
    | list[str],
]:
    """Pad a list of variable-length layouts into a training batch."""
    if not rows:
        raise ValueError("Cannot collate an empty LayoutGAN++ batch")

    max_elements = max(len(row.labels) for row in rows)
    bbox = torch.zeros((len(rows), max_elements, 4), dtype=torch.float32)
    labels = torch.zeros((len(rows), max_elements), dtype=torch.long)
    mask = torch.zeros((len(rows), max_elements), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row.labels)
        bbox[index, :count] = row.bbox
        labels[index, :count] = row.labels
        mask[index, :count] = True

    return {
        "bbox": bbox,
        "labels": labels,
        "mask": mask,
        "names": [row.name for row in rows],
    }
