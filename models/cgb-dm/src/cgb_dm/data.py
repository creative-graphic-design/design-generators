"""Dataset utilities for CGB-DM original zip extracts."""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from laygen.common.bbox import ltrb_to_xywh

from .processing_cgb_dm import CGBDMProcessor


@dataclass(frozen=True)
class CGBDMDataPaths:
    """Paths for one CGB-DM split in an extracted dataset tree."""

    root: Path
    split: str = "train"

    @property
    def inpaint_dir(self) -> Path:
        """Return the inpaint image directory."""
        return self.root / self.split / "inpaint"

    @property
    def saliency_dir(self) -> Path:
        """Return the first saliency directory."""
        return self.root / self.split / "saliency"

    @property
    def saliency_sub_dir(self) -> Path:
        """Return the second saliency directory."""
        return self.root / self.split / "saliency_sub"

    @property
    def annotation_csv(self) -> Path:
        """Return the element annotation CSV path."""
        return self.root / "csv" / f"{self.split}.csv"

    @property
    def saliency_csv(self) -> Path:
        """Return the saliency-box CSV path."""
        return self.root / "csv" / f"{self.split}_sal.csv"


class CGBDMOriginalDataset(Dataset[dict[str, torch.Tensor]]):
    """Read an extracted CGB-DM dataset split without downloading assets.

    Args:
        root: Extracted dataset root.
        split: Dataset split name.
        processor: Processor used for image/layout normalization.

    Examples:
        >>> CGBDMDataPaths(Path("/tmp/data")).annotation_csv.name
        'train.csv'
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: Literal["train", "val", "test"] | str = "train",
        processor: CGBDMProcessor | None = None,
        name_manifest: str | Path | list[str] | tuple[str, ...] | None = None,
        encoding: Literal["public", "reference"] = "public",
    ) -> None:
        """Initialize file lists and CSV indexes."""
        self.paths = CGBDMDataPaths(Path(root), split)
        self.processor = processor or CGBDMProcessor()
        self.names = _load_names(self.paths.inpaint_dir, name_manifest)
        self.encoding = encoding
        self.annotations = _read_grouped_boxes(self.paths.annotation_csv)
        self.saliency_boxes = _read_grouped_boxes(self.paths.saliency_csv)

    def __len__(self) -> int:
        """Return number of image rows."""
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one normalized CGB-DM training row."""
        name = self.names[index]
        image_path = self.paths.inpaint_dir / name
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        saliency = Image.open(self.paths.saliency_dir / name).convert("L")
        saliency_sub = Image.open(self.paths.saliency_sub_dir / name).convert("L")
        if self.encoding == "reference":
            return _encode_reference_row(
                image=image,
                saliency=saliency,
                saliency_sub=saliency_sub,
                annotations=self.annotations[name],
                saliency_box=self.saliency_boxes[name][0],
                width=width,
                height=height,
                max_seq_length=self.processor.max_seq_length,
                num_labels=self.processor.num_labels,
                image_size=self.processor.image_size,
            )
        if self.encoding != "public":
            raise ValueError(f"Unsupported CGB-DM dataset encoding: {self.encoding}")
        content = self.processor(
            image,
            saliency_isnet=saliency,
            saliency_basnet=saliency_sub,
            saliency_box=_normalize_ltrb(self.saliency_boxes[name][0], width, height),
        )
        boxes, labels = zip(*self.annotations[name], strict=False)
        public_labels = (
            [label - 1 for label in labels]
            if self.processor.dataset_name == "pku_posterlayout"
            else list(labels)
        )
        layout = self.processor.encode_layout(
            bbox=[[_normalize_ltrb(box, width, height).tolist() for box in boxes]],
            labels=[public_labels],
        )["layout"][0]
        return {
            "pixel_values": content["pixel_values"][0],
            "layout": layout,
            "saliency_box": content["saliency_box"][0],
        }


def _load_names(
    inpaint_dir: Path, manifest: str | Path | list[str] | tuple[str, ...] | None
) -> list[str]:
    if manifest is None:
        return sorted(path.name for path in inpaint_dir.iterdir())
    if isinstance(manifest, str | Path):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload["names"]
        return [str(name) for name in payload]
    return [str(name) for name in manifest]


def _read_grouped_boxes(
    path: Path,
) -> dict[str, list[tuple[tuple[float, float, float, float], int]]]:
    rows: dict[str, list[tuple[tuple[float, float, float, float], int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = ast.literal_eval(row["box_elem"])
            box = (
                float(parsed[0]),
                float(parsed[1]),
                float(parsed[2]),
                float(parsed[3]),
            )
            cls = int(row.get("cls_elem", 0))
            rows.setdefault(row["poster_path"], []).append((box, cls))
    return rows


def _normalize_ltrb(
    box: tuple[float, float, float, float]
    | tuple[tuple[float, float, float, float], int],
    width: int,
    height: int,
) -> torch.Tensor:
    if isinstance(box[0], tuple):
        box = box[0]
    tensor = torch.tensor(box, dtype=torch.float32)
    tensor = tensor / torch.tensor((width, height, width, height), dtype=torch.float32)
    return ltrb_to_xywh(tensor).clamp(0.0, 1.0)


def _encode_reference_row(
    *,
    image: Image.Image,
    saliency: Image.Image,
    saliency_sub: Image.Image,
    annotations: list[tuple[tuple[float, float, float, float], int]],
    saliency_box: tuple[tuple[float, float, float, float], int],
    width: int,
    height: int,
    max_seq_length: int,
    num_labels: int,
    image_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    rgb = _pil_to_normalized_tensor(image, image_size, mode="RGB")
    saliency_map = Image.fromarray(
        np.maximum(np.asarray(saliency), np.asarray(saliency_sub))
    )
    saliency_tensor = _pil_to_normalized_tensor(saliency_map, image_size, mode="L")
    return {
        "pixel_values": torch.cat((rgb, saliency_tensor), dim=0),
        "layout": _encode_reference_layout(
            annotations, width, height, max_seq_length, num_labels
        ),
        "saliency_box": _encode_reference_saliency_box(saliency_box, width, height),
    }


def _pil_to_normalized_tensor(
    image: Image.Image,
    image_size: tuple[int, int],
    *,
    mode: Literal["RGB", "L"],
) -> torch.Tensor:
    image = image.convert(mode).resize(
        (image_size[1], image_size[0]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    if mode == "RGB":
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor * 2 - 1


def _encode_reference_layout(
    annotations: list[tuple[tuple[float, float, float, float], int]],
    width: int,
    height: int,
    max_seq_length: int,
    num_labels: int,
) -> torch.Tensor:
    label_cls = np.zeros((max_seq_length, num_labels))
    label_box = np.zeros((max_seq_length, 4))
    for index, (box, label) in enumerate(annotations[:max_seq_length]):
        label_cls[index][int(label)] = 1.0
        left, top, right, bottom = box
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        label_box[index] = ltrb_to_xywh(
            torch.tensor((left, top, right, bottom), dtype=torch.float32)
        ).numpy()
        label_box[index][::2] /= width
        label_box[index][1::2] /= height
    for index in range(min(len(annotations), max_seq_length), max_seq_length):
        label_cls[index][0] = 1.0
    label = np.concatenate((label_cls, label_box), axis=1)
    label[:, num_labels:] = 2 * (label[:, num_labels:] - 0.5)
    return torch.tensor(label).float()


def _encode_reference_saliency_box(
    saliency_box: tuple[tuple[float, float, float, float], int],
    width: int,
    height: int,
) -> torch.Tensor:
    box, _ = saliency_box
    tensor = ltrb_to_xywh(torch.tensor([box], dtype=torch.float32))
    tensor[::2] /= width
    tensor[1::2] /= height
    return 2 * (tensor - 0.5)
