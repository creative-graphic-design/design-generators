"""Approved-source to reference-layout data bridge for DS-GAN."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from jaxtyping import Shaped
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from ..processing_ds_gan import (
    DSGANExampleValue,
    DSGANImageInput,
    DSGANProcessor,
    annotations_from_pku_example,
)

SOURCE_DATASET = "creative-graphic-design/PKU-PosterLayout"
SOURCE_CONFIG = "default"
SOURCE_FIELDS = {
    "train.inpainted_poster": "inpainted_poster",
    "test.canvas": "canvas",
    "train.saliency_pfpnet": "pfpn_saliency_map",
    "test.saliency_pfpnet": "pfpn_saliency_map",
    "train.saliency_basnet": "basnet_saliency_map",
    "test.saliency_basnet": "basnet_saliency_map",
    "train.annotations": "annotations",
}

DatasetScalar: TypeAlias = str | int | float | bool | None
DatasetValue: TypeAlias = (
    DatasetScalar | Image.Image | list["DatasetValue"] | dict[str, "DatasetValue"]
)
ManifestValue: TypeAlias = (
    DatasetScalar | list["ManifestValue"] | dict[str, "ManifestValue"]
)


def bridge_example(
    example: Mapping[str, DatasetValue],
    *,
    split: str,
) -> dict[str, DatasetValue]:
    """Map one approved-source row to the fields consumed by the reference loader."""
    required = (
        ("canvas", "pfpn_saliency_map", "basnet_saliency_map")
        if split == "test"
        else (
            "inpainted_poster",
            "pfpn_saliency_map",
            "basnet_saliency_map",
            "annotations",
        )
    )
    missing = [
        field for field in required if field not in example or example[field] is None
    ]
    if missing:
        raise KeyError(f"approved source lacks required input fields: {missing}")

    return {
        "inpainted_poster": example.get("inpainted_poster"),
        "canvas": example.get("canvas"),
        "saliency_pfpnet": example.get("pfpn_saliency_map"),
        "saliency_basnet": example.get("basnet_saliency_map"),
        "annotations": example.get("annotations"),
    }


class DSGANDataset(TorchDataset[dict[str, Shaped[torch.Tensor, "..."]]]):
    """Lazy DS-GAN dataset using the approved source and reference preprocessing."""

    def __init__(
        self,
        rows: Dataset,
        *,
        split: str,
        processor: DSGANProcessor | None = None,
        max_elem: int = 32,
    ) -> None:
        """Initialize the source bridge."""
        self.rows = rows
        self.split = split
        self.processor = processor or DSGANProcessor()
        self.max_elem = max_elem

    def __len__(self) -> int:
        """Return the number of source rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Convert one source row into model inputs and padded labels."""
        source = bridge_example(
            cast(Mapping[str, DatasetValue], self.rows[index]), split=self.split
        )
        image = cast(
            DSGANImageInput,
            source["canvas"] if self.split == "test" else source["inpainted_poster"],
        )
        encoded = self.processor(
            image,
            saliency_pfpnet=cast(DSGANImageInput, source["saliency_pfpnet"]),
            saliency_basnet=cast(DSGANImageInput, source["saliency_basnet"]),
        )
        pixel_values = encoded["pixel_values"][0]
        if self.split == "test":
            return {"pixel_values": pixel_values}

        layout_fields = annotations_from_pku_example(
            cast(
                Mapping[str, DSGANExampleValue],
                {
                    "annotations": cast(DSGANExampleValue, source["annotations"]),
                    "inpainted_poster": image,
                },
            ),
            max_elem=self.max_elem,
        )
        bbox = cast(torch.Tensor, layout_fields["bbox"]).squeeze(0)
        labels = cast(torch.Tensor, layout_fields["labels"]).squeeze(0)
        mask = cast(torch.Tensor, layout_fields["mask"]).squeeze(0)
        padded_bbox, padded_labels, padded_mask = self.processor.pad(
            bbox.unsqueeze(0),
            labels.unsqueeze(0),
            mask.unsqueeze(0),
            max_elem=self.max_elem,
        )
        model_labels = torch.zeros_like(padded_labels)
        model_labels[padded_mask] = padded_labels[padded_mask] + 1
        classes = torch.nn.functional.one_hot(model_labels, num_classes=4).to(
            dtype=padded_bbox.dtype
        )
        layout = torch.stack((classes, padded_bbox), dim=2).squeeze(0)
        return {
            "pixel_values": pixel_values,
            "layout": layout,
            "labels": model_labels.squeeze(0),
            "boxes": padded_bbox.squeeze(0),
            "mask": padded_mask.squeeze(0),
        }


def build_bridge_manifest(
    *,
    source_revision: str,
    train_count: int,
    test_count: int,
    source_hashes: Mapping[str, str],
    source_fields: Mapping[str, str] | None = None,
    available_fields: Sequence[str] | None = None,
) -> dict[str, ManifestValue]:
    """Build a hash-manifested record of the approved-source bridge."""
    mappings = dict(source_fields or SOURCE_FIELDS)
    expected = set(SOURCE_FIELDS)
    available = set(available_fields or mappings.values())
    missing = sorted(
        field for field in expected if mappings.get(field) not in available
    )
    return {
        "source": {
            "dataset": SOURCE_DATASET,
            "config": SOURCE_CONFIG,
            "revision": source_revision,
            "split_counts": {"train": train_count, "test": test_count},
        },
        "vendor_layout": {
            "train": {
                "inpainted_poster": mappings.get("train.inpainted_poster"),
                "saliency_pfpnet": mappings.get("train.saliency_pfpnet"),
                "saliency_basnet": mappings.get("train.saliency_basnet"),
                "annotations": mappings.get("train.annotations"),
            },
            "test": {
                "canvas": mappings.get("test.canvas"),
                "saliency_pfpnet": mappings.get("test.saliency_pfpnet"),
                "saliency_basnet": mappings.get("test.saliency_basnet"),
            },
        },
        "missing_source_fields": missing,
        "hashes": dict(source_hashes),
        "transform": {
            "resize": [350, 240],
            "saliency_merge": "pixelwise_max",
            "box_format": "ltrb_to_normalized_cxcywh",
            "canvas_size": [513, 750],
            "invalid_class": "filtered",
        },
    }


def hash_file(path: Path) -> str:
    """Hash a source metadata or arrow artifact in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_cached_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Load the approved source from a standard datasets cache without downloading."""
    root = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "datasets")
    candidates = sorted(
        root.glob(
            "creative-graphic-design___pku-poster_layout/default/0.0.0/*/dataset_info.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "cached PKU PosterLayout dataset metadata is unavailable"
        )

    dataset_dir = candidates[-1].parent
    train_files = sorted(dataset_dir.glob("pku-poster_layout-train-*.arrow"))
    test_file = dataset_dir / "pku-poster_layout-test.arrow"
    if not train_files or not test_file.exists():
        raise FileNotFoundError(
            "cached PKU PosterLayout train/test Arrow files are incomplete"
        )

    train = concatenate_datasets([Dataset.from_file(str(path)) for path in train_files])
    return DatasetDict({"train": train, "test": Dataset.from_file(str(test_file))})


def load_cached_test(cache_dir: str | Path | None = None) -> Dataset:
    """Load only the approved source TEST Arrow stream from cache."""
    root = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "datasets")
    candidates = sorted(
        root.glob(
            "creative-graphic-design___pku-poster_layout/default/0.0.0/*/dataset_info.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "cached PKU PosterLayout dataset metadata is unavailable"
        )

    test_file = candidates[-1].parent / "pku-poster_layout-test.arrow"
    if not test_file.exists():
        raise FileNotFoundError(
            "cached PKU PosterLayout TEST Arrow file is unavailable"
        )

    return Dataset.from_file(str(test_file))


def manifest_from_cached_dataset(
    cache_dir: str | Path | None = None,
) -> dict[str, ManifestValue]:
    """Create a bridge manifest from cached source metadata and Arrow hashes."""
    root = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "datasets")
    metadata = sorted(
        root.glob(
            "creative-graphic-design___pku-poster_layout/default/0.0.0/*/dataset_info.json"
        )
    )[-1]
    dataset_dir = metadata.parent
    train_files = sorted(dataset_dir.glob("pku-poster_layout-train-*.arrow"))
    test_file = dataset_dir / "pku-poster_layout-test.arrow"
    info = json.loads(metadata.read_text())
    return build_bridge_manifest(
        source_revision=info.get("download_checksums", {})
        .popitem()[0]
        .split("@")[1]
        .split("/")[0],
        train_count=int(info["splits"]["train"]["num_examples"]),
        test_count=int(info["splits"]["test"]["num_examples"]),
        source_hashes={
            "dataset_info.json": hash_file(metadata),
            **{path.name: hash_file(path) for path in (*train_files, test_file)},
        },
        available_fields=list(info["features"]),
    )
