"""Prepare stable private Crello splits and audit every document-stage PNG."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

from canvas_vae.data import (
    CRELLO_DOCUMENT_COUNTS,
    CRELLO_IMAGE_TYPES,
    CRELLO_SPLITS,
    CRELLO_TRAINING_IMAGE_COUNTS,
    CRELLO_V1_SHA256,
    CrelloDocument,
    CrelloElement,
    document_id,
    fixture_image_ids,
    image_id,
    load_crello_split,
)


class _BytesList(Protocol):
    value: Sequence[bytes]


class _Int64List(Protocol):
    value: Sequence[int]


class _FloatList(Protocol):
    value: Sequence[float]


class _Feature(Protocol):
    """TensorFlow feature fields used by the archive reader."""

    bytes_list: _BytesList
    int64_list: _Int64List
    float_list: _FloatList

    def WhichOneof(self, field: str) -> str | None: ...


class _FeatureSequence(Protocol):
    """SequenceExample feature entries."""

    feature: Sequence[_Feature]


class _Context(Protocol):
    """SequenceExample context map."""

    feature: Mapping[str, _Feature]


class _FeatureLists(Protocol):
    """SequenceExample feature-list map."""

    feature_list: Mapping[str, _FeatureSequence]


class _SequenceExample(Protocol):
    """SequenceExample fields consumed by the Crello cache reader."""

    context: _Context
    feature_lists: _FeatureLists


class _DimensionAudit(TypedDict):
    status: str
    shape_hw: list[int]
    channels: int


class _Manifest(TypedDict):
    source_archive_sha256: str
    documents: dict[str, int]
    document_stage_png_count: dict[str, int]
    pixelvae_training_png_count: dict[str, int]
    document_stage_image_ids_first_occurrence: list[str]
    pixelvae_training_image_ids: list[str]
    dimension_audit: _DimensionAudit


def _single_feature(
    feature: _Feature, *, decode_bytes: bool = True
) -> bytes | str | int | float | list[bytes | str | int | float]:
    kind = feature.WhichOneof("kind")
    values: list[bytes | str | int | float] = []
    if kind == "bytes_list":
        if decode_bytes:
            values.extend(value.decode("utf-8") for value in feature.bytes_list.value)
        else:
            values.extend(feature.bytes_list.value)
    elif kind == "int64_list":
        values.extend(int(value) for value in feature.int64_list.value)
    elif kind == "float_list":
        values.extend(float(value) for value in feature.float_list.value)
    else:
        raise ValueError("empty TensorFlow Feature in Crello record")

    if len(values) == 1:
        return values[0]

    return values


def prepare_crello_cache(source_dir: Path, output_dir: Path) -> _Manifest:
    """Parse the pinned source and verify each unique document-stage PNG as RGBA."""
    import tensorflow as tf

    marker = source_dir / ".crello-v1-sha256"
    if marker.read_text("ascii").strip() != CRELLO_V1_SHA256:
        raise ValueError("source directory does not have the verified Crello v1 marker")

    raw_documents: dict[str, list[CrelloDocument]] = {
        split: [] for split in CRELLO_SPLITS
    }
    all_image_ids: dict[str, set[str]] = {split: set() for split in CRELLO_SPLITS}
    train_image_ids: dict[str, set[str]] = {split: set() for split in CRELLO_SPLITS}
    image_types: dict[str, set[str]] = {}
    dimensions: dict[str, tuple[int, int]] = {}

    for split in CRELLO_SPLITS:
        record_files = sorted(source_dir.glob(f"{split}-*"))
        if not record_files:
            raise FileNotFoundError(f"no {split} TFRecord shards under {source_dir}")

        for record_file in record_files:
            for serialized in tf.compat.v1.io.tf_record_iterator(str(record_file)):
                example = cast(
                    _SequenceExample,
                    tf.train.SequenceExample.FromString(serialized),
                )
                context = {
                    key: cast(str | int, _single_feature(example.context.feature[key]))
                    for key in (
                        "id",
                        "length",
                        "group",
                        "format",
                        "canvas_width",
                        "canvas_height",
                        "category",
                    )
                }
                key = document_id(split, str(context["id"]))
                feature_lists = example.feature_lists.feature_list
                types = [
                    str(_single_feature(row)) for row in feature_lists["type"].feature
                ]
                geometry = {
                    field: [
                        float(cast(float | int, _single_feature(row)))
                        for row in feature_lists[field].feature
                    ]
                    for field in ("left", "top", "width", "height", "opacity")
                }
                colors = [
                    [
                        int(value)
                        for value in cast(
                            list[bytes | str | int | float], _single_feature(row)
                        )
                    ]
                    for row in feature_lists["color"].feature
                ]
                pngs = [
                    cast(bytes, _single_feature(row, decode_bytes=False))
                    for row in feature_lists["image_bytes"].feature
                ]
                lengths = {
                    len(types),
                    *(len(values) for values in geometry.values()),
                    len(colors),
                    len(pngs),
                }
                if (
                    len(lengths) != 1
                    or next(iter(lengths)) == 0
                    or any(len(rgb) != 3 for rgb in colors)
                ):
                    raise ValueError(
                        f"invalid element feature lengths in {key}: {lengths}"
                    )

                elements: list[CrelloElement] = []
                for index, png in enumerate(pngs):
                    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError(
                            f"non-PNG image_bytes in {key} element {index}"
                        )

                    current_id = image_id(split, png)
                    element_type = types[index]
                    image_types.setdefault(current_id, set()).add(element_type)
                    split_ids = all_image_ids[split]
                    if current_id not in split_ids:
                        image_path = (
                            output_dir
                            / "images"
                            / split
                            / f"{current_id.rsplit('/', 1)[1]}.png"
                        )
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(png)
                        decoded = tf.io.decode_png(png, channels=4)
                        decoded_height, decoded_width, _ = (
                            int(value) for value in tf.shape(decoded).numpy()
                        )
                        dimensions[current_id] = (decoded_height, decoded_width)
                        split_ids.add(current_id)

                    if element_type in CRELLO_IMAGE_TYPES:
                        train_image_ids[split].add(current_id)

                    elements.append(
                        {
                            "type": element_type,
                            "left": geometry["left"][index],
                            "top": geometry["top"][index],
                            "width": geometry["width"][index],
                            "height": geometry["height"][index],
                            "opacity": geometry["opacity"][index],
                            "color": colors[index],
                            "image_id": current_id,
                        }
                    )

                raw_documents[split].append(
                    {
                        "split": split,
                        "document_id": key,
                        "context": context,
                        "elements": elements,
                    }
                )

    invalid = {
        key: {"shape_hw": list(shape), "element_types": sorted(image_types[key])}
        for key, shape in dimensions.items()
        if shape != (256, 256)
    }
    if invalid:
        raise ValueError(
            "non-256x256 Crello PNGs found: "
            + json.dumps(invalid, sort_keys=True, separators=(",", ":"))
        )

    counts: dict[str, int] = {}
    for split in CRELLO_SPLITS:
        rows = sorted(
            raw_documents[split], key=lambda row: row["document_id"].encode("utf-8")
        )
        if len({row["document_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate document IDs in {split}")

        (output_dir / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        counts[split] = len(rows)

    if counts != CRELLO_DOCUMENT_COUNTS:
        raise ValueError(f"Crello v1 document counts mismatch: {counts}")

    training_counts = {split: len(train_image_ids[split]) for split in CRELLO_SPLITS}
    if training_counts != CRELLO_TRAINING_IMAGE_COUNTS:
        raise ValueError(f"Crello v1 filtered image counts mismatch: {training_counts}")

    (output_dir / "vocabulary.json").write_bytes(
        (source_dir / "vocabulary.json").read_bytes()
    )
    (output_dir / "count.json").write_text(
        json.dumps(counts, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    ordered_documents = [
        row for split in CRELLO_SPLITS for row in load_crello_split(output_dir, split)
    ]
    manifest: _Manifest = {
        "source_archive_sha256": CRELLO_V1_SHA256,
        "documents": counts,
        "document_stage_png_count": {
            split: len(all_image_ids[split]) for split in CRELLO_SPLITS
        },
        "pixelvae_training_png_count": training_counts,
        "document_stage_image_ids_first_occurrence": fixture_image_ids(
            ordered_documents
        ),
        "pixelvae_training_image_ids": [
            key for split in CRELLO_SPLITS for key in sorted(train_image_ids[split])
        ],
        "dimension_audit": {"status": "passed", "shape_hw": [256, 256], "channels": 4},
    }
    (output_dir / "crello-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Parse the extracted source and report only aggregate audit results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare_crello_cache(args.source_dir, args.output_dir)
    summary = {
        key: value
        for key, value in manifest.items()
        if key
        not in (
            "document_stage_image_ids_first_occurrence",
            "pixelvae_training_image_ids",
        )
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
