"""Audit every Crello v1 document PNG header without retaining image bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final

SOURCE_BYTES: Final[int] = 2_989_732_284
SOURCE_SHA256: Final[str] = (
    "f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e"
)
SOURCE_URL: Final[str] = (
    "https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip"
)
EXPECTED_DOCUMENTS: Final[dict[str, int]] = {
    "train": 18_768,
    "val": 2_315,
    "test": 2_278,
}
PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    """Parse archive audit paths."""
    parser = argparse.ArgumentParser(
        description="Verify and scan all Crello v1 document PNG dimensions."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help=f"private local path to the v1 archive (source: {SOURCE_URL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/pixel-vae/crello-v1-dimensions.json"),
        help="private JSON report path (default: .cache/pixel-vae/crello-v1-dimensions.json)",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path(".cache/pixel-vae/crello-v1-tfrecords"),
        help="temporary TFRecord extraction directory (default: .cache/pixel-vae/crello-v1-tfrecords)",
    )
    return parser.parse_args()


def main() -> None:
    """Verify the archive and count PNG dimensions grouped by element type."""
    import tensorflow as tf

    args = parse_args()
    if args.archive.stat().st_size != SOURCE_BYTES:
        raise ValueError(
            f"unexpected Crello v1 archive size: {args.archive.stat().st_size}"
        )
    archive_digest = _file_sha256(args.archive)
    if archive_digest != SOURCE_SHA256:
        raise ValueError(f"unexpected Crello v1 SHA-256: {archive_digest}")

    args.extract_dir.mkdir(parents=True, exist_ok=True)
    record_paths = _extract_tfrecords(args.archive, args.extract_dir)
    document_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    dimension_counts: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_pngs: Counter[str] = Counter()
    non_256_examples: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    text_element_counts: Counter[str] = Counter()
    text_element_hashes: dict[str, set[str]] = {
        split: set() for split in EXPECTED_DOCUMENTS
    }

    for split in ("train", "val", "test"):
        split_paths = sorted(
            path for path in record_paths if path.name.startswith(f"{split}-")
        )
        for serialized in tf.data.TFRecordDataset([str(path) for path in split_paths]):
            example = tf.train.SequenceExample.FromString(serialized.numpy())
            document_counts[split] += 1
            types = example.feature_lists.feature_list["type"].feature
            images = example.feature_lists.feature_list["image_bytes"].feature
            if len(types) != len(images):
                raise ValueError(f"type/image_bytes sequence length differs in {split}")
            document_id = (
                example.context.feature["id"].bytes_list.value[0].decode("utf-8")
            )
            for element_index, (element_type, encoded) in enumerate(
                zip(types, images, strict=True)
            ):
                kind = element_type.bytes_list.value[0].decode("utf-8")
                png_bytes = encoded.bytes_list.value[0]
                type_counts[kind] += 1
                if kind == "textElement":
                    text_element_counts[split] += 1
                    text_element_hashes[split].add(
                        hashlib.sha256(png_bytes).hexdigest()
                    )
                try:
                    height, width = png_dimensions(png_bytes)
                    decoded = tf.io.decode_png(png_bytes, channels=4)
                    if tuple(decoded.shape) != (height, width, 4):
                        raise ValueError(
                            "decoded RGBA shape disagrees with the PNG header"
                        )
                except (ValueError, tf.errors.InvalidArgumentError):
                    invalid_pngs[kind] += 1
                    continue
                size = f"{width}x{height}"
                dimension_counts[kind][size] += 1
                if (height, width) != (256, 256) and len(non_256_examples[kind]) < 10:
                    non_256_examples[kind].append(
                        {
                            "split": split,
                            "document_id": document_id,
                            "element_index": element_index,
                            "size": size,
                        }
                    )

    if dict(document_counts) != EXPECTED_DOCUMENTS:
        raise ValueError(f"unexpected document counts: {dict(document_counts)}")
    text_element_identity, all_splits_text_identical = (
        summarize_text_element_png_identity(text_element_counts, text_element_hashes)
    )
    report = {
        "source_url": SOURCE_URL,
        "source_bytes": SOURCE_BYTES,
        "source_sha256": archive_digest,
        "document_counts": dict(document_counts),
        "element_counts": dict(type_counts),
        "dimensions_by_element_type": {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(dimension_counts.items())
        },
        "invalid_png_counts": dict(invalid_pngs),
        "non_256_examples": dict(non_256_examples),
        "text_element_png_identity": {
            "by_split": text_element_identity,
            "across_splits_byte_identical": all_splits_text_identical,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """Read the PNG IHDR dimensions without decoding or storing pixels."""
    if (
        len(png_bytes) < 24
        or png_bytes[:8] != PNG_SIGNATURE
        or png_bytes[12:16] != b"IHDR"
    ):
        raise ValueError("image_bytes does not start with a PNG IHDR")
    width, height = struct.unpack(">II", png_bytes[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    return height, width


def summarize_text_element_png_identity(
    counts_by_split: Mapping[str, int],
    hashes_by_split: Mapping[str, set[str]],
) -> tuple[dict[str, dict[str, int | bool]], bool]:
    """Summarize whether text-element PNG bytes are unique within and across splits."""
    by_split = {
        split: {
            "element_count": counts_by_split.get(split, 0),
            "unique_png_count": len(hashes_by_split.get(split, set())),
            "within_split_byte_identical": counts_by_split.get(split, 0) > 0
            and len(hashes_by_split.get(split, set())) == 1,
        }
        for split in EXPECTED_DOCUMENTS
    }
    all_hashes = set().union(
        *(hashes_by_split.get(split, set()) for split in EXPECTED_DOCUMENTS)
    )
    across_splits = (
        all(
            counts_by_split.get(split, 0) > 0
            and len(hashes_by_split.get(split, set())) == 1
            for split in EXPECTED_DOCUMENTS
        )
        and len(all_hashes) == 1
    )
    return by_split, across_splits


def _extract_tfrecords(archive_path: Path, output_dir: Path) -> list[Path]:
    record_paths: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".tfrecord"):
                continue
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member path: {info.filename}")
            target = output_dir.joinpath(*relative.parts).resolve()
            if output_dir.resolve() not in target.parents:
                raise ValueError(
                    f"archive member escapes extraction directory: {info.filename}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
            record_paths.append(target)
    if not record_paths:
        raise ValueError("archive contains no TFRecord files")
    return record_paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
