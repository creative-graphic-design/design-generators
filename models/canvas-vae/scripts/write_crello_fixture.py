"""Encode canonical unique Crello PNGs into the private embedding fixture."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Protocol, Sequence, TypedDict, cast

import numpy as np
from jaxtyping import Float, UInt8

from canvas_vae.data import (
    CRELLO_SPLITS,
    CrelloDocument,
    CrelloEmbeddingManifest,
    fixture_image_ids,
    image_id,
    load_crello_split,
    write_embedding_fixture,
)

_ENCODER_BATCH_SIZE = 32
_RGBA_SHAPE = (256, 256, 4)


class _BytesList(Protocol):
    value: Sequence[bytes]


class _FloatList(Protocol):
    value: Sequence[float]


class _Int64List(Protocol):
    value: Sequence[int]


class _Feature(Protocol):
    bytes_list: _BytesList
    float_list: _FloatList
    int64_list: _Int64List

    def WhichOneof(self, field: str) -> str | None: ...


class _Context(Protocol):
    feature: dict[str, _Feature]


class _FeatureSequence(Protocol):
    feature: Sequence[_Feature]


class _FeatureLists(Protocol):
    feature_list: dict[str, _FeatureSequence]


class _SequenceExample(Protocol):
    context: _Context
    feature_lists: _FeatureLists


class _BatchPosition(TypedDict):
    document_id: str
    element_index: int
    batch_size: int


class _DuplicateDiagnostic(TypedDict):
    max_abs_difference: float
    image_id: str | None
    first_occurrence: _BatchPosition | None
    duplicate_occurrence: _BatchPosition | None


def _source_id(record: _SequenceExample) -> str:
    feature = record.context.feature["id"]
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        return feature.bytes_list.value[0].decode("utf-8")

    if kind == "int64_list":
        return str(feature.int64_list.value[0])

    raise ValueError("Crello reference record has no scalar context id")


def _reference_records(reference_dir: Path, split: str) -> dict[str, bytes]:
    import tensorflow as tf

    paths = sorted((reference_dir / "crello-document").glob(f"{split}-*"))
    if not paths:
        raise FileNotFoundError(
            f"no {split} original document records under {reference_dir}"
        )

    output: dict[str, bytes] = {}
    for path in paths:
        for serialized in tf.compat.v1.io.tf_record_iterator(str(path)):
            record = cast(
                _SequenceExample,
                tf.train.SequenceExample.FromString(serialized),
            )
            key = f"crello-v1/{split}/{_source_id(record)}"
            if key in output:
                raise ValueError(f"duplicate original document ID: {key}")

            output[key] = serialized

    return output


def _duplicate_embedding_diagnostic(
    documents: Sequence[CrelloDocument], reference_dir: Path
) -> _DuplicateDiagnostic:
    """Report the largest legacy duplicate difference against its first occurrence."""
    import tensorflow as tf

    ordered_documents = sorted(
        documents,
        key=lambda row: (
            CRELLO_SPLITS.index(row["split"]),
            row["document_id"].encode("utf-8"),
        ),
    )
    first: dict[str, tuple[Float[np.ndarray, "256"], _BatchPosition]] = {}
    report: _DuplicateDiagnostic = {
        "max_abs_difference": 0.0,
        "image_id": None,
        "first_occurrence": None,
        "duplicate_occurrence": None,
    }
    by_split = {
        split: _reference_records(reference_dir, split) for split in CRELLO_SPLITS
    }
    for split in CRELLO_SPLITS:
        expected = {
            document["document_id"]
            for document in ordered_documents
            if document["split"] == split
        }
        if expected != set(by_split[split]):
            raise ValueError(f"original document IDs differ for {split}")

    for document in ordered_documents:
        try:
            serialized = by_split[document["split"]][document["document_id"]]
        except KeyError as error:
            raise ValueError(
                f"original preprocessing omitted {document['document_id']}"
            ) from error

        record = tf.train.SequenceExample.FromString(serialized)
        features = record.feature_lists.feature_list["image_embedding"].feature
        if len(features) != len(document["elements"]):
            raise ValueError(
                f"original element count differs for {document['document_id']}"
            )

        batch_size = len(features)
        for element_index, (element, feature) in enumerate(
            zip(document["elements"], features, strict=True)
        ):
            value = np.asarray(feature.float_list.value, dtype=np.float32)
            if value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError(
                    f"invalid reference posterior mean for {element['image_id']}"
                )

            position: _BatchPosition = {
                "document_id": document["document_id"],
                "element_index": element_index,
                "batch_size": batch_size,
            }
            previous = first.get(element["image_id"])
            if previous is None:
                first[element["image_id"]] = (value, position)
                continue

            difference = float(np.max(np.abs(previous[0] - value)))
            if report["image_id"] is None or difference > report["max_abs_difference"]:
                report = {
                    "max_abs_difference": difference,
                    "image_id": element["image_id"],
                    "first_occurrence": previous[1],
                    "duplicate_occurrence": position,
                }

    return report


def _read_png(package_dir: Path, key: str) -> UInt8[np.ndarray, "256 256 4"]:
    import tensorflow as tf

    namespace, split, digest = key.split("/", maxsplit=2)
    if (
        namespace != "crello-v1"
        or split not in CRELLO_SPLITS
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError(f"invalid Crello fixture image ID: {key}")

    path = package_dir / "images" / split / f"{digest}.png"
    payload = path.read_bytes()
    if image_id(split, payload) != key:
        raise ValueError(f"cached PNG bytes do not match image ID: {key}")

    decoded = tf.io.decode_png(payload, channels=4)
    value = decoded.numpy()
    if value.dtype != np.uint8 or value.shape != _RGBA_SHAPE:
        raise ValueError(f"invalid RGBA PNG dimensions for {key}: {value.shape}")

    return value


def _encode_unique_images(
    image_ids: Sequence[str], package_dir: Path, encoder_path: Path
) -> Float[np.ndarray, "images 256"]:
    """Encode each canonical ID once using a constant padded CPU batch shape."""
    import tensorflow as tf

    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("fixture encoder input IDs must be non-empty and unique")

    encoder = tf.keras.models.load_model(str(encoder_path), compile=False)
    output = np.empty((len(image_ids), 256), dtype=np.float32)
    for start in range(0, len(image_ids), _ENCODER_BATCH_SIZE):
        batch_ids = image_ids[start : start + _ENCODER_BATCH_SIZE]
        images = [_read_png(package_dir, key) for key in batch_ids]
        images.extend([images[-1]] * (_ENCODER_BATCH_SIZE - len(images)))
        inputs = tf.convert_to_tensor(np.stack(images))
        encoded = np.asarray(encoder(inputs, training=False).numpy())
        if encoded.shape != (_ENCODER_BATCH_SIZE, 256):
            raise ValueError(f"encoder returned invalid shape: {encoded.shape}")

        if encoded.dtype != np.float32 or not np.isfinite(encoded).all():
            raise ValueError("encoder returned non-finite or non-float32 values")

        output[start : start + len(batch_ids)] = encoded[: len(batch_ids)]

    return output


def create_fixture(
    documents: Sequence[CrelloDocument],
    reference_dir: Path,
    package_dir: Path,
    fixture_dir: Path,
    encoder_path: Path,
    encoder_state_sha256: str,
) -> tuple[CrelloEmbeddingManifest, _DuplicateDiagnostic]:
    """Build canonical per-image embeddings and diagnose legacy batch variation."""
    image_ids = fixture_image_ids(documents)
    diagnostic = _duplicate_embedding_diagnostic(documents, reference_dir)
    values = _encode_unique_images(image_ids, package_dir, encoder_path)
    manifest = write_embedding_fixture(
        fixture_dir,
        image_ids,
        values,
        encoder_state_sha256=encoder_state_sha256,
    )
    return manifest, diagnostic


def main() -> None:
    """Build the all-document PNG fixture without copying source images into it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--encoder-state-sha256", required=True)
    args = parser.parse_args()
    documents = [
        document
        for split in CRELLO_SPLITS
        for document in load_crello_split(args.package_dir, split)
    ]
    manifest, diagnostic = create_fixture(
        documents,
        args.reference_run,
        args.package_dir,
        args.fixture_dir,
        args.encoder_path,
        args.encoder_state_sha256,
    )
    print(
        json.dumps(
            {
                "fixture_rows": len(manifest["records"]),
                "encoder_batch_size": _ENCODER_BATCH_SIZE,
                "shape": manifest["shape"],
                "dtype": manifest["dtype"],
                "array_sha256": manifest["array_sha256"],
                "duplicate_embedding_diagnostic": diagnostic,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
