"""Serialize first-occurrence embeddings from original document reference records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol, Sequence, cast

import numpy as np
from jaxtyping import Float

from canvas_vae.data import (
    CRELLO_SPLITS,
    CrelloDocument,
    CrelloEmbeddingManifest,
    fixture_image_ids,
    load_crello_split,
    write_embedding_fixture,
)


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


def _source_id(record: _SequenceExample) -> str:
    feature = record.context.feature["id"]
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        return feature.bytes_list.value[0].decode("utf-8")

    if kind == "int64_list":
        return str(feature.int64_list.value[0])

    raise ValueError("Crello reference record has no scalar context id")


def _reference_embeddings(reference_dir: Path, split: str) -> dict[str, bytes]:
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


def create_fixture(
    documents: Sequence[CrelloDocument],
    reference_dir: Path,
    fixture_dir: Path,
    encoder_state_sha256: str,
) -> CrelloEmbeddingManifest:
    """Join original encoder output by canonical document and element order."""
    import tensorflow as tf

    vectors: dict[str, Float[np.ndarray, "256"]] = {}
    ordered_documents = sorted(
        documents,
        key=lambda row: (
            CRELLO_SPLITS.index(row["split"]),
            row["document_id"].encode("utf-8"),
        ),
    )
    by_split = {
        split: _reference_embeddings(reference_dir, split) for split in CRELLO_SPLITS
    }
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

        for element, feature in zip(document["elements"], features, strict=True):
            value = np.asarray(feature.float_list.value, dtype=np.float32)
            if value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError(
                    f"invalid reference posterior mean for {element['image_id']}"
                )

            previous = vectors.setdefault(element["image_id"], value)
            if not np.array_equal(previous, value):
                raise ValueError(
                    f"duplicate image embedding differs for {element['image_id']}"
                )

    ordered_ids = fixture_image_ids(documents)
    if list(vectors) != ordered_ids:
        raise ValueError(
            "reference image traversal differs from the canonical fixture order"
        )

    matrix = np.stack([vectors[key] for key in ordered_ids])
    return write_embedding_fixture(
        fixture_dir,
        ordered_ids,
        matrix,
        encoder_state_sha256=encoder_state_sha256,
    )


def main() -> None:
    """Build the all-document PNG fixture without copying source images into it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--encoder-state-sha256", required=True)
    args = parser.parse_args()
    documents = [
        document
        for split in CRELLO_SPLITS
        for document in load_crello_split(args.package_dir, split)
    ]
    manifest = create_fixture(
        documents,
        args.reference_run,
        args.fixture_dir,
        args.encoder_state_sha256,
    )
    print(
        json.dumps(
            {
                "fixture_rows": len(manifest["records"]),
                "shape": manifest["shape"],
                "dtype": manifest["dtype"],
                "array_sha256": manifest["array_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
