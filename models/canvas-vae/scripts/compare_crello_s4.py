"""Compare original TensorFlow Crello preprocessing with the package loader."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Float, Shaped
from transformers import BatchFeature

from canvas_vae.data import (
    CRELLO_CONTEXT_FIELDS,
    CRELLO_SEQUENCE_FIELDS,
    CRELLO_SPLITS,
    CrelloDocument,
    CrelloProcessor,
    fixture_image_ids,
    image_id,
    load_crello_split,
    load_crello_vocabularies,
    load_embedding_fixture,
    training_image_ids,
)


class _TensorOutput(Protocol):
    """Reference TensorFlow tensor needed for exact NumPy comparison."""

    def numpy(self) -> Shaped[np.ndarray, "..."]: ...


class _BytesList(Protocol):
    value: Sequence[bytes]


class _Int64List(Protocol):
    value: Sequence[int]


class _Feature(Protocol):
    bytes_list: _BytesList
    int64_list: _Int64List

    def WhichOneof(self, field: str) -> str | None: ...


class _FeatureSequence(Protocol):
    feature: Sequence[_Feature]


class _FeatureLists(Protocol):
    feature_list: Mapping[str, _FeatureSequence]


class _Context(Protocol):
    feature: Mapping[str, _Feature]


class _SequenceExample(Protocol):
    context: _Context
    feature_lists: _FeatureLists


class _SplitSummary(TypedDict):
    """Count and equality details for one data split."""

    documents: int
    document_stage_pngs: int
    pixelvae_training_pngs: int
    processed_fields: list[str]
    masks: list[str]


class _S4Summary(TypedDict):
    """Aggregate exact data equality result."""

    exact: bool
    embedding_contract: str
    splits: dict[str, _SplitSummary]


def _source_id(record: _SequenceExample) -> str:
    feature = record.context.feature["id"]
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        return feature.bytes_list.value[0].decode("utf-8")

    if kind == "int64_list":
        return str(feature.int64_list.value[0])

    raise ValueError("Crello source record has no scalar context id")


def _reference_records(reference_run: Path, split: str) -> dict[str, bytes]:
    import tensorflow as tf

    paths = sorted((reference_run / "crello-document").glob(f"{split}-*"))
    if not paths:
        raise FileNotFoundError(
            f"no {split} original document records under {reference_run}"
        )

    records: dict[str, bytes] = {}
    for path in paths:
        for serialized in tf.compat.v1.io.tf_record_iterator(str(path)):
            record = cast(
                _SequenceExample, tf.train.SequenceExample.FromString(serialized)
            )
            key = f"crello-v1/{split}/{_source_id(record)}"
            if key in records:
                raise ValueError(f"duplicate original document ID: {key}")

            records[key] = serialized

    return dict(sorted(records.items(), key=lambda row: row[0].encode("utf-8")))


def _reference_training_image_ids(reference_run: Path, split: str) -> set[str]:
    import tensorflow as tf

    paths = sorted((reference_run / "crello-image").glob(f"{split}-*"))
    if not paths:
        raise FileNotFoundError(
            f"no {split} original image records under {reference_run}"
        )

    keys: set[str] = set()
    for path in paths:
        for serialized in tf.compat.v1.io.tf_record_iterator(str(path)):
            record = tf.train.Example.FromString(serialized)
            png = record.features.feature["image"].bytes_list.value[0]
            keys.add(image_id(split, png))

    return keys


def _canonical_embedding_values(
    documents: Sequence[CrelloDocument],
    embeddings: Mapping[str, Float[np.ndarray, "256"]],
) -> Float[np.ndarray, "documents elements 256"]:
    lengths = [len(document["elements"]) for document in documents]
    values = np.zeros((len(documents), max(lengths, default=0), 256), dtype=np.float32)
    for row_index, document in enumerate(documents):
        for element_index, element in enumerate(document["elements"]):
            values[row_index, element_index] = embeddings[element["image_id"]]

    return values


def _assert_batch_equal(
    actual: BatchFeature, expected: Mapping[str, _TensorOutput]
) -> None:
    for name in (*CRELLO_CONTEXT_FIELDS, *CRELLO_SEQUENCE_FIELDS):
        actual_value = cast(Shaped[torch.Tensor, "..."], actual[name])
        expected_value = expected[name]
        actual_array = actual_value.detach().cpu().numpy()
        expected_array = expected_value.numpy()
        if actual_array.dtype != expected_array.dtype or not np.array_equal(
            actual_array, expected_array
        ):
            raise ValueError(f"exact S4 field mismatch: {name}")


def _assert_replay_equal(first: BatchFeature, second: BatchFeature) -> None:
    for name in first:
        left = first[name]
        right = second[name]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                raise ValueError(f"deterministic loader replay differs at {name}")
        elif isinstance(right, torch.Tensor) or left != right:
            raise ValueError(f"deterministic loader replay differs at {name}")


def main() -> None:
    """Check all source splits, processed fields, masks, IDs, and replay order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor_root / "src" / "canvas-vae"))

    import tensorflow as tf
    from canvasvae.data.spec import DataSpec

    splits = {
        split: load_crello_split(args.package_dir, split) for split in CRELLO_SPLITS
    }
    all_documents = [document for split in CRELLO_SPLITS for document in splits[split]]
    expected_ids = fixture_image_ids(all_documents)
    embeddings = load_embedding_fixture(args.fixture_dir, expected_ids)
    vocabularies = load_crello_vocabularies(args.package_dir)
    processor = CrelloProcessor(vocabularies, embeddings)
    summary: _S4Summary = {
        "exact": True,
        "embedding_contract": "canonical unique-image fixture; legacy duplicate outputs are diagnostic-only",
        "splits": {},
    }
    reference_dir = args.reference_run / "crello-document"
    data_spec = DataSpec(
        "crello-document", str(reference_dir), batch_size=args.batch_size
    )
    for field, values in vocabularies.items():
        if data_spec.preprocessor[field].get_vocabulary() != values:
            raise ValueError(f"source/package vocabulary order differs for {field}")

    type_loss_masks = {
        field: np.asarray(
            data_spec.make_input_columns()[field]["loss_condition"]["mask"], dtype=bool
        )
        for field in ("color", "image_embedding")
    }

    for split in CRELLO_SPLITS:
        documents = splits[split]
        if documents != load_crello_split(args.package_dir, split):
            raise ValueError(f"package split loader replay differs for {split}")

        original_by_id = _reference_records(args.reference_run, split)
        document_ids = [document["document_id"] for document in documents]
        if document_ids != list(original_by_id):
            raise ValueError(
                f"original/package document IDs or order differ for {split}"
            )

        expected_image_ids = set(training_image_ids(documents))
        if expected_image_ids != _reference_training_image_ids(
            args.reference_run, split
        ):
            raise ValueError(f"original/package filtered image IDs differ for {split}")

        serialized = list(original_by_id.values())
        for start in range(0, len(documents), args.batch_size):
            package_rows = documents[start : start + args.batch_size]
            reference_rows = serialized[start : start + args.batch_size]
            actual = processor(package_rows)
            replay = processor(package_rows)
            _assert_replay_equal(actual, replay)
            expected = dict(data_spec.parse_fn(tf.constant(reference_rows)))
            expected["image_embedding"] = tf.convert_to_tensor(
                _canonical_embedding_values(package_rows, embeddings)
            )
            _assert_batch_equal(actual, expected)

            lengths = np.asarray([len(row["elements"]) for row in package_rows])
            if not np.array_equal(actual["num_elements"].numpy(), lengths):
                raise ValueError(f"exact S4 element counts differ for {split}")

            type_ids = expected["type"].numpy()[..., 0]
            element_mask = np.arange(type_ids.shape[1])[None, :] < lengths[:, None]
            for field, key in (
                ("color", "color_mask"),
                ("image_embedding", "image_embedding_mask"),
            ):
                mask = type_loss_masks[field][type_ids] & element_mask
                if not np.array_equal(actual[key].numpy(), mask):
                    raise ValueError(f"exact S4 mask mismatch: {key}")

            if not np.array_equal(actual["element_mask"].numpy(), element_mask):
                raise ValueError("exact S4 mask mismatch: element_mask")

        summary["splits"][split] = {
            "documents": len(documents),
            "document_stage_pngs": len(
                {
                    element["image_id"]
                    for row in documents
                    for element in row["elements"]
                }
            ),
            "pixelvae_training_pngs": len(expected_image_ids),
            "processed_fields": [*CRELLO_CONTEXT_FIELDS, *CRELLO_SEQUENCE_FIELDS],
            "masks": ["element_mask", "color_mask", "image_embedding_mask"],
        }

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
