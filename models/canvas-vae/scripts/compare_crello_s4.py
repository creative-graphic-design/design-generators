"""Compare original TensorFlow preprocessing with the production data loader."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
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
    CrelloSplit,
    fixture_image_ids,
    image_id,
    load_crello_split,
    training_image_ids,
)
from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
from canvas_vae.training.sampling import (
    CrossEpochBatchSampler,
    sequential_batches,
    wrapping_batches,
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
    loader_batches: int
    loader_samples: int
    document_stage_pngs: int
    pixelvae_training_pngs: int
    processed_fields: list[str]
    masks: list[str]


class _S4Summary(TypedDict):
    """Aggregate exact data equality result."""

    exact: bool
    embedding_contract: str
    splits: dict[str, _SplitSummary]


class _RecordingCrossEpochBatchSampler(CrossEpochBatchSampler):
    """Record the ordered batches consumed by the production DataLoader."""

    def __init__(
        self,
        num_records: int,
        batch_size: int,
        generator: torch.Generator,
        *,
        ordered_indices: Sequence[int],
    ) -> None:
        super().__init__(
            num_records,
            batch_size,
            generator,
            ordered_indices=ordered_indices,
        )
        self.produced_batches: list[list[int]] = []

    def __iter__(self) -> Iterator[list[int]]:
        for batch in super().__iter__():
            self.produced_batches.append(batch)
            yield batch


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


def _production_loader_replay(
    data_module: CanvasVAECrelloDataModule,
    replay_data_module: CanvasVAECrelloDataModule,
    split: str,
    batch_size: int,
) -> Iterator[tuple[BatchFeature, BatchFeature, list[str]]]:
    selected = CrelloSplit(split)
    documents = data_module.splits[selected].documents
    document_ids = [document["document_id"] for document in documents]
    index_by_id = {document_id: index for index, document_id in enumerate(document_ids)}
    first_sampler: _RecordingCrossEpochBatchSampler | None = None
    replay_sampler: _RecordingCrossEpochBatchSampler | None = None

    if selected == CrelloSplit.train:
        if (
            data_module.train_sampler is None
            or replay_data_module.train_sampler is None
        ):
            raise RuntimeError(
                "Crello data module setup did not create its train sampler"
            )

        total_samples = len(data_module.train_sampler) * batch_size
        ordered_indices = [
            index_by_id[document_id]
            for document_id in (
                document_ids * math.ceil(total_samples / len(document_ids))
            )[:total_samples]
        ]
        first_sampler = _RecordingCrossEpochBatchSampler(
            len(documents),
            batch_size,
            torch.Generator().manual_seed(0),
            ordered_indices=ordered_indices,
        )
        replay_sampler = _RecordingCrossEpochBatchSampler(
            len(documents),
            batch_size,
            torch.Generator().manual_seed(0),
            ordered_indices=ordered_indices,
        )
        data_module.train_sampler = first_sampler
        actual_loader = data_module.train_dataloader()
        replay_data_module.train_sampler = replay_sampler
        replay_loader = replay_data_module.train_dataloader()
        expected_batches = [
            ordered_indices[start : start + batch_size]
            for start in range(0, total_samples, batch_size)
        ]
    else:
        expected_batches = (
            wrapping_batches(len(documents), batch_size)
            if selected == CrelloSplit.val
            else sequential_batches(len(documents), batch_size)
        )
        actual_loader = (
            data_module.val_dataloader()
            if selected == CrelloSplit.val
            else data_module.test_dataloader()
        )
        replay_loader = (
            replay_data_module.val_dataloader()
            if selected == CrelloSplit.val
            else replay_data_module.test_dataloader()
        )
        batch_sampler = actual_loader.batch_sampler
        if batch_sampler is None or list(batch_sampler) != expected_batches:
            raise ValueError(f"production sampler order differs for {split}")

    for actual, replay, index_batch in zip(
        actual_loader, replay_loader, expected_batches, strict=True
    ):
        expected_ids = [document_ids[index] for index in index_batch]
        if actual["document_id"] != expected_ids:
            raise ValueError(f"production DataLoader IDs or order differ for {split}")

        _assert_replay_equal(actual, replay)
        yield actual, replay, expected_ids

    if selected == CrelloSplit.train:
        if first_sampler is None or replay_sampler is None:
            raise RuntimeError("deterministic train samplers were not created")

        if first_sampler.produced_batches != expected_batches:
            raise ValueError("production train sampler order differs")
        if replay_sampler.produced_batches != expected_batches:
            raise ValueError("replayed train sampler order differs")


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

    data_module = CanvasVAECrelloDataModule(
        data_dir=str(args.package_dir),
        fixture_dir=str(args.fixture_dir),
        batch_size=args.batch_size,
        num_workers=0,
        seed=0,
    )
    data_module.setup("fit")
    replay_data_module = CanvasVAECrelloDataModule(
        data_dir=str(args.package_dir),
        fixture_dir=str(args.fixture_dir),
        batch_size=args.batch_size,
        num_workers=0,
        seed=0,
    )
    replay_data_module.setup("fit")
    splits = {
        split: data_module.splits[CrelloSplit(split)].documents
        for split in CRELLO_SPLITS
    }
    all_documents = [document for split in CRELLO_SPLITS for document in splits[split]]
    expected_ids = fixture_image_ids(all_documents)
    if data_module.processor is None or replay_data_module.processor is None:
        raise RuntimeError("Crello data module setup did not create its processor")

    embeddings = data_module.processor.embeddings
    if list(embeddings) != expected_ids:
        raise ValueError(
            "data module embedding IDs differ from canonical fixture order"
        )

    vocabularies = data_module.processor.vocabularies
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
            raise ValueError(f"production data module changed package split {split}")

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

        document_by_id = {document["document_id"]: document for document in documents}

        observed_ids: list[str] = []
        loader_batches = 0
        for actual, _replay, expected_batch_ids in _production_loader_replay(
            data_module,
            replay_data_module,
            split,
            args.batch_size,
        ):
            loader_batches += 1
            observed_ids.extend(expected_batch_ids)
            package_rows = [
                document_by_id[document_id] for document_id in expected_batch_ids
            ]
            reference_rows = [
                original_by_id[document_id] for document_id in expected_batch_ids
            ]
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
                    raise ValueError(f"exact S4 mask mismatch: {key} in {split}")

            if not np.array_equal(actual["element_mask"].numpy(), element_mask):
                raise ValueError(f"exact S4 mask mismatch: element_mask in {split}")

        summary["splits"][split] = {
            "documents": len(documents),
            "loader_batches": loader_batches,
            "loader_samples": len(observed_ids),
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
