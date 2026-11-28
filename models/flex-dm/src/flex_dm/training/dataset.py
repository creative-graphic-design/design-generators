"""Read and replay the original Flex-DM training TFRecords."""

import json
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import torch
from jaxtyping import Int, Shaped
from tfrecord import example_pb2
from tfrecord.reader import extract_feature_dict
from tfrecord.tools.tfrecord2idx import create_index
from tfrecord.writer import TFRecordWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..configuration_flex_dm import (
    FlexDmColumnSpec,
    FlexDmConfigValue,
    FlexDmDatasetName,
)
from ..data_specs import (
    CRELLO_MODEL_TYPE_VOCABULARY,
    FlexDmVocabularyValue,
    build_column_specs,
    load_builtin_spec,
)

_RecordRef = tuple[Path, int, int]
_DISCRETIZATION = {
    "left": (0.0, 1.0, 64),
    "top": (0.0, 1.0, 64),
    "width": (0.0, 1.0, 64),
    "height": (0.0, 1.0, 64),
    "opacity": (0.0, 1.0, 8),
    "color": (0.0, 255.0, 16),
}
_FREQUENCY_LOOKUPS = {"font_family", "icon", "text_button"}
_INTEGER_LOOKUPS = {"canvas_width", "canvas_height"}
_FEATURE_TYPES = {"string": "byte", "int64": "int", "float32": "float"}


def _read_record(record_file: BinaryIO) -> bytes | None:
    offset = record_file.tell()
    length_bytes = record_file.read(8)
    if not length_bytes:
        return None

    if len(length_bytes) != 8:
        raise RuntimeError(f"Failed to read the record size at byte {offset}.")

    length_crc = record_file.read(4)
    if len(length_crc) != 4:
        raise RuntimeError(f"Failed to read the length checksum at byte {offset}.")

    if length_crc != TFRecordWriter.masked_crc(length_bytes):
        raise RuntimeError(f"Invalid length checksum at byte {offset}.")

    (length,) = struct.unpack("<Q", length_bytes)
    payload = record_file.read(length)
    if len(payload) != length:
        raise RuntimeError(f"Failed to read the record at byte {offset}.")

    payload_crc = record_file.read(4)
    if len(payload_crc) != 4:
        raise RuntimeError(f"Failed to read the payload checksum at byte {offset}.")

    if payload_crc != TFRecordWriter.masked_crc(payload):
        raise RuntimeError(f"Invalid payload checksum at byte {offset}.")

    return payload


def _read_json(path: Path) -> dict[str, FlexDmConfigValue]:
    if not isinstance(value := json.loads(path.read_text(encoding="utf-8")), dict):
        raise TypeError(f"expected a JSON object in {path}")

    return cast(dict[str, FlexDmConfigValue], value)


def _lookup_ids(
    key: str,
    values: Sequence[str],
    dataset: FlexDmDatasetName,
    vocabulary: Mapping[str, FlexDmVocabularyValue],
) -> Int[np.ndarray, "..."]:
    if dataset is FlexDmDatasetName.crello and key == "type":
        mapping = {
            value: index for index, value in enumerate(CRELLO_MODEL_TYPE_VOCABULARY)
        }
        return np.asarray([mapping[value] for value in values], dtype=np.int64)

    raw = vocabulary.get(key)
    if raw is None:
        raise ValueError(f"missing lookup vocabulary for {key}")

    if not isinstance(raw, dict):
        raise TypeError(f"expected a lookup vocabulary for {key}")

    minimum = 500 if key in _FREQUENCY_LOOKUPS else 1
    mapping = {
        str(value): index + 1
        for index, (value, count) in enumerate(
            item for item in raw.items() if int(cast(int, item[1])) >= minimum
        )
    }
    strict = key in {"group", "format", "category"}
    if strict:
        mapping[""] = 0

    return np.asarray(
        [mapping[value] if strict else mapping.get(value, 0) for value in values],
        dtype=np.int64,
    )


def _transform(
    key: str,
    value: Shaped[np.ndarray, "..."] | list[Shaped[np.ndarray, "..."]],
    column: FlexDmColumnSpec,
    dataset: FlexDmDatasetName,
    vocabulary: Mapping[str, FlexDmVocabularyValue],
) -> Shaped[torch.Tensor, "..."]:
    array = np.asarray(value)
    if array.dtype.kind == "S":
        array = _lookup_ids(
            key,
            [item.decode("utf-8") for item in array.reshape(-1)],
            dataset,
            vocabulary,
        )
    elif key in _DISCRETIZATION:
        minimum, maximum, bins = _DISCRETIZATION[key]
        boundaries = np.linspace(minimum, maximum, bins)[1:]
        array = np.searchsorted(boundaries, array, side="right")
    elif key == "length":
        array = array.astype(np.int64) - 1
    elif key in _INTEGER_LOOKUPS:
        raw = vocabulary.get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"missing lookup vocabulary for {key}")

        mapping = {int(v): i for i, v in enumerate(cast(Mapping[str, int], raw))}
        array = np.asarray([mapping[int(item)] for item in array.reshape(-1)])

    target_shape = (-1, *column["shape"]) if column["is_sequence"] else column["shape"]
    array = np.asarray(array).reshape(target_shape)
    dtype = torch.float32 if column["type"] == "numerical" else torch.long
    return torch.as_tensor(array, dtype=dtype)


@dataclass(slots=True)
class FlexDmTfrecordDataset(Dataset[dict[str, Shaped[torch.Tensor, "..."]]]):
    """Map-style reader with stable record IDs and deterministic order replay."""

    data_dir: Path
    dataset_name: FlexDmDatasetName | str
    split: str
    record_order: Sequence[int] | None = None
    sample_order: Sequence[int] | None = None
    _columns: dict[str, FlexDmColumnSpec] = field(init=False, repr=False)
    _vocabulary: dict[str, FlexDmVocabularyValue] = field(init=False, repr=False)
    _records: tuple[_RecordRef, ...] = field(init=False, repr=False)
    _context_description: dict[str, str] = field(init=False, repr=False)
    _features_description: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate source metadata and prepare indexed record references."""
        self.data_dir = Path(self.data_dir)
        try:
            self.dataset_name = FlexDmDatasetName(str(self.dataset_name).lower())
        except ValueError as exc:
            raise ValueError(
                f"unsupported Flex-DM dataset: {self.dataset_name}"
            ) from exc

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Flex-DM split: {self.split}")

        source_columns = load_builtin_spec(self.dataset_name)["columns"]
        self._vocabulary = cast(
            dict[str, FlexDmVocabularyValue],
            _read_json(self.data_dir / "vocabulary.json"),
        )
        self._columns = build_column_specs(
            dataset_name=self.dataset_name, vocabulary=self._vocabulary
        )
        counts = _read_json(self.data_dir / "count.json")
        expected_count = int(cast(int, counts[self.split]))
        descriptions: dict[bool, dict[str, str]] = {False: {}, True: {}}
        for key, column in source_columns.items():
            descriptions[column["is_sequence"]][key] = _FEATURE_TYPES[column["dtype"]]

        self._context_description, self._features_description = (
            descriptions[False],
            descriptions[True],
        )

        records: list[_RecordRef] = []
        for record_path in sorted(self.data_dir.glob(f"{self.split}-*.tfrecord")):
            index_path = record_path.with_suffix(record_path.suffix + ".idx")
            if not index_path.exists():
                temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
                create_index(str(record_path), str(temporary_path))
                os.replace(temporary_path, index_path)

            offsets = np.loadtxt(index_path, dtype=np.int64, ndmin=2)
            records.extend(
                (record_path, record_index, int(offset))
                for record_index, offset in enumerate(offsets[:, 0])
            )

        if len(records) != expected_count:
            raise ValueError(
                f"{self.split} record count {len(records)} does not match count.json {expected_count}"
            )

        if self.record_order is not None and self.sample_order is not None:
            raise ValueError("record_order and sample_order cannot both be set")

        if self.record_order is not None:
            order = tuple(self.record_order)
            if sorted(order) != list(range(len(records))):
                raise ValueError(
                    "record_order must be a permutation of all split indices"
                )

            records = [records[index] for index in order]
            self.record_order = order

        if self.sample_order is not None:
            order = tuple(self.sample_order)
            if any(index < 0 or index >= len(records) for index in order):
                raise ValueError(
                    "sample_order indices must refer to records in this split"
                )

            records = [records[index] for index in order]
            self.sample_order = order

        self._records = tuple(records)

    def __len__(self) -> int:
        """Return the number of records in this split."""
        return len(self._records)

    def sample_id(self, index: int) -> str:
        """Return the source file and record index for one example."""
        record_path, record_index, _offset = self._records[index]
        return f"{record_path.name}:{record_index}"

    def __getitem__(self, index: int) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Decode and preprocess one SequenceExample record."""
        record_path, _record_index, offset = self._records[index]
        with record_path.open("rb") as record_file:
            record_file.seek(offset)
            payload = _read_record(record_file)

        if payload is None:
            raise ValueError(
                f"missing TFRecord at byte offset {offset} in {record_path}"
            )

        example = example_pb2.SequenceExample()
        example.ParseFromString(payload)
        typename_mapping = {
            "byte": "bytes_list",
            "float": "float_list",
            "int": "int64_list",
        }
        context = extract_feature_dict(
            example.context, self._context_description, typename_mapping
        )
        features = extract_feature_dict(
            example.feature_lists, self._features_description, typename_mapping
        )
        output = {}
        for key, column in self._columns.items():
            value: Shaped[np.ndarray, "..."] | list[Shaped[np.ndarray, "..."]] = (  # noqa: UP037
                features[key] if column["is_sequence"] else context[key]
            )
            output[key] = _transform(
                key,
                value,
                column,
                cast(FlexDmDatasetName, self.dataset_name),
                self._vocabulary,
            )

        return output

    def collate_fn(
        self, examples: Sequence[dict[str, Shaped[torch.Tensor, "..."]]]
    ) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Pad variable-length fields to the longest record in a batch."""
        return {
            key: pad_sequence(
                [x[key] for x in examples],
                batch_first=True,
                padding_value=self._padding_value(key),
            )
            if column["is_sequence"]
            else torch.stack([x[key] for x in examples])
            for key, column in self._columns.items()
        }

    def _padding_value(self, key: str) -> int:
        if key not in _FREQUENCY_LOOKUPS:
            return 0

        return int(
            _lookup_ids(
                key,
                [""],
                cast(FlexDmDatasetName, self.dataset_name),
                self._vocabulary,
            )[0]
        )
