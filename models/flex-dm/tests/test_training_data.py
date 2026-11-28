"""Flex-DM training TFRecord reader tests."""

import json
from io import BytesIO

import numpy as np
import pytest
from flex_dm.training import dataset as dataset_module
from flex_dm.training.dataset import FlexDmTfrecordDataset
from tfrecord.writer import TFRecordWriter


def test_read_record_validates_both_checksums() -> None:
    payload = b"sequence example"
    length_bytes = len(payload).to_bytes(8, "little")
    record = (
        length_bytes
        + TFRecordWriter.masked_crc(length_bytes)
        + payload
        + TFRecordWriter.masked_crc(payload)
    )

    assert dataset_module._read_record(BytesIO(record)) == payload
    bad_length_checksum = length_bytes + bytes(4) + payload + record[-4:]
    with pytest.raises(RuntimeError, match="Invalid length checksum"):
        dataset_module._read_record(BytesIO(bad_length_checksum))

    bad_payload_checksum = record[:-4] + bytes(4)
    with pytest.raises(RuntimeError, match="Invalid payload checksum"):
        dataset_module._read_record(BytesIO(bad_payload_checksum))


def _write_rico_record(
    writer: TFRecordWriter, *, types: list[bytes], left: list[float]
) -> None:
    length = len(types)
    writer.write(
        {"length": ([length], "int")},
        {
            "left": ([[value] for value in left], "float"),
            "top": ([[0.25] for _ in types], "float"),
            "width": ([[0.5] for _ in types], "float"),
            "height": ([[0.75] for _ in types], "float"),
            "clickable": ([[1] for _ in types], "int"),
            "type": ([[value] for value in types], "byte"),
            "icon": ([[b"menu"] for _ in types], "byte"),
            "text_button": ([[b"login"] for _ in types], "byte"),
        },
    )


def test_tfrecord_dataset_replays_ids_and_pads_variable_sequences(
    tmp_path, monkeypatch
) -> None:
    """The reader preserves record IDs, vocabulary IDs, and batch padding."""
    (tmp_path / "count.json").write_text(json.dumps({"train": 2}))
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            {
                "type": {"Text": 100, "Button": 80},
                "icon": {"": 600, "menu": 600},
                "text_button": {"": 700, "login": 700},
            }
        )
    )
    record_path = tmp_path / "train-00000-of-00001.tfrecord"
    writer = TFRecordWriter(str(record_path))
    _write_rico_record(writer, types=[b"Text"], left=[0.5])
    _write_rico_record(writer, types=[b"Button", b"Text"], left=[0.0, 1.0])
    writer.close()
    original_loadtxt = np.loadtxt
    index_reads = 0

    def count_index_reads(*args, **kwargs):
        nonlocal index_reads
        index_reads += 1
        return original_loadtxt(*args, **kwargs)

    monkeypatch.setattr(dataset_module.np, "loadtxt", count_index_reads)

    dataset = FlexDmTfrecordDataset(
        tmp_path,
        dataset_name="rico",
        split="train",
        record_order=(1, 0),
    )

    first = dataset[0]
    second = dataset[1]
    expected_left = int(np.searchsorted(np.linspace(0, 1, 64)[1:], 0.0, side="right"))
    assert dataset.sample_id(0) == "train-00000-of-00001.tfrecord:1"
    assert dataset.sample_id(1) == "train-00000-of-00001.tfrecord:0"
    assert first["length"].tolist() == [1]
    assert first["type"].squeeze(-1).tolist() == [2, 1]
    assert first["left"].squeeze(-1).tolist() == [expected_left, 63]
    assert first["icon"].squeeze(-1).tolist() == [2, 2]
    assert first["text_button"].squeeze(-1).tolist() == [2, 2]
    assert second["type"].squeeze(-1).tolist() == [1]
    assert index_reads == 1

    replay = FlexDmTfrecordDataset(
        tmp_path,
        dataset_name="rico",
        split="train",
        sample_order=(1, 0, 1),
    )
    assert len(replay) == 3
    assert [replay.sample_id(index) for index in range(len(replay))] == [
        "train-00000-of-00001.tfrecord:1",
        "train-00000-of-00001.tfrecord:0",
        "train-00000-of-00001.tfrecord:1",
    ]

    batch = dataset.collate_fn([second, first])
    assert batch["length"].squeeze(-1).tolist() == [0, 1]
    assert batch["type"].squeeze(-1).tolist() == [[1, 0], [2, 1]]
    assert batch["icon"].squeeze(-1).tolist() == [[2, 1], [2, 2]]
    assert batch["text_button"].squeeze(-1).tolist() == [[2, 1], [2, 2]]
    assert batch["left"].shape == (2, 2, 1)
    assert index_reads == 2
