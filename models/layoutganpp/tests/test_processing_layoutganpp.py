from typing import Literal, cast

import torch
import pytest

from layoutganpp import (
    DatasetName,
    MAGAZINE_LABELS,
    PUBLAYNET_LABELS,
    RICO_LABELS,
    LayoutGANPPProcessor,
    label2id_for_dataset,
    labels_for_dataset,
)


def test_dataset_vocabularies_match_vendor_order():
    assert labels_for_dataset("rico") == RICO_LABELS
    assert labels_for_dataset("publaynet") == PUBLAYNET_LABELS
    assert labels_for_dataset("magazine") == MAGAZINE_LABELS
    assert labels_for_dataset(DatasetName.rico13)[0] == "Toolbar"
    assert label2id_for_dataset("rico")["Toolbar"] == 0
    with pytest.raises(ValueError, match="Unknown"):
        labels_for_dataset("missing")


def test_processor_encodes_strings_and_padding():
    processor = LayoutGANPPProcessor("publaynet")
    encoded = processor([["text", "figure"], ["title"]])
    assert torch.equal(encoded["labels"], torch.tensor([[0, 4], [1, 0]]))
    assert torch.equal(
        encoded["attention_mask"], torch.tensor([[True, True], [True, False]])
    )


def test_processor_decode_records():
    processor = LayoutGANPPProcessor("publaynet")
    records = processor.batch_decode(
        bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]]),
        labels=torch.tensor([[0, 1]]),
        attention_mask=torch.tensor([[True, False]]),
    )
    assert records[0][0]["label"] == "text"
    assert records[0][0]["label_id"] == 0
    torch.testing.assert_close(
        torch.tensor(records[0][0]["bbox"]),
        torch.tensor([0.5, 0.5, 0.2, 0.2]),
    )
    unbatched = processor.batch_decode(
        bbox=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        labels=torch.tensor([0]),
    )
    assert unbatched[0][0]["label"] == "text"


def test_processor_save_load_roundtrip(tmp_path):
    processor = LayoutGANPPProcessor("magazine")
    processor.save_pretrained(str(tmp_path))
    loaded = LayoutGANPPProcessor.from_pretrained(str(tmp_path))
    assert loaded.id2label == processor.id2label


def test_processor_error_branches_and_tensor_rows():
    processor = LayoutGANPPProcessor("publaynet")
    encoded = processor(torch.tensor([0, 1]))
    assert encoded["labels"].tolist() == [[0, 1]]
    batched = processor(torch.tensor([[0, 1], [2, 3]]))
    assert batched["attention_mask"].tolist() == [[True, True], [True, True]]
    with pytest.raises(ValueError, match="return_tensors"):
        processor(["text"], return_tensors=cast(Literal["pt"], "np"))
    with pytest.raises(ValueError, match="Ragged"):
        processor([["text"], ["title", "figure"]], padding=False)
    with pytest.raises(ValueError, match="empty"):
        processor([])
    with pytest.raises(ValueError, match="Unknown label id"):
        processor([99])
    with pytest.raises(ValueError, match="Unknown label"):
        processor(["missing"])


def test_processor_for_dataset():
    processor = LayoutGANPPProcessor(DatasetName.magazine)
    assert processor.dataset_name == "magazine"
