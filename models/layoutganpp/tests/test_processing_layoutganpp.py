import torch

from layoutganpp import (
    MAGAZINE_LABELS,
    PUBLAYNET_LABELS,
    RICO_LABELS,
    LayoutGANPPProcessor,
    labels_for_dataset,
)


def test_dataset_vocabularies_match_vendor_order():
    assert labels_for_dataset("rico") == RICO_LABELS
    assert labels_for_dataset("publaynet") == PUBLAYNET_LABELS
    assert labels_for_dataset("magazine") == MAGAZINE_LABELS


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


def test_processor_save_load_roundtrip(tmp_path):
    processor = LayoutGANPPProcessor("magazine")
    processor.save_pretrained(str(tmp_path))
    loaded = LayoutGANPPProcessor.from_pretrained(str(tmp_path))
    assert loaded.id2label == processor.id2label
