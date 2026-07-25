import pytest
import torch
from typing import Literal, cast

from layoutvae import LayoutVAEProcessor


def test_processor_encodes_public_labels():
    processor = LayoutVAEProcessor()
    encoded = processor([["text", "figure"], ["title"]])
    assert encoded["label_set"].tolist() == [
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]


def test_processor_maps_internal_empty_to_mask():
    processor = LayoutVAEProcessor()
    labels, mask = processor.public_from_internal(torch.tensor([[0, 1, 5]]))
    assert labels.tolist() == [[0, 0, 4]]
    assert mask.tolist() == [[False, True, True]]


def test_processor_decodes_and_rejects_unknown_label():
    processor = LayoutVAEProcessor()
    records = processor.batch_decode(torch.zeros(1, 1, 4), torch.tensor([[0]]))
    assert records[0][0]["label"] == "text"
    with pytest.raises(ValueError, match="Unknown label"):
        processor(["unknown"])


def test_processor_tensor_rows_masks_and_errors():
    processor = LayoutVAEProcessor()
    encoded = processor(torch.tensor([[0, 4]]))
    assert encoded["label_set"].shape == (1, 6)
    records = processor.batch_decode(
        torch.zeros(1, 2, 4),
        torch.tensor([[0, 4]]),
        torch.tensor([True, False]),
    )
    assert len(records[0]) == 1
    with pytest.raises(ValueError, match="return_tensors"):
        processor(["text"], return_tensors=cast(Literal["pt"], "np"))
    with pytest.raises(ValueError, match="one or two"):
        processor(torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="must not be empty"):
        processor([])
    with pytest.raises(ValueError, match="flat list"):
        processor(cast(list[list[str | int]], [["text"], 1]))
    with pytest.raises(ValueError, match="flat list"):
        processor(cast(list[str | int], ["text", ["figure"]]))
    with pytest.raises(ValueError, match="Unknown label id"):
        processor([99])
    with pytest.raises(ValueError, match="PubLayNet"):
        LayoutVAEProcessor(dataset_name="rico25")
