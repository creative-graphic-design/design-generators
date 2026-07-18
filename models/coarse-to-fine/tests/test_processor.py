import tempfile
from typing import cast

import pytest
import torch

from coarse_to_fine import CoarseToFineProcessor


def test_processor_label_maps_and_encoding_round_trip():
    processor = CoarseToFineProcessor(dataset="publaynet")

    encoded = processor(
        labels=[["text", "figure"]],
        bbox=[[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]],
    )

    assert processor.id2label[0] == "text"
    assert encoded["labels"].tolist() == [[1, 5]]
    assert encoded["bbox"].shape == (1, 2, 4)
    assert encoded["mask"].dtype is torch.bool


def test_processor_builds_hierarchy_batch():
    processor = CoarseToFineProcessor(dataset="publaynet")
    labels = cast(torch.LongTensor, torch.tensor([[0, 1, 0]]))
    bbox = cast(
        torch.FloatTensor,
        torch.tensor(
            [
                [
                    [0.1, 0.1, 0.1, 0.1],
                    [0.5, 0.1, 0.1, 0.1],
                    [0.1, 0.6, 0.1, 0.1],
                ]
            ]
        ),
    )
    mask = cast(torch.BoolTensor, torch.ones((1, 3), dtype=torch.bool))

    out = processor.build_hierarchy_batch(labels, bbox, mask)

    assert out["group_bounding_box"].dim() == 3
    assert out["label_in_one_group"].shape[-1] == len(processor.id2label) + 2
    assert out["grouped_bbox"].dim() == 4


def test_processor_save_and_load_pretrained():
    processor = CoarseToFineProcessor(dataset="publaynet")

    with tempfile.TemporaryDirectory() as tmp:
        processor.save_pretrained(tmp)
        loaded = CoarseToFineProcessor.from_pretrained(tmp)

    assert loaded.dataset == "publaynet"
    assert loaded.id2label == processor.id2label


def test_processor_rejects_missing_or_pixel_inputs():
    processor = CoarseToFineProcessor(dataset="publaynet")

    with pytest.raises(ValueError, match="labels and bbox"):
        processor(labels=[[0]], bbox=None)
    with pytest.raises(ValueError, match="normalized"):
        processor(labels=[[0]], bbox=[[[1, 2, 3, 4]]], normalized=False)
