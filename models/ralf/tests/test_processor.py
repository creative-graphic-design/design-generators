from typing import cast

import torch

from laygen.common.conditions import ConditionType
from ralf import RalfConfig, RalfProcessor


def test_processor_normalizes_aliases_and_box_formats() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=2))

    encoded = processor(
        condition_type="cwh",
        labels=[["logo"]],
        bbox=[[[0.0, 0.0, 0.2, 0.2]]],
        box_format="ltwh",
    )

    assert encoded["condition_type"] is ConditionType.label_size
    assert torch.allclose(
        encoded["constraint_bbox"][0, 0], torch.tensor([0.1, 0.1, 0.2, 0.2])
    )
    assert encoded["constraint_labels"].tolist() == [[0]]


def test_processor_explicit_retrieval_container_wins() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=2, top_k=2))
    retrieval = {
        "items": {
            "bbox": torch.zeros(1, 2, 2, 4),
            "labels": torch.zeros(1, 2, 2, dtype=torch.long),
            "mask": torch.ones(1, 2, 2, dtype=torch.bool),
        },
        "ids": [[3, 4]],
    }

    encoded = processor(condition_type="unconditional", retrieval=retrieval)

    assert encoded["condition_type"] is ConditionType.unconditional
    assert encoded["retrieval"].indexes.tolist() == [[3, 4]]


def test_processor_mask_and_output_dict_paths() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=2))
    encoded = processor(
        labels=torch.tensor([0, 1]),
        bbox=torch.zeros(2, 4),
        mask=torch.tensor([True, False]),
    )

    output = processor.post_process_layouts(
        encoded["input_ids"],
        output_type="dict",
        intermediates={"retrieval": {"indexes": torch.tensor([[1]])}},
    )
    output = cast(dict[str, object], output)
    mask_out = torch.as_tensor(output["mask"])
    intermediates = cast(dict[str, object], output["intermediates"])

    assert mask_out.tolist() == [[True, False]]
    assert "retrieval" in intermediates


def test_processor_requires_canvas_for_pixel_boxes() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))

    try:
        processor(labels=[0], bbox=[[[0, 0, 10, 10]]], normalized=False)
    except ValueError as exc:
        assert "canvas_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
