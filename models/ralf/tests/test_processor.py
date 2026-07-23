from pathlib import Path
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

    pku_typo = processor(condition_type="chw")
    assert pku_typo["condition_type"] is ConditionType.label_size


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


def test_processor_save_load_helpers_and_empty_inputs(tmp_path: Path) -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))
    processor.save_pretrained(tmp_path)

    loaded = RalfProcessor.from_pretrained(tmp_path)
    image_processor = RalfProcessor._load_image_processor_from_pretrained(
        "RalfImageProcessor",
        tmp_path,
    )
    tokenizer = RalfProcessor._load_layout_tokenizer_from_pretrained(
        "RalfLayoutTokenizer",
        tmp_path,
        local_files_only=True,
    )

    assert loaded.config.max_seq_length == 1
    assert image_processor.image_size == processor.image_processor.image_size
    assert tokenizer.config.max_seq_length == 1
    assert processor(labels=[], batch_size=2)["constraint_labels"].shape == (2, 0)
    assert processor.normalize_condition_type("uncond") is ConditionType.unconditional


def test_processor_ltrb_and_tensor_label_paths() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))

    encoded = processor(
        labels=torch.tensor([1]),
        bbox=torch.tensor([[0.1, 0.2, 0.5, 0.6]]),
        box_format="ltrb",
    )

    assert encoded["constraint_labels"].tolist() == [[1]]
    assert torch.allclose(
        encoded["constraint_bbox"][0, 0],
        torch.tensor([0.3, 0.4, 0.4, 0.4]),
    )


def test_processor_normalizes_pixel_boxes_with_canvas_size() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))

    encoded = processor(
        labels=[0],
        bbox=[[[0, 0, 50, 100]]],
        normalized=False,
        canvas_size=(100, 200),
        box_format="ltwh",
    )

    assert torch.allclose(
        encoded["constraint_bbox"][0, 0],
        torch.tensor([0.25, 0.25, 0.5, 0.5]),
    )


def test_processor_requires_canvas_for_pixel_boxes() -> None:
    processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))

    try:
        processor(labels=[0], bbox=[[[0, 0, 10, 10]]], normalized=False)
    except ValueError as exc:
        assert "canvas_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
