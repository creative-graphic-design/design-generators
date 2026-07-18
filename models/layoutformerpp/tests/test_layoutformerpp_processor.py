from typing import get_args, get_type_hints, cast

import torch
import pytest

from laygen.common.bbox import BoxFormat
from laygen.common.labels import DatasetName

from layoutformerpp import (
    ConditionType,
    LayoutFormerPPProcessor,
    LayoutFormerPPTask,
    LayoutGenerationOutput,
    OutputType,
)


def test_processor_label_condition_and_postprocess() -> None:
    processor = LayoutFormerPPProcessor.from_config(
        dataset=DatasetName.rico25, task=LayoutFormerPPTask.gen_t
    )
    batch = processor(
        condition_type="label_size",
        labels=[["Text"]],
        bbox=torch.tensor([[[0.5, 0.5, 0.25, 0.25]]]),
    )
    assert batch["input_ids"].shape[0] == 1
    ids = processor.tokenizer.encode_text("label_1 0 0 10 10 |")["input_ids"]
    out = processor.post_process_layouts(ids)
    assert isinstance(out, LayoutGenerationOutput)
    assert out.labels.tolist() == [[0]]
    assert out.mask.tolist() == [[True]]
    assert out.id2label[0] == "Text"


def test_processor_masks_padded_condition_rows_and_normalizes_pixels() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="completion")

    masked = processor(
        condition_type="completion",
        labels=[["Text", "Image"]],
        bbox=[[[0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]],
        mask=torch.tensor([[True, False]]),
    )
    decoded = processor.tokenizer.batch_decode(
        masked["input_ids"], skip_special_tokens=True
    )

    assert decoded == ["label_1 0 0 25 25"]

    pixel = processor(
        condition_type="label_size",
        labels=[["Text"]],
        bbox=torch.tensor([[[50.0, 50.0, 20.0, 20.0]]]),
        normalized=False,
        canvas_size=(100, 100),
    )
    decoded_pixel = processor.tokenizer.batch_decode(
        pixel["input_ids"], skip_special_tokens=True
    )

    assert decoded_pixel == ["label_1 25 25"]

    with pytest.raises(ValueError, match="canvas_size is required"):
        processor(
            condition_type="label_size",
            labels=[["Text"]],
            bbox=torch.tensor([[[50.0, 50.0, 20.0, 20.0]]]),
            normalized=False,
        )


def test_processor_condition_aliases_and_error_paths() -> None:
    processor = LayoutFormerPPProcessor.from_config(
        dataset="rico", task=ConditionType.relation
    )

    assert processor.normalize_condition_type("gen_t") is ConditionType.label
    assert (
        processor.normalize_condition_type(ConditionType.relation)
        is ConditionType.relation
    )
    with pytest.raises(ValueError, match="Unsupported condition_type"):
        processor.normalize_condition_type("bad")
    with pytest.raises(ValueError, match="Unknown label"):
        processor(condition_type="label", labels=[["missing"]])
    call_hints = get_type_hints(processor.__call__)
    assert get_args(call_hints["return_tensors"]) == ("pt",)

    relation = processor(
        condition_type=ConditionType.relation,
        labels=[["Text", "Image"]],
        relations=[[(2, 1, 1, 0, 3)]],
    )
    assert relation["input_ids"].shape[0] == 1
    assert processor.dataset == "rico"
    assert processor.task == "gen_r"


def test_processor_postprocess_padding_dict_and_errors() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
    sequences = processor.tokenizer.encode_text(
        ["label_1 0 0 10 10 | label_2 1 1 2 2 |", ""],
        add_eos=True,
    )["input_ids"]

    out = processor.post_process_layouts(
        sequences, box_format=BoxFormat.ltwh, output_type=OutputType.dict
    )
    assert isinstance(out, dict)
    labels = out["labels"]
    mask = out["mask"]
    intermediates = cast(dict[str, object], out["intermediates"])
    assert isinstance(labels, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(intermediates, dict)
    assert labels.shape == (2, 2)
    assert mask.tolist() == [[True, True], [False, False]]
    assert intermediates["box_format"] is BoxFormat.ltwh

    with pytest.raises(ValueError, match="Unsupported output_type"):
        processor.post_process_layouts(sequences, output_type="bad")
    postprocess_hints = get_type_hints(processor.post_process_layouts)
    assert get_args(postprocess_hints["return_tensors"]) == ("pt",)
