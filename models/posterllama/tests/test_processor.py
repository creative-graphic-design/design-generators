from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from laygen.common.conditions import ConditionType
from posterllama import PosterLlamaConfig, PosterLlamaProcessor


def test_vendor_condition_aliases_build_prompt() -> None:
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(canvas_size=(100, 200))
    )

    prompt = cast(
        str,
        processor.build_prompt(
            condition_type="cond_cate_size_to_pos",
            labels=["text"],
            bbox=[[[0.5, 0.5, 0.2, 0.4]]],
            canvas_size=(100, 200),
        ),
    )

    assert "categories and size and image" in prompt
    assert 'data-category="text"' in prompt
    assert 'x="<FILL_1>"' in prompt
    assert 'width="20"' in prompt
    assert prompt.endswith(" <MID>")


@pytest.mark.parametrize(
    ("alias", "condition"),
    [
        ("cond_cate_to_size_pos", ConditionType.label),
        ("cond_recover_mask", ConditionType.completion),
        ("cond_cate_pos_to_size", ConditionType.refinement),
    ],
)
def test_normalize_vendor_condition_aliases(
    alias: str,
    condition: ConditionType,
) -> None:
    processor = PosterLlamaProcessor.from_config(PosterLlamaConfig())

    assert processor.normalize_condition_type(alias) is condition


def test_unsupported_text_condition_raises() -> None:
    processor = PosterLlamaProcessor.from_config(PosterLlamaConfig())

    with pytest.raises(NotImplementedError, match="text"):
        processor.build_prompt(condition_type="text")


def test_parse_output_dict_includes_intermediates() -> None:
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(canvas_size=(100, 100))
    )
    output = processor.parse_output(
        '<svg width="100" height="100"><rect data-category="text" x="0" y="0" width="10" height="10"/></svg>',
        output_type="dict",
        return_intermediates=True,
    )

    labels = cast(torch.Tensor, output["labels"])
    intermediates = cast(dict[str, object], output["intermediates"])
    assert labels.tolist() == [[1]]
    assert intermediates["canvas_size"] == (100, 100)


def test_processor_call_and_prompt_variants() -> None:
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(canvas_size=(100, 100))
    )

    batch = processor(
        images=None,
        prompt=["custom"],
        texts=[["headline", "body"]],
        labels=torch.tensor([1]),
        bbox=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        condition_type="cond_cate_pos_to_size",
        batch_size=1,
    )

    assert batch["condition_type"] == ConditionType.refinement
    assert "Text: headline | body" in batch["prompts"][0]
    assert 'width="<FILL_1>"' in batch["prompts"][0]


def test_parse_output_requires_canvas_when_missing() -> None:
    processor = PosterLlamaProcessor.from_config(PosterLlamaConfig(canvas_size=None))

    with pytest.raises(ValueError, match="canvas_size is required"):
        processor.parse_output(
            '<rect data-category="text" x="0" y="0" width="1" height="1"/>'
        )


def test_invalid_output_type_raises() -> None:
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(canvas_size=(100, 100))
    )

    with pytest.raises(ValueError, match="Unsupported output_type"):
        processor.parse_output(
            '<svg width="100" height="100"><rect data-category="text" x="0" y="0" width="1" height="1"/></svg>',
            output_type=cast(Literal["dataclass", "dict"], "tuple"),
        )


def test_num_elements_sequence_and_tensor() -> None:
    processor = PosterLlamaProcessor.from_config(PosterLlamaConfig())

    seq_prompt = processor.build_prompt(
        condition_type="unconditional", num_elements=[2]
    )
    tensor_prompt = processor.build_prompt(
        condition_type="content_image",
        num_elements=torch.tensor([3]),
    )

    assert seq_prompt.count("<FILL_") == 0
    assert tensor_prompt.count("<FILL_") == 0


def test_prompt_helpers_cover_default_and_sequence_paths() -> None:
    processor = PosterLlamaProcessor.from_config(PosterLlamaConfig(canvas_size=None))

    assert processor._resolve_canvas_size(None) == (360, 504)
    assert processor._first_prompt(None) == ""
    assert processor._first_prompt(("first", "second")) == "first"
    assert processor._texts_line(None) == ""
    assert processor._texts_line("single") == "single"
    assert processor._texts_line(["title", "body"]) == "title | body"
    assert processor._texts_line([["1", "body"]]) == "1 | body"
    assert processor._num_elements(None) == 1
    assert processor._num_elements((4, 5)) == 4


def test_constraint_markup_fill_and_condition_variants() -> None:
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(canvas_size=(100, 100), id2label={0: "text"})
    )

    label_only = processor._constraint_markup(
        condition=ConditionType.label,
        labels=["text"],
        bbox=[[[0.1, 0.2, 0.3, 0.4]]],
        mask=None,
        num_elements=None,
        box_format="xywh",
        normalized=True,
        canvas_size=(100, 100),
    )
    assert 'data-category="text"' in label_only
    assert 'x="<FILL_1>"' in label_only

    label_size = processor._constraint_markup(
        condition=ConditionType.label_size,
        labels=[0],
        bbox=[[[0.1, 0.2, 0.3, 0.4]]],
        mask=None,
        num_elements=None,
        box_format="xywh",
        normalized=True,
        canvas_size=(100, 100),
    )
    assert 'x="<FILL_1>"' in label_size
    assert 'width="30"' in label_size or 'width="29"' in label_size

    assert (
        processor._constraint_markup(
            condition=ConditionType.completion,
            labels=None,
            bbox=None,
            mask=None,
            num_elements=[2],
            box_format="xywh",
            normalized=True,
            canvas_size=(100, 100),
        ).count("<FILL_")
        == 10
    )
