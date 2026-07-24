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

    prompt = processor.build_prompt(
        condition_type="cond_cate_size_to_pos",
        labels=["text"],
        bbox=[[[0.5, 0.5, 0.2, 0.4]]],
        canvas_size=(100, 200),
    )

    assert "Condition: label_size" in prompt
    assert 'data-category="text"' in prompt
    assert 'x="<FILL_0>"' in prompt
    assert 'width="20.0"' in prompt


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
    assert "Texts: headline | body" in batch["prompts"][0]
    assert 'width="<FILL_0>"' in batch["prompts"][0]


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

    assert seq_prompt.count("<FILL_") == 10
    assert tensor_prompt.count("<FILL_") == 15
