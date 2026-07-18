from __future__ import annotations

from typing import Literal, cast

import pytest
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from parse_then_place import ParseThenPlaceProcessor


class FakeTokenizer(PreTrainedTokenizerBase):
    model_input_names = ["input_ids", "attention_mask"]

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> BatchEncoding:
        _ = (return_tensors, padding)
        batch = [texts] if isinstance(texts, str) else texts
        input_ids = torch.arange(len(batch) * 2, dtype=torch.long).reshape(
            len(batch), 2
        )
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )

    def batch_decode(  # ty: ignore[invalid-method-override]
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs: object,
    ) -> list[str]:
        _ = (skip_special_tokens, clean_up_tokenization_spaces, kwargs)
        return ["TEXT 0 0 10 20" for _ in range(len(sequences))]


def test_rico_layout_text_to_output_normalizes_ltwh_to_center_xywh() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    output = processor.layout_text_to_output(["text 0 0 10 20"])

    output = cast(LayoutGenerationOutput, output)
    assert_layout_output_schema(output, batch_size=1)
    expected = torch.tensor([[[5 / 144, 10 / 256, 10 / 144, 20 / 256]]])
    assert torch.allclose(output.bbox, expected)
    assert output.labels.tolist() == [[4]]
    assert output.mask.tolist() == [[True]]
    assert output.id2label[4] == "text"


def test_web_textarea_label_uses_vendor_id_order() -> None:
    processor = ParseThenPlaceProcessor.from_config("web")

    output = processor.layout_text_to_output(["textarea 0 0 10 20"])

    output = cast(LayoutGenerationOutput, output)
    assert output.labels.tolist() == [[11]]
    assert output.id2label[11] == "textarea"


def test_unknown_generated_labels_are_skipped() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    output = processor.layout_text_to_output(
        ["unknown 0 0 10 20"],
        return_intermediates=True,
    )

    output = cast(LayoutGenerationOutput, output)
    assert output.mask.tolist() == [[False]]
    assert isinstance(output.intermediates, dict)
    intermediates = cast(dict[str, object], output.intermediates)
    assert intermediates["layout_text"] == ["unknown 0 0 10 20"]


def test_prompt_preprocessing_matches_vendor_text_rules() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    encoded = processor.preprocess_prompt(
        ' #Create “Title” and Bob’s "CTA"  ',
        replace_explicit_value=True,
    )

    assert encoded["prompt"] == 'create "value_0" and bob\'s "value_1"'
    assert encoded["value_map"] == {"value_0": "title", "value_1": "cta"}


def test_layout_decode_validates_candidate_count() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    with pytest.raises(ValueError, match="batch_size"):
        processor.decode_layout_sequences(
            ["text 0 0 10 20"],
            batch_size=2,
            num_return_sequences=1,
        )


def test_processor_tokenizer_paths_and_ir_recovery() -> None:
    tokenizer = FakeTokenizer()
    processor = ParseThenPlaceProcessor(
        parser_tokenizer=tokenizer,
        placement_tokenizer=tokenizer,
        id2label={4: "text"},
    )

    encoded = processor(["Create 'A, B'.", "Create Text"])
    assert encoded["input_ids"].shape == (2, 2)
    assert encoded["prompt_text"] == ["create 'a, b'.", "create text"]
    assert encoded["value_maps"] == [{}, {}]

    logical_forms = processor.postprocess_ir(
        torch.tensor([[1, 2]]),
        value_maps=[{"value_0": "CTA"}],
    )
    assert logical_forms == ["text 0 0 10 20"]

    placement = processor.encode_placement_inputs(["singleinfo : text | image"])
    assert placement["input_ids"].shape == (1, 2)
    assert placement["placement_text"] == ["singleinfo : text | image"]

    decoded = processor.decode_layout_sequences(
        torch.tensor([[1, 2], [3, 4]]),
        batch_size=1,
        num_return_sequences=2,
    )
    assert decoded == [["TEXT 0 0 10 20", "TEXT 0 0 10 20"]]


def test_processor_candidate_selection_and_output_types() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    best = processor.layout_text_to_output(
        [["unknown 0 0 1 1", "text 0 0 10 20 image 10 20 30 40"]],
        output_candidate="best",
        output_type="dict",
        return_intermediates=True,
    )

    assert isinstance(best, dict)
    assert cast(torch.Tensor, best["mask"]).tolist() == [[True, True]]
    assert cast(dict[str, object], best["intermediates"])["layout_text"] == [
        "text 0 0 10 20 image 10 20 30 40"
    ]

    all_candidates = processor.layout_text_to_output(
        [["text 0 0 10 20", "image 10 20 30 40"]],
        output_candidate="all",
    )
    all_candidates = cast(LayoutGenerationOutput, all_candidates)
    assert all_candidates.labels.tolist() == [[4, 10]]

    with pytest.raises(ValueError, match="Unsupported output_candidate"):
        processor.layout_text_to_output(
            ["text 0 0 10 20"],
            output_candidate=cast(Literal["first", "all", "best"], "bad"),
        )

    with pytest.raises(ValueError, match="Unsupported output_type"):
        processor.layout_text_to_output(
            ["text 0 0 10 20"],
            output_type=cast(Literal["dataclass", "dict"], "tuple"),
        )


def test_processor_requires_tokenizers_for_tensor_decoding() -> None:
    processor = ParseThenPlaceProcessor.from_config("rico")

    with pytest.raises(ValueError, match="parser_tokenizer"):
        processor.postprocess_ir(torch.tensor([[1, 2]]))

    with pytest.raises(ValueError, match="placement_tokenizer"):
        processor.decode_layout_sequences(
            torch.tensor([[1, 2]]),
            batch_size=1,
            num_return_sequences=1,
        )
