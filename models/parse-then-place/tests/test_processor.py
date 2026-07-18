from __future__ import annotations

from typing import cast

import pytest
import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from parse_then_place import ParseThenPlaceProcessor


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
