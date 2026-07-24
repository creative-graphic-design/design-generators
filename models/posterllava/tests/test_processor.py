from __future__ import annotations

import torch
from typing import cast
from transformers import PreTrainedTokenizerBase

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from posterllava.processing_posterllava import PosterLlavaProcessor


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [ord(char) % 23 + 1 for char in text]
        if add_special_tokens:
            ids = [1, *ids]
        return {"input_ids": ids}


def test_build_prompt_matches_llava_v0_image_token_contract() -> None:
    processor = PosterLlavaProcessor.from_config()

    prompt = processor.build_prompt(num_elements=2, conv_mode="llava_v0")

    assert prompt.startswith("A chat between a curious human")
    assert "###Human: <image>\n" in prompt
    assert prompt.endswith("###Assistant:")
    assert "Generate 2 layout elements" in prompt


def test_build_prompt_supports_llava_v1_wrapper() -> None:
    processor = PosterLlavaProcessor.from_config()

    prompt = processor.build_prompt(num_elements=1, conv_mode="llava_v1")

    assert prompt.startswith("USER: <image>\n")
    assert prompt.endswith("\nASSISTANT:")


def test_parse_output_uses_first_json_span_and_single_quote_repair() -> None:
    processor = PosterLlavaProcessor.from_config()

    parsed = processor.parse_output(
        "prefix [{'label': 'text', 'box': [0.1, 0.2, 0.3, 0.4]}] suffix"
    )

    assert parsed == [{"label": "text", "box": [0.1, 0.2, 0.3, 0.4]}]


def test_parse_output_rejects_invalid_shapes() -> None:
    processor = PosterLlavaProcessor.from_config()

    for text in (
        "no json",
        "{}",
        "[1]",
        "[{'label': 1, 'box': [0, 0, 1, 1]}]",
        "[{'label': 'x', 'box': [0]}]",
    ):
        try:
            processor.parse_output(text)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {text}")


def test_decode_layout_returns_batch_local_open_vocab_schema() -> None:
    processor = PosterLlavaProcessor.from_config()

    output = cast(
        LayoutGenerationOutput,
        processor.decode_layout(
            [
                "[{'label': 'headline', 'box': [0.0, 0.0, 0.4, 0.4]}]",
                "[{'label': 'logo', 'box': [0.5, 0.5, 1.0, 1.0]}]",
            ],
            return_intermediates=True,
        ),
    )

    assert_layout_output_schema(output, batch_size=2)
    assert output.id2label == {0: "headline", 1: "logo"}
    assert torch.allclose(output.bbox[0, 0], torch.tensor([0.2, 0.2, 0.4, 0.4]))
    assert output.intermediates is not None

    as_dict = processor.decode_layout(
        "[{'label': 'empty', 'box': [0, 0, 0, 0]}]",
        output_type="dict",
    )
    assert as_dict["id2label"] == {0: "empty"}


def test_decode_layout_handles_empty_layout() -> None:
    processor = PosterLlavaProcessor.from_config()

    output = cast(LayoutGenerationOutput, processor.decode_layout("[]"))

    assert_layout_output_schema(output, batch_size=1)
    assert not output.mask.any()


def test_processor_save_load_round_trip(tmp_path) -> None:
    processor = PosterLlavaProcessor.from_config(
        dataset_name="ad_banner",
        id2label={0: "text"},
        default_domain_name="poster",
    )

    processor.save_pretrained(tmp_path)
    loaded = PosterLlavaProcessor.from_pretrained(tmp_path)

    assert loaded.id2label == {0: "text"}
    assert loaded.default_domain_name == "poster"

    (tmp_path / "processor_config.json").write_text('{"canvas_size": [1]}')
    try:
        PosterLlavaProcessor.from_pretrained(tmp_path)
    except ValueError as exc:
        assert "canvas_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_initial_json_and_tokenization_paths() -> None:
    processor = PosterLlavaProcessor(
        tokenizer=cast(PreTrainedTokenizerBase, TinyTokenizer()),
    )

    labels_only = processor.build_initial_json(labels=["headline"])
    assert labels_only == [{"label": "headline", "box": []}]

    with_boxes = processor.build_initial_json(
        labels=torch.tensor([0, 1]),
        bbox=torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1]]),
        mask=torch.tensor([True, False]),
    )
    assert with_boxes[0]["label"] == "header"
    assert len(with_boxes) == 1

    encoded = processor(["<image>\na", "<image>\nb"])
    assert encoded["input_ids"].shape[0] == 2


def test_build_prompt_rejects_non_positive_count() -> None:
    processor = PosterLlavaProcessor.from_config()

    try:
        processor.build_prompt(num_elements=0)
    except ValueError as exc:
        assert "num_elements" in str(exc)
    else:
        raise AssertionError("expected ValueError")
