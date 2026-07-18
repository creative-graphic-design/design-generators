from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from transformers import (
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from parse_then_place import (
    ParseThenPlaceConfig,
    ParseThenPlacePipeline,
    ParseThenPlaceProcessor,
)


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


class FakeGenerationModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, generated: torch.Tensor) -> None:
        super().__init__(PretrainedConfig())
        self.placeholder = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.generated = generated
        self.generate_kwargs: dict[str, object] | None = None
        self.saved_kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return self.generated

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        is_main_process: bool = True,
        **kwargs: object,
    ) -> None:
        self.saved_kwargs = {"is_main_process": is_main_process, **kwargs}
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}")
        (root / "fake_model.txt").write_text("saved")


def _pipeline() -> ParseThenPlacePipeline:
    return ParseThenPlacePipeline(
        config=ParseThenPlaceConfig(dataset_name="rico"),
        processor=ParseThenPlaceProcessor.from_config("rico"),
    )


def test_pipeline_accepts_text_aliases_with_layout_text_smoke() -> None:
    pipe = _pipeline()

    output = pipe(
        prompt="create a simple screen",
        condition_type="prompt",
        layout_text="text 0 0 10 20",
    )

    output = cast(LayoutGenerationOutput, output)
    assert_layout_output_schema(output, batch_size=1)


def test_pipeline_rejects_non_text_conditions_explicitly() -> None:
    pipe = _pipeline()

    with pytest.raises(NotImplementedError, match="condition_type='text'"):
        pipe(
            prompt="create a simple screen",
            condition_type="unconditional",
            layout_text="text 0 0 10 20",
        )


def test_pipeline_requires_prompt_without_layout_text_shortcut() -> None:
    pipe = _pipeline()

    with pytest.raises(ValueError, match="prompt is required"):
        pipe(condition_type="text")


def test_generator_prevents_seed_global_set_seed() -> None:
    pipe = _pipeline()

    with patch("parse_then_place.pipeline_parse_then_place.set_seed") as mocked:
        pipe(
            prompt="create a simple screen",
            seed=123,
            generator=torch.Generator().manual_seed(123),
            layout_text="text 0 0 10 20",
        )

    mocked.assert_not_called()


def test_pipeline_parse_place_and_save_stages() -> None:
    parser = FakeGenerationModel(torch.tensor([[1, 2]]))
    placement = FakeGenerationModel(torch.tensor([[3, 4]]))
    pipe = ParseThenPlacePipeline(
        config=ParseThenPlaceConfig(dataset_name="rico"),
        processor=ParseThenPlaceProcessor.from_config("rico"),
        parser=parser,
        placement=placement,
    )
    generator = torch.Generator().manual_seed(7)

    parsed = pipe.parse(torch.tensor([[0, 1]]), generation_max_length=12)
    placed = pipe.place(
        torch.tensor([[0, 1]]),
        generation_max_length=13,
        num_return_sequences=2,
        temperature=0.2,
        generator=generator,
    )

    assert parsed.tolist() == [[1, 2]]
    assert placed.tolist() == [[3, 4]]
    assert parser.generate_kwargs is not None
    assert parser.generate_kwargs["max_length"] == 12
    assert placement.generate_kwargs is not None
    assert placement.generate_kwargs["generator"] is generator

    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(pipe.processor, "save_pretrained") as save_processor:
            pipe.save_pretrained(tmp, is_main_process=False)
            save_processor.assert_called_once_with(Path(tmp))
            output_dir = Path(tmp)
            assert (output_dir / "config.json").exists()
            assert (output_dir / "semantic_parser" / "fake_model.txt").exists()
            assert (output_dir / "placement" / "fake_model.txt").exists()

    assert parser.saved_kwargs == {"is_main_process": False}
    assert placement.saved_kwargs == {"is_main_process": False}


def test_pipeline_parse_and_place_require_stages() -> None:
    pipe = _pipeline()

    with pytest.raises(ValueError, match="Parser stage"):
        pipe.parse(torch.tensor([[0, 1]]))

    with pytest.raises(ValueError, match="Placement stage"):
        pipe.place(torch.tensor([[0, 1]]))


def test_pipeline_from_pretrained_loads_standard_stage_subfolders() -> None:
    config = ParseThenPlaceConfig(dataset_name="web")
    parser = FakeGenerationModel(torch.tensor([[1, 2]]))
    placement = FakeGenerationModel(torch.tensor([[3, 4]]))
    processor = ParseThenPlaceProcessor.from_config("web")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config.save_pretrained(root)
        (root / "semantic_parser").mkdir()
        (root / "placement").mkdir()
        (root / "semantic_parser" / "config.json").write_text("{}")
        with patch(
            "parse_then_place.pipeline_parse_then_place.AutoModelForSeq2SeqLM.from_pretrained",
            side_effect=[parser, placement],
        ) as load_model:
            loaded = ParseThenPlacePipeline.from_pretrained(
                root,
                config=PretrainedConfig.from_dict(config.to_dict()),
                processor=processor,
                local_files_only=True,
            )

    assert loaded.parser is parser
    assert loaded.placement is placement
    assert loaded.processor is processor
    assert load_model.call_count == 2

    with pytest.raises(TypeError, match="config must be"):
        ParseThenPlacePipeline.from_pretrained(
            Path("missing"),
            config=object(),  # ty: ignore[invalid-argument-type]
        )


def test_pipeline_generate_full_path_and_dict_output() -> None:
    tokenizer = FakeTokenizer()
    parser = FakeGenerationModel(torch.tensor([[1, 2], [3, 4]]))
    placement = FakeGenerationModel(torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]))
    pipe = ParseThenPlacePipeline(
        config=ParseThenPlaceConfig(dataset_name="rico", num_return_sequences=2),
        processor=ParseThenPlaceProcessor(
            parser_tokenizer=tokenizer,
            placement_tokenizer=tokenizer,
        ),
        parser=parser,
        placement=placement,
    )

    with patch("parse_then_place.pipeline_parse_then_place.set_seed") as mocked_seed:
        output = pipe(
            prompt=["create text", "create image"],
            seed=99,
            output_candidate="best",
            output_type="dict",
            return_intermediates=True,
        )

    mocked_seed.assert_called_once_with(99)
    assert isinstance(output, dict)
    assert cast(torch.Tensor, output["bbox"]).shape == (2, 1, 4)
    intermediates = cast(dict[str, object], output["intermediates"])
    assert intermediates["prompt"] == ["create text", "create image"]
    assert intermediates["logical_forms"] == ["text 0 0 10 20", "text 0 0 10 20"]


def test_pipeline_requires_processor_tokenizers_for_inference() -> None:
    pipe = _pipeline()

    with pytest.raises(ValueError, match="parser_tokenizer"):
        pipe(prompt="create text")
