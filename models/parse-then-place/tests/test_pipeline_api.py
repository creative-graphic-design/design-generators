from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from parse_then_place import (
    ParseThenPlaceConfig,
    ParseThenPlaceForConditionalGeneration,
    ParseThenPlacePipeline,
    ParseThenPlaceProcessor,
)


def _pipeline() -> ParseThenPlacePipeline:
    config = ParseThenPlaceConfig(dataset_name="rico")
    model = ParseThenPlaceForConditionalGeneration(config)
    processor = ParseThenPlaceProcessor.from_config("rico")
    return ParseThenPlacePipeline(model=model, processor=processor)


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

    with patch("parse_then_place.modeling_parse_then_place.set_seed") as mocked:
        pipe(
            prompt="create a simple screen",
            seed=123,
            generator=torch.Generator().manual_seed(123),
            layout_text="text 0 0 10 20",
        )

    mocked.assert_not_called()


def test_model_save_and_from_pretrained_smoke() -> None:
    config = ParseThenPlaceConfig(dataset_name="rico")
    model = ParseThenPlaceForConditionalGeneration(config)

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        model.save_pretrained(output_dir)
        loaded = ParseThenPlaceForConditionalGeneration.from_pretrained(
            output_dir,
            local_files_only=True,
        )

    assert loaded.config.dataset_name == "rico"
    assert loaded.config.id2label[4] == "text"
