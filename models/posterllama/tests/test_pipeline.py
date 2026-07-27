from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from laygen.modeling_outputs import LayoutGenerationOutput
from posterllama import PosterLlamaConfig, PosterLlamaPipeline, PosterLlamaProcessor
from posterllama.modeling_posterllama import PosterLlamaRuntime


def _pipeline(runtime: PosterLlamaRuntime | None = None) -> PosterLlamaPipeline:
    config = PosterLlamaConfig(canvas_size=(100, 100))
    return PosterLlamaPipeline(
        config=config,
        processor=PosterLlamaProcessor.from_config(config),
        runtime=runtime,
    )


def test_processor_only_save_load_round_trip(tmp_path: Path) -> None:
    pipe = _pipeline()
    pipe.save_pretrained(tmp_path)

    loaded = PosterLlamaPipeline.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.runtime is None
    assert loaded.config.canvas_size == (100, 100)


def test_missing_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="runtime assets are missing"):
        _pipeline()(images=None)


def test_fake_runtime_generation_and_intermediates() -> None:
    text = (
        '<svg width="100" height="100">'
        '<rect data-category="text" x="0" y="0" width="10" height="20"/>'
        "</svg>"
    )
    generator = torch.Generator().manual_seed(0)

    output = cast(
        LayoutGenerationOutput,
        _pipeline(PosterLlamaRuntime(text))(
            images=None,
            generator=generator,
            seed=123,
            return_intermediates=True,
        ),
    )
    intermediates = cast(dict[str, object], output.intermediates)
    generation_args = cast(dict[str, object], intermediates["generation_args"])

    assert output.labels.tolist() == [[1]]
    assert generation_args["do_sample"] is True
    assert intermediates["raw_generated_text"] == text


def test_pipeline_output_type_dict() -> None:
    text = (
        '<svg width="100" height="100">'
        '<rect data-category="logo" x="0" y="0" width="10" height="10"/>'
        "</svg>"
    )

    output = cast(
        dict[str, object],
        _pipeline(PosterLlamaRuntime(text))(images=None, output_type="dict"),
    )
    labels = cast(torch.Tensor, output["labels"])

    assert labels.tolist() == [[0]]


def test_pipeline_rejects_invalid_output_type_after_generation() -> None:
    text = (
        '<svg width="100" height="100">'
        '<rect data-category="logo" x="0" y="0" width="10" height="10"/>'
        "</svg>"
    )

    with pytest.raises(ValueError, match="Unsupported output_type"):
        _pipeline(PosterLlamaRuntime(text))(
            images=None,
            output_type=cast(Literal["dataclass", "dict"], "tuple"),
        )
