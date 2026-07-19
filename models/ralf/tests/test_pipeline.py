from pathlib import Path
from typing import cast

import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from ralf import (
    RalfConfig,
    RalfForConditionalLayoutGeneration,
    RalfPipeline,
    RalfProcessor,
)


def _pipeline() -> RalfPipeline:
    config = RalfConfig(
        max_seq_length=2,
        num_bin=8,
        decoder_d_model=16,
        decoder_layers=1,
        num_attention_heads=4,
        d_model=16,
    )
    return RalfPipeline(
        model=RalfForConditionalLayoutGeneration(config),
        processor=RalfProcessor.from_config(config),
    )


def test_pipeline_smoke_and_schema() -> None:
    pipe = _pipeline()

    output = pipe(
        condition_type="unconditional",
        batch_size=1,
        seed=1,
        top_k=5,
        return_intermediates=True,
    )
    output = cast(LayoutGenerationOutput, output)

    assert_layout_output_schema(output, batch_size=1)
    sequences = cast(torch.Tensor, output.sequences)
    assert sequences.shape[0] == 1


def test_pipeline_generator_wins_over_seed() -> None:
    pipe = _pipeline()
    generator_a = torch.Generator().manual_seed(4)
    generator_b = torch.Generator().manual_seed(4)

    first = pipe(seed=1, generator=generator_a, top_k=5)
    second = pipe(seed=999, generator=generator_b, top_k=5)
    first = cast(LayoutGenerationOutput, first)
    second = cast(LayoutGenerationOutput, second)

    first_sequences = cast(torch.Tensor, first.sequences)
    second_sequences = cast(torch.Tensor, second.sequences)
    assert torch.equal(first_sequences, second_sequences)


def test_pipeline_save_pretrained_round_trip(tmp_path: Path) -> None:
    pipe = _pipeline()

    pipe.save_pretrained(tmp_path)
    loaded = RalfPipeline.from_pretrained(tmp_path, local_files_only=True)

    assert loaded.config.model_type == "ralf"
    assert loaded.processor.config.max_seq_length == 2


def test_pipeline_retrieval_table_intermediates_and_unsupported_condition() -> None:
    from ralf import RalfRetrievalTable

    pipe = _pipeline()
    table = RalfRetrievalTable({"q": [1, 2]}, top_k=2)

    output = pipe(
        condition_type="retrieval",
        retrieval_table=table,
        query_ids=["q"],
        return_intermediates=True,
        top_k=5,
        seed=0,
    )
    output = cast(LayoutGenerationOutput, output)

    intermediates = cast(dict[str, object], output.intermediates)
    retrieval = cast(dict[str, torch.Tensor], intermediates["retrieval"])
    assert retrieval["indexes"].tolist() == [[1, 2]]
    try:
        pipe(condition_type="text")
    except NotImplementedError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")
