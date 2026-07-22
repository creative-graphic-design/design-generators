from collections.abc import Mapping
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
from ralf.modeling_ralf import RalfTaskPreprocessor


def _pipeline() -> RalfPipeline:
    config = RalfConfig(
        max_seq_length=2,
        num_bin=8,
        decoder_d_model=16,
        decoder_layers=1,
        num_attention_heads=4,
        d_model=16,
    )
    model = RalfForConditionalLayoutGeneration(config)

    def encode(
        inputs: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        image = cast(torch.Tensor, inputs["image"])
        return {
            "memory": torch.zeros(
                image.size(0),
                1,
                config.decoder_d_model,
                device=image.device,
            )
        }

    model._encode_into_memory = encode  # ty: ignore[invalid-assignment]
    return RalfPipeline(model=model, processor=RalfProcessor.from_config(config))


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


def test_pipeline_retrieval_table_intermediates_and_retrieval_condition() -> None:
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


def test_pipeline_explicit_retrieval_indexes_are_returned() -> None:
    pipe = _pipeline()
    cfg = pipe.config
    retrieved_layouts = {
        "bbox": torch.zeros(1, cfg.top_k, cfg.max_seq_length, 4),
        "labels": torch.zeros(1, cfg.top_k, cfg.max_seq_length, dtype=torch.long),
        "mask": torch.ones(1, cfg.top_k, cfg.max_seq_length, dtype=torch.bool),
    }

    output = pipe(
        condition_type="unconditional",
        retrieved_layouts=retrieved_layouts,
        retrieved_images=torch.zeros(1, cfg.top_k, 3, 1, 1),
        retrieved_saliency=torch.zeros(1, cfg.top_k, 1, 1, 1),
        retrieved_indexes=torch.arange(cfg.top_k).reshape(1, cfg.top_k),
        return_intermediates=True,
        top_k=5,
        seed=3,
    )
    output = cast(LayoutGenerationOutput, output)

    intermediates = cast(dict[str, object], output.intermediates)
    retrieval = cast(dict[str, torch.Tensor], intermediates["retrieval"])
    assert retrieval["indexes"].shape == (1, cfg.top_k)
    try:
        pipe(condition_type="text")
    except NotImplementedError as exc:
        assert "text" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


def test_pipeline_passes_relation_constraints_to_model_runtime() -> None:
    pipe = _pipeline()
    relation = [
        pipe.model.tokenizer.names[0],
        "A",
        "left",
        pipe.model.tokenizer.names[1],
        "B",
    ]
    encoded = pipe.processor(
        condition_type="relation",
        labels=torch.tensor([[0, 1]]),
        bbox=torch.full((1, 2, 4), 0.5),
        mask=torch.tensor([[True, True]]),
    )
    prepared = pipe.model._prepare_conditional_inputs(
        pixel_values=cast(torch.Tensor, encoded["pixel_values"]),
        saliency=cast(torch.Tensor, encoded["saliency"]),
        retrieved=None,
        batch_size=1,
        condition_type="relation",
        constraint_input_ids=cast(torch.Tensor, encoded["input_ids"]),
        constraint_mask=cast(torch.Tensor, encoded["attention_mask"]),
        constraint_element_mask=cast(torch.Tensor, encoded["constraint_mask"]),
        relationship_table={"sample": [relation]},
        sample_ids=["sample"],
    )

    relation_sep = RalfTaskPreprocessor(
        pipe.model.tokenizer,
        task="relation",
    ).name_to_id("relation_sep")
    seq_layout_const = cast(torch.Tensor, prepared["seq_layout_const"])
    assert relation_sep in seq_layout_const[0].tolist()
