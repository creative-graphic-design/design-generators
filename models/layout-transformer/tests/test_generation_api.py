import torch
from typing import cast

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from layout_transformer import (
    LayoutObject,
    LayoutRelation,
    LayoutTransformerConfig,
    LayoutTransformerForLayoutGeneration,
    LayoutTransformerPipeline,
    LayoutTransformerProcessor,
)


def build_pipeline():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person", 2: "table"},
        relation_id2label={1: "left of"},
        max_sequence_length=8,
    )
    config = LayoutTransformerConfig(
        vocab_size=processor.tokenizer.vocab_size,
        obj_classes_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_sequence_length=8,
        id2label=processor.id2label,
        relation_id2label=processor.relation_id2label,
    )
    return LayoutTransformerPipeline(
        model=LayoutTransformerForLayoutGeneration(config),
        processor=processor,
    )


def build_stochastic_pipeline():
    processor = LayoutTransformerProcessor.from_config(
        id2label={0: "__image__", 1: "person", 2: "table"},
        relation_id2label={1: "left of"},
        max_sequence_length=8,
    )
    config = LayoutTransformerConfig(
        vocab_size=processor.tokenizer.vocab_size,
        obj_classes_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        dropout=0.0,
        max_sequence_length=8,
        decoder_greedy=False,
        id2label=processor.id2label,
        relation_id2label=processor.relation_id2label,
    )
    return LayoutTransformerPipeline(
        model=LayoutTransformerForLayoutGeneration(config),
        processor=processor,
    )


def test_pipeline_returns_common_schema():
    pipe = build_pipeline()
    output = pipe(
        objects=[
            LayoutObject(id="a", label="person"),
            LayoutObject(id="b", label="table"),
        ],
        relations=[LayoutRelation(subject="a", predicate="left of", object="b")],
        generator=torch.Generator().manual_seed(1),
        seed=999,
    )

    assert_layout_output_schema(output, batch_size=1)
    assert output.mask.any()


def test_pipeline_output_type_dict():
    pipe = build_pipeline()
    output = pipe(
        objects=[LayoutObject(id="a", label="person")],
        output_type="dict",
    )

    assert set(output) >= {"bbox", "labels", "mask", "id2label"}


def test_pipeline_rejects_denormalized_output():
    pipe = build_pipeline()

    try:
        pipe(objects=[LayoutObject(id="a", label="person")], normalized=False)
    except ValueError as exc:
        assert "normalized boxes only" in str(exc)
    else:
        raise AssertionError("expected normalized output error")


def test_pipeline_save_pretrained_round_trip(tmp_path):
    pipe = build_pipeline()
    pipe.save_pretrained(tmp_path)

    loaded = LayoutTransformerPipeline.from_pretrained(tmp_path, local_files_only=True)
    output = loaded(objects=[LayoutObject(id="a", label="person")])

    assert_layout_output_schema(cast(LayoutGenerationOutput, output), batch_size=1)


def test_pipeline_runs_model_in_eval_mode_during_generation(monkeypatch):
    pipe = build_pipeline()
    pipe.model.train()
    original_generate = pipe.model._generate_boxes

    def wrapped_generate(*args, **kwargs):
        assert pipe.model.training is False
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(pipe.model, "_generate_boxes", wrapped_generate)
    pipe(objects=[LayoutObject(id="a", label="person")])

    assert pipe.model.training is True


def test_pipeline_generator_reproducibility_and_seed_variation():
    pipe = build_stochastic_pipeline()
    objects = [
        LayoutObject(id="a", label="person"),
        LayoutObject(id="b", label="table"),
    ]
    relations = [LayoutRelation(subject="a", predicate="left of", object="b")]

    first = cast(
        LayoutGenerationOutput,
        pipe(
            objects=objects,
            relations=relations,
            generator=torch.Generator().manual_seed(123),
        ),
    )
    second = cast(
        LayoutGenerationOutput,
        pipe(
            objects=objects,
            relations=relations,
            generator=torch.Generator().manual_seed(123),
        ),
    )
    different = cast(
        LayoutGenerationOutput,
        pipe(objects=objects, relations=relations, seed=456),
    )

    first_bbox = cast(torch.Tensor, first.bbox)
    second_bbox = cast(torch.Tensor, second.bbox)
    different_bbox = cast(torch.Tensor, different.bbox)
    assert torch.equal(first_bbox, second_bbox)
    assert not torch.equal(first_bbox, different_bbox)
