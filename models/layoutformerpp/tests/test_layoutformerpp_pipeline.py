from pathlib import Path
from typing import cast

import torch
from pytest import MonkeyPatch

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType
from laygen.pipelines import LayoutGenerationPipeline

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
    LayoutGenerationOutput,
)


def test_save_load_smoke(tmp_path: Path) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="gen_t")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    pipe = LayoutFormerPPPipeline(model=model, processor=processor)
    assert isinstance(pipe, LayoutGenerationPipeline)

    pipe.save_pretrained(tmp_path)
    loaded_pipe = LayoutFormerPPPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded_pipe, LayoutGenerationPipeline)

    pipe = loaded_pipe
    out = pipe(condition_type="label", labels=[["Text"]], max_length=2)
    assert isinstance(out, LayoutGenerationOutput)
    assert out.sequences is not None
    assert out.sequences.shape[0] == 1


def test_pipeline_passes_public_condition_arguments_to_processor(
    monkeypatch: MonkeyPatch,
) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico", task="completion")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    pipe = LayoutFormerPPPipeline(
        model=LayoutFormerPPForConditionalGeneration(config),
        processor=processor,
    )
    captured: dict[str, object] = {}
    original_call = LayoutFormerPPProcessor.__call__

    def record_call(self: LayoutFormerPPProcessor, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_call(
            self,
            condition_type=cast(
                ConditionType | str,
                kwargs.get("condition_type", ConditionType.unconditional),
            ),
            labels=cast(list[list[int | str]] | None, kwargs.get("labels")),
            bbox=kwargs.get("bbox"),
            mask=cast(
                torch.Tensor | list[list[bool]] | list[bool] | None,
                kwargs.get("mask"),
            ),
            relations=cast(
                list[list[tuple[int, int, int, int, int]]] | None,
                kwargs.get("relations"),
            ),
            batch_size=cast(int | None, kwargs.get("batch_size")),
            box_format=cast(BoxFormat | str, kwargs.get("box_format", BoxFormat.xywh)),
            normalized=cast(bool, kwargs.get("normalized", True)),
            canvas_size=cast(tuple[int, int] | None, kwargs.get("canvas_size")),
            return_tensors="pt",
        )

    monkeypatch.setattr(LayoutFormerPPProcessor, "__call__", record_call)
    mask = torch.tensor([[True, False]])

    pipe(
        condition_type="completion",
        labels=[["Text", "Image"]],
        bbox=torch.tensor([[[50.0, 50.0, 20.0, 20.0], [80.0, 80.0, 10.0, 10.0]]]),
        mask=mask,
        normalized=False,
        canvas_size=(100, 100),
        max_length=1,
    )

    assert captured["mask"] is mask
    assert captured["normalized"] is False
    assert captured["canvas_size"] == (100, 100)
