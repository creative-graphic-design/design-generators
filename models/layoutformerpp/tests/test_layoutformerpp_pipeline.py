import json
from pathlib import Path
from typing import cast

import torch
import pytest
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
from layoutformerpp.constraints import LayoutFormerPPGenerationConstraint
from pytest import MonkeyPatch


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
    assert cast(torch.Tensor, out.sequences).shape[0] == 1


def test_config_roundtrip_preserves_token_fields() -> None:
    config = LayoutFormerPPConfig(
        vocab_size=17,
        bos_token_id=3,
        eos_token_id=4,
        pad_token_id=5,
        name_or_path="layoutformerpp-test",
        task_specific_params={"decode_max_length": 9},
    )

    restored = LayoutFormerPPConfig.from_dict(config.to_dict())

    assert restored.vocab_size == 17
    assert restored.bos_token_id == 3
    assert restored.eos_token_id == 4
    assert restored.pad_token_id == 5
    assert restored.name_or_path == "layoutformerpp-test"
    assert restored.task_specific_params == {"decode_max_length": 9}


def test_pipeline_honors_custom_config_decode_length(monkeypatch: MonkeyPatch) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="ugen")
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        dataset="rico25",
        task="ugen",
        decode_max_length=7,
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    captured: dict[str, int | None] = {}

    def capture_max_length(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["max_length"] = cast(int | None, kwargs["max_length"])
        return input_ids.new_tensor([[model.eos_token_id]])

    monkeypatch.setattr(model, "_generate_sequences", capture_max_length)
    pipe = LayoutFormerPPPipeline(model=model, processor=processor)

    pipe(condition_type=ConditionType.unconditional)

    assert captured["max_length"] == 7


@pytest.mark.parametrize(
    ("condition", "task", "expected_max_length", "constrained", "do_sample"),
    [
        (ConditionType.unconditional, "ugen", 120, False, True),
        (ConditionType.label, "gen_t", 120, True, False),
        (ConditionType.label_size, "gen_ts", 120, True, False),
        (ConditionType.completion, "completion", 120, False, True),
        (ConditionType.refinement, "refinement", None, False, False),
        (ConditionType.relation, "gen_r", 150, True, False),
    ],
)
def test_pipeline_defaults_match_vendor_decode_lengths(
    condition: ConditionType,
    task: str,
    expected_max_length: int | None,
    constrained: bool,
    do_sample: bool,
    monkeypatch: MonkeyPatch,
) -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="publaynet", task=task)
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        dataset="publaynet",
        task=task,
        decode_max_length=150,
        max_position_embeddings=400,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    captured: dict[str, int | bool] = {}

    def capture_generation_options(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["max_length"] = cast(int, kwargs["max_length"])
        captured["input_length"] = input_ids.size(-1)
        captured["constrained_decoding"] = cast(bool, kwargs["constrained_decoding"])
        captured["do_sample"] = cast(bool, kwargs["do_sample"])
        return input_ids.new_tensor([[model.eos_token_id]])

    monkeypatch.setattr(model, "_generate_sequences", capture_generation_options)
    pipe = LayoutFormerPPPipeline(model=model, processor=processor)

    pipe(
        condition_type=condition,
        labels=[["Text"]],
        bbox=[[[0.5, 0.5, 0.2, 0.2]]],
        relations=[[]],
    )

    max_length = captured["max_length"]
    if expected_max_length is None:
        expected_max_length = captured["input_length"] + 1

    assert max_length == expected_max_length
    assert captured["constrained_decoding"] is constrained
    assert captured["do_sample"] is do_sample


def test_pipeline_passes_separator_setting_to_generation_constraint(
    monkeypatch: MonkeyPatch,
) -> None:
    processor = LayoutFormerPPProcessor.from_config(
        dataset="rico25",
        task="gen_t",
        add_sep_token=False,
    )
    config = LayoutFormerPPConfig(
        vocab_size=processor.tokenizer.vocab_size,
        dataset="rico25",
        task="gen_t",
        add_sep_token=False,
        max_position_embeddings=16,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        dropout=0.0,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    captured: dict[str, object] = {}

    def capture_constraint(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["constraint"] = kwargs["generation_constraint_fn"]
        return input_ids.new_tensor([[model.eos_token_id]])

    monkeypatch.setattr(model, "_generate_sequences", capture_constraint)
    pipe = LayoutFormerPPPipeline(model=model, processor=processor)

    pipe(condition_type=ConditionType.label, labels=[["Text"]])

    constraint = cast(LayoutFormerPPGenerationConstraint, captured["constraint"])
    assert constraint.add_sep_token is False


def test_config_from_pretrained_accepts_legacy_and_unknown_keys(
    tmp_path: Path,
) -> None:
    config = LayoutFormerPPConfig(vocab_size=17)
    config.save_pretrained(tmp_path)

    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload.update(
        {
            "add_cross_attention": True,
            "use_cache": False,
            "num_beams": 3,
            "legacy_unknown": "kept",
        }
    )
    config_path.write_text(json.dumps(payload))

    restored = LayoutFormerPPConfig.from_pretrained(tmp_path)

    assert restored.add_cross_attention is True
    assert restored.use_cache is False
    assert restored.num_beams == 3
    assert restored.legacy_unknown == "kept"


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
            bbox=cast(
                torch.Tensor | list[list[int | float]] | None, kwargs.get("bbox")
            ),
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
