from typing import Literal, cast

import torch
import pytest

from layoutdiffusion import (
    LayoutDiffusionConfig,
    LayoutDiffusionPipeline,
    LayoutDiffusionScheduler,
    LayoutDiffusionTokenizer,
    LayoutDiffusionTransformer,
)
from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput


def _pipe() -> LayoutDiffusionPipeline:
    cfg = LayoutDiffusionConfig(
        dataset_name="publaynet",
        diffusion_steps=10,
        num_channels=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    tokenizer = LayoutDiffusionTokenizer(cfg)
    return LayoutDiffusionPipeline(
        LayoutDiffusionTransformer(
            vocab_size=cfg.vocab_size,
            num_channels=cfg.num_channels,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            intermediate_size=cfg.intermediate_size,
        ),
        LayoutDiffusionScheduler(
            num_train_timesteps=cfg.diffusion_steps,
            vocab_size=cfg.vocab_size,
            mask_token_id=cfg.mask_token_id,
            type_classes=cfg.type_classes,
        ),
        tokenizer,
    )


def test_pipeline_schema_and_aliases() -> None:
    pipe = _pipe()
    out = cast(
        LayoutGenerationOutput,
        pipe(batch_size=1, condition_type="ugen", seed=0, num_inference_steps=1),
    )
    assert_layout_output_schema(out, batch_size=1)
    assert out.sequences is None


def test_pipeline_generator_precedence_over_seed() -> None:
    pipe = _pipe()
    out1 = cast(
        LayoutGenerationOutput,
        pipe(
            batch_size=1,
            seed=999,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=1,
        ),
    )
    out2 = cast(
        LayoutGenerationOutput,
        pipe(
            batch_size=1,
            seed=123,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=1,
        ),
    )
    assert torch.equal(out1.labels, out2.labels)
    assert torch.equal(out1.mask, out2.mask)


def test_pipeline_save_and_load_roundtrip(tmp_path) -> None:
    pipe = _pipe()
    pipe.save_pretrained(tmp_path)
    loaded = LayoutDiffusionPipeline.from_pretrained(tmp_path)
    out = cast(
        LayoutGenerationOutput, loaded(batch_size=1, seed=0, num_inference_steps=1)
    )
    assert_layout_output_schema(out, batch_size=1)


def test_pipeline_output_type_dict_and_invalid() -> None:
    pipe = _pipe()
    out = pipe(batch_size=1, seed=0, num_inference_steps=1, output_type="dict")
    assert isinstance(out, dict)
    assert set(out) >= {"bbox", "labels", "mask", "id2label"}
    with pytest.raises(ValueError):
        pipe(
            batch_size=1,
            seed=0,
            num_inference_steps=1,
            output_type=cast(Literal["dataclass", "dict"], "bad"),
        )


def test_pipeline_rejects_unsupported_condition() -> None:
    pipe = _pipe()
    with pytest.raises(NotImplementedError):
        pipe(condition_type="label_size", num_inference_steps=1)
