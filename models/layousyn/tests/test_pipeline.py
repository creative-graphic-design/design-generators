from typing import cast

import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from layousyn import (
    LayouSynDiTModel,
    LayouSynPipeline,
    LayouSynProcessor,
    LayouSynScheduler,
)


def tiny_pipe() -> LayouSynPipeline:
    return LayouSynPipeline(
        model=LayouSynDiTModel(
            model_name="DiT-D1-H32-N1",
            max_in_len=2,
            max_y_len=3,
            concept_in_channels=4,
            y_in_channels=4,
            class_dropout_prob=0.0,
        ),
        scheduler=LayouSynScheduler(num_train_timesteps=100),
        processor=LayouSynProcessor(
            max_in_len=2,
            max_y_len=3,
            concept_in_channels=4,
            y_in_channels=4,
        ),
    )


def test_pipeline_output_schema_and_generator_precedence() -> None:
    pipe = tiny_pipe()
    caption_embeds = torch.zeros(1, 3, 4)
    caption_padding_mask = torch.ones(1, 3, dtype=torch.bool)
    caption_padding_mask[:, 0] = False
    concept_embeds = torch.zeros(1, 1, 4)
    out1 = pipe(
        labels=[["cat"]],
        caption_embeds=caption_embeds,
        caption_padding_mask=caption_padding_mask,
        concept_embeds=concept_embeds,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=123,
        generator=torch.Generator().manual_seed(0),
    )
    out2 = pipe(
        labels=[["cat"]],
        caption_embeds=caption_embeds,
        caption_padding_mask=caption_padding_mask,
        concept_embeds=concept_embeds,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=999,
        generator=torch.Generator().manual_seed(0),
    )
    assert isinstance(out1, LayoutGenerationOutput)
    assert isinstance(out2, LayoutGenerationOutput)
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.allclose(out1.bbox, out2.bbox)


def test_pipeline_no_cfg_dict_intermediates_and_num_elements() -> None:
    pipe = tiny_pipe()
    out = cast(
        dict[str, object],
        pipe(
            labels=[["cat"]],
            caption_embeds=torch.zeros(1, 3, 4),
            caption_padding_mask=torch.tensor([[False, True, True]]),
            concept_embeds=torch.zeros(1, 1, 4),
            num_inference_steps=1,
            guidance_scale=1.0,
            output_type="dict",
            return_intermediates=True,
            seed=0,
        ),
    )
    assert cast(torch.Tensor, out["bbox"]).shape == (1, 2, 4)
    intermediates = cast(dict[str, object], out["intermediates"])
    nested_intermediates = cast(dict[str, object], intermediates["intermediates"])
    assert nested_intermediates["trajectory"]
    try:
        pipe(
            labels=[["cat"]],
            caption_embeds=torch.zeros(1, 3, 4),
            caption_padding_mask=torch.tensor([[False, True, True]]),
            concept_embeds=torch.zeros(1, 1, 4),
            num_elements=2,
            num_inference_steps=1,
        )
    except ValueError as exc:
        assert "num_elements" in str(exc)
    else:
        raise AssertionError("num_elements mismatch should fail")


def test_pipeline_rejects_unsupported_condition() -> None:
    pipe = tiny_pipe()
    try:
        pipe(condition_type="label", labels=[["cat"]])
    except NotImplementedError as exc:
        assert "condition_type='text'" in str(exc)
    else:
        raise AssertionError("unsupported condition should fail")


def test_pipeline_save_from_pretrained_smoke(tmp_path) -> None:
    pipe = tiny_pipe()
    pipe.save_pretrained(tmp_path)
    pipe.processor.save_pretrained(tmp_path)
    loaded = LayouSynPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded, LayouSynPipeline)
    assert loaded.processor.max_in_len == 2
