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
        scheduler=LayouSynScheduler(num_train_timesteps=2),
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
    concept_embeds = torch.zeros(1, 1, 4)
    out1 = pipe(
        labels=[["cat"]],
        caption_embeds=caption_embeds,
        caption_padding_mask=caption_padding_mask,
        concept_embeds=concept_embeds,
        num_inference_steps=1,
        seed=123,
        generator=torch.Generator().manual_seed(0),
    )
    out2 = pipe(
        labels=[["cat"]],
        caption_embeds=caption_embeds,
        caption_padding_mask=caption_padding_mask,
        concept_embeds=concept_embeds,
        num_inference_steps=1,
        seed=999,
        generator=torch.Generator().manual_seed(0),
    )
    assert isinstance(out1, LayoutGenerationOutput)
    assert isinstance(out2, LayoutGenerationOutput)
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.allclose(out1.bbox, out2.bbox)


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
