import torch
from typing import Literal, cast

from laygen.common.discrete import SamplingMode
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from laygen.common.testing import assert_layout_output_schema
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.denoiser import LayoutDMDenoiser
from layout_dm.pipeline import LayoutDMPipeline
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.scheduler import LayoutDMScheduler
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def make_pipeline() -> LayoutDMPipeline:
    cfg = LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=2,
        num_bin_bboxes=4,
        bbox_quantization="linear",
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        num_timesteps=4,
    )
    tokenizer = LayoutDMTokenizer(cfg)
    denoiser = LayoutDMDenoiser(
        vocab_size=cfg.vocab_size,
        max_token_length=cfg.max_token_length,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_hidden_layers=cfg.num_hidden_layers,
        intermediate_size=cfg.intermediate_size,
    )
    scheduler = LayoutDMScheduler(
        num_timesteps=cfg.num_timesteps,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        pad_token_id=cfg.pad_token_id,
        token_mask=tokenizer.token_mask().tolist(),
        per_var_full_ids=tokenizer.full_id_maps(),
    )
    return LayoutDMPipeline(
        denoiser=denoiser,
        scheduler=scheduler,
        tokenizer=tokenizer,
        processor=LayoutDMProcessor(tokenizer),
    )


def test_pipeline_contract_and_seed_reproducible():
    pipe = make_pipeline()
    out1 = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
    out2 = pipe(batch_size=1, seed=999, num_inference_steps=1, sampling="deterministic")
    assert isinstance(out1, LayoutGenerationOutput)
    assert isinstance(out2, LayoutGenerationOutput)
    assert out1.sequences is not None
    assert out2.sequences is not None
    assert_layout_output_schema(out1, batch_size=1)
    assert torch.equal(
        cast(torch.Tensor, out1.sequences), cast(torch.Tensor, out2.sequences)
    )


def test_pipeline_conditional_dict_and_intermediates():
    pipe = make_pipeline()
    out = pipe(
        condition_type="cat_cond",
        bbox=[[[0.5, 0.5, 0.25, 0.25]]],
        labels=[[1]],
        mask=[[True]],
        batch_size=3,
        seed=0,
        num_inference_steps=1,
        sampling="deterministic",
        output_type="dict",
        return_intermediates=True,
    )

    assert out["bbox"].shape[0] == 1
    assert out["intermediates"] == {"condition_type": "label"}
    assert len(out["trajectory"]) == 1


def test_pipeline_validates_condition_and_output_type():
    pipe = make_pipeline()
    try:
        pipe(condition_type="label", labels=[[0]], num_inference_steps=1)
    except ValueError as exc:
        assert "bbox and labels are required" in str(exc)
    else:
        raise AssertionError("missing conditional bbox should fail")

    try:
        pipe(
            batch_size=1,
            num_inference_steps=1,
            output_type=cast(Literal["dataclass"], "tuple"),
        )
    except ValueError as exc:
        assert "Unsupported output_type" in str(exc)
    else:
        raise AssertionError("unsupported output_type should fail")


def test_pipeline_save_load_roundtrip(tmp_path):
    pipe = make_pipeline()
    pipe.save_pretrained(tmp_path)
    loaded = LayoutDMPipeline.from_pretrained(tmp_path)
    out = loaded(
        batch_size=1,
        seed=0,
        num_inference_steps=1,
        sampling=SamplingMode.deterministic,
    )
    assert isinstance(out, LayoutGenerationOutput)
    assert_layout_output_schema(out, batch_size=1)
