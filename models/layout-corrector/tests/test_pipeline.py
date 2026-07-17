import torch

from layout_corrector import LayoutCorrectorModel, LayoutCorrectorPipeline
from layout_dm import (
    LayoutDMDenoiser,
    LayoutDMPipeline,
    LayoutDMScheduler,
    LayoutDMTokenizer,
)
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_generation_common.outputs import LayoutGenerationOutput


def tiny_layout_dm():
    config = LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=2,
        num_bin_bboxes=2,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        num_timesteps=4,
        bbox_quantization="linear",
    )
    tokenizer = LayoutDMTokenizer(config=config)
    scheduler = LayoutDMScheduler(
        num_timesteps=config.num_timesteps,
        vocab_size=config.vocab_size,
        mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
        token_mask=tokenizer.token_mask().tolist(),
        per_var_full_ids=tokenizer.full_id_maps(),
    )
    denoiser = LayoutDMDenoiser(
        vocab_size=config.vocab_size,
        max_token_length=config.max_token_length,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        timestep_type=config.timestep_type,
    )
    return LayoutDMPipeline(
        denoiser=denoiser,
        scheduler=scheduler,
        tokenizer=tokenizer,
    )


def tiny_corrector(vocab_size):
    return LayoutCorrectorModel(
        dataset_name="publaynet",
        vocab_size=vocab_size,
        max_seq_length=2,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        num_timesteps=4,
        corrector_t_list=(2,),
        use_gumbel_noise=False,
    )


def tiny_pipeline():
    layout_dm = tiny_layout_dm()
    return LayoutCorrectorPipeline(
        layout_dm=layout_dm,
        corrector=tiny_corrector(layout_dm.tokenizer.config.vocab_size),
    )


def test_pipeline_returns_common_output_schema():
    pipe = tiny_pipeline()

    output = pipe(
        batch_size=2,
        seed=5,
        num_inference_steps=2,
        sampling="deterministic",
        corrector_t_list=(),
    )

    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.shape == (2, 2, 4)
    assert output.labels.shape == (2, 2)
    assert output.mask.shape == (2, 2)
    assert output.id2label[0] == "text"


def test_pipeline_generator_overrides_seed():
    pipe = tiny_pipeline()
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(11)

    first = pipe(
        batch_size=1,
        seed=123,
        generator=generator_a,
        num_inference_steps=2,
        corrector_t_list=(),
    )
    second = pipe(
        batch_size=1,
        seed=999,
        generator=generator_b,
        num_inference_steps=2,
        corrector_t_list=(),
    )

    assert torch.equal(first.sequences, second.sequences)


def test_pipeline_save_load_roundtrip(tmp_path):
    pipe = tiny_pipeline()
    pipe.save_pretrained(tmp_path)

    loaded = LayoutCorrectorPipeline.from_pretrained(tmp_path)
    output = loaded(
        batch_size=1,
        seed=3,
        num_inference_steps=1,
        sampling="deterministic",
        corrector_t_list=(),
    )

    assert isinstance(output, LayoutGenerationOutput)
    assert output.id2label[4] == "figure"
