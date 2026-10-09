import torch
import pytest

from layout_corrector import CorrectorMaskMode
from layout_corrector import LayoutCorrectorModel, LayoutCorrectorPipeline
from layout_corrector.pipeline_layout_corrector import OutputType
from layout_dm import (
    LayoutDMDenoiser,
    LayoutDMPipeline,
    LayoutDMScheduler,
    LayoutDMTokenizer,
)
from layout_dm.configuration_layout_dm import LayoutDMConfig
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from laygen.common.discrete import index_to_log_onehot, log_onehot_to_index
from laygen.common.randomness import multinomial
from layout_corrector.sampling import select_tokens_to_remask


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


def test_pipeline_to_propagates_dtype_to_nested_layout_dm():
    pipe = tiny_pipeline()
    pipe.layout_dm.denoiser.to(dtype=torch.float64)
    pipe.corrector.to(dtype=torch.float64)

    pipe.to(dtype=torch.float32)

    assert next(pipe.layout_dm.denoiser.parameters()).dtype == torch.float32
    assert next(pipe.corrector.parameters()).dtype == torch.float32


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
    assert output.id2label is not None
    assert output.id2label[4] == "figure"


def test_pipeline_accepts_enum_modes_and_dict_output():
    pipe = tiny_pipeline()

    output = pipe(
        batch_size=1,
        seed=7,
        num_inference_steps=1,
        sampling="deterministic",
        corrector_t_list=(),
        corrector_mask_mode=CorrectorMaskMode.topk,
        output_type=OutputType.dict,
    )

    assert isinstance(output, dict)
    assert output["bbox"].shape == (1, 2, 4)


def test_pipeline_rejects_unknown_output_type():
    pipe = tiny_pipeline()

    with pytest.raises(ValueError, match="Unsupported output_type"):
        pipe(batch_size=1, num_inference_steps=1, output_type="tuple")


def test_pipeline_runs_conditioned_corrector_branch_with_intermediates():
    pipe = tiny_pipeline()

    output = pipe(
        seed=0,
        condition_type="label",
        labels=[[0, 1]],
        bbox=[[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]],
        mask=[[True, False]],
        num_inference_steps=2,
        sampling="deterministic",
        corrector_t_list=(2,),
        corrector_steps=1,
        corrector_mask_mode=CorrectorMaskMode.thresh,
        corrector_mask_threshold=0.5,
        use_gumbel_noise=True,
        time_adaptive_temperature=True,
        return_intermediates=True,
    )

    assert isinstance(output, LayoutGenerationOutput)
    assert output.scores is not None
    assert output.trajectory is not None
    assert output.intermediates == {"condition_type": "label"}


def test_pipeline_matches_vendor_token_trajectory_through_corrector_steps():
    pipe = tiny_pipeline()
    pipe.layout_dm.scheduler.config.num_timesteps = 100
    pipe.layout_dm.scheduler.set_timesteps(100)
    generator = torch.Generator().manual_seed(17)

    expected_sample = pipe.layout_dm.scheduler.initial_sample(
        1,
        pipe.layout_dm.tokenizer.config.max_token_length,
        device=torch.device("cpu"),
    )
    expected_trajectory = []
    for timestep in pipe.layout_dm.scheduler.timesteps:
        timestep_value = int(timestep.item())
        timestep_batch = torch.full((1,), timestep_value, dtype=torch.long)
        input_ids = log_onehot_to_index(expected_sample)
        denoiser_logits = pipe.layout_dm.denoiser(
            input_ids=input_ids, timesteps=timestep_batch
        ).logits
        logits = denoiser_logits[:, :, :-1].permute(0, 2, 1)
        log_pred = torch.log_softmax(logits.double(), dim=1).float()
        mask_col = torch.full(
            (log_pred.shape[0], 1, log_pred.shape[-1]), -70.0, dtype=log_pred.dtype
        )
        model_log_prob = torch.cat((log_pred, mask_col), dim=1).clamp(-70.0, 0.0)

        if timestep_value in (30, 20, 10):
            x0_recon_ids = multinomial(
                model_log_prob.softmax(dim=1)
                .permute(0, 2, 1)
                .reshape(-1, model_log_prob.size(1)),
                1,
                generator=generator,
                device=torch.device("cpu"),
            ).reshape(1, -1)
            confidence = pipe.corrector.calc_confidence_score(
                x0_recon_ids,
                timestep_batch,
                padding_mask=x0_recon_ids == pipe.layout_dm.tokenizer.pad_token_id,
            )
            remask = select_tokens_to_remask(
                confidence,
                mask_ratio=timestep_value / 100.0,
                mode="thresh",
                threshold=0.7,
                temperature=1.0,
            )
            expected_sample = index_to_log_onehot(
                x0_recon_ids.masked_fill(
                    remask, pipe.layout_dm.tokenizer.mask_token_id
                ),
                pipe.layout_dm.scheduler.vocab_size,
            )
        else:
            model_log_prob = pipe.layout_dm.scheduler.q_posterior(
                model_log_prob, expected_sample, timestep_batch
            )
            ids = multinomial(
                model_log_prob.softmax(dim=1)
                .permute(0, 2, 1)
                .reshape(-1, model_log_prob.size(1)),
                1,
                generator=generator,
                device=torch.device("cpu"),
            ).reshape(1, -1)
            expected_sample = index_to_log_onehot(
                ids, pipe.layout_dm.scheduler.vocab_size
            )
        expected_trajectory.append(log_onehot_to_index(expected_sample).clone())

    actual = pipe(
        batch_size=1,
        generator=torch.Generator().manual_seed(17),
        num_inference_steps=100,
        sampling="random",
        corrector_steps=1,
        corrector_t_list=(10, 20, 30),
        corrector_mask_mode="thresh",
        corrector_mask_threshold=0.7,
        use_gumbel_noise=False,
        return_intermediates=True,
    )

    assert actual.trajectory is not None
    assert len(actual.trajectory) == 100
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(
            actual.trajectory, expected_trajectory, strict=True
        )
    )


def test_pipeline_requires_conditioned_bbox_and_labels():
    pipe = tiny_pipeline()

    with pytest.raises(ValueError, match="bbox and labels"):
        pipe(condition_type="label", labels=[[0]], num_inference_steps=1)


def test_pipeline_mask_ratio_clamps_timestep_range():
    pipe = tiny_pipeline()

    assert pipe._mask_ratio(torch.tensor([-1])) == 0.0
    assert pipe._mask_ratio(torch.tensor([999])) == 1.0
