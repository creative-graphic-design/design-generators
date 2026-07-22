import json
import os
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from laygen.schedulers.continuous import get_layousyn_beta_schedule
from layousyn import LayouSynPipeline


def _parity_paths() -> tuple[Path, Path]:
    reference_dir = Path(
        os.environ.get("LAYOUSYN_REFERENCE_DIR", "/tmp/layousyn-reference")
    )
    converted_dir = Path(
        os.environ.get("LAYOUSYN_CONVERTED_DIR", "/tmp/layousyn-converted")
    )
    missing = [
        path
        for path in (
            reference_dir / "inputs.json",
            reference_dir / "inputs.pt",
            reference_dir / "reference.pt",
            converted_dir / "model" / "model_config.json",
        )
        if not path.exists()
    ]
    if missing:
        skip_or_fail_vendor_parity(
            "LayouSyn vendor parity requires generated references and a converted checkpoint: "
            + ", ".join(str(path) for path in missing),
            missing_paths=missing,
            regeneration_hint=(
                "run models/layousyn/scripts/generate_reference_outputs.py and "
                "models/layousyn/scripts/convert_original_checkpoint.py"
            ),
        )
    return reference_dir, converted_dir


def _device() -> torch.device:
    if not torch.cuda.is_available():
        skip_or_fail_vendor_parity(
            "LayouSyn vendor parity requires CUDA",
            missing_paths=["CUDA device"],
            regeneration_hint="rerun on a CUDA-enabled host with generated LayouSyn parity assets",
        )
    return torch.device("cuda")


@pytest.fixture(scope="module")
def parity_assets() -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    LayouSynPipeline,
    torch.device,
]:
    reference_dir, converted_dir = _parity_paths()
    device = _device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    metadata = cast(
        dict[str, object], json.loads((reference_dir / "inputs.json").read_text())
    )
    inputs = torch.load(reference_dir / "inputs.pt", map_location=device)
    reference = torch.load(reference_dir / "reference.pt", map_location=device)
    pipe = LayouSynPipeline.from_pretrained(converted_dir).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.model.eval()
    return metadata, inputs, reference, pipe, device


@pytest.mark.vendor_parity
def test_layousyn_alpha_scale_and_respacing_match_vendor(
    parity_assets: tuple[
        dict[str, object],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        LayouSynPipeline,
        torch.device,
    ],
) -> None:
    metadata, _inputs, reference, pipe, device = parity_assets
    pipe.scheduler.set_timesteps(
        cast(int, metadata["num_sampling_steps"]), device=device
    )
    assert torch.equal(pipe.scheduler.timestep_map, reference["timestep_map"])
    helper_betas = get_layousyn_beta_schedule(
        cast(Literal["linear", "squaredcos_cap_v2"], metadata["noise_schedule"]),
        cast(int, metadata["diffusion_steps"]),
        alpha_scale=cast(float, metadata["alpha_scale"]),
    )
    assert torch.allclose(
        helper_betas,
        reference["original_betas"].cpu().double(),
        atol=0,
        rtol=0,
    )
    assert torch.allclose(
        pipe.scheduler.original_betas.cpu().double(),
        reference["original_betas"].cpu().double(),
        atol=0,
        rtol=0,
    )
    assert torch.allclose(
        pipe.scheduler.betas.cpu().double(),
        reference["respaced_betas"].cpu().double(),
        atol=0,
        rtol=0,
    )


@pytest.mark.vendor_parity
def test_layousyn_denoiser_logits_match_vendor(
    parity_assets: tuple[
        dict[str, object],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        LayouSynPipeline,
        torch.device,
    ],
) -> None:
    metadata, inputs, reference, pipe, _device = parity_assets
    with torch.no_grad():
        logits = pipe.model.forward_with_cfg(
            inputs["initial_sample"],
            inputs["model_timestep"],
            x_padding_mask=inputs["concept_padding_mask"],
            aspect_ratio=inputs["aspect_ratio"],
            concept_embeds=inputs["concept_embeds"],
            caption_embeds=inputs["caption_embeds"],
            caption_padding_mask=inputs["caption_padding_mask"],
            guidance_scale=cast(float, metadata["cfg_scale"]),
        )
    assert torch.allclose(logits, reference["denoiser_logits"], atol=0, rtol=0)


@pytest.mark.vendor_parity
def test_layousyn_scheduler_first_step_matches_vendor(
    parity_assets: tuple[
        dict[str, object],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        LayouSynPipeline,
        torch.device,
    ],
) -> None:
    metadata, inputs, reference, pipe, device = parity_assets
    pipe.scheduler.set_timesteps(
        cast(int, metadata["num_sampling_steps"]), device=device
    )
    with torch.no_grad():
        step = pipe.scheduler.step(
            reference["denoiser_logits"],
            inputs["scheduler_timestep"],
            inputs["initial_sample"],
            sampling_type="ddim",
            clip_denoised=False,
        )
    assert torch.allclose(
        step.pred_original_sample,
        reference["first_pred_xstart"],
        atol=0,
        rtol=0,
    )
    assert torch.allclose(
        step.prev_sample, reference["first_prev_sample"], atol=0, rtol=0
    )


@pytest.mark.vendor_parity
def test_layousyn_full_sample_matches_vendor(
    parity_assets: tuple[
        dict[str, object],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        LayouSynPipeline,
        torch.device,
    ],
) -> None:
    metadata, inputs, reference, pipe, device = parity_assets
    generator = torch.Generator(device=device).manual_seed(cast(int, metadata["seed"]))
    with torch.no_grad():
        output = pipe(
            prompt=cast(str, metadata["caption"]),
            labels=[cast(list[str], metadata["concepts"])],
            caption_embeds=inputs["pipeline_caption_embeds"],
            caption_padding_mask=inputs["pipeline_caption_padding_mask"],
            concept_embeds=inputs["pipeline_concept_embeds"],
            aspect_ratio=inputs["pipeline_aspect_ratio"],
            num_inference_steps=cast(int, metadata["num_sampling_steps"]),
            guidance_scale=cast(float, metadata["cfg_scale"]),
            sampling_type="ddim",
            generator=generator,
        )
    output = cast(LayoutGenerationOutput, output)
    assert torch.allclose(
        output.bbox.to(device), reference["public_bbox"], atol=0, rtol=0
    )
