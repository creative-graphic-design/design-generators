from __future__ import annotations

from typing import cast

import pytest
import sawna.pipeline_sawna as pipeline_module
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion as sd_pipeline
import diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl as sdxl_pipeline
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from sawna import SAWNAStableDiffusionPipeline, SAWNAStableDiffusionXLPipeline
from sawna.pipeline_sawna import SAWNAHubPipeline
from sawna.processing_sawna import apply_sawna_noise
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection

StableDiffusionPipeline = sd_pipeline.StableDiffusionPipeline
StableDiffusionXLPipeline = sdxl_pipeline.StableDiffusionXLPipeline
SAWNAPipeline = SAWNAStableDiffusionPipeline | SAWNAStableDiffusionXLPipeline
BasePipeline = StableDiffusionPipeline | StableDiffusionXLPipeline


@pytest.mark.parametrize(
    ("pipeline_class", "base_class"),
    [
        (SAWNAStableDiffusionPipeline, StableDiffusionPipeline),
        (SAWNAStableDiffusionXLPipeline, StableDiffusionXLPipeline),
    ],
)
def test_call_blends_raw_noise_before_scheduler_scaling(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_class: type[SAWNAPipeline],
    base_class: type[BasePipeline],
) -> None:
    raw_noise = (
        torch.linspace(-1, 1, 64 * 64).reshape(1, 1, 64, 64).expand(1, 4, -1, -1)
    )
    generator = torch.Generator().manual_seed(1234)
    received: list[
        tuple[str | list[str] | None, torch.Generator | list[torch.Generator] | None]
    ] = []

    def base_call(
        self: BasePipeline,
        prompt: str | list[str] | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        received.append((prompt, generator))
        return cast(SAWNAPipeline, self).prepare_latents(
            batch_size=1,
            num_channels_latents=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
            latents=latents,
        )

    monkeypatch.setattr(base_class, "__call__", base_call)
    pipeline = cast(SAWNAPipeline, object.__new__(pipeline_class))
    pipeline.vae_scale_factor = 1
    pipeline.scheduler = cast(
        KarrasDiffusionSchedulers,
        type("Scheduler", (), {"init_noise_sigma": 2.0})(),
    )
    boxes = [(0.3, 0.3, 0.7, 0.7)]

    result = pipeline(
        "reserved region",
        bounding_boxes=boxes,
        generator=generator,
        latents=raw_noise,
    )

    expected_noise = apply_sawna_noise(raw_noise, boxes) * 2
    assert torch.equal(cast(torch.Tensor, result), expected_noise)
    assert received == [("reserved region", generator)]


def test_call_without_boxes_preserves_standard_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_noise = torch.zeros((1, 4, 64, 64))
    monkeypatch.setattr(
        pipeline_module, "randn_tensor", lambda *args, **kwargs: raw_noise
    )

    def base_call(
        self: StableDiffusionPipeline,
        prompt: str | list[str] | None = None,
    ) -> torch.Tensor:
        return cast(SAWNAStableDiffusionPipeline, self).prepare_latents(
            batch_size=1,
            num_channels_latents=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
            generator=None,
        )

    monkeypatch.setattr(StableDiffusionPipeline, "__call__", base_call)
    pipeline = cast(
        SAWNAStableDiffusionPipeline,
        object.__new__(SAWNAStableDiffusionPipeline),
    )
    pipeline.vae_scale_factor = 1
    pipeline.scheduler = cast(
        KarrasDiffusionSchedulers,
        type("Scheduler", (), {"init_noise_sigma": 2.0})(),
    )

    result = cast(torch.Tensor, pipeline("ordinary generation"))
    assert torch.equal(result, raw_noise * 2)


def test_call_resets_scoped_parameters_after_base_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base_call(
        self: StableDiffusionPipeline,
        prompt: str | list[str] | None = None,
    ) -> None:
        raise RuntimeError("base pipeline failure")

    monkeypatch.setattr(StableDiffusionPipeline, "__call__", base_call)
    pipeline = cast(
        SAWNAStableDiffusionPipeline,
        object.__new__(SAWNAStableDiffusionPipeline),
    )

    with pytest.raises(RuntimeError, match="base pipeline failure"):
        pipeline("reserved region", bounding_boxes=[(0.3, 0.3, 0.7, 0.7)])

    assert pipeline_module._SAWNA_CALL_PARAMETERS.get() is None


def test_hub_pipeline_rejects_incomplete_sdxl_components() -> None:
    with pytest.raises(ValueError, match="both second text-encoder components"):
        SAWNAHubPipeline.__new__(
            SAWNAHubPipeline,
            vae=cast(AutoencoderKL, object()),
            text_encoder=cast(CLIPTextModel, object()),
            tokenizer=cast(CLIPTokenizer, object()),
            unet=cast(UNet2DConditionModel, object()),
            scheduler=cast(KarrasDiffusionSchedulers, object()),
            text_encoder_2=cast(CLIPTextModelWithProjection, object()),
        )


def test_hub_pipeline_exposes_components_to_diffusers_loader() -> None:
    expected_modules, optional_kwargs = DiffusionPipeline._get_signature_keys(
        SAWNAHubPipeline
    )

    assert expected_modules == ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    assert set(optional_kwargs) == {
        "add_watermarker",
        "feature_extractor",
        "force_zeros_for_empty_prompt",
        "image_encoder",
        "requires_safety_checker",
        "safety_checker",
        "text_encoder_2",
        "tokenizer_2",
    }


def test_hub_pipeline_selects_matching_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SAWNAHubPipeline, "stable_pipeline_class", lambda **kwargs: "stable"
    )
    monkeypatch.setattr(SAWNAHubPipeline, "sdxl_pipeline_class", lambda **kwargs: "xl")
    components = cast(
        tuple[
            AutoencoderKL,
            CLIPTextModel,
            CLIPTokenizer,
            UNet2DConditionModel,
            KarrasDiffusionSchedulers,
        ],
        (None, None, None, None, None),
    )
    stable = cast(str, SAWNAHubPipeline(*components))
    xl = cast(
        str,
        SAWNAHubPipeline(
            *components,
            text_encoder_2=cast(CLIPTextModelWithProjection, object()),
            tokenizer_2=cast(CLIPTokenizer, object()),
        ),
    )

    assert (stable, xl) == ("stable", "xl")
