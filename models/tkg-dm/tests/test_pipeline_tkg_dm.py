from __future__ import annotations

import importlib
from typing import cast

import pytest
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from tkg_dm import (
    DEFAULT_SD15_TARGET_SHIFT,
    DEFAULT_SDXL_TARGET_SHIFT,
    TKGDMHubPipeline,
    TKGStableDiffusionPipeline,
    TKGStableDiffusionXLPipeline,
    apply_tkg_noise,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


@pytest.mark.parametrize(
    ("pipeline_class", "base_class", "target_shift"),
    [
        (
            TKGStableDiffusionPipeline,
            StableDiffusionPipeline,
            DEFAULT_SD15_TARGET_SHIFT,
        ),
        (
            TKGStableDiffusionXLPipeline,
            StableDiffusionXLPipeline,
            DEFAULT_SDXL_TARGET_SHIFT,
        ),
    ],
)
def test_prepare_latents_preserves_caller_tensor_and_applies_family_default(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_class: type[TKGStableDiffusionPipeline | TKGStableDiffusionXLPipeline],
    base_class: type[StableDiffusionPipeline | StableDiffusionXLPipeline],
    target_shift: float,
) -> None:
    channel_noise = torch.linspace(-1, 1, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    supplied_latents = channel_noise.expand(1, 4, -1, -1).clone()
    prepared_latents = supplied_latents * 2
    received_latents: torch.Tensor | None = None

    def prepare_base_latents(
        self: StableDiffusionPipeline | StableDiffusionXLPipeline,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device | str,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal received_latents
        del (
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
        )
        received_latents = latents
        return prepared_latents

    monkeypatch.setattr(base_class, "prepare_latents", prepare_base_latents)
    pipeline = pipeline_class.__new__(pipeline_class)

    actual = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=4,
        height=8,
        width=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
        latents=supplied_latents,
    )

    expected = apply_tkg_noise(prepared_latents, target_shift=target_shift)
    torch.testing.assert_close(actual, expected)
    assert received_latents is supplied_latents


def test_stable_diffusion_pipeline_requires_ddim_scheduler() -> None:
    with pytest.raises(TypeError, match="requires a DDIMScheduler"):
        TKGStableDiffusionPipeline(
            vae=cast(AutoencoderKL, None),
            text_encoder=cast(CLIPTextModel, None),
            tokenizer=cast(CLIPTokenizer, None),
            unet=cast(UNet2DConditionModel, None),
            scheduler=cast(KarrasDiffusionSchedulers, object()),
            safety_checker=cast(StableDiffusionSafetyChecker | None, None),
            feature_extractor=cast(CLIPImageProcessor | None, None),
        )


def test_sdxl_pipeline_requires_ddim_scheduler() -> None:
    with pytest.raises(TypeError, match="requires a DDIMScheduler"):
        TKGStableDiffusionXLPipeline(
            vae=cast(AutoencoderKL, None),
            text_encoder=cast(CLIPTextModel, None),
            text_encoder_2=cast(CLIPTextModelWithProjection, None),
            tokenizer=cast(CLIPTokenizer, None),
            tokenizer_2=cast(CLIPTokenizer, None),
            unet=cast(UNet2DConditionModel, None),
            scheduler=cast(KarrasDiffusionSchedulers, object()),
        )


@pytest.mark.parametrize("sdxl", [False, True])
def test_hub_pipeline_routes_to_the_matching_model_family(
    monkeypatch: pytest.MonkeyPatch,
    sdxl: bool,
) -> None:
    class PipelineProbe:
        def __init__(self, **components: object) -> None:
            self.components = components

    pipeline_module = importlib.import_module("tkg_dm.pipeline_tkg_dm")
    monkeypatch.setattr(pipeline_module, "TKGStableDiffusionPipeline", PipelineProbe)
    monkeypatch.setattr(
        pipeline_module,
        "TKGStableDiffusionXLPipeline",
        PipelineProbe,
    )
    components: dict[str, object] = {
        "vae": object(),
        "text_encoder": object(),
        "tokenizer": object(),
        "unet": object(),
        "scheduler": object(),
    }
    if sdxl:
        components["text_encoder_2"] = object()
        components["tokenizer_2"] = object()

    pipeline = TKGDMHubPipeline.__new__(TKGDMHubPipeline, **components)

    assert isinstance(pipeline, PipelineProbe)
    if sdxl:
        assert "text_encoder_2" in pipeline.components
    else:
        assert "text_encoder_2" not in pipeline.components


def test_hub_pipeline_rejects_a_partial_sdxl_text_encoder_pair() -> None:
    with pytest.raises(ValueError, match="both second text-encoder components"):
        TKGDMHubPipeline.__new__(
            TKGDMHubPipeline,
            vae=object(),
            text_encoder=object(),
            tokenizer=object(),
            unet=object(),
            scheduler=object(),
            text_encoder_2=object(),
            tokenizer_2=None,
        )


def test_hub_pipeline_init_exposes_component_parameters_to_diffusers_loader() -> None:
    pipeline = object.__new__(TKGDMHubPipeline)

    TKGDMHubPipeline.__init__(
        pipeline,
        vae=cast(AutoencoderKL, object()),
        text_encoder=cast(CLIPTextModel, object()),
        tokenizer=cast(CLIPTokenizer, object()),
        unet=cast(UNet2DConditionModel, object()),
        scheduler=cast(KarrasDiffusionSchedulers, object()),
    )

    assert isinstance(pipeline, TKGDMHubPipeline)
