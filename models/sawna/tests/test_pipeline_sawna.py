from __future__ import annotations

from typing import cast

import pytest
import sawna.pipeline_sawna as pipeline_module
import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from sawna import SAWNAStableDiffusionPipeline, SAWNAStableDiffusionXLPipeline
from sawna.pipeline_sawna import SAWNAHubPipeline
from sawna.processing_sawna import apply_sawna_noise
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

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
    received: dict[str, object] = {}

    def base_call(
        self: BasePipeline,
        prompt: str | list[str] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        received["prompt"] = prompt
        received["generator"] = kwargs.get("generator")
        return self.prepare_latents(
            batch_size=1,
            num_channels_latents=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
            generator=cast(torch.Generator, kwargs.get("generator")),
            latents=cast(torch.Tensor | None, kwargs.get("latents")),
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

    assert torch.equal(result, apply_sawna_noise(raw_noise, boxes) * 2)
    assert received == {"prompt": "reserved region", "generator": generator}


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
        **kwargs: object,
    ) -> torch.Tensor:
        return self.prepare_latents(
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

    assert torch.equal(pipeline("ordinary generation"), raw_noise * 2)


def test_call_resets_scoped_parameters_after_base_pipeline_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base_call(
        self: StableDiffusionPipeline,
        prompt: str | list[str] | None = None,
        **kwargs: object,
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
            scheduler=cast(object, object()),
            text_encoder_2=cast(CLIPTextModelWithProjection, object()),
        )


def test_hub_pipeline_selects_matching_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SAWNAHubPipeline, "stable_pipeline_class", lambda **kwargs: ("stable", kwargs)
    )
    monkeypatch.setattr(
        SAWNAHubPipeline, "sdxl_pipeline_class", lambda **kwargs: ("xl", kwargs)
    )
    components = {
        "vae": cast(AutoencoderKL, object()),
        "text_encoder": cast(CLIPTextModel, object()),
        "tokenizer": cast(CLIPTokenizer, object()),
        "unet": cast(UNet2DConditionModel, object()),
        "scheduler": cast(KarrasDiffusionSchedulers, object()),
    }

    stable = cast(
        tuple[str, dict[str, object]],
        SAWNAHubPipeline.__new__(SAWNAHubPipeline, **components),
    )
    xl = cast(
        tuple[str, dict[str, object]],
        SAWNAHubPipeline.__new__(
            SAWNAHubPipeline,
            **components,
            text_encoder_2=cast(CLIPTextModelWithProjection, object()),
            tokenizer_2=cast(CLIPTokenizer, object()),
        ),
    )

    assert stable[0] == "stable"
    assert xl[0] == "xl"
