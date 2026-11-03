from __future__ import annotations

import json
import math
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Literal, cast

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from PIL import Image, ImageFilter
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from collagediffusion import (
    CollageDiffusionPipeline,
    CollageLayer,
    compose_collage_layers,
    make_layer_noise_map,
)
from collagediffusion.pipeline_collagediffusion import (
    _CollageCrossAttentionProcessor,
    _find_token_span,
    _inject_textual_inversion_tokens,
    _prompt_token_spans,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        tokens = {
            "a red cat": [1, 2, 3],
            " red cat": [2, 3],
            "red cat": [2, 3],
            " blue dog": [5, 6],
            "blue dog": [5, 6],
            "a <cat> red cat": [1, 4, 2, 3],
            "a <cat> red cat, a blue dog": [1, 4, 2, 3, 7, 1, 5, 6],
            " <cat> red cat": [4, 2, 3],
            "<cat> red cat": [4, 2, 3],
        }.get(text, [])
        return [101, *tokens, 102] if add_special_tokens else tokens


class FakeProgress(AbstractContextManager["FakeProgress"]):
    def __enter__(self) -> FakeProgress:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self) -> None:
        return None


class FakeLatentDistribution:
    def sample(self, *, generator: torch.Generator | None) -> torch.Tensor:
        del generator
        return torch.zeros((1, 4, 8, 8))


class FakeVAE:
    config = SimpleNamespace(scaling_factor=1.0, block_out_channels=[1, 2, 3, 4])

    def encode(self, image: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=FakeLatentDistribution())

    def decode(
        self, latents: torch.Tensor, *, return_dict: bool
    ) -> tuple[torch.Tensor]:
        del return_dict
        return (
            torch.zeros(
                (latents.shape[0], 3, *[size * 8 for size in latents.shape[-2:]])
            ),
        )


class FakeScheduler:
    config = {"steps_offset": 0}
    order = 1

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        self.timesteps = torch.linspace(900, 100, steps, device=device).round()
        self.sigmas = torch.linspace(2.0, 0.1, steps + 1, device=device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        scale = timesteps.to(original_samples.dtype).flatten()[0] / 1000
        return original_samples + noise * scale

    def scale_model_input(
        self, latents: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        del timestep
        return latents

    def step(
        self,
        noise_prediction: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del timestep, kwargs
        return SimpleNamespace(prev_sample=latents - noise_prediction * 0.1)


class FakeUNet:
    dtype = torch.float32

    def __init__(self) -> None:
        self.attn_processors: dict[str, object] = {"down.attn2.processor": object()}
        self.attention_processor_sigmas: list[float] = []

    def set_attn_processor(self, processors: dict[str, object]) -> None:
        self.attn_processors = dict(processors)
        processors.clear()

    def __call__(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep, encoder_hidden_states
        self.attention_processor_sigmas.extend(
            float(processor.sigma)
            for processor in self.attn_processors.values()
            if isinstance(processor, _CollageCrossAttentionProcessor)
        )
        return SimpleNamespace(sample=torch.zeros_like(latents))


class FakeImageProcessor:
    def preprocess(
        self, image: Image.Image, *, height: int, width: int
    ) -> torch.Tensor:
        del image
        return torch.zeros((1, 3, height, width))

    def postprocess(
        self,
        image: torch.Tensor,
        *,
        output_type: str,
        do_denormalize: list[bool] | None,
    ) -> list[Image.Image] | np.ndarray:
        del do_denormalize
        if output_type == "np":
            return np.zeros(
                (image.shape[0], image.shape[-2], image.shape[-1], 3), dtype=np.float32
            )

        return [
            Image.new("RGB", (image.shape[-1], image.shape[-2]))
            for _ in range(image.shape[0])
        ]


class FakePipeline(CollageDiffusionPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def vae_scale_factor(self) -> int:
        return 8

    def encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        *,
        negative_prompt: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del prompt, negative_prompt
        positive = torch.ones((num_images_per_prompt, 5, 4), device=device)
        negative = torch.zeros_like(positive) if do_classifier_free_guidance else None
        return positive, negative

    def progress_bar(self, *, total: int) -> FakeProgress:
        del total
        return FakeProgress()

    def prepare_extra_step_kwargs(
        self, generator: torch.Generator | None, eta: float
    ) -> dict[str, object]:
        return {"generator": generator, "eta": eta}

    def run_safety_checker(
        self,
        image: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[bool] | None]:
        del device, dtype
        return image, None

    def maybe_free_model_hooks(self) -> None:
        return None


def make_pipeline() -> FakePipeline:
    pipeline = FakePipeline.__new__(FakePipeline)
    DiffusionPipeline.__init__(pipeline)
    pipeline.tokenizer = FakeTokenizer()
    pipeline.image_processor = FakeImageProcessor()
    pipeline.vae = FakeVAE()
    pipeline.unet = FakeUNet()
    pipeline.scheduler = FakeScheduler()
    pipeline.loaded_embeddings = []
    pipeline.load_textual_inversion = lambda path, *, token: (
        pipeline.loaded_embeddings.append((path, token))
    )
    return pipeline


def make_layer(
    image: Image.Image | None = None,
    *,
    prompt: str = "red cat",
    position: tuple[int, int] = (0, 0),
    noise_level: float = 0.4,
    positive_attention_weight: float = 0.0,
    negative_attention_weight: float = 0.0,
    textual_inversion_token: str | None = None,
) -> CollageLayer:
    return CollageLayer(
        image or Image.new("RGBA", (2, 2), (220, 30, 10, 255)),
        prompt,
        position,
        noise_level,
        positive_attention_weight,
        negative_attention_weight,
        textual_inversion_token,
    )


@pytest.mark.parametrize(
    (
        "image",
        "prompt",
        "position",
        "noise_level",
        "positive",
        "negative",
        "token",
        "error",
    ),
    [
        (None, "red cat", (0, 0), 0.4, 0.0, 0.0, None, TypeError),
        (Image.new("RGB", (2, 2)), "red cat", (0, 0), 0.4, 0.0, 0.0, None, ValueError),
        (Image.new("RGBA", (2, 2)), " ", (0, 0), 0.4, 0.0, 0.0, None, ValueError),
        (Image.new("RGBA", (2, 2)), "red cat", (0,), 0.4, 0.0, 0.0, None, TypeError),
        (Image.new("RGBA", (2, 2)), "red cat", (0, 0), 1.1, 0.0, 0.0, None, ValueError),
        (
            Image.new("RGBA", (2, 2)),
            "red cat",
            (0, 0),
            0.4,
            -1.0,
            0.0,
            None,
            ValueError,
        ),
        (
            Image.new("RGBA", (2, 2)),
            "red cat",
            (0, 0),
            0.4,
            0.0,
            math.inf,
            None,
            ValueError,
        ),
        (Image.new("RGBA", (2, 2)), "red cat", (0, 0), 0.4, 0.0, 0.0, " ", ValueError),
    ],
)
def test_layer_validation(
    image: Image.Image | None,
    prompt: str,
    position: tuple[int, ...],
    noise_level: float,
    positive: float,
    negative: float,
    token: str | None,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CollageLayer(
            cast(Image.Image, image),
            prompt,
            cast(tuple[int, int], position),
            noise_level,
            positive,
            negative,
            token,
        )


def test_compose_layers_matches_notebook_alpha_mask_paste_semantics() -> None:
    translucent = Image.new("RGBA", (2, 1), (255, 0, 0, 128))
    top = Image.new("RGBA", (2, 1), (0, 0, 255, 128))
    top.putpixel((1, 0), (0, 0, 255, 255))
    layers = [
        make_layer(translucent),
        make_layer(top, prompt="blue square"),
    ]

    collage, masks = compose_collage_layers(layers, width=2, height=1)

    assert collage.size == (2, 1)
    assert collage.getpixel((1, 0)) == (0, 0, 255)
    assert masks[0].dtype == np.float64
    assert masks[1].dtype == np.float64
    pasted_alpha = 64 / 255
    assert masks[0][0, 0] == pytest.approx(pasted_alpha * (1 - pasted_alpha))
    assert masks[0][0, 1] == 0
    assert masks[1][0, 0] == pytest.approx(pasted_alpha)
    assert masks[1][0, 1] == pytest.approx(1)


@pytest.mark.parametrize(
    ("width", "height", "background", "layers", "error"),
    [
        (0, 2, (0, 0, 0), [make_layer()], ValueError),
        (2, 2, (0, 0), [make_layer()], ValueError),
        (2, 2, (0, 0, 256), [make_layer()], ValueError),
        (2, 2, (0, 0, 0), [], ValueError),
    ],
)
def test_compose_layers_rejects_invalid_inputs(
    width: int,
    height: int,
    background: tuple[int, ...],
    layers: list[CollageLayer],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        compose_collage_layers(
            layers,
            width=width,
            height=height,
            background_color=cast(tuple[int, int, int], background),
        )


def test_noise_map_matches_pil_layer_blending_and_blur() -> None:
    masks = [
        np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32),
    ]
    layers = [
        make_layer(noise_level=0.2),
        make_layer(noise_level=0.8, prompt="blue square"),
    ]

    actual = make_layer_noise_map(masks, layers, strength=0.6, blur_radius=0)

    expected = Image.new("L", (2, 2), int(0.6 * 255))
    for mask, layer in zip(masks, layers, strict=True):
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        layer_image = Image.fromarray((mask * layer.noise_level * 255).astype(np.uint8))
        expected.paste(layer_image, (0, 0), mask_image)
    np.testing.assert_array_equal(
        actual, np.asarray(expected, dtype=np.float32) / 255.0
    )

    blurred = make_layer_noise_map(masks, layers, strength=0.6, blur_radius=1)
    expected_blur = expected.filter(ImageFilter.GaussianBlur(1))
    np.testing.assert_array_equal(
        blurred, np.asarray(expected_blur, dtype=np.float32) / 255.0
    )


@pytest.mark.parametrize(
    ("masks", "layers", "strength", "blur_radius"),
    [
        ([], [], 0.5, 0),
        ([np.ones((2, 2))], [], 0.5, 0),
        ([np.ones((2, 2))], [make_layer()], 0, 0),
        ([np.ones((2, 2))], [make_layer()], 0.5, -1),
        (
            [np.ones((2, 2)), np.ones((3, 2))],
            [make_layer(), make_layer(prompt="blue square")],
            0.5,
            0,
        ),
        ([np.full((2, 2), np.nan)], [make_layer()], 0.5, 0),
        ([np.full((2, 2), 2.0)], [make_layer()], 0.5, 0),
    ],
)
def test_noise_map_rejects_invalid_inputs(
    masks: list[np.ndarray],
    layers: list[CollageLayer],
    strength: float,
    blur_radius: float,
) -> None:
    with pytest.raises(ValueError):
        make_layer_noise_map(masks, layers, strength=strength, blur_radius=blur_radius)


def test_prompt_layer_spans_include_textual_inversion_token() -> None:
    layer = make_layer(textual_inversion_token="<cat>")

    assert _prompt_token_spans(FakeTokenizer(), "a <cat> red cat", [layer]) == (
        (2, 3, 4),
    )
    assert _prompt_token_spans(FakeTokenizer(), "a red cat", [make_layer()]) == (
        (2, 3),
    )


def test_prompt_layer_spans_match_notebook_shift_with_textual_inversion() -> None:
    layers = [
        make_layer(textual_inversion_token="<cat>"),
        make_layer(prompt="blue dog"),
    ]

    assert _prompt_token_spans(
        FakeTokenizer(), "a <cat> red cat, a blue dog", layers
    ) == ((2, 3, 4), (6, 7, 8))


@pytest.mark.parametrize(
    ("prompt_ids", "layer_ids"),
    [([], [1]), ([1, 2], []), ([101, 1, 2], [3])],
)
def test_find_token_span_rejects_missing_spans(
    prompt_ids: list[int], layer_ids: list[int]
) -> None:
    with pytest.raises(ValueError):
        _find_token_span(prompt_ids, layer_ids)


def test_textual_inversion_prompt_injection_validates_requests() -> None:
    layer = make_layer(textual_inversion_token="<cat>")
    assert (
        _inject_textual_inversion_tokens("a red cat", [layer], {"<cat>": "cat.pt"})
        == "a <cat> red cat"
    )

    with pytest.raises(ValueError, match="no matching layer"):
        _inject_textual_inversion_tokens("a red cat", [layer], {"<dog>": "dog.pt"})
    with pytest.raises(ValueError, match="exactly once"):
        _inject_textual_inversion_tokens(
            "red cat and red cat", [layer], {"<cat>": "cat.pt"}
        )


def test_attention_guidance_changes_only_conditional_token_scores() -> None:
    layers = [make_layer(positive_attention_weight=1.0, negative_attention_weight=2.0)]
    processor = _CollageCrossAttentionProcessor(
        visible_masks=[np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)],
        token_spans=[(2,)],
        layers=layers,
        output_size=(2, 2),
        batch_size=1,
        classifier_free_guidance=True,
    )
    processor.sigma = 0.5
    scores = torch.zeros((2, 2, 4, 5))
    scores[:, :, :, 2] = 2.0

    processor._apply_layer_guidance(scores)

    assert torch.equal(scores[0, :, :, 2], torch.full_like(scores[0, :, :, 2], 2.0))
    assert torch.all(scores[1, :, [0, 2], 2] > 2.0)
    assert torch.all(scores[1, :, [1, 3], 2] < 2.0)
    assert torch.equal(scores[:, :, :, 1], torch.zeros_like(scores[:, :, :, 1]))


def test_attention_guidance_uses_antialiased_mask_resize() -> None:
    visible_mask = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    processor = _CollageCrossAttentionProcessor(
        visible_masks=[visible_mask],
        token_spans=[(2,)],
        layers=[make_layer(positive_attention_weight=1.0)],
        output_size=(2, 2),
        batch_size=1,
        classifier_free_guidance=False,
    )
    processor.sigma = 0.5
    scores = torch.zeros((1, 1, 4, 3))
    scores[:, :, :, 2] = 2.0
    expected_mask = torch.nn.functional.interpolate(
        torch.as_tensor(visible_mask)[None, None],
        size=(2, 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0].reshape(4)
    expected_scores = scores.clone()
    expected_scores[0, 0, :, 2] += (
        expected_mask * math.log(math.log(processor.sigma + 1.01) + 1.0) * 2.0
    )

    processor._apply_layer_guidance(scores)

    torch.testing.assert_close(scores, expected_scores)


def test_attention_processor_returns_projected_hidden_states() -> None:
    attention = SimpleNamespace(
        spatial_norm=None,
        group_norm=None,
        heads=2,
        norm_cross=False,
        norm_q=None,
        norm_k=None,
        upcast_attention=False,
        upcast_softmax=True,
        scale=0.5,
        residual_connection=False,
        rescale_output_factor=1.0,
        to_q=torch.nn.Linear(4, 4),
        to_k=torch.nn.Linear(4, 4),
        to_v=torch.nn.Linear(4, 4),
        to_out=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()]),
    )
    processor = _CollageCrossAttentionProcessor(
        visible_masks=[np.ones((2, 2), dtype=np.float32)],
        token_spans=[(1,)],
        layers=[make_layer(positive_attention_weight=0.5)],
        output_size=(2, 2),
        batch_size=1,
        classifier_free_guidance=False,
    )

    result = processor(
        cast(Attention, attention),
        torch.ones((1, 4, 4)),
        encoder_hidden_states=torch.ones((1, 5, 4)),
    )

    assert result.shape == (1, 4, 4)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("invalid_case", "error"),
    [
        ("empty-prompt", ValueError),
        ("empty-layers", ValueError),
        ("invalid-layer", TypeError),
        ("strength", ValueError),
        ("steps", ValueError),
        ("guidance", ValueError),
        ("height-zero", ValueError),
        ("height-misaligned", ValueError),
        ("output-type", ValueError),
    ],
)
def test_pipeline_rejects_invalid_inputs(
    invalid_case: Literal[
        "empty-prompt",
        "empty-layers",
        "invalid-layer",
        "strength",
        "steps",
        "guidance",
        "height-zero",
        "height-misaligned",
        "output-type",
    ],
    error: type[Exception],
) -> None:
    pipeline = make_pipeline()

    with pytest.raises(error):
        if invalid_case == "empty-prompt":
            pipeline(" ", [make_layer()])
        elif invalid_case == "empty-layers":
            pipeline("red cat", [])
        elif invalid_case == "invalid-layer":
            pipeline("red cat", cast(list[CollageLayer], [object()]))
        elif invalid_case == "strength":
            pipeline("red cat", [make_layer()], strength=0)
        elif invalid_case == "steps":
            pipeline("red cat", [make_layer()], num_inference_steps=0)
        elif invalid_case == "guidance":
            pipeline("red cat", [make_layer()], guidance_scale=math.inf)
        elif invalid_case == "height-zero":
            pipeline("red cat", [make_layer()], height=0)
        elif invalid_case == "height-misaligned":
            pipeline("red cat", [make_layer()], height=65)
        else:
            pipeline(
                "red cat",
                [make_layer()],
                output_type=cast(Literal["pil", "np"], "tensor"),
            )


def test_pipeline_generates_batch_and_restores_attention_processors() -> None:
    pipeline = make_pipeline()
    original_processors = pipeline.unet.attn_processors.copy()
    latents: list[torch.Tensor] = []

    output = pipeline(
        "a red cat",
        [make_layer(positive_attention_weight=0.5)],
        strength=0.5,
        num_inference_steps=2,
        guidance_scale=2.0,
        num_images_per_prompt=2,
        seed=0,
        height=64,
        width=64,
        step_callback=lambda step, timestep, latent: latents.append(latent.clone()),
    )

    assert isinstance(output, StableDiffusionPipelineOutput)
    assert len(output.images) == 2
    assert pipeline.num_timesteps == 1
    assert len(latents) == 1
    assert latents[0].shape == (2, 4, 8, 8)
    assert pipeline.unet.attention_processor_sigmas == [2.0]
    assert pipeline.num_timesteps == 1
    assert pipeline.unet.attn_processors == original_processors


def test_pipeline_loads_textual_inversion_and_returns_numpy_tuple() -> None:
    pipeline = make_pipeline()
    images, flags = pipeline(
        "a red cat",
        [make_layer(textual_inversion_token="<cat>")],
        strength=1.0,
        num_inference_steps=1,
        guidance_scale=1.0,
        seed=3,
        textual_inversion_embeddings={"<cat>": "cat.pt"},
        height=64,
        width=64,
        noise_blur=0,
        output_type="np",
        return_dict=False,
    )

    assert isinstance(images, np.ndarray)
    assert images.shape == (1, 64, 64, 3)
    assert flags is None
    assert pipeline.loaded_embeddings == [("cat.pt", "<cat>")]


def test_pipeline_generator_takes_precedence_over_seed() -> None:
    def sample(seed: int) -> torch.Tensor:
        pipeline = make_pipeline()
        latents: list[torch.Tensor] = []
        pipeline(
            "a red cat",
            [make_layer()],
            strength=0.5,
            num_inference_steps=2,
            seed=seed,
            generator=torch.Generator().manual_seed(12),
            height=64,
            width=64,
            step_callback=lambda step, timestep, latent: latents.append(latent.clone()),
        )
        return latents[0]

    assert torch.equal(sample(0), sample(999))


def test_pipeline_save_and_load_round_trip_with_tiny_components(tmp_path) -> None:
    from diffusers import (
        AutoencoderKL,
        EulerAncestralDiscreteScheduler,
        UNet2DConditionModel,
    )

    vocabulary = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "<|pad|>": 2,
        "<|unk|>": 3,
    }
    vocabulary_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(
        vocab_file=str(vocabulary_path),
        merges_file=str(merges_path),
        unk_token="<|unk|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=77,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=32,
    )
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=8,
    )
    scheduler = EulerAncestralDiscreteScheduler(num_train_timesteps=20)
    pipeline = CollageDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipeline.save_pretrained(tmp_path / "pipeline")
    reloaded = CollageDiffusionPipeline.from_pretrained(
        tmp_path / "pipeline", local_files_only=True
    )

    assert isinstance(reloaded, CollageDiffusionPipeline)
    assert isinstance(reloaded.scheduler, EulerAncestralDiscreteScheduler)
    assert reloaded.vae.config.latent_channels == 4
