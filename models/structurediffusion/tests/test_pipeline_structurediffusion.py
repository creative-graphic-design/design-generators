from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from jaxtyping import Float

from structurediffusion.modeling_structurediffusion import StructuredAttentionProcessor
from structurediffusion.pipeline_structurediffusion import (
    ParserType,
    StructureDiffusionPipeline,
    normalize_parser_type,
)
from structurediffusion.processing_structurediffusion import NounPhrase


class WordTokenizer:
    model_max_length = 9

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()


class OriginalProcessor:
    def __call__(self) -> None:
        return None


class FakeUNet:
    def __init__(self, processors: dict[str, object]) -> None:
        self._processors: dict[str, object] = processors

    @property
    def attn_processors(self) -> dict[str, object]:
        return self._processors

    def set_attn_processor(self, processors: dict[str, object]) -> None:
        self._processors = processors


def _pipeline() -> StructureDiffusionPipeline:
    pipe = StructureDiffusionPipeline.__new__(StructureDiffusionPipeline)
    pipe.__dict__["config"] = {}
    pipe.tokenizer = WordTokenizer()
    pipe.unet = FakeUNet(
        {
            "block.attn2.processor": OriginalProcessor(),
            "block.attn1.processor": OriginalProcessor(),
        }
    )
    return pipe


def _encode_prompt(
    self: StructureDiffusionPipeline,
    prompt: str | list[str] | None,
    device: torch.device,
    num_images_per_prompt: int,
    do_classifier_free_guidance: bool,
    negative_prompt: str | list[str] | None = None,
    prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
    negative_prompt_embeds: Float[torch.Tensor, "batch tokens dim"] | None = None,
    lora_scale: float | None = None,
    clip_skip: int | None = None,
) -> tuple[
    Float[torch.Tensor, "batch tokens dim"],
    Float[torch.Tensor, "batch tokens dim"] | None,
]:
    del self, device, negative_prompt, negative_prompt_embeds, lora_scale, clip_skip
    if prompt_embeds is not None:
        encoded = prompt_embeds
    else:
        if prompt is None:
            raise ValueError("Prompt text or prompt embeddings are required.")
        prompts = [prompt] if isinstance(prompt, str) else prompt
        encoded = torch.stack(
            [torch.full((9, 4), float(index + 1)) for index, _ in enumerate(prompts)]
        )
    encoded = encoded.repeat_interleave(num_images_per_prompt, dim=0)
    negative = torch.zeros_like(encoded) if do_classifier_free_guidance else None
    return encoded, negative


def test_normalize_parser_type_accepts_supported_spellings() -> None:
    assert normalize_parser_type("constituency") is ParserType.constituency
    assert normalize_parser_type("scene-graph") is ParserType.scene_graph
    assert normalize_parser_type(ParserType.scene_graph) is ParserType.scene_graph


def test_normalize_parser_type_rejects_unknown_parser() -> None:
    with pytest.raises(ValueError, match="choose one of"):
        normalize_parser_type("dependency")


@pytest.mark.parametrize(
    "parser_type", [ParserType.constituency, ParserType.scene_graph]
)
def test_parse_prompt_selects_requested_parser(
    monkeypatch: pytest.MonkeyPatch,
    parser_type: ParserType,
) -> None:
    import structurediffusion.pipeline_structurediffusion as module

    phrase = NounPhrase("red car", 2, 9)
    monkeypatch.setattr(module, "constituency_noun_phrases", lambda prompt: [phrase])
    monkeypatch.setattr(module, "scene_graph_noun_phrases", lambda prompt: [phrase])
    assert StructureDiffusionPipeline._parse_prompt("a red car", parser_type) == [
        phrase
    ]


def test_encode_prompt_passes_through_when_not_inside_structured_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import StableDiffusionPipeline

    monkeypatch.setattr(StableDiffusionPipeline, "encode_prompt", _encode_prompt)
    pipe = _pipeline()
    encoded, negative = pipe.encode_prompt("a car", torch.device("cpu"), 1, False)
    assert encoded.shape == (1, 9, 4)
    assert negative is None


def test_call_installs_conditioning_and_restores_processors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import StableDiffusionPipeline

    import structurediffusion.pipeline_structurediffusion as module

    monkeypatch.setattr(StableDiffusionPipeline, "encode_prompt", _encode_prompt)
    monkeypatch.setattr(
        module,
        "constituency_noun_phrases",
        lambda prompt: [
            NounPhrase("a red car", 0, 9),
            NounPhrase("a white sheep", 14, 27),
        ],
    )
    original_call = object()
    pipe = _pipeline()
    original_processors = dict(pipe.unet.attn_processors)

    def base_call(self: StructureDiffusionPipeline, **kwargs: object) -> object:
        prompt = cast(str | list[str], kwargs["prompt"])
        assert prompt == "a red car and a white sheep"
        self.encode_prompt(
            prompt=prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=2,
            do_classifier_free_guidance=True,
        )
        structured = cast(
            StructuredAttentionProcessor,
            self.unet.attn_processors["block.attn2.processor"],
        )
        assert structured.conditioning is not None
        assert len(structured.conditioning.keys) == 1
        assert len(structured.conditioning.values) == 3
        assert structured.conditioning.keys[0].shape == (2, 9, 4)
        assert structured.conditioning.do_classifier_free_guidance
        return original_call

    monkeypatch.setattr(StableDiffusionPipeline, "__call__", base_call)
    result = pipe("A red car and a white sheep.", num_images_per_prompt=2)

    assert result is original_call
    assert pipe.unet.attn_processors == original_processors
    assert pipe.attention_maps == ()


def test_conjunction_uses_phrase_keys_and_one_combined_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import StableDiffusionPipeline

    import structurediffusion.pipeline_structurediffusion as module

    monkeypatch.setattr(StableDiffusionPipeline, "encode_prompt", _encode_prompt)
    monkeypatch.setattr(
        module,
        "constituency_noun_phrases",
        lambda prompt: [NounPhrase("red car", 2, 9)],
    )
    pipe = _pipeline()

    def base_call(self: StructureDiffusionPipeline, **kwargs: object) -> str:
        prompt = cast(str | list[str], kwargs["prompt"])
        self.encode_prompt(
            prompt=prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        structured = cast(
            StructuredAttentionProcessor,
            self.unet.attn_processors["block.attn2.processor"],
        )
        assert structured.conditioning is not None
        assert len(structured.conditioning.keys) == 2
        assert len(structured.conditioning.values) == 1
        return "ok"

    monkeypatch.setattr(StableDiffusionPipeline, "__call__", base_call)
    assert pipe("a red car", conjunction=True) == "ok"


def test_call_rejects_empty_prompt_and_no_cross_attention_layers() -> None:
    pipe = _pipeline()
    with pytest.raises(ValueError, match="non-empty string"):
        pipe("  ")

    object.__setattr__(
        pipe,
        "unet",
        FakeUNet({"block.attn1.processor": OriginalProcessor()}),
    )
    with pytest.raises(RuntimeError, match="no cross-attention processors"):
        pipe("a car")


def test_call_restores_processors_when_base_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import StableDiffusionPipeline

    monkeypatch.setattr(StableDiffusionPipeline, "encode_prompt", _encode_prompt)
    pipe = _pipeline()
    original_processors = dict(pipe.unet.attn_processors)

    def failing_call(self: StructureDiffusionPipeline, **kwargs: object) -> None:
        del self, kwargs
        raise RuntimeError("base generation failed")

    monkeypatch.setattr(StableDiffusionPipeline, "__call__", failing_call)
    with pytest.raises(RuntimeError, match="base generation failed"):
        pipe("a red car")
    assert pipe.unet.attn_processors == original_processors


def test_install_conditioning_rejects_encoded_batch_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _pipeline()
    pipe._processors_for_call = []
    from structurediffusion.pipeline_structurediffusion import _StructuredCall

    with pytest.raises(ValueError, match="does not match"):
        pipe._install_conditioning(
            _StructuredCall(("a car",), ParserType.constituency, False, False),
            torch.zeros((2, 9, 4)),
            torch.device("cpu"),
            1,
            False,
            None,
            None,
        )


def test_pipeline_save_pretrained_from_pretrained_round_trip(tmp_path: Path) -> None:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers.models.clip.configuration_clip import CLIPTextConfig
    from transformers.models.clip.modeling_clip import CLIPTextModel
    from transformers.models.clip.tokenization_clip import CLIPTokenizer

    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values.copy()
    next_unicode_value = 0
    for byte_value in range(256):
        if byte_value in byte_values:
            continue
        byte_values.append(byte_value)
        unicode_values.append(256 + next_unicode_value)
        next_unicode_value += 1
    byte_tokens = [chr(value) for value in unicode_values]
    vocabulary = {
        token: index
        for index, token in enumerate(
            [
                *byte_tokens,
                *(f"{token}</w>" for token in byte_tokens),
                "<|startoftext|>",
                "<|endoftext|>",
            ]
        )
    }
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path.write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(vocab_path), str(merges_path))
    tokenizer.model_max_length = 8

    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(vocabulary),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            bos_token_id=vocabulary["<|startoftext|>"],
            eos_token_id=vocabulary["<|endoftext|>"],
            pad_token_id=vocabulary["<|endoftext|>"],
        )
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(32, 64),
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
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=8,
        attention_head_dim=4,
        norm_num_groups=8,
    )
    pipeline = StructureDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler(
            num_train_timesteps=20,
            steps_offset=1,
            clip_sample=False,
        ),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipeline.save_pretrained(tmp_path / "saved")
    restored = StructureDiffusionPipeline.from_pretrained(
        tmp_path / "saved",
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )

    assert type(restored) is StructureDiffusionPipeline
    assert restored.unet.config.in_channels == pipeline.unet.config.in_channels
    assert list(restored.unet.config.block_out_channels) == list(
        pipeline.unet.config.block_out_channels
    )
    assert restored.tokenizer.vocab_size == tokenizer.vocab_size
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            pipeline.unet.parameters(),
            restored.unet.parameters(),
            strict=True,
        )
    )
