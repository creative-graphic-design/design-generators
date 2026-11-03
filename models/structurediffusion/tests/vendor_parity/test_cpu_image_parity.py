from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from diffusers import PNDMScheduler
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from jaxtyping import Bool, Float
from nltk.tree import Tree
from PIL import Image

from structurediffusion.modeling_structurediffusion import (
    StructuredAttentionProcessor,
    StructuredConditioning,
)
from structurediffusion.pipeline_structurediffusion import (
    ParserType,
    StructureDiffusionPipeline,
    _StructuredCall,
)
from structurediffusion.processing_structurediffusion import (
    NounPhrase,
    _constituency_pipeline,
    normalize_prompt,
    token_span_from_char_span,
)

PROMPT = "A red car and a white sheep"
GUIDANCE_SCALE = 7.5
INFERENCE_STEPS = 10
SEED = 42
CONDITIONING_ATOL = 1e-6


class _VendorGetAllNps(Protocol):
    def __call__(
        self,
        tree: Tree,
        _full_sent: str,
        tokens: list[str],
    ) -> tuple[
        list[str],
        list[tuple[int, int]],
        list[list[str | tuple[int, int]]],
    ]: ...


class _VendorMultiAlign(Protocol):
    def __call__(
        self,
        _main_seq: torch.Tensor,
        seq: torch.Tensor,
        span: tuple[int, int],
        dim: int = 1,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class _VendorPromptFunctions:
    preprocess_prompts: Callable[[str], str]
    get_all_nps: _VendorGetAllNps
    multi_align: _VendorMultiAlign


@dataclass(frozen=True)
class _VendorConditioning:
    """Reference keys and values assembled through the vendor alignment path."""

    keys: tuple[Float[torch.Tensor, "batch tokens dim"], ...]
    values: tuple[Float[torch.Tensor, "batch tokens dim"], ...]


class VendorAttentionReferenceProcessor:
    """Run the pinned vendor multi_qkv with weights from a Diffusers layer."""

    def __init__(
        self,
        layer_name: str,
        attention: Attention,
        conditioning: _VendorConditioning,
    ) -> None:
        from structured_stable_diffusion.modules.attention import (
            CrossAttention as VendorCrossAttention,
        )

        if attention.to_q is None or attention.to_k is None or attention.to_v is None:
            raise ValueError(
                "The Diffusers attention layer must define Q/K/V projections."
            )

        self.layer_name = layer_name
        self.conditioning = conditioning
        self.reference = VendorCrossAttention(
            query_dim=attention.query_dim,
            context_dim=attention.cross_attention_dim,
            heads=attention.heads,
            dim_head=attention.inner_dim // attention.heads,
            struct_attn=True,
        )
        self.reference.to_q = attention.to_q
        self.reference.to_k = attention.to_k
        self.reference.to_v = attention.to_v

    def __call__(
        self,
        attn: Attention,
        hidden_states: Float[torch.Tensor, "..."],
        encoder_hidden_states: Float[torch.Tensor, "batch tokens dim"] | None = None,
        attention_mask: Bool[torch.Tensor, "..."]
        | Float[torch.Tensor, "..."]
        | None = None,
        temb: Float[torch.Tensor, "batch channels"] | None = None,
        scale: float | None = None,
    ) -> Float[torch.Tensor, "..."]:
        del scale
        if encoder_hidden_states is None or attention_mask is not None:
            raise ValueError(
                "The Stable Diffusion reference expects unmasked text context."
            )
        if attn.to_q is None or attn.to_out is None:
            raise ValueError(
                "The Diffusers attention layer must define Q and output projections."
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        spatial_dimensions: tuple[int, int, int] | None = None
        if hidden_states.ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_dimensions = (channel, height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            batch_size, _, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        if attn.norm_cross or attn.norm_q is not None or attn.norm_k is not None:
            raise ValueError("Unexpected normalized attention layer in SD 1.4.")

        query = attn.to_q(hidden_states)
        base_batch = self.conditioning.keys[0].shape[0]
        if batch_size != base_batch * 2:
            raise ValueError(
                "The vendor reference requires matching classifier-free guidance batches."
            )

        output = self.reference.multi_qkv(
            query,
            encoder_hidden_states[:base_batch],
            list(self.conditioning.keys),
            list(self.conditioning.values),
            mask=None,
        )
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        if spatial_dimensions is not None:
            channel, height, width = spatial_dimensions
            output = output.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,
            )
        if attn.residual_connection:
            output = output + residual

        return output / attn.rescale_output_factor


def _vendor_prompt_functions(repo_root: Path) -> _VendorPromptFunctions:
    source_path = (
        repo_root / "vendor/structured-diffusion-guidance/scripts/txt2img_demo.py"
    )
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    requested = {
        "get_all_nps",
        "get_token_alignment_map",
        "multi_align",
        "preprocess_prompts",
    }
    definitions = [
        node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in requested
    ]
    if {definition.name for definition in definitions} != requested:
        raise RuntimeError("The pinned vendor prompt-alignment functions are missing.")

    module = ast.Module(body=cast(list[ast.stmt], definitions), type_ignores=[])
    namespace: dict[str, object] = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return _VendorPromptFunctions(
        preprocess_prompts=cast(Callable[[str], str], namespace["preprocess_prompts"]),
        get_all_nps=cast(_VendorGetAllNps, namespace["get_all_nps"]),
        multi_align=cast(_VendorMultiAlign, namespace["multi_align"]),
    )


def _image_array(images: object) -> Float[np.ndarray, "batch height width channels"]:
    if not isinstance(images, np.ndarray):
        raise TypeError("Diffusers did not return a NumPy image batch.")
    return cast(Float[np.ndarray, "batch height width channels"], images)


def _configure_plms_scheduler(pipe: StableDiffusionPipeline) -> None:
    scheduler = PNDMScheduler.from_config(
        pipe.scheduler.config,
        skip_prk_steps=True,
        timestep_spacing="leading",
        steps_offset=1,
    )
    if isinstance(scheduler, tuple):
        raise TypeError("PNDMScheduler.from_config returned unused configuration.")

    pipe.scheduler = scheduler


def _capture_package_conditioning(
    pipe: StructureDiffusionPipeline,
    prompt: str,
) -> tuple[StructuredConditioning, list[NounPhrase], list[tuple[int, int]]]:
    normalized_prompt = normalize_prompt(prompt)
    request = _StructuredCall(
        prompts=(normalized_prompt,),
        parser_type=ParserType.constituency,
        conjunction=False,
        save_attention_maps=False,
    )
    phrases = pipe._parse_prompt(normalized_prompt, ParserType.constituency)
    package_spans = [
        token_span_from_char_span(
            pipe.tokenizer,
            normalized_prompt,
            phrase.start,
            phrase.end,
            pipe.tokenizer.model_max_length,
        )
        for phrase in phrases
    ]
    spans = [span for span in package_spans if span is not None]
    if len(spans) != len(package_spans):
        raise AssertionError("The package could not align every extracted phrase.")

    processor = StructuredAttentionProcessor("conditioning-capture", False)
    pipe._processors_for_call = [processor]
    try:
        prompt_embeds, _ = StableDiffusionPipeline.encode_prompt(
            pipe,
            prompt=normalized_prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        pipe._install_conditioning(
            request,
            prompt_embeds,
            torch.device("cpu"),
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            lora_scale=None,
            clip_skip=None,
        )
        conditioning = processor.conditioning
        if conditioning is None:
            raise AssertionError("The package did not install structured conditioning.")

        return conditioning, phrases, spans
    finally:
        del pipe._processors_for_call


def _build_vendor_conditioning(
    pipe: StableDiffusionPipeline,
    raw_prompt: str,
    vendor: _VendorPromptFunctions,
) -> tuple[_VendorConditioning, str, list[str], list[tuple[int, int]]]:
    prompt = vendor.preprocess_prompts(raw_prompt)
    sentence = _constituency_pipeline()(prompt).sentences[0]
    tree = Tree.fromstring(str(sentence.constituency))
    phrases, spans, _ = vendor.get_all_nps(
        tree,
        prompt,
        pipe.tokenizer.tokenize(prompt),
    )
    if not phrases or len(phrases) != len(spans):
        raise AssertionError("The vendor did not return matching phrases and spans.")

    encoded_contexts, _ = StableDiffusionPipeline.encode_prompt(
        pipe,
        prompt=phrases,
        device=torch.device("cpu"),
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    base_context = encoded_contexts[:1]
    aligned_phrase_contexts = tuple(
        vendor.multi_align(
            base_context.clone(), phrase_context.unsqueeze(0), span, dim=1
        )
        for phrase_context, span in zip(encoded_contexts[1:], spans[1:], strict=True)
    )
    return (
        _VendorConditioning(
            keys=(base_context,),
            values=(base_context, *aligned_phrase_contexts),
        ),
        prompt,
        phrases,
        [(start + 1, end + 1) for start, end in spans[1:]],
    )


def _conditioning_difference(
    package: StructuredConditioning,
    vendor: _VendorConditioning,
    field: str,
) -> float:
    package_tensors = package.keys if field == "keys" else package.values
    vendor_tensors = vendor.keys if field == "keys" else vendor.values
    if len(package_tensors) != len(vendor_tensors):
        raise AssertionError(f"Package and vendor {field} counts differ.")

    differences: list[float] = []
    for package_tensor, vendor_tensor in zip(
        package_tensors, vendor_tensors, strict=True
    ):
        if package_tensor.shape != vendor_tensor.shape:
            raise AssertionError(f"Package and vendor {field} shapes differ.")

        differences.append(float((package_tensor - vendor_tensor).abs().max().item()))

    return max(differences, default=0.0)


def _assert_conditionings_match(
    package: StructuredConditioning,
    vendor: _VendorConditioning,
) -> tuple[float, float]:
    key_difference = _conditioning_difference(package, vendor, "keys")
    value_difference = _conditioning_difference(package, vendor, "values")
    if key_difference > CONDITIONING_ATOL or value_difference > CONDITIONING_ATOL:
        raise AssertionError(
            "The package and vendor conditioning differs: "
            f"max key difference={key_difference}, "
            f"max value difference={value_difference}."
        )

    return key_difference, value_difference


def _generate_with_vendor_attention(
    pipe: StableDiffusionPipeline,
    prompt: str,
    latents: Float[torch.Tensor, "batch channels height width"],
    conditioning: _VendorConditioning,
) -> Float[np.ndarray, "batch height width channels"]:
    original_processors = dict(pipe.unet.attn_processors)
    replacements: dict[str, object] = {}
    for name, processor in original_processors.items():
        if name.endswith("attn2.processor"):
            layer = pipe.unet.get_submodule(name.removesuffix(".processor"))
            replacements[name] = VendorAttentionReferenceProcessor(
                name,
                layer,
                conditioning,
            )
        else:
            replacements[name] = processor

    if all(not name.endswith("attn2.processor") for name in original_processors):
        raise RuntimeError("The U-Net has no cross-attention layers to compare.")

    pipe.unet.set_attn_processor(replacements)
    try:
        result = pipe(
            prompt=prompt,
            height=latents.shape[-2] * pipe.vae_scale_factor,
            width=latents.shape[-1] * pipe.vae_scale_factor,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            latents=latents.clone(),
            output_type="np",
        )
        return _image_array(result.images)
    finally:
        pipe.unet.set_attn_processor(original_processors)


def _save_image(
    path: Path, image_batch: Float[np.ndarray, "batch height width channels"]
) -> None:
    pixels = np.round(np.clip(image_batch[0], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.mark.vendor_parity
def test_cpu_image_matches_independent_vendor_path() -> None:
    """Check vendor phrase spans, K/V composition, attention, and generated pixels."""
    if os.getenv("PARITY_REQUIRE") != "1":
        pytest.skip("Set PARITY_REQUIRE=1 to run the Stable Diffusion CPU comparison.")

    repo_root = Path(__file__).resolve().parents[4]
    vendor_root = repo_root / "vendor/structured-diffusion-guidance"
    if not (vendor_root / "structured_stable_diffusion/modules/attention.py").is_file():
        pytest.fail("Initialize vendor/structured-diffusion-guidance before parity.")
    if torch.cuda.is_available():
        pytest.fail("The authorized parity comparison is CPU-only.")

    resolution = int(os.getenv("SDG_COMPARE_RESOLUTION", "64"))
    if resolution not in {64, 512}:
        raise ValueError("SDG_COMPARE_RESOLUTION must be 64 for rehearsal or 512.")

    torch.set_num_threads(int(os.getenv("SDG_CPU_THREADS", "2")))
    started = time.perf_counter()
    model_id = os.getenv("SDG_MODEL_ID", "CompVis/stable-diffusion-v1-4")
    package_load_started = time.perf_counter()
    package_pipe = StructureDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    _configure_plms_scheduler(package_pipe)
    package_pipe.to("cpu")
    package_model_load_seconds = time.perf_counter() - package_load_started

    package_conditioning, package_phrases, package_spans = (
        _capture_package_conditioning(package_pipe, PROMPT)
    )
    vendor = _vendor_prompt_functions(repo_root)
    vendor_prompt = vendor.preprocess_prompts(PROMPT)
    if vendor_prompt != normalize_prompt(PROMPT):
        raise AssertionError("Package and vendor prompt preprocessing differs.")

    latent_side = resolution // package_pipe.vae_scale_factor
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    latents = torch.randn(
        (1, package_pipe.unet.config.in_channels, latent_side, latent_side),
        generator=generator,
        dtype=torch.float32,
    )
    latent_digest = hashlib.sha256(latents.numpy().tobytes()).hexdigest()
    package_started = time.perf_counter()
    package_result = package_pipe(
        PROMPT,
        parser_type=ParserType.constituency,
        conjunction=False,
        height=resolution,
        width=resolution,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        latents=latents.clone(),
        output_type="np",
    )
    package_images = _image_array(package_result.images)
    package_generation_seconds = time.perf_counter() - package_started
    del package_pipe
    gc.collect()

    reference_load_started = time.perf_counter()
    reference_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    _configure_plms_scheduler(reference_pipe)
    reference_pipe.to("cpu")
    reference_model_load_seconds = time.perf_counter() - reference_load_started

    conditioning_started = time.perf_counter()
    (
        vendor_conditioning,
        vendor_prompt,
        reference_phrases,
        reference_spans,
    ) = _build_vendor_conditioning(reference_pipe, PROMPT, vendor)
    reference_conditioning_seconds = time.perf_counter() - conditioning_started
    if reference_phrases[1:] != [phrase.text for phrase in package_phrases]:
        raise AssertionError("Vendor phrase extraction changed between pipelines.")
    if reference_spans != package_spans:
        raise AssertionError("Vendor token spans changed between pipelines.")
    key_conditioning_difference, value_conditioning_difference = (
        _assert_conditionings_match(package_conditioning, vendor_conditioning)
    )

    reference_started = time.perf_counter()
    reference_images = _generate_with_vendor_attention(
        reference_pipe,
        vendor_prompt,
        latents,
        vendor_conditioning,
    )
    reference_generation_seconds = time.perf_counter() - reference_started
    del reference_pipe
    gc.collect()

    difference = np.abs(package_images - reference_images)
    metrics = {
        "input_prompt": PROMPT,
        "vendor_preprocessed_prompt": vendor_prompt,
        "resolution": resolution,
        "seed": SEED,
        "steps": INFERENCE_STEPS,
        "scheduler": (
            "Diffusers PNDMScheduler PLMS path, skip_prk_steps=True, "
            "timestep_spacing=leading, steps_offset=1"
        ),
        "guidance_scale": GUIDANCE_SCALE,
        "parser": "Stanza constituency; phrase extraction and BPE mapping checked against vendor functions",
        "vendor_phrases": reference_phrases,
        "package_phrases": [phrase.text for phrase in package_phrases],
        "vendor_token_spans_with_bos": reference_spans,
        "package_token_spans_with_bos": package_spans,
        "initial_latent_shape": list(latents.shape),
        "initial_latent_sha256": latent_digest,
        "vendor_key_count": len(vendor_conditioning.keys),
        "vendor_value_count": len(vendor_conditioning.values),
        "max_abs_key_conditioning_difference": key_conditioning_difference,
        "max_abs_value_conditioning_difference": value_conditioning_difference,
        "package_pipeline_class": "StructureDiffusionPipeline",
        "reference_pipeline_class": "StableDiffusionPipeline",
        "reference_attention": "Pinned vendor CrossAttention.multi_qkv",
        "model_loading_and_sampler": "Diffusers for both separate pipeline instances",
        "device": "cpu",
        "package_model_load_seconds": package_model_load_seconds,
        "package_generation_seconds": package_generation_seconds,
        "reference_model_load_seconds": reference_model_load_seconds,
        "reference_conditioning_seconds": reference_conditioning_seconds,
        "reference_generation_seconds": reference_generation_seconds,
        "max_abs_pixel_difference": float(difference.max()),
        "mean_abs_pixel_difference": float(difference.mean()),
        "root_mean_square_pixel_difference": float(np.sqrt(np.mean(difference**2))),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output_dir = Path(
        os.getenv(
            "STRUCTUREDIFFUSION_PARITY_OUTPUT",
            str(repo_root / ".cache/structurediffusion/parity"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cpu-{resolution}.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _save_image(output_dir / f"package-{resolution}.png", package_images)
    _save_image(output_dir / f"reference-{resolution}.png", reference_images)
    print(json.dumps(metrics, indent=2))

    assert (
        reference_images.shape
        == package_images.shape
        == (
            1,
            resolution,
            resolution,
            3,
        )
    )
    assert np.isfinite(reference_images).all()
    assert np.isfinite(package_images).all()
    assert metrics["mean_abs_pixel_difference"] < 1e-3
