from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from jaxtyping import Bool, Float
from nltk.tree import Tree

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
    token_span_from_char_span,
)

PROMPT = "A red car and a white sheep"
GUIDANCE_SCALE = 7.5
INFERENCE_STEPS = 10
SEED = 42


class VendorAttentionReferenceProcessor:
    """Call the pinned source's multi_qkv implementation with Diffusers weights."""

    def __init__(self, layer_name: str, attention: Attention) -> None:
        from structured_stable_diffusion.modules.attention import (
            CrossAttention as VendorCrossAttention,
        )

        if attention.to_q is None or attention.to_k is None or attention.to_v is None:
            raise ValueError(
                "The Diffusers attention layer must define Q/K/V projections."
            )

        self.layer_name = layer_name
        self.conditioning: StructuredConditioning | None = None
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
        conditioning = self.conditioning
        if conditioning is None or not conditioning.do_classifier_free_guidance:
            raise ValueError("The vendor reference requires classifier-free guidance.")
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
        input_ndim = hidden_states.ndim
        spatial_dimensions: tuple[int, int, int] | None = None
        if input_ndim == 4:
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
        base_batch = conditioning.keys[0].shape[0]
        if batch_size != base_batch * 2:
            raise ValueError(
                "The vendor reference requires matching CFG query batches."
            )
        unconditional_context = encoder_hidden_states[:base_batch]
        output = self.reference.multi_qkv(
            query,
            unconditional_context,
            list(conditioning.keys),
            list(conditioning.values),
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


def _vendor_parser_function(
    repo_root: Path,
) -> Callable[
    [Tree, str, list[str]],
    tuple[list[str], list[tuple[int, int]], list[tuple[str, tuple[int, int]]]],
]:
    source_path = (
        repo_root / "vendor/structured-diffusion-guidance/scripts/txt2img_demo.py"
    )
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    requested = {"get_all_nps", "get_token_alignment_map"}
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
    return cast(
        Callable[
            [Tree, str, list[str]],
            tuple[list[str], list[tuple[int, int]], list[tuple[str, tuple[int, int]]]],
        ],
        namespace["get_all_nps"],
    )


def _image_array(images: object) -> Float[np.ndarray, "batch height width channels"]:
    if not isinstance(images, np.ndarray):
        raise TypeError("Diffusers did not return a NumPy image batch.")
    return cast(Float[np.ndarray, "batch height width channels"], images)


def _generate_with_vendor_attention(
    pipe: StructureDiffusionPipeline,
    prompt: str,
    latents: Float[torch.Tensor, "batch channels height width"],
) -> Float[np.ndarray, "batch height width channels"]:
    original_processors = dict(pipe.unet.attn_processors)
    reference_processors: list[VendorAttentionReferenceProcessor] = []
    replacements: dict[str, object] = {}
    for name, processor in original_processors.items():
        if name.endswith("attn2.processor"):
            layer = pipe.unet.get_submodule(name.removesuffix(".processor"))
            reference = VendorAttentionReferenceProcessor(name, layer)
            reference_processors.append(reference)
            replacements[name] = reference
        else:
            replacements[name] = processor

    if not reference_processors:
        raise RuntimeError("The U-Net has no cross-attention layers to compare.")

    pipe.unet.set_attn_processor(replacements)
    pipe._call_request = _StructuredCall(
        prompts=(prompt,),
        parser_type=ParserType.constituency,
        conjunction=False,
        save_attention_maps=False,
    )
    pipe._processors_for_call = cast(
        list[StructuredAttentionProcessor],
        reference_processors,
    )
    try:
        result = StableDiffusionPipeline.__call__(
            pipe,
            prompt=prompt,
            height=latents.shape[-2] * pipe.vae_scale_factor,
            width=latents.shape[-1] * pipe.vae_scale_factor,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            eta=0.0,
            latents=latents.clone(),
            output_type="np",
        )
        return _image_array(result.images)
    finally:
        del pipe._call_request
        del pipe._processors_for_call
        pipe.unet.set_attn_processor(original_processors)


@pytest.mark.vendor_parity
def test_cpu_image_matches_pinned_vendor_attention() -> None:
    """Compare one CPU image with the official source's structured attention."""
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
    pipe = StructureDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="leading",
        steps_offset=1,
        clip_sample=False,
    )
    pipe.to("cpu")
    model_load_seconds = time.perf_counter() - started

    prompt = PROMPT
    phrases = pipe._parse_prompt(prompt, ParserType.constituency)
    tokenizer = pipe.tokenizer
    vendor_get_all_nps = _vendor_parser_function(repo_root)
    from structurediffusion.processing_structurediffusion import (
        _constituency_pipeline,
    )

    sentence = _constituency_pipeline()(prompt).sentences[0]
    tree = Tree.fromstring(str(sentence.constituency))
    vendor_phrases, vendor_spans, _ = vendor_get_all_nps(
        tree,
        prompt,
        tokenizer.tokenize(prompt),
    )
    package_spans = [
        token_span_from_char_span(
            tokenizer,
            prompt,
            phrase.start,
            phrase.end,
            tokenizer.model_max_length,
        )
        for phrase in phrases
    ]
    vendor_package_spans = [(start + 1, end + 1) for start, end in vendor_spans[1:]]
    if [phrase.text for phrase in phrases] != vendor_phrases[1:]:
        raise AssertionError(
            "The package and vendor parsers extracted different phrases."
        )
    if package_spans != vendor_package_spans:
        raise AssertionError(
            "The package and vendor parsers aligned different CLIP tokens."
        )

    latent_side = resolution // pipe.vae_scale_factor
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_side, latent_side),
        generator=generator,
        dtype=torch.float32,
    )
    latent_digest = hashlib.sha256(latents.numpy().tobytes()).hexdigest()
    reference_started = time.perf_counter()
    reference_images = _generate_with_vendor_attention(pipe, prompt, latents)
    reference_seconds = time.perf_counter() - reference_started
    package_started = time.perf_counter()
    package_result = pipe(
        prompt,
        parser_type=ParserType.constituency,
        conjunction=False,
        height=resolution,
        width=resolution,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        eta=0.0,
        latents=latents.clone(),
        output_type="np",
    )
    package_seconds = time.perf_counter() - package_started
    package_images = _image_array(package_result.images)
    difference = np.abs(package_images - reference_images)
    metrics = {
        "resolution": resolution,
        "prompt": PROMPT,
        "seed": SEED,
        "steps": INFERENCE_STEPS,
        "scheduler": "DDIM, leading spacing, steps_offset=1, eta=0",
        "guidance_scale": GUIDANCE_SCALE,
        "parser": "Stanza constituency",
        "phrases": [phrase.text for phrase in phrases],
        "token_spans": package_spans,
        "initial_latent_shape": list(latents.shape),
        "initial_latent_sha256": latent_digest,
        "device": "cpu",
        "model_load_seconds": model_load_seconds,
        "reference_generation_seconds": reference_seconds,
        "package_generation_seconds": package_seconds,
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
