from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import snapshot_download
from PIL import Image
from sawna import SAWNAStableDiffusionPipeline, SAWNAStableDiffusionXLPipeline
from sawna.processing_sawna import apply_sawna_noise

SPACE_ID = "EQUES/Space-Aware_Text-to-Image_Generation"
SPACE_REVISION = "e0a8f843a0b4e62c0650b03d5f335271ecb36d0b"
PROMPT = "A majestic lion in a natural landscape"
BOXES = [(0.3, 0.3, 0.7, 0.7)]
SEED = 1234
MODELS = (
    (
        "stable-diffusion-1-5",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        512,
        "stable",
    ),
    (
        "stable-diffusion-2-1",
        "sd2-community/stable-diffusion-2-1",
        "bb2154823665391b4fb29b0b9cf82a198964ee05",
        768,
        "stable",
    ),
    (
        "stable-diffusion-xl",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "462165984030d82259a11f4367a4eed129e94a7b",
        1024,
        "xl",
    ),
)


class SpaceNoiseOptimizer(Protocol):
    def apply_color_shift(
        self,
        _noise_tensor: torch.Tensor,
        _target_color: list[float],
        _target_shifts: list[float],
    ) -> torch.Tensor: ...

    def create_bounding_box_mask(
        self,
        _height: int,
        _width: int,
        _bounding_boxes: list[tuple[float, float, float, float]],
        _sigma: float | None = None,
    ) -> torch.Tensor: ...

    def blend_noise_with_mask(
        self,
        _original_noise: torch.Tensor,
        _shifted_noise: torch.Tensor,
        _mask: torch.Tensor,
    ) -> torch.Tensor: ...


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _config_sha256(config: Mapping[str, object]) -> str:
    serialized = json.dumps(dict(config), sort_keys=True, default=str).encode()
    return _sha256_bytes(serialized)


def _reference_optimizer(source_path: Path) -> tuple[SpaceNoiseOptimizer, str]:
    source = source_path.read_bytes()
    spec = importlib.util.spec_from_file_location("sawna_space_reference", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned Space reference module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    optimizer = cast(
        SpaceNoiseOptimizer,
        module.TKGDMNoiseOptimizer(device="cuda", dtype=torch.float16),
    )
    return optimizer, _sha256_bytes(source)


def _run_pipeline(
    pipeline: StableDiffusionPipeline | StableDiffusionXLPipeline,
    *,
    size: int,
    latents: torch.Tensor,
) -> torch.Tensor:
    if isinstance(pipeline, StableDiffusionXLPipeline):
        result = pipeline(
            prompt=PROMPT,
            height=size,
            width=size,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            latents=latents,
            output_type="pt",
            original_size=(size, size),
            target_size=(size, size),
            crops_coords_top_left=(0, 0),
        )
    else:
        result = pipeline(
            prompt=PROMPT,
            height=size,
            width=size,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            latents=latents,
            output_type="pt",
        )

    return result.images


def _save_example(images: torch.Tensor, path: Path) -> None:
    pixels = images[0].detach().float().clamp(0, 1).permute(1, 2, 0).mul(255).round()
    Image.fromarray(pixels.byte().cpu().numpy()).save(path)


@pytest.mark.vendor_parity
@pytest.mark.integration
def test_pinned_space_noise_and_checkpoint_outputs_are_bitwise_equal() -> None:
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("Set PARITY_REQUIRE=1 to run asset-backed SAWNA acceptance")

    if not torch.cuda.is_available():
        pytest.fail("SAWNA acceptance requires a CUDA GPU")

    if importlib.metadata.version("diffusers") != "0.39.0":
        pytest.fail("SAWNA acceptance requires Diffusers 0.39.0")

    output_root = Path(os.environ.get("SAWNA_PARITY_OUTPUT_DIR", ".cache/sawna/parity"))
    output_root.mkdir(parents=True, exist_ok=True)
    space_root = Path(
        snapshot_download(repo_id=SPACE_ID, repo_type="space", revision=SPACE_REVISION)
    )
    optimizer, source_hash = _reference_optimizer(space_root / "tkg_dm.py")
    results: list[dict[str, object]] = []
    requested = os.environ.get("SAWNA_PARITY_MODELS")
    selected = (
        {name.strip() for name in requested.split(",") if name.strip()}
        if requested is not None
        else {model[0] for model in MODELS}
    )
    unknown = selected - {model[0] for model in MODELS}
    if unknown or not selected:
        pytest.fail("SAWNA_PARITY_MODELS contains an unknown or empty model selection")

    selected_models = tuple(model for model in MODELS if model[0] in selected)

    for name, model_id, revision, size, family in selected_models:
        if family == "xl":
            package_pipe = SAWNAStableDiffusionXLPipeline.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=torch.float16,
            ).to("cuda")
            base_class = StableDiffusionXLPipeline
        else:
            package_pipe = SAWNAStableDiffusionPipeline.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=torch.float16,
            ).to("cuda")
            base_class = StableDiffusionPipeline
        package_pipe.set_progress_bar_config(disable=True)

        reference_components = dict(package_pipe.components)
        reference_components["scheduler"] = type(package_pipe.scheduler).from_config(
            package_pipe.scheduler.config
        )
        reference_pipe = base_class(**reference_components).to("cuda")
        reference_pipe.set_progress_bar_config(disable=True)
        latent_size = size // 8
        epsilon = torch.randn(
            (1, 4, latent_size, latent_size),
            device="cuda",
            dtype=torch.float16,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        )
        shifted = optimizer.apply_color_shift(
            epsilon,
            [0.0, 0.0, 0.0],
            [0.0, 0.07, 0.07, 0.0],
        )
        reference_mask = optimizer.create_bounding_box_mask(
            latent_size,
            latent_size,
            BOXES,
            sigma=None,
        )
        reference_noise = optimizer.blend_noise_with_mask(
            epsilon,
            shifted,
            reference_mask,
        )
        package_noise = apply_sawna_noise(epsilon, BOXES)
        assert torch.equal(package_noise, reference_noise), f"{name} noise differs"

        reference_images = _run_pipeline(
            reference_pipe,
            size=size,
            latents=reference_noise,
        )
        package_images = _run_pipeline(
            package_pipe,
            size=size,
            latents=package_noise,
        )
        repeated_images = _run_pipeline(
            package_pipe,
            size=size,
            latents=package_noise,
        )
        assert torch.equal(package_images, reference_images), f"{name} output differs"
        assert torch.equal(package_images, repeated_images), f"{name} is not repeatable"

        image_path = output_root / f"sawna-{name}-lion-seed{SEED}.png"
        _save_example(package_images, image_path)
        torch.save(
            {
                "epsilon": epsilon.cpu(),
                "epsilon_masked": package_noise.cpu(),
                "reference_images": reference_images.cpu(),
                "package_images": package_images.cpu(),
            },
            output_root / f"sawna-{name}-parity.pt",
        )
        results.append(
            {
                "name": name,
                "model_id": model_id,
                "model_revision": revision,
                "size": [size, size],
                "scheduler": package_pipe.scheduler.config._class_name,
                "scheduler_config_sha256": _config_sha256(
                    package_pipe.scheduler.config
                ),
                "epsilon_masked_bitwise_equal": True,
                "generated_images_bitwise_equal": True,
                "repeatable": True,
                "image_sha256": _sha256_bytes(image_path.read_bytes()),
            }
        )
        del package_pipe, reference_pipe
        gc.collect()
        torch.cuda.empty_cache()

    run = {
        "repository_revision": (
            os.environ["SAWNA_REPOSITORY_REVISION"]
            if "SAWNA_REPOSITORY_REVISION" in os.environ
            else subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        ),
        "space_id": SPACE_ID,
        "space_revision": SPACE_REVISION,
        "space_tkg_dm_sha256": source_hash,
        "space_app_sha256": _sha256_bytes((space_root / "app.py").read_bytes()),
        "prompt": PROMPT,
        "boxes": BOXES,
        "target_shift": 0.07,
        "blur_sigma": None,
        "seed": SEED,
        "dtype": "float16",
        "steps": 25,
        "guidance_scale": 7.5,
        "negative_prompt": None,
        "torch": torch.__version__,
        "diffusers": importlib.metadata.version("diffusers"),
        "transformers": importlib.metadata.version("transformers"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "models": results,
    }
    result_name = (
        "parity-results.json"
        if selected_models == MODELS
        else f"parity-results-{'-'.join(result['name'] for result in results)}.json"
    )
    (output_root / result_name).write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
