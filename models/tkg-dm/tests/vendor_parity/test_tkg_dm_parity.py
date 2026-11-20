from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from diffusers import __version__ as diffusers_version
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from PIL import Image
from tkg_dm import (
    TKGStableDiffusionPipeline,
    TKGStableDiffusionXLPipeline,
    apply_tkg_noise,
)
from transformers import __version__ as transformers_version

AUTHOR_REVISION = "6dae8837021e4dd2eeea8e53b9ca7c5c8447bd10"
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
SEED = 1234
PROMPT = "yellow lemon and slice, realistic, photo-realistic, 4K, high resolution, high quality"
NEGATIVE_PROMPT = "background, character, cartoon, anime, text, fail, low resolution"
ACTIVE_PROMPT = ", realistic, photo-realistic, 4K, high resolution, high quality"
NOISE_RTOL = 1e-5
NOISE_ATOL = 1e-6
SDXL_MIN_PSNR_DB = 48.0
SDXL_MAX_MAE = 1 / 255
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class ChannelMeanShift(Protocol):
    """Type of the reference noise-channel shift function."""

    def __call__(
        self,
        tensor: torch.Tensor,
        target_shift: float = 0.07,
    ) -> torch.Tensor: ...


class TKGNoise(Protocol):
    """Type of the reference noise transformation."""

    def __call__(self, latents: torch.Tensor, *, device: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class AuthorModules:
    """Author modules loaded from the temporary external checkout."""

    utils: ModuleType
    pipeline: ModuleType


@pytest.fixture
def author_modules() -> Iterator[AuthorModules]:
    """Load the pinned reference modules without copying them into the package."""
    repository = Path(os.environ.get("TKG_DM_AUTHOR_REPO", ""))
    utils_path = repository / "utils.py"
    pipeline_path = repository / "pipeline.py"
    config_path = repository / "config.py"
    if not all(path.is_file() for path in (utils_path, pipeline_path, config_path)):
        message = "Set TKG_DM_AUTHOR_REPO to the temporary pinned author checkout."
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(message)

        pytest.skip(message)

    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != AUTHOR_REVISION:
        pytest.fail(
            f"Author reference must be pinned to {AUTHOR_REVISION}; got {revision}."
        )

    previous_modules = {
        name: sys.modules.get(name) for name in ("config", "utils", "pipeline")
    }
    try:
        config = _load_module("config", config_path)
        utils = _load_module("utils", utils_path)
        pipeline = _load_module("pipeline", pipeline_path)
        del config
        yield AuthorModules(utils=utils, pipeline=pipeline)
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reference module {path.name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _local_custom_pipeline(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source = PACKAGE_ROOT / "src" / "tkg_dm"
    (directory / "pipeline.py").write_bytes(
        b"from .pipeline_tkg_dm import TKGDMHubPipeline\n"
    )
    shutil.copyfile(source / "pipeline_tkg_dm.py", directory / "pipeline_tkg_dm.py")
    return directory


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail("TKG-DM acceptance checks require one selected CUDA GPU.")

        pytest.skip("TKG-DM acceptance checks require one selected CUDA GPU.")


def _record_output(name: str, image: Image.Image) -> str | None:
    output_directory = os.environ.get("TKG_DM_PARITY_OUTPUT_DIR")
    if output_directory is None:
        return None

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / f"{name}.png"
    image.convert("RGB").save(image_path)
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def _environment_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "diffusers": diffusers_version,
        "transformers": transformers_version,
        "gpu": torch.cuda.get_device_name(0),
        "colab_runtime": os.environ.get("COLAB_RELEASE_TAG", "unreported"),
        "author_revision": AUTHOR_REVISION,
        "sd15_revision": SD15_MODEL_REVISION,
        "sdxl_revision": SDXL_MODEL_REVISION,
    }


def _write_result(name: str, result: dict[str, object]) -> None:
    output_directory = os.environ.get("TKG_DM_PARITY_OUTPUT_DIR")
    if output_directory is None:
        return

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "results.json"
    records = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    records["environment"] = _environment_record()
    records[name] = result
    result_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.vendor_parity
@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "target_shift"),
    [("sd15", 0.07), ("sdxl", 0.11)],
)
def test_noise_transformation_matches_author_reference(
    author_modules: AuthorModules,
    family: str,
    target_shift: float,
) -> None:
    _require_cuda()
    values = torch.linspace(-3, 3, 128 * 128, dtype=torch.float32, device="cuda")
    latents = values.reshape(1, 1, 128, 128).expand(1, 4, -1, -1).to(torch.float16)
    original_channel_mean_shift = cast(
        ChannelMeanShift,
        getattr(author_modules.utils, "channel_mean_shift"),
    )
    reference_noise = cast(TKGNoise, getattr(author_modules.utils, "tkg_noise"))

    def channel_mean_shift(
        tensor: torch.Tensor,
        target_shift: float = target_shift,
    ) -> torch.Tensor:
        return original_channel_mean_shift(tensor, target_shift)

    previous_utils = sys.modules.get("utils")
    setattr(author_modules.utils, "channel_mean_shift", channel_mean_shift)
    sys.modules["utils"] = author_modules.utils
    try:
        reference = reference_noise(latents, device="cuda")
    finally:
        setattr(
            author_modules.utils,
            "channel_mean_shift",
            original_channel_mean_shift,
        )
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils

    actual = apply_tkg_noise(latents, target_shift=target_shift)
    difference = (actual.float() - reference.float()).abs()
    torch.testing.assert_close(actual, reference, rtol=NOISE_RTOL, atol=NOISE_ATOL)
    max_absolute_error = float(difference.max().item())
    print(f"{family} noise: max absolute error={max_absolute_error:.8g}")
    _write_result(
        f"{family}_noise",
        {
            "target_shift": target_shift,
            "rtol": NOISE_RTOL,
            "atol": NOISE_ATOL,
            "max_absolute_error": max_absolute_error,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
        },
    )


@pytest.mark.vendor_parity
@pytest.mark.integration
def test_sdxl_image_matches_author_reference(
    author_modules: AuthorModules,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _require_cuda()
    original_scheduler_loader = DDIMScheduler.from_pretrained.__func__

    def pinned_scheduler_loader(
        cls: type[DDIMScheduler],
        pretrained_model_name_or_path: str,
        subfolder: str | None = None,
        **kwargs: object,
    ) -> DDIMScheduler:
        kwargs["revision"] = SDXL_MODEL_REVISION
        return original_scheduler_loader(
            cls,
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )

    monkeypatch.setattr(
        DDIMScheduler,
        "from_pretrained",
        classmethod(pinned_scheduler_loader),
    )
    original_pipeline_loader = AutoPipelineForText2Image.from_pretrained.__func__

    def pinned_pipeline_loader(
        cls: type[AutoPipelineForText2Image],
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> StableDiffusionXLPipeline:
        kwargs["revision"] = SDXL_MODEL_REVISION
        return cast(
            StableDiffusionXLPipeline,
            original_pipeline_loader(cls, pretrained_model_name_or_path, **kwargs),
        )

    monkeypatch.setattr(
        AutoPipelineForText2Image,
        "from_pretrained",
        classmethod(pinned_pipeline_loader),
    )
    original_call = StableDiffusionXLPipeline.__call__

    def negative_prompt_alias(
        self: StableDiffusionXLPipeline,
        *args: object,
        negative_prompts: str | None = None,
        **kwargs: object,
    ) -> object:
        if negative_prompts is not None:
            kwargs["negative_prompt"] = negative_prompts

        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(StableDiffusionXLPipeline, "__call__", negative_prompt_alias)
    reference_generator = author_modules.pipeline.StableDiffusionGenerator(
        method="tkg",
        device="cuda:0",
        seed=SEED,
    )
    _, reference_images = reference_generator.generate_images(
        base_prompts=["yellow lemon and slice"],
        active_prompts=ACTIVE_PROMPT,
        negative_prompts=NEGATIVE_PROMPT,
        guidance_scale=7.5,
        latent_size=128,
        steps=50,
    )
    reference_image = reference_images[0].convert("RGB")
    del reference_images, reference_generator
    torch.cuda.empty_cache()
    package_pipeline = cast(
        TKGStableDiffusionXLPipeline,
        DiffusionPipeline.from_pretrained(
            SDXL_MODEL_ID,
            revision=SDXL_MODEL_REVISION,
            scheduler=DDIMScheduler.from_pretrained(
                SDXL_MODEL_ID,
                subfolder="scheduler",
                revision=SDXL_MODEL_REVISION,
            ),
            custom_pipeline=str(_local_custom_pipeline(tmp_path / "custom_sdxl")),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            variant="fp16",
        ),
    ).to("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    package_image = (
        package_pipeline(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5,
        )
        .images[0]
        .convert("RGB")
    )

    reference_rgb = np.asarray(reference_image, dtype=np.float64) / 255
    package_rgb = np.asarray(package_image, dtype=np.float64) / 255
    absolute_error = np.abs(reference_rgb - package_rgb)
    mae = float(absolute_error.mean())
    mse = float(np.square(reference_rgb - package_rgb).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(1 / mse)
    assert psnr >= SDXL_MIN_PSNR_DB
    assert mae <= SDXL_MAX_MAE
    print(f"SDXL image: PSNR={psnr:.6f} dB, MAE={mae:.8g}")

    reference_hash = _record_output("sdxl_reference", reference_image)
    package_hash = _record_output("sdxl_package_parity", package_image)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    example_image = (
        package_pipeline(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            num_inference_steps=50,
            guidance_scale=7.5,
        )
        .images[0]
        .convert("RGB")
    )
    example_hash = _record_output("sdxl_readme_example", example_image)
    _write_result(
        "sdxl_image",
        {
            "target_shift": 0.11,
            "psnr_db": "infinity" if math.isinf(psnr) else psnr,
            "minimum_psnr_db": SDXL_MIN_PSNR_DB,
            "mae": mae,
            "maximum_mae": SDXL_MAX_MAE,
            "reference_sha256": reference_hash,
            "package_sha256": package_hash,
            "readme_example_sha256": example_hash,
            "image_size": list(package_image.size),
        },
    )

    del package_pipeline
    torch.cuda.empty_cache()


@pytest.mark.vendor_parity
@pytest.mark.integration
def test_sd15_package_inference_repeats_with_identical_image_hashes(
    tmp_path: Path,
) -> None:
    _require_cuda()
    scheduler = DDIMScheduler.from_pretrained(
        SD15_MODEL_ID,
        subfolder="scheduler",
        revision=SD15_MODEL_REVISION,
    )
    pipeline = cast(
        TKGStableDiffusionPipeline,
        DiffusionPipeline.from_pretrained(
            SD15_MODEL_ID,
            revision=SD15_MODEL_REVISION,
            scheduler=scheduler,
            custom_pipeline=str(_local_custom_pipeline(tmp_path / "custom_sd15")),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            variant="fp16",
        ),
    ).to("cuda")

    def generate() -> Image.Image:
        return (
            pipeline(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                generator=torch.Generator(device="cpu").manual_seed(SEED),
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
            .images[0]
            .convert("RGB")
        )

    first_image = generate()
    second_image = generate()
    first_pixels = np.asarray(first_image)
    second_pixels = np.asarray(second_image)
    assert np.array_equal(first_pixels, second_pixels)
    first_png_hash = _record_output("sd15_readme_example", first_image)
    first_hash = hashlib.sha256(first_pixels.tobytes()).hexdigest()
    second_hash = hashlib.sha256(second_pixels.tobytes()).hexdigest()
    print(f"SD 1.5 repeatability: image hash={second_hash}")
    _write_result(
        "sd15_repeatability",
        {
            "target_shift": 0.07,
            "identical_pixels": True,
            "first_rgb_sha256": first_hash,
            "second_rgb_sha256": second_hash,
            "example_png_sha256": first_png_hash,
            "image_size": list(first_image.size),
        },
    )
    del pipeline
    torch.cuda.empty_cache()
