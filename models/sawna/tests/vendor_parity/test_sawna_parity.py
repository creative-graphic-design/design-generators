from __future__ import annotations

import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import NoReturn, Protocol, cast

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


class SpacePipeline(Protocol):
    pipe: StableDiffusionPipeline | StableDiffusionXLPipeline | None
    noise_optimizer: SpaceNoiseOptimizer


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _config_sha256(config: Mapping[str, object]) -> str:
    serialized = json.dumps(dict(config), sort_keys=True, default=str).encode()
    return _sha256_bytes(serialized)


def _reference_tkgdm(source_path: Path) -> tuple[ModuleType, str]:
    source = source_path.read_bytes()
    spec = importlib.util.spec_from_file_location("sawna_space_reference", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned Space reference module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, _sha256_bytes(source)


def _reference_app(
    source_path: Path,
    pipeline_factory: Callable[..., object],
) -> tuple[Callable[..., Image.Image], str]:
    source = source_path.read_bytes()
    tree = ast.parse(source, filename=str(source_path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"parse_bounding_boxes", "generate_tkg_dm_image"}
    ]
    if len(functions) != 2:
        raise RuntimeError("Pinned Space app functions are missing")

    def fail_on_demo_fallback(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("Pinned Space app fell back to its demo image")

    namespace: dict[str, object] = {
        "torch": torch,
        "TKGDMPipeline": pipeline_factory,
        "create_demo_visualization": fail_on_demo_fallback,
    }
    # The source is pinned; only its generation functions execute in this harness.
    exec(  # noqa: S102
        compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return cast(Callable[..., Image.Image], namespace["generate_tkg_dm_image"]), (
        _sha256_bytes(source)
    )


def _run_pipeline(
    pipeline: StableDiffusionPipeline | StableDiffusionXLPipeline,
    *,
    size: int,
    latents: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        generator = torch.Generator(device="cuda").manual_seed(SEED)

    if isinstance(pipeline, StableDiffusionXLPipeline):
        result = pipeline(
            prompt=PROMPT,
            height=size,
            width=size,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator,
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
            generator=generator,
            latents=latents,
            output_type="pt",
        )

    return result.images


def _save_example(images: torch.Tensor, path: Path) -> None:
    pixels = _pixels_from_tensor(images)
    Image.fromarray(pixels.byte().cpu().numpy()).save(path)


def _pixels_from_tensor(images: torch.Tensor) -> torch.Tensor:
    return images[0].detach().float().clamp(0, 1).permute(1, 2, 0).mul(255).round()


def _pixels_from_image(image: Image.Image) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8
    ).reshape(image.height, image.width, 3)


@pytest.mark.vendor_parity
@pytest.mark.integration
def test_pinned_space_noise_and_checkpoint_outputs_are_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    tkgdm, source_hash = _reference_tkgdm(space_root / "tkg_dm.py")
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
        space_model_type = {
            "stable-diffusion-1-5": "sd1.5",
            "stable-diffusion-2-1": "sd2.1",
            "stable-diffusion-xl": "sdxl",
        }[name]
        base_class = (
            StableDiffusionXLPipeline if family == "xl" else StableDiffusionPipeline
        )
        space_pipeline_class = tkgdm.TKGDMPipeline
        model_configs = {
            key: dict(value)
            for key, value in space_pipeline_class.MODEL_CONFIGS.items()
        }
        model_configs[space_model_type]["default_model"] = model_id
        monkeypatch.setattr(space_pipeline_class, "MODEL_CONFIGS", model_configs)
        space_instances: list[object] = []
        reference_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        scheduler_records: list[tuple[str, str]] = []

        with monkeypatch.context() as model_patch:
            original_loader = base_class.from_pretrained

            def load_pinned_checkpoint(
                _cls: type,
                checkpoint: str,
                _revision: str = revision,
                _loader: Callable[..., object] = original_loader,
                **kwargs: object,
            ) -> object:
                kwargs["revision"] = _revision
                return _loader(checkpoint, **kwargs)

            model_patch.setattr(
                base_class,
                "from_pretrained",
                classmethod(load_pinned_checkpoint),
            )

            def create_space_pipeline(
                *,
                model_id: str | None,
                model_type: str,
                device: str,
                _factory: Callable[..., object] = space_pipeline_class,
                _instances: list[object] = space_instances,
                _samples: list[tuple[torch.Tensor, torch.Tensor]] = reference_samples,
                _records: list[tuple[str, str]] = scheduler_records,
                _patch: pytest.MonkeyPatch = model_patch,
            ) -> object:
                space_pipeline = _factory(
                    model_id=model_id,
                    model_type=model_type,
                    device=device,
                )
                _instances.append(space_pipeline)
                space_view = cast(SpacePipeline, space_pipeline)
                space_pipe = space_view.pipe
                if space_pipe is None:
                    raise RuntimeError("Pinned Space pipeline failed to load")

                optimizer = space_view.noise_optimizer
                original_blend = optimizer.blend_noise_with_mask

                def record_space_noise(
                    original_noise: torch.Tensor,
                    shifted: torch.Tensor,
                    mask: torch.Tensor,
                    _blend: Callable[..., torch.Tensor] = original_blend,
                    _captured: list[tuple[torch.Tensor, torch.Tensor]] = _samples,
                ) -> torch.Tensor:
                    result = _blend(original_noise, shifted, mask)
                    _captured.append(
                        (original_noise.detach().clone(), result.detach().clone())
                    )
                    return result

                _patch.setattr(optimizer, "blend_noise_with_mask", record_space_noise)
                _records.append(
                    (
                        space_pipe.scheduler.config._class_name,
                        _config_sha256(space_pipe.scheduler.config),
                    )
                )
                return space_pipeline

            generate_space_image, app_hash = _reference_app(
                space_root / "app.py",
                create_space_pipeline,
            )
            torch.manual_seed(SEED)
            reference_image = generate_space_image(
                PROMPT,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                25,
                0.07,
                0.0,
                space_model_type,
                "",
                ";".join(",".join(map(str, box)) for box in BOXES),
            )

        if len(reference_samples) != 1 or len(scheduler_records) != 1:
            raise RuntimeError("Pinned Space app did not use one expected noise sample")

        epsilon, reference_noise = reference_samples[0]
        package_noise = apply_sawna_noise(epsilon, BOXES)
        assert torch.equal(package_noise, reference_noise), f"{name} noise differs"

        def package_generator(
            _epsilon: torch.Tensor = epsilon,
            _name: str = name,
        ) -> torch.Generator:
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            seeded_epsilon = torch.randn(
                _epsilon.shape,
                device="cuda",
                dtype=_epsilon.dtype,
                generator=generator,
            )
            assert torch.equal(seeded_epsilon, _epsilon), f"{_name} seed differs"
            return generator

        generator = package_generator()
        repeated_generator = package_generator()
        epsilon = epsilon.cpu()
        reference_noise = reference_noise.cpu()
        package_noise = package_noise.cpu()
        del (
            generate_space_image,
            create_space_pipeline,
            space_instances,
            reference_samples,
        )
        gc.collect()
        torch.cuda.empty_cache()

        package_arguments: dict[str, object] = {
            "revision": revision,
            "torch_dtype": torch.float16,
        }
        if family == "xl":
            package_arguments.update(use_safetensors=True, variant="fp16")
            package_pipe = SAWNAStableDiffusionXLPipeline.from_pretrained(
                model_id,
                **package_arguments,
            ).to("cuda")
        else:
            package_arguments.update(
                safety_checker=None,
                requires_safety_checker=False,
            )
            package_pipe = SAWNAStableDiffusionPipeline.from_pretrained(
                model_id,
                **package_arguments,
            ).to("cuda")
        package_pipe.set_progress_bar_config(disable=True)
        package_images = _run_pipeline(
            package_pipe,
            size=size,
            latents=package_noise,
            generator=generator,
        )
        repeated_images = _run_pipeline(
            package_pipe,
            size=size,
            latents=package_noise,
            generator=repeated_generator,
        )
        reference_pixels = _pixels_from_image(reference_image)
        package_pixels = _pixels_from_tensor(package_images).byte().cpu()
        assert torch.equal(package_pixels, reference_pixels), f"{name} output differs"
        assert torch.equal(package_images, repeated_images), f"{name} is not repeatable"

        image_path = output_root / f"sawna-{name}-lion-seed{SEED}.png"
        _save_example(package_images, image_path)
        torch.save(
            {
                "epsilon": epsilon.cpu(),
                "epsilon_masked": package_noise.cpu(),
                "reference_images": reference_pixels.cpu(),
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
                "scheduler": scheduler_records[0][0],
                "scheduler_config_sha256": scheduler_records[0][1],
                "epsilon_masked_bitwise_equal": True,
                "generated_images_bitwise_equal": True,
                "repeatable": True,
                "image_sha256": _sha256_bytes(image_path.read_bytes()),
            }
        )
        del package_pipe
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
        "space_app_sha256": app_hash,
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
