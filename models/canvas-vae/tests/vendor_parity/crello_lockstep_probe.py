"""Compare original and package Crello training for the lockstep gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

ProbeJson: TypeAlias = (
    str | int | float | bool | None | list["ProbeJson"] | dict[str, "ProbeJson"]
)
LOSS_RELATIVE_LIMIT: float = 1e-3


@dataclass(frozen=True)
class ProbeConfig:
    data_dir: Path
    original_data_dir: Path
    fixture_dir: Path
    device: Literal["cpu", "cuda"]
    steps: int
    batch_size: int
    seed: int
    jsonl_output: Path
    summary_output: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_summary(path: Path, payload: dict[str, ProbeJson]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_probe(config: ProbeConfig) -> dict[str, ProbeJson]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if config.device == "cuda" else ""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    import numpy as np
    import tensorflow as tf
    import torch

    repository_root = Path(__file__).resolve().parents[4]
    scripts_path = repository_root / "models" / "canvas-vae" / "scripts"
    sys.path.insert(0, str(scripts_path))
    import compare_crello_training as comparison
    from generate_vendor_reference import disable_tensorflow_tf32
    from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
    from canvas_vae.data import CrelloBatch, CrelloSplit
    from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
    from canvas_vae.training.precision import disable_torch_tf32
    from canvas_vae.training.sampling import CrossEpochBatchSampler
    from traingen.optim import KerasAdam

    comparison_path = Path(comparison.__file__).resolve()
    if comparison_path.parent != scripts_path.resolve():
        raise RuntimeError(
            "the Crello comparison helper loaded from an unexpected path"
        )

    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("probe steps and batch size must be positive")
    disable_tensorflow_tf32(tf)
    disable_torch_tf32()
    if config.device == "cuda":
        if not tf.config.list_physical_devices("GPU") or not torch.cuda.is_available():
            raise RuntimeError(
                "the lockstep probe requested CUDA but no GPU is available"
            )
        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
        torch_device = torch.device("cuda:0")
        torch.cuda.set_device(torch_device)
    else:
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("CPU probe unexpectedly sees a TensorFlow GPU")
        torch_device = torch.device("cpu")

    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    data_module = CanvasVAECrelloDataModule(
        data_dir=str(config.data_dir),
        fixture_dir=str(config.fixture_dir),
        batch_size=config.batch_size,
        seed=config.seed,
    )
    data_module.setup("fit")
    documents = data_module.splits[CrelloSplit.train].documents
    if not documents:
        raise ValueError("the Crello training split is empty")
    if data_module.processor is None:
        raise RuntimeError("the Crello data module did not construct its processor")

    processor = data_module.processor
    vocabularies = processor.vocabularies
    model_config = CanvasVAECrelloConfig(
        vocabularies=vocabularies,
        latent_dim=comparison.LATENT_DIM,
        dropout=0.0,
        kl_weight=comparison.KL_WEIGHT,
        l2_weight=comparison.L2_WEIGHT,
    )
    tensorflow_device = "/GPU:0" if config.device == "cuda" else "/CPU:0"
    with tf.device(tensorflow_device):
        _, _columns, original_model = comparison._crello_model(
            tf, config.original_data_dir, dropout=0.0
        )

    original_initial_state = comparison.convert_tensorflow_crello_variables(
        comparison._vendor_variables(original_model), model_config
    )
    package_model = comparison.CanvasVAECrelloModel(model_config).to(torch_device)
    original_initial_state_on_device = {
        key: comparison._tensor_on_model_device(value, package_model)
        for key, value in original_initial_state.items()
    }
    package_model.load_state_dict(
        original_initial_state_on_device,
        strict=True,
    )
    initial_state_sha256 = comparison._torch_state_sha256(package_model.state_dict())
    initial_state_errors = [
        key
        for key, value in package_model.state_dict().items()
        if not torch.equal(value, original_initial_state_on_device[key])
    ]
    del original_initial_state_on_device
    del original_initial_state
    static_errors = comparison._common(original_model, package_model)
    if initial_state_errors or static_errors:
        raise RuntimeError(
            "initial model parity failed: "
            + json.dumps(
                {
                    "initial_state_mismatch_keys": initial_state_errors,
                    "static_errors": static_errors,
                },
                sort_keys=True,
            )
        )

    noise_floor_package = copy.deepcopy(package_model)
    noise_floor_perturbed = copy.deepcopy(package_model)
    noise_floor_initial_sha256 = comparison._torch_state_sha256(
        noise_floor_package.state_dict()
    )
    perturbed_parameter_values = comparison._perturb_parameters_by_one_ulp(
        noise_floor_perturbed
    )
    if comparison._torch_state_sha256(noise_floor_package.state_dict()) != (
        initial_state_sha256
    ):
        raise RuntimeError(
            "noise-floor package copy differs from the cross-system initialization"
        )
    if noise_floor_initial_sha256 != initial_state_sha256:
        raise RuntimeError(
            "noise-floor initial state does not match the cross-system state"
        )

    def batch_stream():
        sampler_generator = torch.Generator(device="cpu").manual_seed(config.seed)
        sampler = CrossEpochBatchSampler(
            len(documents), config.batch_size, sampler_generator
        )
        sampler_batches = iter(sampler)
        noise_generator = np.random.default_rng(config.seed)
        for step in range(config.steps):
            try:
                batch_indices = next(sampler_batches)
            except StopIteration:
                sampler_batches = iter(sampler)
                batch_indices = next(sampler_batches)

            rows = [documents[index] for index in batch_indices]
            batch = cast(CrelloBatch, processor(rows))
            batch_digest = comparison._batch_digest(batch)
            noise = noise_generator.standard_normal(
                (len(rows), comparison.LATENT_DIM), dtype=np.float32
            )
            sampler_state: dict[str, ProbeJson] = {
                "seed": config.seed,
                "epoch": step // len(sampler),
                "batch_in_epoch": step % len(sampler),
                "source_indices": batch_indices,
                "generator_state_sha256": _sha256_bytes(
                    sampler_generator.get_state().numpy().tobytes()
                ),
                "batch_sha256": batch_digest,
                "posterior_noise_sha256": _sha256_bytes(noise.tobytes()),
            }
            yield batch, noise, sampler_state

    def ensure_finite(record: dict[str, ProbeJson]) -> None:
        for key in (
            "loss_left",
            "loss_right",
            "relative_loss_difference",
            "gradient_norm_left",
            "gradient_norm_right",
            "parameter_max_abs_difference",
        ):
            value = record[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"lockstep probe produced a non-finite {key}")

    def first_over_tolerance(
        records: list[dict[str, ProbeJson]],
    ) -> dict[str, ProbeJson] | None:
        for record in records:
            difference = cast(float, record["relative_loss_difference"])
            if difference > LOSS_RELATIVE_LIMIT:
                return {
                    "step": record["step"],
                    "relative_loss_difference": difference,
                    "parameter_with_max_difference": record[
                        "parameter_with_max_difference"
                    ],
                    "parameter_max_abs_difference": record[
                        "parameter_max_abs_difference"
                    ],
                }

        return None

    phase_records: dict[str, list[dict[str, ProbeJson]]] = {
        "cross_system": [],
        "noise_floor": [],
    }
    sampler_signatures: dict[str, list[str]] = {
        "cross_system": [],
        "noise_floor": [],
    }
    package_optimizer = KerasAdam(
        package_model.parameters(), lr=comparison.LEARNING_RATE
    )
    noise_floor_optimizer = KerasAdam(
        noise_floor_package.parameters(), lr=comparison.LEARNING_RATE
    )
    noise_floor_perturbed_optimizer = KerasAdam(
        noise_floor_perturbed.parameters(), lr=comparison.LEARNING_RATE
    )

    config.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    with config.jsonl_output.open("x", encoding="utf-8") as output:
        for phase in ("cross_system", "noise_floor"):
            for step, (batch, noise, sampler_state) in enumerate(batch_stream()):
                step_started = time.monotonic()
                sampler_signatures[phase].append(
                    json.dumps(sampler_state, sort_keys=True)
                )
                if phase == "cross_system":
                    reseed = config.seed + step
                    torch.manual_seed(reseed)
                    if config.device == "cuda":
                        torch.cuda.manual_seed_all(reseed)

                    step_summary: dict[str, ProbeJson] = {}
                    comparison._train_step(
                        tf,
                        original_model,
                        package_model,
                        package_optimizer,
                        batch,
                        noise,
                        step,
                        [],
                        [],
                        [],
                        [],
                        [],
                        step_summary,
                    )
                    metric_values = {
                        "loss_left": step_summary["loss_original"],
                        "loss_right": step_summary["loss_package"],
                        "relative_loss_difference": abs(
                            cast(float, step_summary["loss_package"])
                            - cast(float, step_summary["loss_original"])
                        )
                        / max(abs(cast(float, step_summary["loss_original"])), 1e-12),
                        "gradient_norm_left": step_summary["gradient_norm_original"],
                        "gradient_norm_right": step_summary["gradient_norm_package"],
                        "parameter_max_abs_difference": step_summary[
                            "parameter_max_abs_difference"
                        ],
                        "parameter_with_max_difference": step_summary[
                            "parameter_with_max_difference"
                        ],
                    }
                    comparison_name = "tensorflow_vs_package"
                else:
                    metric_values = comparison._package_lockstep_step(
                        noise_floor_package,
                        noise_floor_perturbed,
                        noise_floor_optimizer,
                        noise_floor_perturbed_optimizer,
                        batch,
                        noise,
                        config.seed + step,
                    )
                    comparison_name = "package_vs_one_ulp_perturbed_package"

                step_seconds = time.monotonic() - step_started
                record: dict[str, ProbeJson] = {
                    "phase": phase,
                    "comparison": comparison_name,
                    "step": step + 1,
                    **metric_values,
                    "sampler_state": sampler_state,
                    "step_seconds_report_only": step_seconds,
                    "documents_per_second_report_only": len(
                        cast(list[str], sampler_state["source_indices"])
                    )
                    / max(step_seconds, 1e-12),
                }
                ensure_finite(record)
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                phase_records[phase].append(record)

    if sampler_signatures["cross_system"] != sampler_signatures["noise_floor"]:
        raise RuntimeError(
            "noise-floor sampler or posterior reseeding differs from cross-system probe"
        )

    cross_first = first_over_tolerance(phase_records["cross_system"])
    noise_first = first_over_tolerance(phase_records["noise_floor"])
    cross_step = cast(int | None, cross_first["step"] if cross_first else None)
    noise_step = cast(int | None, noise_first["step"] if noise_first else None)
    attribution_passed = cross_step is None or (
        noise_step is not None and cross_step >= noise_step
    )
    phases: dict[str, ProbeJson] = {}
    cross_sampler_batch_sha256 = ""
    for phase, records in phase_records.items():
        first = cross_first if phase == "cross_system" else noise_first
        batch_hashes: list[str] = []
        for record in records:
            sampler_state = record.get("sampler_state")
            batch_hash = (
                sampler_state.get("batch_sha256")
                if isinstance(sampler_state, dict)
                else None
            )
            if not isinstance(batch_hash, str):
                raise RuntimeError("lockstep probe sampler batch hash is missing")
            batch_hashes.append(batch_hash)
        phase_sampler_batch_sha256 = _sha256_bytes(
            "\n".join(batch_hashes).encode("ascii")
        )
        if phase == "cross_system":
            cross_sampler_batch_sha256 = phase_sampler_batch_sha256

        phases[phase] = {
            "required_steps": config.steps,
            "completed_steps": len(records),
            "first_step_over_tolerance": first,
            "maximum_relative_loss_difference": max(
                cast(float, record["relative_loss_difference"]) for record in records
            ),
            "sampler_batch_sha256": phase_sampler_batch_sha256,
        }

    return {
        "protocol": "crello-real-scale-lockstep-v2",
        "source_commit": os.environ.get("CANVAS_VAE_S5_SOURCE_COMMIT", "unknown"),
        "probe_script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "comparison_script_sha256": _sha256_bytes(comparison_path.read_bytes()),
        "status": "completed",
        "device": config.device,
        "required_steps": config.steps,
        "completed_steps": config.steps,
        "completed_steps_per_phase": config.steps,
        "batch_size": config.batch_size,
        "document_count": len(documents),
        "initial_state_sha256": initial_state_sha256,
        "noise_floor_initial_state_sha256": noise_floor_initial_sha256,
        "noise_floor_perturbed_parameter_values": perturbed_parameter_values,
        "relative_loss_limit": LOSS_RELATIVE_LIMIT,
        "phases": phases,
        "first_step_over_tolerance": cross_first,
        "noise_floor_first_step_over_tolerance": noise_first,
        "noise_floor_attribution_passed": attribution_passed,
        "sampler": "CrossEpochBatchSampler",
        "sampler_batch_sha256": cross_sampler_batch_sha256,
        "gate_fields": ["finite_metrics", "noise_floor_attribution"],
        "throughput_fields_are_report_only": True,
    }


def _parse_args() -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--original-data-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    return ProbeConfig(
        data_dir=args.data_dir,
        original_data_dir=args.original_data_dir,
        fixture_dir=args.fixture_dir,
        device=cast(Literal["cpu", "cuda"], args.device),
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        jsonl_output=args.jsonl_output,
        summary_output=args.summary_output,
    )


def main() -> None:
    config = _parse_args()
    if config.jsonl_output.exists() or config.summary_output.exists():
        raise FileExistsError("refusing to overwrite a lockstep probe artifact")

    try:
        summary = _run_probe(config)
    except Exception as error:
        _write_summary(
            config.summary_output,
            {
                "protocol": "crello-real-scale-lockstep-v2",
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "completed_steps": 0,
                "required_steps": config.steps,
            },
        )
        raise

    _write_summary(config.summary_output, summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    if (
        summary["status"] != "completed"
        or summary["completed_steps_per_phase"] != config.steps
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
