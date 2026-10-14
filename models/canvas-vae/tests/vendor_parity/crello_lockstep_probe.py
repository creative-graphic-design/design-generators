"""Compare original and package Crello training for the lockstep gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
    from canvas_vae.data import CrelloBatch, CrelloSplit
    from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
    from canvas_vae.training.sampling import CrossEpochBatchSampler
    from traingen.optim import KerasAdam

    comparison_path = Path(comparison.__file__).resolve()
    if comparison_path.parent != scripts_path.resolve():
        raise RuntimeError(
            "the Crello comparison helper loaded from an unexpected path"
        )

    if config.steps < 1 or config.batch_size < 1:
        raise ValueError("probe steps and batch size must be positive")
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

    package_optimizer = KerasAdam(
        package_model.parameters(), lr=comparison.LEARNING_RATE
    )
    sampler_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    sampler = CrossEpochBatchSampler(
        len(documents), config.batch_size, sampler_generator
    )
    sampler_batches = iter(sampler)
    noise_generator = np.random.default_rng(config.seed)
    batch_digests: list[str] = []
    maximum_relative_loss_difference = 0.0
    first_failure: dict[str, ProbeJson] | None = None

    config.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    with config.jsonl_output.open("x", encoding="utf-8") as output:
        for step in range(config.steps):
            step_started = time.monotonic()
            try:
                batch_indices = next(sampler_batches)
            except StopIteration:
                sampler_batches = iter(sampler)
                batch_indices = next(sampler_batches)

            rows = [documents[index] for index in batch_indices]
            batch = cast(CrelloBatch, processor(rows))
            batch_digest = comparison._batch_digest(batch)
            batch_digests.append(batch_digest)
            noise = noise_generator.standard_normal(
                (len(rows), comparison.LATENT_DIM), dtype=np.float32
            )
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
            loss_original = cast(float, step_summary["loss_original"])
            loss_package = cast(float, step_summary["loss_package"])
            relative_loss_difference = abs(loss_package - loss_original) / max(
                abs(loss_original), 1e-12
            )
            maximum_relative_loss_difference = max(
                maximum_relative_loss_difference, relative_loss_difference
            )
            step_seconds = time.monotonic() - step_started
            sampler_state: dict[str, ProbeJson] = {
                "seed": config.seed,
                "epoch": step // len(sampler),
                "batch_in_epoch": step % len(sampler),
                "source_indices": batch_indices,
                "generator_state_sha256": _sha256_bytes(
                    sampler_generator.get_state().numpy().tobytes()
                ),
                "batch_sha256": batch_digest,
            }
            record: dict[str, ProbeJson] = {
                "step": step + 1,
                "loss_original": loss_original,
                "loss_package": loss_package,
                "relative_loss_difference": relative_loss_difference,
                "gradient_norm_original": cast(
                    float, step_summary["gradient_norm_original"]
                ),
                "gradient_norm_package": cast(
                    float, step_summary["gradient_norm_package"]
                ),
                "parameter_max_abs_difference": cast(
                    float, step_summary["parameter_max_abs_difference"]
                ),
                "parameter_with_max_difference": cast(
                    str | None, step_summary["parameter_with_max_difference"]
                ),
                "sampler_state": sampler_state,
                "step_seconds_report_only": step_seconds,
                "documents_per_second_report_only": len(rows)
                / max(step_seconds, 1e-12),
            }
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()

            if relative_loss_difference > LOSS_RELATIVE_LIMIT:
                first_failure = {
                    "step": step + 1,
                    "relative_loss_difference": relative_loss_difference,
                    "parameter_with_max_difference": cast(
                        str | None, step_summary["parameter_with_max_difference"]
                    ),
                    "parameter_max_abs_difference": cast(
                        float, step_summary["parameter_max_abs_difference"]
                    ),
                }
                break

    completed_steps = len(batch_digests)
    return {
        "protocol": "crello-real-scale-lockstep-v1",
        "source_commit": os.environ.get("CANVAS_VAE_S5_SOURCE_COMMIT", "unknown"),
        "probe_script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "comparison_script_sha256": _sha256_bytes(comparison_path.read_bytes()),
        "status": "failed" if first_failure is not None else "passed",
        "device": config.device,
        "required_steps": config.steps,
        "completed_steps": completed_steps,
        "batch_size": config.batch_size,
        "document_count": len(documents),
        "initial_state_sha256": initial_state_sha256,
        "relative_loss_limit": LOSS_RELATIVE_LIMIT,
        "maximum_relative_loss_difference": maximum_relative_loss_difference,
        "first_step_over_tolerance": first_failure,
        "sampler": "CrossEpochBatchSampler",
        "sampler_batch_sha256": _sha256_bytes("\n".join(batch_digests).encode("ascii")),
        "gate_fields": ["relative_loss_difference"],
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
                "protocol": "crello-real-scale-lockstep-v1",
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "completed_steps": 0,
                "required_steps": config.steps,
            },
        )
        raise

    _write_summary(config.summary_output, summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    if summary["status"] != "passed" or summary["completed_steps"] != config.steps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
