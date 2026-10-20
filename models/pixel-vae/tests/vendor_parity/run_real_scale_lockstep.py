from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import tensorflow as tf
import torch

from pixel_vae import PixelVAEConfig, PixelVAEModel
from pixel_vae.conversion import (
    convert_tensorflow_variables,
    extract_tensorflow_variable_map,
    extract_tensorflow_weights,
    tensorflow_arrays_to_torch,
)
from pixel_vae.training import PixelVAEDataModule
from traingen.optim import KerasAdam

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "vendor" / "canvas-vae" / "src" / "pixel-vae"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixelvae.model import PixelVAE  # noqa: E402
from test_pixel_vae_cpu_parity import (  # noqa: E402
    _compare_batch_norm_state,
    _compare_trainable_parameters,
    _source_gradient_arrays,
    _tensorflow_train_step,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure_devices(device: str) -> torch.device:
    physical_gpus = tf.config.list_physical_devices("GPU")
    expected_count = 1 if device == "cuda" else 0
    if len(physical_gpus) != expected_count:
        raise RuntimeError(
            f"TensorFlow sees {len(physical_gpus)} GPUs; expected {expected_count}"
        )

    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    visible_torch_gpus = torch.cuda.device_count()
    if visible_torch_gpus != expected_count:
        raise RuntimeError(
            f"PyTorch sees {visible_torch_gpus} GPUs; expected {expected_count}"
        )

    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(0)

    return torch.device(device)


def _gradient_norm(gradients: list[tf.Tensor]) -> float:
    return float(tf.linalg.global_norm(gradients).numpy())


def _package_gradient_norm(model: PixelVAEModel) -> float:
    gradients = [
        parameter.grad.detach().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("the package model produced no gradients")

    return float(torch.linalg.vector_norm(torch.stack(gradients)).cpu())


def _batch_digest(pixels: torch.Tensor) -> str:
    return hashlib.sha256(
        pixels.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    args = _arguments()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")

    device = _configure_devices(args.device)
    reference = PixelVAE(latent_dim=256, kl=100.0, l2=1e-6)
    source_weights = extract_tensorflow_weights(reference)
    model = PixelVAEModel(PixelVAEConfig())
    conversion = convert_tensorflow_variables(source_weights, model)
    transformed, mapping_report = tensorflow_arrays_to_torch(
        source_weights, model, strict=True
    )
    if mapping_report.assigned_tensors != conversion.assigned_tensors:
        raise RuntimeError("strict conversion and state-map tensor counts differ")

    model.load_state_dict(transformed)
    model.to(device).train()
    source_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
    package_optimizer = KerasAdam(model.parameters(), lr=1e-4, eps=1e-7)
    variable_map = extract_tensorflow_variable_map(reference)

    data = PixelVAEDataModule(
        str(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data.setup("fit")
    sampler = data.samplers["train"]
    sampler.set_epoch(0)
    sampler_order = list(iter(sampler))
    sampler.set_epoch(0)
    loader = data.train_dataloader()
    iterator = iter(loader)
    if loader.num_workers != args.num_workers or not loader.persistent_workers:
        raise RuntimeError("lockstep loader settings differ from the production config")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite lockstep evidence")

    first_divergence = None
    records_written = 0
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            for step in range(args.steps):
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "the real train stream ended before the probe"
                    ) from error

                pixels = cast(torch.Tensor, batch["pixel_values"])
                if pixels.shape[0] != args.batch_size:
                    raise RuntimeError("the lockstep batch has an unexpected size")

                tf_batch = tf.convert_to_tensor(
                    pixels.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
                )
                epsilon = (
                    np.random.default_rng(310 + step)
                    .standard_normal((args.batch_size, 256))
                    .astype(np.float32)
                )
                source_loss, source_gradients, _ = _tensorflow_train_step(
                    reference, source_optimizer, tf_batch, epsilon
                )
                package_optimizer.zero_grad(set_to_none=True)
                package_output = model(
                    pixels.to(device, non_blocking=True),
                    sample=True,
                    epsilon=torch.from_numpy(epsilon).to(device),
                )
                if package_output.loss is None:
                    raise RuntimeError("the package model did not return a loss")
                package_output.loss.backward()

                source_gradient_arrays = _source_gradient_arrays(
                    reference, variable_map, source_gradients
                )
                mapped_gradients, _ = tensorflow_arrays_to_torch(
                    source_gradient_arrays, model, strict=False
                )
                package_gradients = {
                    name: parameter.grad.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                if set(mapped_gradients) != set(package_gradients):
                    raise RuntimeError("training gradient keys differ between systems")

                source_value = float(source_loss.numpy())
                package_value = float(package_output.loss.detach().cpu())
                relative_loss_difference = abs(source_value - package_value) / max(
                    abs(source_value), 1e-12
                )
                gradient_difference = max(
                    float(
                        torch.max(
                            torch.abs(mapped_gradients[name] - package_gradients[name])
                        )
                    )
                    for name in mapped_gradients
                )
                batch_start = step * args.batch_size
                batch_indices = sampler_order[
                    batch_start : batch_start + args.batch_size
                ]
                if len(batch_indices) != args.batch_size:
                    raise RuntimeError("the sampler did not provide a complete batch")

                source_optimizer.apply_gradients(
                    zip(source_gradients, reference.trainable_variables, strict=True)
                )
                package_optimizer.step()
                parameter_difference = _compare_trainable_parameters(reference, model)
                batch_norm_difference = _compare_batch_norm_state(reference, model)
                record = {
                    "step": step + 1,
                    "source_loss": source_value,
                    "package_loss": package_value,
                    "relative_loss_difference": relative_loss_difference,
                    "source_gradient_norm": _gradient_norm(source_gradients),
                    "package_gradient_norm": _package_gradient_norm(model),
                    "max_gradient_difference": gradient_difference,
                    "max_parameter_difference": parameter_difference,
                    "max_batch_norm_difference": batch_norm_difference,
                    "sampler_state": {
                        "epoch": 0,
                        "batch_index": step,
                        "record_indices": batch_indices,
                        "batch_tensor_sha256": _batch_digest(pixels),
                    },
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                records_written += 1
                if relative_loss_difference > 1e-3:
                    first_divergence = record
                    break
    finally:
        iterator_state = getattr(loader, "_iterator", None)
        if iterator_state is not None:
            iterator_state._shutdown_workers()

    summary = {
        "status": "passed" if first_divergence is None else "diverged",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "steps_requested": args.steps,
        "steps_completed": records_written,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": loader.pin_memory,
        "persistent_workers": loader.persistent_workers,
        "first_relative_loss_difference_over_1e-3": first_divergence,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    if first_divergence is not None:
        raise SystemExit("real-scale lockstep exceeded the relative loss limit")
    if records_written != args.steps:
        raise SystemExit("real-scale lockstep did not complete the requested steps")


if __name__ == "__main__":
    main()
