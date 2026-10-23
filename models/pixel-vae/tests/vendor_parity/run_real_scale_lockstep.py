from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, TypedDict, cast

import numpy as np
import tensorflow as tf
import torch

from pixel_vae import PixelVAEConfig, PixelVAEImageProcessor, PixelVAEModel
from pixel_vae.conversion import (
    convert_tensorflow_variables,
    extract_tensorflow_variable_map,
    extract_tensorflow_weights,
    tensorflow_arrays_to_torch,
)
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


class LockstepSamplerManifest(TypedDict):
    sampler_order: list[int]
    record_locations: list[list[int]]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-only", action="store_true")
    return parser.parse_args()


def _should_stop_after_divergence(
    first_divergence: dict[str, object] | None, record_only: bool
) -> bool:
    return first_divergence is not None and not record_only


def _retain_first_divergence(
    first_divergence: dict[str, object] | None, current_record: dict[str, object]
) -> dict[str, object]:
    return first_divergence if first_divergence is not None else current_record


def _configure_devices(device: str) -> torch.device:
    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        raise RuntimeError(
            f"TensorFlow sees {len(physical_gpus)} GPUs; the vendor reference must use CPU"
        )

    expected_torch_gpus = 1 if device == "cuda" else 0
    visible_torch_gpus = torch.cuda.device_count()
    if visible_torch_gpus != expected_torch_gpus:
        raise RuntimeError(
            f"PyTorch sees {visible_torch_gpus} GPUs; expected {expected_torch_gpus}"
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


def _sampler_manifest(args: argparse.Namespace) -> LockstepSamplerManifest:
    with tempfile.TemporaryDirectory(prefix="pixelvae-lockstep-") as temporary:
        manifest_path = Path(temporary) / "sampler.json"
        worker_env = os.environ.copy()
        tests_path = str(Path(__file__).resolve().parents[1])
        current_pythonpath = worker_env.get("PYTHONPATH")
        worker_env["PYTHONPATH"] = (
            tests_path
            if not current_pythonpath
            else os.pathsep.join((tests_path, current_pythonpath))
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vendor_parity.run_lockstep_sampler_manifest",
                "--data-dir",
                str(args.data_dir),
                "--batch-size",
                str(args.batch_size),
                "--steps",
                str(args.steps),
                "--output",
                str(manifest_path),
            ],
            check=True,
            env=worker_env,
        )
        manifest = cast(
            LockstepSamplerManifest,
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    sample_count = args.batch_size * args.steps
    if (
        len(manifest["sampler_order"]) != sample_count
        or len(manifest["record_locations"]) != sample_count
    ):
        raise RuntimeError("the sampler manifest has an unexpected number of entries")

    return manifest


def _read_image_batch(
    record_locations: list[list[int]],
    record_paths: list[Path],
    handles: dict[int, BinaryIO],
) -> list[bytes]:
    images: list[bytes] = []
    for file_index, payload_offset, payload_size in record_locations:
        if file_index < 0 or file_index >= len(record_paths):
            raise RuntimeError("the sampler manifest contains an invalid shard index")

        handle = handles.get(file_index)
        if handle is None:
            handle = record_paths[file_index].open("rb")
            handles[file_index] = handle

        handle.seek(payload_offset)
        payload = handle.read(payload_size)
        if len(payload) != payload_size:
            raise RuntimeError("a lockstep TFRecord payload was truncated")

        example = tf.train.Example.FromString(payload)
        try:
            images.append(example.features.feature["image"].bytes_list.value[0])
        except IndexError as error:
            raise ValueError("a lockstep TFRecord has no PNG image") from error

    return images


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

    if args.num_workers <= 0:
        raise ValueError("lockstep requires persistent training workers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite lockstep evidence")

    manifest = _sampler_manifest(args)
    sampler_order = manifest["sampler_order"]
    record_locations = manifest["record_locations"]
    record_paths = sorted(args.data_dir.glob("train-*.tfrecord"))
    if not record_paths:
        raise FileNotFoundError("no train TFRecord shards were found")

    processor = PixelVAEImageProcessor()
    handles: dict[int, BinaryIO] = {}
    first_divergence = None
    records_written = 0
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            for step in range(args.steps):
                batch_start = step * args.batch_size
                batch_end = batch_start + args.batch_size
                images = _read_image_batch(
                    record_locations[batch_start:batch_end], record_paths, handles
                )
                pixels = cast(torch.Tensor, processor(images)["pixel_values"])
                if pixels.shape[0] != args.batch_size:
                    raise RuntimeError("the lockstep batch has an unexpected size")

                tf_batch = tf.stack(
                    [tf.io.decode_png(image, channels=4) for image in images]
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
                    first_divergence = _retain_first_divergence(
                        first_divergence, record
                    )
                    if _should_stop_after_divergence(
                        first_divergence, args.record_only
                    ):
                        break
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "status": "passed" if first_divergence is None else "diverged",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "steps_requested": args.steps,
        "steps_completed": records_written,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "first_relative_loss_difference_over_1e-3": first_divergence,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    if _should_stop_after_divergence(first_divergence, args.record_only):
        raise SystemExit("real-scale lockstep exceeded the relative loss limit")
    if records_written != args.steps:
        raise SystemExit("real-scale lockstep did not complete the requested steps")


if __name__ == "__main__":
    main()
