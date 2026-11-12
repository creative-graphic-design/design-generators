"""Run and compare the Crello lockstep in framework-isolated processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any  # noqa: TID251 -- bridges TensorFlow, PyTorch, and NumPy runtime values

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
EXPECTED_INITIAL_STATE_SHA256 = (
    "5e6ddfb3fa5c8e25148b3e969ed5f47b02d5107993423a366b643c14c58efa93"
)
EXPECTED_BATCH_SEQUENCE_SHA256 = (
    "b60c2b5dc6074c932343d81d41494d258535fc6bd6ba0991b88b0d3f02689b68"
)
RELATIVE_LOSS_LIMIT = 1e-3


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _write_record(output: Any, record: Mapping[str, Any]) -> None:
    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    output.flush()


def _read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    if not records or records[0].get("record") != "metadata":
        raise ValueError(f"missing metadata record in {path.name}")

    return records[0], records[1:]


def _state_sha256(state: Mapping[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(array.tobytes())

    return digest.hexdigest()


def _relative_loss_difference(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), 1e-12)


def _first_over_limit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        difference = float(record["relative_loss_difference"])
        if difference > RELATIVE_LOSS_LIMIT:
            return {
                "step": int(record["step"]),
                "relative_loss_difference": difference,
                "parameter_with_max_difference": record[
                    "parameter_with_max_difference"
                ],
                "parameter_max_abs_difference": float(
                    record["parameter_max_abs_difference"]
                ),
            }

    return None


def _attribution_passed(
    cross_first_step: int | None, noise_first_step: int | None
) -> bool:
    return cross_first_step is None or (
        noise_first_step is not None and cross_first_step >= noise_first_step
    )


def _parameter_difference(
    left: Mapping[str, Any], right: Mapping[str, Any], names: Sequence[str]
) -> tuple[float, str | None]:
    import numpy as np

    maximum = 0.0
    maximum_name: str | None = None
    for name in names:
        delta = np.abs(np.asarray(left[name]) - np.asarray(right[name]))
        value = float(np.max(delta, initial=0.0))
        if value > maximum:
            maximum = value
            maximum_name = name

    return maximum, maximum_name


def _make_config_and_mapping(data_dir: Path, fixture_dir: Path, batch_size: int):
    from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
    from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModel
    from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
    from canvas_vae.conversion import tensorflow_crello_key_map

    data_module = CanvasVAECrelloDataModule(
        data_dir=str(data_dir),
        fixture_dir=str(fixture_dir),
        batch_size=batch_size,
        seed=0,
    )
    data_module.setup("fit")
    if data_module.processor is None:
        raise RuntimeError("the Crello processor was not initialized")

    comparison_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "compare_crello_training.py"
    )
    sys.path.insert(0, str(comparison_path.parent))
    import compare_crello_training as comparison

    config = CanvasVAECrelloConfig(
        vocabularies=data_module.processor.vocabularies,
        latent_dim=comparison.LATENT_DIM,
        dropout=0.0,
        kl_weight=comparison.KL_WEIGHT,
        l2_weight=comparison.L2_WEIGHT,
    )
    model = CanvasVAECrelloModel(config)
    mapping = tensorflow_crello_key_map(config)
    state_keys = sorted(mapping)
    state_dict = model.state_dict()
    if set(state_keys) != set(state_dict):
        raise RuntimeError("TensorFlow mapping does not cover the package state")

    parameter_keys = [name for name, _ in model.named_parameters()]
    if not set(parameter_keys).issubset(state_keys):
        raise RuntimeError("TensorFlow mapping omits package parameters")

    manifest = {
        "state_keys": state_keys,
        "parameter_keys": parameter_keys,
        "parameter_shapes": {
            name: list(state_dict[name].shape) for name in parameter_keys
        },
        "state_shapes": {name: list(state_dict[name].shape) for name in state_keys},
        "source_map": {
            key: {"source_key": value.key, "transpose": value.transpose}
            for key, value in mapping.items()
        },
    }

    return data_module, config, comparison, manifest


def _prepare(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from canvas_vae.data import CrelloSplit
    from canvas_vae.training.sampling import CrossEpochBatchSampler

    run_dir = args.run_dir.resolve()
    inputs_dir = run_dir / "inputs"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir.name}")
    batches_dir = inputs_dir / "batches"
    batches_dir.mkdir(parents=True)

    data_module, _config, comparison, mapping = _make_config_and_mapping(
        args.data_dir, args.fixture_dir, args.batch_size
    )
    documents = data_module.splits[CrelloSplit.train].documents
    if not documents or data_module.processor is None:
        raise RuntimeError("production Crello training data is empty")

    (inputs_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sampler_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sampler = CrossEpochBatchSampler(len(documents), args.batch_size, sampler_generator)
    sampler_batches = iter(sampler)
    batch_indices: list[list[int]] = []
    sampler_state_hashes: list[str] = []
    for _ in range(args.steps):
        try:
            indices = next(sampler_batches)
        except StopIteration:
            sampler_batches = iter(sampler)
            indices = next(sampler_batches)

        batch_indices.append(indices)
        sampler_state_hashes.append(
            _sha256_bytes(sampler_generator.get_state().numpy().tobytes())
        )

    loader = DataLoader(
        data_module.splits[CrelloSplit.train],
        batch_sampler=batch_indices,
        collate_fn=data_module.processor,
        num_workers=args.num_workers,
    )
    noise_generator = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    batch_hashes: list[str] = []
    for step, (indices, batch) in enumerate(zip(batch_indices, loader, strict=True)):
        batch_hash = comparison._batch_digest(batch)
        batch_hashes.append(batch_hash)
        noise = noise_generator.standard_normal(
            (len(indices), comparison.LATENT_DIM), dtype=np.float32
        )
        arrays: dict[str, np.ndarray] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().contiguous().numpy()
            elif key in {"document_id", "split"}:
                arrays[key] = np.asarray(value, dtype=str)
            else:
                raise TypeError(f"unsupported batch value for {key}")
        arrays["posterior_noise"] = noise
        relative_path = Path("batches") / f"batch-{step + 1:04d}.npz"
        np.savez(inputs_dir / relative_path, allow_pickle=False, **arrays)
        records.append(
            {
                "record": "step",
                "step": step + 1,
                "epoch": step // len(sampler),
                "batch_in_epoch": step % len(sampler),
                "source_indices": indices,
                "generator_state_sha256": sampler_state_hashes[step],
                "batch_sha256": batch_hash,
                "posterior_noise_sha256": _sha256_bytes(noise.tobytes()),
                "batch_file": relative_path.as_posix(),
                "batch_file_sha256": _sha256_file(inputs_dir / relative_path),
            }
        )

    sequence_hash = _sha256_bytes("\n".join(batch_hashes).encode("ascii"))
    if sequence_hash != args.expected_batch_sha256:
        raise RuntimeError(
            f"batch sequence hash mismatch: expected {args.expected_batch_sha256}, got {sequence_hash}"
        )

    _write_jsonl(
        inputs_dir / "batch-plan.jsonl",
        [
            {
                "record": "metadata",
                "schema_version": 1,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "preparation_workers": args.num_workers,
                "seed": args.seed,
                "document_count": len(documents),
                "batch_sequence_sha256": sequence_hash,
            },
            *records,
        ],
    )
    cache_manifest = args.fixture_dir / "manifest.json"
    cache_array = args.fixture_dir / "posterior_means.npy"
    input_manifest = {
        "schema_version": 1,
        "batch_sequence_sha256": sequence_hash,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "preparation_workers": args.num_workers,
        "seed": args.seed,
        "artifacts": {
            "mapping.json": _sha256_file(inputs_dir / "mapping.json"),
            "batch-plan.jsonl": _sha256_file(inputs_dir / "batch-plan.jsonl"),
            "cache/manifest.json": _sha256_file(cache_manifest),
            "cache/posterior_means.npy": _sha256_file(cache_array),
            "original-data/train-00000-of-00001.tfrecord": _sha256_file(
                args.original_data_dir / "train-00000-of-00001.tfrecord"
            ),
        },
    }
    _write_json(inputs_dir / "manifest.json", input_manifest)
    print(json.dumps(input_manifest, sort_keys=True), flush=True)


def _read_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    inputs_dir = run_dir / "inputs"
    plan_metadata, plan_rows = _read_jsonl(inputs_dir / "batch-plan.jsonl")
    mapping = json.loads((inputs_dir / "mapping.json").read_text("utf-8"))
    if int(plan_metadata["steps"]) != len(plan_rows):
        raise ValueError("batch plan length does not match its metadata")
    actual_sequence_hash = _sha256_bytes(
        "\n".join(row["batch_sha256"] for row in plan_rows).encode("ascii")
    )
    if actual_sequence_hash != EXPECTED_BATCH_SEQUENCE_SHA256:
        raise ValueError(f"batch sequence hash mismatch: {actual_sequence_hash}")

    for row in plan_rows:
        batch_path = inputs_dir / row["batch_file"]
        if _sha256_file(batch_path) != row["batch_file_sha256"]:
            raise ValueError(f"batch file hash mismatch at step {row['step']}")

    return plan_metadata, plan_rows, mapping


def _parameter_layout(
    mapping: Mapping[str, Any],
) -> tuple[list[str], dict[str, tuple[int, int, list[int]]], int]:
    names = list(mapping["parameter_keys"])
    offsets: dict[str, tuple[int, int, list[int]]] = {}
    offset = 0
    for name in names:
        shape = [int(value) for value in mapping["parameter_shapes"][name]]
        size = math.prod(shape)
        offsets[name] = (offset, offset + size, shape)
        offset += size

    return names, offsets, offset


def _parameter_row_hash(
    row: Any, names: Sequence[str], offsets: Mapping[str, tuple[int, int, list[int]]]
) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name in names:
        start, stop, _shape = offsets[name]
        value = np.ascontiguousarray(row[start:stop], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes())

    return digest.hexdigest()


def _parameter_arrays_hash(arrays: Mapping[str, Any], names: Sequence[str]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name in names:
        value = np.ascontiguousarray(arrays[name], dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes())

    return digest.hexdigest()


def _torch_state_digest(model: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())

    return digest.hexdigest()


def _write_stack(run_dir: Path, framework: str) -> dict[str, Any]:
    import importlib.metadata
    import numpy as np

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name",
            "--format=csv,noheader",
            "--id",
            visible.split(",")[0],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    driver, gpu = [
        value.strip() for value in query.stdout.splitlines()[0].split(",", 1)
    ]
    stack: dict[str, Any] = {
        "numpy": np.__version__,
        "driver": driver,
        "gpu": gpu,
        "cuda_runtime_package": importlib.metadata.version("nvidia-cuda-runtime-cu12"),
        "cudnn_package": importlib.metadata.version("nvidia-cudnn-cu12"),
        "nccl_package": importlib.metadata.version("nvidia-nccl-cu12"),
    }
    if framework == "torch":
        import torch
        from torch.cuda import nccl

        nccl_version = nccl.version()
        stack.update(
            {
                "framework": "torch",
                "framework_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_runtime": torch.backends.cudnn.version(),
                "nccl_runtime": ".".join(str(value) for value in nccl_version),
            }
        )
    else:
        import tensorflow as tf

        build = tf.sysconfig.get_build_info()
        stack.update(
            {
                "framework": "tensorflow",
                "framework_version": tf.__version__,
                "cuda_runtime": build.get("cuda_version"),
                "cudnn_runtime": build.get("cudnn_version"),
            }
        )

    _write_json(run_dir / "stack.json", stack)
    from canvas_vae.run_metadata import write_run_metadata

    write_run_metadata(run_dir / "run-metadata.json", framework)

    return stack


def _save_initial_state(
    path: Path, state: Mapping[str, Any], state_keys: Sequence[str]
) -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, np.ndarray] = {
        f"v{index:05d}": np.ascontiguousarray(state[name], dtype=np.float32)
        for index, name in enumerate(state_keys)
    }
    with path.open("xb") as output:
        np.savez_compressed(output, allow_pickle=False, **arrays)
    return {
        "file": path.name,
        "file_sha256": _sha256_file(path),
        "state_sha256": _state_sha256(state),
        "state_keys": list(state_keys),
    }


def _load_initial_state(path: Path, mapping: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    state_keys = list(mapping["state_keys"])
    with np.load(path, allow_pickle=False) as archive:
        state = {
            name: np.ascontiguousarray(archive[f"v{index:05d}"], dtype=np.float32)
            for index, name in enumerate(state_keys)
        }
    if _state_sha256(state) != EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(
            "the shared initial state does not match the registered hash"
        )

    return state


def _tf_vendor_state(model: Any, mapping: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    variables = {
        path.replace(".", "/"): variable.numpy()
        for path, variable in model.get_weight_paths().items()
        if not path.startswith("optimizer")
    }
    source_map = mapping["source_map"]
    expected_sources = {entry["source_key"] for entry in source_map.values()}
    missing = sorted(expected_sources - variables.keys())
    unexpected = sorted(variables.keys() - expected_sources)
    if missing or unexpected:
        raise RuntimeError(
            "TensorFlow weight paths differ from the package map: "
            + json.dumps({"missing": missing, "unexpected": unexpected})
        )

    state: dict[str, Any] = {}
    for key in mapping["state_keys"]:
        source = source_map[key]
        value = np.asarray(variables[source["source_key"]], dtype=np.float32)
        state[key] = np.ascontiguousarray(value.T if source["transpose"] else value)

    return state


def _tf_parameter_arrays(
    model: Any, mapping: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    state = _tf_vendor_state(model, mapping)

    return {name: state[name] for name in names}


def _write_row(
    memory: Any,
    index: int,
    arrays: Mapping[str, Any],
    names: Sequence[str],
    offsets: Mapping[str, tuple[int, int, list[int]]],
) -> str:
    import numpy as np

    for name in names:
        start, stop, shape = offsets[name]
        value = np.ascontiguousarray(arrays[name], dtype=np.float32).reshape(-1)
        if value.size != stop - start or list(arrays[name].shape) != shape:
            raise ValueError(f"parameter shape changed for {name}")
        memory[index, start:stop] = value

    return _parameter_row_hash(memory[index], names, offsets)


def _finish_state_file(
    path: Path,
    memory: Any,
    names: Sequence[str],
    offsets: Mapping[str, tuple[int, int, list[int]]],
    row_hashes: Sequence[str],
) -> dict[str, Any]:
    memory.flush()
    del memory

    return {
        "file": path.name,
        "file_sha256": _sha256_file(path),
        "dtype": "float32",
        "steps": len(row_hashes),
        "parameter_keys": list(names),
        "parameter_shapes": {name: offsets[name][2] for name in names},
        "parameter_values_per_step": max(
            stop for _start, stop, _shape in offsets.values()
        ),
        "row_sha256": list(row_hashes),
    }


def _tf_import_and_device() -> tuple[Any, Any]:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    from generate_vendor_reference import disable_tensorflow_tf32, import_original

    tf, spec, _encoder = import_original()
    disable_tensorflow_tf32(tf)
    if tf.config.experimental.tensor_float_32_execution_enabled():
        raise RuntimeError("TensorFlow TF32 must be disabled")
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("TensorFlow found no selected GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_op_determinism()

    return tf, spec


def _preflight_original(args: argparse.Namespace) -> None:
    tf, _spec = _tf_import_and_device()
    stack = _write_stack(args.output_dir, "tensorflow")
    print(
        json.dumps({"preflight": "passed", "stack": stack}, sort_keys=True), flush=True
    )


def _preflight_package(args: argparse.Namespace) -> None:
    import torch

    scripts = args.source_root / "models" / "canvas-vae" / "scripts"
    sys.path.insert(0, str(scripts))
    import compare_crello_training as comparison
    from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModel
    from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
    from traingen_parity.determinism import DeterminismConfig, apply_determinism

    if (
        Path(comparison.__file__).resolve()
        != (scripts / "compare_crello_training.py").resolve()
    ):
        raise RuntimeError(
            "the package comparison helper loaded from an unexpected path"
        )
    apply_determinism(DeterminismConfig(seed=0))
    if not torch.cuda.is_available():
        raise RuntimeError("Torch found no selected GPU")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Torch TF32 must be disabled")
    torch.testing.assert_close(
        torch.ones((64, 64), device="cuda") @ torch.ones((64, 64), device="cuda"),
        torch.full((64, 64), 64.0, device="cuda"),
    )
    del CanvasVAECrelloModel, CanvasVAECrelloDataModule
    stack = _write_stack(args.output_dir, "torch")
    print(
        json.dumps({"preflight": "passed", "stack": stack}, sort_keys=True), flush=True
    )


def _build_original_model(
    tf: Any, spec: Any, data_dir: Path, batch_size: int
) -> tuple[Any, Mapping[str, Any]]:
    from canvasvae.models.vae import VAE

    dataspec = spec.DataSpec("crello-document", str(data_dir), batch_size=batch_size)
    columns = dataspec.make_input_columns()
    model = VAE(
        columns,
        latent_dim=512,
        decoder_type="oneshot",
        num_blocks=1,
        block_type="deepsvg",
        dropout=0.0,
        kl=32.0,
        l2=1e-6,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0))
    model.optimizer.build(model.trainable_variables)

    return model, columns


def _install_fixed_noise(tf: Any, model: Any, noise_box: list[Any]) -> None:
    from types import MethodType

    layer = model.encoder.head

    def call(self, inputs, training=False):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)
        kl_divergence = -0.5 * tf.reduce_mean(
            1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        )
        self.add_loss(self.kl * kl_divergence)
        self.add_metric(kl_divergence, name="kl_divergence")
        if training:
            z_mean += tf.exp(0.5 * z_log_sigma) * tf.convert_to_tensor(noise_box[0])

        return z_mean

    layer.call = MethodType(call, layer)


def _run_original(args: argparse.Namespace) -> None:
    import numpy as np

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.name}")
    plan_metadata, plan_rows, mapping = _read_inputs(args.run_dir)
    if int(plan_metadata["steps"]) != args.steps:
        raise ValueError("requested step count differs from the batch plan")
    tf, spec = _tf_import_and_device()
    tf.keras.utils.set_random_seed(args.seed)
    with tf.device("/GPU:0"):
        model, columns = _build_original_model(
            tf, spec, args.original_data_dir, args.batch_size
        )
    stack = _write_stack(args.output_dir, "tensorflow")
    variables = _tf_vendor_state(model, mapping)
    initial_hash = _state_sha256(variables)
    if initial_hash != args.expected_initial_state_sha256:
        raise RuntimeError(f"initial state hash mismatch: {initial_hash}")
    initial_manifest = _save_initial_state(
        args.output_dir / "initial-state.npz", variables, mapping["state_keys"]
    )
    _write_json(args.output_dir / "initial-state.json", initial_manifest)
    names, offsets, parameter_count = _parameter_layout(mapping)
    states_path = args.output_dir / "parameters.npy"
    if states_path.exists():
        raise FileExistsError(f"refusing to overwrite {states_path.name}")
    states = np.lib.format.open_memmap(
        states_path, mode="w+", dtype="float32", shape=(args.steps, parameter_count)
    )
    noise_box: list[Any] = [None]
    _install_fixed_noise(tf, model, noise_box)
    trace_path = args.output_dir / "steps.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:
        _write_record(
            trace,
            {
                "record": "metadata",
                "framework": "tensorflow",
                "stack": stack,
                "steps": args.steps,
                "batch_sequence_sha256": plan_metadata["batch_sequence_sha256"],
                "initial_state_sha256": initial_hash,
                "initial_state_file_sha256": initial_manifest["file_sha256"],
            },
        )
        row_hashes: list[str] = []
        inputs_dir = args.run_dir / "inputs"
        for index, plan_row in enumerate(plan_rows):
            step_started = time.monotonic()
            with np.load(
                inputs_dir / plan_row["batch_file"], allow_pickle=False
            ) as archive:
                arrays = {name: archive[name] for name in archive.files}
            tf_inputs = {key: tf.convert_to_tensor(arrays[key]) for key in columns}
            noise_box[0] = arrays["posterior_noise"]
            tf.keras.utils.set_random_seed(args.seed + index)
            model.reset_metrics()
            with tf.GradientTape() as tape:
                outputs = model(tf_inputs, training=True)
                loss = model.compute_loss(tf_inputs, None, outputs, None)
            gradients = tape.gradient(loss, model.trainable_variables)
            gradient_norm = float(
                tf.linalg.global_norm(
                    [gradient for gradient in gradients if gradient is not None]
                ).numpy()
            )
            loss_value = float(loss.numpy())
            if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
                raise RuntimeError(f"non-finite TensorFlow metric at step {index + 1}")
            model.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables, strict=True)
            )
            parameters = _tf_parameter_arrays(model, mapping, names)
            row_hash = _write_row(states, index, parameters, names, offsets)
            row_hashes.append(row_hash)
            _write_record(
                trace,
                {
                    "record": "step",
                    "step": index + 1,
                    "loss": loss_value,
                    "gradient_norm": gradient_norm,
                    "batch_sha256": plan_row["batch_sha256"],
                    "posterior_noise_sha256": plan_row["posterior_noise_sha256"],
                    "sampler_state_sha256": _sha256_bytes(
                        json.dumps(plan_row, sort_keys=True).encode("utf-8")
                    ),
                    "parameter_row_sha256": row_hash,
                    "step_seconds_report_only": time.monotonic() - step_started,
                },
            )

    state_manifest = _finish_state_file(states_path, states, names, offsets, row_hashes)
    _write_json(args.output_dir / "parameters.json", state_manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "steps": args.steps,
                "state_file_sha256": state_manifest["file_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _torch_batch(arrays: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.tolist()
        if key in {"document_id", "split"}
        else torch.as_tensor(value, device=device)
        for key, value in arrays.items()
        if key != "posterior_noise"
    }


def _torch_parameter_arrays(model: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous().numpy()
        for name, parameter in model.named_parameters()
        if name in names
    }


def _package_update(
    model: Any,
    optimizer: Any,
    arrays: Mapping[str, Any],
    seed: int,
    step: int,
    device: Any,
    comparison: Any,
) -> tuple[float, float]:
    import random

    import numpy as np
    import torch

    random.seed(seed + step)
    np.random.seed(seed + step)
    torch.manual_seed(seed + step)
    torch.cuda.manual_seed_all(seed + step)
    batch = _torch_batch(arrays, device)
    noise = torch.as_tensor(arrays["posterior_noise"], device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch, posterior_noise=noise)
    if output.loss is None or output.kl_divergence is None:
        raise RuntimeError("package lockstep forward returned no losses")
    loss = output.loss + comparison.l2_penalty(model, comparison.L2_WEIGHT)
    loss_value = float(loss.detach().cpu())
    loss.backward()
    gradient_norms = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradient_norms:
        raise RuntimeError("package lockstep produced no gradients")
    gradient_norm = float(torch.linalg.vector_norm(torch.stack(gradient_norms)).cpu())
    if not math.isfinite(loss_value) or not math.isfinite(gradient_norm):
        raise RuntimeError(f"non-finite package metric at step {step + 1}")
    comparison.clip_gradients_by_norm(model.parameters(), comparison.CLIP_NORM)
    optimizer.step()

    return loss_value, gradient_norm


def _run_package(args: argparse.Namespace) -> None:
    import copy
    import numpy as np
    import torch
    from traingen_parity.determinism import DeterminismConfig, apply_determinism

    plan_metadata, plan_rows, mapping = _read_inputs(args.run_dir)
    if int(plan_metadata["steps"]) != args.steps:
        raise ValueError("requested step count differs from the batch plan")
    apply_determinism(DeterminismConfig(seed=args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("package lockstep requires the selected CUDA device")
    scripts = args.source_root / "models" / "canvas-vae" / "scripts"
    sys.path.insert(0, str(scripts))
    import compare_crello_training as comparison
    from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModel
    from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
    from canvas_vae.training.precision import disable_torch_tf32
    from traingen.optim import KerasAdam

    disable_torch_tf32()
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Torch TF32 must be disabled")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_module = CanvasVAECrelloDataModule(
        data_dir=str(args.data_dir),
        fixture_dir=str(args.fixture_dir),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    data_module.setup("fit")
    if data_module.processor is None:
        raise RuntimeError("package data module did not construct its processor")
    config = comparison.CanvasVAECrelloConfig(
        vocabularies=data_module.processor.vocabularies,
        latent_dim=comparison.LATENT_DIM,
        dropout=0.0,
        kl_weight=comparison.KL_WEIGHT,
        l2_weight=comparison.L2_WEIGHT,
    )
    device = torch.device("cuda:0")
    model = CanvasVAECrelloModel(config).to(device)
    initial_path = args.run_dir / "original" / "initial-state.npz"
    initial_manifest = json.loads(
        (args.run_dir / "original" / "initial-state.json").read_text("utf-8")
    )
    if _sha256_file(initial_path) != initial_manifest["file_sha256"]:
        raise RuntimeError("shared initial-state file hash mismatch")
    initial_state = _load_initial_state(initial_path, mapping)
    model.load_state_dict(
        {
            key: torch.as_tensor(value, device=device)
            for key, value in initial_state.items()
        },
        strict=True,
    )
    initial_hash = comparison._torch_state_sha256(model.state_dict())
    if initial_hash != args.expected_initial_state_sha256:
        raise RuntimeError(f"package initial state hash mismatch: {initial_hash}")
    if initial_hash != initial_manifest["state_sha256"]:
        raise RuntimeError("the original and package initial-state hashes differ")

    cross_model = model
    noise_model = copy.deepcopy(model)
    perturbed_model = copy.deepcopy(model)
    perturbed_count = comparison._perturb_parameters_by_one_ulp(perturbed_model)
    noise_initial_hash = comparison._torch_state_sha256(noise_model.state_dict())
    if noise_initial_hash != initial_hash:
        raise RuntimeError(
            "noise-floor package state differs from the cross initialization"
        )

    package_dir = args.output_dir
    if package_dir.exists():
        raise FileExistsError(f"refusing to overwrite {package_dir.name}")
    package_dir.mkdir(parents=True)
    stack = _write_stack(package_dir, "torch")
    names, offsets, parameter_count = _parameter_layout(mapping)
    cross_states_path = package_dir / "cross-parameters.npy"
    noise_states_path = package_dir / "noise-perturbed-parameters.npy"
    cross_states = np.lib.format.open_memmap(
        cross_states_path,
        mode="w+",
        dtype="float32",
        shape=(args.steps, parameter_count),
    )
    noise_states = np.lib.format.open_memmap(
        noise_states_path,
        mode="w+",
        dtype="float32",
        shape=(args.steps, parameter_count),
    )
    cross_optimizer = KerasAdam(cross_model.parameters(), lr=comparison.LEARNING_RATE)
    noise_optimizer = KerasAdam(noise_model.parameters(), lr=comparison.LEARNING_RATE)
    perturbed_optimizer = KerasAdam(
        perturbed_model.parameters(), lr=comparison.LEARNING_RATE
    )
    inputs_dir = args.run_dir / "inputs"
    cross_path = package_dir / "cross-steps.jsonl"
    noise_path = package_dir / "noise-steps.jsonl"
    cross_hashes: list[str] = []
    noise_hashes: list[str] = []
    cross_state_hashes: list[str] = []
    with cross_path.open("x", encoding="utf-8") as cross_trace:
        _write_record(
            cross_trace,
            {
                "record": "metadata",
                "framework": "torch",
                "stack": stack,
                "steps": args.steps,
                "batch_sequence_sha256": plan_metadata["batch_sequence_sha256"],
                "initial_state_sha256": initial_hash,
            },
        )
        for index, plan_row in enumerate(plan_rows):
            step_started = time.monotonic()
            with np.load(
                inputs_dir / plan_row["batch_file"], allow_pickle=False
            ) as archive:
                arrays = {name: archive[name] for name in archive.files}
            loss, gradient_norm = _package_update(
                cross_model,
                cross_optimizer,
                arrays,
                args.seed,
                index,
                device,
                comparison,
            )
            state_hash = _torch_state_digest(cross_model)
            parameter_arrays = _torch_parameter_arrays(cross_model, names)
            row_hash = _write_row(cross_states, index, parameter_arrays, names, offsets)
            cross_state_hashes.append(state_hash)
            cross_hashes.append(row_hash)
            _write_record(
                cross_trace,
                {
                    "record": "step",
                    "step": index + 1,
                    "loss": loss,
                    "gradient_norm": gradient_norm,
                    "state_sha256": state_hash,
                    "parameter_row_sha256": row_hash,
                    "batch_sha256": plan_row["batch_sha256"],
                    "posterior_noise_sha256": plan_row["posterior_noise_sha256"],
                    "sampler_state_sha256": _sha256_bytes(
                        json.dumps(plan_row, sort_keys=True).encode("utf-8")
                    ),
                    "step_seconds_report_only": time.monotonic() - step_started,
                },
            )

    with noise_path.open("x", encoding="utf-8") as noise_trace:
        _write_record(
            noise_trace,
            {
                "record": "metadata",
                "framework": "torch",
                "stack": stack,
                "steps": args.steps,
                "batch_sequence_sha256": plan_metadata["batch_sequence_sha256"],
                "initial_state_sha256": initial_hash,
                "perturbed_parameter_values": perturbed_count,
            },
        )
        for index, plan_row in enumerate(plan_rows):
            step_started = time.monotonic()
            with np.load(
                inputs_dir / plan_row["batch_file"], allow_pickle=False
            ) as archive:
                arrays = {name: archive[name] for name in archive.files}
            normal_loss, normal_gradient_norm = _package_update(
                noise_model,
                noise_optimizer,
                arrays,
                args.seed,
                index,
                device,
                comparison,
            )
            perturbed_loss, perturbed_gradient_norm = _package_update(
                perturbed_model,
                perturbed_optimizer,
                arrays,
                args.seed,
                index,
                device,
                comparison,
            )
            normal_state_hash = _torch_state_digest(noise_model)
            if normal_state_hash != cross_state_hashes[index]:
                raise RuntimeError(
                    f"package noise-floor baseline diverged at step {index + 1}"
                )
            normal_parameter_hash = _parameter_arrays_hash(
                _torch_parameter_arrays(noise_model, names), names
            )
            perturbed_state_hash = _torch_state_digest(perturbed_model)
            perturbed_arrays = _torch_parameter_arrays(perturbed_model, names)
            row_hash = _write_row(noise_states, index, perturbed_arrays, names, offsets)
            noise_hashes.append(row_hash)
            _write_record(
                noise_trace,
                {
                    "record": "step",
                    "step": index + 1,
                    "normal_loss": normal_loss,
                    "normal_gradient_norm": normal_gradient_norm,
                    "normal_state_sha256": normal_state_hash,
                    "normal_parameter_row_sha256": normal_parameter_hash,
                    "perturbed_loss": perturbed_loss,
                    "perturbed_gradient_norm": perturbed_gradient_norm,
                    "perturbed_state_sha256": perturbed_state_hash,
                    "perturbed_parameter_row_sha256": row_hash,
                    "batch_sha256": plan_row["batch_sha256"],
                    "posterior_noise_sha256": plan_row["posterior_noise_sha256"],
                    "sampler_state_sha256": _sha256_bytes(
                        json.dumps(plan_row, sort_keys=True).encode("utf-8")
                    ),
                    "step_seconds_report_only": time.monotonic() - step_started,
                },
            )

    cross_manifest = _finish_state_file(
        cross_states_path, cross_states, names, offsets, cross_hashes
    )
    noise_manifest = _finish_state_file(
        noise_states_path, noise_states, names, offsets, noise_hashes
    )
    _write_json(package_dir / "cross-parameters.json", cross_manifest)
    _write_json(package_dir / "noise-perturbed-parameters.json", noise_manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "steps": args.steps,
                "initial_state_sha256": initial_hash,
                "cross_state_file_sha256": cross_manifest["file_sha256"],
                "noise_state_file_sha256": noise_manifest["file_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _row_arrays(
    row: Any,
    names: Sequence[str],
    offsets: Mapping[str, tuple[int, int, list[int]]],
) -> dict[str, Any]:
    import numpy as np

    return {
        name: np.asarray(row[start:stop]).reshape(shape)
        for name in names
        for start, stop, shape in (offsets[name],)
    }


def _load_state_matrix(run_dir: Path, manifest_name: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    manifest_path = run_dir / manifest_name
    manifest = json.loads(manifest_path.read_text("utf-8"))
    state_path = run_dir / manifest["file"]
    if _sha256_file(state_path) != manifest["file_sha256"]:
        raise ValueError(f"state artifact hash mismatch: {state_path.name}")
    matrix = np.load(state_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape[0] != int(manifest["steps"]):
        raise ValueError(f"state artifact step count mismatch: {state_path.name}")

    return matrix, manifest


def _validate_step_inputs(
    record: Mapping[str, Any], plan_row: Mapping[str, Any], step: int
) -> None:
    expected_sampler_hash = _sha256_bytes(
        json.dumps(plan_row, sort_keys=True).encode("utf-8")
    )
    for key in ("batch_sha256", "posterior_noise_sha256"):
        if record[key] != plan_row[key]:
            raise ValueError(f"{key} differs at step {step}")
    if record["sampler_state_sha256"] != expected_sampler_hash:
        raise ValueError(f"sampler state differs at step {step}")


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    plan_metadata, plan_rows, mapping = _read_inputs(args.run_dir)
    original_dir = args.run_dir / "original"
    package_dir = args.run_dir / "package"
    original_metadata, original_rows = _read_jsonl(original_dir / "steps.jsonl")
    package_metadata, package_rows = _read_jsonl(package_dir / "cross-steps.jsonl")
    noise_metadata, noise_rows = _read_jsonl(package_dir / "noise-steps.jsonl")
    initial_manifest = json.loads(
        (original_dir / "initial-state.json").read_text("utf-8")
    )
    initial_path = original_dir / initial_manifest["file"]
    if _sha256_file(initial_path) != initial_manifest["file_sha256"]:
        raise ValueError("initial-state artifact hash mismatch")
    if initial_manifest["state_sha256"] != EXPECTED_INITIAL_STATE_SHA256:
        raise ValueError("initial-state hash differs from the required value")
    if plan_metadata["batch_sequence_sha256"] != EXPECTED_BATCH_SEQUENCE_SHA256:
        raise ValueError("batch sequence hash differs from the required value")
    steps = int(plan_metadata["steps"])
    if not (len(original_rows) == len(package_rows) == len(noise_rows) == steps):
        raise ValueError("a framework trace does not contain the full step count")

    names, offsets, parameter_count = _parameter_layout(mapping)
    original_states, original_state_manifest = _load_state_matrix(
        original_dir, "parameters.json"
    )
    package_states, package_state_manifest = _load_state_matrix(
        package_dir, "cross-parameters.json"
    )
    noise_states, noise_state_manifest = _load_state_matrix(
        package_dir, "noise-perturbed-parameters.json"
    )
    for matrix in (original_states, package_states, noise_states):
        if matrix.shape != (steps, parameter_count):
            raise ValueError("parameter snapshot shape does not match the manifest")

    cross_records: list[dict[str, Any]] = []
    noise_records: list[dict[str, Any]] = []
    batch_hashes: list[str] = []
    for index, plan_row in enumerate(plan_rows):
        original = original_rows[index]
        package = package_rows[index]
        noise = noise_rows[index]
        step = index + 1
        for record in (original, package, noise):
            if int(record["step"]) != step:
                raise ValueError(f"non-consecutive trace at step {step}")
            _validate_step_inputs(record, plan_row, step)
        for value in (
            original["loss"],
            original["gradient_norm"],
            package["loss"],
            package["gradient_norm"],
            noise["normal_loss"],
            noise["normal_gradient_norm"],
            noise["perturbed_loss"],
            noise["perturbed_gradient_norm"],
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite per-step metric at step {step}")
        if noise["normal_state_sha256"] != package["state_sha256"]:
            raise ValueError(f"package baseline state differs at step {step}")
        if noise["normal_parameter_row_sha256"] != package["parameter_row_sha256"]:
            raise ValueError(f"package baseline parameters differ at step {step}")
        if (
            noise["normal_loss"] != package["loss"]
            or noise["normal_gradient_norm"] != package["gradient_norm"]
        ):
            raise ValueError(f"package baseline metrics differ at step {step}")

        original_row = original_states[index]
        package_row = package_states[index]
        noise_row = noise_states[index]
        original_row_hash = _parameter_row_hash(original_row, names, offsets)
        package_row_hash = _parameter_row_hash(package_row, names, offsets)
        noise_row_hash = _parameter_row_hash(noise_row, names, offsets)
        if original_row_hash != original["parameter_row_sha256"]:
            raise ValueError(f"TensorFlow parameter row hash mismatch at step {step}")
        if package_row_hash != package["parameter_row_sha256"]:
            raise ValueError(f"package parameter row hash mismatch at step {step}")
        if noise_row_hash != noise["perturbed_parameter_row_sha256"]:
            raise ValueError(f"noise-floor parameter row hash mismatch at step {step}")

        cross_difference, cross_name = _parameter_difference(
            _row_arrays(original_row, names, offsets),
            _row_arrays(package_row, names, offsets),
            names,
        )
        noise_difference, noise_name = _parameter_difference(
            _row_arrays(package_row, names, offsets),
            _row_arrays(noise_row, names, offsets),
            names,
        )
        cross_records.append(
            {
                "step": step,
                "loss_left": float(original["loss"]),
                "loss_right": float(package["loss"]),
                "relative_loss_difference": _relative_loss_difference(
                    float(original["loss"]), float(package["loss"])
                ),
                "gradient_norm_left": float(original["gradient_norm"]),
                "gradient_norm_right": float(package["gradient_norm"]),
                "parameter_max_abs_difference": cross_difference,
                "parameter_with_max_difference": cross_name,
                "batch_sha256": plan_row["batch_sha256"],
                "posterior_noise_sha256": plan_row["posterior_noise_sha256"],
            }
        )
        noise_records.append(
            {
                "step": step,
                "loss_left": float(noise["normal_loss"]),
                "loss_right": float(noise["perturbed_loss"]),
                "relative_loss_difference": _relative_loss_difference(
                    float(noise["normal_loss"]), float(noise["perturbed_loss"])
                ),
                "gradient_norm_left": float(noise["normal_gradient_norm"]),
                "gradient_norm_right": float(noise["perturbed_gradient_norm"]),
                "parameter_max_abs_difference": noise_difference,
                "parameter_with_max_difference": noise_name,
                "batch_sha256": plan_row["batch_sha256"],
                "posterior_noise_sha256": plan_row["posterior_noise_sha256"],
            }
        )
        batch_hashes.append(str(plan_row["batch_sha256"]))

    cross_first = _first_over_limit(cross_records)
    noise_first = _first_over_limit(noise_records)
    cross_step = int(cross_first["step"]) if cross_first else None
    noise_step = int(noise_first["step"]) if noise_first else None
    attribution_passed = _attribution_passed(cross_step, noise_step)
    batch_sequence_hash = _sha256_bytes("\n".join(batch_hashes).encode("ascii"))
    if batch_sequence_hash != EXPECTED_BATCH_SEQUENCE_SHA256:
        raise ValueError("compared batch sequence does not match the required hash")
    summary = {
        "protocol": "crello-real-scale-lockstep-v2-separated-processes",
        "source_commit": os.environ.get("CANVAS_VAE_S5_SOURCE_COMMIT", "unknown"),
        "probe_script_sha256": _sha256_file(Path(__file__)),
        "status": "completed",
        "required_steps": steps,
        "completed_steps": steps,
        "completed_steps_per_phase": steps,
        "batch_size": int(plan_metadata["batch_size"]),
        "device": "cuda",
        "initial_state_sha256": initial_manifest["state_sha256"],
        "noise_floor_initial_state_sha256": initial_manifest["state_sha256"],
        "noise_floor_perturbed_parameter_values": int(
            noise_metadata["perturbed_parameter_values"]
        ),
        "relative_loss_limit": RELATIVE_LOSS_LIMIT,
        "phases": {
            "cross_system": {
                "required_steps": steps,
                "completed_steps": len(cross_records),
                "first_step_over_tolerance": cross_first,
                "maximum_relative_loss_difference": max(
                    row["relative_loss_difference"] for row in cross_records
                ),
                "sampler_batch_sha256": batch_sequence_hash,
            },
            "noise_floor": {
                "required_steps": steps,
                "completed_steps": len(noise_records),
                "first_step_over_tolerance": noise_first,
                "maximum_relative_loss_difference": max(
                    row["relative_loss_difference"] for row in noise_records
                ),
                "sampler_batch_sha256": batch_sequence_hash,
            },
        },
        "first_step_over_tolerance": cross_first,
        "noise_floor_first_step_over_tolerance": noise_first,
        "noise_floor_attribution_passed": attribution_passed,
        "sampler": "CrossEpochBatchSampler",
        "sampler_batch_sha256": batch_sequence_hash,
        "gate_fields": ["finite_metrics", "noise_floor_attribution"],
        "throughput_fields_are_report_only": True,
        "stacks": {
            "tensorflow": original_metadata["stack"],
            "torch": package_metadata["stack"],
        },
        "artifacts": {
            "batch_plan_sha256": _sha256_file(
                args.run_dir / "inputs" / "batch-plan.jsonl"
            ),
            "mapping_sha256": _sha256_file(args.run_dir / "inputs" / "mapping.json"),
            "initial_state_sha256": initial_manifest["file_sha256"],
            "tensorflow_parameter_states_sha256": original_state_manifest[
                "file_sha256"
            ],
            "package_parameter_states_sha256": package_state_manifest["file_sha256"],
            "noise_floor_parameter_states_sha256": noise_state_manifest["file_sha256"],
        },
    }
    if not summary["phases"]["cross_system"]["completed_steps"] == steps:
        raise ValueError("cross-system phase is incomplete")
    if not summary["phases"]["noise_floor"]["completed_steps"] == steps:
        raise ValueError("noise-floor phase is incomplete")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--fixture-dir", type=Path, required=True)
    prepare.add_argument("--original-data-dir", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--steps", type=int, default=300)
    prepare.add_argument("--batch-size", type=int, default=1024)
    prepare.add_argument("--num-workers", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument(
        "--expected-batch-sha256", default=EXPECTED_BATCH_SEQUENCE_SHA256
    )
    package_preflight = subparsers.add_parser("preflight-package")
    package_preflight.add_argument("--source-root", type=Path, required=True)
    package_preflight.add_argument("--output-dir", type=Path, required=True)
    original_preflight = subparsers.add_parser("preflight-original")
    original_preflight.add_argument("--output-dir", type=Path, required=True)
    original = subparsers.add_parser("original")
    original.add_argument("--original-data-dir", type=Path, required=True)
    original.add_argument("--run-dir", type=Path, required=True)
    original.add_argument("--output-dir", type=Path, required=True)
    original.add_argument("--steps", type=int, default=300)
    original.add_argument("--batch-size", type=int, default=1024)
    original.add_argument("--seed", type=int, default=0)
    original.add_argument(
        "--expected-initial-state-sha256", default=EXPECTED_INITIAL_STATE_SHA256
    )
    package = subparsers.add_parser("package")
    package.add_argument("--source-root", type=Path, required=True)
    package.add_argument("--data-dir", type=Path, required=True)
    package.add_argument("--fixture-dir", type=Path, required=True)
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--steps", type=int, default=300)
    package.add_argument("--batch-size", type=int, default=1024)
    package.add_argument("--seed", type=int, default=0)
    package.add_argument(
        "--expected-initial-state-sha256", default=EXPECTED_INITIAL_STATE_SHA256
    )
    compare = subparsers.add_parser("compare")
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "preflight-package":
        _preflight_package(args)
    elif args.command == "preflight-original":
        _preflight_original(args)
    elif args.command == "original":
        _run_original(args)
    elif args.command == "package":
        _run_package(args)
    else:
        if args.summary_output.exists():
            raise FileExistsError("refusing to overwrite comparison summary")
        summary = _compare(args)
        _write_json(args.summary_output, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if not summary["noise_floor_attribution_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
