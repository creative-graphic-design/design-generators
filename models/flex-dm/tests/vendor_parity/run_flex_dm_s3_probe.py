"""Record FlexDM natural training and real-scale lockstep evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch

from flex_dm.conversion import map_tensor_by_rule
from flex_dm.training.lightning_module import FlexDmTrainingModule
from .test_flex_dm_training_s1 import (
    _capture_dropout_masks,
    _replay_dropout_masks,
)
from .test_flex_dm_training_s0 import _first_batch, _source_model_state
from .training_reference import (
    DatasetName,
    build_original,
    original_asset_root,
    original_variant_root,
)
from traingen_parity.determinism import (
    RNGState,
    capture_rng_state,
    restore_rng_state,
)
from traingen_parity.trace import build_step_trace, tensor_sha256
from traingen_parity.compare import (
    TensorTolerance,
    compare_step_trace,
)

ROOT: Final[Path] = Path(__file__).resolve().parents[4]
DATASETS: Final[tuple[DatasetName, ...]] = ("crello", "rico")
LOSS_LIMIT: Final[float] = 1e-3
OUTPUT_LIMIT: Final[float] = 1e-5
SCALAR_LOSS_LIMIT: Final[float] = 1e-6
S2_DIRECT_LIMIT: Final[float] = 3.5e-4
S2_FLOAT64_LIMIT: Final[float] = 4.1e-3
S2_ZERO_GRADIENT_LIMIT: Final[float] = 1e-6


def _paths(dataset: DatasetName) -> dict[str, Path]:
    data = original_asset_root() / "data" / dataset
    if (data / dataset).is_dir():
        data /= dataset
    return {
        "args": original_variant_root(dataset) / "args.json",
        "checkpoint": original_variant_root(dataset) / "checkpoints" / "best.ckpt",
        "data": data,
        "converted": ROOT / ".cache" / "flex-dm" / "converted" / f"flex-dm-{dataset}",
        "training_config": ROOT
        / "models"
        / "flex-dm"
        / "configs"
        / "training"
        / f"flex_dm_{dataset}.yaml",
    }


def _validate_inputs(dataset: DatasetName) -> None:
    paths = _paths(dataset)
    required = (
        paths["args"],
        paths["checkpoint"].with_suffix(".ckpt.index"),
        paths["checkpoint"].with_suffix(".ckpt.data-00000-of-00001"),
        paths["data"] / "count.json",
        paths["converted"] / "config.json",
        paths["converted"] / "model.safetensors",
        paths["training_config"],
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local inputs: {missing}")

    counts = json.loads((paths["data"] / "count.json").read_text(encoding="utf-8"))
    expected = {"crello": 18_587, "rico": 45_012}[dataset]
    if int(counts["train"]) != expected:
        raise RuntimeError(f"unexpected {dataset} training split count")
    args = json.loads(paths["args"].read_text(encoding="utf-8"))
    expected_masking = {
        "crello": "elem_pos_attr_img_txt",
        "rico": "elem_pos_attr",
    }[dataset]
    if (
        args.get("arch_type") != "oneshot"
        or args.get("block_type") != "deepsvg"
        or args.get("masking_method") != expected_masking
        or args.get("input_dtype") != "set"
        or args.get("batch_size") != 256
        or args.get("seed") != 0
        or float(args.get("learning_rate", -1)) != 1e-4
        or float(args.get("l2", -1)) != 0.01
        or float(args.get("dropout", -1)) != 0.1
    ):
        raise RuntimeError("the probe requires the released EXP fine-tuning config")
    if not list(paths["data"].glob("train-*.tfrecord")):
        raise FileNotFoundError(f"missing {dataset} training TFRecord shards")


def _preflight(dataset: DatasetName) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("preflight requires CUDA to be hidden")
    _validate_inputs(dataset)
    print(json.dumps({"mode": "preflight", "dataset": dataset, "inputs": "available"}))


def _configure_tf32(tf) -> None:
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be 0")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if tf.config.experimental.tensor_float_32_execution_enabled():
        raise RuntimeError("TensorFlow TF32 is enabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("PyTorch TF32 is enabled")


def _assert_gpu_environment(tf) -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        raise RuntimeError("the probe requires one selected GPU")
    if torch.cuda.device_count() != 1 or not tf.config.list_logical_devices("GPU"):
        raise RuntimeError("both frameworks must see one GPU")
    _configure_tf32(tf)
    return torch.device("cuda")


def _seed(seed: int, tf) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


def _source_name_map(model, checkpoint: Path, tf) -> dict[str, str]:
    from tensorflow.core.protobuf import trackable_object_graph_pb2

    root = tf.train.Checkpoint(**model._trackable_children())
    reader = tf.train.load_checkpoint(str(checkpoint))
    graph = trackable_object_graph_pb2.TrackableObjectGraph()
    graph.ParseFromString(reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH"))
    names: dict[str, str] = {}

    def visit(trackable, node_id: int) -> None:
        children = {
            edge.local_name: edge.node_id for edge in graph.nodes[node_id].children
        }
        for name, child in trackable._trackable_children().items():
            if name not in children:
                continue
            child_id = children[name]
            if isinstance(child, tf.Variable):
                key = next(
                    (
                        item.checkpoint_key
                        for item in graph.nodes[child_id].attributes
                        if item.name == "VARIABLE_VALUE"
                    ),
                    None,
                )
                if key is not None:
                    names[child.name] = key.removesuffix("/.ATTRIBUTES/VARIABLE_VALUE")
            else:
                visit(child, child_id)

    visit(root, 0)
    if any(variable.name not in names for variable in model.trainable_variables):
        raise RuntimeError("the released checkpoint did not map every trainable tensor")
    return names


def _tensor_dict(
    value: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if isinstance(item, torch.Tensor):
            tensors[key] = item.to(device)
            continue
        if isinstance(item, np.ndarray):
            array = item
        elif hasattr(item, "numpy"):
            array = np.asarray(item.numpy())
        else:
            continue
        if array.dtype.kind in "USO":
            continue
        tensors[key] = torch.as_tensor(array, device=device)
    return tensors


def _batch_hash(batch: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    tf = _tensorflow()
    for key in sorted(batch):
        digest.update(tf.io.serialize_tensor(batch[key]).numpy())
        digest.update(key.encode())
    return digest.hexdigest()


def _source_state(
    values: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for source, value in values.items():
        result = map_tensor_by_rule(source, value)
        if result is None:
            continue
        key, tensor = result
        if key in mapped:
            raise RuntimeError(f"duplicate source mapping for {key}")
        mapped[key] = tensor.to(device)
    return mapped


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        return math.inf
    numerator = torch.linalg.vector_norm(actual.double() - expected.double())
    denominator = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def _max_difference(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> tuple[float, str]:
    if set(first) != set(second):
        raise RuntimeError("parameter names differ between training systems")
    differences = {
        key: float((first[key] - second[key]).abs().max().item()) for key in first
    }
    if not all(math.isfinite(value) for value in differences.values()):
        raise RuntimeError("non-finite parameter difference")
    name = max(differences, key=differences.__getitem__)
    return differences[name], name


def _gradient_norm(gradients: Mapping[str, torch.Tensor]) -> float:
    total = sum(
        float(value.double().square().sum().item()) for value in gradients.values()
    )
    return math.sqrt(total)


def _rng_equal(first: RNGState, second: RNGState) -> bool:
    return (
        first.python == second.python
        and first.numpy[0] == second.numpy[0]
        and np.array_equal(first.numpy[1], second.numpy[1])
        and first.numpy[2:] == second.numpy[2:]
        and torch.equal(first.torch_cpu, second.torch_cpu)
        and len(first.torch_cuda) == len(second.torch_cuda)
        and all(
            torch.equal(a, b)
            for a, b in zip(first.torch_cuda, second.torch_cuda, strict=True)
        )
    )


def _capture_original(
    model, tf, current: dict[str, object], names: Mapping[str, str]
) -> Callable[[], None]:
    from mfp.models import mfp as vendor_mfp

    original_preprocess = vendor_mfp.preprocess_for_train
    original_model = model.model
    original_call = original_model.call
    optimizer = model.optimizer
    clip_gradients = optimizer._clip_gradients
    apply_gradients = optimizer.apply_gradients
    step_variables = []
    random_functions = {
        name: getattr(tf.random, name) for name in ("uniform", "normal")
    }

    def capture_preprocess(*args, **kwargs):
        result = original_preprocess(*args, **kwargs)
        targets, inputs, masks = result
        current["tasks"] = args[2].numpy().copy()
        current["prepared"] = {
            "inputs": {key: value.numpy().copy() for key, value in inputs.items()},
            "labels": {key: value.numpy().copy() for key, value in targets.items()},
            "masks": {key: value.numpy().copy() for key, value in masks.items()},
        }
        return result

    def capture_call(*args, **kwargs):
        result = original_call(*args, **kwargs)
        training = kwargs.get("training", args[1] if len(args) > 1 else False)
        if training:
            current["outputs"] = {
                key: value.numpy().copy()
                for key, value in result.items()
                if key != "tasks" and hasattr(value, "numpy")
            }
        return result

    def capture_clip(gradients):
        gradients = list(gradients)
        active = [
            variable for gradient, variable in step_variables if gradient is not None
        ]
        if len(gradients) != len(active):
            raise RuntimeError("Keras changed the gradient-variable pairing")
        clipped = clip_gradients(gradients)
        current["clipped"] = {
            names[variable.name]: tf.convert_to_tensor(value).numpy().copy()
            for variable, value in zip(active, clipped, strict=True)
            if value is not None
        }
        return clipped

    def capture_apply(gradients_and_variables, *args, **kwargs):
        nonlocal step_variables
        step_variables = list(gradients_and_variables)
        if model.model.losses:
            current["l2"] = tf.add_n(model.model.losses).numpy().copy()
        current["before"] = {
            names[variable.name]: variable.numpy().copy()
            for variable in model.trainable_variables
        }
        current["raw"] = {
            names[variable.name]: tf.convert_to_tensor(gradient).numpy().copy()
            for gradient, variable in step_variables
            if gradient is not None
        }
        indices = getattr(optimizer, "_index_dict", {})
        momentums = getattr(optimizer, "_momentums", ())
        velocities = getattr(optimizer, "_velocities", ())
        current["optimizer_step_before"] = int(optimizer.iterations.numpy())
        current["before_m"] = {}
        current["before_v"] = {}
        for gradient, variable in step_variables:
            if gradient is None:
                continue
            index = indices.get(optimizer._var_key(variable))
            current["before_m"][names[variable.name]] = (
                momentums[index].numpy().copy()
                if index is not None and index < len(momentums)
                else np.zeros_like(variable.numpy())
            )
            current["before_v"][names[variable.name]] = (
                velocities[index].numpy().copy()
                if index is not None and index < len(velocities)
                else np.zeros_like(variable.numpy())
            )
        return apply_gradients(step_variables, *args, **kwargs)

    def capture_random(name, function):
        def record(*args, **kwargs):
            value = function(*args, **kwargs)
            array = np.ascontiguousarray(value.numpy())
            current.setdefault("random_draws", []).append(
                {
                    "operation": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
            )
            return value

        return record

    vendor_mfp.preprocess_for_train = capture_preprocess
    original_model.call = capture_call
    optimizer._clip_gradients = capture_clip
    optimizer.apply_gradients = capture_apply
    for name, function in random_functions.items():
        setattr(tf.random, name, capture_random(name, function))

    def restore() -> None:
        vendor_mfp.preprocess_for_train = original_preprocess
        original_model.call = original_call
        optimizer._clip_gradients = clip_gradients
        optimizer.apply_gradients = apply_gradients
        for name, function in random_functions.items():
            setattr(tf.random, name, function)

    return restore


def _reference_boundary(
    model,
    names: Mapping[str, str],
    current: Mapping[str, object],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    optimizer = model.optimizer
    boundary: dict[str, dict[str, torch.Tensor]] = {
        group: _source_state(cast(Mapping[str, np.ndarray], current[group]), device)
        for group in ("raw", "clipped")
    }
    moments: dict[str, dict[str, np.ndarray]] = {"m": {}, "v": {}}
    after: dict[str, np.ndarray] = {}
    for variable in model.trainable_variables:
        source = names[variable.name]
        target = map_tensor_by_rule(source, variable.numpy())
        if target is None:
            continue
        key, _ = target
        after[source] = variable.numpy().copy()
        index = optimizer._index_dict.get(optimizer._var_key(variable))
        if index is not None:
            moments["m"][source] = optimizer._momentums[index].numpy().copy()
            moments["v"][source] = optimizer._velocities[index].numpy().copy()
    boundary["m"] = _source_state(moments["m"], device)
    boundary["v"] = _source_state(moments["v"], device)
    boundary["before_m"] = _source_state(
        cast(Mapping[str, np.ndarray], current["before_m"]), device
    )
    boundary["before_v"] = _source_state(
        cast(Mapping[str, np.ndarray], current["before_v"]), device
    )
    boundary["after"] = _source_state(after, device)
    return boundary


def _synchronize_package_boundary(
    original,
    names: Mapping[str, str],
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    module: FlexDmTrainingModule,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model_state = {}
    for variable in original.model.variables:
        source = names.get(variable.name)
        if source is None:
            raise RuntimeError(
                f"missing source name for model variable {variable.name}"
            )
        mapped = map_tensor_by_rule(source, variable.numpy())
        if mapped is not None:
            key, value = mapped
            if key in model_state:
                raise RuntimeError(f"duplicate synchronized model tensor {key}")
            model_state[key] = value.to(device)
    if set(model_state) != set(module.model.state_dict()):
        raise RuntimeError("synchronized model parameters and buffers do not match")
    module.model.load_state_dict(model_state, strict=True)

    moments = reference["m"]
    velocities = reference["v"]
    if set(moments) != set(velocities):
        raise RuntimeError("synchronized optimizer moment names differ")
    parameter_names = {
        id(value): name for name, value in module.model.named_parameters()
    }
    optimizer_state = optimizer.state_dict()
    synchronized_state = {}
    iteration = int(original.optimizer.iterations.numpy())
    for live_group, saved_group in zip(
        optimizer.param_groups, optimizer_state["param_groups"], strict=True
    ):
        for parameter, state_id in zip(
            live_group["params"], saved_group["params"], strict=True
        ):
            name = parameter_names[id(parameter)]
            if name in moments:
                synchronized_state[state_id] = {
                    "step": iteration,
                    "exp_avg": moments[name],
                    "exp_avg_sq": velocities[name],
                }
    source_optimizer_state = {
        "state": synchronized_state,
        "param_groups": optimizer_state["param_groups"],
    }
    optimizer.load_state_dict(copy.deepcopy(source_optimizer_state))
    for parameter in module.model.parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        name = parameter_names[id(parameter)]
        for key, source_values in (("exp_avg", moments), ("exp_avg_sq", velocities)):
            if (
                name in source_values
                and state[key].data_ptr() == source_values[name].data_ptr()
            ):
                raise RuntimeError(
                    f"synchronized optimizer state shares storage: {name}.{key}"
                )


def _package_update(
    module: FlexDmTrainingModule,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    *,
    update: int,
    device: torch.device,
    prepared: Mapping[str, object] | None = None,
    dropout_masks: list[np.ndarray] | None = None,
    captured_prepared: dict[str, object] | None = None,
    captured_dropout_masks: list[np.ndarray] | None = None,
) -> tuple[float, dict[str, object], dict[str, dict[str, torch.Tensor]], RNGState]:
    captured: dict[str, object] = {}
    original_forward = module.model.forward
    original_prepare = module.prepare_batch
    initial_state = {
        key: value.detach().clone() for key, value in module.model.state_dict().items()
    }
    incoming_m = {
        name: optimizer.state.get(parameter, {})
        .get("exp_avg", torch.zeros_like(parameter))
        .detach()
        .clone()
        for name, parameter in module.model.named_parameters()
    }
    incoming_v = {
        name: optimizer.state.get(parameter, {})
        .get("exp_avg_sq", torch.zeros_like(parameter))
        .detach()
        .clone()
        for name, parameter in module.model.named_parameters()
    }
    dropout_originals = []

    def capture_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        captured["output"] = output
        return output

    module.model.forward = capture_forward
    if prepared is not None:
        inputs = _tensor_dict(cast(Mapping[str, object], prepared["inputs"]), device)
        labels = _tensor_dict(cast(Mapping[str, object], prepared["labels"]), device)
        masks = _tensor_dict(cast(Mapping[str, object], prepared["masks"]), device)
        tasks = torch.as_tensor(prepared["tasks"], dtype=torch.long, device=device)
        if captured_prepared is not None:
            captured_prepared.update(
                inputs=inputs, labels=labels, masks=masks, tasks=tasks
            )
        module.prepare_batch = lambda _batch: (inputs, labels, masks, tasks)
    elif captured_prepared is not None:

        def capture_prepare(batch):
            values = original_prepare(batch)
            captured_prepared.update(
                inputs=values[0], labels=values[1], masks=values[2], tasks=values[3]
            )
            return values

        module.prepare_batch = capture_prepare
    if captured_dropout_masks is not None:
        if dropout_masks is not None:
            raise ValueError("cannot capture and replay package dropout masks together")
        for layer in module.model.modules():
            if not isinstance(layer, torch.nn.Dropout):
                continue
            original_dropout = layer.forward

            def capture_dropout(values, *, forward=original_dropout):
                output = forward(values)
                mask = torch.logical_or(values == 0.0, output != 0.0)
                captured_dropout_masks.append(mask.detach().cpu().numpy())
                return output

            dropout_originals.append((layer, original_dropout))
            layer.forward = capture_dropout
    consumed = (
        _replay_dropout_masks(module.model, dropout_masks) if dropout_masks else None
    )

    try:
        _configure_tf32(_tensorflow())
        optimizer.zero_grad(set_to_none=True)
        loss = module.training_step(dict(batch), update)
        loss.backward()
        rng = capture_rng_state()
        raw = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.model.named_parameters()
            if parameter.grad is not None
        }
        module.configure_gradient_clipping(optimizer, gradient_clip_val=0.0)
        clipped = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.model.named_parameters()
            if parameter.grad is not None
        }
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if consumed is not None and consumed[0] != len(dropout_masks or ()):
            raise RuntimeError(
                "the package did not replay every reference dropout mask"
            )
        output = cast(object, captured["output"])
        output_losses = {
            **cast(Mapping[str, torch.Tensor], getattr(output, "loss_components")),
            "l2": cast(torch.Tensor, getattr(output, "l2_loss")),
            "total": loss.detach(),
        }
        states = {
            "_before_model_state": initial_state,
            "before_m": {name: incoming_m[name] for name in raw},
            "before_v": {name: incoming_v[name] for name in raw},
            "raw": raw,
            "clipped": clipped,
            "m": {
                name: optimizer.state[parameter]["exp_avg"].detach().clone()
                for name, parameter in module.model.named_parameters()
                if parameter in optimizer.state
            },
            "v": {
                name: optimizer.state[parameter]["exp_avg_sq"].detach().clone()
                for name, parameter in module.model.named_parameters()
                if parameter in optimizer.state
            },
            "after": {
                name: parameter.detach().clone()
                for name, parameter in module.model.named_parameters()
            },
        }
        result = {
            "outputs": cast(Mapping[str, torch.Tensor], getattr(output, "logits")),
            "losses": output_losses,
            "gradient_norm": _gradient_norm(raw),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "state_sha256": _state_hash(states["after"]),
        }
    finally:
        module.model.forward = original_forward
        module.prepare_batch = original_prepare
        for layer, forward in dropout_originals:
            layer.forward = forward
    return float(loss.detach().item()), result, states, rng


@cache
def _tensorflow():
    import tensorflow as tf

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

    return tf


def _loss_difference(first: float, second: float) -> float:
    return abs(second - first) / max(abs(first), float(torch.finfo(torch.float32).eps))


def _gate_passed(cross_step: int | None, noise_step: int | None) -> bool:
    return noise_step is not None and (cross_step is None or cross_step >= noise_step)


def _s3_passed(
    natural_repeats: list[bool], synchronized_passed: bool, lockstep_attributable: bool
) -> bool:
    return lockstep_attributable and (
        (bool(natural_repeats) and all(natural_repeats)) or synchronized_passed
    )


def _write_row(stream, row: Mapping[str, object]) -> None:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()


def _make_package(dataset: DatasetName, device: torch.device):
    module = FlexDmTrainingModule(
        str(ROOT / ".cache" / "flex-dm" / "converted" / f"flex-dm-{dataset}")
    ).to(device)
    module.model.load_state_dict(_source_model_state(dataset), strict=True)
    return module, module.configure_optimizers()


def _make_pair(dataset: DatasetName, tf, device: torch.device):
    dataspec, original, args = build_original(
        dataset, batch_size=256, restore_checkpoint=False
    )
    rng = capture_rng_state()
    tf_rng = tf.random.get_global_generator()
    tf_rng_state = tf.identity(tf_rng.state)
    try:
        batch = _first_batch(dataspec, "test")
        inputs = {key: batch[key] for key in original.input_columns if key in batch}
        original.model(inputs, training=False)
    finally:
        restore_rng_state(rng)
        tf_rng.state.assign(tf_rng_state)

    checkpoint = original_variant_root(dataset) / "checkpoints" / "best.ckpt"
    restore = tf.train.Checkpoint(**original._trackable_children()).restore(
        str(checkpoint)
    )
    restore.assert_nontrivial_match()
    restore.expect_partial()

    original.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(args["learning_rate"]), clipnorm=1.0
        ),
        run_eagerly=True,
    )
    module, optimizer = _make_package(dataset, device)
    names = _source_name_map(original, checkpoint, tf)
    return dataspec, original, module, optimizer, names


def _source_batches(dataspec, steps: int, seed: int, tf):
    tf.random.set_seed(seed)
    data = dataspec.make_dataset(
        "train", shuffle=True, repeat=True, cache=True, parallel=1
    )
    iterator = iter(data)
    for _ in range(steps):
        yield next(iterator)


def _reference_step(
    model, batch, current, dropout_masks, tf, device: torch.device
) -> tuple[float, dict[str, object]]:
    for metric in model.metrics:
        metric.reset_state()
    dropout_masks.clear()
    current["random_draws"] = []
    _configure_tf32(tf)
    metrics = model.train_step(batch)
    if "loss" not in metrics:
        raise RuntimeError("original training entrypoint did not report its objective")
    components = {
        metric.name.removesuffix("_loss"): torch.as_tensor(
            metric.result().numpy(), device=device
        )
        for metric in model.loss_layer.metrics
        if metric.name.endswith("_loss")
    }
    if "l2" not in current:
        raise RuntimeError("original training step did not capture its L2 regularizer")
    l2_loss = torch.as_tensor(current["l2"], device=device)
    return float(metrics["loss"].numpy()), {
        "outputs": _tensor_dict(cast(Mapping[str, object], current["outputs"]), device),
        "losses": {
            **components,
            "l2": l2_loss,
            "total": torch.as_tensor(metrics["loss"].numpy(), device=device),
        },
        "gradient_norm": _gradient_norm(
            _source_state(cast(Mapping[str, np.ndarray], current["raw"]), device)
        ),
        "learning_rate": float(model.optimizer.learning_rate.numpy()),
        "random_trace": {
            "tasks_sha256": hashlib.sha256(
                np.asarray(current["tasks"]).tobytes()
            ).hexdigest(),
            "prepared_sha256": hashlib.sha256(
                b"".join(
                    np.ascontiguousarray(value).tobytes()
                    for group in cast(
                        Mapping[str, Mapping[str, np.ndarray]], current["prepared"]
                    ).values()
                    for _, value in sorted(group.items())
                )
            ).hexdigest(),
            "dropout_masks": len(dropout_masks),
            "draw_trace": current["random_draws"],
        },
    }


def _step_comparison(
    reference: Mapping[str, object],
    package: Mapping[str, object],
    ref_state: Mapping[str, Mapping[str, torch.Tensor]],
    pkg_state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, object]:
    reference_tensors = {
        **{
            f"output/{key}": value
            for key, value in cast(
                Mapping[str, torch.Tensor], reference["outputs"]
            ).items()
        },
        **{
            f"loss/{key}": value
            for key, value in cast(
                Mapping[str, torch.Tensor], reference["losses"]
            ).items()
        },
    }
    package_tensors = {
        **{
            f"output/{key}": value
            for key, value in cast(
                Mapping[str, torch.Tensor], package["outputs"]
            ).items()
        },
        **{
            f"loss/{key}": value
            for key, value in cast(
                Mapping[str, torch.Tensor], package["losses"]
            ).items()
        },
    }
    if set(reference_tensors) != set(package_tensors):
        raise RuntimeError(
            "S1 tensor keys differ: "
            f"reference-only={sorted(set(reference_tensors) - set(package_tensors))}, "
            f"package-only={sorted(set(package_tensors) - set(reference_tensors))}"
        )
    tolerances = {
        key: TensorTolerance(
            atol=max(float(value.abs().max().item()), 1e-12)
            * (OUTPUT_LIMIT if key.startswith("output/") else SCALAR_LOSS_LIMIT),
            rtol=0.0,
        )
        for key, value in reference_tensors.items()
    }
    trace_ref = build_step_trace("original", reference_tensors)
    trace_pkg = build_step_trace("package", package_tensors)
    s1 = compare_step_trace(trace_ref, trace_pkg, tolerances)
    s1_relative_to_max = {
        item.name: item.max_abs_diff
        / max(float(reference_tensors[item.name].abs().max().item()), 1e-12)
        for item in s1.comparisons
    }
    s1_worst_tensor = max(s1_relative_to_max, key=s1_relative_to_max.__getitem__)
    s1_max = s1_relative_to_max[s1_worst_tensor]
    direct: dict[str, float] = {}
    direct_failures: dict[str, dict[str, float]] = {}
    direct_worst: dict[str, dict[str, object]] = {}
    for group in ("raw", "clipped", "m", "v"):
        if set(ref_state[group]) != set(pkg_state[group]):
            raise RuntimeError(f"{group} optimizer tensors differ")
        errors = {
            name: _relative_l2(pkg_state[group][name], ref_state[group][name])
            for name in ref_state[group]
            if not name.endswith("attention.k_proj.bias")
        }
        direct[group] = max(errors.values(), default=0.0)
        if errors:
            worst = max(errors, key=errors.__getitem__)
            direct_worst[group] = {
                "tensor": worst,
                "relative_l2": errors[worst],
            }
        direct_failures[group] = {
            name: error for name, error in errors.items() if error > S2_DIRECT_LIMIT
        }
    key_biases = [
        name for name in ref_state["raw"] if name.endswith("attention.k_proj.bias")
    ]
    zero_gradient = max(
        (
            float(
                torch.maximum(
                    ref_state["raw"][name].abs().max(),
                    pkg_state["raw"][name].abs().max(),
                ).item()
            )
            for name in key_biases
        ),
        default=0.0,
    )
    parameter_difference, largest_parameter = _max_difference(
        ref_state["after"], pkg_state["after"]
    )
    s2_max = max(direct.values(), default=0.0)
    if not all(
        math.isfinite(value)
        for value in (s1_max, *direct.values(), zero_gradient, parameter_difference)
    ):
        raise RuntimeError("non-finite step comparison metric")
    return {
        "s1_pass": s1.passed,
        "s1_max_rel_to_max": s1_max,
        "s1_missing": list(s1.missing),
        "s1_comparison_keys": sorted(reference_tensors),
        "s1_worst_tensor": s1_worst_tensor,
        "s2_direct_rel_l2_max": direct,
        "s2_direct_failures": direct_failures,
        "s2_direct_worst_by_group": direct_worst,
        "s2_key_bias_max_abs": zero_gradient,
        "s2_direct_pass": s2_max <= S2_DIRECT_LIMIT
        and zero_gradient <= S2_ZERO_GRADIENT_LIMIT,
        "s2_pass": s2_max <= S2_DIRECT_LIMIT
        and zero_gradient <= S2_ZERO_GRADIENT_LIMIT,
        "max_parameter_difference": parameter_difference,
        "largest_difference_parameter": largest_parameter,
        "s1_comparisons": [
            {
                **item.__dict__,
                "relative_to_tensor_max": s1_relative_to_max[item.name],
            }
            for item in s1.comparisons
        ],
    }


def _synchronized_update(
    dataset: DatasetName,
    update: int,
    seed: int,
    batch: Mapping[str, object],
    original,
    names: Mapping[str, str],
    current: dict[str, object],
    reference_loss: float,
    reference: Mapping[str, object],
    reference_state: Mapping[str, Mapping[str, torch.Tensor]],
    module: FlexDmTrainingModule,
    optimizer: torch.optim.Optimizer,
    dropout_masks: list[np.ndarray],
    tf,
    device: torch.device,
) -> tuple[float, dict[str, object], dict[str, object], float, dict[str, object]]:
    prepared = cast(dict[str, object], current["prepared"])
    prepared["tasks"] = current["tasks"]
    if int(current["optimizer_step_before"]) != update - 1:
        raise RuntimeError("reference Adam step differs from synchronized update")
    torch.manual_seed(seed + update)
    package_loss, package, package_state, _ = _package_update(
        module,
        optimizer,
        _tensor_dict(batch, device),
        update=update - 1,
        device=device,
        prepared=prepared,
        dropout_masks=dropout_masks,
    )
    _synchronize_package_boundary(
        original, names, reference_state, module, optimizer, device
    )
    comparison = _step_comparison(reference, package, reference_state, package_state)
    comparison.update(
        _s2_arbitration(
            update,
            module,
            reference_state,
            package_state,
            cast(Mapping[str, Mapping[str, float]], comparison["s2_direct_failures"]),
            cast(Mapping[str, np.ndarray], current["before"]),
            prepared,
            prepared,
            dropout_masks,
            dropout_masks,
            device,
            arbitrate_every_update=True,
            optimizer=optimizer,
        )
    )
    sort_diagnostic = (
        _rico_sort_diagnostic(original, current, module.model, package, tf)
        if dataset == "rico"
        else {"status": "not_applicable"}
    )
    difference = _loss_difference(reference_loss, package_loss)
    task_ids = [original.task_names[int(task)] for task in np.asarray(current["tasks"])]
    row = {
        "record": "update",
        "update": update,
        "task_ids": task_ids,
        "task_counts": dict(sorted(Counter(task_ids).items())),
        "pos_sort_diagnostic": sort_diagnostic,
        "loss_reference": reference_loss,
        "loss_package": package_loss,
        "relative_loss_difference": difference,
        "gradient_norm_reference": reference["gradient_norm"],
        "gradient_norm_package": package["gradient_norm"],
        "max_parameter_difference": comparison["max_parameter_difference"],
        "largest_difference_parameter": comparison["largest_difference_parameter"],
        "learning_rate_reference": reference["learning_rate"],
        "learning_rate_package": package["learning_rate"],
        "scheduler_sync": "not_applicable: neither path configures a scheduler",
        **comparison,
    }
    return package_loss, package, comparison, difference, row


def _rico_sort_diagnostic(
    original, current, package_model, package, tf
) -> dict[str, object]:
    from flex_dm.modeling_flex_dm import _sort_indices
    from mfp.models.tensor_utils import sort_inputs

    task_ids = np.asarray(current["tasks"])
    positions = np.flatnonzero(task_ids == original.task_names.index("pos"))
    if not len(positions):
        return {
            "pos_rows": 0,
            "position_key_tie_groups_reference": 0,
            "position_key_tie_groups_package": 0,
            "target_sort_orders_agree": None,
            "prediction_key_tie_groups_reference": 0,
            "prediction_key_tie_groups_package": 0,
            "prediction_sort_orders_agree": None,
        }

    prepared = cast(Mapping[str, Mapping[str, np.ndarray]], current["prepared"])
    labels = {
        key: tf.convert_to_tensor(value) for key, value in prepared["labels"].items()
    }
    reference_outputs = {
        key: tf.convert_to_tensor(value)
        for key, value in cast(Mapping[str, np.ndarray], current["outputs"]).items()
    }
    reference_outputs["length"] = labels["length"]
    package_outputs = {
        key: value.detach().cpu()
        for key, value in cast(Mapping[str, torch.Tensor], package["outputs"]).items()
    }
    package_labels = {
        key: torch.as_tensor(value.numpy()) for key, value in labels.items()
    }

    def reference_order(values, *, from_logits: bool) -> np.ndarray:
        marker = tf.broadcast_to(
            tf.range(tf.shape(values["type"])[1])[tf.newaxis, :, tf.newaxis],
            tf.concat((tf.shape(values["type"])[0:2], [1]), axis=0),
        )
        columns = {
            **original.input_columns,
            "__s3_index": {"is_sequence": True, "type": "numerical", "input_dim": 1},
        }
        sorted_values = sort_inputs(
            {**values, "__s3_index": marker}, columns, from_logits=from_logits
        )
        return sorted_values["__s3_index"][..., 0].numpy()

    columns = original.input_columns
    package_columns = package_model.config.input_columns

    def package_order(values, *, from_logits: bool) -> np.ndarray:
        values = {key: value for key, value in values.items() if key in columns}
        return _sort_indices(
            values,
            package_columns,
            package_labels["length"],
            from_logits=from_logits,
        ).numpy()

    def priorities(values, *, logits: bool, torch_values: bool) -> np.ndarray:
        keys = []
        for name in ("type", "left", "top", "width", "height"):
            value = values[name]
            spec = (package_columns if torch_values else columns)[name]
            if logits and spec["type"] == "categorical":
                value = (
                    value.argmax(dim=-1) if torch_values else tf.argmax(value, axis=-1)
                )
            if torch_values:
                array = value[..., 0].long().cpu().numpy()
            else:
                array = tf.cast(value[..., 0], tf.int64).numpy()
            keys.append(array)
        priority = np.zeros_like(keys[0], dtype=np.int64)
        for key in keys:
            priority = priority * 100 + key
        return priority

    ref_keys = priorities(labels, logits=False, torch_values=False)[positions]
    pkg_keys = priorities(package_labels, logits=False, torch_values=True)[positions]
    valid_counts = [int(value[0]) + 1 for value in labels["length"].numpy()[positions]]

    def tie_groups(values: np.ndarray, count: int) -> int:
        return int(
            np.count_nonzero(np.unique(values[:count], return_counts=True)[1] > 1)
        )

    position_tie_groups_reference = sum(
        tie_groups(values, count)
        for values, count in zip(ref_keys, valid_counts, strict=True)
    )
    position_tie_groups_package = sum(
        tie_groups(values, count)
        for values, count in zip(pkg_keys, valid_counts, strict=True)
    )
    reference_target_order = reference_order(labels, from_logits=False)[positions]
    package_target_order = package_order(package_labels, from_logits=False)[positions]
    reference_prediction_order = reference_order(reference_outputs, from_logits=True)[
        positions
    ]
    package_prediction_order = package_order(package_outputs, from_logits=True)[
        positions
    ]
    mismatches = sum(
        not np.array_equal(ref, pkg)
        and np.array_equal(
            ref[reference_target_order[row]], pkg[package_target_order[row]]
        )
        for row, (ref, pkg) in enumerate(zip(ref_keys, pkg_keys, strict=True))
    )
    prediction_keys_reference = priorities(
        reference_outputs, logits=True, torch_values=False
    )[positions]
    prediction_keys_package = priorities(
        package_outputs, logits=True, torch_values=True
    )[positions]
    prediction_tie_groups_reference = sum(
        tie_groups(values, count)
        for values, count in zip(prediction_keys_reference, valid_counts, strict=True)
    )
    prediction_tie_groups_package = sum(
        tie_groups(values, count)
        for values, count in zip(prediction_keys_package, valid_counts, strict=True)
    )
    prediction_order_mismatches = sum(
        not np.array_equal(ref_order, pkg_order)
        for ref_order, pkg_order in zip(
            reference_prediction_order, package_prediction_order, strict=True
        )
    )
    prediction_tie_mismatches = sum(
        not np.array_equal(ref_order, pkg_order)
        and np.array_equal(
            ref_keys_row[ref_order[: valid_counts[row]]],
            pkg_keys_row[pkg_order[: valid_counts[row]]],
        )
        for row, (ref_order, pkg_order, ref_keys_row, pkg_keys_row) in enumerate(
            zip(
                reference_prediction_order,
                package_prediction_order,
                prediction_keys_reference,
                prediction_keys_package,
                strict=True,
            )
        )
    )
    target_order_mismatches = sum(
        not np.array_equal(ref_order, pkg_order)
        for ref_order, pkg_order in zip(
            reference_target_order, package_target_order, strict=True
        )
    )
    return {
        "pos_rows": len(positions),
        "position_key_tie_groups_reference": position_tie_groups_reference,
        "position_key_tie_groups_package": position_tie_groups_package,
        "target_sort_orders_agree": target_order_mismatches == 0,
        "target_tie_order_mismatches": mismatches,
        "prediction_key_tie_groups_reference": prediction_tie_groups_reference,
        "prediction_key_tie_groups_package": prediction_tie_groups_package,
        "prediction_sort_orders_agree": prediction_order_mismatches == 0,
        "prediction_order_mismatches": int(prediction_order_mismatches),
        "prediction_tie_order_mismatches": int(prediction_tie_mismatches),
        "tie_order_explains_loss_gap": bool(mismatches or prediction_tie_mismatches),
    }


def _state_hash(values: Mapping[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        "".join(tensor_sha256(value) for _, value in sorted(values.items())).encode()
    ).hexdigest()


def _s2_arbitration(
    update: int,
    module: FlexDmTrainingModule,
    reference: Mapping[str, Mapping[str, torch.Tensor]],
    package: Mapping[str, Mapping[str, torch.Tensor]],
    direct_failures: Mapping[str, Mapping[str, float]],
    reference_before: Mapping[str, np.ndarray],
    reference_prepared: Mapping[str, object],
    package_prepared: Mapping[str, object],
    reference_dropout_masks: list[np.ndarray],
    package_dropout_masks: list[np.ndarray],
    device: torch.device,
    *,
    arbitrate_every_update: bool = False,
    later_updates_report_only: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    direct_failures = {
        group: dict(values) for group, values in direct_failures.items() if values
    }
    key_bias = max(
        (
            float(
                torch.maximum(
                    reference["raw"][name].abs().max(), package["raw"][name].abs().max()
                ).item()
            )
            for name in reference["raw"]
            if name.endswith("attention.k_proj.bias")
        ),
        default=0.0,
    )
    arbitration_by_group: dict[str, dict[str, object]] = {
        group: {"required": False, "max_rel_l2": None, "tensors": {}}
        for group in ("raw", "clipped", "m", "v")
    }
    if not direct_failures:
        return {
            "s2_pass": key_bias <= S2_ZERO_GRADIENT_LIMIT,
            "s2_float64_arbitration_max_rel": None,
            "s2_float64_arbitration_pass": None,
            "s2_float64_arbitration_applied": False,
            "s2_float64_arbitration_required": False,
            "s2_float64_arbitration_by_group": arbitration_by_group,
        }
    if update != 1 and not arbitrate_every_update and later_updates_report_only:
        return {
            "s2_pass": None,
            "s2_float64_arbitration_max_rel": None,
            "s2_float64_arbitration_pass": None,
            "s2_float64_arbitration_applied": False,
            "s2_float64_arbitration_required": False,
            "s2_float64_arbitration_by_group": arbitration_by_group,
        }
    if update != 1 and not arbitrate_every_update:
        return {
            "s2_pass": False,
            "s2_float64_arbitration_max_rel": None,
            "s2_float64_arbitration_pass": False,
            "s2_float64_arbitration_applied": False,
            "s2_float64_arbitration_required": False,
            "s2_float64_arbitration_by_group": arbitration_by_group,
        }

    from .test_flex_dm_training_s2 import _float64_model_trace

    package_initial = cast(dict[str, torch.Tensor], package["_before_model_state"])
    reference_initial = {name: value.clone() for name, value in package_initial.items()}
    reference_parameters = _source_state(reference_before, device)
    reference_initial.update(reference_parameters)
    reference_inputs = _tensor_dict(
        cast(Mapping[str, object], reference_prepared["inputs"]), device
    )
    reference_labels = _tensor_dict(
        cast(Mapping[str, object], reference_prepared["labels"]), device
    )
    reference_masks = _tensor_dict(
        cast(Mapping[str, object], reference_prepared["masks"]), device
    )
    reference_tasks = torch.as_tensor(
        reference_prepared["tasks"], dtype=torch.long, device=device
    )
    reference_m = cast(dict[str, torch.Tensor], reference["before_m"])
    reference_v = cast(dict[str, torch.Tensor], reference["before_v"])
    package_inputs = _tensor_dict(
        cast(Mapping[str, object], package_prepared["inputs"]), device
    )
    package_labels = _tensor_dict(
        cast(Mapping[str, object], package_prepared["labels"]), device
    )
    package_masks = _tensor_dict(
        cast(Mapping[str, object], package_prepared["masks"]), device
    )
    package_tasks = torch.as_tensor(
        package_prepared["tasks"], dtype=torch.long, device=device
    )
    package_m = cast(dict[str, torch.Tensor], package["before_m"])
    package_v = cast(dict[str, torch.Tensor], package["before_v"])
    rng = capture_rng_state()
    try:
        exact_reference = _float64_model_trace(
            module,
            reference_initial,
            reference_inputs,
            reference_labels,
            reference_masks,
            reference_tasks,
            reference_dropout_masks,
            reference_m,
            reference_v,
            update,
            optimizer,
        )
        exact_package = _float64_model_trace(
            module,
            package_initial,
            package_inputs,
            package_labels,
            package_masks,
            package_tasks,
            package_dropout_masks,
            package_m,
            package_v,
            update,
            optimizer,
        )
    finally:
        from traingen_parity.determinism import restore_rng_state

        restore_rng_state(rng)
    arbitration_errors = []
    for group, names in direct_failures.items():
        group_errors = {}
        for name in names:
            package_error = _relative_l2(
                package[group][name], exact_package[group][name]
            )
            reference_error = _relative_l2(
                reference[group][name], exact_reference[group][name]
            )
            maximum_error = max(package_error, reference_error)
            group_errors[name] = {
                "direct_rel_l2": direct_failures[group][name],
                "reference_vs_float64_rel_l2": reference_error,
                "package_vs_float64_rel_l2": package_error,
                "max_rel_l2": maximum_error,
            }
            arbitration_errors.extend((package_error, reference_error))
        arbitration_by_group[group] = {
            "required": True,
            "max_rel_l2": max(
                (value["max_rel_l2"] for value in group_errors.values()),
                default=0.0,
            ),
            "tensors": group_errors,
        }
    maximum = max(arbitration_errors, default=0.0)
    passed = maximum <= S2_FLOAT64_LIMIT and key_bias <= S2_ZERO_GRADIENT_LIMIT
    return {
        "s2_pass": passed,
        "s2_float64_arbitration_max_rel": maximum,
        "s2_float64_arbitration_pass": passed,
        "s2_float64_arbitration_applied": True,
        "s2_float64_arbitration_required": True,
        "s2_float64_arbitration_by_group": arbitration_by_group,
    }


def _finite(*values: float, update: int, phase: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite {phase} metric at update {update}")


def _run_natural(
    dataset: DatasetName,
    steps: int,
    output: Path,
    seed: int,
    tf,
    device: torch.device,
) -> dict[str, object]:
    runs: list[list[dict[str, object]]] = []
    first_divergences: list[int | None] = []
    for repeat in range(2):
        _seed(seed, tf)
        dataspec, original, module, optimizer, names = _make_pair(dataset, tf, device)
        current: dict[str, object] = {}
        restore = _capture_original(original, tf, current, names)
        dropout_masks = _capture_dropout_masks(original.model, tf)
        rows = []
        trace_path = output / f"natural-repeat-{repeat}.jsonl"
        trace_path.touch(exist_ok=False)
        try:
            for update, batch in enumerate(
                _source_batches(dataspec, steps, seed, tf), 1
            ):
                reference_loss, reference = _reference_step(
                    original, batch, current, dropout_masks, tf, device
                )
                package_batch = _tensor_dict(batch, device)
                package_prepared: dict[str, object] = {}
                package_dropout_masks: list[np.ndarray] = []
                package_loss, package, pkg_state, _ = _package_update(
                    module,
                    optimizer,
                    package_batch,
                    update=update - 1,
                    device=device,
                    captured_prepared=package_prepared,
                    captured_dropout_masks=package_dropout_masks,
                )
                ref_state = _reference_boundary(original, names, current, device)
                comparison = _step_comparison(reference, package, ref_state, pkg_state)
                reference_prepared = dict(
                    cast(Mapping[str, object], current["prepared"])
                )
                reference_prepared["tasks"] = current["tasks"]
                comparison.update(
                    _s2_arbitration(
                        update,
                        module,
                        ref_state,
                        pkg_state,
                        cast(
                            Mapping[str, Mapping[str, float]],
                            comparison["s2_direct_failures"],
                        ),
                        cast(Mapping[str, np.ndarray], current["before"]),
                        reference_prepared,
                        package_prepared,
                        dropout_masks,
                        package_dropout_masks,
                        device,
                        optimizer=optimizer,
                    )
                )
                reference_state_hash = _state_hash(ref_state["after"])
                row = {
                    "record": "update",
                    "repeat": repeat,
                    "update": update,
                    "batch_sha256": _batch_hash(batch),
                    "reference_loss": reference_loss,
                    "package_loss": package_loss,
                    "relative_loss_difference": _loss_difference(
                        reference_loss, package_loss
                    ),
                    "reference_gradient_norm": reference["gradient_norm"],
                    "package_gradient_norm": package["gradient_norm"],
                    "reference_learning_rate": reference["learning_rate"],
                    "package_learning_rate": package["learning_rate"],
                    "reference_random_trace": reference["random_trace"],
                    "package_random_trace": {
                        "dropout_masks": len(package_dropout_masks),
                        "dropout_mask_sha256": [
                            hashlib.sha256(mask.tobytes()).hexdigest()
                            for mask in package_dropout_masks
                        ],
                    },
                    "reference_gradient_sha256": {
                        group: _state_hash(ref_state[group])
                        for group in ("raw", "clipped")
                    },
                    "package_gradient_sha256": {
                        group: _state_hash(pkg_state[group])
                        for group in ("raw", "clipped")
                    },
                    "reference_optimizer_sha256": {
                        group: _state_hash(ref_state[group]) for group in ("m", "v")
                    },
                    "package_optimizer_sha256": {
                        group: _state_hash(pkg_state[group]) for group in ("m", "v")
                    },
                    "reference_state_sha256": reference_state_hash,
                    "package_state_sha256": package["state_sha256"],
                    **comparison,
                }
                rows.append(row)
                with trace_path.open("a", encoding="utf-8") as trace:
                    _write_row(trace, row)
                _finite(
                    reference_loss,
                    package_loss,
                    float(row["relative_loss_difference"]),
                    float(reference["gradient_norm"]),
                    float(package["gradient_norm"]),
                    float(comparison["max_parameter_difference"]),
                    update=update,
                    phase="natural",
                )
        finally:
            restore()
        if len(rows) != steps:
            raise RuntimeError(
                "natural trajectory stopped before the requested updates"
            )
        runs.append(rows)
        first_divergences.append(
            next(
                (
                    int(row["update"])
                    for row in rows
                    if not bool(row["s1_pass"]) or not bool(row["s2_pass"])
                ),
                None,
            )
        )

    if any(
        first["batch_sha256"] != second["batch_sha256"]
        for first, second in zip(*runs, strict=True)
    ):
        raise RuntimeError("natural repeats did not receive the same batches")

    fields = (
        "reference_loss",
        "package_loss",
        "reference_gradient_norm",
        "package_gradient_norm",
        "max_parameter_difference",
    )
    envelope = {
        key: max(
            (
                abs(float(first[key]) - float(second[key]))
                for first, second in zip(*runs, strict=True)
            ),
            default=0.0,
        )
        for key in fields
    }
    envelope["reference_state_hash_mismatches"] = sum(
        first["reference_state_sha256"] != second["reference_state_sha256"]
        for first, second in zip(*runs, strict=True)
    )
    envelope["package_state_hash_mismatches"] = sum(
        first["package_state_sha256"] != second["package_state_sha256"]
        for first, second in zip(*runs, strict=True)
    )
    direct_pass = [
        all(bool(row["s1_pass"]) and bool(row["s2_pass"]) for row in rows)
        for rows in runs
    ]
    return {
        "phase": "natural",
        "updates_per_repeat": steps,
        "repeats": len(runs),
        "first_s0_s2_divergence_by_repeat": first_divergences,
        "s3_pass_each_repeat": direct_pass,
        "run_to_run_envelope": envelope,
        "records": [f"natural-repeat-{index}.jsonl" for index in range(len(runs))],
    }


def _run_lockstep(
    dataset: DatasetName,
    steps: int,
    output: Path,
    seed: int,
    tf,
    device: torch.device,
) -> dict[str, object]:
    _seed(seed, tf)
    dataspec, original, cross_model, cross_optimizer, names = _make_pair(
        dataset, tf, device
    )
    synchronized_model, synchronized_optimizer = _make_package(dataset, device)
    noise_model, noise_optimizer = _make_package(dataset, device)
    perturbed_model, perturbed_optimizer = _make_package(dataset, device)
    with torch.no_grad():
        for parameter in perturbed_model.model.parameters():
            if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
                raise RuntimeError("noise-floor parameters must be finite float32")
            parameter.copy_(
                torch.nextafter(parameter, torch.full_like(parameter, float("inf")))
            )

    current: dict[str, object] = {}
    restore = _capture_original(original, tf, current, names)
    dropout_masks = _capture_dropout_masks(original.model, tf)
    cross_rows = []
    synchronized_rows = []
    noise_rows = []
    cross_exceedance = None
    noise_exceedance = None
    cross_exceedance_parameter = None
    noise_exceedance_parameter = None
    started = time.perf_counter()
    cross_path = output / "cross-system.jsonl"
    synchronized_path = output / "synchronized.jsonl"
    noise_path = output / "noise-floor.jsonl"
    cross_path.touch(exist_ok=False)
    synchronized_path.touch(exist_ok=False)
    noise_path.touch(exist_ok=False)
    try:
        for update, batch in enumerate(_source_batches(dataspec, steps, seed, tf), 1):
            _seed(seed + update, tf)
            reference_loss, reference = _reference_step(
                original, batch, current, dropout_masks, tf, device
            )
            package_batch = _tensor_dict(batch, device)
            prepared = cast(dict[str, object], current["prepared"])
            prepared["tasks"] = current["tasks"]
            torch.manual_seed(seed + update)
            cross_loss, cross_result, cross_state, _ = _package_update(
                cross_model,
                cross_optimizer,
                package_batch,
                update=update - 1,
                device=device,
                prepared=prepared,
                dropout_masks=dropout_masks,
            )
            reference_state = _reference_boundary(original, names, current, device)
            comparison = _step_comparison(
                reference, cross_result, reference_state, cross_state
            )
            comparison.update(
                _s2_arbitration(
                    update,
                    cross_model,
                    reference_state,
                    cross_state,
                    cast(
                        Mapping[str, Mapping[str, float]],
                        comparison["s2_direct_failures"],
                    ),
                    cast(Mapping[str, np.ndarray], current["before"]),
                    prepared,
                    prepared,
                    dropout_masks,
                    dropout_masks,
                    device,
                    later_updates_report_only=True,
                    optimizer=cross_optimizer,
                )
            )
            (
                synchronized_loss,
                synchronized_result,
                synchronized_comparison,
                synchronized_difference,
                synchronized_row,
            ) = _synchronized_update(
                dataset,
                update,
                seed,
                batch,
                original,
                names,
                current,
                reference_loss,
                reference,
                reference_state,
                synchronized_model,
                synchronized_optimizer,
                dropout_masks,
                tf,
                device,
            )

            cross_difference = _loss_difference(reference_loss, cross_loss)
            if cross_difference > LOSS_LIMIT and cross_exceedance is None:
                cross_exceedance = update
                cross_exceedance_parameter = comparison["largest_difference_parameter"]

            torch.manual_seed(seed + update)
            first_loss, first_result, first_state, first_rng = _package_update(
                noise_model,
                noise_optimizer,
                package_batch,
                update=update - 1,
                device=device,
                prepared=prepared,
                dropout_masks=dropout_masks,
            )
            torch.manual_seed(seed + update)
            second_loss, second_result, second_state, second_rng = _package_update(
                perturbed_model,
                perturbed_optimizer,
                package_batch,
                update=update - 1,
                device=device,
                prepared=prepared,
                dropout_masks=dropout_masks,
            )
            if not _rng_equal(first_rng, second_rng):
                raise RuntimeError(
                    f"noise-floor random draws diverged at update {update}"
                )
            noise_difference = _loss_difference(first_loss, second_loss)
            noise_maximum, noise_parameter = _max_difference(
                first_state["after"], second_state["after"]
            )
            if noise_difference > LOSS_LIMIT and noise_exceedance is None:
                noise_exceedance = update
                noise_exceedance_parameter = noise_parameter

            random_trace = reference["random_trace"]
            sampler_state = {
                "batch_sha256": _batch_hash(batch),
                "task_sha256": random_trace["tasks_sha256"],
                "prepared_sha256": random_trace["prepared_sha256"],
                "draw_trace": random_trace["draw_trace"],
                "dropout_masks": len(dropout_masks),
            }
            cross_rows.append(
                {
                    "record": "update",
                    "update": update,
                    "loss_reference": reference_loss,
                    "loss_package": cross_loss,
                    "relative_loss_difference": cross_difference,
                    "gradient_norm_reference": reference["gradient_norm"],
                    "gradient_norm_package": cross_result["gradient_norm"],
                    "max_parameter_difference": comparison["max_parameter_difference"],
                    "largest_difference_parameter": comparison[
                        "largest_difference_parameter"
                    ],
                    "sampler_state": sampler_state,
                    "learning_rate_reference": reference["learning_rate"],
                    "learning_rate_package": cross_result["learning_rate"],
                    "s1_pass": comparison["s1_pass"],
                    "s1_max_rel_to_max": comparison["s1_max_rel_to_max"],
                    "s1_worst_tensor": comparison["s1_worst_tensor"],
                    "s1_comparisons": comparison["s1_comparisons"],
                    "s1_missing": comparison["s1_missing"],
                    "s1_comparison_keys": comparison["s1_comparison_keys"],
                    "s2_direct_pass": comparison["s2_direct_pass"],
                    "s2_pass": comparison["s2_pass"],
                    "s2_direct_rel_l2_max": comparison["s2_direct_rel_l2_max"],
                    "s2_float64_arbitration_max_rel": comparison[
                        "s2_float64_arbitration_max_rel"
                    ],
                    "s2_float64_arbitration_pass": comparison[
                        "s2_float64_arbitration_pass"
                    ],
                    "s2_float64_arbitration_applied": comparison[
                        "s2_float64_arbitration_applied"
                    ],
                    "s2_float64_arbitration_required": comparison[
                        "s2_float64_arbitration_required"
                    ],
                }
            )
            synchronized_rows.append(synchronized_row)
            noise_rows.append(
                {
                    "record": "update",
                    "update": update,
                    "loss_first": first_loss,
                    "loss_second": second_loss,
                    "relative_loss_difference": noise_difference,
                    "gradient_norm_first": first_result["gradient_norm"],
                    "gradient_norm_second": second_result["gradient_norm"],
                    "max_parameter_difference": noise_maximum,
                    "largest_difference_parameter": noise_parameter,
                    "sampler_state": sampler_state,
                    "rng_equal": True,
                }
            )
            with cross_path.open("a", encoding="utf-8") as trace:
                _write_row(trace, cross_rows[-1])
            with synchronized_path.open("a", encoding="utf-8") as trace:
                _write_row(trace, synchronized_rows[-1])
            with noise_path.open("a", encoding="utf-8") as trace:
                _write_row(trace, noise_rows[-1])
            _finite(
                reference_loss,
                cross_loss,
                cross_difference,
                float(reference["gradient_norm"]),
                float(cross_result["gradient_norm"]),
                float(comparison["max_parameter_difference"]),
                first_loss,
                second_loss,
                noise_difference,
                noise_maximum,
                synchronized_loss,
                synchronized_difference,
                float(first_result["gradient_norm"]),
                float(second_result["gradient_norm"]),
                update=update,
                phase="lockstep",
            )
    finally:
        restore()

    if (
        len(cross_rows) != steps
        or len(synchronized_rows) != steps
        or len(noise_rows) != steps
    ):
        raise RuntimeError("a lockstep phase stopped before the requested update count")
    attribution_pass = _gate_passed(cross_exceedance, noise_exceedance)
    cross_s0_s2_pass = all(
        bool(row["s1_pass"]) and bool(row["s2_pass"]) for row in cross_rows
    )
    synchronized_s0_s2_pass = all(
        bool(row["s1_pass"]) and bool(row["s2_pass"]) for row in synchronized_rows
    )
    elapsed = time.perf_counter() - started
    return {
        "phase": "lockstep",
        "updates_per_phase": steps,
        "cross_system_first_exceedance": cross_exceedance,
        "cross_system_first_exceedance_parameter": cross_exceedance_parameter,
        "noise_floor_first_exceedance": noise_exceedance,
        "noise_floor_first_exceedance_parameter": noise_exceedance_parameter,
        "cross_system_s0_s2_all_rows_pass_report_only": cross_s0_s2_pass,
        "synchronized_within_s0_s2": synchronized_s0_s2_pass,
        "synchronized_first_s0_s2_divergence": next(
            (
                int(row["update"])
                for row in synchronized_rows
                if not bool(row["s1_pass"]) or not bool(row["s2_pass"])
            ),
            None,
        ),
        "noise_floor_attributable": attribution_pass,
        "elapsed_seconds": elapsed,
        "updates_per_second_report_only": steps / elapsed,
        "records": ["cross-system.jsonl", "noise-floor.jsonl", "synchronized.jsonl"],
    }


def _run_synchronized_only(
    dataset: DatasetName,
    steps: int,
    output: Path,
    seed: int,
    tf,
    device: torch.device,
) -> dict[str, object]:
    _seed(seed, tf)
    dataspec, original, synchronized_model, synchronized_optimizer, names = _make_pair(
        dataset, tf, device
    )
    current: dict[str, object] = {}
    restore = _capture_original(original, tf, current, names)
    dropout_masks = _capture_dropout_masks(original.model, tf)
    rows = []
    trace_path = output / "synchronized.jsonl"
    trace_path.touch(exist_ok=False)
    started = time.perf_counter()
    try:
        for update, batch in enumerate(_source_batches(dataspec, steps, seed, tf), 1):
            _seed(seed + update, tf)
            reference_loss, reference = _reference_step(
                original, batch, current, dropout_masks, tf, device
            )
            reference_state = _reference_boundary(original, names, current, device)
            package_loss, package_result, comparison, difference, row = (
                _synchronized_update(
                    dataset,
                    update,
                    seed,
                    batch,
                    original,
                    names,
                    current,
                    reference_loss,
                    reference,
                    reference_state,
                    synchronized_model,
                    synchronized_optimizer,
                    dropout_masks,
                    tf,
                    device,
                )
            )
            rows.append(row)
            with trace_path.open("a", encoding="utf-8") as trace:
                _write_row(trace, row)
            _finite(
                reference_loss,
                package_loss,
                difference,
                float(reference["gradient_norm"]),
                float(package_result["gradient_norm"]),
                float(comparison["max_parameter_difference"]),
                update=update,
                phase="synchronized",
            )
    finally:
        restore()

    if len(rows) != steps:
        raise RuntimeError(
            "the synchronized phase stopped before the requested update count"
        )
    first_divergence = next(
        (
            int(row["update"])
            for row in rows
            if not bool(row["s1_pass"]) or not bool(row["s2_pass"])
        ),
        None,
    )
    elapsed = time.perf_counter() - started
    return {
        "phase": "synchronized",
        "updates_per_phase": steps,
        "synchronized_within_s0_s2": all(
            bool(row["s1_pass"]) and bool(row["s2_pass"]) for row in rows
        ),
        "synchronized_first_s0_s2_divergence": first_divergence,
        "elapsed_seconds": elapsed,
        "updates_per_second_report_only": steps / elapsed,
        "records": ["synchronized.jsonl"],
    }


def _run_output_path(parser: argparse.ArgumentParser, raw_path: Path) -> Path:
    output = raw_path.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        parser.error("output must be a new directory outside the repository")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--phase", choices=("all", "synchronized"), default="all")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        if args.dry_run or args.output_dir is not None:
            parser.error("preflight does not accept --dry-run or --output-dir")
        _preflight(args.dataset)
        return
    if args.output_dir is None:
        parser.error("run mode requires --output-dir")
    if args.steps < 300:
        parser.error("S3 evidence phases require at least 300 updates")
    output = _run_output_path(parser, args.output_dir)
    if os.environ.get("PARITY_REQUIRE") != "1":
        parser.error("run mode requires PARITY_REQUIRE=1")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "dataset": args.dataset,
                    "phase_selector": args.phase,
                    "steps_per_phase": args.steps,
                    "output_dir": str(output),
                },
                sort_keys=True,
            )
        )
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        parser.error("run mode requires one selected GPU")
    _validate_inputs(args.dataset)
    output.mkdir(parents=True)
    tf = _tensorflow()
    device = _assert_gpu_environment(tf)
    _seed(args.seed, tf)
    if args.phase == "synchronized":
        synchronized = _run_synchronized_only(
            args.dataset, args.steps, output, args.seed, tf, device
        )
        result = {
            "dataset": args.dataset,
            "phase_selector": args.phase,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "vendor_commit": subprocess.check_output(
                ["git", "-C", "vendor/flex-dm", "rev-parse", "HEAD"], text=True
            ).strip(),
            "seed": args.seed,
            "tf32": "disabled in TensorFlow and PyTorch",
            "tensorflow_gpu_memory_growth": "enabled",
            "synchronized": synchronized,
            "gate_eligible": False,
        }
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return
    natural = _run_natural(args.dataset, args.steps, output, args.seed, tf, device)
    lockstep = _run_lockstep(args.dataset, args.steps, output, args.seed, tf, device)
    natural_passed = all(cast(list[bool], natural["s3_pass_each_repeat"]))
    lockstep_attributable = bool(lockstep["noise_floor_attributable"])
    synchronized_passed = bool(lockstep["synchronized_within_s0_s2"])
    result = {
        "dataset": args.dataset,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "vendor_commit": subprocess.check_output(
            ["git", "-C", "vendor/flex-dm", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": args.seed,
        "phase_selector": args.phase,
        "tf32": "disabled in TensorFlow and PyTorch",
        "tensorflow_gpu_memory_growth": "enabled",
        "natural": natural,
        "lockstep": lockstep,
        "natural_s3_pass": natural_passed,
        "synchronized_diagnostic_required": not natural_passed,
        "passed": _s3_passed(
            cast(list[bool], natural["s3_pass_each_repeat"]),
            synchronized_passed,
            lockstep_attributable,
        ),
        "gate_eligible": True,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
