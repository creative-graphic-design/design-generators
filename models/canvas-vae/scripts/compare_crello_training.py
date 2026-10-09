"""Compare Crello training steps in package PyTorch and vendor TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Final, NamedTuple, Protocol, TypeAlias, TypedDict, cast

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
from jaxtyping import Float, Shaped
from laygen.common.vendor import vendor_root

from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
from canvas_vae.conversion import (
    TensorFlowSource,
    convert_tensorflow_crello_variables,
    tensorflow_crello_key_map,
)
from canvas_vae.data import (
    CrelloBatch,
    CrelloProcessor,
    CrelloSplit,
    fixture_image_ids,
    load_crello_split,
    load_crello_vocabularies,
    load_embedding_fixture,
)
from canvas_vae.modeling_canvas_vae import (
    CanvasVAECrelloModel,
    CanvasVAECrelloModelOutput,
)
from canvas_vae.training.optim import KerasAdam, clip_gradients_by_norm, l2_penalty
from canvas_vae.training.sampling import sequential_batches

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
VENDOR_ROOT: Final = vendor_root(
    "canvas-vae", marker="src/canvas-vae/canvasvae/train.py"
)
DATA_DIR: Final = Path(
    os.environ.get("CANVAS_VAE_DATA_DIR", ".cache/canvas-vae/crello/package-run-1")
)
FIXTURE_DIR: Final = Path(
    os.environ.get("CANVAS_VAE_FIXTURE_DIR", ".cache/canvas-vae/crello/fixture")
)
REPORT_DIR: Final = Path(
    os.environ.get(
        "CANVAS_VAE_REFERENCE_DIR", ".cache/canvas-vae/reference/crello-model"
    )
)
BATCH_SIZE: Final = 1024
LATENT_DIM: Final = 512
KL_WEIGHT: Final = 32.0
LEARNING_RATE: Final = 1e-3
CLIP_NORM: Final = 1.0
L2_WEIGHT: Final = 1e-6
CALIBRATION_SLICES: Final[tuple[tuple[int, int], ...]] = (
    (0, BATCH_SIZE),
    (BATCH_SIZE, 2 * BATCH_SIZE),
    (2 * BATCH_SIZE, 3 * BATCH_SIZE),
)
CALIBRATION_SELECTION: Final = (
    "first three non-overlapping 1,024-document slices of canonical train.jsonl: "
    "indices [0,1024), [1024,2048), [2048,3072)"
)


class _Comparison(NamedTuple):
    group: str
    name: str
    batch: int
    metric: str
    value: float | None
    actual_shape: list[int]
    expected_shape: list[int]
    shape_match: bool


class _TensorFlowTensor(Protocol):
    def numpy(self) -> Shaped[np.ndarray, "..."]: ...


class _TensorFlowModule(Protocol):
    def convert_to_tensor(
        self, value: Shaped[np.ndarray, "..."]
    ) -> _TensorFlowTensor: ...


class _TrainingTrace(TypedDict):
    z_mean: Float[np.ndarray, "batch latent"]
    z_log_var: Float[np.ndarray, "batch latent"]
    kl_divergence: float


class _FrozenLimit(TypedDict):
    metric: str
    L: float
    max_calibration_error: float
    limit: float


class _FrozenLimits(TypedDict):
    commit: str
    calibration_sha256: str
    selection: str
    formula: str
    limits: dict[str, _FrozenLimit]


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


def _comparison(
    group: str,
    name: str,
    batch: int,
    actual: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."] | float,
    expected: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."] | float,
    metric: str,
) -> _Comparison:
    actual_array = np.asarray(
        actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else actual,
        dtype=np.float64,
    )
    expected_array = np.asarray(
        expected.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor)
        else expected,
        dtype=np.float64,
    )
    shape_match = actual_array.shape == expected_array.shape
    if not shape_match:
        return _Comparison(
            group,
            name,
            batch,
            metric,
            None,
            list(actual_array.shape),
            list(expected_array.shape),
            False,
        )

    delta = np.abs(actual_array - expected_array)
    if metric == "max_rel_to_max":
        scale = max(float(np.max(np.abs(expected_array), initial=0.0)), 1e-12)
        value = float(np.max(delta, initial=0.0) / scale)
    else:
        scale = max(float(np.linalg.norm(expected_array)), 1e-30)
        value = float(np.linalg.norm(delta) / scale)

    return _Comparison(
        group,
        name,
        batch,
        metric,
        value,
        list(actual_array.shape),
        list(expected_array.shape),
        True,
    )


def _ceil_two_significant_digits(value: float) -> float:
    if value <= 0:
        return 0.0

    place = 10 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / place) * place


def _batch_digest(batch: CrelloBatch) -> str:
    digest = hashlib.sha256()
    for document_id in batch["document_id"]:
        digest.update(document_id.encode("utf-8"))

    tensor_batch = cast(Mapping[str, Shaped[torch.Tensor, "..."]], batch)
    for key in (
        "length",
        "group",
        "format",
        "canvas_width",
        "canvas_height",
        "category",
        "type",
        "left",
        "top",
        "width",
        "height",
        "opacity",
        "color",
        "image_embedding",
        "element_mask",
        "color_mask",
        "image_embedding_mask",
    ):
        value = tensor_batch[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(value.tobytes())

    return digest.hexdigest()


def _reference_inputs(
    tf: _TensorFlowModule,
    batch: CrelloBatch,
    columns: Iterable[str],
) -> dict[str, _TensorFlowTensor]:
    tensor_batch = cast(Mapping[str, Shaped[torch.Tensor, "..."]], batch)
    return {
        key: tf.convert_to_tensor(tensor_batch[key].detach().cpu().numpy())
        for key in columns
    }


def _crello_model(tf, data_dir: Path, *, dropout: float):
    from traingen_parity.tensorflow_compat import install_keras_preprocessing_compat

    install_keras_preprocessing_compat()
    batch_norm_base = tf.keras.layers.BatchNormalization

    class IgnorePropagatedMask(batch_norm_base):
        def call(self, inputs, training=None):
            return super().call(inputs, training=training)

    tf.keras.layers.BatchNormalization = IgnorePropagatedMask
    sys.path.insert(0, str(VENDOR_ROOT / "src" / "canvas-vae"))
    from canvasvae.data.spec import DataSpec
    from canvasvae.models.vae import VAE

    dataspec = DataSpec("crello-document", str(data_dir), batch_size=BATCH_SIZE)
    columns = dataspec.make_input_columns()
    model = VAE(
        columns,
        latent_dim=LATENT_DIM,
        decoder_type="oneshot",
        num_blocks=1,
        block_type="deepsvg",
        dropout=dropout,
        kl=KL_WEIGHT,
        l2=L2_WEIGHT,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM
        )
    )
    model.optimizer.build(model.trainable_variables)
    return dataspec, columns, model


def _install_fixed_noise_call(
    tf,
    model,
    noise_box: list[Float[np.ndarray, "batch latent"]],
    trace_box: list[_TrainingTrace],
) -> None:
    layer = model.encoder.head

    def call(self, inputs, training=False):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)
        kl_divergence = -0.5 * tf.reduce_mean(
            1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        )
        trace_box.append(
            {
                "z_mean": z_mean.numpy(),
                "z_log_var": z_log_sigma.numpy(),
                "kl_divergence": float(kl_divergence.numpy()),
            }
        )
        self.add_loss(self.kl * kl_divergence)
        self.add_metric(kl_divergence, name="kl_divergence")
        if training:
            z_mean += tf.exp(0.5 * z_log_sigma) * tf.convert_to_tensor(noise_box[0])

        return z_mean

    layer.call = MethodType(call, layer)


def _vendor_variables(model) -> dict[str, Shaped[np.ndarray, "..."]]:
    return {
        path.replace(".", "/"): variable.numpy()
        for path, variable in model.get_weight_paths().items()
        if not path.startswith("optimizer")
    }


def _regularization_loss(tf, model):
    terms = []
    for layer in model.submodules:
        for attribute in ("kernel", "bias", "embeddings"):
            regularizer = getattr(layer, f"{attribute}_regularizer", None)
            variable = getattr(layer, attribute, None)
            if regularizer is not None and variable is not None:
                terms.append(regularizer(variable))

    return tf.add_n(terms)


def _torch_outputs(
    output: CanvasVAECrelloModelOutput,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    if output.length_logits is None:
        raise ValueError("Crello comparison needs length logits")

    values = {"length": output.length_logits}
    values.update(output.context_logits or {})
    values.update(output.sequence_logits or {})
    values.update(output.numerical_predictions or {})
    return values


def _compare_outputs(
    records: list[_Comparison],
    group: str,
    batch_index: int,
    torch_output: CanvasVAECrelloModelOutput,
    tf_outputs: Mapping[str, Shaped[np.ndarray, "..."]],
) -> None:
    values = _torch_outputs(torch_output)
    for key, actual in values.items():
        expected = tf_outputs[key]
        if key == "length":
            expected = expected[:, 0, :]
        elif key == "color" or key == "image_embedding":
            pass
        elif key in (torch_output.context_logits or {}):
            pass
        else:
            expected = expected.squeeze(axis=2)

        records.append(
            _comparison(group, key, batch_index, actual, expected, "max_rel_to_max")
        )


def _compare_state_tensor(
    records: list[_Comparison],
    group: str,
    batch_index: int,
    name: str,
    actual: Shaped[torch.Tensor, "..."],
    expected: Shaped[np.ndarray, "..."],
    source: TensorFlowSource,
) -> None:
    if source.transpose:
        expected = expected.T

    records.append(_comparison(group, name, batch_index, actual, expected, "norm_rel"))


def _train_step(
    tf,
    vendor_model,
    package_model: CanvasVAECrelloModel,
    optimizer: KerasAdam,
    batch: CrelloBatch,
    noise: Float[np.ndarray, "batch latent"],
    batch_index: int,
    records: list[_Comparison],
) -> None:
    tf_inputs = _reference_inputs(tf, batch, vendor_model.input_columns)
    noise_box = [noise]
    trace_box: list[_TrainingTrace] = []
    _install_fixed_noise_call(tf, vendor_model, noise_box, trace_box)
    vendor_model.reset_metrics()
    with tf.GradientTape() as tape:
        vendor_outputs = vendor_model(tf_inputs, training=True)
        vendor_loss = vendor_model.compute_loss(tf_inputs, None, vendor_outputs, None)

    vendor_gradients = tape.gradient(vendor_loss, vendor_model.trainable_variables)
    vendor_paths = {
        id(variable): path.replace(".", "/")
        for path, variable in vendor_model.get_weight_paths().items()
    }
    gradients_by_path = {
        vendor_paths[id(variable)]: gradient
        for variable, gradient in zip(
            vendor_model.trainable_variables, vendor_gradients, strict=True
        )
    }
    vendor_variables = {
        path.replace(".", "/"): variable
        for path, variable in vendor_model.get_weight_paths().items()
    }
    package_model.train()
    package_output = package_model(batch, posterior_noise=torch.from_numpy(noise))
    if package_output.loss is None or package_output.kl_divergence is None:
        raise RuntimeError("package training forward returned no losses")

    package_penalty = l2_penalty(package_model, L2_WEIGHT)
    package_loss = package_output.loss + package_penalty
    records.append(
        _comparison(
            "s1_training_total_loss",
            "total_loss",
            batch_index,
            package_loss,
            float(vendor_loss.numpy()),
            "max_rel_to_max",
        )
    )
    records.append(
        _comparison(
            "s1_l2_regularization",
            "l2",
            batch_index,
            package_penalty,
            float(_regularization_loss(tf, vendor_model).numpy()),
            "max_rel_to_max",
        )
    )
    _compare_outputs(
        records,
        "s1_training_outputs",
        batch_index,
        package_output,
        {key: value.numpy() for key, value in vendor_outputs.items()},
    )
    if package_output.z_mean is not None and package_output.z_log_var is not None:
        trace = trace_box[-1]
        records.append(
            _comparison(
                "s1_posterior_statistics",
                "z_mean",
                batch_index,
                package_output.z_mean,
                trace["z_mean"],
                "max_rel_to_max",
            )
        )
        records.append(
            _comparison(
                "s1_posterior_statistics",
                "z_log_var",
                batch_index,
                package_output.z_log_var,
                trace["z_log_var"],
                "max_rel_to_max",
            )
        )
        records.append(
            _comparison(
                "s1_posterior_statistics",
                "kl_divergence",
                batch_index,
                package_output.kl_divergence,
                float(trace["kl_divergence"]),
                "max_rel_to_max",
            )
        )

    vendor_metrics = {
        metric.name: float(metric.result().numpy()) for metric in vendor_model.metrics
    }
    for key, value in (package_output.reconstruction_losses or {}).items():
        metric_value = vendor_metrics.get(f"{key}_loss")
        if metric_value is None:
            records.append(
                _Comparison(
                    "s1_reconstruction_losses",
                    key,
                    batch_index,
                    "max_rel_to_max",
                    None,
                    [],
                    [],
                    False,
                )
            )
            continue

        records.append(
            _comparison(
                "s1_reconstruction_losses",
                key,
                batch_index,
                value,
                metric_value,
                "max_rel_to_max",
            )
        )

    package_loss.backward()
    mapping = tensorflow_crello_key_map(package_model.config)
    package_parameters = dict(package_model.named_parameters())
    for state_key, source in mapping.items():
        if state_key not in package_parameters:
            continue

        tf_variable = vendor_variables[source.key]
        tf_gradient = gradients_by_path[source.key]
        if tf_gradient is None:
            records.append(
                _Comparison(
                    "s2_gradients",
                    state_key,
                    batch_index,
                    "norm_rel",
                    None,
                    [],
                    [],
                    False,
                )
            )
            continue

        package_parameter = package_parameters[state_key]
        if package_parameter.grad is None:
            records.append(
                _Comparison(
                    "s2_gradients",
                    state_key,
                    batch_index,
                    "norm_rel",
                    None,
                    [],
                    [],
                    False,
                )
            )
            continue

        _compare_state_tensor(
            records,
            "s2_gradients",
            batch_index,
            state_key,
            package_parameter.grad,
            tf_gradient.numpy(),
            source,
        )

    vendor_clipped = {
        id(variable): tf.clip_by_norm(gradient, CLIP_NORM).numpy()
        for variable, gradient in zip(
            vendor_model.trainable_variables, vendor_gradients, strict=True
        )
        if gradient is not None
    }
    clip_gradients_by_norm(package_model.parameters(), CLIP_NORM)
    for state_key, source in mapping.items():
        if state_key not in package_parameters:
            continue

        tf_variable = vendor_variables[source.key]
        package_parameter = package_parameters[state_key]
        if package_parameter.grad is None:
            records.append(
                _Comparison(
                    "s2_clipped_gradients",
                    state_key,
                    batch_index,
                    "norm_rel",
                    None,
                    [],
                    [],
                    False,
                )
            )
            continue

        _compare_state_tensor(
            records,
            "s2_clipped_gradients",
            batch_index,
            state_key,
            package_parameter.grad,
            vendor_clipped[id(tf_variable)],
            source,
        )

    vendor_model.optimizer.apply_gradients(
        zip(vendor_gradients, vendor_model.trainable_variables, strict=True)
    )
    optimizer.step()
    for state_key, source in mapping.items():
        if state_key not in package_parameters:
            continue

        tf_variable = vendor_variables[source.key]
        package_parameter = package_parameters[state_key]
        _compare_state_tensor(
            records,
            "s2_updated_parameters",
            batch_index,
            state_key,
            package_parameter,
            tf_variable.numpy(),
            source,
        )
        index = vendor_model.optimizer._index_dict[
            vendor_model.optimizer._var_key(tf_variable)
        ]
        _compare_state_tensor(
            records,
            "s2_first_moments",
            batch_index,
            state_key,
            optimizer.state[package_parameter]["exp_avg"],
            vendor_model.optimizer._momentums[index].numpy(),
            source,
        )
        _compare_state_tensor(
            records,
            "s2_second_moments",
            batch_index,
            state_key,
            optimizer.state[package_parameter]["exp_avg_sq"],
            vendor_model.optimizer._velocities[index].numpy(),
            source,
        )

    records.append(
        _comparison(
            "s3_batch_norm_mean",
            "running_mean",
            batch_index,
            package_model.encoder.norm.running_mean,
            vendor_model.encoder.norm.moving_mean.numpy(),
            "max_rel_to_max",
        )
    )
    records.append(
        _comparison(
            "s3_batch_norm_variance",
            "running_variance",
            batch_index,
            package_model.encoder.norm.running_var,
            vendor_model.encoder.norm.moving_variance.numpy(),
            "max_rel_to_max",
        )
    )


def _cpu_report_context() -> dict[str, str | int | float]:
    return {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip(),
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "kl_weight": KL_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "l2_weight": L2_WEIGHT,
        "device": "cpu",
    }


def _write_json(path: Path, payload: JSONValue) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(serialized)
    return serialized


def _record_payload(record: _Comparison) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], record._asdict())


def _has_shape_errors(records: Sequence[_Comparison]) -> bool:
    return any(not row.shape_match for row in records)


def _limits(records: Sequence[_Comparison]) -> dict[str, _FrozenLimit]:
    grouped: dict[str, list[float]] = defaultdict(list)
    metrics: dict[str, str] = {}
    for row in records:
        if row.value is not None:
            grouped[row.group].append(row.value)
            metrics[row.group] = row.metric

    return {
        group: {
            "metric": metrics[group],
            "L": 0.0,
            "max_calibration_error": maximum,
            "limit": max(0.0, _ceil_two_significant_digits(1.5 * maximum)),
        }
        for group, values in grouped.items()
        if (maximum := max(values, default=0.0)) >= 0
    }


def _common(model, package_model: CanvasVAECrelloModel) -> list[str]:
    checks = []
    if set(tensorflow_crello_key_map(package_model.config)) != set(
        package_model.state_dict()
    ):
        checks.append("TensorFlow key map does not cover the package state dict")
    vendor_parameter_count = sum(
        int(np.prod(variable.shape)) for variable in model.trainable_variables
    )
    package_parameters = tuple(package_model.parameters())
    if vendor_parameter_count != sum(
        parameter.numel() for parameter in package_parameters
    ):
        checks.append("trainable parameter counts differ")
    if len(model.trainable_variables) != len(package_parameters):
        checks.append("trainable parameter tensor counts differ")
    if package_model.config.latent_dim != LATENT_DIM:
        checks.append("package latent dimension differs from the Crello recipe")
    if package_model.config.kl_weight != KL_WEIGHT:
        checks.append("package KL coefficient differs from the Crello recipe")

    return checks


def _load_frozen_limits(report_dir: Path, commit: str) -> tuple[bytes, _FrozenLimits]:
    limits_bytes = (report_dir / "limits.json").read_bytes()
    expected_sha256 = (report_dir / "limits.sha256").read_text("ascii").strip()
    if hashlib.sha256(limits_bytes).hexdigest() != expected_sha256:
        raise ValueError("frozen Crello parity limits SHA-256 mismatch")

    limits = cast(_FrozenLimits, json.loads(limits_bytes))
    if limits["commit"] != commit:
        raise ValueError("held-out run must use the exact calibration commit")

    return limits_bytes, limits


def _run(
    phase: str,
    *,
    data_dir: Path = DATA_DIR,
    fixture_dir: Path = FIXTURE_DIR,
    report_dir: Path = REPORT_DIR,
) -> dict[str, JSONValue]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Crello model parity is CPU-only")

    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    report_context = _cpu_report_context()
    frozen_bytes = None
    frozen = None
    if phase == "heldout":
        frozen_bytes, frozen = _load_frozen_limits(
            report_dir, str(report_context["commit"])
        )
    else:
        for filename in ("limits.json", "limits.sha256"):
            (report_dir / filename).unlink(missing_ok=True)

    documents = {split: load_crello_split(data_dir, split) for split in CrelloSplit}
    if phase == "calibrate":
        if len(documents[CrelloSplit.train]) < 3 * BATCH_SIZE:
            raise ValueError("Crello calibration requires three full train batches")

        selected = [
            (index, documents[CrelloSplit.train][start:stop])
            for index, (start, stop) in enumerate(CALIBRATION_SLICES)
        ]
    else:
        selected = [
            (index, documents[CrelloSplit.test][start:stop])
            for index, (start, stop) in enumerate(
                sequential_batches(len(documents[CrelloSplit.test]), BATCH_SIZE)
            )
        ]

    ordered = [row for split in CrelloSplit for row in documents[split]]
    embeddings = load_embedding_fixture(fixture_dir, fixture_image_ids(ordered))
    vocabularies = load_crello_vocabularies(data_dir)
    processor = CrelloProcessor(vocabularies, embeddings)
    config = CanvasVAECrelloConfig(
        vocabularies=vocabularies,
        latent_dim=LATENT_DIM,
        dropout=0.0,
        kl_weight=KL_WEIGHT,
        l2_weight=L2_WEIGHT,
    )
    dataspec, columns, vendor_model = _crello_model(tf, data_dir, dropout=0.0)
    state = convert_tensorflow_crello_variables(_vendor_variables(vendor_model), config)
    package_model = CanvasVAECrelloModel(config)
    package_model.load_state_dict(state, strict=True)
    optimizer = KerasAdam(package_model.parameters(), lr=LEARNING_RATE)

    records: list[_Comparison] = []
    static_errors = _common(vendor_model, package_model)
    initial_errors = []
    for key, package_value in package_model.state_dict().items():
        reference = state[key]
        if not torch.equal(package_value, reference):
            initial_errors.append(key)

    batch_digests = []
    for batch_index, rows in selected:
        batch = cast(CrelloBatch, processor(rows))
        batch_digests.append(_batch_digest(batch))
        tf_batch = _reference_inputs(tf, batch, columns)
        package_model.eval()
        eval_output = package_model(batch)
        eval_reference = vendor_model(tf_batch, training=False)
        _compare_outputs(
            records,
            "s1_eval_outputs",
            batch_index,
            eval_output,
            {key: value.numpy() for key, value in eval_reference.items()},
        )
        package_mean, _ = package_model.encoder(batch)
        tf_mean = vendor_model.encoder(tf_batch, training=False)
        records.append(
            _comparison(
                "s1_eval_posterior_mean",
                "z_mean",
                batch_index,
                package_mean,
                tf_mean.numpy(),
                "max_rel_to_max",
            )
        )

        noise = np.random.default_rng(batch_index).standard_normal(
            (len(rows), LATENT_DIM), dtype=np.float32
        )
        _train_step(
            tf,
            vendor_model,
            package_model,
            optimizer,
            batch,
            noise,
            batch_index,
            records,
        )

    distinct_input_batches = len(set(batch_digests)) == len(batch_digests)
    if phase == "calibrate" and not distinct_input_batches:
        static_errors.append("calibration inputs are not three distinct batches")

    output_path = report_dir / f"{phase}.json"
    report: dict[str, JSONValue] = {
        **report_context,
        "phase": phase,
        "input_selection": CALIBRATION_SELECTION
        if phase == "calibrate"
        else "all canonical test documents in sequential 1,024-document batches",
        "batch_digests": batch_digests,
        "distinct_input_batches": distinct_input_batches,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_errors,
        "measurements": [_record_payload(row) for row in records],
    }

    if phase == "calibrate":
        report["calibration_formula"] = (
            "limit=max(L,ceil2(1.5*max(three distinct train batches))); L=0"
        )
        report["limits"] = _limits(records)
        report_bytes = _write_json(output_path, report)
        if _has_shape_errors(records) or static_errors or initial_errors:
            raise AssertionError(
                f"calibration checks failed; report written to {output_path}"
            )

        limits_payload: dict[str, JSONValue] = {
            "commit": report_context["commit"],
            "calibration_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "selection": CALIBRATION_SELECTION,
            "formula": report["calibration_formula"],
            "limits": report["limits"],
        }
        limits_bytes = _write_json(report_dir / "limits.json", limits_payload)
        (report_dir / "limits.sha256").write_text(
            hashlib.sha256(limits_bytes).hexdigest() + "\n", encoding="ascii"
        )
        return report

    if frozen_bytes is None or frozen is None:
        raise RuntimeError("held-out parity did not load frozen calibration limits")

    limits = frozen["limits"]
    heldout_errors: list[dict[str, JSONValue]] = []
    for row in records:
        limit = limits.get(row.group)
        if limit is None or row.value is None or row.value > limit["limit"]:
            heldout_errors.append(
                {
                    "group": row.group,
                    "name": row.name,
                    "batch": row.batch,
                    "value": row.value,
                    "limit": None if limit is None else limit["limit"],
                    "shape_match": row.shape_match,
                }
            )

    report["frozen_limits_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
    report["heldout_errors"] = heldout_errors
    _write_json(output_path, report)
    if _has_shape_errors(records) or static_errors or initial_errors or heldout_errors:
        raise AssertionError(f"held-out checks failed; report written to {output_path}")

    return report


def main() -> None:
    """Run distinct-batch calibration or the frozen-limit held-out check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "heldout"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    report = _run(
        args.phase,
        data_dir=args.data_dir,
        fixture_dir=args.fixture_dir,
        report_dir=args.report_dir,
    )
    print(
        json.dumps({"phase": report["phase"], "report": str(args.report_dir)}, indent=1)
    )


if __name__ == "__main__":
    main()
