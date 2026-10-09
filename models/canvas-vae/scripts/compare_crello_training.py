"""Compare Crello training steps in package PyTorch and vendor TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from types import MethodType
from typing import Final, NamedTuple, Protocol, TypeAlias, TypedDict, cast
from urllib.parse import unquote, urlsplit

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
    CrelloDocument,
    CrelloProcessor,
    CrelloSplit,
    fixture_image_ids,
    load_crello_split,
    load_crello_vocabularies,
    load_embedding_fixture,
)
from canvas_vae.metrics import crello_reconstruction_scores
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
CALIBRATION_FORMULA: Final = "limit=max(L,ceil2(1.5*M)); L=0; M=max(per-metric maxima from three independent processes)"


class _Comparison(NamedTuple):
    group: str
    name: str
    batch: int
    metric: str
    value: float | None
    actual_shape: list[int]
    expected_shape: list[int]
    shape_match: bool


class _TensorFlowShape(Protocol):
    def as_list(self) -> list[int | None]: ...


class _TensorFlowTensor(Protocol):
    shape: _TensorFlowShape

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
    calibration_batch_maxima: list[float]
    max_calibration_error: float
    limit: float


class _FrozenLimits(TypedDict):
    commit: str
    calibration_sha256: str
    selection: str
    formula: str
    limits: dict[str, _FrozenLimit]


class _HubDirectory(NamedTuple):
    repo_id: str
    repo_type: str | None
    revision: str
    path: str


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


def _torch_state_sha256(state: Mapping[str, Shaped[torch.Tensor, "..."]]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())

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


def _crello_model(
    tf,
    data_dir: Path,
    *,
    dropout: float,
    metric_shape_report: list[dict[str, JSONValue]] | None = None,
):
    from traingen_parity.tensorflow_compat import (
        install_keras_preprocessing_compat,
        install_sparse_categorical_accuracy_compat,
    )

    install_keras_preprocessing_compat()
    install_sparse_categorical_accuracy_compat()
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
    model_kwargs = {
        "latent_dim": LATENT_DIM,
        "decoder_type": "oneshot",
        "num_blocks": 1,
        "block_type": "deepsvg",
        "dropout": dropout,
        "kl": KL_WEIGHT,
        "l2": L2_WEIGHT,
    }
    if metric_shape_report is None:
        model = VAE(columns, **model_kwargs)
    else:
        from canvasvae.models import metrics as vendor_metrics

        scalar_fields = [
            key
            for key, column in columns.items()
            if key != "length" and not column["is_sequence"]
        ]
        categorical_fields = [
            key
            for key, column in columns.items()
            if column["is_sequence"] and column["type"] == "categorical"
        ]
        numerical_fields = [
            key
            for key, column in columns.items()
            if column["is_sequence"] and column["type"] == "numerical"
        ]

        def tensor_shape(value: _TensorFlowTensor) -> list[int | None]:
            return value.shape.as_list()

        def record_shape(
            field: str,
            metric: str,
            y_true: _TensorFlowTensor,
            y_pred: _TensorFlowTensor,
            output: _TensorFlowTensor,
        ) -> None:
            metric_shape_report.append(
                {
                    "field": field,
                    "metric": metric,
                    "y_true_shape": cast(JSONValue, tensor_shape(y_true)),
                    "y_pred_shape": cast(JSONValue, tensor_shape(y_pred)),
                    "output_shape": cast(JSONValue, tensor_shape(output)),
                }
            )

        original_accuracy = tf.keras.metrics.sparse_categorical_accuracy
        original_bleu1 = vendor_metrics.bleu1
        original_cosine = vendor_metrics.scaled_mean_cosine_similarity

        def record_metric_shapes(
            metric: str,
            fields: Sequence[str],
            original: Callable[..., _TensorFlowTensor],
        ) -> Callable[..., _TensorFlowTensor]:
            index = 0

            def wrapped(
                y_true: _TensorFlowTensor | tuple[_TensorFlowTensor, _TensorFlowTensor],
                y_pred: _TensorFlowTensor | tuple[_TensorFlowTensor, _TensorFlowTensor],
            ) -> _TensorFlowTensor:
                nonlocal index
                field = fields[index]
                index += 1
                true_tensor = cast(
                    _TensorFlowTensor,
                    y_true[0] if isinstance(y_true, tuple) else y_true,
                )
                pred_tensor = cast(
                    _TensorFlowTensor,
                    y_pred[0] if isinstance(y_pred, tuple) else y_pred,
                )
                output = original(y_true, y_pred)
                record_shape(field, metric, true_tensor, pred_tensor, output)
                return output

            return wrapped

        tf.keras.metrics.sparse_categorical_accuracy = record_metric_shapes(
            "sparse_categorical_accuracy", scalar_fields, original_accuracy
        )
        vendor_metrics.bleu1 = record_metric_shapes(
            "bleu1", categorical_fields, original_bleu1
        )
        vendor_metrics.scaled_mean_cosine_similarity = record_metric_shapes(
            "scaled_mean_cosine_similarity", numerical_fields, original_cosine
        )
        try:
            model = VAE(columns, **model_kwargs)
        finally:
            tf.keras.metrics.sparse_categorical_accuracy = original_accuracy
            vendor_metrics.bleu1 = original_bleu1
            vendor_metrics.scaled_mean_cosine_similarity = original_cosine

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM
        )
    )
    model.optimizer.build(model.trainable_variables)
    return dataspec, columns, model


def _parse_hub_location(location: str) -> _HubDirectory:
    parsed = urlsplit(location)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("fixture Hub location must be an HTTPS Hugging Face tree URL")

    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "fixture Hub location must not contain credentials or query data"
        )

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    repo_type = None
    if parts and parts[0] in {"datasets", "spaces"}:
        repo_type = parts.pop(0)[:-1]

    if len(parts) < 5 or parts[2] != "tree":
        raise ValueError(
            "fixture Hub location must name a directory under /tree/<revision>/"
        )

    owner, repository, _, revision, *path_parts = parts
    path = PurePosixPath(*path_parts)
    if (
        not revision
        or not path_parts
        or path.is_absolute()
        or any(part in {".", ".."} or "/" in part for part in path_parts)
    ):
        raise ValueError("fixture Hub location contains an unsafe revision or path")

    return _HubDirectory(f"{owner}/{repository}", repo_type, revision, path.as_posix())


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_sha256(value: str, description: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError(
            f"{description} SHA-256 must contain 64 hexadecimal characters"
        )


def _download_embedding_fixture(
    location: str,
    expected_array_sha256: str,
    expected_manifest_sha256: str,
    fixture_dir: Path,
    token_file: Path | None,
) -> None:
    _validate_sha256(expected_array_sha256, "fixture array")
    _validate_sha256(expected_manifest_sha256, "fixture manifest")
    token = None
    try:
        hub_directory = _parse_hub_location(location)
        if token_file is not None:
            if token_file.stat().st_mode & 0o077:
                raise PermissionError("fixture token file must have mode 0600")

            token = token_file.read_text("utf-8").strip()
            if not token:
                raise ValueError("fixture token file is empty")

        from huggingface_hub import hf_hub_download

        fixture_dir.mkdir(parents=True, exist_ok=False)
        for filename in ("manifest.json", "manifest.sha256", "posterior_means.npy"):
            cached_path = hf_hub_download(
                repo_id=hub_directory.repo_id,
                filename=f"{hub_directory.path}/{filename}",
                repo_type=hub_directory.repo_type,
                revision=hub_directory.revision,
                token=token,
            )
            shutil.copyfile(cached_path, fixture_dir / filename)
    finally:
        if token_file is not None:
            token_file.unlink(missing_ok=True)

    manifest_path = fixture_dir / "manifest.json"
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256.lower() != expected_manifest_sha256.lower():
        raise ValueError(
            f"Crello fixture manifest SHA-256 mismatch: {actual_manifest_sha256}"
        )

    sidecar_sha256 = (fixture_dir / "manifest.sha256").read_text("ascii").strip()
    if sidecar_sha256.lower() != expected_manifest_sha256.lower():
        raise ValueError("Crello fixture manifest sidecar SHA-256 mismatch")

    actual_array_sha256 = _sha256_file(fixture_dir / "posterior_means.npy")
    if actual_array_sha256.lower() != expected_array_sha256.lower():
        raise ValueError(
            f"Crello fixture array SHA-256 mismatch: {actual_array_sha256}"
        )

    manifest = cast(dict[str, JSONValue], json.loads(manifest_path.read_text("utf-8")))
    if (
        manifest.get("array_file") != "posterior_means.npy"
        or manifest.get("array_sha256") != expected_array_sha256.lower()
    ):
        raise ValueError("Crello fixture manifest does not match the supplied array")


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


def _compare_metric_mappings(
    records: list[_Comparison],
    group: str,
    batch_index: int,
    package_metrics: Mapping[str, Shaped[torch.Tensor, "..."]],
    vendor_metrics: Mapping[str, Shaped[np.ndarray, "..."]],
) -> None:
    for name in sorted(package_metrics.keys() | vendor_metrics.keys()):
        metric_group = f"{group}/{name}"
        actual = package_metrics.get(name)
        expected = vendor_metrics.get(name)
        if actual is None or expected is None:
            records.append(
                _Comparison(
                    metric_group,
                    name,
                    batch_index,
                    "max_rel_to_max",
                    None,
                    list(actual.shape) if actual is not None else [],
                    list(expected.shape) if expected is not None else [],
                    False,
                )
            )
            continue

        records.append(
            _comparison(
                metric_group, name, batch_index, actual, expected, "max_rel_to_max"
            )
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
        "python_executable": sys.executable,
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
    serialized = (
        json.dumps(payload, indent=1, sort_keys=True, default=_json_default) + "\n"
    ).encode()
    path.write_bytes(serialized)
    return serialized


def _json_default(value: np.generic) -> JSONValue:
    if isinstance(value, np.generic):
        return cast(JSONValue, value.item())

    raise TypeError(f"cannot serialize {type(value).__name__} in a parity report")


def _record_payload(record: _Comparison) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], record._asdict())


def _has_shape_errors(records: Sequence[_Comparison]) -> bool:
    return any(not row.shape_match for row in records)


def _limits(records: Sequence[_Comparison]) -> dict[str, _FrozenLimit]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics: dict[str, str] = {}
    for row in records:
        if row.value is not None:
            grouped[row.group][row.batch].append(row.value)
            metrics[row.group] = row.metric

    limits: dict[str, _FrozenLimit] = {}
    for group, batches in grouped.items():
        batch_maxima = [
            max(batches.get(index, ()), default=0.0)
            for index in range(len(CALIBRATION_SLICES))
        ]
        maximum = max(batch_maxima, default=0.0)
        limits[group] = {
            "metric": metrics[group],
            "L": 0.0,
            "calibration_batch_maxima": batch_maxima,
            "max_calibration_error": maximum,
            "limit": max(0.0, _ceil_two_significant_digits(1.5 * maximum)),
        }

    return limits


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


def _run_s0(
    *, data_dir: Path = DATA_DIR, report_dir: Path = REPORT_DIR
) -> dict[str, JSONValue]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Crello model parity is CPU-only")

    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    torch.manual_seed(0)
    report_context = _cpu_report_context()
    vocabularies = load_crello_vocabularies(data_dir)
    config = CanvasVAECrelloConfig(
        vocabularies=vocabularies,
        latent_dim=LATENT_DIM,
        dropout=0.0,
        kl_weight=KL_WEIGHT,
        l2_weight=L2_WEIGHT,
    )
    vendor_metric_shapes: list[dict[str, JSONValue]] = []
    dataspec, columns, vendor_model = _crello_model(
        tf, data_dir, dropout=0.0, metric_shape_report=vendor_metric_shapes
    )
    package_model = CanvasVAECrelloModel(config)
    errors = _common(vendor_model, package_model)
    expected_metric_fields = set(columns) - {"length"}
    observed_metric_fields = {
        cast(str, entry["field"]) for entry in vendor_metric_shapes if "field" in entry
    }
    if (
        len(vendor_metric_shapes) != len(expected_metric_fields)
        or observed_metric_fields != expected_metric_fields
    ):
        errors.append(
            "vendor metric shape report does not cover each field exactly once"
        )

    vocabulary_matches: dict[str, bool] = {}
    for field, expected in vocabularies.items():
        layer = dataspec.preprocessor.get(field)
        if layer is None:
            vocabulary_matches[field] = False
        else:
            actual = [
                value.item() if isinstance(value, np.generic) else value
                for value in layer.get_vocabulary()
            ]
            vocabulary_matches[field] = actual == expected
        if not vocabulary_matches[field]:
            errors.append(f"vendor vocabulary differs for {field}")

    field_report: dict[str, JSONValue] = {}
    for field, column in columns.items():
        input_dim = column.get("input_dim")
        field_report[field] = {
            "type": column["type"],
            "shape": list(column["shape"]),
            "is_sequence": column["is_sequence"],
            "input_dim": input_dim,
        }

    for field, size in config.context_field_sizes.items():
        column = columns.get(field)
        if (
            column is None
            or column["type"] != "categorical"
            or column["input_dim"] != size
        ):
            errors.append(f"context input dimension differs for {field}")
        if package_model.encoder.context_embeddings[field].num_embeddings != size:
            errors.append(f"context embedding dimension differs for {field}")

        head = (
            package_model.decoder.length_head
            if field == "length"
            else package_model.decoder.context_heads[field]
        )
        if head.out_features != size:
            errors.append(f"context output dimension differs for {field}")

    for field, size in config.sequence_field_sizes.items():
        column = columns.get(field)
        if (
            column is None
            or column["type"] != "categorical"
            or column["input_dim"] != size
        ):
            errors.append(f"sequence input dimension differs for {field}")
        if column is not None and (
            not column["is_sequence"]
            or column["shape"] != (3 if field == "color" else 1,)
        ):
            errors.append(f"sequence input shape differs for {field}")

        embedding = package_model.encoder.sequence_embeddings[f"field_{field}"]
        if embedding.num_embeddings != size:
            errors.append(f"sequence embedding dimension differs for {field}")

        head = package_model.decoder.sequence_heads[f"field_{field}"]
        expected_output_dim = size * (3 if field == "color" else 1)
        if head.out_features != expected_output_dim:
            errors.append(f"sequence output dimension differs for {field}")

    embedding_column = columns.get("image_embedding")
    projection = package_model.encoder.numerical_projections["image_embedding"]
    numerical_head = package_model.decoder.numerical_heads["image_embedding"]
    if (
        embedding_column is None
        or embedding_column["type"] != "numerical"
        or embedding_column["shape"] != (256,)
        or not embedding_column["is_sequence"]
        or projection.in_features != 256
        or projection.out_features != LATENT_DIM
        or numerical_head.in_features != LATENT_DIM
        or numerical_head.out_features != 256
    ):
        errors.append("image_embedding dimensions differ from the vendor schema")

    conditional_masks: dict[str, bool] = {}
    for field in ("color", "image_embedding"):
        condition = columns[field].get("loss_condition", {})
        expected_ids = config.conditional_type_ids(field)
        expected_mask = tuple(
            index in expected_ids for index in range(len(vocabularies["type"]))
        )
        actual_mask = tuple(bool(value) for value in condition.get("mask", ()))
        matches = condition.get("key") == "type" and actual_mask == expected_mask
        conditional_masks[field] = matches
        if not matches:
            errors.append(f"type-conditioned loss mask differs for {field}")

    mapping = tensorflow_crello_key_map(config)
    variables = _vendor_variables(vendor_model)
    initialization_error = None
    initial_mismatch_keys: list[str] = []
    state_digest = None
    try:
        state = convert_tensorflow_crello_variables(variables, config)
        package_model.load_state_dict(state, strict=True)
        initial_mismatch_keys = [
            key
            for key, value in package_model.state_dict().items()
            if not torch.equal(value, state[key])
        ]
        if initial_mismatch_keys:
            errors.append("package initial state differs after TensorFlow mapping")
        state_digest = _torch_state_sha256(package_model.state_dict())
    except (KeyError, RuntimeError) as error:
        initialization_error = str(error)
        errors.append("vendor initialization could not be mapped into the package")

    vendor_optimizer = vendor_model.optimizer
    vendor_optimizer_config = vendor_optimizer.get_config()
    package_optimizer = KerasAdam(package_model.parameters(), lr=LEARNING_RATE)
    package_optimizer_config = package_optimizer.param_groups[0]
    optimizer_matches = (
        math.isclose(
            float(vendor_optimizer.learning_rate.numpy()),
            float(package_optimizer_config["lr"]),
            rel_tol=1e-6,
        )
        and float(vendor_optimizer.beta_1) == package_optimizer_config["betas"][0]
        and float(vendor_optimizer.beta_2) == package_optimizer_config["betas"][1]
        and float(vendor_optimizer.epsilon) == package_optimizer_config["eps"]
        and float(vendor_optimizer.clipnorm) == CLIP_NORM
    )
    if not optimizer_matches:
        errors.append("vendor and package optimizer defaults differ")

    report: dict[str, JSONValue] = {
        **report_context,
        "phase": "s0",
        "metadata": {
            "count_json_sha256": _sha256_file(data_dir / "count.json"),
            "vocabulary_json_sha256": _sha256_file(data_dir / "vocabulary.json"),
            "vocabulary_sizes": {
                field: len(values) for field, values in vocabularies.items()
            },
            "vocabulary_matches": vocabulary_matches,
        },
        "vendor_fields": field_report,
        "vendor_metric_shapes_before_concat": vendor_metric_shapes,
        "package_field_sizes": {
            "context": config.context_field_sizes,
            "sequence": config.sequence_field_sizes,
        },
        "conditional_masks_match": conditional_masks,
        "parameter_mapping": {
            "mapped_keys": len(mapping),
            "package_state_keys": len(package_model.state_dict()),
            "vendor_weight_keys": len(variables),
            "initial_state_sha256": state_digest,
            "initial_state_mismatch_keys": initial_mismatch_keys,
            "initialization_error": initialization_error,
        },
        "parameter_counts": {
            "vendor_trainable": sum(
                int(np.prod(variable.shape))
                for variable in vendor_model.trainable_variables
            ),
            "package_trainable": sum(
                parameter.numel() for parameter in package_model.parameters()
            ),
            "vendor_tensors": len(vendor_model.trainable_variables),
            "package_tensors": len(tuple(package_model.parameters())),
        },
        "optimizer": {
            "vendor": vendor_optimizer_config,
            "package": {
                "learning_rate": package_optimizer_config["lr"],
                "betas": list(package_optimizer_config["betas"]),
                "epsilon": package_optimizer_config["eps"],
                "clip_norm": CLIP_NORM,
            },
            "defaults_match": optimizer_matches,
            "scheduler": None,
        },
        "static_errors": errors,
    }
    report_path = report_dir / "s0.json"
    _write_json(report_path, report)
    if errors:
        raise AssertionError(f"S0 checks failed; report written to {report_path}")

    return report


def _load_frozen_limits(report_dir: Path, commit: str) -> tuple[bytes, _FrozenLimits]:
    limits_bytes = (report_dir / "limits.json").read_bytes()
    expected_sha256 = (report_dir / "limits.sha256").read_text("ascii").strip()
    if hashlib.sha256(limits_bytes).hexdigest() != expected_sha256:
        raise ValueError("frozen Crello parity limits SHA-256 mismatch")

    limits = cast(_FrozenLimits, json.loads(limits_bytes))
    if limits["commit"] != commit:
        raise ValueError("held-out run must use the exact calibration commit")

    return limits_bytes, limits


def _heldout_test_batches(
    documents: Sequence[CrelloDocument],
) -> list[tuple[int, list[CrelloDocument]]]:
    return [
        (batch_index, [documents[index] for index in indices])
        for batch_index, indices in enumerate(
            sequential_batches(len(documents), BATCH_SIZE)
        )
    ]


def _calibrate(
    *,
    data_dir: Path = DATA_DIR,
    fixture_dir: Path = FIXTURE_DIR,
    report_dir: Path = REPORT_DIR,
    fixture_array_sha256: str | None = None,
    fixture_manifest_sha256: str | None = None,
) -> dict[str, JSONValue]:
    report_context = _cpu_report_context()
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "calibrate.json"
    plan_path = report_dir / "calibration-plan.json"
    for path in (
        plan_path,
        output_path,
        report_dir / "limits.json",
        report_dir / "limits.sha256",
        *(report_dir / f"calibration-batch-{index}.json" for index in range(3)),
    ):
        path.unlink(missing_ok=True)

    plan = {
        **report_context,
        "phase": "calibration_plan",
        "input_selection": CALIBRATION_SELECTION,
        "batches": [
            {"batch_index": index, "split": "train", "start": start, "stop": stop}
            for index, (start, stop) in enumerate(CALIBRATION_SLICES)
        ],
        "formula": CALIBRATION_FORMULA,
    }
    plan_bytes = _write_json(plan_path, plan)

    records: list[_Comparison] = []
    static_errors: list[str] = []
    process_reports: list[dict[str, JSONValue]] = []
    batch_digests: list[str] = []
    initial_state_hashes: list[str] = []
    initial_mismatch_keys: list[str] = []
    script = Path(__file__).resolve()
    for batch_index in range(len(CALIBRATION_SLICES)):
        child_report_path = report_dir / f"calibration-batch-{batch_index}.json"
        command = [
            sys.executable,
            str(script),
            "calibration-batch",
            "--batch-index",
            str(batch_index),
            "--data-dir",
            str(data_dir),
            "--fixture-dir",
            str(fixture_dir),
            "--report-dir",
            str(report_dir),
        ]
        if fixture_array_sha256 is not None:
            command.extend(("--fixture-array-sha256", fixture_array_sha256))
        if fixture_manifest_sha256 is not None:
            command.extend(("--fixture-manifest-sha256", fixture_manifest_sha256))

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        process_report: dict[str, JSONValue] = {
            "batch_index": batch_index,
            "returncode": completed.returncode,
            "report_file": child_report_path.name,
            "report_sha256": None,
        }
        if not child_report_path.exists():
            static_errors.append(
                f"calibration process {batch_index} did not write its report"
            )
            process_reports.append(process_report)
            continue

        child_bytes = child_report_path.read_bytes()
        child = cast(dict[str, JSONValue], json.loads(child_bytes))
        process_report["report_sha256"] = hashlib.sha256(child_bytes).hexdigest()
        process_reports.append(process_report)
        if completed.returncode != 0:
            static_errors.append(
                f"calibration process {batch_index} exited with {completed.returncode}"
            )

        child_errors = cast(list[str], child["static_errors"])
        static_errors.extend(child_errors)
        child_digests = cast(list[str], child["batch_digests"])
        if len(child_digests) != 1:
            static_errors.append(
                f"calibration process {batch_index} did not use exactly one batch"
            )
        batch_digests.extend(child_digests)
        initial_state_hashes.append(cast(str, child["initial_state_sha256"]))
        initial_mismatch_keys.extend(
            cast(list[str], child["initial_state_mismatch_keys"])
        )
        for measurement in cast(list[dict[str, JSONValue]], child["measurements"]):
            records.append(
                _Comparison(
                    group=cast(str, measurement["group"]),
                    name=cast(str, measurement["name"]),
                    batch=cast(int, measurement["batch"]),
                    metric=cast(str, measurement["metric"]),
                    value=cast(float | None, measurement["value"]),
                    actual_shape=cast(list[int], measurement["actual_shape"]),
                    expected_shape=cast(list[int], measurement["expected_shape"]),
                    shape_match=cast(bool, measurement["shape_match"]),
                )
            )

    distinct_input_batches = len(batch_digests) == len(CALIBRATION_SLICES) and len(
        set(batch_digests)
    ) == len(CALIBRATION_SLICES)
    if not distinct_input_batches:
        static_errors.append("calibration inputs are not three distinct batches")

    if (
        len(initial_state_hashes) != len(CALIBRATION_SLICES)
        or len(set(initial_state_hashes)) != 1
    ):
        static_errors.append(
            "calibration processes did not share the prescribed initial state"
        )

    groups_per_batch = {
        index: {
            row.group for row in records if row.batch == index and row.value is not None
        }
        for index in range(len(CALIBRATION_SLICES))
    }
    all_groups = set().union(*groups_per_batch.values())
    if not all_groups:
        static_errors.append("calibration produced no metric measurements")
    for group in all_groups:
        if any(group not in groups_per_batch[index] for index in groups_per_batch):
            static_errors.append(
                f"calibration metric group {group} is missing from a batch"
            )

    report: dict[str, JSONValue] = {
        **report_context,
        "phase": "calibrate",
        "calibration_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "fixture_array_sha256": fixture_array_sha256,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "input_selection": CALIBRATION_SELECTION,
        "calibration_processes": process_reports,
        "batch_digests": batch_digests,
        "distinct_input_batches": distinct_input_batches,
        "initial_state_sha256": initial_state_hashes[0]
        if initial_state_hashes
        else None,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_mismatch_keys,
        "measurements": [_record_payload(row) for row in records],
        "calibration_formula": CALIBRATION_FORMULA,
        "limits": _limits(records),
    }
    report_bytes = _write_json(output_path, report)
    if (
        static_errors
        or initial_mismatch_keys
        or _has_shape_errors(records)
        or len(process_reports) != len(CALIBRATION_SLICES)
    ):
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


def _run(
    phase: str,
    *,
    data_dir: Path = DATA_DIR,
    fixture_dir: Path = FIXTURE_DIR,
    report_dir: Path = REPORT_DIR,
    fixture_array_sha256: str | None = None,
    fixture_manifest_sha256: str | None = None,
    batch_index: int | None = None,
) -> dict[str, JSONValue]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Crello model parity is CPU-only")

    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    report_context = _cpu_report_context()
    frozen_bytes = None
    frozen = None
    if phase == "heldout":
        frozen_bytes, frozen = _load_frozen_limits(
            report_dir, str(report_context["commit"])
        )
    elif phase != "calibration-batch" or batch_index not in range(
        len(CALIBRATION_SLICES)
    ):
        raise ValueError("Crello parity phase must be heldout or one calibration batch")

    documents = {split: load_crello_split(data_dir, split) for split in CrelloSplit}
    if phase == "calibration-batch":
        if batch_index is None:
            raise ValueError("a calibration process needs a batch index")

        start, stop = CALIBRATION_SLICES[batch_index]
        if len(documents[CrelloSplit.train]) < stop:
            raise ValueError("Crello calibration requires three full train batches")

        selected = [(batch_index, documents[CrelloSplit.train][start:stop])]
    else:
        selected = _heldout_test_batches(documents[CrelloSplit.test])

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
    initial_state_hash = _torch_state_sha256(package_model.state_dict())
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
        vendor_reconstruction_metrics = vendor_model.vector_metric(
            (tf_batch, eval_reference), training=False
        )
        package_metrics = crello_reconstruction_scores(batch, eval_output, config)
        _compare_metric_mappings(
            records,
            "s1_reconstruction_metrics",
            batch_index,
            {
                key: value
                for key, value in package_metrics.items()
                if key not in {"layout_acc", "layout_miou"}
            },
            {
                key: value.numpy()
                for key, value in vendor_reconstruction_metrics.items()
            },
        )
        vendor_layout_metrics = vendor_model.layout_metric(
            (tf_batch, eval_reference), training=False
        )
        _compare_metric_mappings(
            records,
            "s1_layout_metrics",
            batch_index,
            {key: package_metrics[key] for key in ("layout_acc", "layout_miou")},
            {key: value.numpy() for key, value in vendor_layout_metrics.items()},
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

    report_phase = "calibration_batch" if phase == "calibration-batch" else phase
    output_name = (
        f"calibration-batch-{batch_index}.json"
        if phase == "calibration-batch"
        else "heldout.json"
    )
    output_path = report_dir / output_name
    report: dict[str, JSONValue] = {
        **report_context,
        "phase": report_phase,
        "fixture_array_sha256": fixture_array_sha256,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "batch_index": batch_index,
        "initial_state_sha256": initial_state_hash,
        "input_selection": (
            f"canonical train.jsonl indices {CALIBRATION_SLICES[batch_index][0]}:{CALIBRATION_SLICES[batch_index][1]}"
            if phase == "calibration-batch" and batch_index is not None
            else "all canonical test documents in sequential 1,024-document batches"
        ),
        "batch_digests": batch_digests,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_errors,
        "measurements": [_record_payload(row) for row in records],
    }

    if phase == "calibration-batch":
        _write_json(output_path, report)
        if _has_shape_errors(records) or static_errors or initial_errors:
            raise AssertionError(
                f"calibration batch checks failed; report written to {output_path}"
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
    """Run fixture-independent configuration checks or the calibrated model checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("s0", "calibrate", "calibration-batch", "heldout", "run")
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--fixture-hub-location")
    parser.add_argument("--fixture-array-sha256")
    parser.add_argument("--fixture-manifest-sha256")
    parser.add_argument("--fixture-token-file", type=Path)
    parser.add_argument("--batch-index", type=int)
    args = parser.parse_args()
    if args.phase == "s0":
        report = _run_s0(data_dir=args.data_dir, report_dir=args.report_dir)
        print(
            json.dumps(
                {"phase": report["phase"], "report": str(args.report_dir)}, indent=1
            )
        )
        return

    if args.phase == "run":
        if (
            not args.fixture_hub_location
            or not args.fixture_array_sha256
            or not args.fixture_manifest_sha256
        ):
            parser.error(
                "run requires --fixture-hub-location, --fixture-array-sha256, "
                "and --fixture-manifest-sha256"
            )

        token_file = args.fixture_token_file
        if token_file is None and os.environ.get("CANVAS_VAE_FIXTURE_TOKEN_FILE"):
            token_file = Path(os.environ["CANVAS_VAE_FIXTURE_TOKEN_FILE"])

        with tempfile.TemporaryDirectory(
            prefix="canvas-vae-crello-parity-"
        ) as temporary:
            temporary_dir = Path(temporary)
            fixture_dir = temporary_dir / "fixture"
            _download_embedding_fixture(
                args.fixture_hub_location,
                args.fixture_array_sha256,
                args.fixture_manifest_sha256,
                fixture_dir,
                token_file,
            )
            calibration = _calibrate(
                data_dir=args.data_dir,
                fixture_dir=fixture_dir,
                report_dir=args.report_dir,
                fixture_array_sha256=args.fixture_array_sha256.lower(),
                fixture_manifest_sha256=args.fixture_manifest_sha256.lower(),
            )
            heldout = _run(
                "heldout",
                data_dir=args.data_dir,
                fixture_dir=fixture_dir,
                report_dir=args.report_dir,
                fixture_array_sha256=args.fixture_array_sha256.lower(),
                fixture_manifest_sha256=args.fixture_manifest_sha256.lower(),
            )
        print(
            json.dumps(
                {
                    "phase": "run",
                    "calibration": calibration["phase"],
                    "heldout": heldout["phase"],
                    "report": str(args.report_dir),
                },
                indent=1,
            )
        )
        return

    if args.phase == "calibrate":
        report = _calibrate(
            data_dir=args.data_dir,
            fixture_dir=args.fixture_dir,
            report_dir=args.report_dir,
            fixture_array_sha256=args.fixture_array_sha256,
            fixture_manifest_sha256=args.fixture_manifest_sha256,
        )
    elif args.phase == "calibration-batch":
        if args.batch_index not in range(len(CALIBRATION_SLICES)):
            parser.error("calibration-batch requires --batch-index 0, 1, or 2")

        report = _run(
            "calibration-batch",
            data_dir=args.data_dir,
            fixture_dir=args.fixture_dir,
            report_dir=args.report_dir,
            fixture_array_sha256=args.fixture_array_sha256,
            fixture_manifest_sha256=args.fixture_manifest_sha256,
            batch_index=args.batch_index,
        )
    else:
        report = _run(
            args.phase,
            data_dir=args.data_dir,
            fixture_dir=args.fixture_dir,
            report_dir=args.report_dir,
            fixture_array_sha256=args.fixture_array_sha256,
            fixture_manifest_sha256=args.fixture_manifest_sha256,
        )

    print(
        json.dumps({"phase": report["phase"], "report": str(args.report_dir)}, indent=1)
    )


if __name__ == "__main__":
    main()
