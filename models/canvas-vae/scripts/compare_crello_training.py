"""Compare Crello training steps in package PyTorch and vendor TensorFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
from canvas_vae.metrics import (
    _allowed_type_mask,
    crello_reconstruction_scores,
)
from canvas_vae.modeling_canvas_vae import (
    CanvasVAECrelloModel,
    CanvasVAECrelloModelOutput,
)
from canvas_vae.training.optim import KerasAdam, clip_gradients_by_norm, l2_penalty
from canvas_vae.training.parity import (
    KERAS_ADAM_EPSILON,
    S2_ADAM_RULE_LIMIT,
    WELL_CONDITIONED_SQRT_V,
    check_keras_adam_update,
    check_zero_gradient,
    split_attention_key_biases,
)
from canvas_vae.training.sampling import sequential_batches

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
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
METRIC_SCORE_FLOOR: Final = 2 * 2**-24
ZERO_GRADIENT_FLOOR: Final = 1e-6
CALIBRATION_FORMULA: Final = (
    "limit=max(L,ceil2(1.5*M)); L=2*2^-24 for reconstruction/layout scores, "
    "L=1e-6 for attention key-bias max-absolute gradients, L=0 for well-conditioned "
    "Adam updates and other metrics; "
    "M=max(per-metric maxima from three independent processes)"
)
S2_DIAGNOSTIC_GROUPS: Final = (
    "s2_gradients",
    "s2_clipped_gradients",
    "s2_first_moments",
    "s2_second_moments",
    "s2_updated_parameters",
    "s2_zero_gradients",
)
S3_DIAGNOSTIC_GROUPS: Final = (
    "s3_batch_norm_mean",
    "s3_batch_norm_variance",
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


class _TensorFlowShape(Protocol):
    def as_list(self) -> list[int | None]: ...


class _TensorFlowTensor(Protocol):
    shape: _TensorFlowShape

    def numpy(self) -> Shaped[np.ndarray, "..."]: ...


class _TensorFlowModule(Protocol):
    def convert_to_tensor(
        self, value: Shaped[np.ndarray, "..."]
    ) -> _TensorFlowTensor: ...


class _VendorColumn(TypedDict, total=False):
    type: str
    is_sequence: bool


class _VendorVectorMetric(Protocol):
    def _get_masks(
        self,
        y_true: Mapping[str, _TensorFlowTensor],
        y_pred: Mapping[str, _TensorFlowTensor],
        training: bool,
    ) -> tuple[_TensorFlowTensor, _TensorFlowTensor]: ...

    def _conditional_mask(
        self,
        column: _VendorColumn,
        y_true: Mapping[str, _TensorFlowTensor],
        y_pred: Mapping[str, _TensorFlowTensor],
        mask_true: _TensorFlowTensor,
        mask_pred: _TensorFlowTensor,
    ) -> tuple[_TensorFlowTensor, _TensorFlowTensor]: ...


class _VendorModelMetrics(Protocol):
    vector_metric: _VendorVectorMetric


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


def _process_output_excerpt(output: str | None) -> str:
    if not output:
        return ""

    redacted = re.sub(r"hf_[A-Za-z0-9_-]+", "[redacted token]", output)
    return redacted[-8192:]


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
    vendor_path = vendor_root("canvas-vae", marker="src/canvas-vae/canvasvae/train.py")
    sys.path.insert(0, str(vendor_path / "src" / "canvas-vae"))
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


def _as_numpy(
    value: Shaped[torch.Tensor, "..."] | Shaped[np.ndarray, "..."] | _TensorFlowTensor,
) -> Shaped[np.ndarray, "..."]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return cast(_TensorFlowTensor, value).numpy()


def _float32_ulp_delta(
    actual: Shaped[np.ndarray, "..."], expected: Shaped[np.ndarray, "..."]
) -> Shaped[np.ndarray, "..."]:
    actual32 = np.ascontiguousarray(actual, dtype=np.float32)
    expected32 = np.ascontiguousarray(expected, dtype=np.float32)
    if np.any(actual32 < 0) or np.any(expected32 < 0):
        raise ValueError("reconstruction scores must be non-negative for ULP distance")

    actual_bits = actual32.view(np.uint32).astype(np.int64)
    expected_bits = expected32.view(np.uint32).astype(np.int64)
    return actual_bits - expected_bits


def _normalized_category_ids(
    target_ids: Shaped[np.ndarray, "..."], predicted_ids: Shaped[np.ndarray, "..."]
) -> tuple[Shaped[np.ndarray, "..."], Shaped[np.ndarray, "..."]]:
    if target_ids.ndim == 3 and target_ids.shape[-1] == 1:
        target_ids = target_ids[..., 0]

    if predicted_ids.ndim == 3 and predicted_ids.shape[-1] == 1:
        predicted_ids = predicted_ids[..., 0]

    if target_ids.ndim == 2:
        target_ids = target_ids[..., np.newaxis]

    if predicted_ids.ndim == 2:
        predicted_ids = predicted_ids[..., np.newaxis]

    if (
        target_ids.ndim != 3
        or predicted_ids.ndim != 3
        or target_ids.shape[0] != predicted_ids.shape[0]
        or target_ids.shape[2] != predicted_ids.shape[2]
    ):
        raise ValueError(
            "categorical diagnostic ids must share batch and channel dimensions; "
            f"got {target_ids.shape} and {predicted_ids.shape}"
        )

    return target_ids.astype(np.int64), predicted_ids.astype(np.int64)


def _bleu_diagnostic(
    target_ids: Shaped[np.ndarray, "..."],
    target_mask: Shaped[np.ndarray, "..."],
    predicted_ids: Shaped[np.ndarray, "..."],
    predicted_mask: Shaped[np.ndarray, "..."],
    num_classes: int,
) -> dict[str, JSONValue]:
    target_ids, predicted_ids = _normalized_category_ids(target_ids, predicted_ids)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    predicted_mask = np.asarray(predicted_mask, dtype=np.bool_)
    if target_ids.shape[0] != predicted_ids.shape[0]:
        raise ValueError("BLEU diagnostic batch sizes differ")

    if target_ids.shape[1] != target_mask.shape[1]:
        raise ValueError("BLEU target ids and mask lengths differ")

    if predicted_ids.shape[1] != predicted_mask.shape[1]:
        raise ValueError("BLEU prediction ids and mask lengths differ")

    batch_size, _, channels = target_ids.shape
    matches = np.zeros((batch_size, channels), dtype=np.int64)
    target_count = target_mask.sum(axis=1, dtype=np.int64)
    predicted_count = predicted_mask.sum(axis=1, dtype=np.int64)
    for row in range(batch_size):
        for channel in range(channels):
            target_bow = np.bincount(
                target_ids[row, target_mask[row], channel], minlength=num_classes
            )
            predicted_bow = np.bincount(
                predicted_ids[row, predicted_mask[row], channel], minlength=num_classes
            )
            matches[row, channel] = np.minimum(target_bow, predicted_bow).sum()

    def score_dtype(
        float_type: type[np.float32] | type[np.float64],
    ) -> Shaped[np.ndarray, "batch channels"]:
        target_length = target_count.astype(float_type) + float_type(1e-9)
        predicted_length = predicted_count.astype(float_type) + float_type(1e-9)
        precision = matches.astype(float_type) / predicted_length[:, np.newaxis]
        ratio = (
            float_type(1.0)
            - target_length[:, np.newaxis] / predicted_length[:, np.newaxis]
        )
        brevity_exponent = np.minimum(float_type(0.0), ratio)
        score = np.exp(brevity_exponent) * np.sqrt(precision)
        return np.clip(score, 0.0, 1.0)

    target_length32 = target_count.astype(np.float32) + np.float32(1e-9)
    predicted_length32 = predicted_count.astype(np.float32) + np.float32(1e-9)
    precision32 = matches.astype(np.float32) / predicted_length32[:, np.newaxis]
    ratio32 = (
        np.float32(1.0)
        - target_length32[:, np.newaxis] / predicted_length32[:, np.newaxis]
    )
    exponent32 = np.minimum(np.float32(0.0), ratio32)
    brevity32 = np.exp(exponent32)
    root32 = np.sqrt(precision32)
    raw32 = brevity32 * root32
    return {
        "float32_intermediates": {
            "target_token_count": np.repeat(
                target_count[:, None], channels, axis=1
            ).tolist(),
            "predicted_token_count": np.repeat(
                predicted_count[:, None], channels, axis=1
            ).tolist(),
            "matching_token_count": matches.tolist(),
            "target_length_with_epsilon": np.repeat(
                target_length32[:, None], channels, axis=1
            ).tolist(),
            "predicted_length_with_epsilon": np.repeat(
                predicted_length32[:, None], channels, axis=1
            ).tolist(),
            "precision": precision32.tolist(),
            "length_ratio": ratio32.tolist(),
            "brevity_exponent": exponent32.tolist(),
            "brevity_factor": brevity32.tolist(),
            "sqrt_precision": root32.tolist(),
            "unclipped_score": raw32.tolist(),
            "clipped_score": np.clip(raw32, 0.0, 1.0).tolist(),
        },
        "float64_scores": score_dtype(np.float64).tolist(),
    }


def _scaled_cosine_diagnostic(
    target: Shaped[np.ndarray, "batch target_elements features"],
    target_mask: Shaped[np.ndarray, "batch target_elements"],
    prediction: Shaped[np.ndarray, "batch predicted_elements features"],
    prediction_mask: Shaped[np.ndarray, "batch predicted_elements"],
) -> dict[str, JSONValue]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    prediction_mask = np.asarray(prediction_mask, dtype=np.bool_)
    if (
        target.ndim != 3
        or prediction.ndim != 3
        or target.shape[0] != prediction.shape[0]
        or target.shape[2] != prediction.shape[2]
        or target.shape[:2] != target_mask.shape
        or prediction.shape[:2] != prediction_mask.shape
    ):
        raise ValueError("cosine diagnostic inputs have incompatible shapes")

    target_count = target_mask.sum(axis=1, dtype=np.float64)
    prediction_count = prediction_mask.sum(axis=1, dtype=np.float64)
    target_length = target_count + 1e-9
    prediction_length = prediction_count + 1e-9
    target_mean = (
        np.sum(target * target_mask[:, :, np.newaxis], axis=1)
        / target_length[:, np.newaxis]
    )
    prediction_mean = (
        np.sum(prediction * prediction_mask[:, :, np.newaxis], axis=1)
        / prediction_length[:, np.newaxis]
    )
    target_norm_squared = np.sum(np.square(target_mean), axis=1)
    prediction_norm_squared = np.sum(np.square(prediction_mean), axis=1)
    target_norm = np.sqrt(target_norm_squared)
    prediction_norm = np.sqrt(prediction_norm_squared)
    target_normalized = (
        target_mean / np.sqrt(np.maximum(target_norm_squared, 1e-12))[:, np.newaxis]
    )
    prediction_normalized = (
        prediction_mean
        / np.sqrt(np.maximum(prediction_norm_squared, 1e-12))[:, np.newaxis]
    )
    tensorflow_cosine = np.sum(target_normalized * prediction_normalized, axis=1)
    torch_normalized_target = target_mean / np.maximum(target_norm, 1e-6)[:, np.newaxis]
    torch_normalized_prediction = (
        prediction_mean / np.maximum(prediction_norm, 1e-6)[:, np.newaxis]
    )
    torch_cosine = np.sum(torch_normalized_target * torch_normalized_prediction, axis=1)
    length_penalty = np.exp(np.minimum(0.0, 1.0 - target_length / prediction_length))
    tensorflow_similarity = (1.0 + tensorflow_cosine) / 2.0
    torch_similarity = (1.0 + torch_cosine) / 2.0
    return {
        "target_count": target_count.tolist(),
        "prediction_count": prediction_count.tolist(),
        "target_mean_l2_norm": target_norm.tolist(),
        "prediction_mean_l2_norm": prediction_norm.tolist(),
        "tensorflow_cosine_similarity": tensorflow_cosine.tolist(),
        "torch_cosine_similarity": torch_cosine.tolist(),
        "length_penalty": length_penalty.tolist(),
        "tensorflow_score": np.clip(
            length_penalty * tensorflow_similarity, 0.0, 1.0
        ).tolist(),
        "torch_score": np.clip(length_penalty * torch_similarity, 0.0, 1.0).tolist(),
    }


def _array_delta_summary(
    actual: Shaped[np.ndarray, "..."], expected: Shaped[np.ndarray, "..."]
) -> dict[str, JSONValue]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {
            "equal": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }

    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "equal": bool(np.array_equal(actual, expected)),
        "shape": list(actual.shape),
        "different_values": int(np.count_nonzero(delta)),
        "max_absolute_difference": float(np.max(delta, initial=0.0)),
        "max_relative_difference": float(
            np.max(
                delta / np.maximum(np.abs(expected.astype(np.float64)), 1e-30),
                initial=0.0,
            )
        ),
    }


def _metric_diagnostic_report(
    batch: CrelloBatch,
    package_output: CanvasVAECrelloModelOutput,
    vendor_inputs: Mapping[str, _TensorFlowTensor],
    vendor_output: Mapping[str, _TensorFlowTensor],
    columns: Mapping[str, _VendorColumn],
    package_metrics: Mapping[str, Shaped[torch.Tensor, "..."]],
    vendor_metrics: Mapping[str, _TensorFlowTensor],
    config: CanvasVAECrelloConfig,
    vendor_model: _VendorModelMetrics,
) -> dict[str, JSONValue]:
    if package_output.context_logits is None or package_output.sequence_logits is None:
        raise ValueError("metric diagnosis requires decoded categorical logits")

    if package_output.mask is None or package_output.numerical_predictions is None:
        raise ValueError("metric diagnosis requires decoded masks and numeric outputs")

    vendor_metric = vendor_model.vector_metric
    vendor_true_mask, vendor_pred_mask = vendor_metric._get_masks(
        vendor_inputs, vendor_output, training=False
    )
    package_type = package_output.sequence_logits["type"].argmax(dim=-1)
    package_fields = cast(Mapping[str, Shaped[torch.Tensor, "..."]], batch)
    field_reports: dict[str, JSONValue] = {}

    for field in sorted(package_metrics.keys() & vendor_metrics.keys()):
        if field in {"total", "layout_acc", "layout_miou"}:
            continue

        column = columns[field]
        is_sequence = column["is_sequence"]
        package_target = _as_numpy(package_fields[field])
        vendor_target = _as_numpy(vendor_inputs[field])
        package_true: Shaped[np.ndarray, "..."] | None = None
        vendor_true: Shaped[np.ndarray, "..."] | None = None
        package_pred: Shaped[np.ndarray, "..."] | None = None
        vendor_pred: Shaped[np.ndarray, "..."] | None = None
        package_mask_equal: bool | None = None
        if is_sequence:
            package_true_mask = package_fields["element_mask"]
            package_pred_mask = package_output.mask
            if field in config.conditional_types:
                package_true_mask = package_fields[f"{field}_mask"]
                package_pred_mask = package_pred_mask & _allowed_type_mask(
                    package_type, config.conditional_type_ids(field)
                )

            vendor_true_tensor, vendor_pred_tensor = vendor_metric._conditional_mask(
                column,
                vendor_inputs,
                vendor_output,
                vendor_true_mask,
                vendor_pred_mask,
            )
            package_true = _as_numpy(package_true_mask)
            package_pred = _as_numpy(package_pred_mask)
            vendor_true = _as_numpy(vendor_true_tensor)
            vendor_pred = _as_numpy(vendor_pred_tensor)
            package_mask_equal = np.array_equal(
                package_true, vendor_true
            ) and np.array_equal(package_pred, vendor_pred)

        is_categorical = column["type"] == "categorical"
        package_bleu: dict[str, JSONValue] | None = None
        vendor_bleu: dict[str, JSONValue] | None = None
        package_cosine: dict[str, JSONValue] | None = None
        vendor_cosine: dict[str, JSONValue] | None = None
        numeric_input_differences: dict[str, JSONValue] | None = None
        same_input_formula_scores_equal: dict[str, bool] | None = None
        metric_inputs_equal: bool | None = None
        float64_equal: bool | None = None
        if is_sequence and is_categorical:
            package_logits = package_output.sequence_logits[field]
            vendor_logits = vendor_output[field]
            package_ids = package_logits.argmax(dim=-1).detach().cpu().numpy()
            vendor_ids = np.asarray(vendor_logits.numpy()).argmax(axis=-1)
            vendor_depth = vendor_logits.shape.as_list()[-1]
            if vendor_depth is None:
                raise ValueError(f"vendor logits for {field!r} have no class dimension")

            package_ids, package_predicted_ids = _normalized_category_ids(
                package_target, package_ids
            )
            vendor_ids, vendor_predicted_ids = _normalized_category_ids(
                vendor_target, vendor_ids
            )
            num_classes = int(package_logits.shape[-1])
            package_bleu = _bleu_diagnostic(
                package_ids,
                cast(Shaped[np.ndarray, "..."], package_true),
                package_predicted_ids,
                cast(Shaped[np.ndarray, "..."], package_pred),
                num_classes,
            )
            vendor_bleu = _bleu_diagnostic(
                vendor_ids,
                cast(Shaped[np.ndarray, "..."], vendor_true),
                vendor_predicted_ids,
                cast(Shaped[np.ndarray, "..."], vendor_pred),
                vendor_depth,
            )
            metric_inputs_equal = (
                np.array_equal(package_ids, vendor_ids)
                and np.array_equal(package_predicted_ids, vendor_predicted_ids)
                and package_mask_equal is True
            )
        elif is_sequence:
            package_prediction = _as_numpy(package_output.numerical_predictions[field])
            vendor_prediction = _as_numpy(vendor_output[field])
            package_cosine = _scaled_cosine_diagnostic(
                package_target,
                cast(Shaped[np.ndarray, "batch elements"], package_true),
                package_prediction,
                cast(Shaped[np.ndarray, "batch elements"], package_pred),
            )
            vendor_cosine = _scaled_cosine_diagnostic(
                vendor_target,
                cast(Shaped[np.ndarray, "batch elements"], vendor_true),
                vendor_prediction,
                cast(Shaped[np.ndarray, "batch elements"], vendor_pred),
            )
            numeric_input_differences = {
                "target": _array_delta_summary(package_target, vendor_target),
                "prediction": _array_delta_summary(
                    package_prediction, vendor_prediction
                ),
                "target_mask": _array_delta_summary(
                    cast(Shaped[np.ndarray, "..."], package_true),
                    cast(Shaped[np.ndarray, "..."], vendor_true),
                ),
                "prediction_mask": _array_delta_summary(
                    cast(Shaped[np.ndarray, "..."], package_pred),
                    cast(Shaped[np.ndarray, "..."], vendor_pred),
                ),
            }
            metric_inputs_equal = (
                np.array_equal(package_target, vendor_target)
                and np.array_equal(package_prediction, vendor_prediction)
                and package_mask_equal is True
            )
        else:
            package_logits = package_output.context_logits[field]
            vendor_logits = vendor_output[field]
            package_pred_ids = package_logits.argmax(dim=-1).detach().cpu().numpy()
            vendor_pred_ids = np.asarray(vendor_logits.numpy()).argmax(axis=-1)
            metric_inputs_equal = np.array_equal(
                package_target, vendor_target
            ) and np.array_equal(package_pred_ids, vendor_pred_ids)

        actual = np.asarray(_as_numpy(package_metrics[field]), dtype=np.float32)
        expected = np.asarray(_as_numpy(vendor_metrics[field]), dtype=np.float32)
        ulp_delta = (
            _float32_ulp_delta(actual, expected)
            if actual.shape == expected.shape
            else None
        )
        if package_bleu is not None and vendor_bleu is not None:
            package_double = np.asarray(
                package_bleu["float64_scores"], dtype=np.float64
            )
            vendor_double = np.asarray(vendor_bleu["float64_scores"], dtype=np.float64)
            float64_equal = np.array_equal(package_double, vendor_double)
            bleu_intermediates: JSONValue = {
                "package": package_bleu["float32_intermediates"],
                "vendor": vendor_bleu["float32_intermediates"],
            }
            effective_intermediates_equal = (
                package_bleu["float32_intermediates"]
                == vendor_bleu["float32_intermediates"]
            )
            float64_scores: JSONValue = {
                "package": package_bleu["float64_scores"],
                "vendor": vendor_bleu["float64_scores"],
            }
            if package_mask_equal is not True or not effective_intermediates_equal:
                diagnosis = "effective BLEU inputs differ"
            elif not float64_equal:
                diagnosis = "same effective BLEU inputs but float64 scores differ; inspect formula semantics"
            elif ulp_delta is not None and np.any(ulp_delta):
                diagnosis = "same effective BLEU inputs and float64 scores; observed discrepancy is float32 rounding"
            else:
                diagnosis = "same effective BLEU inputs and scores"
        elif package_cosine is not None and vendor_cosine is not None:
            package_double = np.asarray(package_cosine["torch_score"], dtype=np.float64)
            vendor_double = np.asarray(
                vendor_cosine["tensorflow_score"], dtype=np.float64
            )
            float64_equal = bool(
                np.allclose(package_double, vendor_double, rtol=1e-14, atol=1e-15)
            )
            float64_scores = {"package": package_cosine, "vendor": vendor_cosine}
            bleu_intermediates = None
            package_torch_scores = np.asarray(
                package_cosine["torch_score"], dtype=np.float64
            )
            package_tensorflow_scores = np.asarray(
                package_cosine["tensorflow_score"], dtype=np.float64
            )
            vendor_torch_scores = np.asarray(
                vendor_cosine["torch_score"], dtype=np.float64
            )
            vendor_tensorflow_scores = np.asarray(
                vendor_cosine["tensorflow_score"], dtype=np.float64
            )
            same_input_formula_scores_equal = {
                "package_inputs": bool(
                    np.allclose(
                        package_torch_scores,
                        package_tensorflow_scores,
                        rtol=1e-14,
                        atol=1e-15,
                    )
                ),
                "vendor_inputs": bool(
                    np.allclose(
                        vendor_torch_scores,
                        vendor_tensorflow_scores,
                        rtol=1e-14,
                        atol=1e-15,
                    )
                ),
            }
            if package_mask_equal is not True:
                diagnosis = "conditional metric masks differ"
            elif not float64_equal and metric_inputs_equal:
                diagnosis = "identical metric inputs but float64 scores differ; inspect formula semantics"
            elif not float64_equal:
                if all(same_input_formula_scores_equal.values()):
                    diagnosis = "float64 formulas agree per side; scores differ because package/vendor metric inputs differ"
                else:
                    diagnosis = "metric inputs differ and float64 package/vendor formula scores differ; inspect both"
            elif ulp_delta is not None and np.any(ulp_delta):
                diagnosis = (
                    "same float64 scores; observed discrepancy is float32 rounding"
                )
            else:
                diagnosis = "same metric inputs and scores"
        else:
            diagnosis = None
            bleu_intermediates = None
            float64_scores = None

        field_reports[field] = {
            "actual_package_scores": actual.tolist(),
            "expected_vendor_scores": expected.tolist(),
            "float32_ulp_delta_from_expected": (
                ulp_delta.tolist() if ulp_delta is not None else None
            ),
            "max_absolute_float32_ulp_distance": (
                int(np.max(np.abs(ulp_delta), initial=0))
                if ulp_delta is not None
                else None
            ),
            "package_vendor_metric_inputs_equal": metric_inputs_equal,
            "package_vendor_masks_equal": package_mask_equal,
            "diagnosis": diagnosis,
            "bleu_intermediates": bleu_intermediates,
            "numeric_metric_input_differences": numeric_input_differences,
            "same_input_float64_formula_scores_equal": same_input_formula_scores_equal,
            "float64_scores": float64_scores,
            "float64_scores_equal": float64_equal,
        }

    actual_total = np.asarray(_as_numpy(package_metrics["total"]), dtype=np.float32)
    expected_total = np.asarray(_as_numpy(vendor_metrics["total"]), dtype=np.float32)
    total_ulp_delta = (
        _float32_ulp_delta(actual_total, expected_total)
        if actual_total.shape == expected_total.shape
        else None
    )
    field_reports["total"] = {
        "actual_package_scores": actual_total.tolist(),
        "expected_vendor_scores": expected_total.tolist(),
        "float32_ulp_delta_from_expected": (
            total_ulp_delta.tolist() if total_ulp_delta is not None else None
        ),
        "max_absolute_float32_ulp_distance": (
            int(np.max(np.abs(total_ulp_delta), initial=0))
            if total_ulp_delta is not None
            else None
        ),
    }

    return {
        "fields": field_reports,
        "float64_method": "BLEU and scaled cosine formulas recomputed from each side's observed inputs and masks in NumPy float64; vendor/package metric layers remain unchanged. Cosine normalization uses an effective norm epsilon of 1e-6, matching TensorFlow l2_normalize's squared-norm epsilon of 1e-12.",
    }


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


def _tensor_values(
    value: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
) -> Float[np.ndarray, "..."]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    return np.asarray(value, dtype=np.float64)


def _aligned_vendor_tensor_values(
    value: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    source: TensorFlowSource,
) -> Float[np.ndarray, "..."]:
    values = _tensor_values(value)
    return values.T if source.transpose else values


def _paired_tensor_norms(
    package: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    vendor: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    source: TensorFlowSource,
) -> dict[str, JSONValue]:
    package_values = _tensor_values(package)
    vendor_values = _tensor_values(vendor)
    if source.transpose:
        vendor_values = vendor_values.T

    return {
        "package": float(np.linalg.norm(package_values)),
        "vendor": float(np.linalg.norm(vendor_values)),
    }


def _distribution_statistics(values: Float[np.ndarray, "..."]) -> dict[str, JSONValue]:
    flattened = values.reshape(-1)
    return {
        "count": int(flattened.size),
        "min": float(np.min(flattened, initial=np.inf)),
        "p50": float(np.median(flattened)),
        "mean": float(np.mean(flattened)),
        "max": float(np.max(flattened, initial=-np.inf)),
    }


def _s2_tensor_diagnostic(
    group: str,
    name: str,
    batch_index: int,
    actual: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    expected: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    source: TensorFlowSource,
    *,
    metric_value: float | None = None,
) -> dict[str, JSONValue]:
    actual_values = _tensor_values(actual)
    expected_values = _tensor_values(expected)
    if source.transpose:
        expected_values = expected_values.T

    difference = np.abs(actual_values - expected_values)
    if group == "s2_zero_gradients":
        metric = "max_abs"
        metric_value = max(
            float(np.max(np.abs(actual_values), initial=0.0)),
            float(np.max(np.abs(expected_values), initial=0.0)),
        )
    else:
        metric = "norm_rel"
        metric_value = float(
            np.linalg.norm(difference)
            / max(float(np.linalg.norm(expected_values)), 1e-30)
        )

    measured_value = (
        max(
            float(np.max(np.abs(actual_values), initial=0.0)),
            float(np.max(np.abs(expected_values), initial=0.0)),
        )
        if group == "s2_zero_gradients"
        else float(
            np.linalg.norm(difference)
            / max(float(np.linalg.norm(expected_values)), 1e-30)
        )
    )
    if metric_value is not None:
        measured_value = metric_value

    diagnostic = {
        "group": group,
        "tensor": name,
        "batch": batch_index,
        "metric": metric,
        "metric_value": measured_value,
        "relative_error": measured_value if metric == "norm_rel" else None,
        "absolute_difference": {
            "max_abs": float(np.max(difference, initial=0.0)),
            "l2": float(np.linalg.norm(difference)),
        },
        "tensor_magnitude": {
            "package_max_abs": float(np.max(np.abs(actual_values), initial=0.0)),
            "vendor_max_abs": float(np.max(np.abs(expected_values), initial=0.0)),
            "package_l2": float(np.linalg.norm(actual_values)),
            "vendor_l2": float(np.linalg.norm(expected_values)),
        },
    }
    if group == "s2_updated_parameters":
        diagnostic["full_vector_relative_l2"] = float(
            np.linalg.norm(difference)
            / max(float(np.linalg.norm(expected_values)), 1e-30)
        )

    return diagnostic


def _optimizer_argmax_diagnostic(
    group: str,
    package: Mapping[str, Float[torch.Tensor, "..."] | Float[np.ndarray, "..."]],
    vendor: Mapping[str, Float[torch.Tensor, "..."] | Float[np.ndarray, "..."]],
) -> dict[str, JSONValue]:
    metric_field = {
        "s2_gradients": "gradient",
        "s2_clipped_gradients": "clipped_gradient",
        "s2_first_moments": "m",
        "s2_second_moments": "v",
        "s2_updated_parameters": "update",
        "s2_zero_gradients": "gradient",
    }[group]
    package_values = {key: _tensor_values(value) for key, value in package.items()}
    vendor_values = {key: _tensor_values(value) for key, value in vendor.items()}
    package_metric = package_values[metric_field]
    vendor_metric = vendor_values[metric_field]
    if group == "s2_zero_gradients":
        source = (
            package_metric
            if np.max(np.abs(package_metric)) >= np.max(np.abs(vendor_metric))
            else vendor_metric
        )
        flat_index = int(np.argmax(np.abs(source)))
        difference = max(
            float(np.max(np.abs(package_metric), initial=0.0)),
            float(np.max(np.abs(vendor_metric), initial=0.0)),
        )
        conditioned = False
    else:
        delta = np.abs(package_metric - vendor_metric)
        if group == "s2_updated_parameters":
            well = np.sqrt(vendor_values["v"]) >= WELL_CONDITIONED_SQRT_V
            if not well.any():
                return {
                    "comparison_field": metric_field,
                    "argmax_coordinate": None,
                    "argmax_absolute_difference": None,
                    "optimizer_values": None,
                }
            delta = np.where(well, delta, -1.0)
            conditioned = True
        else:
            conditioned = False
        flat_index = int(np.argmax(delta))
        difference = float(delta.reshape(-1)[flat_index])

    coordinate = [
        int(value) for value in np.unravel_index(flat_index, package_metric.shape)
    ]
    index = tuple(coordinate)
    return {
        "comparison_field": metric_field,
        "argmax_coordinate": coordinate,
        "argmax_absolute_difference": difference,
        "well_conditioned_coordinate": conditioned,
        "optimizer_values": {
            "package": {
                key: float(values[index])
                for key, values in package_values.items()
                if key != "v"
            },
            "vendor": {
                key: float(values[index])
                for key, values in vendor_values.items()
                if key != "v"
            },
        },
    }


def _batch_norm_tensor_diagnostic(
    group: str,
    tensor: str,
    batch_index: int,
    package: Float[torch.Tensor, "..."],
    vendor: Float[np.ndarray, "..."],
) -> dict[str, JSONValue]:
    package_values = _tensor_values(package)
    vendor_values = _tensor_values(vendor)
    difference = np.abs(package_values - vendor_values)
    flat_index = int(np.argmax(difference))
    coordinate = [
        int(value) for value in np.unravel_index(flat_index, package_values.shape)
    ]
    index = tuple(coordinate)
    return {
        "group": group,
        "tensor": tensor,
        "batch": batch_index,
        "metric": "max_rel_to_max",
        "metric_value": float(
            np.max(difference, initial=0.0)
            / max(float(np.max(np.abs(vendor_values), initial=0.0)), 1e-12)
        ),
        "argmax_coordinate": coordinate,
        "argmax_absolute_difference": float(difference[index]),
        "coordinate_values": {
            side: {
                "value": float(values[index]),
                "gradient": None,
                "m": None,
                "sqrt_v_plus_eps": None,
                "update": None,
            }
            for side, values in (
                ("package", package_values),
                ("vendor", vendor_values),
            )
        },
        "optimizer_state_applicable": False,
    }


def _adam_denominator_statistics(
    second_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
    epsilon: float,
) -> dict[str, JSONValue]:
    if isinstance(second_moment, torch.Tensor):
        second_moment = second_moment.detach().cpu().numpy()

    values = np.asarray(second_moment)
    denominator = np.sqrt(values) + np.asarray(epsilon, dtype=values.dtype)
    return _distribution_statistics(np.asarray(denominator, dtype=np.float64))


def _s2_optimizer_context(
    parameter_norm_before: dict[str, JSONValue],
    parameter_norm_after: dict[str, JSONValue],
    raw_gradient_norm: dict[str, JSONValue] | None,
    clipped_gradient_norm: dict[str, JSONValue] | None,
    package_second_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."] | None,
    package_epsilon: float,
    vendor_second_moment: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."] | None,
    vendor_epsilon: float,
) -> dict[str, JSONValue]:
    denominator = None
    if package_second_moment is not None and vendor_second_moment is not None:
        denominator = {
            "package": _adam_denominator_statistics(
                package_second_moment, package_epsilon
            ),
            "vendor": _adam_denominator_statistics(
                vendor_second_moment, vendor_epsilon
            ),
        }

    return {
        "parameter_norm": {
            "before_step": parameter_norm_before,
            "after_step": parameter_norm_after,
        },
        "gradient_norm": {
            "raw": raw_gradient_norm,
            "clipped": clipped_gradient_norm,
        },
        "adam_denominator": denominator,
    }


def _largest_s2_tensor_diagnostics(
    diagnostics: Sequence[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    summary: dict[str, JSONValue] = {}
    for row in diagnostics:
        group = cast(str, row["group"])
        metric_value = cast(float, row["metric_value"])
        current = cast(dict[str, JSONValue] | None, summary.get(group))
        current_value = (
            -1.0 if current is None else cast(float, current["metric_value"])
        )
        if metric_value > current_value:
            summary[group] = row

    return summary


def _train_step(
    tf,
    vendor_model,
    package_model: CanvasVAECrelloModel,
    optimizer: KerasAdam,
    batch: CrelloBatch,
    noise: Float[np.ndarray, "batch latent"],
    batch_index: int,
    records: list[_Comparison],
    zero_gradient_checks: list[dict[str, JSONValue]],
    s2_tensor_diagnostics: list[dict[str, JSONValue]],
    s3_tensor_diagnostics: list[dict[str, JSONValue]],
    static_errors: list[str],
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
    relative_mapping, zero_gradient_mapping = split_attention_key_biases(mapping)
    all_mapping = {**relative_mapping, **zero_gradient_mapping}
    package_parameters = dict(package_model.named_parameters())
    if not zero_gradient_mapping:
        static_errors.append("no attention key-projection biases were checked")

    parameter_norm_before: dict[str, dict[str, JSONValue]] = {}
    parameter_values_before: dict[
        str, tuple[Float[torch.Tensor, "..."], Float[np.ndarray, "..."]]
    ] = {}
    diagnostics_by_tensor: dict[str, list[dict[str, JSONValue]]] = defaultdict(list)
    raw_gradient_norms: dict[str, dict[str, JSONValue]] = {}
    clipped_gradient_norms: dict[str, dict[str, JSONValue]] = {}
    optimizer_values_package: dict[
        str,
        dict[str, Float[torch.Tensor, "..."] | Float[np.ndarray, "..."]],
    ] = defaultdict(dict)
    optimizer_values_vendor: dict[
        str,
        dict[str, Float[torch.Tensor, "..."] | Float[np.ndarray, "..."]],
    ] = defaultdict(dict)
    for state_key, source in all_mapping.items():
        package_parameter = package_parameters.get(state_key)
        tf_variable = vendor_variables.get(source.key)
        if package_parameter is None or tf_variable is None:
            continue

        parameter_norm_before[state_key] = _paired_tensor_norms(
            package_parameter,
            tf_variable.numpy(),
            source,
        )
        parameter_values_before[state_key] = (
            package_parameter.detach().clone(),
            tf_variable.numpy().copy(),
        )

    for state_key, source in zero_gradient_mapping.items():
        package_parameter = package_parameters.get(state_key)
        tf_gradient = gradients_by_path.get(source.key)
        if (
            package_parameter is None
            or package_parameter.grad is None
            or tf_gradient is None
        ):
            zero_gradient_checks.append(
                {
                    "tensor": state_key,
                    "batch": batch_index,
                    "package_max_abs": None,
                    "vendor_max_abs": None,
                    "calibration_floor": ZERO_GRADIENT_FLOOR,
                }
            )
            static_errors.append(f"zero-gradient tensor {state_key} has no gradient")
            continue

        check = check_zero_gradient(package_parameter.grad, tf_gradient.numpy())
        optimizer_values_package[state_key]["gradient"] = (
            package_parameter.grad.detach().clone()
        )
        optimizer_values_vendor[state_key]["gradient"] = _aligned_vendor_tensor_values(
            tf_gradient.numpy(), source
        )
        raw_gradient_norms[state_key] = _paired_tensor_norms(
            package_parameter.grad,
            tf_gradient.numpy(),
            mapping[state_key],
        )
        diagnostics_by_tensor[state_key].append(
            _s2_tensor_diagnostic(
                "s2_zero_gradients",
                state_key,
                batch_index,
                package_parameter.grad,
                tf_gradient.numpy(),
                source,
            )
        )
        zero_gradient_checks.append(
            {
                "tensor": state_key,
                "batch": batch_index,
                "package_max_abs": check.package_max_abs,
                "vendor_max_abs": check.vendor_max_abs,
                "calibration_floor": ZERO_GRADIENT_FLOOR,
            }
        )
        for side, value in (
            ("package", check.package_max_abs),
            ("vendor", check.vendor_max_abs),
        ):
            records.append(
                _Comparison(
                    "s2_zero_gradients",
                    f"{side}:{state_key}",
                    batch_index,
                    "max_abs",
                    value,
                    [],
                    [],
                    True,
                )
            )

    for state_key, source in relative_mapping.items():
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
        optimizer_values_package[state_key]["gradient"] = (
            package_parameter.grad.detach().clone()
        )
        optimizer_values_vendor[state_key]["gradient"] = _aligned_vendor_tensor_values(
            tf_gradient.numpy(), source
        )
        raw_gradient_norms[state_key] = _paired_tensor_norms(
            package_parameter.grad,
            tf_gradient.numpy(),
            source,
        )
        diagnostics_by_tensor[state_key].append(
            _s2_tensor_diagnostic(
                "s2_gradients",
                state_key,
                batch_index,
                package_parameter.grad,
                tf_gradient.numpy(),
                source,
            )
        )

    vendor_clipped = {
        id(variable): tf.clip_by_norm(gradient, CLIP_NORM).numpy()
        for variable, gradient in zip(
            vendor_model.trainable_variables, vendor_gradients, strict=True
        )
        if gradient is not None
    }
    clip_gradients_by_norm(package_model.parameters(), CLIP_NORM)
    for state_key, source in relative_mapping.items():
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
        optimizer_values_package[state_key]["clipped_gradient"] = (
            package_parameter.grad.detach().clone()
        )
        optimizer_values_vendor[state_key]["clipped_gradient"] = (
            _aligned_vendor_tensor_values(vendor_clipped[id(tf_variable)], source)
        )
        clipped_gradient_norms[state_key] = _paired_tensor_norms(
            package_parameter.grad,
            vendor_clipped[id(tf_variable)],
            source,
        )
        diagnostics_by_tensor[state_key].append(
            _s2_tensor_diagnostic(
                "s2_clipped_gradients",
                state_key,
                batch_index,
                package_parameter.grad,
                vendor_clipped[id(tf_variable)],
                source,
            )
        )

    vendor_model.optimizer.apply_gradients(
        zip(vendor_gradients, vendor_model.trainable_variables, strict=True)
    )
    optimizer.step()
    for state_key, source in all_mapping.items():
        package_parameter = package_parameters.get(state_key)
        tf_variable = vendor_variables.get(source.key)
        if package_parameter is None or tf_variable is None:
            continue

        package_before, vendor_before = parameter_values_before[state_key]
        package_update = package_parameter.detach() - package_before
        vendor_update = tf_variable.numpy() - vendor_before
        vendor_update_aligned = _aligned_vendor_tensor_values(vendor_update, source)
        index = vendor_model.optimizer._index_dict[
            vendor_model.optimizer._var_key(tf_variable)
        ]
        package_state = optimizer.state[package_parameter]
        vendor_first_moment = vendor_model.optimizer._momentums[index].numpy()
        vendor_second_moment = vendor_model.optimizer._velocities[index].numpy()
        package_first_moment = package_state["exp_avg"]
        package_second_moment = package_state["exp_avg_sq"]

        optimizer_values_package[state_key].update(
            {
                "m": package_first_moment.detach().clone(),
                "v": package_second_moment.detach().clone(),
                "sqrt_v_plus_eps": torch.sqrt(package_second_moment)
                + KERAS_ADAM_EPSILON,
                "update": package_update.detach().clone(),
            }
        )
        optimizer_values_vendor[state_key].update(
            {
                "m": _aligned_vendor_tensor_values(vendor_first_moment, source),
                "v": _aligned_vendor_tensor_values(vendor_second_moment, source),
                "sqrt_v_plus_eps": np.sqrt(
                    _aligned_vendor_tensor_values(vendor_second_moment, source)
                )
                + KERAS_ADAM_EPSILON,
                "update": vendor_update_aligned,
            }
        )

        if state_key in relative_mapping:
            zeros = torch.zeros_like(package_update)
            update_check = check_keras_adam_update(
                package_update,
                vendor_update_aligned,
                optimizer_values_package[state_key]["gradient"],
                optimizer_values_vendor[state_key]["gradient"],
                zeros,
                zeros,
                step=1,
                adam_rule_limit=S2_ADAM_RULE_LIMIT,
                well_conditioned_limit=None,
            )
            records.append(
                _Comparison(
                    "s2_updated_parameters",
                    state_key,
                    batch_index,
                    "norm_rel",
                    update_check["well_conditioned_norm_rel"],
                    list(package_update.shape),
                    list(vendor_update_aligned.shape),
                    True,
                )
            )
            for side, check_key in (
                ("package", "package_adam_rule_within"),
                ("vendor", "original_adam_rule_within"),
            ):
                if not update_check[check_key]:
                    static_errors.append(
                        f"{side} Adam update for {state_key} differs from its "
                        "float64 Keras reference by more than "
                        f"{S2_ADAM_RULE_LIMIT}"
                    )

            diagnostic = _s2_tensor_diagnostic(
                "s2_updated_parameters",
                state_key,
                batch_index,
                package_update,
                vendor_update,
                source,
                metric_value=update_check["well_conditioned_norm_rel"],
            )
            diagnostic.update(cast(Mapping[str, JSONValue], update_check))
            diagnostic["float64_reference_relative_l2"] = {
                "package": update_check["package_adam_rule"],
                "vendor": update_check["original_adam_rule"],
            }
            well = np.sqrt(optimizer_values_vendor[state_key]["v"])
            well_mask = well >= WELL_CONDITIONED_SQRT_V
            update_delta = np.abs(
                _tensor_values(package_update) - vendor_update_aligned
            )
            conditioned_delta = update_delta[well_mask]
            diagnostic["well_conditioned_absolute_difference"] = {
                "elements": int(conditioned_delta.size),
                "l2": float(np.linalg.norm(conditioned_delta)),
                "max_abs": float(np.max(conditioned_delta, initial=0.0)),
            }
            diagnostic["update_norm"] = {
                "package": float(torch.linalg.vector_norm(package_update)),
                "vendor": float(np.linalg.norm(vendor_update_aligned)),
            }
            diagnostics_by_tensor[state_key].append(diagnostic)

        if state_key in relative_mapping:
            _compare_state_tensor(
                records,
                "s2_first_moments",
                batch_index,
                state_key,
                package_first_moment,
                vendor_first_moment,
                source,
            )
            diagnostics_by_tensor[state_key].append(
                _s2_tensor_diagnostic(
                    "s2_first_moments",
                    state_key,
                    batch_index,
                    package_first_moment,
                    vendor_first_moment,
                    source,
                )
            )
            _compare_state_tensor(
                records,
                "s2_second_moments",
                batch_index,
                state_key,
                package_second_moment,
                vendor_second_moment,
                source,
            )
            diagnostics_by_tensor[state_key].append(
                _s2_tensor_diagnostic(
                    "s2_second_moments",
                    state_key,
                    batch_index,
                    package_second_moment,
                    vendor_second_moment,
                    source,
                )
            )

    for state_key, source in all_mapping.items():
        package_parameter = package_parameters.get(state_key)
        tf_variable = vendor_variables.get(source.key)
        if package_parameter is None or tf_variable is None:
            continue

        index = vendor_model.optimizer._index_dict[
            vendor_model.optimizer._var_key(tf_variable)
        ]
        package_second_moment = optimizer_values_package[state_key].get("v")
        vendor_second_moment = (
            optimizer_values_vendor[state_key].get("v")
            if index < len(vendor_model.optimizer._velocities)
            else None
        )
        vendor_velocity = vendor_second_moment
        parameter_norm_after = _paired_tensor_norms(
            package_parameter,
            tf_variable.numpy(),
            source,
        )
        context = _s2_optimizer_context(
            parameter_norm_before[state_key],
            parameter_norm_after,
            raw_gradient_norms.get(state_key),
            clipped_gradient_norms.get(state_key),
            package_second_moment,
            float(optimizer.param_groups[0]["eps"]),
            vendor_velocity,
            float(vendor_model.optimizer.epsilon),
        )
        for diagnostic in diagnostics_by_tensor[state_key]:
            diagnostic.update(context)
            diagnostic["argmax_coordinate_diagnostic"] = _optimizer_argmax_diagnostic(
                cast(str, diagnostic["group"]),
                optimizer_values_package[state_key],
                optimizer_values_vendor[state_key],
            )
            s2_tensor_diagnostics.append(diagnostic)

    for group, tensor, package_value, vendor_value in (
        (
            "s3_batch_norm_mean",
            "encoder.norm.running_mean",
            package_model.encoder.norm.running_mean,
            vendor_model.encoder.norm.moving_mean.numpy(),
        ),
        (
            "s3_batch_norm_variance",
            "encoder.norm.running_variance",
            package_model.encoder.norm.running_var,
            vendor_model.encoder.norm.moving_variance.numpy(),
        ),
    ):
        records.append(
            _comparison(
                group,
                tensor,
                batch_index,
                package_value,
                vendor_value,
                "max_rel_to_max",
            )
        )
        s3_tensor_diagnostics.append(
            _batch_norm_tensor_diagnostic(
                group, tensor, batch_index, package_value, vendor_value
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


def _largest_s2_relative_tensors(
    records: Sequence[_Comparison],
) -> dict[str, JSONValue]:
    summary: dict[str, JSONValue] = {}
    groups = sorted({row.group for row in records if row.group.startswith("s2_")})
    for group in groups:
        candidates = [
            row
            for row in records
            if row.group == group and row.metric == "norm_rel" and row.value is not None
        ]
        if not candidates:
            continue

        largest = max(candidates, key=lambda row: row.value or 0.0)
        summary[group] = {
            "tensor": largest.name,
            "batch": largest.batch,
            "metric": largest.metric,
            "relative_error": largest.value,
        }

    return summary


def _group_maxima(records: Sequence[_Comparison]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.value is not None:
            grouped[row.group].append(row.value)

    return {group: max(values) for group, values in grouped.items()}


def _annotate_zero_gradient_checks(
    checks: list[dict[str, JSONValue]], limit: float | None
) -> None:
    for check in checks:
        package_value = cast(float | None, check["package_max_abs"])
        vendor_value = cast(float | None, check["vendor_max_abs"])
        package_within = (
            None if limit is None or package_value is None else package_value <= limit
        )
        vendor_within = (
            None if limit is None or vendor_value is None else vendor_value <= limit
        )
        check["limit"] = limit
        check["package_within_limit"] = package_within
        check["vendor_within_limit"] = vendor_within
        check["within_limit"] = (
            None
            if package_within is None or vendor_within is None
            else package_within and vendor_within
        )


def _limits(records: Sequence[_Comparison]) -> dict[str, _FrozenLimit]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics: dict[str, str] = {}
    for row in records:
        if row.value is not None:
            grouped[row.group][row.batch].append(row.value)
            metrics[row.group] = row.metric

    limits: dict[str, _FrozenLimit] = {}
    for group, batches in grouped.items():
        registered_floor = (
            METRIC_SCORE_FLOOR
            if group.startswith(("s1_reconstruction_metrics/", "s1_layout_metrics/"))
            else ZERO_GRADIENT_FLOOR
            if group == "s2_zero_gradients"
            else 0.0
        )
        batch_maxima = [
            max(batches.get(index, ()), default=0.0)
            for index in range(len(CALIBRATION_SLICES))
        ]
        maximum = max(batch_maxima, default=0.0)
        limits[group] = {
            "metric": metrics[group],
            "L": registered_floor,
            "calibration_batch_maxima": batch_maxima,
            "max_calibration_error": maximum,
            "limit": max(registered_floor, _ceil_two_significant_digits(1.5 * maximum)),
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

    calibration_bytes = (report_dir / "calibrate.json").read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != limits["calibration_sha256"]:
        raise ValueError("frozen Crello calibration report SHA-256 mismatch")

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
    zero_gradient_checks: list[dict[str, JSONValue]] = []
    s2_tensor_diagnostics: list[dict[str, JSONValue]] = []
    s3_tensor_diagnostics: list[dict[str, JSONValue]] = []
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
        if completed.returncode != 0 or not child_report_path.exists():
            process_report["stdout_tail"] = _process_output_excerpt(
                cast(str | None, getattr(completed, "stdout", None))
            )
            process_report["stderr_tail"] = _process_output_excerpt(
                cast(str | None, getattr(completed, "stderr", None))
            )
        if not child_report_path.exists():
            static_errors.append(
                f"calibration process {batch_index} did not write its report"
            )
            process_reports.append(process_report)
            break

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
        zero_gradient_checks.extend(
            cast(
                list[dict[str, JSONValue]],
                child.get("zero_gradient_checks", []),
            )
        )
        child_s2_diagnostics = cast(
            list[dict[str, JSONValue]], child.get("s2_tensor_diagnostics", [])
        )
        s2_tensor_diagnostics.extend(child_s2_diagnostics)
        child_missing_diagnostics = set(S2_DIAGNOSTIC_GROUPS) - set(
            _largest_s2_tensor_diagnostics(child_s2_diagnostics)
        )
        if child_missing_diagnostics:
            static_errors.append(
                f"calibration process {batch_index} is missing S2 tensor diagnostics: "
                + ", ".join(sorted(child_missing_diagnostics))
            )
        child_s3_diagnostics = cast(
            list[dict[str, JSONValue]], child.get("s3_tensor_diagnostics", [])
        )
        s3_tensor_diagnostics.extend(child_s3_diagnostics)
        child_missing_s3_diagnostics = set(S3_DIAGNOSTIC_GROUPS) - set(
            _largest_s2_tensor_diagnostics(child_s3_diagnostics)
        )
        if child_missing_s3_diagnostics:
            static_errors.append(
                f"calibration process {batch_index} is missing S3 tensor diagnostics: "
                + ", ".join(sorted(child_missing_s3_diagnostics))
            )
        child_digests = cast(list[str], child["batch_digests"])
        child_batch_invalid = len(child_digests) != 1
        if len(child_digests) != 1:
            static_errors.append(
                f"calibration process {batch_index} did not use exactly one batch"
            )
        batch_digests.extend(child_digests)
        if len(set(batch_digests)) != len(batch_digests):
            static_errors.append("calibration inputs are not three distinct batches")
            child_batch_invalid = True
        initial_state_hashes.append(cast(str, child["initial_state_sha256"]))
        initial_mismatch_keys.extend(
            cast(list[str], child["initial_state_mismatch_keys"])
        )
        if len(set(initial_state_hashes)) > 1:
            static_errors.append(
                "calibration processes did not share the prescribed initial state"
            )
        child_records = [
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
            for measurement in cast(list[dict[str, JSONValue]], child["measurements"])
        ]
        records.extend(child_records)
        if (
            completed.returncode != 0
            or child_errors
            or child_batch_invalid
            or cast(list[str], child["initial_state_mismatch_keys"])
            or len(set(initial_state_hashes)) > 1
            or _has_shape_errors(child_records)
        ):
            break

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
    if "s2_zero_gradients" not in all_groups:
        static_errors.append("calibration produced no key-bias gradient measurements")
    for group in all_groups:
        if any(group not in groups_per_batch[index] for index in groups_per_batch):
            static_errors.append(
                f"calibration metric group {group} is missing from a batch"
            )

    candidate_limits = _limits(records)
    largest_s2_diagnostics = _largest_s2_tensor_diagnostics(s2_tensor_diagnostics)
    missing_s2_diagnostics = set(S2_DIAGNOSTIC_GROUPS) - set(largest_s2_diagnostics)
    if missing_s2_diagnostics:
        static_errors.append(
            "calibration is missing S2 tensor diagnostics: "
            + ", ".join(sorted(missing_s2_diagnostics))
        )
    largest_s3_diagnostics = _largest_s2_tensor_diagnostics(s3_tensor_diagnostics)
    missing_s3_diagnostics = set(S3_DIAGNOSTIC_GROUPS) - set(largest_s3_diagnostics)
    if missing_s3_diagnostics:
        static_errors.append(
            "calibration is missing S3 tensor diagnostics: "
            + ", ".join(sorted(missing_s3_diagnostics))
        )

    calibration_complete = (
        not static_errors
        and not initial_mismatch_keys
        and not _has_shape_errors(records)
        and len(process_reports) == len(CALIBRATION_SLICES)
    )
    zero_gradient_limit = (
        candidate_limits["s2_zero_gradients"]["limit"] if calibration_complete else None
    )
    _annotate_zero_gradient_checks(zero_gradient_checks, zero_gradient_limit)

    report: dict[str, JSONValue] = {
        **report_context,
        "phase": "calibrate",
        "calibration_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "fixture_array_sha256": fixture_array_sha256,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "input_selection": CALIBRATION_SELECTION,
        "calibration_processes": process_reports,
        "calibration_complete": calibration_complete,
        "incomplete": not calibration_complete,
        "batch_digests": batch_digests,
        "distinct_input_batches": distinct_input_batches,
        "initial_state_sha256": initial_state_hashes[0]
        if initial_state_hashes
        else None,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_mismatch_keys,
        "measurements": [_record_payload(row) for row in records],
        "zero_gradient_checks": zero_gradient_checks,
        "s2_tensor_diagnostics": s2_tensor_diagnostics,
        "s3_tensor_diagnostics": s3_tensor_diagnostics,
        "largest_s2_tensor_diagnostic_by_group": largest_s2_diagnostics,
        "largest_s3_tensor_diagnostic_by_group": largest_s3_diagnostics,
        "largest_relative_tensor_by_s2_group": _largest_s2_relative_tensors(records),
        "calibration_formula": CALIBRATION_FORMULA,
        "limits": candidate_limits if calibration_complete else {},
        "candidate_limits": {} if calibration_complete else candidate_limits,
    }
    report_bytes = _write_json(output_path, report)
    if not calibration_complete:
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


def _heldout(
    *,
    data_dir: Path = DATA_DIR,
    fixture_dir: Path = FIXTURE_DIR,
    report_dir: Path = REPORT_DIR,
    fixture_array_sha256: str | None = None,
    fixture_manifest_sha256: str | None = None,
) -> dict[str, JSONValue]:
    report_context = _cpu_report_context()
    frozen_bytes, frozen = _load_frozen_limits(
        report_dir, str(report_context["commit"])
    )
    documents = load_crello_split(data_dir, CrelloSplit.test)
    batches = _heldout_test_batches(documents)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "heldout.json"
    plan_path = report_dir / "heldout-plan.json"
    for path in (
        output_path,
        plan_path,
        *(report_dir / f"heldout-batch-{index}.json" for index, _ in batches),
    ):
        path.unlink(missing_ok=True)

    plan = {
        **report_context,
        "phase": "heldout_plan",
        "input_selection": (
            "all canonical test.jsonl documents in sequential 1,024-document batches; "
            "each batch starts from the prescribed initial state"
        ),
        "frozen_limits_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "batches": [
            {
                "batch_index": index,
                "split": "test",
                "start": index * BATCH_SIZE,
                "stop": index * BATCH_SIZE + len(rows),
            }
            for index, rows in batches
        ],
    }
    plan_bytes = _write_json(plan_path, plan)

    records: list[_Comparison] = []
    static_errors: list[str] = []
    initial_state_hashes: list[str] = []
    initial_mismatch_keys: list[str] = []
    batch_digests: list[str] = []
    zero_gradient_checks: list[dict[str, JSONValue]] = []
    s2_tensor_diagnostics: list[dict[str, JSONValue]] = []
    s3_tensor_diagnostics: list[dict[str, JSONValue]] = []
    process_reports: list[dict[str, JSONValue]] = []
    script = Path(__file__).resolve()
    calibration_report = cast(
        dict[str, JSONValue], json.loads((report_dir / "calibrate.json").read_bytes())
    )
    for batch_index, _ in batches:
        child_report_path = report_dir / f"heldout-batch-{batch_index}.json"
        command = [
            sys.executable,
            str(script),
            "heldout-batch",
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
        if completed.returncode != 0 or not child_report_path.exists():
            process_report["stdout_tail"] = _process_output_excerpt(
                cast(str | None, getattr(completed, "stdout", None))
            )
            process_report["stderr_tail"] = _process_output_excerpt(
                cast(str | None, getattr(completed, "stderr", None))
            )
        if not child_report_path.exists():
            static_errors.append(
                f"held-out process {batch_index} did not write its report"
            )
            process_reports.append(process_report)
            continue

        child_bytes = child_report_path.read_bytes()
        child = cast(dict[str, JSONValue], json.loads(child_bytes))
        process_report["report_sha256"] = hashlib.sha256(child_bytes).hexdigest()
        process_reports.append(process_report)
        if completed.returncode != 0:
            static_errors.append(
                f"held-out process {batch_index} exited with {completed.returncode}"
            )

        static_errors.extend(cast(list[str], child["static_errors"]))
        zero_gradient_checks.extend(
            cast(
                list[dict[str, JSONValue]],
                child.get("zero_gradient_checks", []),
            )
        )
        child_s2_diagnostics = cast(
            list[dict[str, JSONValue]], child.get("s2_tensor_diagnostics", [])
        )
        s2_tensor_diagnostics.extend(child_s2_diagnostics)
        child_missing_diagnostics = set(S2_DIAGNOSTIC_GROUPS) - set(
            _largest_s2_tensor_diagnostics(child_s2_diagnostics)
        )
        if child_missing_diagnostics:
            static_errors.append(
                f"held-out process {batch_index} is missing S2 tensor diagnostics: "
                + ", ".join(sorted(child_missing_diagnostics))
            )
        child_s3_diagnostics = cast(
            list[dict[str, JSONValue]], child.get("s3_tensor_diagnostics", [])
        )
        s3_tensor_diagnostics.extend(child_s3_diagnostics)
        child_missing_s3_diagnostics = set(S3_DIAGNOSTIC_GROUPS) - set(
            _largest_s2_tensor_diagnostics(child_s3_diagnostics)
        )
        if child_missing_s3_diagnostics:
            static_errors.append(
                f"held-out process {batch_index} is missing S3 tensor diagnostics: "
                + ", ".join(sorted(child_missing_s3_diagnostics))
            )
        child_digests = cast(list[str], child["batch_digests"])
        if len(child_digests) != 1:
            static_errors.append(
                f"held-out process {batch_index} did not use exactly one batch"
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

    if len(initial_state_hashes) != len(batches) or len(set(initial_state_hashes)) != 1:
        static_errors.append(
            "held-out processes did not share the prescribed initial state"
        )
    if initial_state_hashes and any(
        value != calibration_report["initial_state_sha256"]
        for value in initial_state_hashes
    ):
        static_errors.append(
            "held-out processes differed from the calibration initial state"
        )
    if len(process_reports) != len(batches):
        static_errors.append("held-out process reports are incomplete")

    groups_by_batch = {
        index: {
            row.group for row in records if row.batch == index and row.value is not None
        }
        for index, _ in batches
    }
    for index, groups in groups_by_batch.items():
        missing_groups = set(frozen["limits"]) - groups
        if missing_groups:
            static_errors.append(
                f"held-out batch {index} is missing metrics: {', '.join(sorted(missing_groups))}"
            )

    heldout_errors: list[dict[str, JSONValue]] = []
    for row in records:
        limit = frozen["limits"].get(row.group)
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

    zero_gradient_limit = frozen["limits"].get("s2_zero_gradients", {}).get("limit")
    _annotate_zero_gradient_checks(zero_gradient_checks, zero_gradient_limit)
    largest_s2_diagnostics = _largest_s2_tensor_diagnostics(s2_tensor_diagnostics)
    missing_s2_diagnostics = set(S2_DIAGNOSTIC_GROUPS) - set(largest_s2_diagnostics)
    if missing_s2_diagnostics:
        static_errors.append(
            "held-out run is missing S2 tensor diagnostics: "
            + ", ".join(sorted(missing_s2_diagnostics))
        )
    largest_s3_diagnostics = _largest_s2_tensor_diagnostics(s3_tensor_diagnostics)
    missing_s3_diagnostics = set(S3_DIAGNOSTIC_GROUPS) - set(largest_s3_diagnostics)
    if missing_s3_diagnostics:
        static_errors.append(
            "held-out run is missing S3 tensor diagnostics: "
            + ", ".join(sorted(missing_s3_diagnostics))
        )

    report: dict[str, JSONValue] = {
        **report_context,
        "phase": "heldout",
        "heldout_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "fixture_array_sha256": fixture_array_sha256,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "input_selection": plan["input_selection"],
        "batch_digests": batch_digests,
        "initial_state_sha256": initial_state_hashes[0]
        if initial_state_hashes
        else None,
        "heldout_processes": process_reports,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_mismatch_keys,
        "measurements": [_record_payload(row) for row in records],
        "zero_gradient_checks": zero_gradient_checks,
        "s2_tensor_diagnostics": s2_tensor_diagnostics,
        "s3_tensor_diagnostics": s3_tensor_diagnostics,
        "largest_s2_tensor_diagnostic_by_group": largest_s2_diagnostics,
        "largest_s3_tensor_diagnostic_by_group": largest_s3_diagnostics,
        "largest_relative_tensor_by_s2_group": _largest_s2_relative_tensors(records),
        "frozen_limits_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "heldout_errors": heldout_errors,
        "heldout_maxima": _group_maxima(records),
    }
    _write_json(output_path, report)
    if (
        static_errors
        or initial_mismatch_keys
        or _has_shape_errors(records)
        or heldout_errors
        or not records
    ):
        raise AssertionError(f"held-out checks failed; report written to {output_path}")

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
    diagnostic_split: str | None = None,
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
    if phase == "heldout-batch":
        _load_frozen_limits(report_dir, str(report_context["commit"]))
    elif phase == "calibration-batch":
        if batch_index is None:
            raise ValueError("calibration process requires a batch index")
        if batch_index not in range(len(CALIBRATION_SLICES)):
            raise ValueError("calibration process requires batch index 0, 1, or 2")
    elif phase == "metric-diagnostic":
        if diagnostic_split not in {"calibration", "heldout"}:
            raise ValueError("metric diagnosis requires calibration or heldout split")
        if batch_index is None or batch_index < 0:
            raise ValueError("metric diagnosis requires a non-negative batch index")
        if diagnostic_split == "calibration" and batch_index not in range(
            len(CALIBRATION_SLICES)
        ):
            raise ValueError(
                "calibration metric diagnosis requires batch index 0, 1, or 2"
            )
    else:
        raise ValueError(
            "Crello parity phase must be heldout-batch, calibration-batch, or metric-diagnostic"
        )

    documents = {split: load_crello_split(data_dir, split) for split in CrelloSplit}
    if phase == "calibration-batch" or (
        phase == "metric-diagnostic" and diagnostic_split == "calibration"
    ):
        calibration_index = cast(int, batch_index)
        start, stop = CALIBRATION_SLICES[calibration_index]
        if len(documents[CrelloSplit.train]) < stop:
            raise ValueError("Crello calibration requires three full train batches")

        selected = [(calibration_index, documents[CrelloSplit.train][start:stop])]
    else:
        heldout_batches = _heldout_test_batches(documents[CrelloSplit.test])
        if batch_index is None:
            raise ValueError("held-out process requires a canonical test batch index")
        if batch_index not in range(len(heldout_batches)):
            raise ValueError("held-out process requires a canonical test batch index")

        selected = [heldout_batches[batch_index]]

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
    optimizer = (
        None
        if phase == "metric-diagnostic"
        else KerasAdam(package_model.parameters(), lr=LEARNING_RATE)
    )

    records: list[_Comparison] = []
    metric_diagnostics: list[dict[str, JSONValue]] = []
    zero_gradient_checks: list[dict[str, JSONValue]] = []
    s2_tensor_diagnostics: list[dict[str, JSONValue]] = []
    s3_tensor_diagnostics: list[dict[str, JSONValue]] = []
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
        if phase == "metric-diagnostic":
            metric_diagnostics.append(
                _metric_diagnostic_report(
                    batch,
                    eval_output,
                    tf_batch,
                    eval_reference,
                    cast(Mapping[str, _VendorColumn], columns),
                    {
                        key: value
                        for key, value in package_metrics.items()
                        if key not in {"layout_acc", "layout_miou"}
                    },
                    vendor_reconstruction_metrics,
                    config,
                    cast(_VendorModelMetrics, vendor_model),
                )
            )
            continue

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
        if optimizer is None:
            raise RuntimeError("training parity requires an optimizer")

        _train_step(
            tf,
            vendor_model,
            package_model,
            optimizer,
            batch,
            noise,
            batch_index,
            records,
            zero_gradient_checks,
            s2_tensor_diagnostics,
            s3_tensor_diagnostics,
            static_errors,
        )

    report_phase = (
        "metric_diagnostic"
        if phase == "metric-diagnostic"
        else "calibration_batch"
        if phase == "calibration-batch"
        else phase
    )
    if phase == "metric-diagnostic":
        output_name = f"metric-diagnostic-{diagnostic_split}-{batch_index}.json"
    elif phase == "calibration-batch":
        output_name = f"calibration-batch-{batch_index}.json"
    else:
        output_name = f"heldout-batch-{batch_index}.json"
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
            if (
                phase == "calibration-batch"
                or (phase == "metric-diagnostic" and diagnostic_split == "calibration")
            )
            and batch_index is not None
            else f"canonical test.jsonl batch {batch_index}, sequential 1,024-document batches"
        ),
        "batch_digests": batch_digests,
        "static_errors": static_errors,
        "initial_state_mismatch_keys": initial_errors,
        "measurements": [_record_payload(row) for row in records],
        "zero_gradient_checks": zero_gradient_checks,
        "s2_tensor_diagnostics": s2_tensor_diagnostics,
        "s3_tensor_diagnostics": s3_tensor_diagnostics,
        "largest_s2_tensor_diagnostic_by_group": _largest_s2_tensor_diagnostics(
            s2_tensor_diagnostics
        ),
        "largest_s3_tensor_diagnostic_by_group": _largest_s2_tensor_diagnostics(
            s3_tensor_diagnostics
        ),
        "largest_relative_tensor_by_s2_group": _largest_s2_relative_tensors(records),
    }
    if phase == "metric-diagnostic":
        report.update(
            {
                "report_only": True,
                "limits_modified": False,
                "tolerance_assertions_run": False,
                "training_step_run": False,
                "metric_diagnostics": metric_diagnostics,
            }
        )

    _write_json(output_path, report)
    if phase != "metric-diagnostic" and (
        _has_shape_errors(records) or static_errors or initial_errors
    ):
        raise AssertionError(
            f"{report_phase} checks failed; report written to {output_path}"
        )

    return report


def main() -> None:
    """Run fixture-independent configuration checks or the calibrated model checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "s0",
            "calibrate",
            "calibration-batch",
            "heldout-batch",
            "metric-diagnostic",
            "heldout",
            "run",
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--fixture-hub-location")
    parser.add_argument("--fixture-array-sha256")
    parser.add_argument("--fixture-manifest-sha256")
    parser.add_argument("--fixture-token-file", type=Path)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--diagnostic-split", choices=("calibration", "heldout"))
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
            heldout = _heldout(
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
    elif args.phase == "heldout":
        report = _heldout(
            data_dir=args.data_dir,
            fixture_dir=args.fixture_dir,
            report_dir=args.report_dir,
            fixture_array_sha256=args.fixture_array_sha256,
            fixture_manifest_sha256=args.fixture_manifest_sha256,
        )
    elif args.phase in ("calibration-batch", "heldout-batch", "metric-diagnostic"):
        if args.batch_index is None or args.batch_index < 0:
            parser.error(f"{args.phase} requires a non-negative --batch-index")
        if args.phase == "metric-diagnostic" and args.diagnostic_split is None:
            parser.error("metric-diagnostic requires --diagnostic-split")

        report = _run(
            args.phase,
            data_dir=args.data_dir,
            fixture_dir=args.fixture_dir,
            report_dir=args.report_dir,
            fixture_array_sha256=args.fixture_array_sha256,
            fixture_manifest_sha256=args.fixture_manifest_sha256,
            batch_index=args.batch_index,
            diagnostic_split=args.diagnostic_split,
        )
    else:
        raise ValueError(f"unsupported Crello parity phase {args.phase!r}")

    print(
        json.dumps({"phase": report["phase"], "report": str(args.report_dir)}, indent=1)
    )


if __name__ == "__main__":
    main()
