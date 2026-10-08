"""TensorFlow-to-PyTorch weight mapping for PixelVAE checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
import hashlib
import json
from typing import Final, Protocol, assert_never, cast

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from .modeling_pixel_vae import (
    MobileNetLayerKind,
    PixelVAEBatchNorm2d,
    PixelVAEModel,
)


class WeightLayout(StrEnum):
    """Array layouts handled by the converter."""

    direct = auto()
    dense = auto()
    conv2d = auto()
    depthwise = auto()
    conv2d_transpose = auto()


@dataclass(frozen=True)
class WeightAssignment:
    """One source variable to target parameter mapping."""

    source_key: str
    target_key: str
    layout: WeightLayout


@dataclass(frozen=True)
class ConversionReport:
    """Summary of a strict TensorFlow weight conversion."""

    assigned_tensors: int
    unexpected_source_keys: tuple[str, ...]
    missing_source_keys: tuple[str, ...]


class KerasLayer(Protocol):
    """Small structural interface used to extract original layer arrays."""

    name: str
    weights: Sequence[KerasVariable]

    def get_weights(self) -> list[Float[np.ndarray, "..."]]:
        """Return the layer's weights in Keras order."""


class KerasEncoderHead(Protocol):
    """Variational dense layers in the reference encoder."""

    z_mean: KerasLayer
    z_log_sigma: KerasLayer


class KerasEncoder(Protocol):
    """Encoder branch in the original PixelVAE model."""

    cnn: KerasLayerSequence
    head: KerasEncoderHead


class KerasDecoder(Protocol):
    """Decoder branch in the original PixelVAE model."""

    dense: KerasLayer
    cnn: KerasLayerSequence


class KerasLayerSequence(Protocol):
    """Ordered Keras layer collection."""

    layers: Sequence[KerasLayer]


class KerasPixelVAE(Protocol):
    """Original model attributes required for extracting weight arrays."""

    encoder: KerasEncoder
    decoder: KerasDecoder


class KerasVariable(Protocol):
    """TensorFlow variable values needed by the parity trace adapter."""

    def numpy(self) -> Float[np.ndarray, "..."]:
        """Return the current variable value as a NumPy array."""


KERAS_VARIABLES: Final[dict[str, tuple[str, ...]]] = {
    "Dense": ("kernel", "bias"),
    "Conv2D": ("kernel", "bias"),
    "DepthwiseConv2D": ("depthwise_kernel", "bias"),
    "Conv2DTranspose": ("kernel", "bias"),
    "BatchNormalization": (
        "gamma",
        "beta",
        "moving_mean",
        "moving_variance",
    ),
}


def extract_tensorflow_weights(
    reference: KerasPixelVAE,
) -> dict[str, Float[np.ndarray, "..."]]:
    """Collect named arrays from the original Keras PixelVAE model.

    Args:
        reference: Built TensorFlow model with initialized or loaded weights.

    Returns:
        A mapping with stable encoder, posterior-head, and decoder source keys.

    Raises:
        ValueError: If a source layer has an unsupported weight count.
    """
    weights: dict[str, Float[np.ndarray, "..."]] = {}
    for layer in reference.encoder.cnn.layers:
        _collect_layer(weights, f"backbone/{layer.name}", layer)

    _collect_layer(weights, "encoder/head/z_mean", reference.encoder.head.z_mean)
    _collect_layer(
        weights, "encoder/head/z_log_sigma", reference.encoder.head.z_log_sigma
    )
    _collect_layer(weights, "decoder/dense", reference.decoder.dense)
    for index, layer in enumerate(reference.decoder.cnn.layers):
        _collect_layer(weights, f"decoder/cnn/{index}", layer)

    return weights


def extract_tensorflow_variable_map(
    reference: KerasPixelVAE,
) -> dict[str, KerasVariable]:
    """Map stable source keys to the original model's variable objects.

    Args:
        reference: Built TensorFlow model with initialized or loaded weights.

    Returns:
        Variable objects keyed by the same names as
        :func:`extract_tensorflow_weights`.

    Raises:
        ValueError: If a source layer has an unsupported weight count.
    """
    variables: dict[str, KerasVariable] = {}
    for layer in reference.encoder.cnn.layers:
        _collect_layer_variables(variables, f"backbone/{layer.name}", layer)

    _collect_layer_variables(
        variables, "encoder/head/z_mean", reference.encoder.head.z_mean
    )
    _collect_layer_variables(
        variables, "encoder/head/z_log_sigma", reference.encoder.head.z_log_sigma
    )
    _collect_layer_variables(variables, "decoder/dense", reference.decoder.dense)
    for index, layer in enumerate(reference.decoder.cnn.layers):
        _collect_layer_variables(variables, f"decoder/cnn/{index}", layer)

    return variables


def tensorflow_state_sha256(
    weights: Mapping[str, Float[np.ndarray, "..."]],
) -> str:
    """Hash a named TensorFlow state independently of mapping insertion order.

    Args:
        weights: Named Keras arrays.

    Returns:
        SHA-256 digest over each sorted key, shape, dtype, and contiguous bytes.
    """
    digest = hashlib.sha256()
    for key, array in sorted(weights.items()):
        contiguous = np.ascontiguousarray(array)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())

    return digest.hexdigest()


def tensorflow_weight_map(model: PixelVAEModel) -> tuple[WeightAssignment, ...]:
    """Build the complete source-to-target map for a PixelVAE instance.

    Args:
        model: Target model whose layer shapes define expected parameters.

    Returns:
        Ordered assignments for every convolution, normalization, and dense
        parameter in the source model.
    """
    assignments: list[WeightAssignment] = []
    for name, kind in model.encoder.backbone.keras_layer_kinds.items():
        module = model.encoder.backbone.layers[name]
        source = f"backbone/{name}"
        target = f"encoder.backbone.layers.{name}"
        if kind is MobileNetLayerKind.batch_norm:
            assignments.extend(_batch_norm_assignments(source, target))
        elif kind is MobileNetLayerKind.depthwise:
            assignments.extend(
                _conv_assignments(source, target, WeightLayout.depthwise, bias=False)
            )
        elif kind is MobileNetLayerKind.conv:
            assignments.extend(
                _conv_assignments(
                    source, target, WeightLayout.conv2d, bias=module.bias is not None
                )
            )
        else:
            raise ValueError(f"unsupported MobileNetV2 layer kind: {kind}")

    assignments.extend(_dense_assignments("encoder/head/z_mean", "encoder.z_mean"))
    assignments.extend(
        _dense_assignments("encoder/head/z_log_sigma", "encoder.z_log_variance")
    )
    assignments.extend(_dense_assignments("decoder/dense", "decoder.dense"))

    decoder_modules: tuple[tuple[str, nn.Module], ...] = (
        ("bn0", model.decoder.batch_norms[0]),
        ("transpose_convolutions.0", model.decoder.transpose_convolutions[0]),
        ("bn1", model.decoder.batch_norms[1]),
        ("transpose_convolutions.1", model.decoder.transpose_convolutions[1]),
        ("bn2", model.decoder.batch_norms[2]),
        ("transpose_convolutions.2", model.decoder.transpose_convolutions[2]),
        ("bn3", model.decoder.batch_norms[3]),
        ("transpose_convolutions.3", model.decoder.transpose_convolutions[3]),
        ("bn4", model.decoder.batch_norms[4]),
        ("transpose_convolutions.4", model.decoder.transpose_convolutions[4]),
        ("bn5", model.decoder.batch_norms[5]),
        ("output_convolution", model.decoder.output_convolution),
    )
    for index, (target_module, module) in enumerate(decoder_modules):
        source = f"decoder/cnn/{index}"
        target = f"decoder.{target_module}"
        if isinstance(module, PixelVAEBatchNorm2d):
            assignments.extend(_batch_norm_assignments(source, target))
        elif isinstance(module, nn.ConvTranspose2d):
            assignments.extend(
                _conv_assignments(
                    source,
                    target,
                    WeightLayout.conv2d_transpose,
                    bias=module.bias is not None,
                )
            )
        else:
            assignments.extend(
                _conv_assignments(
                    source, target, WeightLayout.conv2d, bias=module.bias is not None
                )
            )

    return tuple(assignments)


def convert_tensorflow_variables(
    weights: Mapping[str, Float[np.ndarray, "..."]],
    model: PixelVAEModel,
    *,
    strict: bool = True,
) -> ConversionReport:
    """Copy original Keras arrays into a PyTorch PixelVAE model.

    Args:
        weights: Source arrays from :func:`extract_tensorflow_weights`.
        model: Target model to update.
        strict: Require every source and target parameter to have a mapping.

    Returns:
        Tensor count and any missing or unexpected source keys.

    Raises:
        ValueError: If strict mapping, expected keys, or tensor shapes differ.
    """
    converted, report = tensorflow_arrays_to_torch(weights, model, strict=strict)
    state = model.state_dict()
    with torch.no_grad():
        for target_key, value in converted.items():
            state[target_key].copy_(value)

    return report


def tensorflow_arrays_to_torch(
    weights: Mapping[str, Float[np.ndarray, "..."]],
    model: PixelVAEModel,
    *,
    strict: bool = False,
) -> tuple[dict[str, Float[torch.Tensor, "..."]], ConversionReport]:
    """Transform a subset or full mapping without mutating the target model.

    Args:
        weights: Named TensorFlow arrays such as weights or gradients.
        model: Target model whose shape and layout map is authoritative.
        strict: Require every expected source key and reject unexpected keys.

    Returns:
        Transformed values keyed by PyTorch state-dict name, plus a mapping
        report.

    Raises:
        ValueError: If strict keys differ or a mapped array has the wrong shape.
    """
    assignments = tensorflow_weight_map(model)
    assignment_targets = [assignment.target_key for assignment in assignments]
    state_keys = set(model.state_dict())
    if len(assignment_targets) != len(set(assignment_targets)):
        raise ValueError(
            "TensorFlow weight map assigns a PyTorch state key more than once"
        )

    if strict and set(assignment_targets) != state_keys:
        raise ValueError(
            "TensorFlow weight map does not cover the complete PyTorch state: "
            f"missing={tuple(sorted(state_keys - set(assignment_targets)))}, "
            f"unexpected={tuple(sorted(set(assignment_targets) - state_keys))}"
        )

    expected_keys = {assignment.source_key for assignment in assignments}
    received_keys = set(weights)
    missing = tuple(sorted(expected_keys - received_keys))
    unexpected = tuple(sorted(received_keys - expected_keys))
    if strict and (missing or unexpected):
        raise ValueError(
            f"TensorFlow weight keys differ; missing={missing}, unexpected={unexpected}"
        )

    state = model.state_dict()
    converted: dict[str, Float[torch.Tensor, "..."]] = {}
    for assignment in assignments:
        if assignment.source_key not in weights:
            continue

        source = torch.from_numpy(np.asarray(weights[assignment.source_key]))
        target = _convert_layout(source, assignment.layout)
        target_tensor = state[assignment.target_key]
        if target.shape != target_tensor.shape:
            raise ValueError(
                f"shape mismatch for {assignment.source_key} -> "
                f"{assignment.target_key}: {tuple(target.shape)} != "
                f"{tuple(target_tensor.shape)}"
            )

        converted[assignment.target_key] = target.to(dtype=target_tensor.dtype)

    return converted, ConversionReport(len(converted), unexpected, missing)


def _collect_layer(
    weights: dict[str, Float[np.ndarray, "..."]], key: str, layer: KerasLayer
) -> None:
    names = KERAS_VARIABLES.get(type(layer).__name__)
    if names is None:
        if layer.get_weights():
            raise ValueError(
                f"unsupported weighted Keras layer: {type(layer).__name__}"
            )

        return

    arrays = layer.get_weights()
    if len(arrays) not in (len(names), len(names) - 1):
        raise ValueError(f"unexpected weight count for {key}: {len(arrays)} weights")

    selected_names = names if len(arrays) == len(names) else names[:-1]
    weights.update(
        (f"{key}/{name}", cast(Float[np.ndarray, "..."], np.asarray(array)))
        for name, array in zip(selected_names, arrays, strict=True)
    )


def _collect_layer_variables(
    variables: dict[str, KerasVariable], key: str, layer: KerasLayer
) -> None:
    names = KERAS_VARIABLES.get(type(layer).__name__)
    if names is None:
        if layer.weights:
            raise ValueError(
                f"unsupported weighted Keras layer: {type(layer).__name__}"
            )

        return

    if len(layer.weights) not in (len(names), len(names) - 1):
        raise ValueError(
            f"unexpected weight count for {key}: {len(layer.weights)} weights"
        )

    selected_names = names if len(layer.weights) == len(names) else names[:-1]
    variables.update(
        (f"{key}/{name}", variable)
        for name, variable in zip(selected_names, layer.weights, strict=True)
    )


def _batch_norm_assignments(source: str, target: str) -> list[WeightAssignment]:
    return [
        WeightAssignment(
            f"{source}/{source_name}", f"{target}.{target_name}", WeightLayout.direct
        )
        for source_name, target_name in (
            ("gamma", "weight"),
            ("beta", "bias"),
            ("moving_mean", "running_mean"),
            ("moving_variance", "running_var"),
        )
    ]


def _dense_assignments(source: str, target: str) -> list[WeightAssignment]:
    return [
        WeightAssignment(f"{source}/kernel", f"{target}.weight", WeightLayout.dense),
        WeightAssignment(f"{source}/bias", f"{target}.bias", WeightLayout.direct),
    ]


def _conv_assignments(
    source: str, target: str, layout: WeightLayout, *, bias: bool
) -> list[WeightAssignment]:
    assignments = [
        WeightAssignment(
            f"{source}/depthwise_kernel"
            if layout is WeightLayout.depthwise
            else f"{source}/kernel",
            f"{target}.weight",
            layout,
        )
    ]
    if bias:
        assignments.append(
            WeightAssignment(f"{source}/bias", f"{target}.bias", WeightLayout.direct)
        )

    return assignments


def _convert_layout(
    source: Float[torch.Tensor, "..."], layout: WeightLayout
) -> Float[torch.Tensor, "..."]:
    if layout is WeightLayout.direct:
        return source

    if layout is WeightLayout.dense:
        return source.T.contiguous()

    if layout is WeightLayout.conv2d:
        return source.permute(3, 2, 0, 1).contiguous()

    if layout is WeightLayout.depthwise:
        return source.permute(2, 3, 0, 1).contiguous()

    if layout is WeightLayout.conv2d_transpose:
        return source.permute(3, 2, 0, 1).contiguous()

    assert_never(layout)
