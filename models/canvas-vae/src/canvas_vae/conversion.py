"""Checkpoint conversion for CanvasVAE.

TensorFlow checkpoints written by the original Keras trainer store variables
under object-graph keys such as
``encoder/seq2seq/seq2seq_0/attn/dense_query/kernel/.ATTRIBUTES/VARIABLE_VALUE``.
This module maps those keys to package state-dict keys, transposing Keras dense
kernels, and rejects missing or unexpected model variables. Lightning
checkpoints written by the package trainer store the model under ``model.``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float, Shaped

from .configuration_canvas_vae import (
    CanvasVAECrelloConfig,
    CanvasVAEConfig,
    CanvasVAEField,
)

VARIABLE_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"
IGNORED_KEY_PARTS = ("OPTIMIZER_SLOT", "optimizer/", "save_counter", "_CHECKPOINTABLE")
LIGHTNING_MODEL_PREFIX = "model."


class TensorFlowSource(NamedTuple):
    """Source variable of one package state-dict entry."""

    key: str
    transpose: bool


def _dense(package: str, original: str) -> dict[str, TensorFlowSource]:
    return {
        f"{package}.weight": TensorFlowSource(f"{original}/kernel", True),
        f"{package}.bias": TensorFlowSource(f"{original}/bias", False),
    }


def _block(package: str, original: str) -> dict[str, TensorFlowSource]:
    mapping = {
        f"{package}.{norm}.weight": TensorFlowSource(f"{original}/{norm}/gamma", False)
        for norm in ("norm1", "norm2")
    } | {
        f"{package}.{norm}.bias": TensorFlowSource(f"{original}/{norm}/beta", False)
        for norm in ("norm1", "norm2")
    }
    projections = {
        "q_proj": "dense_query",
        "k_proj": "dense_key",
        "v_proj": "dense_value",
        "out_proj": "combine_heads",
    }
    for name, source in projections.items():
        mapping |= _dense(f"{package}.attention.{name}", f"{original}/attn/{source}")

    mapping |= _dense(f"{package}.conditional", f"{original}/conditional")
    mapping |= _dense(f"{package}.mlp.0", f"{original}/mlp/layer_with_weights-0")
    return mapping | _dense(f"{package}.mlp.2", f"{original}/mlp/layer_with_weights-1")


def tensorflow_key_map(config: CanvasVAEConfig) -> dict[str, TensorFlowSource]:
    """Return the TensorFlow source of every package state-dict entry.

    Args:
        config: Model config.

    Returns:
        Source keys without the ``/.ATTRIBUTES/VARIABLE_VALUE`` suffix.

    Examples:
        >>> config = CanvasVAEConfig(
        ...     vocabularies={"component": ["[UNK]", ""], "icon": ["[UNK]"],
        ...                   "text_button": ["[UNK]"]}
        ... )
        >>> tensorflow_key_map(config)["encoder.z_log_var.weight"]
        TensorFlowSource(key='encoder/head/z_log_sigma/kernel', transpose=True)
    """
    mapping = {
        "encoder.length_embedding.weight": TensorFlowSource(
            "encoder/input_layer/length/embeddings", False
        ),
        "encoder.position_embedding.embeddings.weight": TensorFlowSource(
            "encoder/input_layer/const/embeddings/embeddings", False
        ),
        "encoder.norm.weight": TensorFlowSource("encoder/norm/gamma", False),
        "encoder.norm.bias": TensorFlowSource("encoder/norm/beta", False),
        "encoder.norm.running_mean": TensorFlowSource(
            "encoder/norm/moving_mean", False
        ),
        "encoder.norm.running_var": TensorFlowSource(
            "encoder/norm/moving_variance", False
        ),
        "decoder.position_embedding.embeddings.weight": TensorFlowSource(
            "decoder/embedding_const/embeddings/embeddings", False
        ),
    }
    mapping |= _dense("encoder.z_mean", "encoder/head/z_mean")
    mapping |= _dense("encoder.z_log_var", "encoder/head/z_log_sigma")
    mapping |= _dense("decoder.length_head", "decoder/head/decoders/length")
    for field in CanvasVAEField:
        mapping[f"encoder.field_embeddings.{field}.weight"] = TensorFlowSource(
            f"encoder/input_layer/{field}/embeddings", False
        )
        mapping |= _dense(
            f"decoder.field_heads.{field}", f"decoder/head/decoders/{field}"
        )

    for index in range(config.num_blocks):
        for part in ("encoder", "decoder"):
            mapping |= _block(
                f"{part}.blocks.{index}", f"{part}/seq2seq/seq2seq_{index}"
            )

    return mapping


def tensorflow_crello_key_map(
    config: CanvasVAECrelloConfig,
) -> dict[str, TensorFlowSource]:
    """Return the original Keras variable for every Crello model tensor."""
    mapping = {
        "encoder.position_embedding.embeddings.weight": TensorFlowSource(
            "encoder/input_layer/const/embeddings/embeddings", False
        ),
        "encoder.norm.weight": TensorFlowSource("encoder/norm/gamma", False),
        "encoder.norm.bias": TensorFlowSource("encoder/norm/beta", False),
        "encoder.norm.running_mean": TensorFlowSource(
            "encoder/norm/moving_mean", False
        ),
        "encoder.norm.running_var": TensorFlowSource(
            "encoder/norm/moving_variance", False
        ),
        "decoder.position_embedding.embeddings.weight": TensorFlowSource(
            "decoder/embedding_const/embeddings/embeddings", False
        ),
    }
    mapping |= _dense("encoder.z_mean", "encoder/head/z_mean")
    mapping |= _dense("encoder.z_log_var", "encoder/head/z_log_sigma")
    mapping |= _dense("decoder.length_head", "decoder/head/decoders/length")
    for field in config.context_fields:
        mapping[f"encoder.context_embeddings.{field}.weight"] = TensorFlowSource(
            f"encoder/input_layer/{field}/embeddings", False
        )

    for field in config.sequence_field_sizes:
        mapping[f"encoder.sequence_embeddings.field_{field}.weight"] = TensorFlowSource(
            f"encoder/input_layer/{field}/embeddings", False
        )
        mapping |= _dense(
            f"decoder.sequence_heads.field_{field}",
            f"decoder/head/decoders/{field}",
        )

    mapping |= _dense(
        "encoder.numerical_projections.image_embedding",
        "encoder/input_layer/image_embedding",
    )
    mapping |= _dense("decoder.context_heads.group", "decoder/head/decoders/group")
    for field in config.context_fields[2:]:
        mapping |= _dense(
            f"decoder.context_heads.{field}", f"decoder/head/decoders/{field}"
        )

    mapping |= _dense(
        "decoder.numerical_heads.image_embedding",
        "decoder/head/decoders/image_embedding",
    )
    for index in range(config.num_blocks):
        for part in ("encoder", "decoder"):
            mapping |= _block(
                f"{part}.blocks.{index}", f"{part}/seq2seq/seq2seq_{index}"
            )

    return mapping


def convert_tensorflow_variables(
    variables: Mapping[str, Shaped[np.ndarray, ...]],
    config: CanvasVAEConfig,
) -> dict[str, Float[torch.Tensor, ...]]:
    """Convert TensorFlow checkpoint variables into a package state dict.

    Args:
        variables: Checkpoint variables keyed by object-graph key, with or
            without the ``/.ATTRIBUTES/VARIABLE_VALUE`` suffix. Optimizer and
            bookkeeping entries are ignored.
        config: Model config.

    Returns:
        Package state dict.

    Raises:
        KeyError: If model variables are missing or unexpected.
    """
    return _convert_variables(variables, tensorflow_key_map(config))


def convert_tensorflow_crello_variables(
    variables: Mapping[str, Shaped[np.ndarray, ...]],
    config: CanvasVAECrelloConfig,
) -> dict[str, Float[torch.Tensor, ...]]:
    """Map original TensorFlow variables into a Crello model state dict."""
    return _convert_variables(variables, tensorflow_crello_key_map(config))


def _convert_variables(
    variables: Mapping[str, Shaped[np.ndarray, ...]],
    mapping: Mapping[str, TensorFlowSource],
) -> dict[str, Float[torch.Tensor, ...]]:
    model_variables = {
        key.removesuffix(VARIABLE_SUFFIX): value
        for key, value in variables.items()
        if not any(part in key for part in IGNORED_KEY_PARTS)
    }
    expected = {source.key for source in mapping.values()}
    missing = sorted(expected - model_variables.keys())
    unexpected = sorted(model_variables.keys() - expected)
    if missing:
        raise KeyError(f"missing TensorFlow variables: {missing}")

    if unexpected:
        raise KeyError(f"unexpected TensorFlow variables: {unexpected}")

    state_dict = {}
    for key, source in mapping.items():
        value = np.asarray(model_variables[source.key], dtype=np.float32)
        state_dict[key] = torch.from_numpy(
            np.ascontiguousarray(value.T if source.transpose else value)
        )

    return state_dict


def load_lightning_state_dict(
    checkpoint: str | Path,
) -> dict[str, Float[torch.Tensor, ...]]:
    """Load the model state dict from a package Lightning checkpoint.

    Args:
        checkpoint: Lightning ``.ckpt`` file.

    Returns:
        Model state dict without the ``model.`` prefix.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        key.removeprefix(LIGHTNING_MODEL_PREFIX): value
        for key, value in payload["state_dict"].items()
        if key.startswith(LIGHTNING_MODEL_PREFIX)
    }


__all__ = [
    "TensorFlowSource",
    "convert_tensorflow_crello_variables",
    "convert_tensorflow_variables",
    "load_lightning_state_dict",
    "tensorflow_crello_key_map",
    "tensorflow_key_map",
]
