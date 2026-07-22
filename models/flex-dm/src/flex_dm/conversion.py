"""Tensor conversion helpers for TensorFlow-to-PyTorch Flex-DM checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FlexDmConversionReport:
    """Summary of a semantic checkpoint conversion."""

    matched_tensor_count: int
    matched_parameter_count: int
    missing_target_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]


def convert_dense_kernel(tf_kernel: np.ndarray) -> torch.Tensor:
    """Transpose a TensorFlow Dense kernel into PyTorch Linear layout."""
    return torch.from_numpy(tf_kernel.T)


def convert_dense_bias(tf_bias: np.ndarray) -> torch.Tensor:
    """Convert a TensorFlow Dense bias without transposition."""
    return torch.from_numpy(tf_bias)


def convert_embedding(tf_embedding: np.ndarray) -> torch.Tensor:
    """Convert a TensorFlow embedding table without transposition."""
    return torch.from_numpy(tf_embedding)


def convert_layer_norm_gamma_beta(
    gamma: np.ndarray,
    beta: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert TensorFlow LayerNorm gamma/beta to PyTorch weight/bias."""
    return torch.from_numpy(gamma), torch.from_numpy(beta)


def map_tensor_by_rule(
    source_name: str, value: np.ndarray
) -> tuple[str, torch.Tensor] | None:
    """Map one known vendor variable name to a PyTorch state-dict key.

    Args:
        source_name: TensorFlow variable name.
        value: TensorFlow variable value.

    Returns:
        Target state-dict key and converted tensor, or ``None`` when the source
        name is not part of the current semantic mapping.
    """
    name = (
        source_name.removesuffix(":0")
        .removesuffix("/.ATTRIBUTES/VARIABLE_VALUE")
        .removesuffix("/VARIABLE_VALUE")
    )
    if name.startswith("optimizer/") or "/.OPTIMIZER_SLOT/" in source_name:
        return None
    if name.startswith("model/encoder/input_layer/"):
        rest = name.removeprefix("model/encoder/input_layer/")
        if rest.endswith("/embeddings"):
            key = rest.removesuffix("/embeddings")
            if key.endswith("_special"):
                field = key.removesuffix("_special")
                return (
                    f"encoder.special_embeddings.field_{field}.weight",
                    convert_embedding(value),
                )
            return f"encoder.input_embeddings.field_{key}.weight", convert_embedding(
                value
            )
        if rest.endswith("/kernel"):
            key = rest.removesuffix("/kernel")
            return (
                f"encoder.input_projections.field_{key}.weight",
                convert_dense_kernel(value),
            )
        if rest.endswith("/bias"):
            key = rest.removesuffix("/bias")
            return (
                f"encoder.input_projections.field_{key}.bias",
                convert_dense_bias(value),
            )
    if name.startswith("model/decoder/decoders/"):
        rest = name.removeprefix("model/decoder/decoders/")
        if rest.endswith("/kernel"):
            key = rest.removesuffix("/kernel")
            return f"decoder.heads.field_{key}.weight", convert_dense_kernel(value)
        if rest.endswith("/bias"):
            key = rest.removesuffix("/bias")
            return f"decoder.heads.field_{key}.bias", convert_dense_bias(value)
    if name.startswith("model/blocks/seq2seq/seq2seq_"):
        rest = name.removeprefix("model/blocks/seq2seq/seq2seq_")
        block_text, layer = rest.split("/", 1)
        block = int(block_text)
        prefixes = {
            "attn/dense_query": f"blocks.{block}.attention.q_proj",
            "attn/dense_key": f"blocks.{block}.attention.k_proj",
            "attn/dense_value": f"blocks.{block}.attention.v_proj",
            "attn/combine_heads": f"blocks.{block}.attention.out_proj",
            "mlp/layer_with_weights-0": f"blocks.{block}.mlp.0",
            "mlp/layer_with_weights-1": f"blocks.{block}.mlp.2",
            "norm1": f"blocks.{block}.norm1",
            "norm2": f"blocks.{block}.norm2",
        }
        for source_prefix, target_prefix in prefixes.items():
            if layer == f"{source_prefix}/kernel":
                return f"{target_prefix}.weight", convert_dense_kernel(value)
            if layer == f"{source_prefix}/bias":
                return f"{target_prefix}.bias", convert_dense_bias(value)
            if layer == f"{source_prefix}/gamma":
                return f"{target_prefix}.weight", torch.from_numpy(value)
            if layer == f"{source_prefix}/beta":
                return f"{target_prefix}.bias", torch.from_numpy(value)
    return None


def conversion_report(
    *,
    converted: dict[str, torch.Tensor],
    target_keys: set[str],
    source_keys: set[str],
    consumed_source_keys: set[str],
) -> FlexDmConversionReport:
    """Build a deterministic conversion summary."""
    return FlexDmConversionReport(
        matched_tensor_count=len(converted),
        matched_parameter_count=sum(tensor.numel() for tensor in converted.values()),
        missing_target_keys=tuple(sorted(target_keys - converted.keys())),
        unexpected_source_keys=tuple(sorted(source_keys - consumed_source_keys)),
    )
