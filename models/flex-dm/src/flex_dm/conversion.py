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
    name = source_name.removesuffix(":0")
    if "/kernel" in name:
        return name.replace("/", ".").removesuffix(
            ".kernel"
        ) + ".weight", convert_dense_kernel(value)
    if "/bias" in name or "/beta" in name:
        target = name.replace("/", ".").replace(".beta", ".bias")
        return target.removesuffix(".bias") + ".bias", convert_dense_bias(value)
    if "/gamma" in name:
        return name.replace("/", ".").removesuffix(
            ".gamma"
        ) + ".weight", torch.from_numpy(value)
    if name.endswith("/embeddings"):
        return name.replace("/", ".").removesuffix(
            ".embeddings"
        ) + ".weight", convert_embedding(value)
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
