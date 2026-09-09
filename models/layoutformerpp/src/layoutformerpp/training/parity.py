"""Static-state comparison helpers for LayoutFormer++ S0 evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from jaxtyping import Shaped


@dataclass(frozen=True, slots=True)
class StaticStateComparison:
    """Exhaustive state-key and parameter-count comparison."""

    reference_parameter_count: int
    package_parameter_count: int
    missing_package_keys: tuple[str, ...]
    extra_package_keys: tuple[str, ...]
    reference_state_sha256: str
    package_state_sha256: str

    @property
    def passed(self) -> bool:
        """Return whether topology and copied state agree exhaustively."""
        return (
            self.reference_parameter_count == self.package_parameter_count
            and not self.missing_package_keys
            and not self.extra_package_keys
            and self.reference_state_sha256 == self.package_state_sha256
        )


def parameter_count(module: torch.nn.Module) -> int:
    """Count unique parameter storage exposed by a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def state_dict_sha256(
    state_dict: Mapping[str, Shaped[torch.Tensor, "..."]],
) -> str:
    """Hash sorted state names, tensor metadata, and raw CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare_static_state(
    reference: torch.nn.Module,
    package: torch.nn.Module,
) -> StaticStateComparison:
    """Compare exhaustive keys, parameter counts, and copied tensor bytes."""
    reference_state = reference.state_dict()
    package_state = package.state_dict()
    return StaticStateComparison(
        reference_parameter_count=parameter_count(reference),
        package_parameter_count=parameter_count(package),
        missing_package_keys=tuple(sorted(set(reference_state) - set(package_state))),
        extra_package_keys=tuple(sorted(set(package_state) - set(reference_state))),
        reference_state_sha256=state_dict_sha256(reference_state),
        package_state_sha256=state_dict_sha256(package_state),
    )


__all__ = [
    "StaticStateComparison",
    "compare_static_state",
    "parameter_count",
    "state_dict_sha256",
]
