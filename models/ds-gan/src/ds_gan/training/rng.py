"""Deterministic random-state helpers for DS-GAN training."""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np
import torch
from jaxtyping import Float, Int
from laygen.common.randomness import normal

from ..modeling_ds_gan import xyxy_to_xywh


class NumpyChoiceSource(Protocol):
    """NumPy random source exposing the legacy weighted choice operation."""

    def choice(
        self,
        a: int,
        *,
        size: tuple[int, int, int],
        p: Float[np.ndarray, "4"],
    ) -> Int[np.ndarray, "batch elements 1"]:
        """Draw weighted class ids."""


def explicit_setup_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set every random source used by the reference training recipe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def vendor_random_initial_layout(
    batch_size: int,
    max_elem: int,
    *,
    numpy_rng: NumpyChoiceSource,
    torch_generator: torch.Generator | None,
    device: torch.device | str = "cpu",
) -> Float[torch.Tensor, "batch elements 2 4"]:
    """Reproduce the reference class and box draws with explicit RNG objects."""
    class_ids = numpy_rng.choice(
        4,
        size=(batch_size, max_elem, 1),
        p=np.asarray((0.1, 0.8, 1.0, 1.0), dtype=np.float64) / 2.9,
    )
    class_tensor = torch.zeros(
        batch_size, max_elem, 4, device=device, dtype=torch.float32
    )
    class_tensor.scatter_(
        -1,
        torch.as_tensor(class_ids, device=device, dtype=torch.long),
        1.0,
    )
    box_xyxy = normal(
        0.5,
        0.15,
        size=(batch_size, max_elem, 1, 4),
        generator=torch_generator,
        device=device,
        dtype=torch.float32,
    )
    return torch.cat((class_tensor.unsqueeze(2), xyxy_to_xywh(box_xyxy)), dim=2)
