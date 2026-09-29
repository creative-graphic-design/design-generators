"""Generic Lightning training-step logging and reduction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

tensor_type = torch.Tensor


class ScalarLogger(Protocol):
    """Minimal scalar logging protocol implemented by Lightning modules."""

    def log(
        self,
        name: str,
        value: tensor_type,
        *,
        prog_bar: bool = False,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Log a scalar training metric.

        Args:
            name: Metric name.
            value: Scalar tensor value.
            prog_bar: Whether to show the value in the progress bar.
            on_step: Whether to aggregate the value per step.
            on_epoch: Whether to aggregate the value per epoch.
            batch_size: Batch size used by Lightning metric aggregation.
        """
        _ = (name, value, prog_bar, on_step, on_epoch, batch_size)


def sum_loss_values(losses: Mapping[str, tensor_type]) -> tensor_type:
    """Sum scalar loss values with the canonical training reduction.

    Args:
        losses: Mapping from metric names to scalar loss tensors.

    Returns:
        Scalar tensor containing the sum of all loss values.

    Examples:
        >>> import torch
        >>> sum_loss_values({"a": torch.tensor(1.0), "b": torch.tensor(2.0)})
        tensor(3.)
    """
    return torch.stack(tuple(losses.values())).sum()


def log_training_losses(
    logger: ScalarLogger,
    losses: Mapping[str, tensor_type],
    total: tensor_type,
    *,
    batch_size: int = 1,
) -> None:
    """Log per-component and total training losses.

    Args:
        logger: Object exposing Lightning-compatible ``log``.
        losses: Per-component scalar loss values.
        total: Total scalar training loss.
        batch_size: Batch size used by Lightning metric aggregation.

    Returns:
        None.
    """
    for key, value in losses.items():
        logger.log(key, value, on_step=True, on_epoch=True, batch_size=batch_size)

    logger.log(
        "train_loss",
        total,
        prog_bar=True,
        on_step=True,
        on_epoch=True,
        batch_size=batch_size,
    )


def finish_training_step(
    logger: ScalarLogger,
    losses: Mapping[str, tensor_type],
    trace: Mapping[str, tensor_type],
    *,
    batch_size: int = 1,
) -> tuple[tensor_type, dict[str, tensor_type]]:
    """Reduce losses, log training metrics, and append ``train_loss`` to a trace.

    Args:
        logger: Object exposing Lightning-compatible ``log``.
        losses: Per-component scalar loss values.
        trace: Training trace entries produced before the optimizer step.
        batch_size: Batch size used by Lightning metric aggregation.

    Returns:
        Total scalar loss and an updated detached trace mapping.
    """
    total = sum_loss_values(losses)
    log_training_losses(logger, losses, total, batch_size=batch_size)
    return total, {**trace, "train_loss": total.detach()}


def log_validation_loss(
    logger: ScalarLogger,
    total: tensor_type,
    *,
    batch_size: int = 1,
) -> None:
    """Log the canonical validation loss metric.

    Args:
        logger: Object exposing Lightning-compatible ``log``.
        total: Scalar validation loss.
        batch_size: Batch size used by Lightning metric aggregation.

    Returns:
        None.
    """
    logger.log(
        "val_loss",
        total,
        prog_bar=True,
        on_step=False,
        on_epoch=True,
        batch_size=batch_size,
    )
