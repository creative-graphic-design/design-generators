"""Loss functions for LayoutFlow training parity."""

from __future__ import annotations

import torch
from jaxtyping import Float


def layout_flow_losses(
    cond_mask: Float[torch.Tensor, "batch elements channels"],
    ut: Float[torch.Tensor, "batch elements channels"],
    vt: Float[torch.Tensor, "batch elements channels"],
    *,
    geom_dim: int = 4,
    geom_l1_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute the LayoutFlow training losses.

    Args:
        cond_mask: Condition mask where ``1`` marks generated fields.
        ut: Target conditional vector field.
        vt: Predicted vector field.
        geom_dim: Number of geometry channels.
        geom_l1_weight: Weight applied to the geometry L1 auxiliary loss.

    Returns:
        Dictionary with ``flow_loss``, ``geom_l1_loss``, and ``train_loss``.

    Raises:
        RuntimeError: If tensor shapes are incompatible.

    Examples:
        >>> x = torch.ones(1, 2, 3)
        >>> out = layout_flow_losses(x, x, x, geom_dim=2)
        >>> out["train_loss"].item()
        0.0
    """
    flow_loss = torch.nn.functional.mse_loss(cond_mask * vt, cond_mask * ut)
    geom_l1_loss = torch.nn.functional.l1_loss(
        cond_mask[..., :geom_dim] * vt[..., :geom_dim],
        cond_mask[..., :geom_dim] * ut[..., :geom_dim],
    )
    train_loss = flow_loss + geom_l1_weight * geom_l1_loss
    return {
        "flow_loss": flow_loss,
        "geom_l1_loss": geom_l1_loss,
        "train_loss": train_loss,
    }
