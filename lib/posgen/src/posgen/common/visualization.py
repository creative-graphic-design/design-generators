"""Visualization placeholders for future position-generation outputs."""

from .content import PositionContent


def render_position_summary(content: PositionContent) -> str:
    """Return a compact text summary for a position-generation sample.

    Args:
        content: Position content object to summarize.

    Returns:
        Text summary of active positions.

    Raises:
        ValueError: This function does not raise directly.

    Examples:
        >>> import torch
        >>> content = PositionContent(
        ...     positions=torch.zeros(1, 2, 2),
        ...     mask=torch.tensor([[True, False]]),
        ... )
        >>> render_position_summary(content)
        '1 active positions'
    """

    return f"{int(content.mask.sum().item())} active positions"
