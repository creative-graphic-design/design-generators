"""Visualization placeholders for future position-generation outputs."""

from .content import PositionContent


def render_position_summary(content: PositionContent) -> str:
    """Return a compact text summary for a position-generation sample."""
    return f"{int(content.mask.sum().item())} active positions"
