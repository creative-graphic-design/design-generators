"""Testing helpers reserved for future position-generation packages."""

import torch

from .content import PositionContent


def assert_position_content_schema(content: PositionContent) -> None:
    """Assert the minimal position-generation content schema.

    Args:
        content: Position content object to check.

    Returns:
        None.

    Raises:
        AssertionError: If positions and masks do not match the schema.

    Examples:
        >>> import torch
        >>> content = PositionContent(
        ...     positions=torch.zeros(1, 2, 2),
        ...     mask=torch.ones(1, 2, dtype=torch.bool),
        ... )
        >>> assert_position_content_schema(content)
    """

    assert content.positions.shape[:-1] == content.mask.shape
    assert content.positions.shape[-1] == 2
    assert content.positions.dtype.is_floating_point
    assert content.mask.dtype == torch.bool
