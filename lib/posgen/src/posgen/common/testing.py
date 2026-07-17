"""Testing helpers reserved for future position-generation packages."""

import torch

from .content import PositionContent


def assert_position_content_schema(content: PositionContent) -> None:
    """Check the minimal position-generation content schema."""

    assert content.positions.shape[:-1] == content.mask.shape
    assert content.positions.shape[-1] == 2
    assert content.positions.dtype.is_floating_point
    assert content.mask.dtype == torch.bool
