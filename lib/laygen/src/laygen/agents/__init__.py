"""Shared agent building blocks for text-conditioned layout generators.

Importing this module requires the optional ``laygen[agents]`` extra because
provider execution is delegated to Pydantic AI. The rest of ``laygen`` remains
usable without that extra.
"""

from .core import (
    BaseLayoutAgent,
    ExemplarSelector,
    LayoutItem2DLike,
    ModelLike,
    PromptBuilder,
    ResponseParser,
    layout_items_to_output,
    messages_to_text,
)

__all__ = [
    "BaseLayoutAgent",
    "ExemplarSelector",
    "LayoutItem2DLike",
    "ModelLike",
    "PromptBuilder",
    "ResponseParser",
    "layout_items_to_output",
    "messages_to_text",
]
