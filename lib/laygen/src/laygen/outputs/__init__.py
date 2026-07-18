"""Shared output schemas for layout-generation packages."""

from ._spec import (
    LAYOUT_GENERATION_OUTPUT_FIELDS,
    OutputField,
    OutputFieldSpec,
    dataclass_fields,
)
from .diffusers import LayoutGenerationOutput as DiffusersLayoutGenerationOutput
from .transformers import LayoutGenerationOutput as TransformersLayoutGenerationOutput

__all__ = [
    "LAYOUT_GENERATION_OUTPUT_FIELDS",
    "DiffusersLayoutGenerationOutput",
    "OutputField",
    "OutputFieldSpec",
    "TransformersLayoutGenerationOutput",
    "dataclass_fields",
]
