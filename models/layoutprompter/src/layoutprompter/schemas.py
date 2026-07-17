"""Pydantic schemas for LayoutPrompter structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PixelBBox(BaseModel):
    """Top-left pixel ``xywh`` bbox emitted by the language model."""

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class LayoutElement(BaseModel):
    """One predicted layout element."""

    label: str
    bbox: PixelBBox

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        """Normalize model-produced labels for dataset lookup."""
        return value.strip().lower()


class LayoutPrompterOutput(BaseModel):
    """Structured output requested from the Pydantic AI model."""

    elements: list[LayoutElement] = Field(default_factory=list)
