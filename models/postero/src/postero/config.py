"""Configuration for PosterO prompt construction and parsing."""

from __future__ import annotations

from typing import Final

from posgen.common import DatasetName, normalize_dataset_name
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from postero.enums import (
    PosterOInjection,
    PosterOPoolStrategy,
    PosterORankStrategy,
    PosterOStructure,
    PromptTarget,
    coerce_enum,
)

DEFAULT_CANVAS_SIZE: Final[tuple[int, int]] = (513, 750)
PKU_ID2LABEL: Final[dict[int, str]] = {1: "text", 2: "logo", 3: "underlay"}
CGL_ID2LABEL: Final[dict[int, str]] = {
    1: "logo",
    2: "text",
    3: "underlay",
    4: "embellishment",
}


class PosterOConfig(BaseModel):
    """Runtime prompt, retrieval, parser, and provider settings.

    Args:
        dataset_name: Poster dataset name. Public aliases are normalized through
            ``posgen.common``.
        structure: Prompt serialization structure.
        injection: Available-area injection strategy.
        pool_strategy: Candidate pool strategy.
        rank_strategy: Exemplar ranking strategy.
        sample_size: Number of candidate records considered before ranking.
        num_return: Maximum provider attempts for one public call.
        n_valid_layouts: Number of valid parsed layouts requested.
        canvas_size: Canvas size as ``(width, height)``.
        temperature: Provider sampling temperature.
        top_p: Provider nucleus sampling value.
        max_tokens: Maximum response tokens requested from a provider.
        frequency_penalty: Provider frequency penalty.
        presence_penalty: Provider presence penalty.
        stop_token: Provider stop token.
        label_rback: Whether parser labels are mapped back to configured ids.
        prompt_target: Structured response target.
        id2label: Public label ids. Defaults preserve PosterO issue mappings.

    Raises:
        ValueError: If a closed mode or dataset name is unsupported.

    Examples:
        >>> config = PosterOConfig(dataset_name="pku")
        >>> config.id2label[1]
        'text'
    """

    dataset_name: DatasetName | str = DatasetName.pku_posterlayout
    structure: PosterOStructure | str = PosterOStructure.hierarchical
    injection: PosterOInjection | str = PosterOInjection.top
    pool_strategy: PosterOPoolStrategy | str = PosterOPoolStrategy.metric_filter
    rank_strategy: PosterORankStrategy | str = PosterORankStrategy.rank_by_feature
    sample_size: int = Field(default=10, ge=1)
    num_return: int = Field(default=10, ge=1)
    n_valid_layouts: int = Field(default=10, ge=1)
    canvas_size: tuple[int, int] = DEFAULT_CANVAS_SIZE
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 800
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_token: str = "\n\n"
    label_rback: bool = True
    prompt_target: PromptTarget | str = PromptTarget.rect_only
    id2label: dict[int, str] | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("dataset_name")
    @classmethod
    def normalize_dataset(cls, value: DatasetName | str) -> DatasetName:
        """Normalize poster dataset aliases at the config boundary."""
        return normalize_dataset_name(value)

    @field_validator("structure")
    @classmethod
    def normalize_structure(cls, value: PosterOStructure | str) -> PosterOStructure:
        """Normalize prompt structure."""
        return coerce_enum(value, PosterOStructure)

    @field_validator("injection")
    @classmethod
    def normalize_injection(cls, value: PosterOInjection | str) -> PosterOInjection:
        """Normalize available-area injection mode."""
        return coerce_enum(value, PosterOInjection)

    @field_validator("pool_strategy")
    @classmethod
    def normalize_pool_strategy(
        cls, value: PosterOPoolStrategy | str
    ) -> PosterOPoolStrategy:
        """Normalize pool strategy."""
        return coerce_enum(value, PosterOPoolStrategy)

    @field_validator("rank_strategy")
    @classmethod
    def normalize_rank_strategy(
        cls, value: PosterORankStrategy | str
    ) -> PosterORankStrategy:
        """Normalize rank strategy."""
        return coerce_enum(value, PosterORankStrategy)

    @field_validator("prompt_target")
    @classmethod
    def normalize_prompt_target(cls, value: PromptTarget | str) -> PromptTarget:
        """Normalize prompt target."""
        return coerce_enum(value, PromptTarget)

    @model_validator(mode="after")
    def fill_id2label(self) -> "PosterOConfig":
        """Fill dataset-specific label ids when no explicit map is stored."""
        if self.id2label is not None:
            return self
        mapping = (
            CGL_ID2LABEL
            if self.dataset_name in {DatasetName.cgl, DatasetName.cgl_v2}
            else PKU_ID2LABEL
        )
        object.__setattr__(self, "id2label", dict(mapping))
        return self
