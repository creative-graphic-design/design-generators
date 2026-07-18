"""Typed LayoutGPT request and response schemas."""

from typing import Final

from laygen.agents import layout_items_to_output
from laygen.modeling_outputs import LayoutGenerationOutput
from pydantic import BaseModel, ConfigDict, Field, computed_field

from layout_gpt.enums import ICLType, LayoutGPTSetting
from layout_gpt.types import ChatMessage, LayoutGPTIntermediates

DEFAULT_K: Final[int] = 8
DEFAULT_CANVAS_SIZE: Final[int] = 256
DEFAULT_GPT_INPUT_LENGTH_LIMIT: Final[int] = 3000
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 1.0
DEFAULT_N_ITER: Final[int] = 1
DEFAULT_FIXED_RANDOM_SEED: Final[int] = 42


class LayoutItem2D(BaseModel):
    """A parsed 2D CSS layout item."""

    label: str
    left: float = Field(ge=0.0)
    top: float = Field(ge=0.0)
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def bbox_ltrb(self) -> tuple[float, float, float, float]:
        """Normalized ``left, top, right, bottom`` box."""
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @computed_field
    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """Normalized center ``xywh`` box."""
        return (
            self.left + self.width / 2,
            self.top + self.height / 2,
            self.width,
            self.height,
        )


class LayoutItem3D(BaseModel):
    """A parsed 3D CSS layout item."""

    label: str
    length: float
    width: float
    height: float
    orientation: float
    left: float
    top: float
    depth: float

    model_config = ConfigDict(frozen=True)


class RawLayoutResponse(BaseModel):
    """Structured model response before CSS parsing."""

    text: str


class LayoutGPTOutput(BaseModel):
    """Pydantic representation of a parsed LayoutGPT response."""

    prompt: str
    canvas_size: int
    items: list[LayoutItem2D]
    raw_text: str
    id2label: dict[int, str]
    selected_exemplar_ids: list[str | int] = Field(default_factory=list)
    prompt_messages: list[ChatMessage] | None = None

    model_config = ConfigDict(frozen=True)

    def to_layout_generation_output(self) -> LayoutGenerationOutput:
        """Convert to the shared public layout output schema."""
        intermediates: LayoutGPTIntermediates = {
            "prompt": self.prompt,
            "raw_text": self.raw_text,
            "selected_exemplar_ids": self.selected_exemplar_ids,
            "prompt_messages": self.prompt_messages,
        }
        return layout_items_to_output(
            self.items,
            id2label=self.id2label,
            intermediates=intermediates,
        )


class LayoutGPTConfig(BaseModel):
    """Runtime configuration for LayoutGPT prompt and provider behavior."""

    setting: LayoutGPTSetting = LayoutGPTSetting.counting
    icl_type: ICLType = ICLType.k_similar
    k: int = Field(default=DEFAULT_K, ge=0)
    canvas_size: int = Field(default=DEFAULT_CANVAS_SIZE, gt=0)
    gpt_input_length_limit: int = Field(default=DEFAULT_GPT_INPUT_LENGTH_LIMIT, gt=0)
    chat: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    n_iter: int = Field(default=DEFAULT_N_ITER, ge=1)
    fixed_random_seed: int = DEFAULT_FIXED_RANDOM_SEED
