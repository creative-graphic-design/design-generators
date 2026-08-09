"""Configuration for PosterLLaVA recipe checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from typing import Final

from posgen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    normalize_dataset_name,
)
from transformers import PretrainedConfig

PosterLlavaConfigValue = (
    str | int | float | bool | None | list[str] | list[int] | dict[str, str]
)


class OutputType(StrEnum):
    """Closed output container modes supported by PosterLLaVA."""

    dataclass = auto()
    dict = auto()


class ConversationMode(StrEnum):
    """Conversation templates used by LLaVA-family checkpoints."""

    llava_v0 = auto()
    llava_v1 = auto()


DEFAULT_CHECKPOINT_ID: Final[str] = "posterllava/posterllava_v0"
DEFAULT_PROMPT_TEMPLATE: Final[str] = (
    "Hello! Could you please help me to place {num_elements} foreground "
    "elements over the background image of resolution {resolution} to craft "
    "an aesthetically pleasing, harmonious, balanced, and visually appealing "
    "{domain_name}?\n"
    "Finding semantic-meaningful objects or visual foci on the background "
    "image at first might help in designing, and you should avoid any "
    "unnecessary blocking of them.\n"
    "Please return the result by completing the following JSON file. Each "
    "element's location and size should be represented by a bounding box "
    "described as [left, top, right, bottom], and each number is a continuous "
    "digit from 0 to 1.\n"
    "Here is the initial JSON file: {initial_json}"
)


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize a public output-type value.

    Args:
        output_type: Output type enum or string value.

    Returns:
        Normalized output type.

    Raises:
        ValueError: If the value is unsupported.

    Examples:
        >>> str(normalize_output_type("dict"))
        'dict'
    """
    if isinstance(output_type, OutputType):
        return output_type
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


def normalize_conversation_mode(
    conversation_mode: ConversationMode | str,
) -> ConversationMode:
    """Normalize a LLaVA conversation mode.

    Args:
        conversation_mode: Conversation mode enum or string value.

    Returns:
        Normalized conversation mode.

    Raises:
        ValueError: If the mode is unsupported.
    """
    if isinstance(conversation_mode, ConversationMode):
        return conversation_mode
    try:
        return ConversationMode(conversation_mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported conversation mode: {conversation_mode}") from exc


class PosterLlavaConfig(PretrainedConfig):
    """Configuration saved with a PosterLLaVA recipe checkpoint.

    Args:
        checkpoint_id: Upstream LLaVA-style checkpoint id used by local smoke
            scripts and documentation.
        dataset_name: Canonical poster/content dataset metadata key.
        id2label: Persisted label metadata. Open-vocabulary generation uses a
            batch-local map at runtime, but this config records known dataset
            labels for model cards and smoke checks.
        prompt_template: Prompt body template passed through the LLaVA
            conversation wrapper.
        default_conv_mode: Default LLaVA conversation template.
        image_aspect_ratio: Image preprocessing mode; ``"pad"`` matches the
            released checkpoint.
        max_new_tokens: Default token budget for generation.
        default_temperature: Default sampled-generation temperature.
        processor_subfolder: Subfolder used by pipeline component loading.
        model_subfolder: Optional model component subfolder.
        tokenizer_subfolder: Optional tokenizer component subfolder.
        image_processor_subfolder: Optional image processor component subfolder.
        kwargs: Extra Hugging Face config fields.

    Raises:
        ValueError: If numeric fields or enum-like fields are invalid.

    Examples:
        >>> cfg = PosterLlavaConfig(dataset_name="ad_banner")
        >>> cfg.checkpoint_id
        'posterllava/posterllava_v0'
    """

    model_type = "posterllava"
    id2label: dict[int, str]

    def __init__(
        self,
        *,
        checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
        dataset_name: DatasetName | str = DatasetName.ad_banner,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        default_conv_mode: ConversationMode | str = ConversationMode.llava_v0,
        image_aspect_ratio: str = "pad",
        max_new_tokens: int = 1024,
        default_temperature: float = 0.2,
        processor_subfolder: str = "processor",
        model_subfolder: str = "model",
        tokenizer_subfolder: str = "tokenizer",
        image_processor_subfolder: str = "image_processor",
        **kwargs: PosterLlavaConfigValue,
    ) -> None:
        """Initialize PosterLLaVA recipe configuration."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        dataset = normalize_dataset_name(dataset_name)
        conv_mode = normalize_conversation_mode(default_conv_mode)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if default_temperature < 0:
            raise ValueError("default_temperature must be non-negative")
        if image_aspect_ratio != "pad":
            raise ValueError("PosterLLaVA currently supports image_aspect_ratio='pad'")
        self.checkpoint_id = checkpoint_id
        self.dataset_name = str(dataset)
        self.id2label = {
            int(key): str(value)
            for key, value in (id2label or id2label_for_dataset(dataset)).items()
        }
        self.prompt_template = prompt_template
        self.default_conv_mode = str(conv_mode)
        self.image_aspect_ratio = image_aspect_ratio
        self.max_new_tokens = max_new_tokens
        self.default_temperature = default_temperature
        self.processor_subfolder = processor_subfolder
        self.model_subfolder = model_subfolder
        self.tokenizer_subfolder = tokenizer_subfolder
        self.image_processor_subfolder = image_processor_subfolder
