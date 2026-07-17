"""Pydantic AI wrapper for LayoutPrompter."""

from __future__ import annotations

import os
import json
from dataclasses import asdict
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path
from typing import Final, assert_never

from laygen.agents import BaseLayoutAgent, ModelLike
from laygen.common import (
    BoxFormat,
    ConditionType,
    DatasetName,
    LayoutGenerationOutput,
    normalize_box_format,
    normalize_condition_type,
)
from pydantic_ai.settings import ModelSettings

from layoutprompter.data import SupportedDataset, normalize_dataset
from layoutprompter.enums import LayoutPrompterTask, OutputType, PromptFormat
from layoutprompter.parsing import Parser
from layoutprompter.records import LayoutRecordInput
from layoutprompter.schemas import LayoutPrompterOutput
from layoutprompter.selection import create_selector
from layoutprompter.serialization import build_prompt, create_serializer

DEFAULT_CANDIDATE_SIZE: Final[int] = -1
DEFAULT_NUM_PROMPT: Final[int] = 3
DEFAULT_MAX_LENGTH: Final[int] = 8000
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 1.0
DEFAULT_MODEL: Final[str] = "openai:gpt-4o-mini"
DEFAULT_MODEL_ENV_VAR: Final[str] = "LAYOUTPROMPTER_MODEL"
INSTRUCTIONS: Final[str] = (
    "You are LayoutPrompter. Return structured layout elements with dataset labels "
    "and pixel left/top/width/height boxes."
)

LayoutRecord = LayoutRecordInput


TASK_ALIASES: Final[dict[ConditionType, LayoutPrompterTask]] = {
    ConditionType.label: LayoutPrompterTask.gent,
    ConditionType.label_size: LayoutPrompterTask.gents,
    ConditionType.relation: LayoutPrompterTask.genr,
    ConditionType.completion: LayoutPrompterTask.completion,
    ConditionType.refinement: LayoutPrompterTask.refinement,
    ConditionType.text: LayoutPrompterTask.text,
}


def normalize_prompt_format(prompt_format: PromptFormat | str) -> PromptFormat:
    """Return a prompt-format enum from a public string value."""
    try:
        return PromptFormat(prompt_format)
    except ValueError as exc:
        raise ValueError(f"Unsupported prompt format: {prompt_format}") from exc


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Return an output-type enum from a public string value."""
    try:
        return OutputType(output_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported output_type: {output_type}") from exc


@dataclass(frozen=True)
class LayoutPrompterConfig:
    """Runtime configuration for LayoutPrompter prompt generation.

    Args:
        dataset: Dataset vocabulary to use. Public strings are normalized to
            `LayoutPrompterDataset`.
        condition_type: Public condition name or vendor alias.
        input_format: Prompt input format, either `seq` or `html`.
        output_format: Model output format, either `seq` or `html`.
        candidate_size: Number of training candidates to keep before retrieval.
            `-1` keeps all candidates.
        num_prompt: Maximum number of few-shot exemplars in each prompt.
        shuffle: Whether to shuffle selected exemplars after ranking.
        seed: Optional deterministic seed for exemplar selection.
        max_length: Maximum prompt length used while adding exemplars.
        temperature: Model sampling temperature passed to Pydantic AI.
        top_p: Nucleus sampling value passed to Pydantic AI.
        model: Pydantic AI model instance or model string.

    Raises:
        ValueError: If a dataset, condition, or prompt format is unsupported.

    Examples:
        >>> from pydantic_ai.models.test import TestModel
        >>> config = LayoutPrompterConfig(
        ...     dataset="webui",
        ...     condition_type="label",
        ...     model=TestModel(custom_output_args={"elements": []}),
        ... )
        >>> config.task
        'gent'
    """

    dataset: SupportedDataset | str = DatasetName.publaynet
    condition_type: ConditionType | str = ConditionType.label
    input_format: PromptFormat | str = PromptFormat.SEQ
    output_format: PromptFormat | str = PromptFormat.SEQ
    candidate_size: int = DEFAULT_CANDIDATE_SIZE
    num_prompt: int = DEFAULT_NUM_PROMPT
    shuffle: bool = True
    seed: int | None = None
    max_length: int = DEFAULT_MAX_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    model: ModelLike = None

    def __post_init__(self) -> None:
        """Normalize public string modes to enums at the config boundary."""
        object.__setattr__(self, "dataset", normalize_dataset(self.dataset))
        object.__setattr__(
            self, "condition_type", normalize_condition_type(self.condition_type)
        )
        object.__setattr__(
            self, "input_format", normalize_prompt_format(self.input_format)
        )
        object.__setattr__(
            self, "output_format", normalize_prompt_format(self.output_format)
        )

    @property
    def task(self) -> LayoutPrompterTask:
        """Return the vendor task key."""
        condition_type = normalize_condition_type(self.condition_type)
        try:
            return TASK_ALIASES[condition_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported LayoutPrompter condition_type: {condition_type}"
            ) from exc


class LayoutPrompter(BaseLayoutAgent[LayoutPrompterOutput]):
    """High-level LayoutPrompter Pydantic AI agent.

    Args:
        config: Runtime prompt, retrieval, parser, and model settings. Defaults
            to `LayoutPrompterConfig()`.

    Raises:
        ValueError: If the config contains an unsupported mode.

    Examples:
        >>> import torch
        >>> from pydantic_ai.models.test import TestModel
        >>> model = TestModel(custom_output_args={"elements": []})
        >>> agent = LayoutPrompter(LayoutPrompterConfig(model=model, num_prompt=1))
        >>> isinstance(agent, LayoutPrompter)
        True
    """

    def __init__(self, config: LayoutPrompterConfig | None = None) -> None:
        """Create a LayoutPrompter runner from runtime config.

        Args:
            config: Optional agent configuration.

        Raises:
            ValueError: If a config mode cannot be normalized.
        """
        self.config = config or LayoutPrompterConfig()
        super().__init__(
            model=self.config.model,
            model_env_var=DEFAULT_MODEL_ENV_VAR,
            raw_response_type=LayoutPrompterOutput,
            instructions=INSTRUCTIONS,
        )
        self.serializer = create_serializer(
            self.config.dataset,
            self.config.task,
            self.config.input_format,
            self.config.output_format,
        )
        self.parser = Parser(self.config.dataset, self.config.output_format)

    def build_prompt(
        self, train_data: Sequence[LayoutRecord], test_data: LayoutRecord
    ) -> str:
        """Select exemplars and build the final LayoutPrompter prompt.

        Args:
            train_data: Candidate exemplar records with `labels`, `bboxes`, and
                `discrete_gold_bboxes` tensors.
            test_data: Test record containing task-specific constraints.

        Returns:
            The final few-shot prompt sent to the configured model.

        Raises:
            KeyError: If required record fields are missing.

        Examples:
            >>> import torch
            >>> from pydantic_ai.models.test import TestModel
            >>> record = {
            ...     "labels": torch.tensor([0]),
            ...     "bboxes": torch.tensor([[1, 2, 3, 4]]),
            ...     "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
            ... }
            >>> agent = LayoutPrompter(
            ...     LayoutPrompterConfig(model=TestModel(), shuffle=False, num_prompt=1)
            ... )
            >>> "Element Type Constraint" in agent.build_prompt([record], record)
            True
        """
        selector = create_selector(
            self.config.task,
            train_data,
            self.config.candidate_size,
            self.config.num_prompt,
            shuffle=self.config.shuffle,
            seed=self.config.seed,
        )
        exemplars = selector(test_data)
        return build_prompt(
            self.serializer,
            exemplars,
            test_data,
            self.config.dataset,
            max_length=self.config.max_length,
        )

    def run_sync(
        self, train_data: Sequence[LayoutRecord], test_data: LayoutRecord
    ) -> LayoutGenerationOutput:
        """Run the Pydantic AI model and return the common layout schema.

        Args:
            train_data: Candidate exemplar records.
            test_data: Test record containing task-specific constraints.

        Returns:
            A `LayoutGenerationOutput` with normalized center `xywh` boxes.

        Raises:
            RuntimeError: If the model output cannot be parsed.

        Examples:
            >>> import torch
            >>> from pydantic_ai.models.test import TestModel
            >>> model = TestModel(custom_output_args={"elements": []})
            >>> record = {
            ...     "labels": torch.tensor([0]),
            ...     "bboxes": torch.tensor([[1, 2, 3, 4]]),
            ...     "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
            ... }
            >>> agent = LayoutPrompter(LayoutPrompterConfig(model=model, num_prompt=1))
            >>> agent.run_sync([record], record).labels.shape[0]
            1
        """
        prompt = self.build_prompt(train_data, test_data)
        raw = self.run_raw_sync(
            prompt,
            model_settings=ModelSettings(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            ),
        )
        return self.parser.parse_one(raw)

    def __call__(
        self,
        *,
        train_data: Sequence[LayoutRecord],
        test_data: LayoutRecord,
        batch_size: int = 1,
        seed: int | None = None,
        generator: object | None = None,
        condition_type: ConditionType | str | None = None,
        labels: object | None = None,
        bbox: object | None = None,
        mask: object | None = None,
        num_elements: int | list[int] | object | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: OutputType | str = OutputType.DATACLASS,
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Expose the shared generation signature for LayoutPrompter.

        Args:
            train_data: Candidate exemplar records.
            test_data: Test record containing task-specific constraints.
            batch_size: Accepted for shared interface compatibility.
            seed: Accepted for shared interface compatibility.
            generator: Accepted for shared interface compatibility.
            condition_type: Accepted for shared interface compatibility.
            labels: Accepted for shared interface compatibility.
            bbox: Accepted for shared interface compatibility.
            mask: Accepted for shared interface compatibility.
            num_elements: Accepted for shared interface compatibility.
            box_format: Public input box format name; validated at the boundary.
            normalized: Accepted for shared interface compatibility.
            canvas_size: Accepted for shared interface compatibility.
            num_inference_steps: Accepted for shared interface compatibility.
            output_type: `dataclass` for `LayoutGenerationOutput`, or `dict`.
            return_intermediates: Accepted for shared interface compatibility.

        Returns:
            A `LayoutGenerationOutput` or dictionary representation.

        Raises:
            ValueError: If `box_format` or `output_type` is unsupported.

        Examples:
            >>> import torch
            >>> from pydantic_ai.models.test import TestModel
            >>> model = TestModel(custom_output_args={"elements": []})
            >>> record = {
            ...     "labels": torch.tensor([0]),
            ...     "bboxes": torch.tensor([[1, 2, 3, 4]]),
            ...     "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
            ... }
            >>> agent = LayoutPrompter(LayoutPrompterConfig(model=model, num_prompt=1))
            >>> isinstance(agent(train_data=[record], test_data=record), LayoutGenerationOutput)
            True
        """
        if condition_type is not None:
            normalize_condition_type(condition_type)
        del (
            batch_size,
            seed,
            generator,
            condition_type,
            labels,
            bbox,
            mask,
            num_elements,
            normalized,
            canvas_size,
            num_inference_steps,
            return_intermediates,
        )
        normalize_box_format(box_format)
        normalized_output_type = normalize_output_type(output_type)
        output = self.run_sync(train_data, test_data)
        if normalized_output_type is OutputType.DATACLASS:
            return output
        if normalized_output_type is OutputType.DICT:
            return self.output_to_dict(output)
        assert_never(normalized_output_type)

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        """Persist the dataset and prompt configuration.

        Args:
            save_directory: Target directory for `layoutprompter_config.json`.

        Returns:
            None.

        Raises:
            OSError: If the directory or config file cannot be written.

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from pydantic_ai.models.test import TestModel
            >>> agent = LayoutPrompter(LayoutPrompterConfig(model=TestModel()))
            >>> with TemporaryDirectory() as tmpdir:
            ...     agent.save_pretrained(tmpdir)
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        config["model"] = None
        config["dataset"] = str(self.config.dataset)
        config["condition_type"] = str(self.config.condition_type)
        config["input_format"] = str(self.config.input_format)
        config["output_format"] = str(self.config.output_format)
        (path / "layoutprompter_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        model: ModelLike = None,
    ) -> "LayoutPrompter":
        """Load a saved LayoutPrompter dataset and prompt configuration.

        Args:
            pretrained_model_name_or_path: Directory containing
                `layoutprompter_config.json`.
            model: Replacement Pydantic AI model or model string.

        Returns:
            A configured `LayoutPrompter` instance.

        Raises:
            OSError: If the config file cannot be read.
            ValueError: If saved config modes are unsupported.

        Examples:
            >>> from tempfile import TemporaryDirectory
            >>> from pydantic_ai.models.test import TestModel
            >>> model = TestModel(custom_output_args={"elements": []})
            >>> agent = LayoutPrompter(LayoutPrompterConfig(model=model))
            >>> with TemporaryDirectory() as tmpdir:
            ...     agent.save_pretrained(tmpdir)
            ...     loaded = LayoutPrompter.from_pretrained(tmpdir, model=model)
            >>> isinstance(loaded, LayoutPrompter)
            True
        """
        path = Path(pretrained_model_name_or_path) / "layoutprompter_config.json"
        config_data = json.loads(path.read_text(encoding="utf-8"))
        config_data["model"] = model
        return cls(LayoutPrompterConfig(**config_data))

    @staticmethod
    def _resolve_model(model: ModelLike = None) -> ModelLike:
        if model is not None:
            return model
        return (
            os.getenv(DEFAULT_MODEL_ENV_VAR)
            or os.getenv("PYDANTIC_AI_MODEL")
            or DEFAULT_MODEL
        )

    def resolve_model(self, model: ModelLike = None) -> ModelLike:
        """Resolve constructor or per-call model with LayoutPrompter defaults."""
        return self._resolve_model(model)
