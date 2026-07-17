"""Pydantic AI wrapper for LayoutPrompter."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from layout_generation_common.outputs import LayoutGenerationOutput
from layoutprompter.parsing import Parser
from layoutprompter.schemas import LayoutPrompterOutput
from layoutprompter.selection import create_selector
from layoutprompter.serialization import build_prompt, create_serializer


TASK_ALIASES: dict[str, str] = {
    "label": "gent",
    "cat_cond": "gent",
    "gen_t": "gent",
    "label_size": "gents",
    "size_cond": "gents",
    "gen_ts": "gents",
    "relation": "genr",
    "gen_r": "genr",
    "completion": "completion",
    "partial": "completion",
    "refinement": "refinement",
    "refine": "refinement",
    "text": "text",
}


@dataclass(frozen=True)
class LayoutPrompterConfig:
    """Runtime configuration for LayoutPrompter prompt generation."""

    dataset: str = "publaynet"
    condition_type: str = "label"
    input_format: str = "seq"
    output_format: str = "seq"
    candidate_size: int = -1
    num_prompt: int = 3
    shuffle: bool = True
    seed: int | None = None
    max_length: int = 8000
    temperature: float = 0.7
    top_p: float = 1.0
    model: str | Model | None = None

    @property
    def task(self) -> str:
        """Return the vendor task key."""
        try:
            return TASK_ALIASES[self.condition_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported condition_type: {self.condition_type}"
            ) from exc


class LayoutPrompter:
    """High-level LayoutPrompter Pydantic AI agent."""

    def __init__(self, config: LayoutPrompterConfig | None = None) -> None:
        """Create a LayoutPrompter runner from runtime config."""
        self.config = config or LayoutPrompterConfig()
        self.serializer = create_serializer(
            self.config.dataset,
            self.config.task,
            self.config.input_format,
            self.config.output_format,
        )
        self.parser = Parser(self.config.dataset, self.config.output_format)
        self.agent = Agent(
            self._resolve_model(self.config.model),
            output_type=LayoutPrompterOutput,
            model_settings={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
        )

    def build_prompt(
        self, train_data: list[dict[str, Any]], test_data: dict[str, Any]
    ) -> str:
        """Select exemplars and build the final LayoutPrompter prompt."""
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
        self, train_data: list[dict[str, Any]], test_data: dict[str, Any]
    ) -> LayoutGenerationOutput:
        """Run the Pydantic AI model and return the common layout schema."""
        prompt = self.build_prompt(train_data, test_data)
        result = self.agent.run_sync(prompt)
        return self.parser.parse_one(result.output)

    def __call__(
        self,
        *,
        train_data: list[dict[str, Any]],
        test_data: dict[str, Any],
        batch_size: int = 1,
        seed: int | None = None,
        generator: object | None = None,
        condition_type: str | None = None,
        labels: object | None = None,
        bbox: object | None = None,
        mask: object | None = None,
        num_elements: int | list[int] | object | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
        output_type: str = "dataclass",
        return_intermediates: bool = False,
        **model_kwargs: object,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Expose the shared generation signature for LayoutPrompter."""
        del (
            batch_size,
            seed,
            generator,
            condition_type,
            labels,
            bbox,
            mask,
            num_elements,
            box_format,
            normalized,
            canvas_size,
            num_inference_steps,
            return_intermediates,
            model_kwargs,
        )
        output = self.run_sync(train_data, test_data)
        if output_type == "dataclass":
            return output
        if output_type == "dict":
            return asdict(output)
        raise ValueError(f"Unsupported output_type: {output_type}")

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        """Persist the dataset and prompt configuration."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        config["model"] = None
        (path / "layoutprompter_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike[str],
        *,
        model: str | Model | None = None,
    ) -> "LayoutPrompter":
        """Load a saved LayoutPrompter dataset and prompt configuration."""
        path = Path(pretrained_model_name_or_path) / "layoutprompter_config.json"
        config_data = json.loads(path.read_text(encoding="utf-8"))
        config_data["model"] = model
        return cls(LayoutPrompterConfig(**config_data))

    @staticmethod
    def _resolve_model(model: str | Model | None) -> str | Model:
        if model is not None:
            return model
        return (
            os.getenv("LAYOUTPROMPTER_MODEL")
            or os.getenv("PYDANTIC_AI_MODEL")
            or "openai:gpt-4o-mini"
        )
