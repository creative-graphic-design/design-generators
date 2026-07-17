"""Run a small LayoutGPT demo with a configured Pydantic AI provider."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from layout_gpt import LayoutGPTAgent
from layout_gpt.enums import ICLType, LayoutGPTSetting
from layout_gpt.exemplars import load_nsr_examples
from layout_gpt.schema import LayoutGPTConfig

SETTING_CHOICES: Final[tuple[str, ...]] = tuple(
    setting.value for setting in LayoutGPTSetting
)
DEFAULT_SETTING: Final[LayoutGPTSetting] = LayoutGPTSetting.counting
DEFAULT_ICL_TYPE: Final[ICLType] = ICLType.fixed_random
DEFAULT_CANVAS_SIZE: Final[int] = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--setting", choices=SETTING_CHOICES, default=DEFAULT_SETTING.value
    )
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    setting = LayoutGPTSetting(args.setting)
    config = LayoutGPTConfig(
        setting=setting,
        icl_type=DEFAULT_ICL_TYPE,
        canvas_size=args.canvas_size,
        chat=True,
    )
    train_examples = load_nsr_examples(args.train_json, setting=setting)
    agent = LayoutGPTAgent(model=args.model, config=config)
    output = agent.run_sync(args.prompt, train_examples=train_examples)
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
