"""Run a small LayoutGPT demo with a configured Pydantic AI provider."""

from __future__ import annotations

import argparse
from pathlib import Path

from layout_gpt import LayoutGPTAgent
from layout_gpt.exemplars import load_nsr_examples
from layout_gpt.schema import LayoutGPTConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--setting", choices=["counting", "spatial"], default="counting"
    )
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    config = LayoutGPTConfig(
        setting=args.setting,
        icl_type="fixed-random",
        canvas_size=args.canvas_size,
        chat=True,
    )
    train_examples = load_nsr_examples(args.train_json, setting=args.setting)
    agent = LayoutGPTAgent(model=args.model, config=config)
    output = agent.run_sync(args.prompt, train_examples=train_examples)
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
