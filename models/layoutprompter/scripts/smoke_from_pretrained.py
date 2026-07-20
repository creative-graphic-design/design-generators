"""Smoke-test LayoutPrompter prompt configuration loading."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai.models.test import TestModel

from layoutprompter import LayoutPrompter, LayoutPrompterConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Persist and reload a prompt-only LayoutPrompter configuration."""
    args = parse_args()
    model = TestModel(custom_output_args={"elements": []})
    agent = LayoutPrompter(LayoutPrompterConfig(model=model, dataset="webui"))
    agent.save_pretrained(args.path)
    loaded = LayoutPrompter.from_pretrained(args.path, model=model)
    print(loaded.config.dataset)


if __name__ == "__main__":
    main()
