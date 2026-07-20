"""Save a LayoutGPT prompt configuration directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from layout_gpt import LayoutGPTAgent
from layout_gpt.schema import LayoutGPTConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Persist a small prompt-only configuration."""
    args = parse_args()
    agent = LayoutGPTAgent(
        config=LayoutGPTConfig(setting="counting", icl_type="fixed-random", k=1)
    )
    agent.save_pretrained(args.path)
    print(args.path / "layout_gpt_config.json")


if __name__ == "__main__":
    main()
