"""Smoke-test LayoutGPT prompt configuration loading."""

from __future__ import annotations

import argparse
from pathlib import Path

from layout_gpt import LayoutGPTAgent
from layout_gpt.schema import LayoutGPTOutput, LayoutItem2D


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Reload prompt configuration and convert a parsed output."""
    args = parse_args()
    loaded = LayoutGPTAgent.from_pretrained(args.path)
    assert loaded.config.setting == "counting"
    assert loaded.config.icl_type == "fixed-random"
    assert loaded.config.k == 1
    output = LayoutGPTOutput(
        prompt="there is one clock in the image",
        canvas_size=64,
        items=[LayoutItem2D(label="clock", left=0.25, top=0.25, width=0.5, height=0.5)],
        raw_text="clock {height: 32px; width: 32px; top: 16px; left: 16px; }",
        id2label={0: "clock"},
    )
    layout = output.to_layout_generation_output()
    assert layout.bbox.shape == (1, 1, 4)
    assert layout.id2label == {0: "clock"}
    print("LayoutGPT loading smoke passed.")


if __name__ == "__main__":
    main()
