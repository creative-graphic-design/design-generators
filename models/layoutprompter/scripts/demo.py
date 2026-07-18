"""Run a tiny LayoutPrompter demo when an LLM API key is configured."""

from __future__ import annotations

import os

import numpy as np

from layoutprompter import LayoutPrompter, LayoutPrompterConfig


def main() -> None:
    """Run the demo or skip cleanly when no provider key is available."""
    model = os.getenv("LAYOUTPROMPTER_MODEL", "openai:gpt-4o-mini")
    if model.startswith("openai:") and not os.getenv("OPENAI_API_KEY"):
        print("Skipping real LLM demo: OPENAI_API_KEY is not set.")
        return
    train_data = [
        {
            "labels": np.asarray([0, 1]),
            "bboxes": np.asarray([[10, 10, 40, 20], [60, 80, 30, 20]]),
            "discrete_gold_bboxes": np.asarray([[10, 10, 40, 20], [60, 80, 30, 20]]),
        }
    ]
    test_data = {
        "labels": np.asarray([0, 1]),
        "bboxes": np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]]),
        "discrete_gold_bboxes": np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]]),
    }
    prompter = LayoutPrompter(
        LayoutPrompterConfig(model=model, shuffle=False, num_prompt=1)
    )
    print(prompter.run_sync(train_data, test_data))


if __name__ == "__main__":
    main()
