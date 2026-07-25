"""Run a local PosterO save/load and inference smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from postero import PosterOAgent, PosterOConfig
from postero.vendor_parity import fixture_records
from laygen.modeling_outputs import LayoutGenerationOutput


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/postero-smoke"),
        help="Directory used for the temporary saved prompt config.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the smoke test."""
    args = parse_args()

    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=(
                        '{"text":"<svg><rect data-label=\\"text_1\\" x=\\"10\\" '
                        'y=\\"20\\" width=\\"30\\" height=\\"40\\" /></svg>"}'
                    )
                )
            ]
        )

    model = FunctionModel(respond)
    agent = PosterOAgent(model=model, config=PosterOConfig(sample_size=1))
    agent.save_pretrained(args.work_dir)
    loaded = PosterOAgent.from_pretrained(args.work_dir, model=model)
    query, candidates = fixture_records()
    output = loaded(
        query_record=query,
        candidate_records=candidates,
        return_intermediates=True,
    )
    print(cast(LayoutGenerationOutput, output).bbox.shape)


if __name__ == "__main__":
    main()
