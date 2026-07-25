"""Run PosterO with deterministic in-memory records and a fake provider."""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from postero import PosterOAgent, PosterOConfig
from postero.vendor_parity import fixture_records


def main() -> None:
    """Run the offline demo."""

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

    query, candidates = fixture_records()
    output = PosterOAgent(
        model=FunctionModel(respond),
        config=PosterOConfig(sample_size=1, n_valid_layouts=1),
    )(
        query_record=query,
        candidate_records=candidates,
        return_intermediates=True,
    )
    print(output)


if __name__ == "__main__":
    main()
