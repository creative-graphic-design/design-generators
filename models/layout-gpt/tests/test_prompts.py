"""Tests for vendor-compatible LayoutGPT prompt serialization."""

from layout_gpt.exemplars import LayoutExample
from layout_gpt.prompts import (
    create_exemplar_prompt,
    form_prompt_for_chatgpt,
    form_prompt_for_gpt3,
)


def test_create_exemplar_prompt_matches_vendor_css_order() -> None:
    example = LayoutExample(
        id=1,
        prompt="one sink is in the picture",
        objects=(("sink", (0.5, 0.25, 0.25, 0.5)),),
        metadata={},
    )

    assert create_exemplar_prompt(example, canvas_size=64) == (
        "\nPrompt: one sink is in the picture\nLayout:\n"
        "sink {height: 32px; width: 16px; top: 16px; left: 32px; }\n"
    )


def test_chat_prompt_puts_later_selected_example_closer_to_query() -> None:
    first = LayoutExample(id=1, prompt="first", objects=(), metadata={})
    second = LayoutExample(id=2, prompt="second", objects=(), metadata={})

    messages = form_prompt_for_chatgpt(
        "query",
        exemplars=[first, second],
        canvas_size=64,
        token_counter=lambda _text: 1,
    )

    assert [message["content"] for message in messages[1:4:2]] == [
        "Prompt: second\nLayout:",
        "Prompt: first\nLayout:",
    ]


def test_completion_prompt_truncates_by_token_budget() -> None:
    examples = [
        LayoutExample(
            id=1, prompt="first", objects=(("a", (0, 0, 1, 1)),), metadata={}
        ),
        LayoutExample(
            id=2, prompt="second", objects=(("b", (0, 0, 1, 1)),), metadata={}
        ),
    ]

    prompt = form_prompt_for_gpt3(
        "query",
        exemplars=examples,
        canvas_size=64,
        token_counter=lambda text: 10 if "second" in text else 1,
        input_length_limit=5,
    )

    assert "Prompt: first" in prompt
    assert "Prompt: second" not in prompt
