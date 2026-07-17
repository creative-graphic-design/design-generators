"""Prompt serialization ported from the original LayoutGPT scripts."""

from collections.abc import Callable, Sequence

from layout_gpt.exemplars import LayoutExample

TokenCounter = Callable[[str], int]


def default_token_counter(text: str) -> int:
    """Small dependency-free fallback token counter."""
    return len(text.split())


def system_prompt_2d(*, canvas_size: int) -> str:
    """Return the vendor 2D instruction prompt."""
    return (
        "Instruction: Given a sentence prompt that will be used to generate an image, "
        "plan the layout of the image."
        "The generated layout should follow the CSS style, where each line starts "
        "with the object description and is followed by its absolute position. "
        'Formally, each line should be like "object {width: ?px; height: ?px; '
        'left: ?px; top: ?px; }". '
        f"The image is {canvas_size}px wide and {canvas_size}px high. "
        f"Therefore, all properties of the positions should not exceed {canvas_size}px, "
        "including the addition of left and width and the addition of top and height. \n"
    )


def create_exemplar_prompt(
    example: LayoutExample,
    *,
    canvas_size: int,
    is_chat: bool = False,
) -> str:
    """Serialize one exemplar using vendor CSS property order."""
    prompt = "" if is_chat else f"\nPrompt: {example.prompt}\nLayout:\n"
    for category, bbox in example.objects:
        x, y, width, height = [int(value * canvas_size) for value in bbox]
        prompt += (
            f"{category} {{height: {height}px; width: {width}px; "
            f"top: {y}px; left: {x}px; }}\n"
        )
    return prompt


def form_prompt_for_chatgpt(
    text_input: str,
    *,
    exemplars: Sequence[LayoutExample],
    canvas_size: int,
    token_counter: TokenCounter = default_token_counter,
    input_length_limit: int = 3000,
) -> list[dict[str, str]]:
    """Build chat messages with vendor exemplar ordering and token truncation."""
    system_prompt = system_prompt_2d(canvas_size=canvas_size)
    final_prompt = f"Prompt: {text_input}\nLayout:"
    total_length = token_counter(system_prompt + final_prompt)
    messages = [{"role": "system", "content": system_prompt}]

    for exemplar in exemplars:
        user_prompt = f"Prompt: {exemplar.prompt}\nLayout:"
        answer = create_exemplar_prompt(exemplar, canvas_size=canvas_size, is_chat=True)
        current_length = token_counter(user_prompt + answer)
        if total_length + current_length > input_length_limit:
            break
        total_length += current_length
        current_messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]
        messages = messages[:1] + current_messages + messages[1:]

    messages.append({"role": "user", "content": final_prompt})
    return messages


def form_prompt_for_gpt3(
    text_input: str,
    *,
    exemplars: Sequence[LayoutExample],
    canvas_size: int,
    token_counter: TokenCounter = default_token_counter,
    input_length_limit: int = 3000,
) -> str:
    """Build completion prompt with vendor exemplar ordering and truncation."""
    prompt = system_prompt_2d(canvas_size=canvas_size)
    last_example = f"\nPrompt: {text_input}\nLayout:"
    total_length = token_counter(prompt + last_example)
    prompting_examples = ""

    for exemplar in exemplars:
        current = create_exemplar_prompt(exemplar, canvas_size=canvas_size)
        current_length = token_counter(current)
        if total_length + current_length > input_length_limit:
            break
        prompting_examples = current + prompting_examples
        total_length += current_length

    return prompt + prompting_examples + last_example
