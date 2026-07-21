import pytest
import torch

from layout_action import (
    LayoutActionConfig,
    LayoutActionForCausalLM,
    LayoutActionSamplingConfig,
    LayoutActionSamplingMode,
    sample_action_tokens,
    top_k_logits,
)


class FakeModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def get_block_size(self) -> int:
        return 16

    def __call__(self, input_ids):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size)
        logits[:, -1, 3] = 10.0
        return type("Output", (), {"logits": logits})()


def test_top_k_logits_masks_lower_values() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    masked = top_k_logits(logits, 2)

    assert torch.isneginf(masked[0, 0])
    assert masked[0, 2] == 3.0


def test_greedy_generation_and_forcing() -> None:
    prompt = torch.tensor([[1]])
    forced = torch.tensor([[-100, 5]])

    generated = sample_action_tokens(
        FakeModel(8),
        prompt,
        max_new_tokens=2,
        sampling=LayoutActionSamplingConfig(mode=LayoutActionSamplingMode.greedy),
        forced_token_ids=forced,
    )

    assert generated.tolist() == [[1, 3, 5]]


def test_model_generate_token_level() -> None:
    config = LayoutActionConfig(
        dataset_name="publaynet",
        max_elements=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    model = LayoutActionForCausalLM(config)

    generated = model.generate(
        torch.tensor([[config.bos_token_id]]),
        max_new_tokens=2,
        do_sample=False,
    )

    assert generated.shape == (1, 3)


def test_sampling_config_rejects_bad_mode() -> None:
    with pytest.raises(ValueError):
        LayoutActionSamplingConfig.from_values(
            mode="bad",
            temperature=1.0,
            top_k=None,
        )
