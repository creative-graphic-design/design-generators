import pytest
import torch

from layout_action import LayoutActionConfig, LayoutActionForCausalLM


def tiny_config() -> LayoutActionConfig:
    return LayoutActionConfig(
        dataset_name="publaynet",
        max_elements=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )


def test_model_forward_shape_and_pad_loss() -> None:
    config = tiny_config()
    model = LayoutActionForCausalLM(config)
    input_ids = torch.tensor([[config.bos_token_id, config.label_token_id(0)]])
    labels = torch.tensor([[config.label_token_id(0), config.pad_token_id]])

    output = model(input_ids, labels=labels)

    assert output.logits.shape == (1, 2, config.vocab_size)
    assert output.loss is not None


def test_model_save_pretrained_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = LayoutActionForCausalLM(config)
    model.eval()
    input_ids = torch.tensor([[config.bos_token_id, config.label_token_id(0)]])
    expected = model(input_ids).logits

    model.save_pretrained(tmp_path)
    restored = LayoutActionForCausalLM.from_pretrained(tmp_path)
    restored.eval()

    assert torch.equal(restored(input_ids).logits, expected)


def test_model_forward_tuple_and_errors() -> None:
    config = tiny_config()
    model = LayoutActionForCausalLM(config)
    input_ids = torch.tensor([[config.bos_token_id]])

    assert model.get_input_embeddings() is model.tok_emb
    model.set_input_embeddings(model.tok_emb)
    tuple_output = model(input_ids, return_dict=False, output_hidden_states=True)
    assert tuple_output[0].shape[-1] == config.vocab_size
    with pytest.raises(ValueError):
        LayoutActionConfig(n_head=3, n_embd=16)
        LayoutActionForCausalLM(LayoutActionConfig(n_head=3, n_embd=16))
    with pytest.raises(ValueError):
        model(torch.full((1, config.block_size + 1), config.bos_token_id))
