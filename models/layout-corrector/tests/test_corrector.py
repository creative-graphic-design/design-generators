import torch

from layout_corrector import LayoutCorrectorModel


def tiny_model(**kwargs):
    params = {
        "dataset_name": "publaynet",
        "vocab_size": 16,
        "max_seq_length": 2,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_timesteps": 10,
        "use_padding_as_vocab": True,
    }
    params.update(kwargs)
    return LayoutCorrectorModel(**params)


def test_corrector_forward_shape():
    model = tiny_model()
    input_ids = torch.randint(0, 16, (3, 10))
    timesteps = torch.tensor([1, 2, 3])

    output = model(input_ids=input_ids, timesteps=timesteps)

    assert output.logits.shape == (3, 10)


def test_padding_positions_forced_high_when_not_vocab():
    model = tiny_model(use_padding_as_vocab=False)
    input_ids = torch.randint(0, 16, (1, 10))
    timesteps = torch.tensor([1])
    padding_mask = torch.zeros(1, 10, dtype=torch.bool)
    padding_mask[:, -2:] = True

    logits = model.calc_confidence_score(input_ids, timesteps, padding_mask)

    assert torch.equal(logits[:, -2:], torch.full((1, 2), 1000.0))
