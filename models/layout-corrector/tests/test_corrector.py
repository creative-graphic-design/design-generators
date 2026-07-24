import torch
import yaml
import pytest

from layout_corrector import LayoutCorrectorModel
from layout_corrector.modeling_layout_corrector import AggregatedCategoricalTransformer


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


def test_corrector_model_accepts_position_embedding_branch():
    model = tiny_model(pos_emb="default", timestep_type=None)
    output = model(
        input_ids=torch.randint(0, 16, (1, 10)),
        timesteps=torch.tensor([1]),
    )

    assert output.logits.shape == (1, 10)


def test_corrector_model_allows_external_dataset_with_explicit_labels():
    model = tiny_model(dataset_name="crello-bbox", id2label={0: "class_0"})

    assert model.config.dataset_name == "crello-bbox"
    assert (
        yaml.safe_load(yaml.safe_dump(dict(model.config)))["dataset_name"]
        == "crello-bbox"
    )


def test_corrector_model_rejects_unsupported_modes():
    with pytest.raises(ValueError, match="recon_type"):
        tiny_model(recon_type="bad")
    with pytest.raises(ValueError, match="target"):
        tiny_model(target="bad")
    with pytest.raises(ValueError, match="transformer_type"):
        tiny_model(transformer_type="bad")


def test_aggregated_transformer_rejects_bad_token_length():
    with pytest.raises(ValueError, match="max_token_length"):
        AggregatedCategoricalTransformer(
            vocab_size=16,
            max_token_length=11,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            dropout=0.0,
            timestep_type=None,
            pos_emb="none",
            num_attributes_per_element=5,
            num_timesteps=10,
        )
