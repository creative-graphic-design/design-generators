import torch

from cgb_dm import CGBDMTransformerModel


def tiny_model() -> CGBDMTransformerModel:
    return CGBDMTransformerModel(
        num_labels=4,
        max_seq_length=2,
        image_size=(32, 32),
        dim_model=16,
        n_head=2,
        feature_dim=32,
        num_layers=1,
        num_train_timesteps=10,
    )


def test_model_forward_shapes_and_key_families():
    model = tiny_model()
    sample = torch.zeros(1, 2, 8)
    image = torch.zeros(1, 4, 32, 32)
    saliency_box = torch.zeros(1, 1, 4)
    out = model(sample, image, saliency_box, torch.zeros(1, dtype=torch.long))

    assert out.sample.shape == sample.shape
    assert out.cgb_weight is not None
    keys = set(model.state_dict())
    assert any(key.startswith("img_encoder.") for key in keys)
    assert any(key.startswith("layout_encoder.") for key in keys)
    assert any(key.startswith("layout_decoder.") for key in keys)
    assert any(key.startswith("cgbwp.") for key in keys)
    assert any(key.startswith("salbox_encoder.") for key in keys)
