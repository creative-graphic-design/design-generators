import torch

from layoutvae import LayoutVAEConfig, LayoutVAEModel
from layoutvae.modeling_layoutvae import Decoder, Embeder, Encoder, LayoutVAEModelOutput


def _counts():
    return torch.tensor([[7.0, 1.0, 0.0, 0.0, 0.0, 1.0]])


def test_model_forward_shapes_and_mask():
    model = LayoutVAEModel(LayoutVAEConfig()).eval()
    label_set = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    output = model(label_set, class_counts=_counts())
    assert isinstance(output, LayoutVAEModelOutput)
    assert output.bbox.shape == (1, 9, 4)
    assert output.labels.shape == (1, 9)
    assert output.mask.sum().item() == 2
    assert output.bbox[output.mask].min().item() >= 0.0
    assert output.bbox[output.mask].max().item() <= 1.0


def test_model_return_tuple_and_save_load(tmp_path):
    model = LayoutVAEModel(LayoutVAEConfig()).eval()
    label_set = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    output = model(label_set, class_counts=_counts(), return_dict=False)
    assert len(output) == 6
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = LayoutVAEModel.from_pretrained(tmp_path).eval()
    loaded_output = loaded(label_set, class_counts=_counts())
    assert loaded_output.bbox.shape == (1, 9, 4)


def test_model_sampling_and_validation_branches():
    model = LayoutVAEModel(LayoutVAEConfig()).eval()
    label_set = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    generator = torch.Generator().manual_seed(0)
    out = model(label_set, generator=generator)
    assert out.class_counts.shape == (1, 6)
    with torch.no_grad():
        normalized = model._normalize_counts(torch.zeros(1, 6))
        labels = model._labels_from_counts(torch.tensor([[0.0, 0, 0, 0, 0, 12]]))
    assert normalized.shape == (1, 6)
    assert labels.shape == (1, 9, 6)
    try:
        model(torch.zeros(1, 5))
    except ValueError as exc:
        assert "label_set" in str(exc)
    try:
        model(label_set, class_counts=torch.zeros(1, 5))
    except ValueError as exc:
        assert "class_counts" in str(exc)


def test_low_level_blocks_are_callable():
    embeder = Embeder(6)
    embedding = embeder([torch.zeros(1, 6), torch.zeros(1, 6), torch.zeros(1, 6)])
    assert embedding.shape == (1, 128)
    mu, logvar = Encoder()([torch.zeros(1, 1), embedding])
    assert mu.shape == logvar.shape == (1, 32)
    decoded = Decoder(1)([embedding, torch.zeros(1, 32)])
    assert decoded.shape == (1, 1)
