import torch

from layout_fid import LayoutFIDConfig, LayoutFIDModel


def test_model_forward_and_reconstruction_shapes():
    cfg = LayoutFIDConfig(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=3,
        d_model=32,
        nhead=4,
        num_layers=1,
    )
    model = LayoutFIDModel(cfg)
    model.eval()
    out = model(
        bbox=torch.zeros(2, 3, 4),
        labels=torch.zeros(2, 3, dtype=torch.long),
        padding_mask=torch.tensor([[False, False, True], [False, True, True]]),
        output_reconstruction=True,
    )
    assert out.features.shape == (2, 32)
    assert out.discriminator_logits.shape == (2,)
    assert out.class_logits.shape == (3, 6)
    assert out.bbox_pred.shape == (3, 4)
    assert not hasattr(model, "generate_layout")
