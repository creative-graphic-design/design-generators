import torch
import pytest

from radm import RADMDenoiser


def test_denoiser_forward_shapes() -> None:
    model = RADMDenoiser(num_classes=5, hidden_dim=8, text_feature_dim=4)
    out = model(
        boxes_xyxy=torch.zeros(2, 3, 4),
        timesteps=torch.tensor([1.0, 2.0]),
        text_features=torch.ones(2, 2, 4),
        text_mask=torch.tensor([[[True], [False]], [[True], [True]]]),
    )
    assert out.logits.shape == (2, 3, 5)
    assert out.boxes_xyxy.shape == (2, 3, 4)
    assert out.pred_noise.shape == (2, 3, 4)


def test_denoiser_scalar_timestep_no_mask_and_shape_error() -> None:
    model = RADMDenoiser(num_classes=2, hidden_dim=8, text_feature_dim=4)
    out = model(
        boxes_xyxy=torch.zeros(1, 2, 4),
        timesteps=torch.tensor(1.0),
        text_features=torch.ones(1, 2, 4),
    )
    assert out.logits.shape == (1, 2, 2)
    with pytest.raises(ValueError, match="text_features"):
        model(
            boxes_xyxy=torch.zeros(1, 2, 4),
            timesteps=torch.tensor(1.0),
            text_features=torch.ones(2, 4),
        )
