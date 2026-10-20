import torch
import pytest

from radm import RADMConfig, RADMDenoiser
from radm.modeling_radm import RADMGeometryRelationAwareModule


def _config(*, num_classes: int, num_proposals: int) -> RADMConfig:
    return RADMConfig(
        num_classes=num_classes,
        num_proposals=num_proposals,
        hidden_dim=8,
        text_feature_dim=4,
        backbone_depth=18,
    )


def test_denoiser_forward_shapes() -> None:
    model = RADMDenoiser(config=_config(num_classes=5, num_proposals=3))
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
    model = RADMDenoiser(config=_config(num_classes=2, num_proposals=2))
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


def test_gram_frequency_basis_uses_source_device_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the source CPU basis construction before the device transfer."""
    module = RADMGeometryRelationAwareModule(pooled_dim=4, output_dim=64)
    try:
        torch.empty(1, device="cuda:0")
    except (AssertionError, RuntimeError):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    relative_geometry = torch.rand(2, 2, 4, device=device)
    calls: list[tuple[str, str]] = []
    original_arange = torch.arange
    original_full = torch.full
    original_pow = torch.pow

    def tracked_arange(start: int, end: float) -> torch.Tensor:
        result = original_arange(start, end)
        if start == 0 and end == module.out_dim / 8:
            calls.append(("arange", str(result.device)))
        return result

    def tracked_full(size: tuple[int, ...], fill_value: float) -> torch.Tensor:
        result = original_full(size, fill_value)
        if (
            result.shape == (1,)
            and result.numel() == 1
            and result.item() == module.wave_length
        ):
            calls.append(("full", str(result.device)))
        return result

    def tracked_pow(input: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        result = original_pow(input, exponent)
        if (
            isinstance(input, torch.Tensor)
            and input.shape == (1,)
            and input.item() == module.wave_length
        ):
            calls.append(("pow", str(input.device)))
        return result

    monkeypatch.setattr(torch, "arange", tracked_arange)
    monkeypatch.setattr(torch, "full", tracked_full)
    monkeypatch.setattr(torch, "pow", tracked_pow)
    output = module.extract_position_embedding(relative_geometry)

    assert calls == [("arange", "cpu"), ("full", "cpu"), ("pow", "cpu")]
    assert output.device == device
