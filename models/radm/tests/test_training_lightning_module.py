from collections import OrderedDict

import torch
from torch import nn

import radm.training.lightning_module as lightning_module
from radm.training.dataset import RADMTrainingBatch


class TinyBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        batch = images.shape[0]
        return OrderedDict(
            (str(index), torch.ones(batch, 256, 2, 2, device=images.device))
            for index in range(4)
        )


class TinyPooler(nn.Module):
    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        boxes: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        del image_shapes
        first = next(iter(features.values()))
        return first.new_ones(sum(box.shape[0] for box in boxes), 256, 7, 7)


def _batch(mask: torch.Tensor | None = None) -> RADMTrainingBatch:
    return RADMTrainingBatch(
        images=torch.zeros(1, 3, 8, 8),
        boxes_xyxy=torch.tensor([[[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]]]),
        labels=torch.tensor([[1, 2]]),
        mask=mask if mask is not None else torch.tensor([[True, True]]),
        text_features=torch.ones(1, 2, 3),
        text_mask=torch.tensor([[[True], [False]]]),
        canvas_size=torch.tensor([[8, 8]]),
    )


def test_training_module_runs_steps_with_tiny_backbone(monkeypatch) -> None:
    monkeypatch.setattr(
        lightning_module,
        "resnet_fpn_backbone",
        lambda **kwargs: TinyBackbone(),
    )
    monkeypatch.setattr(
        lightning_module,
        "MultiScaleRoIAlign",
        lambda **kwargs: TinyPooler(),
    )
    module = lightning_module.RADMTrainingModule(
        num_classes=3,
        num_proposals=2,
        hidden_dim=4,
        text_feature_dim=3,
        num_train_timesteps=5,
        learning_rate=1e-3,
        lr_steps=(1, 2),
        ota_k=1,
    )
    module.log = lambda *args, **kwargs: None

    module.on_fit_start()
    output = module.model(_batch())
    train_loss = module.training_step(_batch(), 0)
    val_loss = module.validation_step(_batch(torch.tensor([[False, False]])), 0)
    optimizer_config = module.configure_optimizers()

    assert output["logits"].shape == (1, 2, 3)
    assert torch.isfinite(train_loss)
    assert torch.isfinite(val_loss)
    assert "train_loss" in module.latest_step_trace
    assert "optimizer" in optimizer_config
