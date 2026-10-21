from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radm.modeling_radm import RADMDenoiser, RADMDenoiserOutput
from radm.training.config import effective_radm_config
from radm.training.lightning_module import RADMTrainingModule


class _TinyRADMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.tensor(0.0))
        self.second = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(num_classes=1)

    def prepare_diffusion_concat(
        self, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            boxes,
            torch.zeros_like(boxes),
            torch.zeros(1, dtype=torch.long, device=boxes.device),
        )

    def forward(
        self,
        *,
        boxes_xyxy: torch.Tensor,
        timesteps: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        images: torch.Tensor,
        image_scales: torch.Tensor,
        absolute_boxes_xyxy: torch.Tensor | None = None,
    ) -> RADMDenoiserOutput:
        del timesteps, text_features, text_mask, images, image_scales
        batch_size, proposal_count = boxes_xyxy.shape[:2]
        logits = (self.first * 20 + self.second * 2).expand(
            batch_size, proposal_count, 1
        )
        if absolute_boxes_xyxy is None:
            absolute_boxes_xyxy = boxes_xyxy

        return RADMDenoiserOutput(
            logits=logits,
            boxes_xyxy=boxes_xyxy,
            pred_original_sample=boxes_xyxy,
            pred_noise=boxes_xyxy,
            auxiliary_logits=logits.unsqueeze(0),
            auxiliary_boxes_xyxy=boxes_xyxy.unsqueeze(0),
            auxiliary_boxes_absolute_xyxy=absolute_boxes_xyxy.unsqueeze(0),
        )


class _TwoBatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self._batches = [_training_batch(), _training_batch()]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._batches[index]


def _training_batch() -> dict[str, torch.Tensor]:
    return {
        "boxes_xyxy": torch.tensor([[[0.1, 0.1, 0.4, 0.4]]]),
        "mask": torch.tensor([[True]]),
        "labels": torch.tensor([[0]]),
        "image_scales": torch.full((1, 4), 10.0),
        "forward_image_scales": torch.full((1, 4), 10.0),
        "images": torch.zeros(1, 3, 8, 8),
        "text_features": torch.zeros(1, 1, 4),
        "text_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }


def test_lightning_optimizer_clips_current_step_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    effective = replace(
        effective_radm_config(),
        class_weight=1.0,
        l1_weight=0.0,
        giou_weight=0.0,
        learning_rate=0.01,
        weight_decay=0.01,
        warmup_factor=1.0,
        warmup_iters=0,
        milestones=(100, 200),
        max_iter=2,
        ota_k=1,
    )
    reference = RADMTrainingModule(
        model=cast(RADMDenoiser, _TinyRADMModel()), effective=effective
    )
    actual = RADMTrainingModule(
        model=cast(RADMDenoiser, _TinyRADMModel()), effective=effective
    )
    actual.model.load_state_dict(reference.model.state_dict())

    reference_optimizer = torch.optim.AdamW(
        reference.model.parameters(),
        lr=effective.learning_rate,
        betas=effective.betas,
        eps=effective.eps,
        weight_decay=effective.weight_decay,
    )
    reference_batch = _training_batch()
    for step in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = reference._compute_step_loss(
            reference_batch, record_trace=False
        )
        reference_loss.backward()
        preclip_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in reference.model.parameters()
                    if parameter.grad is not None
                ]
            )
        )
        if step == 0:
            assert preclip_norm > effective.gradient_clip_norm
        torch.nn.utils.clip_grad_norm_(
            reference.model.parameters(), effective.gradient_clip_norm
        )
        reference_optimizer.step()

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        max_steps=2,
        limit_train_batches=2,
        num_sanity_val_steps=0,
        gradient_clip_val=0.0,
    )
    trainer.fit(
        actual,
        train_dataloaders=DataLoader(_TwoBatchDataset(), batch_size=None),
    )

    for actual_parameter, expected_parameter in zip(
        actual.model.parameters(), reference.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter)
