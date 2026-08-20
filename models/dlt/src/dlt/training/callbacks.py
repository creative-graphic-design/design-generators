"""Training callbacks for DLT reference-recipe reproduction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch
from jaxtyping import Float, Int
from lightning.pytorch import Callback, LightningModule, Trainer

from .dataset import (
    _mask_all,
    _mask_loc,
    _mask_random,
    _mask_size,
    _mask_whole_box,
    _pad_instance,
)

if TYPE_CHECKING:
    from .lightning_module import DLTTrainingModule


class _ReferenceLayoutDataset(Protocol):
    max_num_comp: int

    def __len__(self) -> int: ...

    def get_data_by_ix(
        self, index: int
    ) -> tuple[
        Float[np.ndarray, "elements 4"],
        Int[np.ndarray, "elements"],
        list[int],
        str,
    ]: ...


class _JointScheduler(Protocol):
    num_cont_steps: int

    def step_jointly(
        self,
        cont_output: Float[torch.Tensor, "batch elements 4"],
        cat_output: dict[str, Float[torch.Tensor, "batch elements categories"]],
        timestep: Int[torch.Tensor, "batch"],
        sample: Float[torch.Tensor, "batch elements 4"],
    ) -> tuple[
        "_JointSchedulerOutput", dict[str, Int[torch.Tensor, "batch elements"]]
    ]: ...


class _JointSchedulerOutput(Protocol):
    prev_sample: Float[torch.Tensor, "batch elements 4"]
    pred_original_sample: Float[torch.Tensor, "batch elements 4"]


class DLTReferenceEpochSamplingCallback(Callback):
    """Consume the reference recipe's per-epoch sampling RNG."""

    def __init__(self, *, num_samples: int = 5) -> None:
        """Store the number of validation layouts sampled after each epoch."""
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run reference-style validation sampling after a training epoch."""
        datamodule = getattr(trainer, "datamodule", None)
        val_data = getattr(datamodule, "val_dataset", None)
        if val_data is None:
            raise RuntimeError(
                "DLTReferenceEpochSamplingCallback requires a datamodule with "
                "a prepared val_dataset"
            )

        consume_reference_epoch_sampling_rng(
            pl_module,
            val_data,
            num_samples=self.num_samples,
        )


def consume_reference_epoch_sampling_rng(
    pl_module: LightningModule,
    val_data: _ReferenceLayoutDataset,
    *,
    num_samples: int = 5,
) -> None:
    """Consume reference post-epoch sampling RNG without logging images."""
    if num_samples > len(val_data):
        raise ValueError("num_samples cannot exceed validation dataset length")

    module = cast("DLTTrainingModule", pl_module)
    model = cast(torch.nn.Module, module.model)
    scheduler = cast(_JointScheduler, module.scheduler)
    categories_num = int(module.dlt_config.categories_num)
    device = torch.device(module.device)
    was_training = model.training
    model.eval()
    try:
        indices = np.random.choice(range(len(val_data)), num_samples, replace=False)
        for sample_index, layout_index in enumerate(indices):
            sample = _reference_condition_sample(
                val_data,
                int(layout_index),
                sample_index,
                device=device,
            )
            _sample_from_model(
                sample,
                model,
                scheduler,
                categories_num=categories_num,
                device=device,
            )
    finally:
        if was_training:
            model.train()


def _reference_condition_sample(
    val_data: _ReferenceLayoutDataset,
    index: int,
    sample_index: int,
    *,
    device: torch.device,
) -> dict[
    str,
    Float[torch.Tensor, "batch elements 4"]
    | Int[torch.Tensor, "batch elements"]
    | Int[torch.Tensor, "batch elements 4"],
]:
    box, cat, _, _ = val_data.get_data_by_ix(index)
    if sample_index == 0:
        mask, mask_cat = _mask_loc(box.shape, r_mask=1.0)
    elif sample_index == 1:
        mask, mask_cat = _mask_size(box.shape, r_mask=1.0)
    elif sample_index == 2:
        mask, mask_cat = _mask_whole_box(box.shape, r_mask=1.0)
    elif sample_index == 3:
        mask, mask_cat = _mask_random(
            box.shape,
            r_mask_box=np.random.uniform(0.5, 1.0, size=1)[0],
            r_mask_cat=np.random.uniform(0.5, 1.0, size=1)[0],
        )
    else:
        mask, mask_cat = _mask_all(box.shape)

    box, cat, mask, mask_cat = _pad_instance(
        box, cat, mask, mask_cat, val_data.max_num_comp
    )
    return {
        "box": torch.tensor(box.astype(np.float32), device=device).unsqueeze(0),
        "cat": torch.tensor(cat.astype(int), device=device).unsqueeze(0),
        "mask_box": torch.tensor(mask.astype(int), device=device).unsqueeze(0),
        "mask_cat": torch.tensor(mask_cat.astype(int), device=device).unsqueeze(0),
        "box_cond": torch.tensor(
            box.copy().astype(np.float32), device=device
        ).unsqueeze(0),
    }


def _sample_from_model(
    sample: dict[
        str,
        Float[torch.Tensor, "batch elements 4"]
        | Int[torch.Tensor, "batch elements"]
        | Int[torch.Tensor, "batch elements 4"],
    ],
    model: torch.nn.Module,
    scheduler: _JointScheduler,
    *,
    categories_num: int,
    device: torch.device,
) -> tuple[
    Float[torch.Tensor, "batch elements 4"],
    Int[torch.Tensor, "batch elements"],
]:
    shape = sample["box_cond"].shape
    noisy_batch = {
        "box": torch.randn(*shape, dtype=torch.float32, device=device),
        "cat": (categories_num - 1)
        * torch.ones((shape[0], shape[1]), dtype=torch.long, device=device),
    }
    bbox_pred = None
    cat_pred = None
    for i in range(scheduler.num_cont_steps - 1, -1, -1):
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            pred_box, pred_cat = model(sample, noisy_batch, timesteps=t)
            bbox_pred, cat_pred = scheduler.step_jointly(
                pred_box,
                {"cat": pred_cat},
                timestep=t,
                sample=noisy_batch["box"],
            )
            noisy_batch["box"] = bbox_pred.prev_sample
            noisy_batch["cat"] = cat_pred["cat"]
    if bbox_pred is None or cat_pred is None:
        raise RuntimeError("Reference epoch sampling did not run")

    return bbox_pred.pred_original_sample, cat_pred["cat"]
