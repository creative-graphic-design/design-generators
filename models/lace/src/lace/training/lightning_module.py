"""Lightning training module matching the effective LACE recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from ..configuration_lace import default_model_config, get_dataset_spec
from ..modeling_lace import LaceTransformerModel
from ..scheduling_lace import LaceScheduler
from .config import LaceSeedMode, LaceTrainingDatasetName
from .ema import LaceEMA
from .losses import lace_losses, rand_fix
from .seed import apply_lace_seed_mode


class LaceTrainingModule(LightningModule):
    """Lightning wrapper for the original plain-PyTorch training loop."""

    def __init__(
        self,
        *,
        dataset_name: LaceTrainingDatasetName = "publaynet",
        model: LaceTransformerModel | None = None,
        learning_rate: float = 1e-5,
        batch_size: int = 256,
        sample_t_max: int = 999,
        feature_dim: int = 2048,
        dim_transformer: int = 1024,
        nhead: int = 16,
        num_layers: int = 4,
        ema_mu: float = 0.9999,
        seed_mode: LaceSeedMode | str = LaceSeedMode.default,
        seed: int = 42975,
    ) -> None:
        """Initialize model, diffusion buffers, and active recipe values."""
        super().__init__()

        spec = get_dataset_spec(dataset_name)
        config = default_model_config(dataset_name)
        config["dim_transformer"] = dim_transformer
        config["dim_feedforward"] = feature_dim
        config["nhead"] = nhead
        config["num_layers"] = num_layers

        self.dataset_name = dataset_name
        self.num_classes = spec.num_classes_with_pad
        self.seq_dim = spec.seq_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_t_max = sample_t_max
        self.ema_mu = ema_mu
        self.seed_mode = LaceSeedMode(seed_mode)
        self.seed = seed

        self.model = model or LaceTransformerModel(**config)
        self.ema_helper = LaceEMA(mu=ema_mu)
        self.ema_helper.register(self.model)

        schedule = LaceScheduler(num_train_timesteps=1000, ddim_num_steps=200)
        model_device = next(self.model.parameters()).device
        alphas_cumprod = schedule.alphas_cumprod_for_device(model_device)

        self.register_buffer("alphas_bar_sqrt", alphas_cumprod.sqrt())
        self.register_buffer("one_minus_alphas_bar_sqrt", (1 - alphas_cumprod).sqrt())
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self.latest_ema_state: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self.save_hyperparameters(ignore=("model",))

    def on_fit_start(self) -> None:
        """Apply the configured seed policy at the training boundary."""
        apply_lace_seed_mode(self.seed_mode, seed=self.seed)
        alphas_cumprod = LaceScheduler(
            num_train_timesteps=1000
        ).alphas_cumprod_for_device(cast(torch.Tensor, self.alphas_bar_sqrt).device)
        cast(torch.Tensor, self.alphas_bar_sqrt).copy_(alphas_cumprod.sqrt())
        cast(torch.Tensor, self.one_minus_alphas_bar_sqrt).copy_(
            (1 - alphas_cumprod).sqrt()
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the reference unscheduled Adam optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=1e-8,
        )

    def on_train_batch_end(
        self,
        outputs: Float[torch.Tensor, ""]
        | Mapping[str, Shaped[torch.Tensor, "..."]]
        | None,
        batch: Mapping[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> None:
        """Update EMA after Lightning completes the optimizer step."""
        del outputs, batch, batch_idx
        self.ema_helper.update(self.model)
        self.latest_ema_state = self.ema_helper.state_dict()

    def forward_t(
        self,
        layout_input: Float[torch.Tensor, "batch elements channels"],
        mask: Bool[torch.Tensor, "batch elements"],
        *,
        timestep: Int[torch.Tensor, "batch"],
    ) -> tuple[
        Float[torch.Tensor, "four_batch elements channels"],
        Float[torch.Tensor, "four_batch elements channels"],
        Float[torch.Tensor, "four_batch elements channels"],
        Int[torch.Tensor, "four_batch"],
    ]:
        """Run the four effective conditioning branches in reference order."""
        noise = torch.randn_like(layout_input)
        sqrt_alpha = cast(torch.Tensor, self.alphas_bar_sqrt)[timestep].reshape(
            -1, 1, 1
        )
        sqrt_one_minus = cast(torch.Tensor, self.one_minus_alphas_bar_sqrt)[
            timestep
        ].reshape(-1, 1, 1)
        noisy = sqrt_alpha * layout_input + sqrt_one_minus * noise
        conditioned_labels = layout_input.clone()
        conditioned_labels[:, :, self.num_classes :] = noisy[:, :, self.num_classes :]
        conditioned_centers = layout_input.clone()
        conditioned_centers[:, :, self.num_classes : self.num_classes + 2] = noisy[
            :, :, self.num_classes : self.num_classes + 2
        ]
        fixed = rand_fix(layout_input.shape[0], mask)
        conditioned_complete = noisy.clone()
        conditioned_complete[fixed] = layout_input[fixed]
        all_inputs = torch.cat(
            (noisy, conditioned_labels, conditioned_centers, conditioned_complete),
            dim=0,
        )
        all_noise = torch.cat((noise,) * 4, dim=0)
        all_timesteps = torch.cat((timestep,) * 4, dim=0)
        output = self.model(sample=all_inputs, timestep=all_timesteps).sample
        sqrt_one_minus_all = cast(torch.Tensor, self.one_minus_alphas_bar_sqrt)[
            all_timesteps
        ].reshape(-1, 1, 1)
        sqrt_alpha_all = (1 - sqrt_one_minus_all.square()).sqrt()
        reconstructed = (
            1 / sqrt_alpha_all * (all_inputs - output * sqrt_one_minus_all)
        ).to(all_inputs.device)
        return output, all_noise, reconstructed, all_timesteps

    def training_step(
        self,
        batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Run one package-local diffusion and constraint step."""
        del batch_idx
        bbox = cast(Float[torch.Tensor, "batch elements 4"], batch["bbox"]).float()
        labels = cast(Int[torch.Tensor, "batch elements"], batch["labels"]).long()
        mask = cast(Bool[torch.Tensor, "batch elements"], batch["mask"]).bool()
        labels = labels.clone()
        labels[~mask] = self.num_classes - 1
        bbox_in = 2 * (bbox - 0.5)
        one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes).to(
            dtype=bbox_in.dtype
        )
        layout_input = torch.cat((one_hot, bbox_in), dim=2)
        timestep = torch.randint(
            low=0,
            high=self.sample_t_max,
            size=(bbox.shape[0],),
            device=bbox.device,
        )
        output, noise, reconstructed, all_timesteps = self.forward_t(
            layout_input, mask, timestep=timestep
        )
        losses = lace_losses(
            layout_input=layout_input,
            model_output=output,
            noise=noise,
            reconstructed=reconstructed,
            bbox=torch.cat((bbox,) * 4, dim=0),
            mask=torch.cat((mask,) * 4, dim=0),
            timesteps=all_timesteps,
            num_classes=self.num_classes,
        )
        self.log("train_loss", losses["train_loss"], on_step=True, on_epoch=True)
        self.latest_step_trace = {
            "bbox": bbox.detach(),
            "labels": labels.detach(),
            "mask": mask.detach(),
            "layout_input": layout_input.detach(),
            "timestep": timestep.detach(),
            "noise": noise.detach(),
            "model_output": output.detach(),
            "reconstructed": reconstructed.detach(),
            "all_timesteps": all_timesteps.detach(),
            **{key: value.detach() for key, value in losses.items()},
        }
        return losses["train_loss"]
