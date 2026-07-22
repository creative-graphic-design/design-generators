"""PyTorch Lightning module for LayoutFlow training."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from layout_flow.configuration_layout_flow import LayoutFlowConfig
from layout_flow.modeling_layout_flow import LayoutFlowTransformerModel
from layout_flow.processing_layout_flow import LayoutFlowProcessor

from .config import LayoutFlowSeedMode
from .losses import layout_flow_losses


class LayoutFlowTrainingModule(LightningModule):
    """Lightning training wrapper around ``LayoutFlowTransformerModel``."""

    def __init__(
        self,
        *,
        config: LayoutFlowConfig | None = None,
        model: LayoutFlowTransformerModel | None = None,
        dataset_name: str = "publaynet",
        learning_rate: float = 0.0005,
        scheduler: str | None = "reduce_on_plateau",
        condition_policy: str = "random4",
        geom_l1_weight: float = 0.2,
        seed_mode: LayoutFlowSeedMode | str = LayoutFlowSeedMode.vendor_compat,
        fid_calc_every_n: int = 20,
    ) -> None:
        """Initialize LayoutFlow training state."""
        super().__init__()
        self.layout_flow_config = config or LayoutFlowConfig(dataset_name=dataset_name)
        self.model = model or LayoutFlowTransformerModel(
            num_labels=self.layout_flow_config.num_labels,
            latent_dim=self.layout_flow_config.latent_dim,
            tr_enc_only=self.layout_flow_config.tr_enc_only,
            d_model=self.layout_flow_config.d_model,
            nhead=self.layout_flow_config.nhead,
            dim_feedforward=self.layout_flow_config.dim_feedforward,
            num_layers=self.layout_flow_config.num_layers,
            dropout=self.layout_flow_config.dropout,
            use_pos_enc=self.layout_flow_config.use_pos_enc,
            attr_encoding=self.layout_flow_config.attr_encoding,
            seq_type=self.layout_flow_config.seq_type,
        )
        self.processor = LayoutFlowProcessor(self.layout_flow_config)
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.condition_policy = condition_policy
        self.geom_l1_weight = geom_l1_weight
        self.seed_mode = LayoutFlowSeedMode(seed_mode)
        self.fid_calc_every_n = fid_calc_every_n
        self.geom_dim = 4
        self.attr_dim = self.layout_flow_config.attr_dim
        self.latest_step_trace: dict[str, torch.Tensor] = {}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return vendor-compatible AdamW and optional ReduceLROnPlateau."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)
        )
        if self.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "FID_Layout",
                    "frequency": self.fid_calc_every_n,
                },
            }
        return optimizer

    def forward(
        self,
        xt: Float[torch.Tensor, "batch elements channels"],
        cond_mask: Int[torch.Tensor, "batch elements channels"],
        timestep: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch elements channels"]:
        """Predict the vector field for LayoutFlow training."""
        return self.model(sample=xt, timestep=timestep, cond_mask=cond_mask).sample

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one vendor-compatible training step."""
        del batch_idx
        prepared = self._prepare_batch(batch)
        cond_mask = self.random4_condition_mask(
            prepared["length"], prepared["bbox"].shape[1]
        )
        x0, x1 = self.get_start_end(prepared)
        t = self.sample_t(x0)
        xt, ut = self.sample_xt(prepared, x0, x1, cond_mask, t)
        vt = self(xt, cond_mask, t.squeeze(-1))
        losses = layout_flow_losses(
            cond_mask.to(vt.dtype),
            ut,
            vt,
            geom_dim=self.geom_dim,
            geom_l1_weight=self.geom_l1_weight,
        )
        for key, value in losses.items():
            self.log(
                key, value, prog_bar=key == "train_loss", on_step=True, on_epoch=True
            )
        self.latest_step_trace = {
            "bbox": prepared["bbox"].detach(),
            "type": prepared["type"].detach(),
            "mask": prepared["mask"].detach(),
            "length": prepared["length"].detach(),
            "cond_mask": cond_mask.detach(),
            "x0": x0.detach(),
            "x1": x1.detach(),
            "t": t.detach(),
            "xt": xt.detach(),
            "ut": ut.detach(),
            "vt": vt.detach(),
            **{key: value.detach() for key, value in losses.items()},
        }
        return losses["train_loss"]

    def get_start_end(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        Float[torch.Tensor, "batch elements channels"],
        Float[torch.Tensor, "batch elements channels"],
    ]:
        """Return vendor ``x0`` and preprocessed data sample ``x1``."""
        bbox = batch["bbox"]
        labels = batch["type"].long()
        conv_type = self.processor.encode_labels(labels)
        gt = torch.cat([bbox, conv_type], dim=-1)
        x0 = torch.zeros_like(gt)
        if self.layout_flow_config.distribution != "gaussian":
            raise ValueError(
                "LayoutFlow training parity currently supports gaussian x0 sampling"
            )
        for i, length_tensor in enumerate(batch["length"]):
            length = int(length_tensor.item())
            x0[i, :length] = torch.randn(
                length, gt.shape[-1], device=gt.device, dtype=gt.dtype
            )
        x1 = self.processor.preprocess_state(gt)
        mask = batch["mask"].unsqueeze(-1)
        x1 = mask * x1 + (~mask) * gt
        return x0, x1

    def sample_t(
        self, x0: Float[torch.Tensor, "batch elements channels"]
    ) -> Float[torch.Tensor, "batch"]:
        """Sample vendor uniform training times."""
        return torch.rand(x0.shape[0]).type_as(x0)

    def sample_xt(
        self,
        batch: dict[str, torch.Tensor],
        x0: Float[torch.Tensor, "batch elements channels"],
        x1: Float[torch.Tensor, "batch elements channels"],
        cond_mask: Int[torch.Tensor, "batch elements channels"],
        t: Float[torch.Tensor, "batch"],
    ) -> tuple[
        Float[torch.Tensor, "batch elements channels"],
        Float[torch.Tensor, "batch elements channels"],
    ]:
        """Return vendor linear ``x_t`` and vector field ``u_t``."""
        del batch
        tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
        xt = (1 - tpad) * x0 + tpad * x1
        ut = x1 - x0
        cond = cond_mask.to(dtype=xt.dtype)
        xt = (1 - cond) * x1 + cond * xt
        return xt, ut

    def random4_condition_mask(
        self,
        lengths: Int[torch.Tensor, "batch"],
        seq_len: int,
    ) -> Int[torch.Tensor, "batch elements channels"]:
        """Return the vendor ``random4`` condition mask."""
        batch = lengths.shape[0]
        device = lengths.device
        cond_mask = torch.ones(
            batch,
            seq_len,
            self.geom_dim + self.attr_dim,
            dtype=torch.int,
            device=device,
        )
        if self.condition_policy != "random4":
            raise ValueError(f"Unsupported condition_policy: {self.condition_policy}")
        div = batch // 4
        for i, length_tensor in enumerate(lengths[:div]):
            length = int(length_tensor.item())
            n = length * 0.2 * torch.rand(1).to(device)
            if length > 1:
                idx = torch.multinomial(
                    torch.arange(length).float(),
                    int(n.item() + 1),
                ).to(device)
                cond_mask[i, idx] = 0
        cond_mask[div : 2 * div, :, self.geom_dim :] = 0
        cond_mask[2 * div : 3 * div, :, 2:] = 0
        return cond_mask

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bbox = batch["bbox"].float()
        labels = batch.get("type", batch.get("labels"))
        if labels is None:
            raise ValueError("LayoutFlow training batch requires 'type' labels")
        mask = batch["mask"]
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        return {
            "bbox": bbox,
            "type": labels.long(),
            "mask": mask.bool(),
            "length": batch["length"].long(),
        }
