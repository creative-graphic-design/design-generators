"""PyTorch Lightning module for LayoutDiffusion discrete training."""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float, Int, Shaped
from laygen.common.discrete import (
    index_to_log_onehot,
    log_onehot_to_index,
    update_loss_history,
)
from laygen.common.training import (
    finish_training_step,
    log_validation_loss,
    sum_loss_values,
)
from lightning.pytorch import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer

from ..configuration_layoutdiffusion import LayoutDiffusionConfig
from ..modeling_layoutdiffusion import LayoutDiffusionTransformer
from ..scheduling_layoutdiffusion import LayoutDiffusionScheduler
from ..tokenization_layoutdiffusion import LayoutDiffusionTokenizer
from .config import (
    LayoutDiffusionSeedMode,
    LayoutDiffusionTimeSampler,
    LayoutDiffusionTrainingScheduler,
)
from .losses import (
    log_categorical,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
    sum_except_batch,
)
from .vocab import build_training_tokenizer

EMA_CHECKPOINT_KEY = "layoutdiffusion_ema_state_dict"


class LayoutDiffusionTrainingModule(LightningModule):
    """Lightning wrapper reproducing LayoutDiffusion categorical diffusion training."""

    lt_history: Float[torch.Tensor, "timesteps"]
    lt_count: Float[torch.Tensor, "timesteps"]

    def __init__(
        self,
        *,
        config: LayoutDiffusionConfig,
        model: LayoutDiffusionTransformer | None = None,
        tokenizer: LayoutDiffusionTokenizer | None = None,
        vocab_file: str | None = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        auxiliary_loss_weight: float = 1e-3,
        time_sampler: LayoutDiffusionTimeSampler = "importance",
        scheduler: LayoutDiffusionTrainingScheduler | None = "linear_anneal",
        lr_anneal_steps: int = 400_000,
        ema_rate: float = 0.9999,
        seed_mode: LayoutDiffusionSeedMode | str = LayoutDiffusionSeedMode.default,
    ) -> None:
        """Initialize LayoutDiffusion training state."""
        super().__init__()
        self.tokenizer = tokenizer or build_training_tokenizer(
            config, vocab_file=vocab_file
        )
        self.layoutdiffusion_config = config
        self.model = model or LayoutDiffusionTransformer(
            vocab_size=config.vocab_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.diffusion_scheduler = LayoutDiffusionScheduler.from_layout_config(config)
        self.num_timesteps = config.diffusion_steps
        self.num_classes = config.vocab_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.time_sampler: LayoutDiffusionTimeSampler = time_sampler
        self.scheduler = scheduler
        self.lr_anneal_steps = lr_anneal_steps
        self.ema_rate = ema_rate
        self.seed_mode = LayoutDiffusionSeedMode(seed_mode)
        self.register_buffer("lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("lt_count", torch.zeros(self.num_timesteps))
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self._ema_params: dict[str, Shaped[torch.Tensor, "..."]] = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def on_fit_start(self) -> None:
        """Validate model/datamodule label order before training starts."""
        datamodule = getattr(self.trainer, "datamodule", None)
        data_config = getattr(datamodule, "config", None)
        data_id2label = getattr(data_config, "id2label", None)
        model_id2label = getattr(self.layoutdiffusion_config, "id2label", None)
        if data_id2label is None or model_id2label is None:
            return
        if dict(data_id2label) == dict(model_id2label):
            return
        data_first = next(iter(dict(data_id2label).items()), None)
        model_first = next(iter(dict(model_id2label).items()), None)
        raise ValueError(
            "LayoutDiffusion model/data id2label mismatch: "
            f"model first entry={model_first}, data first entry={data_first}"
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW and optional linear annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        if self.scheduler == "linear_anneal":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / self.lr_anneal_steps),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }
        return optimizer

    def _sample_time(
        self, batch_size: int, device: torch.device
    ) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        if self.time_sampler == "uniform":
            return sample_time_uniform(
                batch_size, num_timesteps=self.num_timesteps, device=device
            )
        if self.time_sampler == "importance":
            return sample_time_importance(
                batch_size,
                num_timesteps=self.num_timesteps,
                lt_history=self.lt_history,
                lt_count=self.lt_count,
            )
        raise ValueError(f"Unsupported time_sampler: {self.time_sampler}")

    def _q_sample(
        self, x_start: Int[torch.Tensor, "batch tokens"], t: Int[torch.Tensor, "batch"]
    ) -> tuple[
        Float[torch.Tensor, "batch vocab tokens"],
        Int[torch.Tensor, "batch tokens"],
    ]:
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_qpred = self.diffusion_scheduler.q_pred(log_x_start, t)
        log_x_t = self.diffusion_scheduler.log_sample_categorical(log_qpred)
        return log_x_t, log_onehot_to_index(log_x_t)

    def _diffusion_losses(
        self, x_start: Int[torch.Tensor, "batch tokens"], *, is_train: bool
    ) -> tuple[
        dict[str, Float[torch.Tensor, ""]],
        dict[str, Shaped[torch.Tensor, "..."]],
    ]:
        """Compute the LayoutDiffusion variational training loss."""
        batch_size, seq_length = x_start.shape
        t, pt = self._sample_time(batch_size, x_start.device)
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_x_t, xt = self._q_sample(x_start, t)

        logits = self.model(input_ids=xt, timesteps=t).logits
        log_x0_recon = self.diffusion_scheduler.predict_start(
            logits, batch_size, seq_length
        )
        log_model_prob = self.diffusion_scheduler.q_posterior(log_x0_recon, log_x_t, t)
        log_true_prob = self.diffusion_scheduler.q_posterior(log_x_start, log_x_t, t)

        mask_region = xt.eq(self.diffusion_scheduler.mask_token_id).float()
        mask_weight = mask_region + (1.0 - mask_region)
        kl = sum_except_batch(
            multinomial_kl(log_true_prob, log_model_prob) * mask_weight
        )
        decoder_nll = sum_except_batch(-log_categorical(log_x_start, log_model_prob))
        at_zero = (t == 0).float()
        kl_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl
        update_loss_history(
            kl_loss,
            t,
            self.lt_history,
            self.lt_count,
        )

        loss1 = kl_loss / pt
        losses: dict[str, Float[torch.Tensor, ""]] = {"kl_loss": loss1.mean()}
        aux_loss = torch.zeros_like(losses["kl_loss"])
        if self.auxiliary_loss_weight != 0 and is_train:
            kl_aux = sum_except_batch(
                multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
                * mask_weight
            )
            kl_aux_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl_aux
            adaptive_weight = 2.0 - t.float() / self.num_timesteps
            loss2 = adaptive_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            aux_loss = loss2.mean()
            losses["aux_loss"] = aux_loss

        trace: dict[str, Shaped[torch.Tensor, "..."]] = {
            "t": t.detach(),
            "pt": pt.detach(),
            "xt": xt.detach(),
            "log_x_t": log_x_t.detach(),
            "log_x0_recon": log_x0_recon.detach(),
            "log_model_prob": log_model_prob.detach(),
            "log_true_prob": log_true_prob.detach(),
            "kl": kl.detach(),
            "decoder_nll": decoder_nll.detach(),
            "kl_loss": kl_loss.detach(),
            "lt_history": self.lt_history.detach().clone(),
            "lt_count": self.lt_count.detach().clone(),
            "aux_loss": aux_loss.detach(),
        }
        return losses, trace

    def training_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one LayoutDiffusion training step."""
        del batch_idx
        seq = batch["input_ids"].long()
        losses, trace = self._diffusion_losses(seq, is_train=True)
        total, self.latest_step_trace = finish_training_step(self, losses, trace)
        return total

    def validation_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one LayoutDiffusion validation step."""
        del batch_idx
        seq = batch["input_ids"].long()
        losses, _ = self._diffusion_losses(seq, is_train=False)
        total = sum_loss_values(losses)
        log_validation_loss(self, total)
        return total

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Float[torch.Tensor, ""]] | None = None,
    ) -> None:
        """Run the optimizer step and update EMA parameters."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.update_ema()

    def update_ema(self) -> None:
        """Update exponential moving average parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                ema_param = self._ema_params[name].to(device=param.device)
                self._ema_params[name] = ema_param
                ema_param.mul_(self.ema_rate).add_(
                    param.detach(), alpha=1.0 - self.ema_rate
                )

    def ema_state_dict(self) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Return a detached copy of EMA parameters."""
        return {
            name: value.detach().clone() for name, value in self._ema_params.items()
        }

    def on_save_checkpoint(
        self,
        checkpoint: dict[
            str,
            dict[str, Shaped[torch.Tensor, "..."]]
            | Shaped[torch.Tensor, "..."]
            | int
            | float
            | str
            | bool
            | None,
        ],
    ) -> None:
        """Persist EMA parameters in Lightning checkpoints."""
        checkpoint[EMA_CHECKPOINT_KEY] = self.ema_state_dict()

    def on_load_checkpoint(
        self,
        checkpoint: dict[
            str,
            dict[str, Shaped[torch.Tensor, "..."]]
            | Shaped[torch.Tensor, "..."]
            | int
            | float
            | str
            | bool
            | None,
        ],
    ) -> None:
        """Restore EMA parameters from Lightning checkpoints when available."""
        ema_state = checkpoint.get(EMA_CHECKPOINT_KEY)
        if ema_state is None:
            return
        if not isinstance(ema_state, dict):
            raise TypeError(f"{EMA_CHECKPOINT_KEY} must be a dict")
        restored: dict[str, Shaped[torch.Tensor, "..."]] = {}
        for name, value in ema_state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{EMA_CHECKPOINT_KEY}[{name}] must be a tensor")
            restored[str(name)] = value.detach().clone()
        self._ema_params = restored
