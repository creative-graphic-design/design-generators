"""PyTorch Lightning module for LayoutDM discrete-diffusion training."""

from __future__ import annotations

from typing import Literal, cast

import torch
from jaxtyping import Float, Int, Shaped
from laygen.common.discrete import (
    index_to_log_onehot,
    log_sample_categorical,
    update_loss_history,
)
from laygen.common.training import (
    finish_training_step,
    log_validation_loss,
    sum_loss_values,
)
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from ..configuration_layout_dm import LayoutDMConfig
from ..modeling_layout_dm import LayoutDMDenoiser
from ..scheduling_layout_dm import LayoutDMScheduler
from ..tokenization_layout_dm import LayoutDMTokenizer
from .config import (
    LayoutDMSeedMode,
    LayoutDMTimeSampler,
    LayoutDMTrainingScheduler,
)
from .losses import (
    log_categorical,
    mean_except_batch,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
)


class LayoutDMTrainingModule(LightningModule):
    """Lightning wrapper reproducing LayoutDM categorical-diffusion training."""

    def __init__(
        self,
        *,
        config: LayoutDMConfig,
        model: LayoutDMDenoiser | None = None,
        tokenizer: LayoutDMTokenizer | None = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.98),
        auxiliary_loss_weight: float = 0.1,
        adaptive_auxiliary_loss: bool = True,
        time_sampler: LayoutDMTimeSampler = "importance",
        scheduler: LayoutDMTrainingScheduler | None = "reduce_on_plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        scheduler_threshold: float = 1e-2,
        seed_mode: LayoutDMSeedMode | str = LayoutDMSeedMode.default,
    ) -> None:
        """Initialize LayoutDM training state.

        Args:
            config: LayoutDM architecture and tokenizer configuration.
            model: Optional pre-built denoiser. Built from ``config`` otherwise.
            tokenizer: Optional pre-built tokenizer. Built from ``config``
                otherwise.
            learning_rate: AdamW learning rate.
            weight_decay: Weight decay applied to the decay parameter group.
            betas: AdamW beta coefficients.
            auxiliary_loss_weight: Weight of the auxiliary cross-entropy term.
            adaptive_auxiliary_loss: Whether to scale the auxiliary term by the
                per-timestep adaptive weight.
            time_sampler: Timestep-sampling strategy.
            scheduler: Optional learning-rate scheduler name.
            scheduler_factor: ``ReduceLROnPlateau`` multiplicative factor.
            scheduler_patience: ``ReduceLROnPlateau`` patience in epochs.
            scheduler_threshold: ``ReduceLROnPlateau`` improvement threshold.
            seed_mode: Regular or deterministic seed mode.
        """
        super().__init__()
        self.layout_dm_config = config
        self.model = model or LayoutDMDenoiser(
            vocab_size=config.vocab_size,
            max_token_length=config.max_token_length,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            timestep_type=cast(
                'Literal["adalayernorm", "adalayernorm_abs"] | None',
                config.timestep_type,
            ),
        )
        self.tokenizer = tokenizer or LayoutDMTokenizer(config)
        self.var_order = tuple(config.var_order.split("-"))
        per_var_full_ids = (
            self.tokenizer.full_id_maps() if config.q_type == "constrained" else None
        )
        self.diffusion_scheduler = LayoutDMScheduler(
            num_timesteps=config.num_timesteps,
            q_type=config.q_type,  # type: ignore[arg-type]
            vocab_size=config.vocab_size,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
            var_order=self.var_order,
            per_var_full_ids=per_var_full_ids,
            att_1=config.att_1,
            att_T=config.att_T,
            ctt_1=config.ctt_1,
            ctt_T=config.ctt_T,
        )
        self.num_timesteps = config.num_timesteps
        self.num_classes = config.vocab_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.time_sampler: LayoutDMTimeSampler = time_sampler
        self.scheduler = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.seed_mode = LayoutDMSeedMode(seed_mode)
        self.mat_size = {key: len(ids) for key, ids in (per_var_full_ids or {}).items()}
        self.register_buffer("lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("lt_count", torch.zeros(self.num_timesteps))
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}

    # -- optimization ---------------------------------------------------------

    def optim_groups(self) -> list[dict[str, list[nn.Parameter] | float]]:
        """Split parameters into weight-decayed and decay-free groups."""
        decay: set[str] = set()
        no_decay: set[str] = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for module_name, module in self.model.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full)
                elif param_name.endswith("weight") and isinstance(module, whitelist):
                    decay.add(full)
                elif param_name.endswith("weight") and isinstance(module, blacklist):
                    no_decay.add(full)
                else:
                    no_decay.add(full)
        params = dict(self.model.named_parameters())
        inter = decay & no_decay
        assert not inter, f"parameters {inter} in both decay/no_decay groups"
        missing = set(params) - (decay | no_decay)
        assert not missing, f"parameters {missing} were not assigned a group"
        return [
            {
                "params": [params[name] for name in sorted(decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [params[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW and an optional ``ReduceLROnPlateau`` scheduler."""
        optimizer = torch.optim.AdamW(
            self.optim_groups(), lr=self.learning_rate, betas=self.betas
        )
        if self.scheduler == "reduce_on_plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": plateau,
                    "monitor": "val_loss",
                    "interval": "epoch",
                },
            }
        return optimizer

    # -- diffusion loss -------------------------------------------------------

    def _sample_time(
        self, batch_size: int, device: torch.device
    ) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        if self.time_sampler == "uniform":
            return sample_time_uniform(
                batch_size, num_timesteps=self.num_timesteps, device=device
            )
        return sample_time_importance(
            batch_size,
            num_timesteps=self.num_timesteps,
            lt_history=cast(torch.Tensor, self.lt_history),
            lt_count=cast(torch.Tensor, self.lt_count),
        )

    def _q_sample_full(
        self, x_start: Int[torch.Tensor, "batch tokens"], t: Int[torch.Tensor, "batch"]
    ) -> tuple[
        Float[torch.Tensor, "batch vocab tokens"],
        Int[torch.Tensor, "batch tokens"],
    ]:
        """Draw ``x_t`` from the forward diffusion posterior per variable."""
        if self.layout_dm_config.q_type == "vanilla":
            log_x_start = index_to_log_onehot(x_start, self.num_classes)
            log_qpred = self.diffusion_scheduler._vanilla_q_pred(log_x_start, t)
            xt = log_sample_categorical(log_qpred)
            return index_to_log_onehot(xt, self.num_classes), xt
        batch_size = x_start.shape[0]
        step = len(self.var_order)
        seq_len = x_start.shape[1] // step
        reshaped = x_start.reshape(batch_size, seq_len, step)
        log_xt_full_parts: list[Float[torch.Tensor, "batch vocab tokens"]] = []
        xt_full_parts: list[Int[torch.Tensor, "batch tokens"]] = []
        for i, key in enumerate(self.var_order):
            col_full = reshaped[..., i]
            col_partial = self.tokenizer.full_to_partial_ids(col_full, key)
            log_x_start = index_to_log_onehot(col_partial, self.mat_size[key])
            log_qpred = self.diffusion_scheduler._q_pred(log_x_start, t, key)
            sampled = log_sample_categorical(log_qpred)
            log_xt = index_to_log_onehot(sampled, self.mat_size[key])
            log_xt_full_parts.append(
                self.tokenizer.partial_to_full_log_probs(log_xt, key)
            )
            xt_full_parts.append(self.tokenizer.partial_to_full_ids(sampled, key))
        log_x_t_full = torch.stack(log_xt_full_parts, dim=-1).reshape(
            batch_size, self.num_classes, -1
        )
        xt_full = torch.stack(xt_full_parts, dim=-1).reshape(batch_size, -1)
        return log_x_t_full, xt_full

    def _diffusion_losses(
        self, x_start: Int[torch.Tensor, "batch tokens"], *, is_train: bool
    ) -> tuple[
        dict[str, Float[torch.Tensor, ""]],
        dict[str, Shaped[torch.Tensor, "..."]],
    ]:
        """Compute the LayoutDM variational loss for a token batch."""
        batch_size = x_start.shape[0]
        t, pt = self._sample_time(batch_size, x_start.device)

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_x_t, xt = self._q_sample_full(x_start, t)

        denoiser_logits = self.model(input_ids=xt, timesteps=t).logits
        log_x0_recon = self.diffusion_scheduler.predict_start(denoiser_logits)
        log_model_prob = self.diffusion_scheduler.q_posterior(log_x0_recon, log_x_t, t)
        log_true_prob = self.diffusion_scheduler.q_posterior(log_x_start, log_x_t, t)

        kl = multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region + (1.0 - mask_region)
        kl = mean_except_batch(kl * mask_weight)

        decoder_nll = mean_except_batch(-log_categorical(log_x_start, log_model_prob))
        at_zero = (t == 0).float()
        kl_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl

        update_loss_history(
            kl_loss,
            t,
            cast(torch.Tensor, self.lt_history),
            cast(torch.Tensor, self.lt_count),
        )

        losses: dict[str, Float[torch.Tensor, ""]] = {"kl_loss": (kl_loss / pt).mean()}
        if self.auxiliary_loss_weight != 0 and is_train:
            kl_aux = multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = mean_except_batch(kl_aux * mask_weight)
            kl_aux_loss = at_zero * decoder_nll + (1.0 - at_zero) * kl_aux
            weight = (1 - t / self.num_timesteps) + 1.0
            if not self.adaptive_auxiliary_loss:
                weight = torch.ones_like(weight)
            losses["aux_loss"] = (
                weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            ).mean()

        trace: dict[str, Shaped[torch.Tensor, "..."]] = {
            "t": t.detach(),
            "pt": pt.detach(),
            "xt": xt.detach(),
            "log_model_prob": log_model_prob.detach(),
            "kl": kl.detach(),
            "decoder_nll": decoder_nll.detach(),
            "kl_loss": kl_loss.detach(),
            **{key: value.detach() for key, value in losses.items()},
        }
        return losses, trace

    def training_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one LayoutDM training step."""
        del batch_idx
        seq = batch["input_ids"].long()
        losses, trace = self._diffusion_losses(seq, is_train=True)
        total, self.latest_step_trace = finish_training_step(self, losses, trace)
        return total

    def validation_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one LayoutDM validation step."""
        del batch_idx
        seq = batch["input_ids"].long()
        losses, _ = self._diffusion_losses(seq, is_train=True)
        total = sum_loss_values(losses)
        log_validation_loss(self, total)
        return total
