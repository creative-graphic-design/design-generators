"""Categorical Gaussian-refine scheduler for LayoutDiffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from laygen.common.discrete import (
    gumbel_noise_like,
    index_to_log_onehot,
    log_add_exp,
    log_onehot_to_index,
)

from .sampling import LayoutDiffusionSamplingConfig, LayoutDiffusionSamplingName

if TYPE_CHECKING:
    from .conditioning import LayoutDiffusionCondition


@dataclass
class LayoutDiffusionSchedulerOutput(BaseOutput):
    """Output of one LayoutDiffusion scheduler step."""

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor | None = None
    model_log_prob: torch.Tensor | None = None


class LayoutDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """Diffusers scheduler for LayoutDiffusion categorical transitions.

    Args:
        num_train_timesteps: Number of training diffusion steps.
        vocab_size: Full vocabulary size including mask.
        mask_token_id: Mask token id.
        type_classes: Number of label/type classes.
        num_special_tokens: Number of leading special tokens.
        num_coordinate_bins: Coordinate vocabulary size.
        noise_schedule: Vendor schedule name.
        pow_num: Gaussian transition exponent.
        mul_num: Gaussian transition multiplier.
        type_start_step: Label-conditioned start step.
        rico_refine_start_step: RICO refinement start step.
        publaynet_refine_start_step: PubLayNet refinement start step.

    Examples:
        >>> scheduler = LayoutDiffusionScheduler(vocab_size=139, mask_token_id=138, type_classes=5)
        >>> scheduler.q_mats.shape[-2:]
        torch.Size([128, 128])
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_train_timesteps: int = 200,
        vocab_size: int,
        mask_token_id: int,
        type_classes: int,
        num_special_tokens: int = 5,
        num_coordinate_bins: int = 128,
        noise_schedule: str = "gaussian_refine_pow2.5",
        pow_num: float = 2.5,
        mul_num: float = 12.4,
        type_start_step: int = 160,
        rico_refine_start_step: int = 50,
        publaynet_refine_start_step: int = 60,
    ) -> None:
        """Initialize scheduler buffers."""
        self.num_timesteps = num_train_timesteps
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.type_classes = type_classes
        self.num_special_tokens = num_special_tokens
        self.num_coordinate_bins = num_coordinate_bins
        self.noise_schedule = noise_schedule
        self.pow_num = pow_num
        self.mul_num = mul_num
        self.type_start_step = type_start_step
        self.rico_refine_start_step = rico_refine_start_step
        self.publaynet_refine_start_step = publaynet_refine_start_step
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self._init_buffers()

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        *,
        start_step: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set reverse diffusion timesteps."""
        start = self.num_timesteps if start_step is None else start_step
        steps = num_inference_steps or start
        values = [int(i * start / steps) for i in range(steps - 1, -1, -1)]
        if values[-1] != 0:
            values.append(0)
        self.timesteps = torch.tensor(values, dtype=torch.long, device=device)

    def predict_start(
        self,
        logits: torch.Tensor,
        batch_size: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Append the fixed mask logit and clamp model log probabilities."""
        log_pred = torch.log_softmax(logits.double(), dim=1).float()
        zero = (
            torch.zeros(
                batch_size, 1, seq_length, device=logits.device, dtype=logits.dtype
            )
            - 70
        )
        return torch.cat((log_pred, zero), dim=1).clamp(-70.0, 0.0)

    def q_pred_one_timestep(
        self, log_x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``q(x_t | x_{t-1})``."""
        matrix = self._transition_matrix(t, cumulative=False, device=log_x_t.device)
        return matrix.matmul(log_x_t.exp()).clamp(min=1e-30).log()

    def q_pred(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute cumulative ``q(x_t | x_0)``."""
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        matrix = self._transition_matrix(t, cumulative=True, device=log_x_start.device)
        return matrix.matmul(log_x_start.exp()).clamp(min=1e-30).log()

    def q_posterior(
        self,
        log_x_start: torch.Tensor,
        log_x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``p_theta(x_{t-1} | x_t)`` from predicted start logits."""
        if t.min().item() < 0 or t.max().item() >= self.num_timesteps:
            raise ValueError("timestep outside scheduler range")
        batch_size = log_x_start.shape[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = onehot_x_t.eq(self.mask_token_id).unsqueeze(1)
        log_one = torch.zeros(
            batch_size, 1, 1, device=log_x_t.device, dtype=log_x_t.dtype
        )
        log_zero = torch.log(log_one + 1.0e-30).expand(-1, -1, log_x_t.shape[-1])
        log_zero_aux = torch.log(log_one + 1.0e-30).expand(-1, -1, -1)

        log_qt = self.q_pred(log_x_t, t)[:, :-1, :]
        log_cumprod_ct = _extract(
            self.log_cumprod_ct.to(t.device), t, log_x_start.shape
        )
        ct_cumprod = torch.cat(
            [
                log_zero_aux.expand(-1, self.num_special_tokens, -1),
                log_cumprod_ct.expand(
                    -1, self.vocab_size - 1 - self.num_special_tokens, -1
                ),
            ],
            dim=1,
        )
        log_qt = (~mask) * log_qt + mask * ct_cumprod

        log_qt_one = self.q_pred_one_timestep(log_x_t, t)
        log_qt_one = torch.cat((log_qt_one[:, :-1, :], log_zero), dim=1)
        log_ct = _extract(self.log_ct.to(t.device), t, log_x_start.shape)
        ct_vector = torch.cat(
            [
                log_zero_aux.expand(-1, self.num_special_tokens, -1),
                log_ct.expand(-1, self.vocab_size - 1 - self.num_special_tokens, -1),
            ],
            dim=1,
        )
        ct_vector = torch.cat((ct_vector, log_one), dim=1)
        log_qt_one = (~mask) * log_qt_one + mask * ct_vector
        q = torch.cat((log_x_start[:, :-1, :] - log_qt, log_zero), dim=1)
        posterior = self.q_pred(q, t - 1) + log_qt_one
        return posterior.clamp(-70.0, 0.0)

    def step(
        self,
        logits: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        sampling: LayoutDiffusionSamplingConfig | None = None,
        condition: LayoutDiffusionCondition | None = None,
        generator: torch.Generator | None = None,
    ) -> LayoutDiffusionSchedulerOutput:
        """Run one reverse diffusion step."""
        _ = condition
        config = sampling or LayoutDiffusionSamplingConfig()
        log_x_recon = self.predict_start(logits, sample.shape[0], sample.shape[-1])
        model_log_prob = self.q_posterior(log_x_recon, sample, timestep)
        if str(config.name) == str(LayoutDiffusionSamplingName.argmax):
            ids = model_log_prob.argmax(dim=1)
            prev = index_to_log_onehot(ids, self.vocab_size)
        else:
            prev = self.log_sample_categorical(model_log_prob, generator=generator)
        return LayoutDiffusionSchedulerOutput(
            prev_sample=prev,
            pred_original_sample=log_x_recon,
            model_log_prob=model_log_prob,
        )

    def log_sample_categorical(
        self,
        logits: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample log one-hot categorical tokens with Gumbel-max."""
        sample = (gumbel_noise_like(logits, generator=generator) + logits).argmax(dim=1)
        return index_to_log_onehot(sample, self.vocab_size)

    def _init_buffers(self) -> None:
        if self.noise_schedule != "gaussian_refine_pow2.5":
            raise NotImplementedError(
                "Only gaussian_refine_pow2.5 LayoutDiffusion schedule is supported"
            )
        at, at1, bt1, bt2, ct, ct1, att, att1, btt1, btt2, ctt, ctt1 = _alpha_schedule(
            self.num_timesteps, type_classes=self.type_classes
        )
        self.log_at1 = (
            torch.log(torch.tensor(at1, dtype=torch.float64)).clamp(-70, 0).float()
        )
        self.log_ct1 = (
            torch.log(torch.tensor(ct1, dtype=torch.float64)).clamp(-70, 0).float()
        )
        self.log_cumprod_at1 = (
            torch.log(torch.tensor(att1, dtype=torch.float64)).clamp(-70, 0).float()
        )
        self.log_cumprod_ct1 = (
            torch.log(torch.tensor(ctt1, dtype=torch.float64)).clamp(-70, 0).float()
        )
        self.log_1_min_ct1 = _log_1_min_a(self.log_ct1.double()).float()
        self.log_1_min_cumprod_ct1 = _log_1_min_a(self.log_cumprod_ct1.double()).float()

        at_t = torch.tensor(at, dtype=torch.float64)
        bt1_t = torch.tensor(bt1, dtype=torch.float64)
        bt2_t = torch.tensor(bt2, dtype=torch.float64)
        ct_t = torch.tensor(ct, dtype=torch.float64)
        self.log_at = torch.log(at_t).float()
        self.log_bt1 = torch.log(bt1_t).float()
        self.log_bt2 = torch.log(bt2_t).float()
        self.log_ct = torch.log(ct_t).clamp(-70, 0).float()
        self.log_cumprod_at = torch.log(torch.tensor(att, dtype=torch.float64)).float()
        self.log_cumprod_bt1 = torch.log(
            torch.tensor(btt1, dtype=torch.float64)
        ).float()
        self.log_cumprod_bt2 = torch.log(
            torch.tensor(btt2, dtype=torch.float64)
        ).float()
        self.log_cumprod_ct = (
            torch.log(torch.tensor(ctt, dtype=torch.float64)).clamp(-70, 0).float()
        )
        self.log_1_min_ct = _log_1_min_a(self.log_ct.double()).float()
        self.log_1_min_cumprod_ct = _log_1_min_a(self.log_cumprod_ct.double()).float()

        bt2_safe = np.where(bt2 == 0.0, bt2.max(), bt2)
        q_one_step = [
            _gaussian_matrix2(
                t, bt=torch.tensor(bt2_safe).pow(2).pow(self.pow_num / 2) * self.mul_num
            )
            for t in range(self.num_timesteps)
        ]
        q_one_step.append(
            np.ones((self.num_coordinate_bins, self.num_coordinate_bins))
            / (self.num_coordinate_bins**2)
        )
        self.q_onestep_mats = torch.from_numpy(np.stack(q_one_step, axis=0)).float()
        q_mat = self.q_onestep_mats[0].numpy()
        q_mats = [q_mat]
        for t in range(1, self.num_timesteps):
            q_mat = np.tensordot(q_mat, self.q_onestep_mats[t].numpy(), axes=([1], [0]))
            q_mats.append(q_mat)
        q_mats.append(
            np.ones((self.num_coordinate_bins, self.num_coordinate_bins))
            / (self.num_coordinate_bins**2)
        )
        self.q_mats = torch.from_numpy(np.stack(q_mats, axis=0)).float()

    def _transition_matrix(
        self, t: torch.Tensor, *, cumulative: bool, device: torch.device
    ) -> torch.Tensor:
        batch_size = t.shape[0]
        if cumulative:
            log_at = _extract(self.log_cumprod_at.to(device), t, (batch_size, 1, 1))
            log_bt1 = _extract(self.log_cumprod_bt1.to(device), t, (batch_size, 1, 1))
            log_bt2 = _extract(self.log_cumprod_bt2.to(device), t, (batch_size, 1, 1))
            log_ct = _extract(self.log_cumprod_ct.to(device), t, (batch_size, 1, 1))
            log_at1 = _extract(self.log_cumprod_at1.to(device), t, (batch_size, 1, 1))
            log_ct1 = _extract(self.log_cumprod_ct1.to(device), t, (batch_size, 1, 1))
            q_coord = self.q_mats[t].to(device)
        else:
            log_at = _extract(self.log_at.to(device), t, (batch_size, 1, 1))
            log_bt1 = _extract(self.log_bt1.to(device), t, (batch_size, 1, 1))
            log_bt2 = _extract(self.log_bt2.to(device), t, (batch_size, 1, 1))
            log_ct = _extract(self.log_ct.to(device), t, (batch_size, 1, 1))
            log_at1 = _extract(self.log_at1.to(device), t, (batch_size, 1, 1))
            log_ct1 = _extract(self.log_ct1.to(device), t, (batch_size, 1, 1))
            q_coord = self.q_onestep_mats[t].to(device)
        log_1_min_ct = _log_1_min_a(log_ct)
        log_1_min_ct1 = _log_1_min_a(log_ct1)
        eye_special = torch.eye(self.num_special_tokens, device=device).expand(
            batch_size, -1, -1
        )
        zeros_special_rest = torch.zeros(
            batch_size,
            self.num_special_tokens,
            self.vocab_size - self.num_special_tokens,
            device=device,
        )
        type_eye = (
            torch.eye(self.type_classes, device=device)
            .clamp(min=1e-30)
            .log()
            .expand(batch_size, -1, -1)
        )
        coord_eye = (
            torch.eye(self.num_coordinate_bins, device=device)
            .clamp(min=1e-30)
            .log()
            .expand(batch_size, -1, -1)
        )
        matrix_absorb = torch.cat(
            [
                torch.cat([eye_special, zeros_special_rest], dim=-1),
                torch.cat(
                    [
                        torch.zeros(
                            batch_size,
                            self.type_classes,
                            self.num_special_tokens,
                            device=device,
                        ),
                        log_add_exp(type_eye + log_at1, log_bt1).exp(),
                        torch.zeros(
                            batch_size,
                            self.type_classes,
                            self.vocab_size
                            - self.num_special_tokens
                            - self.type_classes,
                            device=device,
                        ),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        torch.zeros(
                            batch_size,
                            self.num_coordinate_bins,
                            self.num_special_tokens + self.type_classes,
                            device=device,
                        ),
                        log_add_exp(coord_eye + log_at, log_bt2).exp(),
                        torch.zeros(
                            batch_size, self.num_coordinate_bins, 1, device=device
                        ),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        torch.zeros(
                            batch_size, 1, self.num_special_tokens, device=device
                        ),
                        log_add_exp(
                            torch.zeros(batch_size, 1, self.type_classes, device=device)
                            .clamp(min=1e-30)
                            .log()
                            + log_1_min_ct1,
                            log_ct1,
                        ).exp(),
                        log_add_exp(
                            torch.zeros(
                                batch_size, 1, self.num_coordinate_bins, device=device
                            )
                            .clamp(min=1e-30)
                            .log()
                            + log_1_min_ct,
                            log_ct,
                        ).exp(),
                        torch.ones(batch_size, 1, 1, device=device),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        matrix_gaussian = matrix_absorb.clone()
        coord_start = self.num_special_tokens + self.type_classes
        matrix_gaussian[
            :,
            coord_start : coord_start + self.num_coordinate_bins,
            coord_start : coord_start + self.num_coordinate_bins,
        ] = q_coord
        early = (t < (self.num_timesteps * 4 // 5)).reshape(batch_size, 1, 1)
        return torch.where(early, matrix_gaussian, matrix_absorb)


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    batch, *_ = timesteps.shape
    out = values.to(timesteps.device).gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def _log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - a.exp() + 1e-40)


def _gaussian_matrix2(t: int, *, bt: torch.Tensor) -> np.ndarray:
    num_pixel_vals = 128
    transition_bands = num_pixel_vals - 1
    beta_t = bt.numpy()[t]
    mat = np.zeros((num_pixel_vals, num_pixel_vals), dtype=np.float64)
    values = np.linspace(0.0, 127.0, num_pixel_vals, endpoint=True, dtype=np.float64)
    values = values * 2.0 / (num_pixel_vals - 1.0)
    values = values[: transition_bands + 1]
    values = -values * values / beta_t
    values = np.concatenate([values[:0:-1], values], axis=0)
    values = np.exp(values) / np.sum(np.exp(values), axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
        off_diag = np.full((num_pixel_vals - k,), values[k], dtype=np.float64)
        mat += np.diag(off_diag, k=k)
        mat += np.diag(off_diag, k=-k)
    mat += np.diag(1.0 - mat.sum(1), k=0)
    return mat


def _alpha_schedule(time_step: int, *, type_classes: int) -> tuple[np.ndarray, ...]:
    sep = 5
    sep_1 = sep - 1
    first = time_step * sep_1 // sep
    att = np.concatenate(
        (
            np.arange(0, first) / (first - 1) * (0.0001 - 0.99999) + 0.99999,
            np.arange(0, time_step - first)
            / (time_step - first - 1)
            * (0.000009 - 0.00009)
            + 0.00009,
        )
    )
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    att1 = np.concatenate(
        (
            np.arange(0, first) / (first - 1) * (0.9999 - 0.99999) + 0.99999,
            np.arange(0, time_step - first)
            / (time_step - first - 1)
            * (0.000009 - 0.9999)
            + 0.9999,
        )
    )
    att1 = np.concatenate(([1], att1))
    at1 = att1[1:] / att1[:-1]
    ctt = np.concatenate(
        (
            np.arange(0, first) / (first - 1) * (0.00009 - 0.000009) + 0.000009,
            np.arange(0, time_step - first)
            / (time_step - first - 1)
            * (0.9999 - 0.0001)
            + 0.0001,
        )
    )
    ctt = np.concatenate(([0], ctt))
    ct = 1 - (1 - ctt[1:]) / (1 - ctt[:-1])
    ctt1 = np.concatenate(
        (
            np.arange(0, first) / (first - 1) * (0.00009 - 0.000009) + 0.000009,
            np.arange(0, time_step - first)
            / (time_step - first - 1)
            * (0.9998 - 0.00009)
            + 0.00009,
        )
    )
    ctt1 = np.concatenate(([0], ctt1))
    ct1 = 1 - (1 - ctt1[1:]) / (1 - ctt1[:-1])
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    att1 = np.concatenate((att1[1:], [1]))
    ctt1 = np.concatenate((ctt1[1:], [0]))
    btt1 = (1 - att1 - ctt1) / type_classes
    btt2 = 1 - att - ctt
    bt1 = (1 - at1 - ct1) / type_classes
    btt2_for_step = np.concatenate(([0], btt2))
    bt2 = 1 - (1 - btt2_for_step[1:]) / (1 - btt2_for_step[:-1])
    btt2 = (1 - att - ctt) / 128
    bt2 = np.concatenate((bt2[:first], at1[first:] / 128))
    at = np.concatenate((at[:first], (1 - ct - bt2 * 128)[first:])).clip(min=1e-30)
    ct = np.concatenate(((1 - at - bt2)[:first], ct[first:])).clip(min=1e-30)
    return at, at1, bt1, bt2, ct, ct1, att, att1, btt1, btt2, ctt, ctt1
