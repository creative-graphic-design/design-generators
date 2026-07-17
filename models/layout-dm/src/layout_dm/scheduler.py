from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

from laygen.common.discrete import (
    index_to_log_onehot,
    log_add_exp,
    log_onehot_to_index,
    sample_categorical,
)

from .sampling import LayoutDMSamplingConfig

if TYPE_CHECKING:
    from .conditioning import LayoutDMCondition


@dataclass
class LayoutDMSchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor | None = None
    model_log_prob: torch.Tensor | None = None


class LayoutDMScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        num_timesteps: int = 100,
        q_type: Literal["constrained", "vanilla"] = "constrained",
        vocab_size: int,
        mask_token_id: int,
        pad_token_id: int,
        var_order: tuple[str, ...] = ("c", "x", "y", "w", "h"),
        token_mask: list[list[bool]] | None = None,
        per_var_full_ids: dict[str, list[int]] | None = None,
        att_1: float = 0.99999,
        att_T: float = 0.000009,
        ctt_1: float = 0.000009,
        ctt_T: float = 0.99999,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.timesteps = torch.arange(num_timesteps - 1, -1, -1)
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.var_order = tuple(var_order)
        self.token_mask = (
            None if token_mask is None else torch.tensor(token_mask, dtype=torch.bool)
        )
        self.per_var_full_ids = per_var_full_ids
        self.att_1 = att_1
        self.att_T = att_T
        self.ctt_1 = ctt_1
        self.ctt_T = ctt_T
        if per_var_full_ids is None:
            self.schedules = {
                "full": _alpha_schedule(
                    num_timesteps, vocab_size - 1, att_1, att_T, ctt_1, ctt_T
                )
            }
        else:
            self.schedules = {
                key: _alpha_schedule(
                    num_timesteps, len(ids) - 1, att_1, att_T, ctt_1, ctt_T
                )
                for key, ids in per_var_full_ids.items()
            }

    def set_timesteps(
        self, num_inference_steps: int | None = None, device: torch.device | None = None
    ) -> None:
        steps = num_inference_steps or self.num_timesteps
        self.timesteps = torch.tensor(
            [int(i * self.num_timesteps / steps) for i in range(steps - 1, -1, -1)],
            dtype=torch.long,
            device=device,
        )

    def initial_sample(
        self,
        batch_size: int,
        token_length: int,
        *,
        device: torch.device,
        condition: LayoutDMCondition | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            ids = condition.input_ids.to(device)
        else:
            ids = torch.full(
                (batch_size, token_length),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
        return index_to_log_onehot(ids, self.vocab_size)

    def predict_start(self, denoiser_output: torch.Tensor) -> torch.Tensor:
        logits = denoiser_output[:, :, :-1]
        log_pred = torch.log_softmax(logits.double(), dim=-1).float()
        mask_col = torch.full(
            (*log_pred.shape[:2], 1),
            -70.0,
            device=log_pred.device,
            dtype=log_pred.dtype,
        )
        return (
            torch.cat((log_pred, mask_col), dim=-1).permute(0, 2, 1).clamp(-70.0, 0.0)
        )

    def q_posterior(
        self, log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.per_var_full_ids is not None:
            return self._constrained_q_posterior(log_x_start, log_x_t, t)
        t = t.clamp(0, self.num_timesteps - 1)
        keep = (
            self.schedules["full"][3].to(t.device).gather(0, t).exp().reshape(-1, 1, 1)
        )
        return torch.logaddexp(
            log_x_start + keep.log(), log_x_t + torch.log1p(-keep).clamp_min(-70.0)
        ).clamp(-70.0, 0.0)

    def _constrained_q_posterior(
        self,
        log_x_start_full: torch.Tensor,
        log_x_t_full: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = log_x_start_full.size(0)
        step = len(self.var_order)
        seq_len = log_x_start_full.shape[-1] // step
        index_x_t_full = log_onehot_to_index(log_x_t_full)
        mask_reshaped = (index_x_t_full == self.mask_token_id).reshape(
            batch_size, seq_len, step
        )
        log_one = torch.zeros(
            batch_size, 1, 1, device=log_x_t_full.device, dtype=log_x_t_full.dtype
        )
        log_zero = torch.log(log_one + 1.0e-30).expand(-1, -1, seq_len)
        full_outputs = []
        for i, key in enumerate(self.var_order):
            mask = mask_reshaped[..., i].unsqueeze(1)
            log_x_start = self._full_to_partial_log(log_x_start_full[..., i::step], key)
            log_x_t = self._full_to_partial_log(log_x_t_full[..., i::step], key)
            log_qt = self._q_pred(log_x_t, t, key)[:, :-1, :]
            log_cumprod_ct = _extract(
                self.schedules[key][5].to(t.device), t, log_x_t.shape
            )
            ct_cumprod = log_cumprod_ct.expand(-1, self._mat_size(key) - 1, -1)
            log_qt = (~mask) * log_qt + mask * ct_cumprod
            log_qt_one = self._q_pred_one_timestep(log_x_t, t, key)
            log_qt_one = torch.cat((log_qt_one[:, :-1, :], log_zero), dim=1)
            log_ct = _extract(self.schedules[key][2].to(t.device), t, log_x_t.shape)
            ct_vector = torch.cat(
                (log_ct.expand(-1, self._mat_size(key) - 1, -1), log_one), dim=1
            )
            log_qt_one = (~mask) * log_qt_one + mask * ct_vector
            q = torch.cat((log_x_start[:, :-1, :] - log_qt, log_zero), dim=1)
            q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
            q = q - q_log_sum_exp
            partial = self._q_pred(q, t - 1, key) + log_qt_one + q_log_sum_exp
            full_outputs.append(
                self._partial_to_full_log(partial.clamp(-70.0, 0.0), key)
            )
        return torch.stack(full_outputs, dim=-1).reshape(
            batch_size, self.vocab_size, -1
        )

    def _q_pred_one_timestep(
        self, log_x_t: torch.Tensor, t: torch.Tensor, key: str
    ) -> torch.Tensor:
        log_at, log_bt, log_ct = (self.schedules[key][i].to(t.device) for i in range(3))
        log_at = _extract(log_at, t, log_x_t.shape)
        log_bt = _extract(log_bt, t, log_x_t.shape)
        log_ct = _extract(log_ct, t, log_x_t.shape)
        log_1_min_ct = _log_1_min_a(log_ct)
        return torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

    def _q_pred(
        self, log_x_start: torch.Tensor, t: torch.Tensor, key: str
    ) -> torch.Tensor:
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at, log_cumprod_bt, log_cumprod_ct = (
            self.schedules[key][i].to(t.device) for i in range(3, 6)
        )
        log_cumprod_at = _extract(log_cumprod_at, t, log_x_start.shape)
        log_cumprod_bt = _extract(log_cumprod_bt, t, log_x_start.shape)
        log_cumprod_ct = _extract(log_cumprod_ct, t, log_x_start.shape)
        log_1_min_cumprod_ct = _log_1_min_a(log_cumprod_ct)
        return torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),
            ],
            dim=1,
        )

    def _mat_size(self, key: str) -> int:
        assert self.per_var_full_ids is not None
        return len(self.per_var_full_ids[key])

    def _full_ids(self, key: str, device: torch.device) -> torch.Tensor:
        assert self.per_var_full_ids is not None
        return torch.tensor(self.per_var_full_ids[key], dtype=torch.long, device=device)

    def _full_to_partial_log(self, inputs: torch.Tensor, key: str) -> torch.Tensor:
        full_ids = self._full_ids(key, inputs.device)
        index = full_ids.reshape(1, -1, 1).expand(inputs.shape[0], -1, inputs.shape[-1])
        return torch.gather(inputs, dim=1, index=index)

    def _partial_to_full_log(self, inputs: torch.Tensor, key: str) -> torch.Tensor:
        full_ids = self._full_ids(key, inputs.device)
        outputs = torch.full(
            (inputs.shape[0], self.vocab_size, inputs.shape[-1]),
            -70.0,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        index = full_ids.reshape(1, -1, 1).expand(inputs.shape[0], -1, inputs.shape[-1])
        return outputs.scatter(dim=1, index=index, src=inputs)

    def step(
        self,
        denoiser_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        previous_timestep: int,
        sampling: LayoutDMSamplingConfig,
        condition: LayoutDMCondition | None = None,
        generator: torch.Generator | None = None,
    ) -> LayoutDMSchedulerOutput:
        log_x_recon = self.predict_start(denoiser_output)
        model_log_prob = self.q_posterior(log_x_recon, sample, timestep)
        if self.token_mask is not None:
            valid = self.token_mask.to(model_log_prob.device).T.unsqueeze(0)
            model_log_prob = model_log_prob.masked_fill(~valid, -70.0)
        if condition is not None:
            strong_mask = condition.mask.to(model_log_prob.device).unsqueeze(1)
            strong_log_prob = index_to_log_onehot(
                condition.input_ids.to(model_log_prob.device), self.vocab_size
            )
            model_log_prob = torch.where(strong_mask, strong_log_prob, model_log_prob)
        logits = model_log_prob.permute(0, 2, 1)
        ids = sample_categorical(
            logits,
            sampling=sampling.name,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            generator=generator,
        )
        prev_sample = index_to_log_onehot(ids, self.vocab_size)
        return LayoutDMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=log_x_recon,
            model_log_prob=model_log_prob,
        )


def _alpha_schedule(
    num_timesteps: int, n: int, att_1: float, att_t: float, ctt_1: float, ctt_t: float
) -> tuple[torch.Tensor, ...]:
    att = np.arange(0, num_timesteps) / (num_timesteps - 1) * (att_t - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, num_timesteps) / (num_timesteps - 1) * (ctt_t - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    ct = 1 - one_minus_ctt[1:] / one_minus_ctt[:-1]
    bt = (1 - at - ct) / n
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / n
    return tuple(
        torch.tensor(x.astype("float64")).log().float()
        for x in (at, bt, ct, att, btt, ctt)
    )


def _extract(
    values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    batch, *_ = timesteps.shape
    out = values.gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def _log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - a.exp() + 1e-40)
