"""Joint continuous/discrete scheduler for DLT pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from einops import rearrange


@dataclass
class DLTJointSchedulerOutput(BaseOutput):
    """Output returned by one joint reverse step."""

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


DiscreteFeatureSpec = tuple[str, int]
_DEFAULT_FEATURES: Final[tuple[DiscreteFeatureSpec, ...]] = (("cat", 7),)


class DLTJointDiffusionScheduler(SchedulerMixin, ConfigMixin):
    """Save/loadable DLT continuous and discrete diffusion scheduler.

    Args:
        alpha: Probability of changing to a non-mask category.
        beta: Probability of changing to the mask/drop category.
        seq_max_length: Maximum number of layout elements.
        discrete_features_names: Discrete feature specs as ``(name, count)``.
        num_discrete_steps: Number of discrete diffusion steps per feature.
        temperature: Categorical sampling temperature.
        num_train_timesteps: Continuous DDPM timesteps.
        beta_schedule: Diffusers DDPM beta schedule.
        prediction_type: DDPM prediction type.
        clip_sample: Whether DDPM steps clamp predicted samples.
    """

    config_name = "scheduler_config.json"
    order = 1

    @register_to_config
    def __init__(
        self,
        *,
        alpha: float = 0.0,
        beta: float = 0.15,
        seq_max_length: int = 9,
        discrete_features_names: Sequence[Sequence[object]] | None = None,
        num_discrete_steps: Sequence[int] | None = None,
        temperature: float = 0.8,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "sample",
        clip_sample: bool = False,
    ) -> None:
        """Initialize the scheduler and defer transition matrix construction."""
        features = discrete_features_names or _DEFAULT_FEATURES
        steps = list(num_discrete_steps or [10 for _ in features])
        if len(features) != len(steps):
            raise ValueError("Each discrete feature requires a step count")
        self.alpha = alpha
        self.beta = beta
        self.seq_max_length = seq_max_length
        parsed_features: list[DiscreteFeatureSpec] = []
        for raw_feature in features:
            name = str(raw_feature[0])
            raw_count = raw_feature[1]
            if isinstance(raw_count, int):
                count = raw_count
            else:
                count = int(str(raw_count))
            parsed_features.append((name, count))
        self.discrete_features_names = parsed_features
        self.num_discrete_steps = [int(step) for step in steps]
        self.temperature = temperature
        self._cont2disc: dict[str, dict[int, int]] | None = None
        self._transition_matrices: dict[str, list[torch.Tensor]] | None = None
        self._ddpm = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
        )
        self.num_cont_steps = num_train_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

    @property
    def cont2disc(self) -> dict[str, dict[int, int]]:
        """Return continuous-to-discrete timestep mappings, computing lazily."""
        if self._cont2disc is None:
            self._cont2disc = {
                name: self.mapping_cont2disc(self.num_train_timesteps, steps)
                for (name, _), steps in zip(
                    self.discrete_features_names, self.num_discrete_steps, strict=True
                )
            }
        return self._cont2disc

    @property
    def transition_matrices(self) -> dict[str, list[torch.Tensor]]:
        """Return discrete transition matrices, computing lazily."""
        if self._transition_matrices is None:
            self._transition_matrices = {
                name: self.generate_transition_mat(count, steps)
                for (name, count), steps in zip(
                    self.discrete_features_names, self.num_discrete_steps, strict=True
                )
            }
        return self._transition_matrices

    def add_noise_jointly(
        self,
        vec_cont: torch.Tensor,
        vec_cat: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add continuous DDPM noise and discrete categorical noise."""
        noised_cont = self._ddpm.add_noise(
            original_samples=vec_cont,
            timesteps=cast(torch.IntTensor, timesteps),
            noise=noise,
        )
        cat_res: dict[str, torch.Tensor] = {}
        for f_name, _ in self.discrete_features_names:
            t_to_discrete_stage = [
                self.cont2disc[f_name][int(t.item())] for t in timesteps
            ]
            prob_mat = [
                self.transition_matrices[f_name][u].to(vec_cont.device)[
                    vec_cat[f_name][i]
                ]
                for i, u in enumerate(t_to_discrete_stage)
            ]
            probs = torch.cat(prob_mat)
            cat_noise = torch.multinomial(
                probs, 1, replacement=True, generator=generator
            )
            cat_res[f_name] = rearrange(
                cat_noise, "(d b) 1 -> d b", d=noised_cont.shape[0]
            )
        return noised_cont, cat_res

    def step_jointly(
        self,
        cont_output: torch.Tensor,
        cat_output: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> tuple[DLTJointSchedulerOutput, dict[str, torch.Tensor]]:
        """Take one reverse step for boxes and categories."""
        bbox = cast(
            DDPMSchedulerOutput,
            self._ddpm.step(
                cont_output,
                int(timestep.flatten()[0].item()),
                sample,
                generator=generator,
                return_dict=True,
            ),
        )
        bbox_out = DLTJointSchedulerOutput(
            prev_sample=bbox.prev_sample,
            pred_original_sample=cast(torch.Tensor, bbox.pred_original_sample),
        )
        step_cat_res: dict[str, torch.Tensor] = {}
        batch_timestep = (
            timestep
            if timestep.numel() == sample.shape[0]
            else timestep.flatten()[0].repeat(sample.shape[0])
        )
        for f_name, f_cat_num in self.discrete_features_names:
            t_to_discrete_stage = [
                self.cont2disc[f_name][int(t.item())] for t in batch_timestep
            ]
            cls, _ = self.denoise_cat(
                cat_output[f_name],
                t_to_discrete_stage,
                f_cat_num,
                self.transition_matrices[f_name],
                generator=generator,
            )
            step_cat_res[f_name] = cls
        return bbox_out, step_cat_res

    def generate_transition_mat(
        self, categories_num: int, num_discrete_steps: int
    ) -> list[torch.Tensor]:
        """Generate Markov transition matrices for one discrete feature."""
        transition_mat = (
            np.eye(categories_num) * (1 - self.alpha - self.beta)
            + self.alpha / categories_num
        )
        transition_mat[:, -1] += self.beta
        transition_mat[-1, :] = 0
        transition_mat[-1, -1] = 1
        transition_mat_list: list[torch.Tensor] = []
        curr_mat = transition_mat.copy()
        for _ in range(num_discrete_steps):
            transition_mat_list.append(torch.tensor(curr_mat, dtype=torch.float32))
            curr_mat = curr_mat @ transition_mat
        return transition_mat_list

    def denoise_cat(
        self,
        pred: torch.Tensor,
        t: list[int],
        cat_num: int,
        transition_mat_list: list[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Denoise a categorical feature using DLT's transition rule."""
        pred_prob = F.softmax(pred, dim=2)
        prob, cls = torch.max(pred_prob, dim=2)
        if t[0] > 1:
            matrix = transition_mat_list[t[0]].to(device=pred.device, dtype=pred.dtype)
            scores = torch.matmul(pred_prob.reshape((-1, cat_num)), matrix)
            scores = scores.reshape(pred_prob.shape)
            scores[:, :, 0] = 0
            logits = scores / self.temperature
            flat = logits.reshape(-1, cat_num)
            res = torch.multinomial(flat, 1, generator=generator).reshape(cls.shape)
        else:
            res = (cat_num - 1) * torch.ones_like(cls, dtype=torch.long)
            top = torch.topk(prob, prob.shape[1], dim=1)
            for row in range(prob.shape[0]):
                res[row, top.indices[row]] = cls[row, top.indices[row]]
        return res, 0

    @staticmethod
    def mapping_cont2disc(
        num_cont_steps: int, num_discrete_steps: int
    ) -> dict[int, int]:
        """Map continuous timesteps onto discrete diffusion stages."""
        block_size = num_cont_steps // num_discrete_steps
        cont2disc: dict[int, int] = {}
        for i in range(num_cont_steps):
            if i >= (num_discrete_steps - 1) * block_size:
                if (
                    num_cont_steps % num_discrete_steps != 0
                    and i >= num_discrete_steps * block_size
                ):
                    cont2disc[i] = num_discrete_steps - 1
                else:
                    cont2disc[i] = i // block_size
            else:
                cont2disc[i] = i // block_size
        return cont2disc
