"""Continuous diffusion scheduler adapters backed by Diffusers.

The beta/DDIM helper names follow CompVis latent-diffusion
``ldm/modules/diffusionmodules/util.py`` utilities as carried by LACE vendor
``diffusion_utils.py``. Common schedules are delegated to Diffusers schedulers.
"""

from __future__ import annotations

import math
from enum import StrEnum, auto
from typing import Literal, assert_never

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class BetaSchedule(StrEnum):
    """Supported DDPM beta schedules.

    Origin:
        These schedule names mirror CompVis latent-diffusion
        ``make_beta_schedule`` aliases used by the LACE vendor scheduler.
    """

    linear = auto()
    const = auto()
    quad = auto()
    jsd = auto()
    sigmoid = auto()
    cosine = auto()
    cosine_reverse = auto()
    cosine_anneal = auto()


class DDIMDiscretization(StrEnum):
    """Supported DDIM timestep discretization methods.

    Origin:
        These discretization names mirror CompVis latent-diffusion
        ``make_ddim_timesteps`` modes used by the LACE vendor scheduler.
    """

    uniform = auto()
    quad = auto()
    new = auto()


def normalize_beta_schedule(schedule: BetaSchedule | str) -> BetaSchedule:
    """Normalize a beta schedule value.

    Origin:
        This preserves the CompVis latent-diffusion schedule aliases exposed by
        the LACE vendor configuration.

    Args:
        schedule: Schedule enum or string value.

    Returns:
        Canonical beta schedule enum.

    Raises:
        ValueError: If the schedule is unsupported.
    """
    if isinstance(schedule, BetaSchedule):
        return schedule
    try:
        return BetaSchedule(schedule)
    except ValueError as exc:
        raise ValueError(f"Unsupported beta schedule: {schedule}") from exc


def normalize_ddim_discretization(
    method: DDIMDiscretization | str,
) -> DDIMDiscretization:
    """Normalize a DDIM timestep discretization method.

    Origin:
        This preserves the CompVis latent-diffusion DDIM discretization aliases
        exposed by the LACE vendor configuration.

    Args:
        method: Method enum or string value.

    Returns:
        Canonical DDIM discretization enum.

    Raises:
        ValueError: If the method is unsupported.
    """
    if isinstance(method, DDIMDiscretization):
        return method
    try:
        return DDIMDiscretization(method)
    except ValueError as exc:
        raise ValueError(f"Unsupported ddim discretization: {method}") from exc


def _diffusers_beta_schedule(
    schedule: BetaSchedule,
) -> Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"] | None:
    if schedule is BetaSchedule.linear:
        return "linear"
    if schedule is BetaSchedule.quad:
        return "scaled_linear"
    if schedule in (BetaSchedule.cosine, BetaSchedule.cosine_reverse):
        return "squaredcos_cap_v2"
    if schedule is BetaSchedule.sigmoid:
        return "sigmoid"
    return None


def get_beta_schedule(
    schedule: BetaSchedule | str = BetaSchedule.cosine,
    num_timesteps: int = 1000,
    start: float = 0.0001,
    end: float = 0.02,
) -> torch.Tensor:
    """Create a beta schedule, delegating common schedules to Diffusers.

    Origin:
        The public API follows CompVis latent-diffusion ``make_beta_schedule``.
        Common DDPM schedules delegate to Diffusers ``DDPMScheduler`` while
        legacy LACE-only aliases remain custom for compatibility.

    Args:
        schedule: Schedule enum or string value.
        num_timesteps: Number of training timesteps.
        start: Initial beta value for schedules that use a range.
        end: Final beta value for schedules that use a range.

    Returns:
        One-dimensional beta tensor.

    Raises:
        ValueError: If the schedule is unsupported.
    """
    canonical = normalize_beta_schedule(schedule)
    diffusers_schedule = _diffusers_beta_schedule(canonical)
    if diffusers_schedule is not None:
        return DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=start,
            beta_end=end,
            beta_schedule=diffusers_schedule,
        ).betas
    if canonical is BetaSchedule.const:
        return end * torch.ones(num_timesteps)
    if canonical is BetaSchedule.jsd:
        return 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    if canonical is BetaSchedule.cosine_anneal:
        return torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    raise ValueError(f"Unsupported beta schedule: {schedule}")


def get_layousyn_beta_schedule(
    schedule: Literal["linear", "squaredcos_cap_v2"] = "linear",
    num_timesteps: int = 100,
    *,
    alpha_scale: float = 1.0,
) -> torch.Tensor:
    """Create LayouSyn/OpenAI-style beta schedules.

    Origin:
        This preserves `vendor/lay-your-scene/layousyn/diffusion/
        gaussian_diffusion.py` exactly, including the LayYourScene-specific
        ``alpha_scale`` transform for both the linear and squared-cosine
        schedules.

    Args:
        schedule: Vendor schedule name.
        num_timesteps: Number of diffusion timesteps.
        alpha_scale: Vendor alpha-bar scaling factor.

    Returns:
        One-dimensional beta tensor in float64 precision.

    Raises:
        ValueError: If the schedule is unsupported.
    """
    if schedule == "linear":
        scale = 1000 / num_timesteps
        betas = torch.linspace(
            scale * 0.0001,
            scale * 0.02,
            num_timesteps,
            dtype=torch.float64,
        )
        if alpha_scale == 1.0:
            return betas
        alpha_cumprod = torch.cumprod(1 - betas, dim=0)
        alpha_scaled = (alpha_scale**2 * alpha_cumprod) / (
            (alpha_scale**2 - 1) * alpha_cumprod + 1.0
        )
        first = 1 - alpha_scaled[:1]
        rest = 1 - alpha_scaled[1:] / alpha_scaled[:-1]
        return torch.cat((first, rest), dim=0)
    if schedule == "squaredcos_cap_v2":
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            alpha_1 = _layousyn_scaled_cosine_alpha_bar(t1, alpha_scale)
            alpha_2 = _layousyn_scaled_cosine_alpha_bar(t2, alpha_scale)
            betas.append(min(1 - alpha_2 / alpha_1, 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    raise ValueError(f"Unsupported LayouSyn beta schedule: {schedule}")


def _layousyn_scaled_cosine_alpha_bar(timestep: float, alpha_scale: float) -> float:
    alpha = math.cos((timestep + 0.008) / 1.008 * math.pi / 2) ** 2
    return (alpha * alpha_scale**2) / (alpha * (alpha_scale**2 - 1) + 1)


def get_ddim_timesteps(
    method: DDIMDiscretization | str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    *,
    steps_offset: int = 1,
) -> np.ndarray:
    """Create ascending vendor-order DDIM timesteps.

    Origin:
        The public API follows CompVis latent-diffusion ``make_ddim_timesteps``.
        The uniform branch adapts Diffusers ``DDIMScheduler`` back to LACE's
        ascending one-indexed vendor order.

    Args:
        method: Discretization enum or string value.
        num_ddim_timesteps: Number of inference timesteps.
        num_ddpm_timesteps: Number of training timesteps.
        steps_offset: Diffusers timestep offset. LACE uses one-indexed
            timesteps, so the default is ``1``.

    Returns:
        NumPy array of timesteps in ascending vendor order.

    Raises:
        ValueError: If the method is unsupported.
    """
    canonical = normalize_ddim_discretization(method)
    if canonical is DDIMDiscretization.uniform:
        scheduler = DDIMScheduler(
            num_train_timesteps=num_ddpm_timesteps,
            timestep_spacing="leading",
            steps_offset=steps_offset,
        )
        scheduler.set_timesteps(num_ddim_timesteps)
        return scheduler.timesteps.cpu().numpy()[::-1].copy()
    if canonical is DDIMDiscretization.quad:
        timesteps = (
            np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2
        ).astype(int)
        return timesteps + steps_offset
    if canonical is DDIMDiscretization.new:
        c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
        timesteps = np.asarray(
            list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c))
        )
        return timesteps + steps_offset
    assert_never(canonical)


make_beta_schedule = get_beta_schedule
make_ddim_timesteps = get_ddim_timesteps
