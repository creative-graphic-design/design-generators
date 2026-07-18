import math

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from laygen.schedulers.continuous import (
    BetaSchedule,
    DDIMDiscretization,
    get_beta_schedule,
    get_ddim_timesteps,
    get_layousyn_beta_schedule,
    make_beta_schedule,
    make_ddim_timesteps,
    normalize_beta_schedule,
    normalize_ddim_discretization,
)


def _vendor_beta_schedule(
    schedule: BetaSchedule,
    *,
    num_timesteps: int,
    start: float = 0.0001,
    end: float = 0.02,
) -> torch.Tensor:
    if schedule is BetaSchedule.linear:
        return torch.linspace(start, end, num_timesteps)
    if schedule is BetaSchedule.const:
        return end * torch.ones(num_timesteps)
    if schedule is BetaSchedule.quad:
        return torch.linspace(start**0.5, end**0.5, num_timesteps) ** 2
    if schedule is BetaSchedule.jsd:
        return 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    if schedule is BetaSchedule.sigmoid:
        betas = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(betas) * (end - start) + start
    if schedule in (BetaSchedule.cosine, BetaSchedule.cosine_reverse):
        max_beta = 0.999
        cosine_s = 0.008
        return torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
    if schedule is BetaSchedule.cosine_anneal:
        return torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    raise AssertionError(f"unhandled schedule: {schedule}")


def _vendor_ddim_timesteps(
    method: DDIMDiscretization,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
) -> np.ndarray:
    if method is DDIMDiscretization.uniform:
        c = num_ddpm_timesteps // num_ddim_timesteps
        timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif method is DDIMDiscretization.quad:
        timesteps = (
            np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2
        ).astype(int)
    elif method is DDIMDiscretization.new:
        c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
        timesteps = np.asarray(
            list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c))
        )
    else:
        raise AssertionError(f"unhandled method: {method}")
    return timesteps + 1


def test_common_beta_schedules_match_vendor_and_diffusers_aliases() -> None:
    cases = [
        (BetaSchedule.linear, "linear"),
        (BetaSchedule.quad, "scaled_linear"),
        (BetaSchedule.sigmoid, "sigmoid"),
        (BetaSchedule.cosine, "squaredcos_cap_v2"),
    ]
    for schedule, diffusers_name in cases:
        actual = get_beta_schedule(schedule, num_timesteps=8)
        expected_vendor = _vendor_beta_schedule(schedule, num_timesteps=8)
        expected_diffusers = DDPMScheduler(
            num_train_timesteps=8,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule=diffusers_name,
        ).betas
        assert torch.equal(actual, expected_vendor)
        assert torch.equal(actual, expected_diffusers)


def test_legacy_lace_beta_schedules_remain_vendor_exact() -> None:
    for schedule in [
        BetaSchedule.const,
        BetaSchedule.jsd,
        BetaSchedule.cosine_reverse,
        BetaSchedule.cosine_anneal,
    ]:
        assert torch.equal(
            make_beta_schedule(schedule, num_timesteps=8),
            _vendor_beta_schedule(schedule, num_timesteps=8),
        )


def test_ddim_timesteps_match_vendor_and_diffusers_uniform_adapter() -> None:
    uniform = get_ddim_timesteps("uniform", 10, 100)
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        timestep_spacing="leading",
        steps_offset=1,
    )
    scheduler.set_timesteps(10)
    assert np.array_equal(uniform, scheduler.timesteps.cpu().numpy()[::-1])
    assert np.array_equal(
        uniform,
        np.asarray([1, 11, 21, 31, 41, 51, 61, 71, 81, 91]),
    )

    for method in [DDIMDiscretization.quad, DDIMDiscretization.new]:
        assert np.array_equal(
            make_ddim_timesteps(method, 60, 100),
            _vendor_ddim_timesteps(method, 60, 100),
        )


def test_scheduler_normalizers_reject_unknown_values() -> None:
    assert normalize_beta_schedule("linear") is BetaSchedule.linear
    assert normalize_ddim_discretization("new") is DDIMDiscretization.new
    try:
        normalize_beta_schedule("bad")
    except ValueError as exc:
        assert "Unsupported beta schedule" in str(exc)
    else:
        raise AssertionError("unsupported beta schedule should fail")
    try:
        normalize_ddim_discretization("bad")
    except ValueError as exc:
        assert "Unsupported ddim discretization" in str(exc)
    else:
        raise AssertionError("unsupported ddim discretization should fail")


def test_layousyn_scaled_cosine_matches_vendor_formula() -> None:
    def alpha_bar(t: float, alpha_scale: float) -> float:
        out = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        return (out * alpha_scale**2) / (out * (alpha_scale**2 - 1) + 1)

    expected = torch.tensor(
        [
            min(1 - alpha_bar((i + 1) / 8, 2.0) / alpha_bar(i / 8, 2.0), 0.999)
            for i in range(8)
        ],
        dtype=torch.float64,
    )
    assert torch.equal(
        get_layousyn_beta_schedule("squaredcos_cap_v2", 8, alpha_scale=2.0),
        expected,
    )


def test_layousyn_linear_alpha_scale_matches_vendor_formula() -> None:
    betas = np.linspace(0.0001 * 125, 0.02 * 125, 8, dtype=np.float64)
    alpha_cumprod = np.cumprod(1 - betas)
    updated = (4.0 * alpha_cumprod) / (3.0 * alpha_cumprod + 1.0)
    expected = torch.from_numpy(
        np.array(
            [1 - updated[0]] + [1 - updated[i] / updated[i - 1] for i in range(1, 8)]
        )
    )
    assert torch.equal(
        get_layousyn_beta_schedule("linear", 8, alpha_scale=2.0), expected
    )
