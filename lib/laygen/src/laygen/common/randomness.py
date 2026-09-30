"""Side-effect-free random-generator helpers shared by layout packages."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from jaxtyping import Shaped

if TYPE_CHECKING:
    import torch


def _draw(
    draw: Callable[[torch.device, torch.Generator | None], Shaped[torch.Tensor, "..."]],
    *,
    generator: torch.Generator | None,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Draw on the generator device and move the result to the target device."""
    import torch

    draw_device = generator.device if generator is not None else torch.device(device)
    result = draw(draw_device, generator)
    if dtype is None:
        return result.to(device=device)

    return result.to(device=device, dtype=dtype)


def randn(
    *size: int | tuple[int, ...] | torch.Size,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample normally distributed values.

    Args:
        size: Output shape, as dimensions or one shape tuple.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        A tensor sampled with the requested shape, device, and dtype.
    """
    import torch

    if len(size) == 1 and isinstance(size[0], tuple):
        shape = size[0]
    elif all(isinstance(dimension, int) for dimension in size):
        shape = cast(tuple[int, ...], tuple(size))
    else:
        raise TypeError("size must be integers or one shape tuple")

    return _draw(
        lambda draw_device, draw_generator: torch.randn(
            shape,
            generator=draw_generator,
            device=draw_device,
            dtype=dtype,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def rand(
    *size: int | tuple[int, ...] | torch.Size,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample uniformly distributed values.

    Args:
        size: Output shape, as dimensions or one shape tuple.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        A tensor sampled with the requested shape, device, and dtype.
    """
    import torch

    if len(size) == 1 and isinstance(size[0], tuple):
        shape = size[0]
    elif all(isinstance(dimension, int) for dimension in size):
        shape = cast(tuple[int, ...], tuple(size))
    else:
        raise TypeError("size must be integers or one shape tuple")

    return _draw(
        lambda draw_device, draw_generator: torch.rand(
            shape,
            generator=draw_generator,
            device=draw_device,
            dtype=dtype,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def randint(
    low: int,
    high: int,
    size: tuple[int, ...] | torch.Size,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample integer values uniformly from ``[low, high)``.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        size: Output shape.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        A tensor sampled with the requested shape, device, and dtype.
    """
    import torch

    return _draw(
        lambda draw_device, draw_generator: torch.randint(
            low,
            high,
            size,
            generator=draw_generator,
            device=draw_device,
            dtype=dtype,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def randperm(
    n: int,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample a random permutation of ``range(n)``.

    Args:
        n: Number of values in the permutation.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        A permutation tensor on the requested device.
    """
    import torch

    return _draw(
        lambda draw_device, draw_generator: torch.randperm(
            n,
            generator=draw_generator,
            device=draw_device,
            dtype=dtype,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def multinomial(
    probs: Shaped[torch.Tensor, "..."],
    num_samples: int,
    replacement: bool = False,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample indices from a categorical probability tensor.

    Args:
        probs: Non-negative categorical probabilities.
        num_samples: Number of indices per row.
        replacement: Whether to sample with replacement.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        Sampled indices on the requested device.
    """
    import torch

    return _draw(
        lambda draw_device, draw_generator: torch.multinomial(
            probs.to(draw_device),
            num_samples,
            replacement=replacement,
            generator=draw_generator,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def bernoulli(
    probs: Shaped[torch.Tensor, "..."],
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample binary values from probability values.

    Args:
        probs: Probability tensor with values in ``[0, 1]``.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        Sampled binary values on the requested device.
    """
    import torch

    return _draw(
        lambda draw_device, draw_generator: torch.bernoulli(
            probs.to(draw_device), generator=draw_generator
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def normal(
    mean: int | float | Shaped[torch.Tensor, "..."],
    std: int | float | Shaped[torch.Tensor, "..."],
    *,
    size: tuple[int, ...] | torch.Size | None = None,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample normally distributed values from mean and standard deviation.

    Args:
        mean: Mean tensor or scalar.
        std: Standard-deviation tensor or scalar.
        size: Output shape when both parameters are scalars.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        Normally distributed values on the requested device.
    """
    import torch

    def draw(
        draw_device: torch.device, draw_generator: torch.Generator | None
    ) -> Shaped[torch.Tensor, "..."]:
        draw_mean = mean.to(draw_device) if isinstance(mean, torch.Tensor) else mean
        draw_std = std.to(draw_device) if isinstance(std, torch.Tensor) else std
        if size is None:
            if isinstance(draw_mean, torch.Tensor):
                return torch.normal(draw_mean, draw_std, generator=draw_generator)

            if isinstance(draw_std, torch.Tensor):
                return torch.normal(draw_mean, draw_std, generator=draw_generator)

            raise ValueError("size is required when mean and std are scalars")

        if isinstance(draw_mean, torch.Tensor) or isinstance(draw_std, torch.Tensor):
            raise ValueError("size requires scalar mean and std")

        return torch.normal(
            cast(int | float, draw_mean),
            cast(int | float, draw_std),
            size=size,
            generator=draw_generator,
            device=draw_device,
        )

    return _draw(draw, generator=generator, device=device, dtype=dtype)


def poisson(
    rates: Shaped[torch.Tensor, "..."],
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> Shaped[torch.Tensor, "..."]:
    """Sample Poisson-distributed values from rate values.

    Args:
        rates: Non-negative Poisson rate tensor.
        generator: Optional random generator.
        device: Target device.
        dtype: Optional output dtype.

    Returns:
        Sampled values on the requested device.
    """
    import torch

    return _draw(
        lambda draw_device, draw_generator: torch.poisson(
            rates.to(draw_device), generator=draw_generator
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def resolve_torch_generator(
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> torch.Generator | None:
    """Resolve an explicit or locally seeded Torch generator.

    Args:
        generator: Explicit generator, which takes precedence over ``seed``.
        seed: Seed for a new local generator when ``generator`` is absent.

    Returns:
        The explicit generator, a new CPU-local generator, or ``None``.
    """
    if generator is not None:
        return generator

    if seed is None:
        return None

    return _seeded_generator(seed=seed, device=device)


def _seeded_generator(*, seed: int, device: str | torch.device) -> torch.Generator:
    import torch

    return torch.Generator().manual_seed(seed)
