from __future__ import annotations

import torch


def sample_initial_state(
    *,
    batch_size: int,
    max_length: int,
    lengths: torch.LongTensor,
    dim: int,
    distribution: str = "gaussian",
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else lengths.device
    if distribution == "gaussian":
        sample = torch.randn(
            batch_size,
            max_length,
            dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    elif distribution == "uniform":
        sample = (
            2
            * torch.rand(
                batch_size,
                max_length,
                dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            - 1
        )
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    mask = torch.arange(max_length, device=device)[None, :] < lengths[:, None].to(
        device
    )
    return sample * mask.unsqueeze(-1)
