from __future__ import annotations

import torch
import torch.nn.functional as F

LOG_EPS = -70.0


def index_to_log_onehot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if input_ids.numel() and input_ids.max().item() >= vocab_size:
        raise ValueError(
            f"input id {input_ids.max().item()} exceeds vocab_size {vocab_size}"
        )
    onehot = F.one_hot(input_ids.long(), vocab_size)
    order = (0, -1) + tuple(range(1, input_ids.ndim))
    return torch.log(onehot.permute(order).float().clamp(min=1e-30))


def log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    return log_x.argmax(dim=1)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    maximum = torch.maximum(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(
    values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    batch, *_ = timesteps.shape
    out = values.to(timesteps.device).gather(-1, timesteps)
    return out.reshape(batch, *((1,) * (len(broadcast_shape) - 1)))


def gumbel_noise_like(
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    uniform = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return -torch.log(-torch.log(uniform + 1e-30) + 1e-30)


def log_sample_categorical(
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return (logits + gumbel_noise_like(logits, generator=generator)).argmax(dim=1)


def top_k_logits(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    if k <= 0 or k >= logits.size(dim):
        return logits
    values = torch.topk(logits, k, dim=dim).values
    threshold = values.select(dim, k - 1).unsqueeze(dim)
    return logits.masked_fill(logits < threshold, LOG_EPS)


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, LOG_EPS)
    return torch.empty_like(logits).scatter(
        dim=-1, index=sorted_indices, src=sorted_logits
    )


def sample_categorical(
    logits: torch.Tensor,
    *,
    sampling: str = "random",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if sampling == "deterministic":
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    if sampling in {"top_k", "top_k_top_p"} and top_k is not None:
        scaled = top_k_logits(scaled, top_k, dim=-1)
    if sampling in {"top_p", "top_k_top_p"} and top_p is not None:
        scaled = _top_p_logits(scaled, top_p)
    if sampling == "gumbel":
        return (scaled + gumbel_noise_like(scaled, generator=generator)).argmax(dim=-1)
    probs = scaled.softmax(dim=-1).reshape(-1, scaled.size(-1))
    sampled = torch.multinomial(probs, 1, generator=generator).reshape(
        scaled.shape[:-1]
    )
    return sampled


def batch_topk_mask(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    max_k = int(k.max().item()) if k.numel() else 0
    if max_k == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    _, indices = torch.topk(scores, max_k, dim=1)
    ranks = torch.arange(max_k, device=scores.device).unsqueeze(0)
    active = ranks < k.to(scores.device).unsqueeze(1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, active)
