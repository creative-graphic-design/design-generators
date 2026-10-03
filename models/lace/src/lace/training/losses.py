"""Package-local copies of the effective LACE constraint losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Bool, Float, Int


def xywh_to_ltrb_reference(
    bbox: Float[torch.Tensor, "batch elements 4"],
) -> Float[torch.Tensor, "batch elements 4"]:
    """Convert center boxes with the reference absolute-width behavior."""
    xy = torch.abs(bbox[:, :, :2])
    wh = torch.abs(bbox[:, :, 2:])
    return torch.cat((xy - 0.5 * wh, xy + 0.5 * wh), dim=-1)


def pairwise_iou(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Float[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements elements"]:
    """Return the pairwise intersection-over-union matrix."""
    ltrb = xywh_to_ltrb_reference(bbox)
    n_box = ltrb.shape[1]

    area = (ltrb[:, :, 2] - ltrb[:, :, 0]) * (ltrb[:, :, 3] - ltrb[:, :, 1])
    area_sum = area.unsqueeze(-1) + area.unsqueeze(-2)
    lt = ltrb[:, :, [0, 1]].swapaxes(1, 2)
    rb = ltrb[:, :, [2, 3]].swapaxes(1, 2)
    inter_lt = torch.max(lt.unsqueeze(-1), lt.unsqueeze(-2))
    inter_rb = torch.min(rb.unsqueeze(-1), rb.unsqueeze(-2))
    inter_wh = F.relu(inter_rb - inter_lt)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    iou = inter_area / (area_sum - inter_area + 1e-10)
    diagonal = torch.eye(n_box, dtype=torch.bool, device=bbox.device)
    iou.masked_fill_(diagonal, 0)
    select_mask = torch.matmul(mask.unsqueeze(2), mask.unsqueeze(1))

    return iou * select_mask


def alignment_matrix(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 6 elements"]:
    """Return the raw coordinate-alignment matrix used by LACE."""
    ltrb = xywh_to_ltrb_reference(bbox).permute(2, 0, 1)
    xl, yt, xr, yb = ltrb
    bbox_t = bbox.permute(2, 0, 1)
    xc, yc = bbox_t[:2]
    coords = torch.stack((xl, xc, xr, yt, yc, yb), dim=1)
    coords = coords.unsqueeze(-1) - coords.unsqueeze(-2)
    idx = torch.arange(coords.size(2), device=coords.device)
    coords[:, :, idx, idx] = 1.0
    coords = coords.abs().permute(0, 2, 1, 3)
    coords[~mask] = 1.0
    return coords


def constraint_temporal_weight(
    timestep: Int[torch.Tensor, "batch"],
    *,
    num_timesteps: int = 1000,
    end: float = 1e-2,
) -> Float[torch.Tensor, "batch"]:
    """Return the reference constant temporal constraint weight."""
    weight = 1 - 4 * end * torch.ones(num_timesteps)
    return weight.cumprod(dim=0).to(timestep.device)[timestep]


def rand_fix(
    batch_size: int,
    mask: Bool[torch.Tensor, "batch elements"],
    *,
    ratio: float = 0.2,
) -> Bool[torch.Tensor, "batch elements"]:
    """Sample the reference random completion mask."""
    n_elements = mask.shape[1]
    indices = (torch.rand(batch_size, n_elements) <= torch.rand(1).item() * ratio).to(
        mask.device
    )
    return indices * mask.bool()


def lace_losses(
    *,
    layout_input: Float[torch.Tensor, "batch elements channels"],
    model_output: Float[torch.Tensor, "four_batch elements channels"],
    noise: Float[torch.Tensor, "four_batch elements channels"],
    reconstructed: Float[torch.Tensor, "four_batch elements channels"],
    bbox: Float[torch.Tensor, "four_batch elements 4"],
    mask: Bool[torch.Tensor, "four_batch elements"],
    timesteps: Int[torch.Tensor, "four_batch"],
    num_classes: int,
) -> dict[str, Float[torch.Tensor, ""]]:
    """Compute the reference-ordered diffusion, constraint, and reconstruction losses."""
    bbox_rep = reconstructed[:, :, num_classes:].clamp(-1, 1) / 2 + 0.5
    _, align_loss = layout_alignment(bbox_rep, mask)
    align_loss = 20 * align_loss
    piou = pairwise_iou(bbox_rep, mask.float())
    pdist = torch.cdist(bbox_rep[:, :, :2], bbox_rep[:, :, :2], p=2)
    overlap_loss = piou.mean(dim=(1, 2)) + (piou.ne(0) * torch.exp(-pdist)).mean(
        dim=(1, 2)
    )
    reconstruct_loss = F.mse_loss(
        torch.cat((layout_input,) * 4, dim=0)[:, :, num_classes:],
        reconstructed[:, :, num_classes:],
    )
    weight = constraint_temporal_weight(timesteps)
    constraint_loss = torch.mean((align_loss + overlap_loss) * weight)
    diffusion_loss = F.mse_loss(noise, model_output)
    total = diffusion_loss + constraint_loss + reconstruct_loss
    return {
        "diffusion_loss": diffusion_loss,
        "alignment_loss": align_loss.mean(),
        "overlap_loss": overlap_loss.mean(),
        "constraint_loss": constraint_loss,
        "reconstruct_loss": reconstruct_loss,
        "train_loss": total,
    }


def layout_alignment(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """Return the reference local alignment score and normalized score."""
    coords = alignment_matrix(bbox, mask)
    values = coords.min(-1).values.min(-1).values
    values.masked_fill_(values.eq(1.0), 0.0)
    values = -torch.log(1 - values)
    score = reduce(values, "b s -> b", "sum")
    normalized = score / reduce(mask, "b s -> b", "sum")
    normalized[torch.isnan(normalized)] = 0.0
    return score, normalized
