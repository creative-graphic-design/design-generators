"""Package-local copies of the effective LACE constraint losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Bool, Float, Int


def xywh_to_ltrb_reference(
    bbox: Float[torch.Tensor, "batch elements 4"],
) -> Float[torch.Tensor, "batch elements 4"]:
    """Convert center boxes with the reference allocation and assignments."""
    bbox_ltrb = torch.zeros(bbox.shape).to(bbox.device)
    bbox_xy = torch.abs(bbox[:, :, :2])
    bbox_wh = torch.abs(bbox[:, :, 2:])
    bbox_ltrb[:, :, :2] = bbox_xy - 0.5 * bbox_wh
    bbox_ltrb[:, :, 2:] = bbox_xy + 0.5 * bbox_wh
    return bbox_ltrb


def _piou_ltrb(
    bbox_ltrb: Float[torch.Tensor, "batch elements 4"],
    mask: Float[torch.Tensor, "batch elements"] | None = None,
) -> Float[torch.Tensor, "batch elements elements"]:
    """Compute the reference pairwise IoU operation in its source order."""
    n_box = bbox_ltrb.shape[1]
    device = bbox_ltrb.device

    area_bbox = (bbox_ltrb[:, :, 2] - bbox_ltrb[:, :, 0]) * (
        bbox_ltrb[:, :, 3] - bbox_ltrb[:, :, 1]
    )
    area_bbox_psum = area_bbox.unsqueeze(-1) + area_bbox.unsqueeze(-2)

    x1y1 = bbox_ltrb[:, :, [0, 1]]
    x1y1 = torch.swapaxes(x1y1, 1, 2)
    x1y1_i = torch.max(x1y1.unsqueeze(-1), x1y1.unsqueeze(-2))

    x2y2 = bbox_ltrb[:, :, [2, 3]]
    x2y2 = torch.swapaxes(x2y2, 1, 2)
    x2y2_i = torch.min(x2y2.unsqueeze(-1), x2y2.unsqueeze(-2))

    wh_i = F.relu(x2y2_i - x1y1_i)
    area_i = wh_i[:, 0, :, :] * wh_i[:, 1, :, :]
    piou = area_i / (area_bbox_psum - area_i + 1e-10)
    piou.masked_fill_(torch.eye(n_box, n_box).to(torch.bool).to(device), 0)

    if mask is not None:
        mask = mask.unsqueeze(2)
        select_mask = torch.matmul(mask, torch.transpose(mask, dim0=1, dim1=2))
        piou = piou * select_mask.to(device)

    return piou


def pairwise_iou(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Float[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements elements"]:
    """Return the reference pairwise intersection-over-union matrix."""
    return _piou_ltrb(xywh_to_ltrb_reference(bbox), mask)


def alignment_matrix(
    bbox: Float[torch.Tensor, "batch elements 4"],
    mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 6 elements"]:
    """Return the reference raw coordinate-alignment matrix."""
    bbox_t = bbox.permute(2, 0, 1)
    xc, yc, width, height = bbox_t
    xl = xc - width / 2
    yt = yc - height / 2
    xr = xc + width / 2
    yb = yc + height / 2
    x = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    x = x.unsqueeze(-1) - x.unsqueeze(-2)
    idx = torch.arange(x.size(2), device=x.device)
    x[:, :, idx, idx] = 1.0
    x = x.abs().permute(0, 2, 1, 3)
    x[~mask] = 1.0
    return x


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
