from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xywh_to_ltrb_split(bbox: torch.Tensor) -> list[torch.Tensor]:
    xc, yc, w, h = bbox
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def _xywh_to_ltrb(bbox_xywh: torch.Tensor) -> torch.Tensor:
    bbox_ltrb = torch.zeros_like(bbox_xywh)
    bbox_xy = torch.abs(bbox_xywh[:, :, :2])
    bbox_wh = torch.abs(bbox_xywh[:, :, 2:])
    bbox_ltrb[:, :, :2] = bbox_xy - 0.5 * bbox_wh
    bbox_ltrb[:, :, 2:] = bbox_xy + 0.5 * bbox_wh
    return bbox_ltrb


def _pairwise_iou_xywh(bbox_xywh: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bbox_ltrb = _xywh_to_ltrb(bbox_xywh)
    n_box = bbox_ltrb.shape[1]
    area = (bbox_ltrb[:, :, 2] - bbox_ltrb[:, :, 0]) * (
        bbox_ltrb[:, :, 3] - bbox_ltrb[:, :, 1]
    )
    area_sum = area.unsqueeze(-1) + area.unsqueeze(-2)
    left_top = bbox_ltrb[:, :, [0, 1]].swapaxes(1, 2)
    right_bottom = bbox_ltrb[:, :, [2, 3]].swapaxes(1, 2)
    inter_lt = torch.max(left_top.unsqueeze(-1), left_top.unsqueeze(-2))
    inter_rb = torch.min(right_bottom.unsqueeze(-1), right_bottom.unsqueeze(-2))
    inter_wh = F.relu(inter_rb - inter_lt)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    iou = inter_area / (area_sum - inter_area + 1e-10)
    iou.masked_fill_(torch.eye(n_box, dtype=torch.bool, device=bbox_xywh.device), 0)
    select_mask = torch.matmul(mask.float().unsqueeze(2), mask.float().unsqueeze(1))
    return iou * select_mask


def _layout_alignment_matrix(bbox: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bbox_t = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = _xywh_to_ltrb_split(bbox_t)
    xc, yc = bbox_t[0], bbox_t[1]
    coords = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    coords = coords.unsqueeze(-1) - coords.unsqueeze(-2)
    idx = torch.arange(coords.size(2), device=coords.device)
    coords[:, :, idx, idx] = 1.0
    coords = coords.abs().permute(0, 2, 1, 3)
    coords[~mask] = 1.0
    return coords


def beautify_layout(
    bbox: torch.Tensor,
    mask: torch.BoolTensor,
    overlap_weight: float = 1.0,
    alignment_weight: float = 1.0,
    xy_only: bool = False,
    num_steps: int = 1000,
    lr: float = 1e-4,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    if torch.sum(mask) == 1:
        return bbox, mask
    bbox_in = bbox
    if xy_only:
        wh = torch.abs(bbox[:, :, 2:].clone().detach())
        bbox_in = torch.cat([bbox[:, :, :2], wh], dim=2)
    bbox_in = bbox_in.clone()
    bbox_in[:, :, [0, 2]] *= 10 / 4
    bbox_in[:, :, [1, 3]] *= 10 / 6
    bbox_initial = bbox_in.clone().detach()
    bbox_param = nn.Parameter(bbox_in)
    optimizer = torch.optim.Adam([bbox_param], lr=lr)
    mse_loss = nn.MSELoss()
    mask_out = mask.clone()
    with torch.enable_grad():
        for _ in range(num_steps):
            bbox_relu = torch.relu(bbox_param)
            align_score = _layout_alignment_matrix(bbox_relu, mask_out)
            align_mask = (align_score < 1 / 64).clone().detach()
            align_loss = torch.mean(align_score * align_mask)
            piou = torch.mean(_pairwise_iou_xywh(bbox_relu, mask_out))
            mse = mse_loss(bbox_relu, bbox_initial)
            loss = mse + alignment_weight * align_loss + overlap_weight * piou
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([bbox_param], 1.0)
            optimizer.step()
            min_wh = torch.min(bbox_relu[:, :, [2, 3]], dim=2).values
            mask_out = mask_out * (min_wh > 0.01)
    bbox_out = torch.relu(bbox_param.detach())
    bbox_out[:, :, [0, 2]] *= 4 / 10
    bbox_out[:, :, [1, 3]] *= 6 / 10
    return bbox_out, mask_out
