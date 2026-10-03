"""Package-local DS-GAN discriminator and layout ordering."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from jaxtyping import Float
from torch import nn

from ..configuration_ds_gan import DSGANConfig
from ..modeling_ds_gan import CNNLSTM, ResnetBackbone

NO_OBJECT: Final = 0


class _LayoutArgmax(torch.autograd.Function):
    """Apply the discriminator's hard layout ordering with identity backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        layout: Float[torch.Tensor, "batch elements 2 4"],
    ) -> Float[torch.Tensor, "batch elements 2 4"]:
        del ctx
        output = layout
        class_ids = output[:, :, 0].argmax(dim=-1)
        output[:, :, 0].zero_()
        output[:, :, 0].scatter_(-1, class_ids.unsqueeze(-1), 1.0)
        boxes = output[:, :, 1]
        boxes[class_ids == NO_OBJECT] = 0.0
        for batch_index in range(output.shape[0]):
            order = _layout_order(output[batch_index, :, 0], output[batch_index, :, 1])
            output[batch_index, :, 1] = output[batch_index, order, 1]

        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Float[torch.Tensor, "batch elements 2 4"],
    ) -> tuple[Float[torch.Tensor, "batch elements 2 4"]]:  # ty: ignore[invalid-method-override]
        del ctx
        return (grad_output,)


def _layout_order(
    class_one_hot: Float[torch.Tensor, "elements classes"],
    boxes: Float[torch.Tensor, "elements 4"],
) -> list[int]:
    """Return the layout order used by the discriminator's box channel."""
    xyxy = _cxcywh_to_xyxy(boxes)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    left_top = torch.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    right_bottom = torch.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    intersection_width_height = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_width_height[..., 0] * intersection_width_height[..., 1]
    union = area[:, None] + area[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.asarray(intersection.detach().cpu()) / np.asarray(union.detach().cpu())

    ids = np.asarray(class_one_hot.detach().cpu())
    areas = np.asarray(area.detach().cpu())
    logos = np.where(ids == 2)[0]
    order_text = sorted(
        [(int(index), float(areas[index])) for index in np.where(ids == 1)[0]],
        key=lambda value: value[1],
        reverse=True,
    )
    order_decorations = sorted(
        [(int(index), float(areas[index])) for index in np.where(ids == 3)[0]],
        key=lambda value: value[1],
    )

    connections: dict[int, int | list[int]] = {}
    reverse: dict[int, list[int]] = {}
    for decoration, _ in order_decorations:
        connected: list[int] = []
        for index in logos:
            if iou[decoration, index]:
                connections[int(index)] = decoration
                connected.append(int(index))

        for index, _ in order_text:
            if iou[decoration, index]:
                connections[index] = decoration
                connected.append(index)

        for index, _ in order_decorations:
            if decoration == index:
                continue

            if bool(iou[decoration, index]):
                if index not in connections:
                    connections[index] = [decoration]
                else:
                    connection = connections[index]
                    if not isinstance(connection, list):
                        raise RuntimeError("layout decoration connection is not a list")

                    connection.append(decoration)

                connected.append(index)

        reverse[decoration] = connected

    order: list[int] = []
    for index in logos:
        index = int(index)
        if index not in connections:
            order.append(index)
            continue

        decoration = connections[index]
        decoration_index = _first_connection(decoration)
        for connected_index in reverse[decoration_index]:
            if connected_index not in order:
                order.append(connected_index)

        if decoration_index not in order:
            order.append(decoration_index)

    for index, _ in order_text:
        if len(order) >= len(ids):
            break

        if index not in connections:
            order.append(index)
            continue

        decoration = connections[index]
        decoration_index = _first_connection(decoration)
        for connected_index in reverse[decoration_index]:
            if connected_index not in order:
                order.append(connected_index)

        if decoration_index not in order:
            order.append(decoration_index)

    if len(order) < len(ids):
        order.extend(int(index) for index in np.where(ids == NO_OBJECT)[0])

    return order[: len(ids)]


def _first_connection(connection: int | list[int]) -> int:
    if isinstance(connection, int):
        return connection

    return connection[0]


def _cxcywh_to_xyxy(
    boxes: Float[torch.Tensor, "elements 4"],
) -> Float[torch.Tensor, "elements 4"]:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


class DSGANDiscriminator(nn.Module):
    """Package-local discriminator matching the original DS-GAN topology."""

    def __init__(self, config: DSGANConfig) -> None:
        """Initialize the discriminator from a ResNet-FPN and CNN-LSTM config."""
        super().__init__()
        self.resnet_fpn = ResnetBackbone(config)
        self.cnnlstm = CNNLSTM(config)
        self.fc_tf = nn.Linear(2 * config.hidden_size, 1)

    def forward(
        self,
        pixel_values: Float[torch.Tensor, "batch 4 height width"],
        layout: Float[torch.Tensor, "batch elements 2 4"],
    ) -> Float[torch.Tensor, "batch 1"]:
        """Score a layout after hard class selection and ordering."""
        h0 = self.resnet_fpn(pixel_values)
        ordered = _LayoutArgmax.apply(layout)
        output = self.cnnlstm(ordered, h0)[:, -1, :]
        return torch.tanh(self.fc_tf(output))


__all__ = ["DSGANDiscriminator"]
