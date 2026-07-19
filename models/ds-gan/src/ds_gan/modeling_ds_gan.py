"""PyTorch/Transformers implementation of the PosterLayout DS-GAN generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_ds_gan import DSGANConfig

if TYPE_CHECKING:
    import torch


@dataclass
class DSGANModelOutput(ModelOutput):
    """Raw DS-GAN generator output.

    Attributes:
        class_probs: Vendor class probabilities with shape
            ``(batch, elements, 4)`` where id 0 is ``no object``.
        bbox: Normalized center ``xywh`` boxes with shape
            ``(batch, elements, 4)``.
        initial_layout: Initial class/box layout passed to the generator.
    """

    class_probs: Float[torch.Tensor, "batch elements 4"]
    bbox: Float[torch.Tensor, "batch elements 4"] = cast(
        Float[torch.Tensor, "batch elements 4"], None
    )
    initial_layout: Float[torch.Tensor, "batch elements 2 4"] | None = None


class ResnetBackbone(nn.Module):
    """Vendor ResNet-FPN encoder used to initialize the DS-GAN LSTM state."""

    def __init__(self, config: DSGANConfig) -> None:
        """Initialize the ResNet-FPN encoder."""
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "DSGANModel requires the optional timm dependency"
            ) from exc

        if config.backbone == "resnet50":
            channels = (1024, 2048)
        elif config.backbone == "resnet18":
            channels = (256, 512)
        else:
            raise ValueError(f"Unsupported DS-GAN backbone: {config.backbone}")
        resnet = timm.create_model(config.backbone, pretrained=False)
        resnet.conv1 = nn.Conv2d(
            4,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        children = list(resnet.children())
        self.resnet_tilconv4 = nn.Sequential(*children[:7])
        self.resnet_conv5 = children[7]
        self.fpn_conv11_4 = nn.Conv2d(channels[0], 256, 1, 1, 0)
        self.fpn_conv11_5 = nn.Conv2d(channels[1], 256, 1, 1, 0)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.proj = nn.Conv2d(512, 8 * config.max_elem, 1, 1, 0)
        self.fc_h0 = nn.Linear(config.backbone_feature_size, config.num_layers * 2)

    def forward(
        self, pixel_values: Float[torch.Tensor, "batch 4 height width"]
    ) -> Float[torch.Tensor, "layers2 batch hidden"]:
        """Encode image/saliency tensors into an LSTM initial hidden state."""
        resnet_f4 = self.resnet_tilconv4(pixel_values)
        resnet_f5 = self.resnet_conv5(resnet_f4)
        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        resnet_f5up = F.interpolate(
            resnet_f5p, size=resnet_f4p.shape[2:], mode="nearest"
        )
        fused = torch.concat(
            [resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1
        )
        projected = self.proj(fused)
        flattened = projected.flatten(start_dim=-2)
        return self.fc_h0(flattened).permute(2, 0, 1)


class CNNLSTM(nn.Module):
    """Vendor CNN-LSTM sequence model."""

    def __init__(self, config: DSGANConfig) -> None:
        """Initialize the CNN-LSTM block."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=1, padding=1),
        )
        self.lstm = nn.LSTM(
            input_size=config.out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        layout: Float[torch.Tensor, "batch elements 2 4"],
        h0: Float[torch.Tensor, "layers2 batch hidden"],
    ) -> Float[torch.Tensor, "batch elements hidden2"]:
        """Run the vendor CNN-LSTM over initial layout tensors."""
        self.lstm.flatten_parameters()
        x = layout.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        x = self.conv(x).permute(0, 2, 1).contiguous()
        output, _ = self.lstm(x, (torch.zeros_like(h0).contiguous(), h0.contiguous()))
        return output


class DSGANModel(PreTrainedModel):
    """Transformers-compatible DS-GAN generator.

    Args:
        config: DS-GAN model configuration.

    Examples:
        >>> config = DSGANConfig(backbone="resnet18", max_elem=4, hidden_size=32, num_layers=2, image_size=(64, 64), backbone_feature_size=16)
        >>> model = DSGANModel(config)
        >>> model.config.model_type
        'ds_gan'
    """

    config_class = DSGANConfig
    base_model_prefix = "ds_gan"
    supports_gradient_checkpointing = False

    def __init__(self, config: DSGANConfig) -> None:
        """Initialize DS-GAN generator layers."""
        super().__init__(config)
        self.resnet_fpn = ResnetBackbone(config)
        self.cnnlstm = CNNLSTM(config)
        self.fc1 = nn.Linear(2 * config.hidden_size, config.output_size // 2)
        self.fc2 = nn.Linear(2 * config.hidden_size, config.output_size // 2)
        self.post_init()

    def forward(
        self,
        pixel_values: Float[torch.Tensor, "batch 4 height width"],
        layout: Float[torch.Tensor, "batch elements 2 4"],
        return_dict: bool = True,
    ) -> DSGANModelOutput | tuple[torch.Tensor, torch.Tensor]:
        """Run a DS-GAN generator forward pass.

        Args:
            pixel_values: RGB plus saliency tensor shaped ``(B, 4, H, W)``.
            layout: Initial vendor layout shaped ``(B, max_elem, 2, 4)``.
            return_dict: Whether to return a dataclass output.

        Returns:
            Raw class probabilities and normalized center ``xywh`` boxes.

        Raises:
            ValueError: If tensor shapes do not match the config.
        """
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 4:
            raise ValueError("pixel_values must have shape (batch, 4, height, width)")
        expected_layout = (pixel_values.shape[0], self.config.max_elem, 2, 4)
        if tuple(layout.shape) != expected_layout:
            raise ValueError(f"layout must have shape {expected_layout}")
        pixel_values = pixel_values.to(dtype=self.dtype)
        layout = layout.to(device=pixel_values.device, dtype=self.dtype)
        h0 = self.resnet_fpn(pixel_values)
        lstm_output = self.cnnlstm(layout, h0)
        class_probs = torch.softmax(self.fc1(lstm_output), dim=-1)
        bbox = torch.sigmoid(self.fc2(lstm_output))
        if not return_dict:
            return class_probs, bbox
        return DSGANModelOutput(
            class_probs=class_probs,
            bbox=bbox,
            initial_layout=layout,
        )


def xyxy_to_xywh(
    bbox: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    """Convert left/top/right/bottom boxes to center ``xywh`` boxes."""
    left, top, right, bottom = bbox.unbind(-1)
    return torch.stack(
        ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top),
        dim=-1,
    )


def random_initial_layout(
    batch_size: int,
    max_elem: int,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    weighted_classes: bool = True,
    use_numpy_classes: bool = False,
) -> Float[torch.Tensor, "batch elements 2 4"]:
    """Sample the vendor DS-GAN initial layout tensor.

    Args:
        batch_size: Batch size.
        max_elem: Number of layout slots.
        generator: Optional torch generator. Takes precedence over ``seed``.
        seed: Convenience seed used only when ``generator`` is absent.
        device: Target torch device.
        dtype: Target floating dtype.
        weighted_classes: Whether to use the vendor inference class prior.
        use_numpy_classes: Use NumPy class sampling for exact vendor script parity.

    Returns:
        Tensor shaped ``(batch, max_elem, 2, 4)``.
    """
    resolved_device = (
        torch.device(device) if device is not None else torch.device("cpu")
    )
    if generator is None and seed is not None:
        generator = torch.Generator(device=resolved_device).manual_seed(seed)
    if weighted_classes:
        probs = torch.tensor((0.1, 0.8, 1.0, 1.0), device=resolved_device)
        probs = probs / probs.sum()
    else:
        probs = torch.full((4,), 0.25, device=resolved_device)
    if use_numpy_classes:
        rng = np.random.default_rng(seed)
        np_probs = probs.detach().cpu().numpy()
        class_ids = torch.as_tensor(
            rng.choice(4, size=(batch_size, max_elem, 1), p=np_probs),
            dtype=torch.long,
            device=resolved_device,
        )
    else:
        class_ids = torch.multinomial(
            probs,
            num_samples=batch_size * max_elem,
            replacement=True,
            generator=generator,
        ).reshape(batch_size, max_elem, 1)
    class_one_hot = torch.zeros(
        batch_size,
        max_elem,
        4,
        dtype=dtype,
        device=resolved_device,
    )
    class_one_hot.scatter_(-1, class_ids, 1)
    box_xyxy = torch.normal(
        mean=0.5,
        std=0.15,
        size=(batch_size, max_elem, 1, 4),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
    )
    bbox = xyxy_to_xywh(box_xyxy)
    return torch.concat([class_one_hot.unsqueeze(2), bbox], dim=2)
