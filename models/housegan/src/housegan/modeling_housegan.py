"""PyTorch House-GAN generator in Transformers ``PreTrainedModel`` form."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_housegan import HouseGanConfig


@dataclass
class HouseGanModelOutput(ModelOutput):
    """Raw House-GAN model output."""

    masks: Float[torch.Tensor, "elements height width"]
    node_features: Float[torch.Tensor, "elements room_labels"] | None = None
    edges: Int[torch.Tensor, "edges 3"] | None = None


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    *,
    act: str,
    upsample: bool = False,
) -> list[nn.Module]:
    layer: nn.Module
    if upsample:
        layer = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
    else:
        layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
    blocks: list[nn.Module] = [layer]
    if "leaky" in act:
        blocks.append(nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        blocks.append(nn.ReLU(True))
    elif "tanh" in act:
        blocks.append(nn.Tanh())
    return blocks


class CMP(nn.Module):
    """House-GAN convolutional message passing block."""

    def __init__(self, in_channels: int) -> None:
        """Initialize the CMP block."""
        super().__init__()
        self.encoder = nn.Sequential(
            *_conv_block(3 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *_conv_block(2 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *_conv_block(2 * in_channels, in_channels, 3, 1, 1, act="leaky"),
        )

    def forward(
        self,
        feats: Float[torch.Tensor, "elements channels height width"],
        edges: Int[torch.Tensor, "edges 3"],
    ) -> Float[torch.Tensor, "elements channels height width"]:
        """Pool positive and negative edge-neighbor features."""
        edges = edges.view(-1, 3)
        elements = feats.size(0)
        pooled_pos = torch.zeros_like(feats)
        pooled_neg = torch.zeros_like(feats)
        if edges.numel() > 0:
            pos_edges = edges[edges[:, 1] > 0]
            neg_edges = edges[edges[:, 1] < 0]
            pooled_pos = _pool_edges(feats, pos_edges, elements)
            pooled_neg = _pool_edges(feats, neg_edges, elements)
        return self.encoder(torch.cat([feats, pooled_pos, pooled_neg], dim=1))


def _pool_edges(
    feats: Float[torch.Tensor, "elements channels height width"],
    edges: Int[torch.Tensor, "edges 3"],
    elements: int,
) -> Float[torch.Tensor, "elements channels height width"]:
    pooled = torch.zeros_like(feats)
    if edges.numel() == 0:
        return pooled
    src = torch.cat([edges[:, 0], edges[:, 2]]).long()
    dst = torch.cat([edges[:, 2], edges[:, 0]]).long()
    src_feats = feats[src]
    dst_index = dst.view(-1, 1, 1, 1).expand_as(src_feats)
    return pooled.scatter_add(0, dst_index, src_feats).view(elements, *feats.shape[1:])


class HouseGanGenerator(PreTrainedModel):
    """Transformers-compatible House-GAN generator.

    Args:
        config: House-GAN configuration.

    Examples:
        >>> model = HouseGanGenerator(HouseGanConfig())
        >>> latents = torch.zeros(2, 128)
        >>> nodes = torch.eye(10)[:2]
        >>> edges = torch.tensor([[0, 1, 1]])
        >>> tuple(model(latents, nodes, edges).masks.shape)
        (2, 32, 32)
    """

    config_class = HouseGanConfig
    base_model_prefix = "housegan"
    main_input_name = "latents"
    supports_gradient_checkpointing = False

    def __init__(self, config: HouseGanConfig) -> None:
        """Initialize generator layers."""
        super().__init__(config)
        init_size = config.mask_size // 4
        in_features = config.latent_dim + config.node_feature_dim
        channels = config.cmp_channels
        self.init_size = init_size
        self.l1 = nn.Sequential(nn.Linear(in_features, channels * init_size**2))
        self.upsample_1 = nn.Sequential(
            *_conv_block(channels, channels, 4, 2, 1, act="leaky", upsample=True)
        )
        self.upsample_2 = nn.Sequential(
            *_conv_block(channels, channels, 4, 2, 1, act="leaky", upsample=True)
        )
        self.cmp_1 = CMP(channels)
        self.cmp_2 = CMP(channels)
        self.decoder = nn.Sequential(
            *_conv_block(channels, 256, 3, 1, 1, act="leaky"),
            *_conv_block(256, 128, 3, 1, 1, act="leaky"),
            *_conv_block(128, 1, 3, 1, 1, act="tanh"),
        )
        self.post_init()

    def forward(
        self,
        latents: Float[torch.Tensor, "elements latent"],
        node_features: Float[torch.Tensor, "elements room_labels"],
        edges: Int[torch.Tensor, "edges 3"],
        return_dict: bool | None = None,
    ) -> HouseGanModelOutput | tuple[torch.Tensor]:
        """Run a House-GAN forward pass.

        Args:
            latents: Per-room latent vectors.
            node_features: Per-room one-hot room features.
            edges: Signed complete graph triples.
            return_dict: Whether to return ``HouseGanModelOutput``.

        Returns:
            Raw generated room masks.
        """
        if latents.ndim != 2 or latents.shape[-1] != self.config.latent_dim:
            raise ValueError("latents must have shape (elements, latent_dim)")
        if node_features.shape != (latents.shape[0], self.config.node_feature_dim):
            raise ValueError(
                "node_features must have shape (elements, node_feature_dim)"
            )
        if edges.ndim != 2 or edges.shape[-1] != 3:
            raise ValueError("edges must have shape (edges, 3)")
        dtype = next(self.parameters()).dtype
        latents = latents.to(dtype=dtype, device=self.device)
        node_features = node_features.to(dtype=dtype, device=self.device)
        edges = edges.to(dtype=torch.long, device=self.device)
        hidden = torch.cat(
            [latents.view(-1, self.config.latent_dim), node_features], dim=1
        )
        hidden = self.l1(hidden)
        hidden = hidden.view(
            -1, self.config.cmp_channels, self.init_size, self.init_size
        )
        hidden = self.cmp_1(hidden, edges).view(-1, *hidden.shape[1:])
        hidden = self.upsample_1(hidden)
        hidden = self.cmp_2(hidden, edges).view(-1, *hidden.shape[1:])
        hidden = self.upsample_2(hidden)
        masks = self.decoder(hidden.view(-1, hidden.shape[1], *hidden.shape[2:]))
        masks = masks.view(-1, *masks.shape[2:])
        if return_dict is False:
            return (masks,)
        return HouseGanModelOutput(
            masks=masks, node_features=node_features, edges=edges
        )
