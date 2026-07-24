"""Transformer denoiser used by CGB-DM checkpoints."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float, Int
from laygen.nn import AdaLayerNorm, SinusoidalPosEmb
from torch import Tensor, nn


@dataclass
class CGBDMModelOutput(BaseOutput):
    """Output returned by the CGB-DM denoiser.

    Attributes:
        sample: Predicted epsilon tensor with the same shape as the input layout.
        cgb_weight: Content-graphic balance weight estimated from image tokens.
    """

    sample: Float[torch.Tensor, "batch elements channels"]
    cgb_weight: Float[torch.Tensor, "batch 1 1"] | None = None


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


class _ImageFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ImageAttention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads) for tensor in qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _ImageTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _ImageAttention(
                            dim, heads=heads, dim_head=dim_head, dropout=dropout
                        ),
                        _ImageFeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_pair in self.layers:
            attn, ff = cast(tuple[nn.Module, nn.Module], tuple(layer_pair.children()))
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class CGBDMImageEncoder(nn.Module):
    """Patch image encoder used for content-aware conditioning."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        dim_model: int,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        """Initialize patch embedding and image transformer blocks."""
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = _pair(patch_size)
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim_model),
            nn.LayerNorm(dim_model),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _ImageTransformer(
            dim_model, depth, heads, dim_head, mlp_dim, dropout
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensors into patch tokens."""
        x = self.to_patch_embedding(image)
        batch, tokens, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : tokens + 1]
        return self.transformer(self.dropout(x))


class _LayoutMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(left, right)
            for left, right in zip(
                [input_dim] + hidden, hidden + [output_dim], strict=True
            )
        )
        self.num_layers = num_layers

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < self.num_layers - 1:
                x = F.softplus(x)
        return x


class _LayoutBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        diffusion_steps: int = 1000,
        timestep_type: str | None = "adalayernorm",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = (
            AdaLayerNorm(d_model, diffusion_steps, timestep_type)
            if timestep_type is not None and "adalayernorm" in timestep_type
            else nn.LayerNorm(d_model, eps=1e-5)
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm4 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        img: Tensor | None,
        cgb_w: Tensor | None,
        salbox_encode: Tensor | None,
        *,
        timestep: Tensor,
    ) -> Tensor:
        x = src
        x = (
            self.norm1(x, timestep)
            if isinstance(self.norm1, AdaLayerNorm)
            else self.norm1(x)
        )
        x = x + self._sa_block(x)
        if img is not None:
            x = self.norm2(x)
            attended = self._ca_block(x, img, img)
            x = x + (attended if cgb_w is None else cgb_w * attended)
        if salbox_encode is not None:
            x = self.norm3(x)
            x = x + self._ca_block(x, salbox_encode, salbox_encode)
        x = x + self._ff_block(self.norm4(x))
        return x

    def _sa_block(self, x: Tensor) -> Tensor:
        return self.dropout1(self.self_attn(x, x, x, need_weights=False)[0])

    def _ca_block(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.dropout2(self.self_attn(x, k, v, need_weights=False)[0])

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


class CGBDMLayoutModule(nn.Module):
    """Timestep-conditioned layout encoder or decoder block stack."""

    def __init__(
        self,
        *,
        seq_dim: int,
        dim_model: int,
        n_head: int,
        feature_dim: int,
        num_layers: int,
        num_train_timesteps: int,
        max_seq_length: int,
        if_encoder: bool,
    ) -> None:
        """Initialize layout projections and timestep-aware blocks."""
        super().__init__()
        self.max_elem = max_seq_length
        self.if_encoder = if_encoder
        self.mlp = (
            _LayoutMLP(seq_dim, dim_model, dim_model, 3)
            if if_encoder
            else _LayoutMLP(dim_model, dim_model, seq_dim, 3)
        )
        self.pos_encoder = SinusoidalPosEmb(num_steps=max_seq_length, dim=dim_model)
        layer = _LayoutBlock(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=feature_dim,
            diffusion_steps=num_train_timesteps,
            timestep_type="adalayernorm",
        )
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        img_encode: Tensor | None,
        cgb_w: Tensor | None,
        salbox_encode: Tensor | None,
        timestep: Tensor,
    ) -> Tensor:
        """Run the layout encoder or decoder path."""
        if self.if_encoder:
            output = F.softplus(self.mlp(src))
            positions = torch.arange(self.max_elem, device=src.device)
            output = output + self.pos_encoder(positions)
        else:
            output = src
        for index, layer in enumerate(self.layers):
            output = layer(
                output,
                img_encode,
                cgb_w,
                salbox_encode,
                timestep=timestep,
            )
            if index < self.num_layers - 1:
                output = F.softplus(output)
        if not self.if_encoder:
            output = self.mlp(output)
        return output


class _TransformerLayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_type = input.dtype
        output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return output.to(original_type)


class _ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = _TransformerLayerNorm,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        attended = self.attn(
            self.ln_1(x),
            self.ln_1(x),
            self.ln_1(x),
            need_weights=False,
            attn_mask=attn_mask,
        )[0]
        x = x + attended
        return x + self.mlp(self.ln_2(x))


class _TokenTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.ModuleList(
            [_ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask)
        return x


class CGBDMQFormer(nn.Module):
    """Estimate a scalar content-graphic balance weight from image tokens."""

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 1,
        num_heads: int = 8,
        num_tokens: int = 1,
        n_layers: int = 2,
    ) -> None:
        """Initialize query-token transformer and scalar projection."""
        super().__init__()
        scale = in_dim**-0.5
        self.num_tokens = num_tokens
        self.scale_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = _TokenTransformer(
            width=in_dim, layers=n_layers, heads=num_heads
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        self.out = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Softplus(),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Pool image tokens into a content-graphic balance weight."""
        scale_emb = self.scale_emb.repeat(image_tokens.shape[0], 1, 1)
        x = torch.cat([scale_emb, image_tokens], dim=1)
        x = self.ln1(x).permute(1, 0, 2)
        x = self.transformer_blocks(x).permute(1, 0, 2)
        x = self.ln2(x[:, : self.num_tokens, :])
        return self.out(x)


class CGBDMMLP(_LayoutMLP):
    """Softplus MLP used for saliency-box embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize saliency-box embedding layers."""
        super().__init__(input_dim, hidden_dim, output_dim, num_layers=3)


class CGBDMTransformerModel(ModelMixin, ConfigMixin):
    """CGB-DM transformer denoiser with image and saliency conditioning.

    Args:
        num_labels: Internal class-channel count including invalid/pad.
        max_seq_length: Maximum number of layout elements.
        image_size: Image tensor size as ``(height, width)``.
        patch_size: Image patch size.
        dim_model: Hidden dimension.
        n_head: Attention head count.
        feature_dim: Feed-forward hidden dimension.
        num_layers: Number of decoder layers.
        num_train_timesteps: Number of training diffusion steps.

    Examples:
        >>> model = CGBDMTransformerModel(num_labels=4, max_seq_length=2, image_size=(32, 32), dim_model=16, n_head=2, feature_dim=32, num_layers=1)
        >>> model.seq_dim
        8
    """

    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        num_labels: int = 4,
        max_seq_length: int = 16,
        image_size: tuple[int, int] | list[int] = (384, 256),
        patch_size: int = 32,
        dim_model: int = 512,
        n_head: int = 8,
        feature_dim: int = 1024,
        num_layers: int = 4,
        num_train_timesteps: int = 1000,
    ) -> None:
        """Initialize the CGB-DM denoising network."""
        super().__init__()
        self.num_labels = int(num_labels)
        self.max_seq_length = int(max_seq_length)
        self.image_size: tuple[int, int] = (int(image_size[0]), int(image_size[1]))
        self.seq_dim = self.num_labels + 4
        self.img_encoder = CGBDMImageEncoder(
            image_size=self.image_size,
            patch_size=patch_size,
            in_channels=4,
            dim_model=dim_model,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.layout_encoder = CGBDMLayoutModule(
            seq_dim=self.seq_dim,
            dim_model=dim_model,
            n_head=n_head,
            feature_dim=feature_dim,
            num_layers=num_layers // 2,
            num_train_timesteps=num_train_timesteps,
            max_seq_length=self.max_seq_length,
            if_encoder=True,
        )
        self.layout_decoder = CGBDMLayoutModule(
            seq_dim=self.seq_dim,
            dim_model=dim_model,
            n_head=n_head,
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_train_timesteps=num_train_timesteps,
            max_seq_length=self.max_seq_length,
            if_encoder=False,
        )
        self.cgbwp = CGBDMQFormer(in_dim=dim_model, num_tokens=1)
        self.salbox_encoder = CGBDMMLP(4, dim_model, dim_model)

    def forward(
        self,
        sample: Float[torch.Tensor, "batch elements channels"],
        image: Float[torch.Tensor, "batch 4 height width"],
        saliency_box: Float[torch.Tensor, "batch 1 4"],
        timestep: Int[torch.Tensor, "batch"],
        return_dict: bool = True,
    ) -> CGBDMModelOutput | tuple[Float[torch.Tensor, "batch elements channels"]]:
        """Predict epsilon for a noisy layout tensor.

        Args:
            sample: Noisy class-plus-box layout tensor.
            image: Four-channel RGB/saliency tensor in ``[-1, 1]``.
            saliency_box: Saliency box tensor in internal ``[-1, 1]`` center xywh.
            timestep: Per-example diffusion timestep ids.
            return_dict: Whether to return ``CGBDMModelOutput``.

        Returns:
            Output dataclass or one-item tuple containing predicted epsilon.
        """
        image_tokens = self.img_encoder(image)
        saliency_tokens = self.salbox_encoder(saliency_box)
        encoded = self.layout_encoder(sample, None, None, None, timestep)
        cgb_weight = self.cgbwp(image_tokens)
        pred = self.layout_decoder(
            encoded,
            image_tokens,
            cgb_weight,
            saliency_tokens,
            timestep,
        )
        if not return_dict:
            return (pred,)
        return CGBDMModelOutput(sample=pred, cgb_weight=cgb_weight)
