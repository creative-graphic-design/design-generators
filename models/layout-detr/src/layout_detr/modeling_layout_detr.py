"""Transformers-compatible LayoutDETR model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from transformers import PreTrainedModel
from transformers import BertConfig, BertLMHeadModel
from transformers.utils import ModelOutput

from .configuration_layout_detr import LayoutDetrConfig


class _CrossAttentionSelfProtocol(Protocol):
    key: nn.Linear
    value: nn.Linear


class _CrossAttentionProtocol(Protocol):
    self: _CrossAttentionSelfProtocol


class _BertLayerWithCrossAttentionProtocol(Protocol):
    crossattention: _CrossAttentionProtocol


@dataclass
class LayoutDetrModelOutput(ModelOutput):
    """Raw LayoutDETR model output."""

    bbox: Float[torch.Tensor, "batch elements 4"]
    labels: Int[torch.Tensor, "batch elements"] = cast(
        Int[torch.Tensor, "batch elements"], None
    )
    mask: Bool[torch.Tensor, "batch elements"] = cast(
        Bool[torch.Tensor, "batch elements"], None
    )
    latents: Float[torch.Tensor, "batch elements latent"] | None = None
    hidden_states: Float[torch.Tensor, "batch elements hidden"] | None = None


class LayoutDetrForConditionalGeneration(PreTrainedModel):
    """A standard ``PreTrainedModel`` wrapper for LayoutDETR forward inference."""

    config_class = LayoutDetrConfig
    base_model_prefix = "layout_detr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: LayoutDetrConfig) -> None:
        """Initialize LayoutDETR layers."""
        super().__init__(config)
        self._is_vendor_architecture = config.architecture == "vendor"
        if self._is_vendor_architecture:  # pragma: no cover
            self._init_vendor_layers(config)
        else:
            self._init_lightweight_layers(config)
        self.post_init()

    def _init_lightweight_layers(self, config: LayoutDetrConfig) -> None:
        self.background_encoder = nn.Sequential(
            nn.Conv2d(config.img_channels, config.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_z = nn.Linear(config.z_dim, config.bert_f_dim)
        self.emb_label = nn.Embedding(config.num_bbox_labels, config.bert_f_dim)
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.bert_f_dim)
        self.text_len_embeddings = nn.Embedding(
            config.max_text_length, config.bert_f_dim
        )
        self.background_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_in = nn.Sequential(
            nn.Linear(config.bert_f_dim * 4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, min(8, config.hidden_dim // 8)),
            dim_feedforward=max(config.hidden_dim * 4, 64),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.bbox_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 4),
        )

    def _init_vendor_layers(self, config: LayoutDetrConfig) -> None:  # pragma: no cover
        self.backbone = _build_backbone(config.backbone_name)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, config.hidden_dim, 1)
        self.fc_z = nn.Linear(config.z_dim * config.max_seq_length, config.bert_f_dim)
        self.emb_label = nn.Embedding(config.num_bbox_labels, config.bert_f_dim)
        self.text_encoder = _build_vendor_bert_model(
            config,
            num_hidden_layers=config.bert_num_encoder_layers,
            encoder_width=config.bert_f_dim,
            add_pooling_layer=False,
            use_cross_attention=True,
            text_encoder=True,
        )
        self.enc_text_len = nn.Embedding(config.max_text_length, config.bert_f_dim)
        self.fc_in = _VendorMLP(
            input_dim=config.bert_f_dim * 4,
            hidden_dim=config.bert_f_dim,
            output_dim=config.hidden_dim,
            num_layers=3,
        )
        self.transformer = _DetrTransformer(
            d_model=config.hidden_dim,
            dropout=0.1,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )
        self.bbox_embed = _VendorMLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=4,
            num_layers=3,
        )
        self.fc_z_rec = nn.Linear(
            config.hidden_dim, config.z_dim * config.max_seq_length
        )
        self.fc_out_cls = nn.Linear(config.hidden_dim, config.num_bbox_labels)
        self.text_decoder = _build_vendor_bert_lm_head(
            config,
            num_hidden_layers=config.bert_num_decoder_layers,
            encoder_width=512,
        )
        self.fc_text_len_rec = nn.Linear(config.hidden_dim, config.max_text_length)

    def forward(
        self,
        *,
        pixel_values: Float[torch.Tensor, "batch channels height width"],
        input_ids: Int[torch.Tensor, "batch elements tokens"],
        text_attention_mask: Bool[torch.Tensor, "batch elements tokens"],
        bbox_labels: Int[torch.Tensor, "batch elements"],
        layout_mask: Bool[torch.Tensor, "batch elements"],
        latents: Float[torch.Tensor, "batch elements latent"],
        text_lengths: Int[torch.Tensor, "batch elements"] | None = None,
        return_dict: bool | None = None,
    ) -> LayoutDetrModelOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the LayoutDETR conditional forward pass."""
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if bbox_labels.ndim != 2:
            raise ValueError("bbox_labels must have shape (batch, elements)")
        if (
            latents.shape[:2] != bbox_labels.shape
            or latents.shape[-1] != self.config.z_dim
        ):
            raise ValueError("latents must have shape (batch, elements, z_dim)")
        if input_ids.shape[:2] != bbox_labels.shape:
            raise ValueError("input_ids must have shape (batch, elements, tokens)")
        labels = bbox_labels.to(dtype=torch.long)
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= self.config.num_bbox_labels
        ):
            raise ValueError("bbox_labels contain ids outside config.num_bbox_labels")
        device = labels.device
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        latents = latents.to(device=device, dtype=self.dtype)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        text_attention_mask = text_attention_mask.to(device=device, dtype=torch.bool)
        layout_mask = layout_mask.to(device=device, dtype=torch.bool)

        if self._is_vendor_architecture:
            bbox, hidden = self._forward_vendor(
                pixel_values=pixel_values,
                input_ids=input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels,
                layout_mask=layout_mask,
                latents=latents,
                text_lengths=text_lengths,
            )
            if not return_dict:
                return bbox, labels, layout_mask
            return LayoutDetrModelOutput(
                bbox=bbox,
                labels=labels,
                mask=layout_mask,
                latents=latents,
                hidden_states=hidden,
            )

        bg = self.background_proj(self.background_encoder(pixel_values)).unsqueeze(1)
        z = self.fc_z(latents)
        label_features = self.emb_label(labels)
        text_tokens = self.text_embeddings(input_ids)
        token_mask = text_attention_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
        denom = token_mask.sum(dim=2).clamp_min(1.0)
        text_features = (text_tokens * token_mask).sum(dim=2) / denom
        lengths = text_attention_mask.sum(dim=-1).clamp_max(
            self.config.max_text_length - 1
        )
        text_len_features = self.text_len_embeddings(lengths)
        hidden = self.fc_in(
            torch.cat([z, label_features, text_features, text_len_features], dim=-1)
        )
        hidden = hidden + bg
        hidden = self.transformer(hidden, src_key_padding_mask=~layout_mask)
        bbox = torch.sigmoid(self.bbox_embed(hidden))
        if not return_dict:
            return bbox, labels, layout_mask
        return LayoutDetrModelOutput(
            bbox=bbox,
            labels=labels,
            mask=layout_mask,
            latents=latents,
            hidden_states=hidden,
        )

    def _forward_vendor(  # pragma: no cover
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        layout_mask: torch.Tensor,
        latents: torch.Tensor,
        text_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bg_nested = _NestedTensor(
            pixel_values,
            torch.zeros(
                pixel_values.shape[0],
                pixel_values.shape[2],
                pixel_values.shape[3],
                dtype=torch.bool,
                device=pixel_values.device,
            ),
        )
        bg_feat, pos = self.backbone(bg_nested)
        bg_tensor, bg_mask = bg_feat[-1].decompose()
        z0 = _normalize_2nd_moment(latents.reshape(latents.shape[0], -1))
        z = self.fc_z(z0).unsqueeze(1).expand(-1, labels.shape[1], -1)
        label_features = self.emb_label(labels)
        flat_input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        flat_attention = text_attention_mask.reshape(-1, text_attention_mask.shape[-1])
        text_output = self.text_encoder(
            flat_input_ids,
            attention_mask=flat_attention,
            return_dict=True,
            mode="text",
        )
        text_features = text_output.last_hidden_state[:, 0, :].view(
            labels.shape[0], labels.shape[1], -1
        )
        if text_lengths is None:
            lengths = flat_attention.sum(dim=-1)
        else:
            lengths = text_lengths.to(device=labels.device, dtype=torch.long).reshape(
                -1
            )
        lengths = lengths.clamp_max(self.config.max_text_length - 1)
        text_len_features = self.enc_text_len(lengths.view(labels.shape))
        hidden = torch.cat(
            [z, label_features, text_features, text_len_features], dim=-1
        )
        hidden = torch.relu(self.fc_in(hidden)).permute(1, 0, 2)
        hidden = self.transformer(
            src=self.input_proj(bg_tensor),
            mask=bg_mask,
            pos_embed=pos[-1],
            tgt=hidden,
            tgt_key_padding_mask=~layout_mask,
        )[0]
        return torch.sigmoid(self.bbox_embed(hidden)), hidden


class _NestedTensor:  # pragma: no cover
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensors, self.mask


class _FrozenBatchNorm2d(nn.Module):  # pragma: no cover
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = cast(torch.Tensor, self.weight).reshape(1, -1, 1, 1)
        bias = cast(torch.Tensor, self.bias).reshape(1, -1, 1, 1)
        running_var = cast(torch.Tensor, self.running_var).reshape(1, -1, 1, 1)
        running_mean = cast(torch.Tensor, self.running_mean).reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        return x * scale + (bias - running_mean * scale)


class _BackboneBase(nn.Module):  # pragma: no cover
    def __init__(self, backbone: nn.Module, num_channels: int) -> None:
        super().__init__()
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: _NestedTensor) -> dict[str, _NestedTensor]:
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(),
                size=x.shape[-2:],
            ).to(torch.bool)[0]
            out[name] = _NestedTensor(x, mask)
        return out


class _Backbone(_BackboneBase):  # pragma: no cover
    def __init__(self, name: str) -> None:
        if name != "resnet50":
            raise ValueError(f"Unsupported LayoutDETR backbone: {name}")
        backbone = resnet50(
            weights=None,
            replace_stride_with_dilation=[False, False, False],
            norm_layer=_FrozenBatchNorm2d,
        )
        super().__init__(backbone, 2048)


class _PositionEmbeddingSine(nn.Module):  # pragma: no cover
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: _NestedTensor) -> torch.Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _Joiner(nn.Module):  # pragma: no cover
    num_channels: int

    def __init__(
        self,
        backbone: _Backbone,
        position_embedding: _PositionEmbeddingSine,
    ) -> None:
        super().__init__()
        self.add_module("0", backbone)
        self.add_module("1", position_embedding)

    def forward(
        self,
        tensor_list: _NestedTensor,
    ) -> tuple[list[_NestedTensor], list[torch.Tensor]]:
        backbone = cast(nn.Module, self._modules["0"])
        position_embedding = cast(nn.Module, self._modules["1"])
        xs = backbone(tensor_list)
        out = []
        pos = []
        for x in xs.values():
            out.append(x)
            pos.append(position_embedding(x).to(x.tensors.dtype))
        return out, pos


def _build_backbone(name: str) -> _Joiner:  # pragma: no cover
    backbone = _Backbone(name)
    position_embedding = _PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    model = _Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class _VendorMLP(nn.Module):  # pragma: no cover
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(
                [input_dim, *hidden],
                [*hidden, output_dim],
                strict=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class _DetrTransformer(nn.Module):  # pragma: no cover
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = _DetrTransformerEncoder(
            _DetrTransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
            ),
            num_encoder_layers,
        )
        self.decoder = _DetrTransformerDecoder(
            _DetrTransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
            ),
            num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.d_model = d_model
        self.nhead = nhead

    def forward(
        self,
        *,
        src: torch.Tensor,
        mask: torch.Tensor,
        pos_embed: torch.Tensor,
        tgt: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = src.shape
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1)
        memory_mask = mask.flatten(1)
        memory = self.encoder(
            src_flat,
            src_key_padding_mask=memory_mask,
            pos=pos_flat,
        )
        hidden = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            pos=pos_flat,
        )
        return (
            hidden.transpose(0, 1),
            memory.permute(1, 2, 0).view(batch_size, channels, height, width),
        )


class _DetrTransformerEncoder(nn.Module):  # pragma: no cover
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = None

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        return output


class _DetrTransformerDecoder(nn.Module):  # pragma: no cover
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = False

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
        return self.norm(output)


class _DetrTransformerEncoderLayer(nn.Module):  # pragma: no cover
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.normalize_before = False

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)


class _DetrTransformerDecoderLayer(nn.Module):  # pragma: no cover
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.normalize_before = False

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = tgt
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=tgt,
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)


def _bert_config(
    config: LayoutDetrConfig,
    *,
    num_hidden_layers: int,
    encoder_width: int,
    use_cross_attention: bool,
) -> BertConfig:  # pragma: no cover
    bert_config = BertConfig(
        vocab_size=config.text_vocab_size,
        hidden_size=config.bert_f_dim,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=config.bert_num_heads,
        intermediate_size=config.bert_f_dim * 4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        add_cross_attention=use_cross_attention,
        is_decoder=use_cross_attention,
    )
    bert_config.encoder_width = encoder_width
    return bert_config


def _build_vendor_bert_model(
    config: LayoutDetrConfig,
    *,
    num_hidden_layers: int,
    encoder_width: int,
    add_pooling_layer: bool,
    use_cross_attention: bool,
    text_encoder: bool = False,
) -> "_VendorBertModel":  # pragma: no cover
    bert_config = _bert_config(
        config,
        num_hidden_layers=num_hidden_layers,
        encoder_width=encoder_width,
        use_cross_attention=use_cross_attention,
    )
    model = _VendorBertModel(bert_config, add_pooling_layer=add_pooling_layer)
    if text_encoder:
        model.config.is_decoder = False
    return model


def _build_vendor_bert_lm_head(
    config: LayoutDetrConfig,
    *,
    num_hidden_layers: int,
    encoder_width: int,
) -> BertLMHeadModel:  # pragma: no cover
    bert_config = _bert_config(
        config,
        num_hidden_layers=num_hidden_layers,
        encoder_width=encoder_width,
        use_cross_attention=True,
    )
    model = BertLMHeadModel(bert_config)
    model.bert.embeddings = _VendorBertEmbeddings(bert_config)  # ty: ignore[invalid-assignment]
    for layer in model.bert.encoder.layer:
        cross_attention = cast(
            _BertLayerWithCrossAttentionProtocol, layer
        ).crossattention.self
        cross_attention.key = nn.Linear(encoder_width, config.bert_f_dim)
        cross_attention.value = nn.Linear(encoder_width, config.bert_f_dim)
    return model


class _VendorBertEmbeddings(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=True,
        )
        self.position_embedding_type = getattr(
            config,
            "position_embedding_type",
            "absolute",
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        del token_type_ids
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("input_ids or inputs_embeds must be provided")
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = cast(torch.Tensor, self.position_ids)[
                :,
                past_key_values_length : seq_length + past_key_values_length,
            ]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


@dataclass
class _VendorBertOutput(ModelOutput):  # pragma: no cover
    last_hidden_state: torch.Tensor


class _VendorBertModel(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig, add_pooling_layer: bool = False) -> None:
        super().__init__()
        self.config = config
        self.embeddings = _VendorBertEmbeddings(config)
        self.encoder = _VendorBertEncoder(config)
        self.pooler = (
            None
            if not add_pooling_layer
            else nn.Linear(
                config.hidden_size,
                config.hidden_size,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        mode: str = "multimodal",
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> _VendorBertOutput | tuple[torch.Tensor]:
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        extended_attention_mask = self._extended_attention_mask(
            attention_mask,
            input_shape,
        )
        encoder_extended_attention_mask = None
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_states.size()[:-1],
                    device=input_ids.device,
                )
            encoder_extended_attention_mask = self._invert_attention_mask(
                encoder_attention_mask
            )
        hidden_states = self.embeddings(input_ids=input_ids)
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            mode=mode,
        )
        if not return_dict:
            return (hidden_states,)
        return _VendorBertOutput(last_hidden_state=hidden_states)

    def _extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: torch.Size,
    ) -> torch.Tensor:
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Unsupported attention_mask shape: {attention_mask.shape}"
            )
        extended = extended.to(dtype=next(self.parameters()).dtype)
        return (1.0 - extended) * -10000.0

    def _invert_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        extended = attention_mask[:, None, None, :].to(
            dtype=next(self.parameters()).dtype
        )
        return (1.0 - extended) * -10000.0


class _VendorBertEncoder(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_VendorBertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        mode: str,
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                mode=mode,
            )
        return hidden_states


class _VendorBertLayer(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = _VendorBertAttention(config)
        if config.add_cross_attention:
            self.crossattention = _VendorBertAttention(
                config,
                is_cross_attention=True,
            )
        self.intermediate = _VendorBertIntermediate(config)
        self.output = _VendorBertOutputLayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        mode: str,
    ) -> torch.Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        if mode == "multimodal":
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states is required for multimodal")
            attention_output = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class _VendorBertAttention(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig, is_cross_attention: bool = False) -> None:
        super().__init__()
        self.self = _VendorBertSelfAttention(config, is_cross_attention)
        self.output = _VendorBertSelfOutput(config)
        self.pruned_heads: set[int] = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_output = self.self(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        return self.output(self_output, hidden_states)


class _VendorBertSelfAttention(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig, is_cross_attention: bool) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        key_value_input = (
            config.encoder_width if is_cross_attention else config.hidden_size
        )
        self.key = nn.Linear(key_value_input, self.all_head_size)
        self.value = nn.Linear(key_value_input, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config,
            "position_embedding_type",
            "absolute",
        )
        self.save_attention = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_value_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key_layer = self.transpose_for_scores(self.key(key_value_states))
        value_layer = self.transpose_for_scores(self.value(key_value_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(*new_shape)


class _VendorBertSelfOutput(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class _VendorBertIntermediate(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class _VendorBertOutputLayer(nn.Module):  # pragma: no cover
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


def _normalize_2nd_moment(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:  # pragma: no cover
    return x * (x.square().mean(dim=1, keepdim=True) + eps).rsqrt()
