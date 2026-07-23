"""PyTorch model wrapper for standalone RALF checkpoints."""

# ruff: noqa: D102,D107

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, einsum
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_ralf import (
    RalfConfig,
    RalfConfigTaskName,
    RalfLayoutVariable,
    RalfTaskName,
)
from .retrieval import RalfRetrievedBatch, retrieved_batch_to_model_inputs

GEO_KEYS: tuple[str, ...] = ("width", "height", "center_x", "center_y")
FID_BBOX_KEYS: tuple[str, ...] = ("center_x", "center_y", "width", "height")
TASK_TOKEN_VOCABULARIES: tuple[str, ...] = (
    "end_of_task",
    "label",
    "label_size",
    "relationship",
    "refinement",
    "completion",
    "uncondition",
)
SPECIAL_TASK_TOKENS: tuple[str, ...] = ("sep", "relation_sep", "canvas")
RELATIONSHIP_TOKENS: tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(10))
RELATIONSHIP_POSITION_TOKENS: tuple[str, ...] = (
    "unknown_loc",
    "left",
    "top",
    "right",
    "bottom",
    "center",
)
RELATIONSHIP_SIZE_TOKENS: tuple[str, ...] = (
    "unknown_size",
    "smaller",
    "equal",
    "larger",
)
RalfTaskTokenName = Literal[
    "uncondition",
    "label",
    "label_size",
    "relationship",
    "refinement",
    "completion",
]

TASK_BY_CONDITION: Final[dict[RalfConfigTaskName, RalfTaskName]] = {
    "unconditional": "uncond",
    "retrieval": "uncond",
    "content_image": "uncond",
    "label": "c",
    "label_size": "cwh",
    "completion": "partial",
    "refinement": "refinement",
    "relation": "relation",
    "uncond": "uncond",
    "c": "c",
    "cwh": "cwh",
    "chw": "cwh",
    "partial": "partial",
}
TASK_TOKEN_BY_TASK: Final[dict[RalfTaskName, RalfTaskTokenName]] = {
    "uncond": "uncondition",
    "c": "label",
    "cwh": "label_size",
    "partial": "completion",
    "refinement": "refinement",
    "relation": "relationship",
}
PREPROCESSOR_VAR_BY_TASK: Final[dict[RalfTaskName, tuple[RalfLayoutVariable, ...]]] = {
    "c": ("label",),
    "cwh": ("label", "width", "height"),
    "partial": ("label", "width", "height", "center_x", "center_y"),
    "refinement": ("label", "width", "height", "center_x", "center_y"),
    "relation": ("label",),
}


class ImageReshaper(nn.Module):
    """Reshape image feature maps to transformer memory."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) != self.d_model:
            raise ValueError(f"{x.size(1)} != {self.d_model}")
        return rearrange(x, "b c h w -> b (h w) c")


class PositionalEncoding1d(nn.Module):
    """RALF sine positional encoding for token sequences."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
        scale_input: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.scale_input = scale_input
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        if batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        h = x * math.sqrt(self.d_model) if self.scale_input else x
        pe = cast(Tensor, self.pe)
        if self.batch_first:
            h = h + pe[:, : h.size(1)]
        else:
            h = h + pe[: h.size(0)]
        return self.dropout(h)


class PositionEmbeddingSine(nn.Module):
    """RALF 2D sine positional encoding."""

    def __init__(
        self,
        d_model: int = 256,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale
        self.reshape = ImageReshaper(d_model)

    def forward(self, input: Tensor) -> Tensor:
        bs, _c, h, w = input.size()
        y, x = torch.meshgrid(
            torch.arange(h).type_as(input),
            torch.arange(w).type_as(input),
            indexing="ij",
        )
        if self.normalize:
            y = y / (h - 1)
            x = x / (w - 1)
            y = y * self.scale
            x = x * self.scale
        dim_t = torch.arange(self.d_model).type_as(input)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.d_model
        )
        pos_x = x.flatten()[None, :, None] / dim_t
        pos_y = y.flatten()[None, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).repeat(bs, 1, 1)
        return self.reshape(input) + pos


class FeedForward(nn.Module):
    """RALF MLP block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        output_dim = dim if output_dim is None else output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """RALF cross-attention block."""

    def __init__(
        self,
        dim_q: int,
        dimvq: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim_q)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim_q, inner_dim, bias=False)
        self.to_kv = nn.Linear(dimvq, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim_q), nn.Dropout(dropout))

    def forward(
        self, x: Tensor, context: Tensor | None = None, kv_include_self: bool = False
    ) -> Tensor:
        b, _n, _d = x.shape
        h = self.heads
        x = self.norm(x)
        context = x if context is None else context
        if kv_include_self:
            context = torch.cat((x, context), dim=1)
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.dropout(self.attend(dots))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", b=b)
        return self.to_out(out)


class ResnetBackbone(nn.Module):
    """RALF ResNet50 FPN feature extractor without external loads."""

    def __init__(
        self, backbone: str = "resnet50", d_model: int = 256, head: str = "transformer"
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("RALF converted checkpoints use resnet50")
        resnet = timm.create_model("resnet50", pretrained=False)
        return_nodes = {"layer4": "layer4", "layer3": "layer3"}
        self.body = create_feature_extractor(resnet, return_nodes=return_nodes)
        params = {
            key: getattr(self.body.conv1, key)
            for key in ["kernel_size", "stride", "padding", "out_channels"]
        }
        conv1 = cast(nn.Conv2d, self.body.conv1)
        weight = conv1.weight.data
        weight = torch.cat([weight, torch.mean(weight, dim=1, keepdim=True)], dim=1)
        self.body.conv1 = nn.Conv2d(in_channels=4, bias=False, **params)
        self.body.conv1.weight.data = weight
        self.fpn_conv11_4 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.fpn_conv11_5 = nn.Conv2d(2048, 256, 1, 1, 0)
        self.fpn_conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.proj = nn.Conv2d(512, d_model, 1, 1, 0)
        if head != "transformer":
            raise ValueError("RALF converted checkpoints use transformer image head")
        self.head = head

    def forward(self, img: Tensor) -> Tensor:
        h = self.body(img)
        resnet_f4 = h["layer3"]
        resnet_f5 = h["layer4"]
        resnet_f4p = self.fpn_conv11_4(resnet_f4)
        resnet_f5p = self.fpn_conv11_5(resnet_f5)
        resnet_f5up = F.interpolate(
            resnet_f5p, size=resnet_f4p.shape[2:], mode="nearest"
        )
        resnet_fused = torch.concat(
            [resnet_f5up, self.fpn_conv33(resnet_f5up + resnet_f4p)], dim=1
        )
        return self.proj(resnet_fused)


class ResnetFeatureExtractor(nn.Module):
    """Container preserving checkpoint key prefix."""

    def __init__(
        self,
        backbone: str = "resnet50",
        d_model: int = 256,
        head: str = "transformer",
    ) -> None:
        super().__init__()
        self.extractor = ResnetBackbone(backbone=backbone, d_model=d_model, head=head)

    def forward(self, *args: object) -> Tensor:
        return self.extractor(*args)


class TransformerWithToken(nn.Module):
    """FIDNet transformer encoder with a learned summary token."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int
    ) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer("token_mask", torch.zeros(1, 1, dtype=torch.bool))
        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def forward(self, x: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        batch = x.size(1)
        token = self.token.expand(-1, batch, -1)
        x = torch.cat([token, x], dim=0)
        token_mask = cast(Tensor, self.token_mask).expand(batch, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)
        return self.core(x, src_key_padding_mask=padding_mask)


class FIDNetFeatureExtractor(nn.Module):
    """FIDNet encoder subset used by RALF retrieval layout features."""

    def __init__(
        self,
        num_label: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        max_bbox: int = 10,
    ) -> None:
        super().__init__()
        _ = max_bbox
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)
        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

    def extract_features(self, inputs: Mapping[str, Tensor]) -> Tensor:
        padding_mask = ~inputs["mask"]
        bbox = torch.stack([inputs[key] for key in FID_BBOX_KEYS], dim=-1)
        h_bbox = self.fc_bbox(bbox)
        h_label = self.emb_label(inputs["label"].long())
        x = self.enc_fc_in(torch.cat([h_bbox, h_label], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]


class BaseDecoder(nn.Module):
    """RALF autoregressive transformer decoder."""

    def __init__(
        self,
        d_label: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        pos_emb: str = "layout",
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        if pos_emb != "layout":
            raise ValueError(
                "RALF converted checkpoints use layout positional encoding"
            )
        self.tie_weights = False
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        self.d_model = d_model
        self.emb = nn.Embedding(d_label, d_model)
        self.pos_emb = PositionalEncoding1d(d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_label, bias=False)
        )
        self.use_paramter_ablation = False

    def init_weight(self) -> None:
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        for module in self.head:
            if isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        h = self.pos_emb(self.emb(tgt))
        if is_causal:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(h.size(1))
            h = self.transformer(
                h,
                memory,
                tgt_mask=tgt_mask.to(h.device),
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        else:
            h = self.transformer(h, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.head(h)


class UserConstraintTransformerEncoder(nn.Module):
    """RALF encoder for task/user constraint tokens."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_label: int,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=0.1,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )
        self.emb = nn.Embedding(d_label, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.pos_emb = PositionalEncoding1d(d_model=d_model)

    def init_weight(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src: Tensor, src_key_padding_mask: Tensor, task_token: Tensor | None
    ) -> Tensor:
        h = self.pos_emb(self.emb(src))
        h = self.encoder(src=h, src_key_padding_mask=src_key_padding_mask)
        if task_token is not None:
            h = h + self.emb(task_token)
        return h


class RalfTokenizerView:
    """Small tokenizer view matching the token ids needed by the model."""

    def __init__(self, config: RalfConfig) -> None:
        self.config = config
        id2label = cast(dict[int, str], config.id2label)
        self.label_names = [
            label
            for _idx, label in sorted(id2label.items(), key=lambda item: int(item[0]))
        ]
        self.var_order = list(config.var_order)
        self.special_tokens = list(config.special_tokens)
        self._label_feature = self
        self.names = self.label_names
        self.num_classes = len(self.label_names)
        self._special_token_name_to_id = {
            token: self.name_to_id(token) for token in self.special_tokens
        }

    @property
    def N_label(self) -> int:
        return self.num_classes

    @property
    def N_bbox_per_var(self) -> int:
        return self.config.num_bin

    @property
    def N_bbox(self) -> int:
        return self.config.num_bbox_tokens

    @property
    def N_sp_token(self) -> int:
        return len(self.special_tokens)

    @property
    def N_total(self) -> int:
        return self.config.vocab_size

    @property
    def max_seq_length(self) -> int:
        return self.config.max_seq_length

    @property
    def max_token_length(self) -> int:
        return self.config.max_token_length

    def name_to_id(self, name: str) -> int:
        return self.config.special_token_id(name)


class RalfTaskPreprocessor:
    """RALF task token preprocessor."""

    def __init__(
        self,
        tokenizer: RalfTokenizerView,
        *,
        task: RalfTaskName,
        global_task_embedding: bool = False,
        relationship_table: Mapping[str, list[object]] | None = None,
        relation_size: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.global_task_embedding = global_task_embedding
        self.task_name = task
        self.relationship_table = (
            {
                str(key): random.sample(values, len(values))
                for key, values in relationship_table.items()
            }
            if relationship_table is not None
            else None
        )
        self.relation_size = int(relation_size)
        self._TASK = TASK_TOKEN_BY_TASK[task]
        self._VAR = PREPROCESSOR_VAR_BY_TASK.get(task, ())
        self.device = torch.device("cpu")
        tokens = (
            TASK_TOKEN_VOCABULARIES
            + SPECIAL_TASK_TOKENS
            + RELATIONSHIP_TOKENS
            + RELATIONSHIP_POSITION_TOKENS
            + RELATIONSHIP_SIZE_TOKENS
        )
        self._preprocess_token_name_to_id = {
            token: idx + self.tokenizer.N_total for idx, token in enumerate(tokens)
        }
        labelname_to_id = {
            name: self.tokenizer.names.index(name) for name in self.tokenizer.names
        }
        self._token_to_name_to_id = {
            **self.tokenizer._special_token_name_to_id,
            **self._preprocess_token_name_to_id,
            **labelname_to_id,
        }

    @property
    def TASK(self) -> RalfTaskTokenName:
        return self._TASK

    @property
    def N_total(self) -> int:
        return self.tokenizer.N_total + len(self._preprocess_token_name_to_id)

    def name_to_id(self, name: str) -> int:
        return self._token_to_name_to_id[name]

    def _relation_item_to_name(self, item: object) -> str:
        name = getattr(item, "name", item)
        class_name = item.__class__.__name__
        if not isinstance(name, str):
            return str(name)
        if name == "UNKNOWN":
            return "unknown_size" if class_name == "RelSize" else "unknown_loc"
        relation_names = {
            "LEFT": "left",
            "TOP": "top",
            "RIGHT": "right",
            "BOTTOM": "bottom",
            "CENTER": "center",
            "SMALLER": "smaller",
            "EQUAL": "equal",
            "LARGER": "larger",
        }
        return relation_names.get(name, name)

    def _relation_item_to_id(self, item: object) -> int:
        return self.name_to_id(self._relation_item_to_name(item))

    def get_token(self, name: str, batch_size: int) -> Tensor:
        return torch.full((batch_size, 1), self.name_to_id(name), device=self.device)

    def create_task_token(self, batch_size: int) -> Tensor:
        return torch.cat(
            [
                self.get_token(self.TASK, batch_size),
                self.get_token("end_of_task", batch_size),
            ],
            dim=-1,
        )

    def create_pad_mask(self, seq: Tensor) -> Tensor:
        return seq == self.name_to_id("pad")

    def _parse_seq_into_vars(self, seq: Tensor) -> dict[str, Tensor]:
        seq = seq.clone()
        seq[seq == self.name_to_id("eos")] = self.name_to_id("pad")
        seq = seq[:, 1:]
        seq = seq.reshape(seq.size(0), -1, len(self.tokenizer.var_order))
        return {key: seq[..., idx] for idx, key in enumerate(self.tokenizer.var_order)}

    def _shuffle_seq_vars(self, seq_vars: dict[str, Tensor]) -> dict[str, Tensor]:
        label = seq_vars["label"]
        non_padding_counts = (label != self.name_to_id("pad")).sum(dim=1)
        shuffled = {key: value.clone() for key, value in seq_vars.items()}
        for batch_idx, count in enumerate(non_padding_counts.tolist()):
            if count <= 1:
                continue
            indexes = torch.randperm(count, device=label.device)
            for key, value in seq_vars.items():
                shuffled[key][batch_idx, :count] = value[batch_idx, indexes]
        return shuffled

    def _valid_element_mask(self, seq_vars: Mapping[str, Tensor]) -> Tensor:
        label = seq_vars["label"]
        return (label != self.name_to_id("pad")) & (label != self.name_to_id("eos"))

    def _geo_sequence(self, inputs: "RalfConditionalInputs") -> Tensor:
        if inputs.seq is None:
            raise ValueError(f"condition_type={self.task_name!r} requires labels")
        seq = inputs.seq
        if self.task_name == "partial" and inputs.mask is not None:
            seq = seq.clone()
            seq[~inputs.mask.bool()] = self.name_to_id("pad")
        seq_vars = self._parse_seq_into_vars(seq)
        if self.task_name == "relation":
            _ = self._shuffle_seq_vars(seq_vars)
            seq_vars = self._shuffle_seq_vars(seq_vars)
        elif self.task_name == "c":
            seq_vars = self._shuffle_seq_vars(seq_vars)
        valid = (
            self._valid_element_mask(seq_vars)
            if inputs.element_mask is None
            else inputs.element_mask[:, : seq_vars["label"].size(1)].bool()
        )
        if self.task_name == "partial":
            valid = torch.zeros_like(valid)
            if valid.size(1) > 0:
                valid[:, 0] = True
        max_valid = int(valid.sum(dim=1).max().item()) if valid.numel() else 0
        if max_valid == 0:
            return self.get_token("pad", inputs.image.size(0))
        pieces: list[Tensor] = []
        sep = self.get_token("sep", inputs.image.size(0))
        for element_idx in range(max_valid):
            for key in self._VAR:
                values = seq_vars[key][:, element_idx : element_idx + 1]
                values = torch.where(
                    valid[:, element_idx : element_idx + 1],
                    values,
                    self.get_token("pad", inputs.image.size(0)),
                )
                pieces.append(values)
            if element_idx != max_valid - 1:
                pieces.append(sep)
        return torch.cat(pieces, dim=1)

    def _relation_ids(self, ids: object, batch_size: int) -> list[str]:
        if ids is None:
            return [""] * batch_size
        if isinstance(ids, Tensor):
            return [str(item) for item in ids.detach().cpu().tolist()]
        if isinstance(ids, (list, tuple)):
            return [str(item) for item in ids]
        return [str(ids)] * batch_size

    def _relation_sequence(
        self, inputs: "RalfConditionalInputs", label_sequence: Tensor
    ) -> Tensor:
        if self.relationship_table is None:
            return label_sequence
        batch = label_sequence.size(0)
        label_mask = self.create_pad_mask(label_sequence)
        label_sequence = label_sequence.clone()
        if not self.global_task_embedding:
            label_sequence[:, 1] = self.get_token(self.TASK, batch)[:, 0]
        label_sequence[label_sequence == self.name_to_id("eos")] = self.name_to_id(
            "relation_sep"
        )

        outputs = []
        max_length = 0
        for batch_idx, item_id in enumerate(self._relation_ids(inputs.id, batch)):
            seq = label_sequence[batch_idx][~label_mask[batch_idx]]
            relations = self.relationship_table.get(item_id, [])
            if not relations:
                seq = torch.cat([seq, self.get_token("eos", 1)[0]], dim=0)
                outputs.append(seq)
                max_length = max(max_length, seq.size(0))
                continue
            sample_size = max(len(relations) * self.relation_size // 100, 1)
            sampled = random.sample(relations, sample_size)
            relation_tokenized = torch.tensor(
                [
                    [
                        self._relation_item_to_id(element)
                        for element in cast(list[object], relation)
                    ]
                    for relation in sampled
                ],
                dtype=torch.long,
                device=self.device,
            )
            sep = self.get_token("sep", relation_tokenized.size(0))
            relation_with_sep = torch.cat([relation_tokenized, sep], dim=1).view(-1)
            relation_with_sep[-1] = self.name_to_id("eos")
            seq = torch.cat([seq, relation_with_sep], dim=0)
            outputs.append(seq)
            max_length = max(max_length, seq.size(0))
        out = torch.full(
            (batch, max_length),
            fill_value=self.name_to_id("pad"),
            dtype=torch.long,
            device=self.device,
        )
        for batch_idx, seq in enumerate(outputs):
            out[batch_idx, : seq.size(0)] = seq
        return out

    def __call__(self, inputs: "RalfConditionalInputs") -> dict[str, Tensor]:
        batch = inputs.image.size(0)
        self.device = inputs.image.device
        bos = self.get_token("bos", batch)
        eos = self.get_token("eos", batch)
        body = (
            torch.empty(batch, 0, dtype=torch.long, device=self.device)
            if self.task_name == "uncond"
            else self._geo_sequence(inputs)
        )
        if self.global_task_embedding:
            seq = torch.cat([bos, body, eos], dim=-1)
        else:
            seq = torch.cat([bos, self.create_task_token(batch), body, eos], dim=-1)
        if self.task_name == "relation":
            seq = self._relation_sequence(inputs, seq)
        return {"seq": seq.long(), "pad_mask": self.create_pad_mask(seq)}


@dataclass
class RalfConditionalInputs:
    """Model-side generation inputs."""

    image: Tensor
    retrieved: dict[str, Tensor]
    seq: Tensor | None = None
    mask: Tensor | None = None
    element_mask: Tensor | None = None
    task: RalfTaskName | None = "uncond"
    id: object = None


def _get_ref_layout_input(
    retrieved_samples: Mapping[str, Tensor], kdx: int
) -> dict[str, Tensor]:
    return {
        "center_x": retrieved_samples["center_x"][:, kdx],
        "center_y": retrieved_samples["center_y"][:, kdx],
        "width": retrieved_samples["width"][:, kdx],
        "height": retrieved_samples["height"][:, kdx],
        "label": retrieved_samples["label"][:, kdx].long(),
        "mask": retrieved_samples["mask"][:, kdx].bool(),
    }


def _extract_retrieved_features(
    *,
    retrieved_samples: Mapping[str, Tensor],
    top_k: int,
    layout_encoder: FIDNetFeatureExtractor,
    layout_adapter: FeedForward,
    pos_emb_1d: PositionalEncoding1d,
) -> Tensor:
    ref_layouts = []
    for kdx in range(top_k):
        ref_layout_input = _get_ref_layout_input(retrieved_samples, kdx)
        with torch.no_grad():
            feature_layout_ref = layout_encoder.extract_features(ref_layout_input)
        ref_layouts.append(layout_adapter(feature_layout_ref))
    stacked = torch.stack(ref_layouts, dim=1)
    return pos_emb_1d(stacked)


def _restrict_reliable_label_or_size(
    *,
    sampling_idx: int,
    condition: Tensor | None,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    if condition is None:
        return logits
    batch = condition.size(0)
    for batch_idx in range(batch):
        given = int(condition[batch_idx, sampling_idx].item())
        first_pad = torch.argmax(condition[batch_idx].eq(pad_id).float())
        first_pad_idx = (
            int(first_pad.item())
            if condition[batch_idx, first_pad].eq(pad_id).item()
            else max_length + 1
        )
        mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
        if sampling_idx < first_pad_idx:
            if given in (pad_id, -1):
                continue
            mask[given] = False
        else:
            mask[eos_id] = False
        logits[batch_idx, mask] = -math.inf
    return logits


def _restrict_only_category(
    *,
    sampling_idx: int,
    condition: Tensor | None,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    if (sampling_idx - 1) % 5 != 0:
        return logits
    return _restrict_reliable_label_or_size(
        sampling_idx=sampling_idx,
        condition=condition,
        logits=logits,
        pad_id=pad_id,
        eos_id=eos_id,
        max_length=max_length,
    )


def _apply_decode_space_restriction(
    *,
    task: RalfTaskName,
    step: int,
    condition: Tensor | None,
    logits: Tensor,
    pad_id: int,
    eos_id: int,
    max_length: int,
) -> Tensor:
    if task in {"c", "cwh"}:
        return _restrict_reliable_label_or_size(
            sampling_idx=step + 1,
            condition=condition,
            logits=logits,
            pad_id=pad_id,
            eos_id=eos_id,
            max_length=max_length,
        )
    if task in {"refinement", "relation"}:
        return _restrict_only_category(
            sampling_idx=step + 1,
            condition=condition,
            logits=logits,
            pad_id=pad_id,
            eos_id=eos_id,
            max_length=max_length,
        )
    return logits


class RalfForConditionalLayoutGeneration(PreTrainedModel):
    """Standalone `PreTrainedModel` for RALF autoregressive decoding."""

    config_class = RalfConfig
    base_model_prefix = "ralf"
    main_input_name = "input_ids"
    _tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: RalfConfig) -> None:
        """Initialize a local module tree matching original RALF checkpoint keys."""
        super().__init__(config)
        self.tokenizer = RalfTokenizerView(config)
        self.dataset_name = config.dataset_name
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        self.use_reference_image = config.use_reference_image
        self.layout_backbone = config.layout_backbone
        self.top_k = config.top_k
        self.weight_init = True
        self.retrieval_backbone = config.retrieval_backbone
        self.random_retrieval = False
        self.saliency_k = str(config.saliency_k)
        self.num_layers = config.encoder_layers
        self.nhead = config.num_attention_heads
        self.dropout = config.dropout
        self.encoder = ResnetFeatureExtractor(
            backbone="resnet50", d_model=config.d_model, head="transformer"
        )
        self.pos_emb_2d = PositionEmbeddingSine(config.d_model, normalize=True)
        self.dim_feedforward = 4 * config.d_model
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_attention_heads,
                batch_first=True,
                dropout=config.dropout,
                norm_first=True,
                dim_feedforward=self.dim_feedforward,
            ),
            num_layers=config.encoder_layers,
        )
        self.decoder = BaseDecoder(
            d_label=self.tokenizer.N_total,
            d_model=config.decoder_d_model,
            num_layers=config.decoder_layers,
            nhead=config.num_attention_heads,
            pos_emb="layout",
            dim_feedforward=self.dim_feedforward,
        )
        self.loss_fn_ce = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=self.tokenizer.name_to_id("pad")
        )
        self.layout_encoer = FIDNetFeatureExtractor(
            num_label=self.tokenizer.N_label,
            d_model=256,
            nhead=4,
            num_layers=4,
            max_bbox=config.max_seq_length,
        )
        self.layout_encoer.enc_transformer.token.requires_grad = False
        for parameter in self.layout_encoer.parameters():
            parameter.requires_grad = False
        self.pos_emb_1d = PositionalEncoding1d(
            d_model=config.d_model,
            max_len=5000 if not config.use_reference_image else 10000,
        )
        self.layout_adapter = FeedForward(
            dim=256, hidden_dim=4 * config.d_model, output_dim=config.d_model
        )
        self.head = FeedForward(dim=config.d_model, hidden_dim=4 * config.d_model)
        self.auxilary_task = self._canonical_to_task_name(config.task)
        self.use_multitask = config.use_multitask
        self.global_task_embedding = config.global_task_embedding
        self.preprocessor = RalfTaskPreprocessor(
            tokenizer=self.tokenizer,
            task=self.auxilary_task,
            global_task_embedding=config.global_task_embedding,
        )
        self.user_const_encoder = UserConstraintTransformerEncoder(
            d_model=config.d_model,
            nhead=config.num_attention_heads,
            num_layers=config.encoder_layers,
            d_label=self.preprocessor.N_total,
            dim_feedforward=self.dim_feedforward,
        )
        self.use_flag_embedding = config.use_flag_embedding
        if self.use_flag_embedding:
            self.task_emb = nn.Embedding(2, 1)
            nn.init.normal_(self.task_emb.weight, mean=0.0, std=0.02)
            self.register_buffer("flag_img", torch.zeros(1).long())
            self.register_buffer("flag_user_const", torch.ones(1).long())
        self.attn = Attention(
            config.d_model, config.d_model, heads=8, dim_head=64, dropout=0.0
        )
        self.all_tied_weights_keys = dict(self._tied_weights_keys)

    @staticmethod
    def _canonical_to_task_name(task: RalfConfigTaskName | str) -> RalfTaskName:
        if task not in TASK_BY_CONDITION:
            raise ValueError(f"Unsupported RALF task or condition: {task}")
        return TASK_BY_CONDITION[cast(RalfConfigTaskName, task)]

    def _default_retrieved(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Tensor]:
        shape = (batch_size, self.config.top_k, self.config.max_seq_length)
        image = torch.zeros(
            batch_size, self.config.top_k, 4, 64, 64, device=device, dtype=dtype
        )
        return {
            "image": image,
            "center_x": torch.zeros(shape, device=device, dtype=dtype),
            "center_y": torch.zeros(shape, device=device, dtype=dtype),
            "width": torch.zeros(shape, device=device, dtype=dtype),
            "height": torch.zeros(shape, device=device, dtype=dtype),
            "label": torch.zeros(shape, device=device, dtype=torch.long),
            "mask": torch.zeros(shape, device=device, dtype=torch.bool),
        }

    def _encode_into_memory(
        self, inputs: Mapping[str, Tensor | Mapping[str, Tensor]]
    ) -> dict[str, Tensor]:
        image = cast(Tensor, inputs["image"])
        retrieved = cast(Mapping[str, Tensor], inputs["retrieved"])
        input_img_feature = self.encoder(image)
        input_img_feature = self.pos_emb_2d(input_img_feature)
        image_memory = self.transformer_encoder(input_img_feature)
        ref_layouts = _extract_retrieved_features(
            retrieved_samples=retrieved,
            top_k=self.top_k,
            layout_encoder=self.layout_encoer,
            layout_adapter=self.layout_adapter,
            pos_emb_1d=self.pos_emb_1d,
        )
        memory_ca = self.attn(image_memory, ref_layouts)
        img_retrieved_layout_memory = self.head(
            torch.cat([image_memory, memory_ca, ref_layouts], dim=1)
        )
        if self.global_task_embedding:
            task_token = self.preprocessor.get_token(
                self.preprocessor.TASK, img_retrieved_layout_memory.size(0)
            ).type_as(cast(Tensor, inputs["seq_layout_const"]))
        else:
            task_token = None
        user_const_feature = self.user_const_encoder(
            src=cast(Tensor, inputs["seq_layout_const"]),
            src_key_padding_mask=cast(Tensor, inputs["seq_layout_const_pad_mask"]),
            task_token=task_token,
        )
        if self.use_flag_embedding:
            img_retrieved_layout_memory = img_retrieved_layout_memory + self.task_emb(
                self.flag_img
            )
            user_const_feature = user_const_feature + self.task_emb(
                self.flag_user_const
            )
        return {
            "memory": torch.cat(
                [img_retrieved_layout_memory, user_const_feature], dim=1
            )
        }

    def _prepare_conditional_inputs(
        self,
        *,
        pixel_values: Tensor | None,
        saliency: Tensor | None,
        retrieved: RalfRetrievedBatch | None,
        batch_size: int,
        condition_type: RalfConfigTaskName | None = None,
        constraint_input_ids: Tensor | None = None,
        constraint_mask: Tensor | None = None,
        constraint_element_mask: Tensor | None = None,
        relationship_table: Mapping[str, list[object]] | None = None,
        sample_ids: object = None,
    ) -> dict[str, Tensor | Mapping[str, Tensor]]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        if pixel_values is None:
            pixel_values = torch.zeros(
                batch_size, 3, 64, 64, device=device, dtype=dtype
            )
        if saliency is None:
            saliency = torch.zeros(
                pixel_values.size(0),
                1,
                pixel_values.size(2),
                pixel_values.size(3),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
        if pixel_values.size(-1) < 64 or pixel_values.size(-2) < 64:
            pixel_values = F.interpolate(
                pixel_values, size=(64, 64), mode="bilinear", align_corners=False
            )
            saliency = F.interpolate(
                saliency, size=(64, 64), mode="bilinear", align_corners=False
            )
        if pixel_values.size(0) == 1 and batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1)
            saliency = saliency.expand(batch_size, -1, -1, -1)
        image = torch.cat([pixel_values, saliency], dim=1).to(
            device=device, dtype=dtype
        )
        retrieved_dict = (
            self._default_retrieved(image.size(0), device, dtype)
            if retrieved is None
            else {
                key: value.to(device=device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device=device)
                for key, value in retrieved_batch_to_model_inputs(retrieved).items()
            }
        )
        if retrieved_dict["image"].size(2) == 3:
            retrieved_dict["image"] = torch.cat(
                [retrieved_dict["image"], retrieved_dict["saliency"]], dim=2
            )
        task = self._canonical_to_task_name(condition_type or self.auxilary_task)
        preprocessor = (
            self.preprocessor
            if task == self.auxilary_task and relationship_table is None
            else RalfTaskPreprocessor(
                tokenizer=self.tokenizer,
                task=task,
                global_task_embedding=self.global_task_embedding,
                relationship_table=relationship_table if task == "relation" else None,
                relation_size=self.config.relation_size,
            )
        )
        cond = RalfConditionalInputs(
            image=image,
            retrieved=retrieved_dict,
            seq=constraint_input_ids,
            mask=constraint_mask,
            element_mask=constraint_element_mask,
            task=task,
            id=sample_ids,
        )
        seq_constraints = preprocessor(cond)
        return {
            "image": image,
            "retrieved": retrieved_dict,
            "seq_layout_const": seq_constraints["seq"],
            "seq_layout_const_pad_mask": seq_constraints["pad_mask"],
        }

    def _prepare_unconditional_inputs(
        self,
        *,
        pixel_values: Tensor | None,
        saliency: Tensor | None,
        retrieved: RalfRetrievedBatch | None,
        batch_size: int,
    ) -> dict[str, Tensor | Mapping[str, Tensor]]:
        return self._prepare_conditional_inputs(
            pixel_values=pixel_values,
            saliency=saliency,
            retrieved=retrieved,
            batch_size=batch_size,
            condition_type="uncond",
        )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"] | None = None,
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        labels: Int[torch.Tensor, "batch tokens"] | None = None,
        retrieved: RalfRetrievedBatch | None = None,
        condition_type: RalfConfigTaskName | None = None,
        constraint_element_mask: Bool[torch.Tensor, "batch elements"] | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        """Run teacher-forced token prediction using the local RALF port."""
        relationship_table = cast(
            Mapping[str, list[object]] | None, kwargs.get("relationship_table")
        )
        sample_ids = kwargs.get("sample_ids")
        if input_ids is None:
            raise ValueError("input_ids is required")
        encoder_inputs = self._prepare_conditional_inputs(
            pixel_values=pixel_values,
            saliency=saliency,
            retrieved=retrieved,
            batch_size=input_ids.size(0),
            condition_type=condition_type,
            constraint_input_ids=input_ids,
            constraint_mask=attention_mask,
            constraint_element_mask=constraint_element_mask,
            relationship_table=relationship_table,
            sample_ids=sample_ids,
        )
        encoded_feat = self._encode_into_memory(encoder_inputs)
        logits = self.decoder(
            tgt=input_ids,
            tgt_key_padding_mask=None
            if attention_mask is None
            else ~attention_mask.bool(),
            is_causal=True,
            **encoded_feat,
        )
        loss = None
        if labels is not None:
            targets = labels.clone()
            targets[targets == self.config.pad_token_id] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=cast(torch.FloatTensor | None, loss), logits=logits)

    @torch.no_grad()
    def _generate_sequences(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        pixel_values: Float[torch.Tensor, "batch channels height width"] | None = None,
        saliency: Float[torch.Tensor, "batch 1 height width"] | None = None,
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        *,
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        token_mask: Bool[torch.Tensor, "tokens vocab"] | None = None,
        retrieved: RalfRetrievedBatch | None = None,
        condition_type: RalfConfigTaskName | None = None,
        constraint_input_ids: Int[torch.Tensor, "batch tokens"] | None = None,
        constraint_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        constraint_element_mask: Bool[torch.Tensor, "batch elements"] | None = None,
        relationship_table: Mapping[str, list[object]] | None = None,
        sample_ids: object = None,
    ) -> Int[torch.Tensor, "batch tokens"]:
        """Run the RALF autoregressive token loop used by `RalfPipeline`."""
        _ = attention_mask
        was_training = self.training
        self.eval()
        task = self._canonical_to_task_name(condition_type or self.auxilary_task)
        generated = input_ids[:, :1].clone()
        start_step = 0
        if task == "partial":
            condition_seq = (
                constraint_input_ids if constraint_input_ids is not None else input_ids
            )
            prefix = condition_seq[:, 1 : 1 + len(self.config.var_order)]
            generated = torch.cat([generated, prefix.to(generated.device)], dim=1)
            start_step = len(self.config.var_order)
        max_length = max_length or self.config.max_token_length
        encoder_inputs = self._prepare_conditional_inputs(
            pixel_values=pixel_values,
            saliency=saliency,
            retrieved=retrieved,
            batch_size=input_ids.size(0),
            condition_type=task,
            constraint_input_ids=constraint_input_ids,
            constraint_mask=constraint_mask,
            constraint_element_mask=constraint_element_mask,
            relationship_table=relationship_table,
            sample_ids=sample_ids,
        )
        encoded_feat = self._encode_into_memory(encoder_inputs)
        try:
            for step in range(start_step, max_length):
                logits = self.decoder(
                    tgt=generated,
                    tgt_key_padding_mask=generated.eq(self.config.pad_token_id),
                    is_causal=True,
                    **encoded_feat,
                )
                next_logits = logits[:, step : step + 1]
                next_logits = rearrange(next_logits, "b 1 c -> b c") / temperature
                if token_mask is not None and step < token_mask.size(0):
                    next_logits = next_logits.masked_fill(
                        ~token_mask[step].to(next_logits.device), -math.inf
                    )
                next_logits = _apply_decode_space_restriction(
                    task=task,
                    step=step,
                    condition=constraint_input_ids,
                    logits=next_logits,
                    pad_id=self.config.pad_token_id,
                    eos_id=self.config.eos_token_id,
                    max_length=self.config.max_token_length,
                )
                if top_k is not None and top_k > 0 and top_k < next_logits.size(-1):
                    values = torch.topk(next_logits, top_k).values
                    next_logits = next_logits.masked_fill(
                        next_logits < values[:, [-1]], -math.inf
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probs, num_samples=1, generator=generator
                )
                generated = torch.cat([generated, next_token], dim=1)
        finally:
            if was_training:
                self.train()
        return generated
