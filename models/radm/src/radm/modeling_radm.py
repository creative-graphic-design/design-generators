"""Relation-aware proposal denoiser used by RADM inference and training."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.utils import BaseOutput
from jaxtyping import Bool, Float, Int
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import roi_align

from .configuration_radm import RADMConfig
from .scheduling_radm import cosine_beta_schedule


@dataclass
class RADMDenoiserOutput(BaseOutput):
    """Predictions from one relation-aware proposal refinement pass."""

    logits: Float[torch.Tensor, "batch proposals classes"]
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"]
    pred_original_sample: Float[torch.Tensor, "batch proposals 4"]
    pred_noise: Float[torch.Tensor, "batch proposals 4"]
    auxiliary_logits: Float[torch.Tensor, "heads batch proposals classes"] | None = None
    auxiliary_boxes_xyxy: Float[torch.Tensor, "heads batch proposals 4"] | None = None
    auxiliary_boxes_absolute_xyxy: (
        Float[torch.Tensor, "heads batch proposals 4"] | None
    ) = None


class RADMFrozenBatchNorm2d(FrozenBatchNorm2d):
    """Frozen normalization with the effective training-time branches."""

    def forward(
        self, x: Float[torch.Tensor, "batch channels height width"]
    ) -> Float[torch.Tensor, "batch channels height width"]:
        """Normalize with a fused no-gradient or explicit gradient path."""
        weight = cast(torch.Tensor, self.weight)
        bias_value = cast(torch.Tensor, self.bias)
        running_mean = cast(torch.Tensor, self.running_mean)
        running_var = cast(torch.Tensor, self.running_var)
        if x.requires_grad:
            scale = weight * (running_var + self.eps).rsqrt()
            bias = bias_value - running_mean * scale
            return x * scale.reshape(1, -1, 1, 1).to(x.dtype) + bias.reshape(
                1, -1, 1, 1
            ).to(x.dtype)
        return F.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias_value,
            training=False,
            eps=self.eps,
        )


class RADMBackbone(nn.Module):
    """ResNet-FPN feature extractor with the four proposal feature levels."""

    def __init__(self, *, depth: Literal[18, 50], freeze_at: int) -> None:
        """Initialize an ImageNet-free ResNet-FPN backbone."""
        super().__init__()
        if freeze_at not in range(6):
            raise ValueError("backbone_freeze_at must be between 0 and 5")
        self.body = resnet_fpn_backbone(
            backbone_name=f"resnet{depth}",
            weights=None,
            trainable_layers=5 - freeze_at,
            norm_layer=RADMFrozenBatchNorm2d,
        )
        self.strides: tuple[int, int, int, int] = (4, 8, 16, 32)

    def _forward_bottom_up(
        self, images: Float[torch.Tensor, "batch channels height width"]
    ) -> OrderedDict[str, Float[torch.Tensor, "batch channels height width"]]:
        """Run the bottom-up graph with the fixed stem operation sequence."""
        body = self.body.body
        x = body.conv1(images)
        x = body.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outputs: OrderedDict[
            str, Float[torch.Tensor, "batch channels height width"]
        ] = OrderedDict()
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(body, name)(x)
            if name in body.return_layers:
                outputs[body.return_layers[name]] = x
        return outputs

    def forward(
        self, images: Float[torch.Tensor, "batch channels height width"]
    ) -> OrderedDict[str, Float[torch.Tensor, "batch channels height width"]]:
        """Return ``p2`` through ``p5`` feature maps."""
        bottom_up = self._forward_bottom_up(images)
        fpn = self.body.fpn
        names = list(bottom_up)
        values = list(bottom_up.values())
        last_inner = fpn.get_result_from_inner_blocks(values[-1], -1)
        results = [fpn.get_result_from_layer_blocks(last_inner, -1)]
        for index in range(len(values) - 2, -1, -1):
            inner_lateral = fpn.get_result_from_inner_blocks(values[index], index)
            inner_top_down = F.interpolate(last_inner, scale_factor=2.0, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, fpn.get_result_from_layer_blocks(last_inner, index))
        if fpn.extra_blocks is not None:
            results, names = fpn.extra_blocks(results, values, names)
        raw = OrderedDict(zip(names, results, strict=True))
        values = list(raw.values())[:4]
        if len(values) != 4:
            raise RuntimeError("RADM FPN must produce p2, p3, p4, and p5")
        return OrderedDict(
            (name, value)
            for name, value in zip(("p2", "p3", "p4", "p5"), values, strict=True)
        )


class RADMVisualTextualRelationAwareModule(nn.Module):
    """Fuse proposal ROI tokens with masked text features."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        text_feature_dim: int,
        key_dim: int,
        value_dim: int,
        num_proposals: int,
        num_heads: int,
    ) -> None:
        """Initialize the visual-textual fusion projections."""
        super().__init__()
        self.vis_project = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, 1),
            nn.GELU(),
            nn.Dropout(0.0),
        )
        self.image_lang_att = RADMVisualTextualAttention(
            visual_dim=hidden_dim,
            text_dim=text_feature_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_proposals=num_proposals,
            out_dim=value_dim,
            num_heads=num_heads,
        )
        self.project_mm = nn.Sequential(
            nn.Conv1d(value_dim, value_dim, 1, 1),
            nn.GELU(),
            nn.Dropout(0.0),
        )

    def forward(
        self,
        roi_tokens: Float[torch.Tensor, "batch_tokens roi_tokens hidden"],
        text_features: Float[torch.Tensor, "batch text_dim text_tokens"],
        text_mask: Bool[torch.Tensor, "batch text_tokens 1"],
        position_embedding: Float[torch.Tensor, "batch_tokens hidden"],
    ) -> Float[torch.Tensor, "batch_tokens roi_tokens hidden"]:
        """Fuse visual ROI tokens with the masked text sequence."""
        visual = self.vis_project(roi_tokens.permute(0, 2, 1))
        textual = self.image_lang_att(
            roi_tokens, text_features, text_mask, position_embedding
        ).permute(0, 2, 1)
        fused = self.project_mm(torch.mul(visual, textual))
        return fused.permute(0, 2, 1)


class RADMVisualTextualAttention(nn.Module):
    """Multi-head text attention with the proposal-position token path."""

    def __init__(
        self,
        *,
        visual_dim: int,
        text_dim: int,
        key_dim: int,
        value_dim: int,
        num_proposals: int,
        out_dim: int,
        num_heads: int,
    ) -> None:
        """Initialize the attention projections used by the released head."""
        super().__init__()
        self.propose_num = num_proposals
        self.v_in_channels = visual_dim
        self.l_in_channels = text_dim
        self.out_channels = out_dim
        self.key_channels = key_dim
        self.value_channels = value_dim
        self.num_heads = num_heads
        self.linear = nn.Linear(50, 49)
        self.f_key = nn.Sequential(nn.Conv1d(text_dim, key_dim, 1, 1))
        self.f_query = nn.Sequential(
            nn.Conv1d(visual_dim, key_dim, 1, 1),
            nn.InstanceNorm1d(key_dim),
        )
        self.f_value = nn.Sequential(nn.Conv1d(text_dim, value_dim, 1, 1))
        self.W = nn.Sequential(
            nn.Conv1d(value_dim, out_dim, 1, 1),
            nn.InstanceNorm1d(out_dim),
        )

    def forward(
        self,
        roi_tokens: Float[torch.Tensor, "batch_tokens roi_tokens hidden"],
        text_features: Float[torch.Tensor, "batch text_dim text_tokens"],
        text_mask: Bool[torch.Tensor, "batch text_tokens 1"],
        position_embedding: Float[torch.Tensor, "batch_tokens hidden"],
    ) -> Float[torch.Tensor, "batch_tokens roi_tokens hidden"]:
        """Apply the checked multi-head text attention layout."""
        if text_mask is None:
            text_mask = torch.ones(
                text_features.shape[0],
                text_features.shape[-1],
                1,
                dtype=torch.bool,
                device=text_features.device,
            )
        text_features = text_features.repeat(self.propose_num, 1, 1)
        text_mask = text_mask.repeat(self.propose_num, 1, 1)
        batch_tokens, roi_tokens_count = roi_tokens.shape[:2]
        visual = roi_tokens.permute(0, 2, 1)
        text_mask_transposed = text_mask.permute(0, 2, 1).to(visual.dtype)
        query = self.f_query(visual).reshape(batch_tokens, self.key_channels, -1)
        position_embedding = position_embedding.view(
            batch_tokens, self.key_channels
        ).unsqueeze(2)
        query = self.linear(torch.cat((query, position_embedding), dim=2))
        query = query.reshape(batch_tokens, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(text_features) * text_mask_transposed
        value = self.f_value(text_features) * text_mask_transposed
        text_count = value.size(-1)
        query = query.reshape(
            batch_tokens,
            roi_tokens_count,
            self.num_heads,
            self.key_channels // self.num_heads,
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch_tokens,
            self.num_heads,
            self.key_channels // self.num_heads,
            text_count,
        )
        value = value.reshape(
            batch_tokens,
            self.num_heads,
            self.value_channels // self.num_heads,
            text_count,
        )
        mask = text_mask_transposed.unsqueeze(1)
        similarity = torch.matmul(query, key) * (self.key_channels**-0.5)
        similarity = similarity + (1e4 * mask - 1e4)
        attention = F.softmax(similarity, dim=-1)
        output = torch.matmul(attention, value.permute(0, 1, 3, 2))
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_tokens, roi_tokens_count, self.value_channels)
        )
        output = self.W(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output


class RADMGeometryRelationAwareModule(nn.Module):
    """Geometry-weighted aggregation of pooled proposal features."""

    def __init__(self, *, pooled_dim: int, output_dim: int = 64) -> None:
        """Initialize geometry weighting and pooled-feature projections."""
        super().__init__()
        self.linear = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(pooled_dim, output_dim)
        self.out_dim = 64
        self.wave_length = 1000
        self.topo_out_dim = output_dim
        self.drop_rate = 0.0
        self.init_weight()

    def init_weight(self) -> None:
        """Initialize geometry projections with the reference standard deviation."""
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.constant_(self.linear2.bias, 0)

    def build_relative_geo(
        self,
        rois: Float[torch.Tensor, "proposals 4"],
        targets: Float[torch.Tensor, "targets 4"],
    ) -> Float[torch.Tensor, "proposals targets 4"]:
        """Build logarithmic pairwise geometry features."""
        if rois.shape[1] != targets.shape[1]:
            raise ValueError("geometry inputs must have four coordinates")
        rois_repeat = rois[..., None].repeat(1, 1, targets.shape[0])
        target_x, target_y, target_w, target_h = targets.unbind(dim=1)
        floor = torch.tensor(1e-3).to(targets.device)
        target_w = target_w.maximum(floor)
        target_h = target_h.maximum(floor)
        relative_x = (target_x - rois_repeat[:, 0, :]).abs() / target_w
        relative_x = relative_x.maximum(floor)
        relative_y = (target_y - rois_repeat[:, 1, :]).abs() / target_h
        relative_y = relative_y.maximum(floor)
        relative_w = rois_repeat[:, 2, :] / target_w
        relative_w = relative_w.maximum(floor)
        relative_h = rois_repeat[:, 3, :] / target_h
        relative_h = relative_h.maximum(floor)
        relative = torch.stack(
            (relative_x, relative_y, relative_w, relative_h), dim=-1
        ).float()
        return relative.log()

    def extract_position_embedding(
        self,
        relative_geometry: Float[torch.Tensor, "proposals targets 4"],
    ) -> Float[torch.Tensor, "proposals targets embedding"]:
        """Encode pairwise geometry with the fixed sinusoidal basis."""
        feature_range = torch.arange(0, self.out_dim / 8)
        dimension = (
            torch.pow(
                torch.full((1,), self.wave_length),
                (8.0 / self.out_dim) * feature_range,
            )
            .to(relative_geometry.device)
            .reshape(1, 1, 1, -1)
        )
        scaled = (100.0 * relative_geometry).unsqueeze(-1) / dimension
        embedding = torch.stack((scaled.sin(), scaled.cos()), dim=-1)
        return embedding.flatten(2)

    def forward(
        self,
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        pooled_features: Float[torch.Tensor, "batch_proposals channels roi roi"],
    ) -> Float[torch.Tensor, "batch_proposals topology"]:
        """Aggregate pooled ROI features using geometry-aware weights."""
        batch, proposals = boxes_xyxy.shape[:2]
        boxes = boxes_xyxy.reshape(-1, 4)
        batch_ids = torch.arange(batch, device=boxes.device).repeat_interleave(
            proposals
        )
        boxes_with_batch = torch.cat((batch_ids[:, None], boxes), dim=1)
        centers = torch.stack(
            (
                boxes_with_batch[:, 0],
                (boxes_with_batch[:, 1] + boxes_with_batch[:, 3]) / 2,
                (boxes_with_batch[:, 2] + boxes_with_batch[:, 4]) / 2,
                boxes_with_batch[:, 3] - boxes_with_batch[:, 1],
                boxes_with_batch[:, 4] - boxes_with_batch[:, 2],
            ),
            dim=1,
        )
        transformed = self.linear2(
            pooled_features.reshape(pooled_features.shape[0] * proposals, -1)
        )
        topology = transformed.new_zeros((boxes_with_batch.shape[0], self.topo_out_dim))
        for batch_index in range(batch):
            selected = boxes_with_batch[:, 0] == batch_index
            relative = self.build_relative_geo(
                centers[selected, 1:], centers[selected, 1:]
            )
            position = self.extract_position_embedding(relative)
            weights = self.relu(self.linear(position)).squeeze(-1)
            weights = F.softmax(weights, dim=-1)
            weights = F.dropout(weights, p=self.drop_rate, training=self.training)
            topology[selected] = torch.mm(weights, transformed[selected])
        return topology


class RADMDynamicConv(nn.Module):
    """Proposal-conditioned feature interaction block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        dim_dynamic: int,
        num_dynamic: int,
        roi_resolution: int,
    ) -> None:
        """Initialize proposal-conditioned dynamic projections."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.dynamic_layer = nn.Linear(
            hidden_dim, self.num_dynamic * hidden_dim * dim_dynamic
        )
        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_layer = nn.Linear(hidden_dim * roi_resolution**2, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        proposal_features: Float[torch.Tensor, "one batch_proposals hidden"],
        roi_features: Float[torch.Tensor, "roi_tokens batch_proposals hidden"],
    ) -> Float[torch.Tensor, "batch_proposals hidden"]:
        """Apply two proposal-conditioned projections to ROI features."""
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(proposal_features).permute(1, 0, 2)
        first_size = self.hidden_dim * self.dim_dynamic
        first, second = parameters.split(first_size, dim=-1)
        first = first.reshape(-1, self.hidden_dim, self.dim_dynamic)
        second = second.reshape(-1, self.dim_dynamic, self.hidden_dim)
        features = torch.bmm(features, first)
        features = self.norm1(features)
        features = self.activation(features)
        features = torch.bmm(features, second)
        features = self.norm2(features)
        features = self.activation(features)
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        return self.activation(features)


class RADMRelationBlock(nn.Module):
    """One proposal self-attention, visual-textual, and geometric block."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        text_feature_dim: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dim_dynamic: int,
        num_dynamic: int,
        num_proposals: int,
        roi_resolution: int,
        num_classes: int,
        num_cls: int,
        num_reg: int,
        with_vtram: bool,
        with_gram: bool,
    ) -> None:
        """Initialize one proposal refinement block."""
        super().__init__()
        self.with_vtram = with_vtram
        self.with_gram = with_gram
        self.d_model = hidden_dim
        self.text_feature_dim = text_feature_dim
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=0.0
        )
        self.inst_interact = RADMDynamicConv(
            hidden_dim=hidden_dim,
            dim_dynamic=dim_dynamic,
            num_dynamic=num_dynamic,
            roi_resolution=roi_resolution,
        )
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        if with_vtram:
            self.vis_text_att = RADMVisualTextualRelationAwareModule(
                hidden_dim=hidden_dim,
                text_feature_dim=text_feature_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_proposals=num_proposals,
                num_heads=2,
            )
            self.linear4 = nn.Linear(4, hidden_dim)
        if with_gram:
            self.GRAM = RADMGeometryRelationAwareModule(
                pooled_dim=hidden_dim * roi_resolution * roi_resolution,
                output_dim=64,
            )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)
        self.dropout3 = nn.Dropout(0.0)
        self.activation = F.relu
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        fused_dim = hidden_dim * (1 + int(with_vtram)) + (64 if with_gram else 0)
        self.cls_module = nn.ModuleList(
            layer
            for _ in range(num_cls)
            for layer in (
                nn.Linear(fused_dim, fused_dim, bias=False),
                nn.LayerNorm(fused_dim),
                nn.ReLU(inplace=True),
            )
        )
        self.reg_module = nn.ModuleList(
            layer
            for _ in range(num_reg)
            for layer in (
                nn.Linear(fused_dim, fused_dim, bias=False),
                nn.LayerNorm(fused_dim),
                nn.ReLU(inplace=True),
            )
        )
        self.class_logits = nn.Linear(fused_dim, num_classes)
        self.bboxes_delta = nn.Linear(fused_dim, 4)
        self.bbox_weights = (2.0, 2.0, 1.0, 1.0)
        self.scale_clamp = math.log(100000.0 / 16)

    def forward(
        self,
        proposal_features: Float[torch.Tensor, "batch proposals hidden"] | None,
        roi_features: Float[torch.Tensor, "batch proposals channels roi roi"],
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        initial_norm_boxes: Float[torch.Tensor, "batch proposals 4"],
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"] | None,
        time_embedding: Float[torch.Tensor, "batch time_hidden"],
    ) -> tuple[
        Float[torch.Tensor, "batch proposals classes"],
        Float[torch.Tensor, "batch proposals 4"],
        Float[torch.Tensor, "batch proposals hidden"],
    ]:
        """Refine proposals with the fixed relation-aware feature order."""
        batch, proposals = boxes_xyxy.shape[:2]
        textual = roi_features.new_zeros((batch * proposals, self.d_model))
        topology = roi_features.new_zeros((batch * proposals, 64))
        if self.with_gram:
            topology = self.GRAM(boxes_xyxy, roi_features)

        roi_sequence = roi_features.view(batch * proposals, self.d_model, -1).permute(
            0, 2, 1
        )
        if self.with_vtram:
            positions = self.linear4(initial_norm_boxes).reshape(batch * proposals, -1)
            textual = self.vis_text_att(
                roi_sequence,
                text_features.reshape(batch, self.text_feature_dim, -1),
                text_mask,
                positions,
            )
            textual = textual.reshape(batch * proposals, self.d_model, -1).mean(-1)

        if proposal_features is None:
            proposal_features = roi_features.reshape(
                batch, proposals, self.d_model, -1
            ).mean(-1)
        elif proposal_features.ndim == 3 and proposal_features.shape[0] == 1:
            proposal_features = proposal_features.reshape(
                batch, proposals, self.d_model
            )
        proposal_sequence = proposal_features.permute(1, 0, 2)
        attended = self.self_attn(
            proposal_sequence, proposal_sequence, value=proposal_sequence
        )[0]
        proposal_sequence = self.norm1(proposal_sequence + self.dropout1(attended))
        proposal_sequence = proposal_sequence.reshape(proposals, batch, self.d_model)
        proposal_sequence = proposal_sequence.permute(1, 0, 2).reshape(
            1, batch * proposals, self.d_model
        )
        interacted = self.inst_interact(
            proposal_sequence, roi_sequence.permute(1, 0, 2)
        )
        proposal_sequence = proposal_sequence + self.dropout2(interacted)
        object_features = self.norm2(proposal_sequence)
        feed_forward = self.linear2(
            self.dropout(self.activation(self.linear1(object_features)))
        )
        object_features = self.norm3(object_features + self.dropout3(feed_forward))
        scale, shift = self.block_time_mlp(time_embedding).chunk(2, dim=1)
        scale = torch.repeat_interleave(scale, proposals, dim=0)
        shift = torch.repeat_interleave(shift, proposals, dim=0)
        fused = object_features.transpose(0, 1).reshape(batch * proposals, -1)
        fused = fused * (scale + 1) + shift
        if self.with_vtram:
            fused = torch.cat((fused, textual), dim=1)
        if self.with_gram:
            fused = torch.cat((topology, fused), dim=1)
        cls_feature = fused.clone()
        for layer in self.cls_module:
            cls_feature = layer(cls_feature)
        reg_feature = fused.clone()
        for layer in self.reg_module:
            reg_feature = layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        deltas = self.bboxes_delta(reg_feature)
        predicted_boxes = _apply_box_deltas(
            boxes_xyxy.reshape(-1, 4),
            deltas,
            bbox_weights=self.bbox_weights,
            scale_clamp=self.scale_clamp,
        ).reshape(batch, proposals, 4)
        next_features = object_features
        return (
            class_logits.reshape(batch, proposals, -1),
            predicted_boxes,
            next_features,
        )


class RADMProposalHead(nn.Module):
    """Repeated proposal refinement head with level-assigned ROI pooling."""

    def __init__(self, config: RADMConfig) -> None:
        """Initialize the repeated proposal refinement head."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RADMRelationBlock(
                    hidden_dim=config.hidden_dim,
                    text_feature_dim=config.text_feature_dim,
                    num_attention_heads=config.num_attention_heads,
                    dim_feedforward=config.dim_feedforward,
                    dim_dynamic=config.dim_dynamic,
                    num_dynamic=config.num_dynamic,
                    num_proposals=config.num_proposals,
                    roi_resolution=config.roi_resolution,
                    num_classes=config.num_classes,
                    num_cls=config.num_cls,
                    num_reg=config.num_reg,
                    with_vtram=config.with_vtram,
                    with_gram=config.with_gram,
                )
                for _ in range(config.num_heads)
            ]
        )
        self.time_mlp = nn.Sequential(
            _SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 4),
        )
        self.roi_resolution = config.roi_resolution
        self.roi_sampling_ratio = config.roi_sampling_ratio
        self.hidden_dim = config.hidden_dim
        self.deep_supervision = config.deep_supervision
        self.feature_projection = nn.ModuleDict(
            {
                name: (
                    nn.Identity()
                    if config.hidden_dim == 256
                    else nn.Conv2d(256, config.hidden_dim, kernel_size=1)
                )
                for name in ("p2", "p3", "p4", "p5")
            }
        )

    def forward(
        self,
        features: Mapping[str, Float[torch.Tensor, "batch channels height width"]],
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"] | None,
        timesteps: Int[torch.Tensor, "batch"],
        image_scales: Float[torch.Tensor, "batch 4"],
        absolute_boxes_xyxy: Float[torch.Tensor, "batch proposals 4"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "heads batch proposals classes"],
        Float[torch.Tensor, "heads batch proposals 4"],
    ]:
        """Run all proposal refinement blocks."""
        time_embedding = self.time_mlp(timesteps.float())
        class_outputs: list[Float[torch.Tensor, "batch proposals classes"]] = []
        box_outputs: list[Float[torch.Tensor, "batch proposals 4"]] = []
        initial_norm_boxes = boxes_xyxy.to(dtype=features["p2"].dtype)
        proposal_features: Float[torch.Tensor, "batch proposals hidden"] | None = None
        image_scale = image_scales[:, None, :].to(device=boxes_xyxy.device)
        current_boxes = (
            boxes_xyxy * image_scale
            if absolute_boxes_xyxy is None
            else absolute_boxes_xyxy.to(device=boxes_xyxy.device)
        )
        for block in self.blocks:
            roi_features = self._roi_features(features, current_boxes)
            logits, predicted_boxes, proposal_features = block(
                proposal_features,
                roi_features,
                current_boxes,
                initial_norm_boxes,
                text_features,
                text_mask,
                time_embedding,
            )
            class_outputs.append(logits)
            # Keep raw absolute boxes attached for the training loss;
            # only the boxes fed into the next repeated head are detached, as
            # in the corresponding repeated-head update.
            box_outputs.append(predicted_boxes)
            current_boxes = predicted_boxes.detach()
        if not self.deep_supervision:
            class_outputs = class_outputs[-1:]
            box_outputs = box_outputs[-1:]
        return torch.stack(class_outputs), torch.stack(box_outputs)

    def _roi_features(
        self,
        features: Mapping[str, Float[torch.Tensor, "batch channels height width"]],
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    ) -> Float[torch.Tensor, "batch proposals roi hidden"]:
        absolute_boxes = boxes_xyxy
        batch_indices = torch.arange(
            boxes_xyxy.shape[0], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype
        )
        rois = torch.cat(
            [
                batch_indices[:, None, None].expand(-1, boxes_xyxy.shape[1], 1),
                absolute_boxes,
            ],
            dim=-1,
        ).reshape(-1, 5)
        pooled = features["p2"].new_zeros(
            boxes_xyxy.shape[0] * boxes_xyxy.shape[1],
            self.hidden_dim,
            self.roi_resolution,
            self.roi_resolution,
        )
        levels = self._assign_pooler_levels(absolute_boxes)
        for level, (name, stride) in enumerate(
            zip(
                ("p2", "p3", "p4", "p5"),
                (4.0, 8.0, 16.0, 32.0),
                strict=True,
            )
        ):
            selected = torch.nonzero(
                levels.flatten() == level, as_tuple=False
            ).flatten()
            feature = self.feature_projection[name](features[name])
            pooled[selected] = roi_align(
                feature,
                rois[selected].to(dtype=feature.dtype),
                output_size=self.roi_resolution,
                spatial_scale=1.0 / stride,
                sampling_ratio=self.roi_sampling_ratio,
                aligned=True,
            )
        return pooled.reshape(
            boxes_xyxy.shape[0],
            boxes_xyxy.shape[1],
            self.hidden_dim,
            self.roi_resolution,
            self.roi_resolution,
        )

    @staticmethod
    def _assign_pooler_levels(
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    ) -> Int[torch.Tensor, "batch proposals"]:
        """Assign each ROI to one FPN level from its square-root area."""
        width = (boxes_xyxy[..., 2] - boxes_xyxy[..., 0]).clamp_min(0)
        height = (boxes_xyxy[..., 3] - boxes_xyxy[..., 1]).clamp_min(0)
        box_size = torch.sqrt(width * height)
        level = torch.floor(4 + torch.log2(box_size / 224 + 1e-8))
        return level.clamp(2, 5).to(torch.long) - 2


class _SinusoidalPositionEmbedding(nn.Module):
    """Deterministic scalar-timestep embedding."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(
        self, timesteps: Float[torch.Tensor, "batch"]
    ) -> Float[torch.Tensor, "batch dimension"]:
        half = self.dimension // 2
        exponent = torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(
            half - 1, 1
        )
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -exponent)
        angles = timesteps[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class RADMDenoiser(ModelMixin, ConfigMixin):
    """Genuine RADM proposal denoiser shared by inference and training."""

    config_name: str = "denoiser_config.json"

    def __init__(
        self,
        *,
        config: RADMConfig | None = None,
        num_classes: int | None = None,
        num_proposals: int | None = None,
        hidden_dim: int | None = None,
        text_feature_dim: int | None = None,
        max_text_num: int | None = None,
        num_heads: int | None = None,
        num_attention_heads: int | None = None,
        dim_feedforward: int | None = None,
        num_dynamic: int | None = None,
        dim_dynamic: int | None = None,
        num_cls: int | None = None,
        num_reg: int | None = None,
        roi_resolution: int | None = None,
        roi_sampling_ratio: int | None = None,
        with_vtram: bool | None = None,
        with_gram: bool | None = None,
        deep_supervision: bool | None = None,
        backbone_depth: int | None = None,
        backbone_freeze_at: int | None = None,
        num_train_timesteps: int | None = None,
        snr_scale: float | None = None,
    ) -> None:
        """Initialize from an explicit package config or serialized model values."""
        super().__init__()
        if config is None:
            values = {
                "num_classes": num_classes,
                "num_proposals": num_proposals,
                "hidden_dim": hidden_dim,
                "text_feature_dim": text_feature_dim,
                "max_text_num": max_text_num,
                "num_heads": num_heads,
                "num_attention_heads": num_attention_heads,
                "dim_feedforward": dim_feedforward,
                "num_dynamic": num_dynamic,
                "dim_dynamic": dim_dynamic,
                "num_cls": num_cls,
                "num_reg": num_reg,
                "roi_resolution": roi_resolution,
                "roi_sampling_ratio": roi_sampling_ratio,
                "with_vtram": with_vtram,
                "with_gram": with_gram,
                "deep_supervision": deep_supervision,
                "backbone_depth": backbone_depth,
                "backbone_freeze_at": backbone_freeze_at,
                "num_train_timesteps": num_train_timesteps,
                "snr_scale": snr_scale,
            }
            if any(value is None for value in values.values()):
                raise TypeError("RADMDenoiser requires an explicit RADMConfig")
            config = RADMConfig(**values)  # type: ignore[arg-type]
        self.radm_config = config
        self.register_to_config(
            **{
                "num_classes": config.num_classes,
                "num_proposals": config.num_proposals,
                "hidden_dim": config.hidden_dim,
                "text_feature_dim": config.text_feature_dim,
                "max_text_num": config.max_text_num,
                "num_heads": config.num_heads,
                "num_attention_heads": config.num_attention_heads,
                "dim_feedforward": config.dim_feedforward,
                "num_dynamic": config.num_dynamic,
                "dim_dynamic": config.dim_dynamic,
                "num_cls": config.num_cls,
                "num_reg": config.num_reg,
                "roi_resolution": config.roi_resolution,
                "roi_sampling_ratio": config.roi_sampling_ratio,
                "with_vtram": config.with_vtram,
                "with_gram": config.with_gram,
                "deep_supervision": config.deep_supervision,
                "backbone_depth": config.backbone_depth,
                "backbone_freeze_at": config.backbone_freeze_at,
                "num_train_timesteps": config.num_train_timesteps,
                "snr_scale": config.snr_scale,
            }
        )
        if config.backbone_depth not in (18, 50):
            raise ValueError("backbone_depth must be 18 or 50")
        self.num_classes = config.num_classes
        self.num_proposals = config.num_proposals
        self.hidden_dim = config.hidden_dim
        self.backbone = RADMBackbone(
            depth=cast(Literal[18, 50], config.backbone_depth),
            freeze_at=config.backbone_freeze_at,
        )
        self.head = RADMProposalHead(config)
        self._register_diffusion_buffers(config)

    def _register_diffusion_buffers(self, config: RADMConfig) -> None:
        betas = cosine_beta_schedule(config.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1.0).sqrt()
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: Float[torch.Tensor, "batch proposals 4"],
        timesteps: Int[torch.Tensor, "batch"],
        noise: Float[torch.Tensor, "batch proposals 4"] | None = None,
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Sample the configured cosine forward diffusion process."""
        sampled_noise = torch.randn_like(x_start) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * sampled_noise
        )

    def predict_noise_from_start(
        self,
        x_t: Float[torch.Tensor, "batch proposals 4"],
        timesteps: Int[torch.Tensor, "batch"],
        x_start: Float[torch.Tensor, "batch proposals 4"],
    ) -> Float[torch.Tensor, "batch proposals 4"]:
        """Recover epsilon from a predicted clean diffusion sample."""
        return (
            _extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - x_start
        ) / _extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)

    def prepare_diffusion_concat(
        self,
        boxes_cxcywh: Float[torch.Tensor, "elements 4"],
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[
        Float[torch.Tensor, "proposals 4"],
        Float[torch.Tensor, "proposals 4"],
        Int[torch.Tensor, "1"],
    ]:
        """Prepare one image's padded proposals using the configured sampler."""
        device = boxes_cxcywh.device
        timestep = torch.randint(
            0,
            self.radm_config.num_train_timesteps,
            (1,),
            device=device,
            generator=generator,
        ).long()
        noise = torch.randn(self.num_proposals, 4, device=device, generator=generator)
        count = boxes_cxcywh.shape[0]
        if count == 0:
            boxes_cxcywh = boxes_cxcywh.new_tensor([[0.5, 0.5, 1.0, 1.0]])
            count = 1
        if count < self.num_proposals:
            placeholders = (
                torch.randn(
                    self.num_proposals - count, 4, device=device, generator=generator
                )
                / 6.0
                + 0.5
            )
            placeholders[:, 2:] = placeholders[:, 2:].clamp_min(1e-4)
            clean = torch.cat((boxes_cxcywh, placeholders), dim=0)
        elif count > self.num_proposals:
            clean = boxes_cxcywh[: self.num_proposals]
        else:
            clean = boxes_cxcywh
        clean = (clean * 2.0 - 1.0) * self.radm_config.snr_scale
        diffused = self.q_sample(clean, timestep, noise=noise)
        diffused = diffused.clamp(
            -self.radm_config.snr_scale, self.radm_config.snr_scale
        )
        diffused = (diffused / self.radm_config.snr_scale + 1.0) / 2.0
        return _cxcywh_to_xyxy(diffused), noise, timestep

    def forward(
        self,
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        timesteps: Int[torch.Tensor, "batch"]
        | Int[torch.Tensor, ""]
        | Float[torch.Tensor, "batch"]
        | Float[torch.Tensor, ""],
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"] | None = None,
        images: Float[torch.Tensor, "batch channels height width"] | None = None,
        image_scales: Float[torch.Tensor, "batch 4"] | None = None,
        absolute_boxes_xyxy: Float[torch.Tensor, "batch proposals 4"] | None = None,
    ) -> RADMDenoiserOutput:
        """Predict proposal classes and denoised boxes."""
        if text_features.ndim != 3:
            raise ValueError("text_features must have shape (batch, text, dim)")
        if boxes_xyxy.shape[1] != self.num_proposals:
            raise ValueError(f"expected {self.num_proposals} proposals")
        if timesteps.ndim == 0:
            timesteps = timesteps.repeat(boxes_xyxy.shape[0])
        timestep_batch = timesteps.to(
            device=boxes_xyxy.device, dtype=torch.long
        ).reshape(-1)
        if images is None:
            features = self._fallback_features(boxes_xyxy)
            default_scale = boxes_xyxy.new_ones((boxes_xyxy.shape[0], 4))
        else:
            features = self.backbone(images)
            default_scale = boxes_xyxy.new_tensor(
                (images.shape[-1], images.shape[-2], images.shape[-1], images.shape[-2])
            ).expand(boxes_xyxy.shape[0], -1)
        resolved_scales = default_scale if image_scales is None else image_scales
        auxiliary_logits, auxiliary_boxes_absolute = self.head(
            features,
            boxes_xyxy,
            text_features,
            text_mask,
            timestep_batch,
            resolved_scales,
            absolute_boxes_xyxy,
        )
        image_scale = resolved_scales[:, None, :].to(device=boxes_xyxy.device)
        auxiliary_boxes = auxiliary_boxes_absolute / image_scale
        logits = auxiliary_logits[-1]
        pred_original = auxiliary_boxes[-1]
        return RADMDenoiserOutput(
            logits=logits,
            boxes_xyxy=pred_original,
            pred_original_sample=pred_original,
            pred_noise=boxes_xyxy - pred_original,
            auxiliary_logits=auxiliary_logits,
            auxiliary_boxes_xyxy=auxiliary_boxes,
            auxiliary_boxes_absolute_xyxy=auxiliary_boxes_absolute,
        )

    def _fallback_features(
        self, boxes_xyxy: Float[torch.Tensor, "batch proposals 4"]
    ) -> dict[str, Float[torch.Tensor, "batch channels height width"]]:
        """Create a deterministic proposal-only feature pyramid for serialization tests."""
        zero = boxes_xyxy.new_zeros(boxes_xyxy.shape[0], 256, 1, 1)
        return {name: zero for name in ("p2", "p3", "p4", "p5")}


def _apply_box_deltas(
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    deltas: Float[torch.Tensor, "batch proposals 4"],
    *,
    bbox_weights: tuple[float, float, float, float] = (2.0, 2.0, 1.0, 1.0),
    scale_clamp: float = math.log(100000.0 / 16),
) -> Float[torch.Tensor, "batch proposals 4"]:
    """Apply center/size deltas to normalized ``xyxy`` boxes."""
    boxes_xyxy = boxes_xyxy.to(deltas.dtype)
    widths = boxes_xyxy[..., 2] - boxes_xyxy[..., 0]
    heights = boxes_xyxy[..., 3] - boxes_xyxy[..., 1]
    center_x = boxes_xyxy[..., 0] + 0.5 * widths
    center_y = boxes_xyxy[..., 1] + 0.5 * heights
    wx, wy, ww, wh = bbox_weights
    dx, dy, dw, dh = deltas.unbind(dim=-1)
    new_center_x = center_x + (dx / wx) * widths
    new_center_y = center_y + (dy / wy) * heights
    new_width = widths * (dw / ww).clamp(max=scale_clamp).exp()
    new_height = heights * (dh / wh).clamp(max=scale_clamp).exp()
    return torch.stack(
        (
            new_center_x - 0.5 * new_width,
            new_center_y - 0.5 * new_height,
            new_center_x + 0.5 * new_width,
            new_center_y + 0.5 * new_height,
        ),
        dim=-1,
    )


def _extract(
    values: Float[torch.Tensor, "timesteps"],
    timesteps: Int[torch.Tensor, "batch"],
    shape: torch.Size,
) -> Float[torch.Tensor, "batch 1 1"]:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def _cxcywh_to_xyxy(
    boxes: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )
