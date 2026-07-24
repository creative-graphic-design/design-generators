"""PyTorch modules for layout FID feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_layout_fid import LayoutFIDConfig, LayoutFIDSource, normalize_source


class TransformerWithToken(nn.Module):
    """Transformer encoder with a learned summary token."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        """Initialize the token encoder."""
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        self.token_mask: torch.Tensor
        self.register_buffer("token_mask", torch.zeros(1, 1, dtype=torch.bool))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.core = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x: Float[torch.Tensor, "elements batch channels"],
        src_key_padding_mask: Bool[torch.Tensor, "batch elements"],
    ) -> Float[torch.Tensor, "elements_plus_token batch channels"]:
        """Encode element features with a prepended summary token."""
        batch_size = x.size(1)
        token = self.token.expand(-1, batch_size, -1)
        x = torch.cat([token, x], dim=0)
        token_mask = self.token_mask.expand(batch_size, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)
        return self.core(x, src_key_padding_mask=padding_mask)


@dataclass
class LayoutFIDOutput(ModelOutput):
    """Output returned by ``LayoutFIDModel.forward``.

    Args:
        features: Batch-level layout feature vectors.
        discriminator_logits: Optional discriminator logits.
        class_logits: Optional per-element or valid-element class logits.
        bbox_pred: Optional reconstructed boxes.
        intermediates: Optional diagnostic tensors.
    """

    features: torch.Tensor
    discriminator_logits: torch.Tensor | None = None
    class_logits: torch.Tensor | None = None
    bbox_pred: torch.Tensor | None = None
    intermediates: dict[str, object] | None = None


class LayoutFIDModel(PreTrainedModel):
    """Feature encoder used for layout FID evaluation."""

    config_class = LayoutFIDConfig
    base_model_prefix = "layout_fid"

    def __init__(self, config: LayoutFIDConfig) -> None:
        """Create a layout FID encoder.

        Args:
            config: Explicit layout FID configuration.
        """
        super().__init__(config)
        self.all_tied_weights_keys: dict[str, str] = {}
        self.emb_label = nn.Embedding(config.num_label_embeddings, config.d_model)
        self.fc_bbox = nn.Linear(4, config.d_model)
        self.enc_fc_in = nn.Linear(config.d_model * 2, config.d_model)
        self.enc_transformer = TransformerWithToken(
            d_model=config.d_model,
            dim_feedforward=config.d_model // 2,
            nhead=config.nhead,
            num_layers=config.num_layers,
        )
        self.fc_out_disc = nn.Linear(config.d_model, 1)
        self.pos_token = nn.Parameter(torch.rand(config.max_length, 1, config.d_model))
        self.dec_fc_in = nn.Linear(config.d_model * 2, config.d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model // 2,
        )
        self.dec_transformer = nn.TransformerEncoder(
            dec_layer, num_layers=config.num_layers
        )
        self.fc_out_cls = nn.Linear(config.d_model, config.num_label_embeddings)
        self.fc_out_bbox = nn.Linear(config.d_model, 4)

    def extract_features(
        self,
        *,
        bbox: Float[torch.Tensor, "batch elements 4"],
        labels: Int[torch.Tensor, "batch elements"],
        padding_mask: Bool[torch.Tensor, "batch elements"],
    ) -> Float[torch.Tensor, "batch channels"]:
        """Extract batch-level feature vectors.

        Args:
            bbox: Model-ready boxes.
            labels: Model-ready label ids.
            padding_mask: Boolean mask where ``True`` marks padded elements.

        Returns:
            Feature tensor shaped ``(batch, d_model)``.

        Raises:
            ValueError: If input shapes are inconsistent.

        Examples:
            >>> from layout_fid import LayoutFIDConfig, LayoutFIDModel
            >>> cfg = LayoutFIDConfig(
            ...     dataset_name="publaynet", architecture="layoutnet",
            ...     source="layoutflow", num_public_labels=5,
            ...     num_label_embeddings=6, max_length=2,
            ... )
            >>> model = LayoutFIDModel(cfg)
            >>> out = model.extract_features(
            ...     bbox=torch.zeros(1, 2, 4),
            ...     labels=torch.zeros(1, 2, dtype=torch.long),
            ...     padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            ... )
            >>> tuple(out.shape)
            (1, 256)
        """
        self._validate_inputs(bbox, labels, padding_mask)
        box_features = self.fc_bbox(bbox)
        label_features = self.emb_label(labels)
        hidden = self.enc_fc_in(torch.cat([box_features, label_features], dim=-1))
        hidden = torch.relu(hidden).permute(1, 0, 2)
        encoded = self.enc_transformer(hidden, padding_mask)
        return encoded[0]

    def forward(
        self,
        *,
        bbox: Float[torch.Tensor, "batch elements 4"],
        labels: Int[torch.Tensor, "batch elements"],
        padding_mask: Bool[torch.Tensor, "batch elements"],
        output_reconstruction: bool = False,
        return_dict: bool = True,
    ) -> LayoutFIDOutput | tuple[torch.Tensor, ...]:
        """Run feature extraction and optional reconstruction heads.

        Args:
            bbox: Model-ready boxes.
            labels: Model-ready label ids.
            padding_mask: Boolean mask where ``True`` marks padded elements.
            output_reconstruction: Whether to return class and bbox predictions.
            return_dict: Whether to return ``LayoutFIDOutput``.

        Returns:
            ``LayoutFIDOutput`` or a tuple with the same non-``None`` fields.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        features = self.extract_features(
            bbox=bbox, labels=labels, padding_mask=padding_mask
        )
        discriminator_logits = self.fc_out_disc(features).squeeze(-1)
        class_logits: torch.Tensor | None = None
        bbox_pred: torch.Tensor | None = None
        if output_reconstruction:
            class_logits, bbox_pred = self._decode(features, padding_mask)
        if not return_dict:
            values = (features, discriminator_logits, class_logits, bbox_pred)
            return tuple(value for value in values if value is not None)
        return LayoutFIDOutput(
            features=features,
            discriminator_logits=discriminator_logits,
            class_logits=class_logits,
            bbox_pred=bbox_pred,
        )

    def _decode(
        self,
        features: Float[torch.Tensor, "batch channels"],
        padding_mask: Bool[torch.Tensor, "batch elements"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, elements = padding_mask.shape
        hidden = features.unsqueeze(0).expand(elements, -1, -1)
        positions = self.pos_token[:elements].expand(-1, batch_size, -1)
        hidden = torch.relu(self.dec_fc_in(torch.cat([hidden, positions], dim=-1)))
        hidden = self.dec_transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = hidden.permute(1, 0, 2)
        class_logits = self.fc_out_cls(hidden)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(hidden))
        if normalize_source(self.config.source) is LayoutFIDSource.layoutflow:
            valid = ~padding_mask
            class_logits = class_logits[valid]
            bbox_pred = bbox_pred[valid]
        return class_logits, bbox_pred

    @staticmethod
    def _validate_inputs(
        bbox: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor
    ) -> None:
        if bbox.ndim != 3 or bbox.shape[-1] != 4:
            raise ValueError("bbox must have shape (batch, elements, 4)")
        if labels.shape != bbox.shape[:2]:
            raise ValueError("labels must have shape (batch, elements)")
        if padding_mask.shape != bbox.shape[:2]:
            raise ValueError("padding_mask must have shape (batch, elements)")
