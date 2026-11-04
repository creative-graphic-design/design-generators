"""Package-local discriminator used by LayoutGAN++ training."""

from __future__ import annotations

import torch
from torch import nn
from typing import cast
from jaxtyping import Bool, Float, Int


class TransformerWithToken(nn.Module):
    """Transformer encoder with a learned layout-summary token."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
    ) -> None:
        """Initialize the encoder and its learned summary token."""
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer("token_mask", torch.zeros(1, 1, dtype=torch.bool))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.core = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        hidden: Float[torch.Tensor, "elements batch hidden"],
        padding_mask: Bool[torch.Tensor, "batch elements"],
    ) -> Float[torch.Tensor, "elements_plus_token batch hidden"]:
        """Encode layout elements together with the summary token."""
        batch_size = hidden.shape[1]
        token = self.token.expand(-1, batch_size, -1)
        hidden = torch.cat([token, hidden], dim=0)
        token_mask = cast(Bool[torch.Tensor, "1 1"], self.token_mask).expand(
            batch_size, -1
        )
        full_padding_mask = torch.cat([token_mask, padding_mask], dim=1)
        return self.core(hidden, src_key_padding_mask=full_padding_mask)


class LayoutGANPPDiscriminator(nn.Module):
    """Discriminator matching the original LayoutGAN++ topology."""

    def __init__(
        self,
        *,
        num_labels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_elements: int,
    ) -> None:
        """Initialize the layout discriminator layers."""
        super().__init__()
        self.emb_label = nn.Embedding(num_labels, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)
        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.fc_out_disc = nn.Linear(d_model, 1)
        self.pos_token = nn.Parameter(torch.rand(max_elements, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)
        self.dec_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model // 2,
            ),
            num_layers=num_layers,
        )
        self.fc_out_cls = nn.Linear(d_model, num_labels)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(
        self,
        bbox: Float[torch.Tensor, "batch elements 4"],
        labels: Int[torch.Tensor, "batch elements"],
        padding_mask: Bool[torch.Tensor, "batch elements"],
        reconst: bool = False,
    ) -> (
        Float[torch.Tensor, "batch"]
        | tuple[
            Float[torch.Tensor, "batch"],
            Float[torch.Tensor, "valid_elements labels"],
            Float[torch.Tensor, "valid_elements 4"],
        ]
    ):
        """Return adversarial logits and optional reconstruction predictions."""
        _, elements, _ = bbox.shape
        bbox_hidden = self.fc_bbox(bbox)
        label_hidden = self.emb_label(labels)
        hidden = self.enc_fc_in(torch.cat([bbox_hidden, label_hidden], dim=-1))
        hidden = torch.relu(hidden).permute(1, 0, 2)
        hidden = self.enc_transformer(hidden, padding_mask)
        summary = hidden[0]
        logit_disc = self.fc_out_disc(summary).squeeze(-1)
        if not reconst:
            return logit_disc

        summary = summary.unsqueeze(0).expand(elements, -1, -1)
        position = self.pos_token[:elements].expand(-1, bbox.shape[0], -1)
        hidden = torch.relu(self.dec_fc_in(torch.cat([summary, position], dim=-1)))
        hidden = self.dec_transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = hidden.permute(1, 0, 2)[~padding_mask]
        return (
            logit_disc,
            self.fc_out_cls(hidden),
            torch.sigmoid(self.fc_out_bbox(hidden)),
        )
