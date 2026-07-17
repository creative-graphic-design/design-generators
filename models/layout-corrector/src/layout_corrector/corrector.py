from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn

from layout_dm.transformer import Block, ElementPositionalEmbedding, TransformerEncoder
from laygen.common.labels import id2label_for_dataset, normalize_dataset_name


@dataclass
class LayoutCorrectorOutput(BaseOutput):
    logits: torch.FloatTensor


class AggregatedCategoricalTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_token_length: int,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        intermediate_size: int,
        dropout: float,
        timestep_type: str | None,
        pos_emb: str,
        num_attributes_per_element: int,
        num_timesteps: int,
    ) -> None:
        super().__init__()
        if max_token_length % num_attributes_per_element:
            raise ValueError(
                "max_token_length must divide by num_attributes_per_element"
            )
        self.num_attributes_per_element = num_attributes_per_element
        self.cat_emb = nn.Embedding(vocab_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.Sequential(
            nn.Linear(num_attributes_per_element * hidden_size, hidden_size),
            nn.ReLU(),
        )
        layer = Block(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            diffusion_step=num_timesteps,
            timestep_type=timestep_type,
        )
        self.backbone = TransformerEncoder(layer, num_hidden_layers)
        self.dec = nn.Sequential(
            nn.Linear(hidden_size, num_attributes_per_element * hidden_size),
            nn.ReLU(),
        )
        self.pos_emb = None
        if pos_emb != "none":
            self.pos_emb = ElementPositionalEmbedding(
                hidden_size,
                max_token_length // num_attributes_per_element,
                n_attr_per_elem=1,
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        timestep: torch.LongTensor | None = None,
        src_key_padding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        batch_size, token_length = input_ids.shape
        step = self.num_attributes_per_element
        hidden = self.drop(self.cat_emb(input_ids))
        hidden = hidden.reshape(
            batch_size, token_length // step, step * hidden.size(-1)
        )
        hidden = self.enc(hidden)
        if self.pos_emb is not None:
            hidden = hidden + self.pos_emb(hidden)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.reshape(
                batch_size, token_length // step, step
            ).any(dim=-1)
        hidden = self.backbone(
            hidden,
            src_key_padding_mask=src_key_padding_mask,
            timestep=timestep,
        )
        hidden = self.dec(hidden)
        hidden = hidden.reshape(batch_size, token_length, -1)
        return self.head(hidden)


class LayoutCorrectorModel(ModelMixin, ConfigMixin):
    config_name = "corrector_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str,
        vocab_size: int,
        id2label: dict[int | str, str] | None = None,
        max_seq_length: int = 25,
        num_attributes_per_element: int = 5,
        hidden_size: int = 464,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 1856,
        dropout: float = 0.0,
        timestep_type: str | None = "adalayernorm",
        num_timesteps: int = 100,
        recon_type: str = "x_t-1",
        target: str = "recon_acc",
        attr_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        use_padding_as_vocab: bool = True,
        pos_emb: str = "none",
        transformer_type: str = "aggregated",
        corrector_steps: int = 1,
        corrector_t_list: tuple[int, ...] = (10, 20, 30),
        corrector_mask_mode: str = "thresh",
        corrector_mask_threshold: float = 0.7,
        corrector_temperature: float = 1.0,
        use_gumbel_noise: bool = True,
        gumbel_temperature: float = 1.0,
        time_adaptive_temperature: bool = False,
    ) -> None:
        super().__init__()
        if recon_type not in {"x_0", "x_t-1"}:
            raise ValueError(f"Unsupported recon_type: {recon_type}")
        if target not in {"mask", "recon_acc"}:
            raise ValueError(f"Unsupported target: {target}")
        if transformer_type != "aggregated":
            raise ValueError("Only transformer_type='aggregated' is supported")
        dataset_name = normalize_dataset_name(dataset_name)
        normalized_id2label = {
            int(k): v
            for k, v in (id2label or id2label_for_dataset(dataset_name)).items()
        }
        self.register_to_config(
            dataset_name=dataset_name,
            id2label=normalized_id2label,
            corrector_t_list=tuple(corrector_t_list),
            attr_loss_weights=tuple(attr_loss_weights),
        )
        self.model = AggregatedCategoricalTransformer(
            vocab_size=vocab_size,
            max_token_length=max_seq_length * num_attributes_per_element,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
            timestep_type=timestep_type,
            pos_emb=pos_emb,
            num_attributes_per_element=num_attributes_per_element,
            num_timesteps=num_timesteps,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        timesteps: torch.LongTensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> LayoutCorrectorOutput:
        src_key_padding_mask = (
            None if self.config.use_padding_as_vocab else padding_mask
        )
        logits = self.model(
            input_ids,
            timestep=timesteps,
            src_key_padding_mask=src_key_padding_mask,
        ).squeeze(-1)
        if not self.config.use_padding_as_vocab and padding_mask is not None:
            logits = logits.masked_fill(padding_mask, 1000.0)
        return LayoutCorrectorOutput(logits=logits)

    def calc_confidence_score(
        self,
        input_ids: torch.LongTensor,
        timesteps: torch.LongTensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        return self(
            input_ids=input_ids,
            timesteps=timesteps,
            padding_mask=padding_mask,
        ).logits
