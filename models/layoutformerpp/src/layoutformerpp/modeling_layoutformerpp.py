"""PyTorch model wrapper for LayoutFormer++."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .configuration_layoutformerpp import LayoutFormerPPConfig


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Create the causal decoder mask used by the checkpoint model."""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, -math.inf).masked_fill(mask == 1, 0.0)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask logits outside the top-k set."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    values = torch.topk(logits, k).values
    out = logits.clone()
    out[out < values[:, [-1]]] = -math.inf
    return out


class PositionalEncoding(nn.Module):
    """Learned positional embeddings matching the original implementation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        """Initialize learned position tokens."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_token = nn.Parameter(torch.rand(max_len, 1, d_model))

    def forward(
        self, x: Float[torch.Tensor, "seq batch channels"]
    ) -> Float[torch.Tensor, "seq batch channels"]:
        """Add learned position tokens to `(seq, batch, hidden)` input."""
        return self.dropout(x + self.pos_token[: x.size(0)])


class LayoutFormerPPForConditionalGeneration(PreTrainedModel):
    """Transformers `PreTrainedModel` with checkpoint-compatible module names."""

    config_class = LayoutFormerPPConfig
    base_model_prefix = "layoutformerpp"
    main_input_name = "input_ids"
    _tied_weights_keys = {
        "dec_embedding.weight": "enc_embedding.weight",
        "out.weight": "dec_embedding.weight",
    }

    def __init__(self, config: LayoutFormerPPConfig) -> None:
        """Initialize checkpoint-compatible encoder/decoder modules."""
        super().__init__(config)
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.bos_token_id = int(config.bos_token_id)
        self.pad_token_id = int(config.pad_token_id)
        self.eos_token_id = int(config.eos_token_id)

        self.enc_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.enc_pos_embedding = PositionalEncoding(
            config.d_model, config.dropout, max_len=config.max_position_embeddings
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_attention_heads,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers
        )

        self.dec_embedding = (
            self.enc_embedding
            if config.share_embedding
            else nn.Embedding(config.vocab_size, config.d_model)
        )
        self.dec_pos_embedding = PositionalEncoding(
            config.d_model, config.dropout, max_len=config.max_position_embeddings
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers
        )
        self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.out.weight = self.dec_embedding.weight

        self.task_embedding = None
        if config.add_task_embedding:
            self.task_embedding = nn.Embedding(6, config.d_model)

        self.task_prompt_embed = None
        if config.add_task_prompt_token_in_model:
            self.num_task_prompt_token = config.num_task_prompt_token
            self.task_prompt_embed = nn.Parameter(
                torch.empty(6, config.num_task_prompt_token, config.d_model)
            )
            nn.init.normal_(self.task_prompt_embed)
        self.tie_weights()
        self.all_tied_weights_keys = dict(self._tied_weights_keys)

    def encode(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        padding_mask: Bool[torch.Tensor, "batch tokens"],
        task_ids: Int[torch.Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "seq batch channels"],
        Bool[torch.Tensor, "batch seq"],
    ]:
        """Encode input token ids with optional task prompt embeddings."""
        if self.task_prompt_embed is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids are required when task prompt embeddings are enabled"
                )
            x = self.enc_embedding(input_ids)
            prompts = self.task_prompt_embed[task_ids]
            x = torch.cat([prompts, x], dim=1).permute(1, 0, 2)
            bsz = input_ids.size(0)
            prompt_mask = padding_mask.new_zeros(
                (bsz, self.num_task_prompt_token)
            ).bool()
            enc_padding_mask = torch.cat([prompt_mask, padding_mask], dim=1)
        else:
            x = self.enc_embedding(input_ids).permute(1, 0, 2)
            enc_padding_mask = padding_mask
        enc_hs = self.encoder(
            self.enc_pos_embedding(x), src_key_padding_mask=enc_padding_mask
        )
        if self.task_embedding is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids are required when task embeddings are enabled"
                )
            enc_hs = enc_hs + self.task_embedding(task_ids).unsqueeze(0)
        return enc_hs, enc_padding_mask

    def prepare_decoder_input_ids_from_labels(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
        """Shift labels right and prepend BOS."""
        bos = labels.new_full((labels.size(0), 1), self.bos_token_id)
        return torch.cat([bos, labels[:, :-1]], dim=1)

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        labels: Int[torch.Tensor, "batch tokens"] | None = None,
        decoder_input_ids: Int[torch.Tensor, "batch tokens"] | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
        return_dict: bool | None = None,
    ) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        """Run teacher-forced LayoutFormer++ decoding."""
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        padding_mask = ~attention_mask.bool()
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("decoder_input_ids or labels must be provided")
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        enc_hs, enc_padding_mask = self.encode(input_ids, padding_mask, task_ids)
        dec_input = self.dec_pos_embedding(
            self.dec_embedding(decoder_input_ids).permute(1, 0, 2)
        )
        tgt_mask = generate_square_subsequent_mask(dec_input.size(0), dec_input.device)
        y = self.decoder(
            tgt=dec_input,
            memory=enc_hs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=enc_padding_mask,
        )
        logits = self.out(y.permute(1, 0, 2))
        loss = None
        if labels is not None:
            targets = labels.clone()
            targets[targets == self.pad_token_id] = -100
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return Seq2SeqLMOutput(loss=cast(torch.FloatTensor | None, loss), logits=logits)

    @torch.no_grad()
    def _generate_sequences(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        *,
        max_length: int | None = None,
        do_sample: bool = False,
        top_k: int = 10,
        temperature: float = 0.7,
        generation_constraint_fn: Callable[
            [int, int, torch.Tensor], tuple[list[int], int | None]
        ]
        | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
        generator: torch.Generator | None = None,
    ) -> Int[torch.Tensor, "batch tokens"]:
        """Run the reference greedy/top-k autoregressive loop."""
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        padding_mask = ~attention_mask.bool()
        max_length = max_length or self.config.decode_max_length
        enc_hs, enc_padding_mask = self.encode(input_ids, padding_mask, task_ids)
        bsz = input_ids.size(0)
        stop = input_ids.new_zeros(bsz, dtype=torch.bool)
        pred_ids = input_ids.new_full((bsz, 1), self.bos_token_id)
        outs: list[torch.Tensor] = []
        for idx in range(max_length):
            dec_input = self.dec_pos_embedding(
                self.dec_embedding(pred_ids).permute(1, 0, 2)
            )
            tgt_mask = generate_square_subsequent_mask(idx + 1, input_ids.device)
            y = self.decoder(
                tgt=dec_input,
                memory=enc_hs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=enc_padding_mask,
            )
            logits = self.out(y.permute(1, 0, 2)[:, -1, :])
            if generation_constraint_fn is not None:
                current = (
                    torch.stack(outs, dim=1) if outs else input_ids.new_empty((bsz, 0))
                )
                for batch_idx in range(bsz):
                    allowed, _ = generation_constraint_fn(
                        batch_idx, idx, current[batch_idx]
                    )
                    mask = torch.ones(
                        logits.size(-1), dtype=torch.bool, device=logits.device
                    )
                    mask[allowed] = False
                    logits[batch_idx].masked_fill_(mask, -math.inf)
            if do_sample:
                probs = F.softmax(top_k_logits(logits / temperature, top_k), dim=-1)
                curr = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            else:
                curr = torch.argmax(logits, dim=-1)
            eos = curr.eq(self.eos_token_id)
            curr[stop] = self.pad_token_id
            outs.append(curr)
            pred_ids = torch.cat([pred_ids, curr.unsqueeze(1)], dim=1)
            stop = torch.logical_or(stop, eos)
            if bool(torch.all(stop)):
                break
        return torch.stack(outs, dim=1)
