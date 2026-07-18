"""PyTorch model wrapper for Coarse-to-Fine layout generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from laygen.common.bbox import BoxFormat
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_coarse_to_fine import CoarseToFineConfig
from .hierarchy import decode_hierarchy_from_logits, flatten_hierarchy
from .types import OutputType, normalize_output_type

if TYPE_CHECKING:
    from .processing_coarse_to_fine import CoarseToFineProcessor


def make_seq_first(arg: torch.Tensor) -> torch.Tensor:
    """Convert ``(batch, seq, ...)`` tensors to ``(seq, batch, ...)``."""
    dims = [1, 0, *range(2, arg.dim())]
    return arg.permute(*dims)


def make_batch_first(arg: torch.Tensor) -> torch.Tensor:
    """Convert ``(seq, batch, ...)`` tensors to ``(batch, seq, ...)``."""
    dims = [1, 0, *range(2, arg.dim())]
    return arg.permute(*dims)


def get_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """Match the vendor cumulative padding-mask convention."""
    return (mask == 0).cumsum(dim=0) > 0


def get_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert batch-first mask to seq-first broadcast mask."""
    return make_seq_first(mask.unsqueeze(2))


def make_group_first(arg: torch.Tensor) -> torch.Tensor:
    """Convert ``(seq, group, batch, ...)`` to ``(group, seq, batch, ...)``."""
    return arg.permute(1, 0, 2, *range(3, arg.dim()))


def pack_group_batch(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Flatten group and batch dimensions in seq-first grouped tensors."""
    return tuple(
        arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:])
        for arg in args
    )


def unpack_group_batch(
    batch_size: int, *args: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Restore ``(seq, group, batch, ...)`` tensors from packed group batches."""
    return tuple(
        arg.reshape(arg.size(0), -1, batch_size, *arg.shape[2:]) for arg in args
    )


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """Create the causal decoder mask used by the vendor model."""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    return (
        mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    )


class LayoutEmbedding(nn.Module):
    """Vendor-compatible label, box, and group-label embeddings."""

    def __init__(self, config: CoarseToFineConfig) -> None:
        """Initialize embedding tables."""
        super().__init__()
        self.config = config
        self.label_embed = nn.Embedding(config.num_labels + 3, 128)
        self.bbox_embed = nn.Embedding(config.bbox_vocab_size, 128)
        self.proj_cat = nn.Linear(128 * 5, config.d_model)
        self.group_label_embed = nn.Linear(config.num_labels + 2, 128)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.kaiming_normal_(self.label_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.bbox_embed.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.proj_cat.weight, mode="fan_in")

    def get_label_embedding(self, label: torch.Tensor) -> torch.Tensor:
        """Embed element labels."""
        return self.label_embed(label)

    def get_box_embedding(self, box: torch.Tensor) -> torch.Tensor:
        """Embed four discrete box-coordinate ids and concatenate them."""
        bbox_vecs = self.bbox_embed(box)
        seq, batch, _, _ = bbox_vecs.shape
        return bbox_vecs.reshape(seq, batch, -1)

    def get_group_label_embedding(self, label: torch.Tensor) -> torch.Tensor:
        """Embed per-group label histograms."""
        return self.group_label_embed(label)

    def forward(self, label: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Embed labels and boxes into transformer hidden states."""
        label_vecs = self.get_label_embedding(label)
        box_vecs = self.get_box_embedding(box)
        return self.proj_cat(torch.cat((label_vecs, box_vecs), dim=-1))


class Encoder(nn.Module):
    """Vendor-compatible layout encoder."""

    def __init__(
        self, config: CoarseToFineConfig, layout_embd: LayoutEmbedding
    ) -> None:
        """Initialize the transformer encoder."""
        super().__init__()
        self.embedding = layout_embd
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        encoder_norm = nn.LayerNorm(config.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers, norm=encoder_norm
        )

    def forward(
        self, labels: torch.Tensor, bboxes: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Encode a seq-first padded layout and mean-pool valid states."""
        key_padding_mask = get_key_padding_mask(masks)
        src = self.embedding(labels, bboxes)
        memory = self.encoder(src=src, src_key_padding_mask=key_padding_mask)
        padding_mask = get_padding_mask(masks)
        return (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(
            dim=0, keepdim=True
        ).clamp_min(1)


class VAE(nn.Module):
    """Vendor-compatible latent sampler."""

    def __init__(self, config: CoarseToFineConfig) -> None:
        """Initialize latent projections."""
        super().__init__()
        self.config = config
        self.enc_mu_fcn = nn.Linear(config.d_model, config.d_z)
        self.enc_sigma_fcn = nn.Linear(config.d_model, config.d_z)
        self.z_fcn = nn.Linear(config.d_z, config.d_model)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(
        self, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent ``z`` from encoded memory."""
        mu = self.enc_mu_fcn(memory)
        logvar = self.enc_sigma_fcn(memory)
        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * torch.randn_like(sigma)
        return z, mu, logvar

    def inference(
        self, z: torch.Tensor | None, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return seq-first latent tensor for generation."""
        if z is None:
            return torch.randn(size=(1, batch_size, self.config.d_z), device=device)
        return (
            make_seq_first(z).to(device)
            if z.dim() == 3 and z.size(0) != 1
            else z.to(device)
        )


class GroupDecoder(nn.Module):
    """Autoregressive group box and label-histogram decoder."""

    def __init__(
        self, config: CoarseToFineConfig, layout_embd: LayoutEmbedding
    ) -> None:
        """Initialize group decoder modules."""
        super().__init__()
        self.config = config
        self.layout_embd = layout_embd
        self.proj_cat_tgt = nn.Linear(128 * 5, config.d_model)
        self.register_buffer(
            "square_subsequent_mask",
            generate_square_subsequent_mask(
                config.max_num_elements + 2, torch.device("cpu")
            ),
            persistent=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        decoder_norm = nn.LayerNorm(config.d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_layers_decoder, norm=decoder_norm
        )
        self.label_fcn = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.Linear(128, config.num_labels + 2),
            nn.ReLU(inplace=True),
        )
        self.box_fcn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.bbox_vocab_size),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        label: torch.Tensor,
        box: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced group decoding."""
        key_padding_mask = get_key_padding_mask(mask)
        tgt_label_vecs = self.layout_embd.get_group_label_embedding(label)
        tgt_box_vecs = self.layout_embd.get_box_embedding(box)
        tgt = self.proj_cat_tgt(torch.cat((tgt_label_vecs, tgt_box_vecs), dim=-1))
        length = tgt.size(0)
        causal_mask = generate_square_subsequent_mask(length, tgt.device)
        out = self.decoder(
            tgt[:length],
            z,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )
        rec_box = self.box_fcn(out)
        rec_label = self.label_fcn(out)
        return out[:-2], rec_box, rec_label

    def inference(
        self, z: torch.Tensor, *, max_group_num: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy autoregressive group decoding."""
        tgt = torch.zeros(max_group_num, z.shape[1], self.config.d_model, device=device)
        rec_labels = torch.zeros(
            max_group_num, z.shape[1], self.config.num_labels + 2, device=device
        )
        rec_bboxes = torch.zeros(
            max_group_num, z.shape[1], 4 * self.config.bbox_vocab_size, device=device
        )
        sos_box = torch.zeros(z.shape[1], 4, dtype=torch.long, device=device)
        sos_label = torch.zeros(z.shape[1], self.config.num_labels + 2, device=device)
        sos_label[:, self.config.group_sos_index] = 1.0
        sos_box_vecs = self.layout_embd.get_box_embedding(sos_box.unsqueeze(0))
        sos_label_vecs = self.layout_embd.group_label_embed(sos_label.unsqueeze(0))
        tgt[0] = self.proj_cat_tgt(
            torch.cat((sos_label_vecs, sos_box_vecs), dim=-1)
        ).squeeze(0)
        out = torch.zeros_like(tgt)
        for idx in range(max_group_num):
            decoded = self.decoder(tgt[: idx + 1], z)
            out[idx] = decoded[idx]
            rec_box_i = self.box_fcn(out[idx])
            rec_label_i = self.label_fcn(out[idx])
            rec_bboxes[idx] = rec_box_i
            rec_labels[idx] = rec_label_i
            if idx < max_group_num - 1:
                next_box = rec_box_i.reshape(-1, 4, self.config.bbox_vocab_size).argmax(
                    -1
                )
                tgt_label_vecs = self.layout_embd.group_label_embed(rec_label_i)
                tgt_box_vecs = self.layout_embd.get_box_embedding(
                    next_box.unsqueeze(0)
                ).squeeze(0)
                tgt[idx + 1] = self.proj_cat_tgt(
                    torch.cat((tgt_label_vecs, tgt_box_vecs), dim=-1)
                )
        return out[:-2], rec_bboxes, rec_labels


class ElementDecoder(nn.Module):
    """Autoregressive element decoder conditioned on group memory."""

    def __init__(
        self, config: CoarseToFineConfig, layout_embd: LayoutEmbedding
    ) -> None:
        """Initialize element decoder modules."""
        super().__init__()
        self.config = config
        self.layout_embd = layout_embd
        self.proj_cat_memory = nn.Sequential(
            nn.Linear(config.d_model + config.d_model, config.d_model)
        )
        self.register_buffer(
            "square_subsequent_mask",
            generate_square_subsequent_mask(
                config.max_num_elements + 2, torch.device("cpu")
            ),
            persistent=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        decoder_norm = nn.LayerNorm(config.d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_layers_decoder, norm=decoder_norm
        )
        self.label_fcn = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.Linear(128, config.num_labels + 3),
            nn.ReLU(inplace=True),
        )
        self.box_fcn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.bbox_vocab_size),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        label: torch.Tensor,
        box: torch.Tensor,
        memory: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced element decoding."""
        z = z.repeat(memory.size(0), 1, 1).unsqueeze(0)
        memory = memory.unsqueeze(0)
        memory, z, label, box = pack_group_batch(memory, z, label, box)
        memory = self.proj_cat_memory(torch.cat((memory, z), dim=-1))
        batch_size, _, _ = mask.shape
        tgt = self.layout_embd(label.squeeze(-1), box)
        length = tgt.size(0)
        causal_mask = generate_square_subsequent_mask(length, tgt.device)
        out = self.decoder(tgt[:length], memory, tgt_mask=causal_mask)
        rec_box = self.box_fcn(out)
        rec_label = self.label_fcn(out)
        rec_box, rec_label = unpack_group_batch(batch_size, rec_box, rec_label)
        return rec_box, rec_label

    def inference(
        self,
        memory: torch.Tensor,
        z: torch.Tensor,
        *,
        max_num_elements: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy autoregressive element decoding."""
        groups, batch_size, _ = memory.shape
        z = z.repeat(memory.size(0), 1, 1).unsqueeze(0)
        memory = memory.unsqueeze(0)
        memory, z = pack_group_batch(memory, z)
        memory = self.proj_cat_memory(torch.cat((memory, z), dim=-1))
        tgt = torch.zeros(
            max_num_elements, z.shape[1], self.config.d_model, device=device
        )
        rec_label = torch.zeros(
            max_num_elements, z.shape[1], self.config.num_labels + 3, device=device
        )
        rec_box = torch.zeros(
            max_num_elements, z.shape[1], 4 * self.config.bbox_vocab_size, device=device
        )
        sos_box = torch.zeros(z.shape[1], 4, dtype=torch.long, device=device)
        sos_label = torch.full(
            (z.shape[1],), self.config.element_sos_id, dtype=torch.long, device=device
        )
        tgt[0] = self.layout_embd(sos_label.unsqueeze(0), sos_box.unsqueeze(0)).squeeze(
            0
        )
        out = torch.zeros_like(tgt)
        for idx in range(max_num_elements):
            decoded = self.decoder(tgt[: idx + 1], memory)
            out[idx] = decoded[idx]
            rec_box_i = self.box_fcn(out[idx])
            rec_label_i = self.label_fcn(out[idx])
            rec_box[idx] = rec_box_i
            rec_label[idx] = rec_label_i
            if idx < max_num_elements - 1:
                next_box = rec_box_i.reshape(-1, 4, self.config.bbox_vocab_size).argmax(
                    -1
                )
                next_label = rec_label_i.argmax(1)
                tgt[idx + 1] = self.layout_embd(
                    next_label.unsqueeze(0), next_box.unsqueeze(0)
                ).squeeze(0)
        rec_box, rec_label = unpack_group_batch(batch_size, rec_box, rec_label)
        _ = groups
        return rec_box, rec_label


class CoarseToFineForLayoutGeneration(PreTrainedModel):
    """Transformers ``PreTrainedModel`` with vendor-compatible module names."""

    config_class = CoarseToFineConfig
    base_model_prefix = "coarse_to_fine"
    main_input_name = "labels"
    _tied_weights_keys = {
        "encoder.embedding.label_embed.weight": "layout_embd.label_embed.weight",
        "group_decoder.layout_embd.label_embed.weight": "layout_embd.label_embed.weight",
        "ele_decoder.layout_embd.label_embed.weight": "layout_embd.label_embed.weight",
        "encoder.embedding.bbox_embed.weight": "layout_embd.bbox_embed.weight",
        "group_decoder.layout_embd.bbox_embed.weight": "layout_embd.bbox_embed.weight",
        "ele_decoder.layout_embd.bbox_embed.weight": "layout_embd.bbox_embed.weight",
        "encoder.embedding.proj_cat.weight": "layout_embd.proj_cat.weight",
        "group_decoder.layout_embd.proj_cat.weight": "layout_embd.proj_cat.weight",
        "ele_decoder.layout_embd.proj_cat.weight": "layout_embd.proj_cat.weight",
        "encoder.embedding.proj_cat.bias": "layout_embd.proj_cat.bias",
        "group_decoder.layout_embd.proj_cat.bias": "layout_embd.proj_cat.bias",
        "ele_decoder.layout_embd.proj_cat.bias": "layout_embd.proj_cat.bias",
        "encoder.embedding.group_label_embed.weight": (
            "layout_embd.group_label_embed.weight"
        ),
        "group_decoder.layout_embd.group_label_embed.weight": (
            "layout_embd.group_label_embed.weight"
        ),
        "ele_decoder.layout_embd.group_label_embed.weight": (
            "layout_embd.group_label_embed.weight"
        ),
        "encoder.embedding.group_label_embed.bias": (
            "layout_embd.group_label_embed.bias"
        ),
        "group_decoder.layout_embd.group_label_embed.bias": (
            "layout_embd.group_label_embed.bias"
        ),
        "ele_decoder.layout_embd.group_label_embed.bias": (
            "layout_embd.group_label_embed.bias"
        ),
    }

    def __init__(self, config: CoarseToFineConfig) -> None:
        """Initialize vendor-compatible modules."""
        super().__init__(config)
        self.layout_embd = LayoutEmbedding(config)
        self.encoder = Encoder(config, self.layout_embd)
        self.vae = VAE(config)
        self.group_decoder = GroupDecoder(config, self.layout_embd)
        self.ele_decoder = ElementDecoder(config, self.layout_embd)
        self.all_tied_weights_keys = dict(self._tied_weights_keys)

    @property
    def device(self) -> torch.device:
        """Return the current parameter device."""
        return next(self.parameters()).device

    def _sample_latent(
        self,
        *,
        batch_size: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor,
            torch.randn(
                (1, batch_size, self.config.d_z),
                generator=generator,
                device=device,
            ),
        )

    def _normalize_generation_condition(
        self, condition_type: ConditionType | str
    ) -> ConditionType:
        condition = normalize_condition_type(condition_type)
        if condition is not ConditionType.unconditional:
            raise NotImplementedError(
                "Coarse-to-Fine released checkpoints support only unconditional generation"
            )
        return condition

    def forward(
        self,
        labels: torch.LongTensor,
        bbox: torch.LongTensor,
        mask: torch.BoolTensor,
        group_bounding_box: torch.LongTensor,
        label_in_one_group: torch.FloatTensor,
        group_mask: torch.BoolTensor,
        grouped_bbox: torch.LongTensor,
        grouped_labels: torch.LongTensor,
        grouped_mask: torch.BoolTensor,
        latent_z: torch.FloatTensor | None = None,
        use_teacher_forcing: bool = True,
        return_dict: bool | None = None,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, ...]:
        """Run teacher-forced or greedy hierarchical decoding.

        Args:
            labels: Batch-first vendor label ids.
            bbox: Batch-first discrete ``ltwh`` ids.
            mask: Batch-first valid element mask.
            group_bounding_box: Batch-first group discrete ``ltwh`` ids.
            label_in_one_group: Batch-first group label histograms.
            group_mask: Batch-first valid group mask.
            grouped_bbox: Batch-first group-relative discrete ``ltwh`` ids.
            grouped_labels: Batch-first group-relative vendor labels.
            grouped_mask: Batch-first valid grouped-element mask.
            latent_z: Optional latent tensor to bypass stochastic sampling.
            use_teacher_forcing: Whether to use provided hierarchy tensors.
            return_dict: Return a dictionary when true.

        Returns:
            Dictionary of raw logits and latent tensors.
        """
        batch_size, groups, seq, dims = grouped_bbox.shape
        seq_bbox = make_seq_first(bbox)
        seq_group_bbox = make_seq_first(group_bounding_box)
        grouped_box = make_seq_first(
            grouped_bbox.reshape(batch_size, groups * seq, dims)
        )
        grouped_box = make_seq_first(
            grouped_box.reshape(groups, seq, batch_size * dims)
        ).reshape(seq, groups, batch_size, dims)
        seq_labels = make_seq_first(labels.unsqueeze(2)).squeeze(2)
        seq_group_labels = make_seq_first(label_in_one_group)
        grouped_label = make_seq_first(
            grouped_labels.reshape(batch_size, groups * seq, 1)
        )
        grouped_label = make_seq_first(
            grouped_label.reshape(groups, seq, batch_size)
        ).unsqueeze(3)
        memory = self.encoder(seq_labels, seq_bbox, mask)
        if latent_z is None:
            z, mu, logvar = self.vae(memory)
        else:
            z = (
                latent_z
                if latent_z.dim() == 3 and latent_z.size(0) == 1
                else make_seq_first(latent_z)
            )
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)
        if use_teacher_forcing:
            group_embd, rec_group_bbox, rec_group_label = self.group_decoder(
                seq_group_labels, seq_group_bbox, z, group_mask
            )
            rec_box, rec_label = self.ele_decoder(
                grouped_label,
                grouped_box,
                group_embd,
                z,
                grouped_mask,
            )
        else:
            group_embd, rec_group_bbox, rec_group_label = self.group_decoder.inference(
                z, max_group_num=seq_group_labels.shape[0], device=self.device
            )
            rec_box, rec_label = self.ele_decoder.inference(
                group_embd,
                z,
                max_num_elements=grouped_label.shape[0],
                device=self.device,
            )
        rec_box = make_group_first(rec_box)
        rec_label = make_group_first(rec_label)
        flat_box = make_batch_first(rec_box.reshape(groups * seq, batch_size, -1))
        flat_label = make_batch_first(rec_label.reshape(groups * seq, batch_size, -1))
        rec_group_bbox = make_batch_first(rec_group_bbox)
        rec_group_label = make_batch_first(rec_group_label)
        output = {
            "group_bounding_box_logits": rec_group_bbox.reshape(
                batch_size, rec_group_bbox.size(1), 4, self.config.bbox_vocab_size
            ),
            "label_in_one_group_logits": rec_group_label,
            "grouped_bbox_logits": flat_box.reshape(
                batch_size, groups, seq, 4, self.config.bbox_vocab_size
            ),
            "grouped_label_logits": flat_label.reshape(batch_size, groups, seq, -1),
            "mu": make_batch_first(mu),
            "logvar": make_batch_first(logvar),
            "latent_z": make_batch_first(z),
        }
        if return_dict is False:
            return tuple(output.values())
        return output

    @torch.no_grad()
    def decode_hierarchy(self, latent_z: torch.FloatTensor) -> dict[str, torch.Tensor]:
        """Decode raw hierarchy logits from a seq-first latent tensor."""
        z = latent_z.to(self.device)
        group_embd, group_bbox, group_label = self.group_decoder.inference(
            z,
            max_group_num=self.config.max_num_elements + 2,
            device=self.device,
        )
        elem_bbox, elem_label = self.ele_decoder.inference(
            group_embd,
            z,
            max_num_elements=self.config.max_num_elements + 2,
            device=self.device,
        )
        elem_bbox = make_group_first(elem_bbox)
        elem_label = make_group_first(elem_label)
        groups, seq, batch_size, _ = elem_bbox.shape
        return {
            "group_bounding_box_logits": make_batch_first(group_bbox).reshape(
                batch_size,
                group_bbox.size(0),
                4,
                self.config.bbox_vocab_size,
            ),
            "label_in_one_group_logits": make_batch_first(group_label),
            "grouped_bbox_logits": make_batch_first(
                elem_bbox.reshape(groups * seq, batch_size, -1)
            ).reshape(batch_size, groups, seq, 4, self.config.bbox_vocab_size),
            "grouped_label_logits": make_batch_first(
                elem_label.reshape(groups * seq, batch_size, -1)
            ).reshape(batch_size, groups, seq, -1),
        }

    @torch.no_grad()
    def generate_layout(
        self,
        *,
        batch_size: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        latent_z: torch.FloatTensor | None = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        processor: CoarseToFineProcessor | None = None,
        output_type: OutputType | str = OutputType.dataclass,
        return_intermediates: bool = False,
        labels: object = None,
        bbox: object = None,
        mask: object = None,
        num_elements: int | list[int] | torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        num_inference_steps: int | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Generate a layout with the released unconditional checkpoint path.

        Args:
            batch_size: Number of layouts to generate.
            seed: Optional seed used only when ``generator`` is absent.
            generator: Local torch generator. Takes precedence over ``seed``.
            latent_z: Optional latent tensor with shape ``(1, B, d_z)``.
            condition_type: Public condition name or alias.
            processor: Optional processor used for output post-processing.
            output_type: ``"dataclass"`` or ``"dict"``.
            return_intermediates: Whether to retain hierarchy metadata.
            labels: Unsupported v1 condition input.
            bbox: Unsupported v1 condition input.
            mask: Unsupported v1 condition input.
            num_elements: Unsupported v1 condition input.
            box_format: Public input box format.
            normalized: Whether inputs are normalized.
            canvas_size: Optional canvas size for future conditioned paths.
            num_inference_steps: Unused; kept for shared API compatibility.

        Returns:
            Shared layout output or dictionary.

        Raises:
            NotImplementedError: If a supported public condition is requested
                but no released Coarse-to-Fine checkpoint implements it.

        Examples:
            >>> config = CoarseToFineConfig(d_model=16, d_z=16, n_heads=4, dim_feedforward=32, n_layers=1, n_layers_decoder=1, max_num_elements=2)
            >>> model = CoarseToFineForLayoutGeneration(config).eval()
            >>> out = model.generate_layout(batch_size=1, seed=0)
            >>> out.bbox.shape[0]
            1
        """
        _ = (
            labels,
            bbox,
            mask,
            num_elements,
            box_format,
            normalized,
            canvas_size,
            num_inference_steps,
        )
        self._normalize_generation_condition(condition_type)
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        if latent_z is None:
            sampled_z = self._sample_latent(
                batch_size=batch_size, generator=generator, device=self.device
            )
        else:
            sampled_z = cast(torch.FloatTensor, latent_z.to(self.device))
            batch_size = (
                sampled_z.size(1) if sampled_z.dim() == 3 else sampled_z.size(0)
            )
        raw = self.decode_hierarchy(sampled_z)
        hierarchy = decode_hierarchy_from_logits(
            group_bbox_logits=raw["group_bounding_box_logits"],
            group_label_logits=raw["label_in_one_group_logits"],
            grouped_bbox_logits=raw["grouped_bbox_logits"],
            grouped_label_logits=raw["grouped_label_logits"],
            num_labels=self.config.num_labels,
            group_eos_index=self.config.group_eos_index,
            element_eos_id=self.config.element_eos_id,
            discrete_x_grid=self.config.discrete_x_grid,
            discrete_y_grid=self.config.discrete_y_grid,
        )
        id2label = (
            dict(processor.id2label)
            if processor is not None
            else {int(key): str(value) for key, value in self.config.id2label.items()}
        )
        output = flatten_hierarchy(
            hierarchy, id2label=id2label, max_num_elements=self.config.max_num_elements
        )
        output.sequences = cast(torch.Tensor, hierarchy.discrete_relative_bbox)
        output.scores = None
        output.trajectory = raw if return_intermediates else None
        if not return_intermediates:
            output.intermediates = None
        normalized_output_type = normalize_output_type(output_type)
        if normalized_output_type is OutputType.dict:
            return dict(output)
        return output
