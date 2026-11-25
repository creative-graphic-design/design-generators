"""PyTorch model classes for Flex-DM masked document modeling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from laygen.nn import MultiHeadSelfAttention

from .configuration_flex_dm import FlexDmColumnSpec, FlexDmConfig
from .masking import get_seq_mask

_RICO_SORT_KEYS = ("type", "left", "top", "width", "height")
_KERAS_EPSILON = 1e-7


def _sort_indices(
    values: Mapping[str, Shaped[torch.Tensor, "..."]],
    columns: Mapping[str, FlexDmColumnSpec],
    lengths: Shaped[torch.Tensor, "batch 1"],
    *,
    from_logits: bool,
) -> Int[torch.Tensor, "batch seq"]:
    seq_len = values["type"].shape[1]
    priority = torch.zeros(
        lengths.shape[0], seq_len, dtype=torch.long, device=lengths.device
    )
    for key in _RICO_SORT_KEYS:
        value = values[key]
        if from_logits and columns[key]["type"] == "categorical":
            value = value.argmax(dim=-1)

        priority = priority * 100 + value[..., 0].long()

    invalid = ~get_seq_mask(lengths, maxlen=seq_len)
    return torch.argsort(priority + invalid.long() * 100 ** len(_RICO_SORT_KEYS), dim=1)


def _gather_sequence(
    values: Mapping[str, Shaped[torch.Tensor, "..."]],
    columns: Mapping[str, FlexDmColumnSpec],
    indices: Int[torch.Tensor, "batch seq"],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    ordered: dict[str, Shaped[torch.Tensor, "..."]] = {}
    for key, value in values.items():
        if columns[key]["is_sequence"]:
            index_shape = (*indices.shape, *((1,) * (value.ndim - 2)))
            index = indices.view(index_shape).expand_as(value)
            ordered[key] = value.gather(1, index)
        else:
            ordered[key] = value

    return ordered


@dataclass
class FlexDmModelOutput(ModelOutput):
    """Output of ``FlexDmForMaskedDocumentModeling``.

    Args:
        logits: Per-column logits or numerical predictions.
        loss: Optional reconstruction objective including regularization.
        loss_components: Optional per-column reconstruction losses.
        l2_loss: Optional sum of configured parameter penalties.
        hidden_states: Optional final hidden states.
        masks: Optional per-column hidden-field masks.
    """

    logits: dict[str, Shaped[torch.Tensor, "..."]]
    loss: Float[torch.Tensor, ""] | None = None
    loss_components: dict[str, Float[torch.Tensor, ""]] | None = None
    l2_loss: Float[torch.Tensor, ""] | None = None
    hidden_states: Float[torch.Tensor, "batch seq channels"] | None = None
    masks: dict[str, Bool[torch.Tensor, "..."]] | None = None

    def __post_init__(self) -> None:
        """Keep the logits dictionary as one ModelOutput field."""
        if self.logits is not None:
            self["logits"] = self.logits
        if self.loss is not None:
            self["loss"] = self.loss
        if self.loss_components is not None:
            self["loss_components"] = self.loss_components
        if self.l2_loss is not None:
            self["l2_loss"] = self.l2_loss
        if self.hidden_states is not None:
            self["hidden_states"] = self.hidden_states
        if self.masks is not None:
            self["masks"] = self.masks


def _module_key(key: str) -> str:
    return f"field_{key}"


class FlexDmPreTrainedModel(PreTrainedModel):
    """Base class for Flex-DM Transformers models."""

    config_class = FlexDmConfig
    base_model_prefix = "flex_dm"
    supports_gradient_checkpointing = False


class FlexDmInputEncoder(nn.Module):
    """Encode heterogeneous Flex-DM input columns into one hidden sequence."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create per-column embeddings and projections."""
        super().__init__()
        self.config = config
        self.input_embeddings = nn.ModuleDict()
        self.input_projections = nn.ModuleDict()
        self.special_embeddings = nn.ModuleDict()
        for key, column in config.input_columns.items():
            if not column["is_sequence"]:
                continue
            if column["type"] == "categorical":
                input_dim = cast(int, column["input_dim"])
                self.input_embeddings[_module_key(key)] = nn.Embedding(
                    input_dim + 2,
                    config.latent_dim,
                )
            else:
                self.input_projections[_module_key(key)] = nn.Linear(
                    int(column["shape"][-1]),
                    config.latent_dim,
                )
                self.special_embeddings[_module_key(key)] = nn.Embedding(
                    2,
                    config.latent_dim,
                )
        self.task_embedding = (
            nn.Embedding(len(config.task_names), config.latent_dim)
            if config.context == "id"
            else None
        )
        self.length_embedding = (
            nn.Embedding(config.max_seq_length + 1, config.latent_dim)
            if config.context == "length"
            else None
        )
        self.position_embedding = (
            nn.Embedding(config.max_seq_length, config.latent_dim)
            if config.input_dtype != "set"
            else None
        )

    def forward(
        self,
        inputs: Mapping[str, Shaped[torch.Tensor, "..."]],
        *,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch seq channels"],
        Bool[torch.Tensor, "batch seq"],
    ]:
        """Encode model inputs.

        Args:
            inputs: Per-column tensors.
            task_ids: Optional task ids.

        Returns:
            Hidden sequence and valid-element mask.
        """
        first_key = self.config.valid_sequence_keys[0]
        batch, seq_len = inputs[first_key].shape[:2]
        device = inputs[first_key].device
        hidden = torch.zeros(
            batch,
            seq_len,
            self.config.latent_dim,
            device=device,
            dtype=torch.float32,
        )
        for key in self.config.valid_sequence_keys:
            column = self.config.input_columns[key]
            value = inputs[key].to(device)
            if column["type"] == "categorical":
                embedded = self.input_embeddings[_module_key(key)](value.long())
                if embedded.ndim == 4:
                    embedded = embedded.sum(dim=-2)
            else:
                projected = self.input_projections[_module_key(key)](value.float())
                masked = (value == 10.0).all(dim=-1)
                unused = (value == 0.0).all(dim=-1)
                special_ids = torch.zeros_like(masked, dtype=torch.long)
                special_ids = torch.where(
                    unused, torch.ones_like(special_ids), special_ids
                )
                special = self.special_embeddings[_module_key(key)](special_ids)
                embedded = torch.where(
                    (masked | unused).unsqueeze(-1), special, projected
                )
            hidden = hidden + embedded
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=device).clamp(
                max=self.config.max_seq_length - 1
            )
            hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        if self.task_embedding is not None and task_ids is not None:
            hidden = hidden + self.task_embedding(task_ids.to(device)).unsqueeze(1)
        if self.length_embedding is not None and "length" in inputs:
            length_ids = (
                inputs["length"]
                .reshape(batch)
                .long()
                .clamp(
                    min=0,
                    max=self.config.max_seq_length,
                )
            )
            hidden = hidden + self.length_embedding(length_ids).unsqueeze(1)
        if "length" in inputs:
            from .masking import get_seq_mask

            seq_mask = get_seq_mask(
                inputs["length"].reshape(batch).long(),
                maxlen=seq_len,
            )
        else:
            seq_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
        return hidden, seq_mask


class FlexDmDeepSvgBlock(nn.Module):
    """DeepSVG-style pre-norm transformer block."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create one transformer block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadSelfAttention(config.latent_dim, num_heads=8)
        self.norm2 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Float[torch.Tensor, "batch seq channels"],
        attention_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq channels"]:
        """Apply pre-norm attention and MLP residuals."""
        hidden_states = hidden_states + self.dropout(
            self.attention(self.norm1(hidden_states), attention_mask)
        )
        return hidden_states + self.dropout(self.mlp(self.norm2(hidden_states)))


class FlexDmDecoder(nn.Module):
    """Decode hidden states into one head per model column."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Create per-column output heads."""
        super().__init__()
        self.config = config
        self.heads = nn.ModuleDict()
        for key in config.valid_sequence_keys:
            column = config.input_columns[key]
            units = int(column["shape"][-1])
            if column["type"] == "categorical":
                units *= cast(int, column["input_dim"])
            self.heads[_module_key(key)] = nn.Linear(config.latent_dim, units)

    def forward(
        self, hidden_states: Float[torch.Tensor, "batch seq channels"]
    ) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Return per-column logits and predictions."""
        outputs: dict[str, Shaped[torch.Tensor, "..."]] = {}
        for key in self.config.valid_sequence_keys:
            head = self.heads[_module_key(key)]
            column = self.config.input_columns[key]
            raw = head(hidden_states)
            if column["type"] == "categorical":
                shape_dim = int(column["shape"][-1])
                input_dim = cast(int, column["input_dim"])
                outputs[key] = raw.view(*raw.shape[:2], shape_dim, input_dim)
            else:
                outputs[key] = raw.view(*raw.shape[:2], int(column["shape"][-1]))
        return outputs


class FlexDmForMaskedDocumentModeling(FlexDmPreTrainedModel):
    """Flex-DM MFP model with a standard Transformers ``forward`` method."""

    def __init__(self, config: FlexDmConfig) -> None:
        """Initialize encoder, transformer blocks, and decoder."""
        super().__init__(config)
        if config.arch_type != "oneshot":
            raise ValueError("Only arch_type='oneshot' is supported")

        if config.block_type != "deepsvg":
            raise ValueError("Only block_type='deepsvg' is supported")

        self.encoder = FlexDmInputEncoder(config)
        self.blocks = nn.ModuleList(
            [FlexDmDeepSvgBlock(config) for _ in range(config.num_blocks)]
        )
        self.decoder = FlexDmDecoder(config)
        self.post_init()

    def forward(
        self,
        *,
        inputs: Mapping[str, Shaped[torch.Tensor, "..."]],
        masks: Mapping[str, Bool[torch.Tensor, "..."]] | None = None,
        labels: Mapping[str, Shaped[torch.Tensor, "..."]] | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
    ) -> (
        FlexDmModelOutput
        | tuple[dict[str, Shaped[torch.Tensor, "..."]], Float[torch.Tensor, ""] | None]
    ):
        """Run a Flex-DM forward pass.

        Args:
            inputs: Per-column model input tensors.
            masks: Per-column hidden-field masks; required with labels.
            labels: Optional per-column reconstruction targets.
            task_ids: Optional task ids.
            output_hidden_states: Whether to include final hidden states.
            return_dict: Whether to return a ``ModelOutput``.

        Returns:
            ``FlexDmModelOutput`` by default.
        """
        hidden_states, seq_mask = self.encoder(inputs, task_ids=task_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, seq_mask)
        logits = self.decoder(hidden_states)
        loss_components = None
        l2_loss = None
        loss = None
        if labels is not None:
            if masks is None:
                raise ValueError("masks are required when labels are provided")

            loss_components = self._compute_loss(logits, labels, masks, task_ids)
            l2_loss = self._compute_l2_loss()
            loss = torch.stack((*loss_components.values(), l2_loss)).sum()

        output = FlexDmModelOutput(
            logits=logits,
            loss=loss,
            loss_components=loss_components,
            l2_loss=l2_loss,
            hidden_states=hidden_states if output_hidden_states else None,
            masks=dict(masks) if masks is not None else None,
        )
        if return_dict is False:
            return logits, loss
        return output

    def _compute_loss(
        self,
        logits: Mapping[str, Shaped[torch.Tensor, "..."]],
        labels: Mapping[str, Shaped[torch.Tensor, "..."]],
        masks: Mapping[str, Bool[torch.Tensor, "..."]],
        task_ids: Int[torch.Tensor, "batch"] | None,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        if self.config.dataset_name == "rico" and task_ids is not None:
            sort_flag = task_ids == self.config.task_names.index("pos")
            label_indices = _sort_indices(
                labels,
                self.config.input_columns,
                labels["length"],
                from_logits=False,
            )
            prediction_indices = _sort_indices(
                logits,
                self.config.input_columns,
                labels["length"],
                from_logits=True,
            )
            sorted_labels = _gather_sequence(
                labels, self.config.input_columns, label_indices
            )
            sorted_logits = _gather_sequence(
                logits, self.config.input_columns, prediction_indices
            )
            labels = {
                key: torch.where(
                    sort_flag.view(-1, *((1,) * (value.ndim - 1))),
                    sorted_labels[key],
                    value,
                )
                if self.config.input_columns[key]["is_sequence"]
                else value
                for key, value in labels.items()
            }
            logits = {
                key: torch.where(
                    sort_flag.view(-1, *((1,) * (value.ndim - 1))),
                    sorted_logits[key],
                    value,
                )
                if self.config.input_columns[key]["is_sequence"]
                else value
                for key, value in logits.items()
            }

        seq_mask = get_seq_mask(
            labels["length"], maxlen=next(iter(logits.values())).shape[1]
        )
        losses: dict[str, Float[torch.Tensor, ""]] = {}
        for key, column in self.config.input_columns.items():
            if not column["is_sequence"] or key not in logits or key not in labels:
                continue
            pred = logits[key]
            target = labels[key]
            if column["type"] == "categorical":
                vocab = cast(int, column["input_dim"])
                # Match the reference Keras clipping of eager softmax probabilities.
                log_probabilities = (
                    pred.softmax(dim=-1)
                    .clamp(min=_KERAS_EPSILON, max=1 - _KERAS_EPSILON)
                    .log()
                )
                loss = F.cross_entropy(
                    log_probabilities.reshape(-1, vocab),
                    target.long().reshape(-1),
                    reduction="none",
                ).reshape(target.shape)
            else:
                loss = (
                    F.mse_loss(pred, target.float(), reduction="none")
                    .mean(dim=-1)
                    .mul(int(column["shape"][-1]))
                    .unsqueeze(-1)
                )

            loss = loss * masks[key].to(device=loss.device, dtype=loss.dtype).unsqueeze(
                -1
            )
            condition = column.get("loss_condition")
            if condition is not None:
                condition_labels = labels[condition["key"]].squeeze(-1).long()
                condition_mask = torch.as_tensor(condition["mask"], device=loss.device)[
                    condition_labels
                ]
                loss = loss * condition_mask.unsqueeze(-1)

            loss = loss * seq_mask.to(dtype=loss.dtype).unsqueeze(-1)
            losses[key] = loss.sum(dim=1).sum(dim=1).mean()

        return losses

    def _compute_l2_loss(self) -> Float[torch.Tensor, ""]:
        if not self.config.l2:
            return next(self.parameters()).new_zeros(())

        # The pinned model regularizes Dense kernels/biases and embedding tables.
        penalties = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                penalties.append(module.weight.square().sum())
                if module.bias is not None:
                    penalties.append(module.bias.square().sum())
            elif isinstance(module, nn.Embedding):
                penalties.append(module.weight.square().sum())

        return torch.stack(penalties).sum() * self.config.l2
