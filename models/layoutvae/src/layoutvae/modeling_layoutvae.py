"""PyTorch model classes for LayoutVAE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final, TypeAlias, cast

import torch
from jaxtyping import Bool, Float, Int
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from laygen.common.bbox import clamp_boxes, ltwh_to_xywh
from laygen.common.enums import normalize_enum_value

from .configuration_layoutvae import LayoutVAEConfig

INTERNAL_EMPTY_LABEL_ID: Final[int] = 0
LayoutVAETupleOutput: TypeAlias = tuple[
    Float[torch.Tensor, "batch elements 4"],
    Float[torch.Tensor, "batch elements 4"],
    Int[torch.Tensor, "batch elements"],
    Bool[torch.Tensor, "batch elements"],
    Float[torch.Tensor, "batch internal_labels"],
    Int[torch.Tensor, "batch elements"],
]


class OutputType(StrEnum):
    """Supported generation output formats."""

    dataclass = auto()
    dict = auto()


def normalize_output_type(output_type: OutputType | str) -> OutputType:
    """Normalize a public output type.

    Args:
        output_type: Output type enum or string.

    Returns:
        Normalized output type enum.

    Raises:
        ValueError: If the value is unsupported.

    Examples:
        >>> str(normalize_output_type("dict"))
        'dict'
    """
    return normalize_enum_value(
        output_type,
        OutputType,
        option_name="output_type",
    )


@dataclass
class LayoutVAEModelOutput(ModelOutput):
    """Raw LayoutVAE model output.

    Args:
        raw_ltwh: Internal normalized left-top-width-height boxes.
        bbox: Public normalized center `xywh` boxes.
        labels: Public label IDs.
        mask: Valid-element mask.
        class_counts: Six-way class-count tensor.
        label_set: Optional label-set input.
        internal_labels: Optional six-way generated label IDs.

    Examples:
        >>> output = LayoutVAEModelOutput(
        ...     raw_ltwh=torch.zeros(1, 1, 4),
        ...     bbox=torch.zeros(1, 1, 4),
        ...     labels=torch.zeros(1, 1, dtype=torch.long),
        ...     mask=torch.ones(1, 1, dtype=torch.bool),
        ...     class_counts=torch.ones(1, 6),
        ... )
        >>> tuple(output.bbox.shape)
        (1, 1, 4)
    """

    raw_ltwh: Float[torch.Tensor, "batch elements 4"]
    bbox: Float[torch.Tensor, "batch elements 4"] = cast(
        Float[torch.Tensor, "batch elements 4"], None
    )
    labels: Int[torch.Tensor, "batch elements"] = cast(
        Int[torch.Tensor, "batch elements"], None
    )
    mask: Bool[torch.Tensor, "batch elements"] = cast(
        Bool[torch.Tensor, "batch elements"], None
    )
    class_counts: Float[torch.Tensor, "batch internal_labels"] = cast(
        Float[torch.Tensor, "batch internal_labels"], None
    )
    label_set: Float[torch.Tensor, "batch internal_labels"] | None = None
    internal_labels: Int[torch.Tensor, "batch elements"] | None = None


class FCBlock(nn.Module):
    """Two-layer fully connected block used by the encoders."""

    def __init__(self, n_class: int) -> None:
        """Initialize the block.

        Args:
            n_class: Input dimension.
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_class, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(
        self, inputs: Float[torch.Tensor, "batch features"]
    ) -> Float[torch.Tensor, "batch 128"]:
        """Run the block."""
        return self.seq(inputs)


class Embeder(nn.Module):
    """Count module embedding network."""

    def __init__(self, n_class: int) -> None:
        """Initialize the embedding network."""
        super().__init__()
        self.fcb1 = FCBlock(n_class)
        self.fcb2 = FCBlock(n_class)
        self.fcb3 = FCBlock(n_class)
        self.fc = nn.Linear(128 * 3, 128)

    def forward(
        self,
        inputs: tuple[
            Float[torch.Tensor, "batch internal_labels"],
            Float[torch.Tensor, "batch internal_labels"],
            Float[torch.Tensor, "batch internal_labels"],
        ],
    ) -> Float[torch.Tensor, "batch 128"]:
        """Embed label-set, current-label, and previous-count tensors."""
        in1, in2, in3 = inputs
        return self.fc(torch.cat((self.fcb1(in1), self.fcb2(in2), self.fcb3(in3)), 1))


class Encoder(nn.Module):
    """Gaussian posterior encoder."""

    def __init__(self, in_dim: int = 1, latent_dim: int = 32) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(256, latent_dim)
        self.fc4 = nn.Linear(latent_dim, latent_dim)
        self.fc5 = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        inputs: tuple[
            Float[torch.Tensor, "batch input_features"],
            Float[torch.Tensor, "batch 128"],
        ],
    ) -> tuple[
        Float[torch.Tensor, "batch latent"],
        Float[torch.Tensor, "batch latent"],
    ]:
        """Encode a target tensor and conditional embedding."""
        in1, in2 = inputs
        out = self.act(self.fc1(in1))
        out = torch.cat((self.fc2(out), in2), 1)
        out = self.act(self.fc3(out))
        return self.fc4(out), self.fc5(out)


class Prior(nn.Module):
    """Conditional Gaussian prior network."""

    def __init__(self, latent_dim: int = 32) -> None:
        """Initialize the prior."""
        super().__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, inputs: Float[torch.Tensor, "batch 128"]
    ) -> tuple[
        Float[torch.Tensor, "batch latent"],
        Float[torch.Tensor, "batch latent"],
    ]:
        """Predict latent mean and log variance."""
        out = self.act(self.fc1(inputs))
        return self.fc2(out), self.fc3(out)


class Decoder(nn.Module):
    """Conditional decoder network."""

    def __init__(self, output_dim: int, latent_dim: int = 32) -> None:
        """Initialize the decoder."""
        super().__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(128 + latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(
        self,
        inputs: tuple[
            Float[torch.Tensor, "batch 128"],
            Float[torch.Tensor, "batch latent"],
        ],
    ) -> Float[torch.Tensor, "batch output"]:
        """Decode from an embedding and latent tensor."""
        in1, in2 = inputs
        out = self.act(self.fc1(torch.cat((in1, in2), 1)))
        out = self.act(self.fc2(out))
        return self.fc3(out)


class EmbedBbox(nn.Module):
    """Autoregressive box embedding network."""

    def __init__(self, n_class: int) -> None:
        """Initialize the box embedding network."""
        super().__init__()
        self.fcb1 = FCBlock(n_class)
        self.fcb2 = FCBlock(n_class)
        self.seq1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.n_class = n_class
        self.fc = nn.Linear(128 * 3, 128)
        self.lstm = nn.LSTM(n_class + 4, hidden_size=128)

    def forward(
        self,
        inputs: tuple[
            Float[torch.Tensor, "batch internal_labels"],
            Float[torch.Tensor, "batch internal_labels"],
            Float[torch.Tensor, "history batch features"],
        ],
    ) -> Float[torch.Tensor, "batch 128"]:
        """Embed counts, current label, and previous elements."""
        in1, in2, in3 = inputs
        _, (h_0, _c_0) = self.lstm(in3)
        hn = h_0.view(-1, 128)
        return self.fc(torch.cat((self.fcb1(in1), self.fcb2(in2), self.seq1(hn)), 1))


def _sample_diag_gaussian(
    mu: Float[torch.Tensor, "batch latent"],
    logvar: Float[torch.Tensor, "batch latent"],
    *,
    generator: torch.Generator | None,
) -> Float[torch.Tensor, "batch latent"]:
    std = torch.exp(logvar / 2)
    eps = torch.randn(
        std.shape, generator=generator, device=std.device, dtype=std.dtype
    )
    return eps * std + mu


class CountVAEModel(nn.Module):
    """Autoregressive count module."""

    def __init__(self, n_class: int, latent_dim: int = 32) -> None:
        """Initialize the count module."""
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.prior = Prior(latent_dim=latent_dim)
        self.decoder = Decoder(1, latent_dim=latent_dim)
        self.embeder = Embeder(n_class)
        self.n_class = n_class

    def forward(
        self,
        label_set: Float[torch.Tensor, "batch internal_labels"],
        *,
        latents: Float[torch.Tensor, "batch internal_labels latent"] | None = None,
        count_samples: Float[torch.Tensor, "batch internal_labels"] | None = None,
        generator: torch.Generator | None = None,
    ) -> Float[torch.Tensor, "batch internal_labels"]:
        """Generate class counts from a label set."""
        previous_counts = torch.zeros_like(label_set)
        samples = (
            count_samples.to(device=label_set.device, dtype=label_set.dtype)
            if count_samples is not None
            else None
        )
        for i in range(self.n_class):
            current_label = torch.zeros_like(previous_counts)
            x_i = label_set[..., i]
            current_label[..., i] = x_i
            embedding = self.embeder((label_set, current_label, previous_counts))
            mu, logvar = self.prior(embedding)
            z = (
                latents[:, i, :].to(device=label_set.device, dtype=label_set.dtype)
                if latents is not None
                else _sample_diag_gaussian(mu, logvar, generator=generator)
            )
            rate = torch.exp(self.decoder((embedding, z)))
            q = (
                samples[:, i].view(-1, 1)
                if samples is not None
                else torch.poisson(rate, generator=generator)
            )
            previous_counts = previous_counts + current_label * (
                q.view(-1, 1) + x_i.view(-1, 1)
            )
        return previous_counts


class BboxVAEModel(nn.Module):
    """Autoregressive bounding-box module."""

    def __init__(
        self,
        n_class: int,
        n_dim: int,
        max_box: int,
        latent_dim: int = 32,
    ) -> None:
        """Initialize the box module."""
        super().__init__()
        self.embeder = EmbedBbox(n_class)
        self.encoder = Encoder(n_dim, latent_dim=latent_dim)
        self.decoder = Decoder(n_dim, latent_dim=latent_dim)
        self.prior = Prior(latent_dim=latent_dim)
        self.n_dim = n_dim
        self.n_class = n_class
        self.max_box = max_box

    def forward(
        self,
        box_counts: Float[torch.Tensor, "batch internal_labels"],
        box_label: Float[torch.Tensor, "batch elements internal_labels"],
        *,
        latents: Float[torch.Tensor, "batch elements latent"] | None = None,
        output_noise: Float[torch.Tensor, "batch elements 4"] | None = None,
        generator: torch.Generator | None = None,
    ) -> Float[torch.Tensor, "batch elements 4"]:
        """Generate normalized left-top-width-height boxes."""
        boxes = []
        prev_label = torch.zeros(
            (1, box_label.shape[0], self.n_class),
            device=box_label.device,
            dtype=box_label.dtype,
        )
        prev_box = torch.zeros(
            (1, box_label.shape[0], 4),
            device=box_label.device,
            dtype=box_label.dtype,
        )
        for i in range(self.max_box):
            if i == 0:
                prev_label = torch.zeros(
                    (1, *box_label[..., i, :].shape),
                    device=box_label.device,
                    dtype=box_label.dtype,
                )
                prev_box = torch.zeros(
                    (1, box_label.shape[0], 4),
                    device=box_label.device,
                    dtype=box_label.dtype,
                )
            current_label = box_label[..., i, :].view(-1, self.n_class)
            embedding = self.embeder(
                (box_counts, current_label, torch.cat([prev_label, prev_box], dim=2))
            )
            mu, logvar = self.prior(embedding)
            z = (
                latents[:, i, :].to(device=box_label.device, dtype=box_label.dtype)
                if latents is not None
                else _sample_diag_gaussian(mu, logvar, generator=generator)
            )
            decoded = self.decoder((embedding, z))
            if output_noise is None:
                eps = torch.rand(
                    decoded.shape,
                    generator=generator,
                    device=decoded.device,
                    dtype=decoded.dtype,
                )
                box = decoded + eps * 0.02
            else:
                box = decoded + output_noise[:, i, :].to(
                    device=decoded.device, dtype=decoded.dtype
                )
            prev_box = torch.cat([prev_box, torch.unsqueeze(box, 0)])
            prev_label = torch.cat([prev_label, torch.unsqueeze(current_label, 0)])
            boxes.append(box)
        return torch.stack(boxes, dim=1)


class LayoutVAEModel(PreTrainedModel):
    """Transformers-compatible LayoutVAE model.

    Args:
        config: LayoutVAE configuration.

    Examples:
        >>> model = LayoutVAEModel(LayoutVAEConfig())
        >>> model.config.model_type
        'layoutvae'
    """

    config_class = LayoutVAEConfig
    base_model_prefix = "layoutvae"
    supports_gradient_checkpointing = False

    def __init__(self, config: LayoutVAEConfig) -> None:
        """Initialize LayoutVAE submodules."""
        super().__init__(config)
        self.countvae = CountVAEModel(
            config.internal_num_labels, config.count_latent_dim
        )
        self.bboxvae = BboxVAEModel(
            config.internal_num_labels,
            4,
            config.max_position_embeddings,
            config.bbox_latent_dim,
        )
        self.post_init()

    def forward(
        self,
        label_set: Float[torch.Tensor, "batch internal_labels"],
        *,
        count_latents: Float[torch.Tensor, "batch internal_labels latent"]
        | None = None,
        bbox_latents: Float[torch.Tensor, "batch elements latent"] | None = None,
        bbox_noise: Float[torch.Tensor, "batch elements 4"] | None = None,
        class_counts: Float[torch.Tensor, "batch internal_labels"] | None = None,
        count_samples: Float[torch.Tensor, "batch internal_labels"] | None = None,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> LayoutVAEModelOutput | LayoutVAETupleOutput:
        """Run label-conditioned layout generation.

        Args:
            label_set: Six-way label-set tensor.
            count_latents: Optional fixed count latents.
            bbox_latents: Optional fixed box latents.
            bbox_noise: Optional fixed output noise.
            class_counts: Optional fixed six-way class counts.
            count_samples: Optional fixed count samples.
            generator: Optional PyTorch random generator.
            return_dict: Whether to return a dataclass.

        Returns:
            Model output dataclass or tuple.

        Raises:
            ValueError: If shapes are invalid.

        Examples:
            >>> model = LayoutVAEModel(LayoutVAEConfig())
            >>> label_set = torch.tensor([[0, 1, 0, 0, 0, 1]], dtype=torch.float32)
            >>> out = model(label_set, class_counts=torch.tensor([[7, 1, 0, 0, 0, 1.]]))
            >>> tuple(out.bbox.shape)
            (1, 9, 4)
        """
        label_set = label_set.to(device=self.device, dtype=self.dtype)
        if label_set.ndim != 2 or label_set.shape[1] != self.config.internal_num_labels:
            raise ValueError(
                "label_set must have shape (batch, config.internal_num_labels)"
            )
        if class_counts is None:
            class_counts = self.countvae(
                label_set,
                latents=count_latents,
                count_samples=count_samples,
                generator=generator,
            )
            class_counts = self._normalize_counts(class_counts)
        else:
            class_counts = class_counts.to(
                device=label_set.device, dtype=label_set.dtype
            )
            if class_counts.shape != label_set.shape:
                raise ValueError("class_counts must match label_set shape")
        class_labels = self._labels_from_counts(class_counts)
        raw_ltwh = self.bboxvae(
            class_counts,
            class_labels,
            latents=bbox_latents,
            output_noise=bbox_noise,
            generator=generator,
        )
        internal_ids = torch.argmax(class_labels, dim=2).to(dtype=torch.long)
        labels = torch.clamp(internal_ids - 1, min=0)
        mask = internal_ids != INTERNAL_EMPTY_LABEL_ID
        public_bbox = clamp_boxes(ltwh_to_xywh(raw_ltwh))
        raw_ltwh = torch.where(mask.unsqueeze(-1), raw_ltwh, torch.zeros_like(raw_ltwh))
        public_bbox = torch.where(
            mask.unsqueeze(-1), public_bbox, torch.zeros_like(public_bbox)
        )
        if not return_dict:
            return raw_ltwh, public_bbox, labels, mask, class_counts, internal_ids
        return LayoutVAEModelOutput(
            raw_ltwh=raw_ltwh,
            bbox=public_bbox,
            labels=labels,
            mask=mask,
            class_counts=class_counts,
            label_set=label_set,
            internal_labels=internal_ids,
        )

    def _normalize_counts(
        self,
        class_counts: Float[torch.Tensor, "batch internal_labels"],
    ) -> Float[torch.Tensor, "batch internal_labels"]:
        counts = class_counts.clamp_min(0)
        denom = counts.sum(dim=1, keepdim=True).clamp_min(1)
        counts = torch.floor(self.config.max_position_embeddings * (counts / denom))
        totals = counts.sum(dim=1)
        shortfall = self.config.max_position_embeddings - totals
        counts[:, INTERNAL_EMPTY_LABEL_ID] = counts[
            :, INTERNAL_EMPTY_LABEL_ID
        ] + torch.clamp(shortfall, min=0)
        return counts

    def _labels_from_counts(
        self,
        class_counts: Float[torch.Tensor, "batch internal_labels"],
    ) -> Float[torch.Tensor, "batch elements internal_labels"]:
        labels = torch.zeros(
            (
                class_counts.shape[0],
                self.config.max_position_embeddings,
                self.config.internal_num_labels,
            ),
            device=class_counts.device,
            dtype=class_counts.dtype,
        )
        for batch_index, counts in enumerate(class_counts):
            position = 0
            for class_index in reversed(range(self.config.internal_num_labels)):
                count = int(counts[class_index].item())
                for _ in range(count):
                    if position >= self.config.max_position_embeddings:
                        break
                    labels[batch_index, position, class_index] = 1.0
                    position += 1
        return labels
