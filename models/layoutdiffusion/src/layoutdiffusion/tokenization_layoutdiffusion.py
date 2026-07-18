"""PreTrainedTokenizer for LayoutDiffusion layout token sequences."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
from transformers import PreTrainedTokenizer

from laygen.common.bbox import (
    BoxFormat,
    clamp_boxes,
    ltwh_to_xywh,
    ltrb_to_xywh,
    normalize_boxes,
    normalize_box_format,
    xywh_to_ltrb,
)

from .configuration_layoutdiffusion import LayoutDiffusionConfig
from .labels import public_id_to_vendor_label, vendor_label_to_public_id


class LayoutDiffusionTokenizer(PreTrainedTokenizer):
    """Tokenizer backed by the original LayoutDiffusion ``vocab.json``.

    Args:
        config: LayoutDiffusion config or serialized config mapping.
        vocab_file: Optional saved vocabulary file.
        layout_config_file: Optional saved layout config file.
        **kwargs: Extra ``PreTrainedTokenizer`` keyword arguments.

    Raises:
        ValueError: If required tokenizer files are absent.

    Examples:
        >>> tok = LayoutDiffusionTokenizer(
        ...     LayoutDiffusionConfig(dataset_name="publaynet")
        ... )
        >>> tok.mask_token
        'MASK'
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "layout_config_file": "layout_config.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        config: LayoutDiffusionConfig | Mapping[str, object] | None = None,
        *,
        vocab_file: str | Path | None = None,
        layout_config_file: str | Path | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the tokenizer."""
        if isinstance(config, LayoutDiffusionConfig):
            pass
        elif config is None:
            config = self._load_config(
                layout_config_file=layout_config_file, kwargs=kwargs
            )
        else:
            config = _config_from_mapping(config)
        if vocab_file is not None and Path(vocab_file).exists():
            raw_vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
            config.vocab = {str(k): int(v) for k, v in raw_vocab.items()}
            if "MASK" not in config.vocab:
                config.vocab["MASK"] = config.vocab_size - 1
            config.vocab_size = max(config.vocab.values()) + 1
        self.config = config
        self._token_to_id = dict(config.vocab)
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        super().__init__(
            pad_token=kwargs.pop("pad_token", "PAD"),
            mask_token=kwargs.pop("mask_token", "MASK"),
            unk_token=kwargs.pop("unk_token", "UNK"),
            model_max_length=kwargs.pop("model_max_length", config.max_token_length),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return full vocabulary size."""
        return self.config.vocab_size

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of token-to-id vocabulary."""
        return dict(self._token_to_id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        """Split a vendor layout token string on whitespace."""
        _ = kwargs
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        """Convert one token string to id."""
        return self._token_to_id.get(token, self.config.special_token_ids["UNK"])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert one token id to string."""
        return self._id_to_token.get(int(index), "UNK")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join LayoutDiffusion tokens for debugging or parity fixtures."""
        return " ".join(tokens)

    def __call__(
        self,
        *,
        bbox: torch.Tensor | list[object],
        labels: torch.Tensor | list[object],
        mask: torch.Tensor | list[object] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode layout tensors into token ids.

        Args:
            bbox: Layout boxes.
            labels: Dataset-local labels.
            mask: Optional valid-element mask.
            box_format: Format of ``bbox``.
            normalized: Whether boxes are already normalized.
            canvas_size: Pixel canvas size for unnormalized boxes.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, and ``mask``.
        """
        return self.encode_layout(
            bbox=torch.as_tensor(bbox),
            labels=torch.as_tensor(labels),
            mask=None if mask is None else torch.as_tensor(mask),
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )

    def encode_layout(
        self,
        *,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode public layout tensors into vendor token ids.

        Args:
            bbox: Boxes shaped ``(B, S, 4)``.
            labels: Labels shaped ``(B, S)``.
            mask: Optional valid mask shaped ``(B, S)``.
            box_format: Input box format.
            normalized: Whether input boxes are normalized.
            canvas_size: Pixel canvas size when ``normalized`` is false.

        Returns:
            Encoded token tensors.

        Raises:
            ValueError: If unnormalized boxes omit ``canvas_size``.
        """
        if bbox.ndim == 2:
            bbox = bbox.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        bbox = bbox.float()
        if normalized:
            fmt = normalize_box_format(box_format)
            if fmt is BoxFormat.xywh:
                xywh = clamp_boxes(bbox)
            elif fmt is BoxFormat.ltrb:
                xywh = clamp_boxes(ltrb_to_xywh(bbox))
            else:
                xywh = clamp_boxes(ltwh_to_xywh(bbox))
        else:
            if canvas_size is None:
                raise ValueError("canvas_size is required when normalized=False")
            xywh = normalize_boxes(bbox, canvas_size=canvas_size, box_format=box_format)
        ltrb_ids = (xywh_to_ltrb(xywh).clamp(0.0, 1.0) * 127).round().long()
        batch_size = labels.shape[0]
        input_ids = torch.full(
            (batch_size, self.config.max_token_length),
            self.config.pad_token_id,
            dtype=torch.long,
        )
        input_ids[:, 0] = self.config.special_token_ids["START"]
        for batch_idx in range(batch_size):
            valid_positions = torch.nonzero(
                mask[batch_idx].bool(), as_tuple=False
            ).flatten()
            valid_positions = valid_positions[: self.config.max_num_elements]
            cursor = 1
            for elem_idx, source_idx in enumerate(valid_positions.tolist()):
                if elem_idx > 0:
                    input_ids[batch_idx, cursor] = self.config.special_token_ids["|"]
                    cursor += 1
                label = public_id_to_vendor_label(
                    self.config.dataset_name, int(labels[batch_idx, source_idx].item())
                )
                token_ids = [self._token_to_id[label]]
                token_ids.extend(
                    self._token_to_id[str(int(v))]
                    for v in ltrb_ids[batch_idx, source_idx].tolist()
                )
                input_ids[batch_idx, cursor : cursor + 5] = torch.tensor(token_ids)
                cursor += 5
            if cursor < self.config.max_token_length:
                input_ids[batch_idx, cursor] = self.config.special_token_ids["END"]
        attention_mask = input_ids.ne(self.config.pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask.bool(),
        }

    def decode_layout(
        self,
        input_ids: torch.Tensor,
        *,
        output_box_format: Literal["xywh", "ltrb"] = "xywh",
    ) -> dict[str, torch.Tensor]:
        """Decode token ids into public layout tensors.

        Args:
            input_ids: Token ids shaped ``(B, L)``.
            output_box_format: ``"xywh"`` or ``"ltrb"``.

        Returns:
            Dictionary with ``bbox``, ``labels``, and ``mask``.

        Raises:
            ValueError: If ``output_box_format`` is unsupported.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_boxes = []
        batch_labels = []
        batch_masks = []
        for row in input_ids.cpu().long():
            tokens = [self._convert_id_to_token(int(idx)) for idx in row.tolist()]
            elements = self._parse_elements(tokens)
            boxes = torch.zeros(self.config.max_num_elements, 4, dtype=torch.float32)
            labels = torch.zeros(self.config.max_num_elements, dtype=torch.long)
            masks = torch.zeros(self.config.max_num_elements, dtype=torch.bool)
            for i, element in enumerate(elements[: self.config.max_num_elements]):
                label, *coords = element
                labels[i] = vendor_label_to_public_id(self.config.dataset_name, label)
                ltrb = torch.tensor([int(v) for v in coords], dtype=torch.float32) / 127
                boxes[i] = ltrb_to_xywh(ltrb) if output_box_format == "xywh" else ltrb
                masks[i] = True
            batch_boxes.append(clamp_boxes(boxes))
            batch_labels.append(labels)
            batch_masks.append(masks)
        if output_box_format not in {"xywh", "ltrb"}:
            raise ValueError(f"Unsupported output_box_format: {output_box_format}")
        return {
            "bbox": torch.stack(batch_boxes, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
            "mask": torch.stack(batch_masks, dim=0),
        }

    def build_initial_tokens(
        self,
        *,
        batch_size: int,
        num_elements: torch.Tensor | list[int] | int | None = None,
        labels: torch.Tensor | None = None,
        condition_type: str = "unconditional",
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the vendor sampling start template.

        Args:
            batch_size: Number of samples.
            num_elements: Optional element counts in ``[1, 20]``.
            labels: Optional labels for label-conditioned generation.
            condition_type: Canonical condition name.
            generator: Optional random generator.
            device: Output device.

        Returns:
            Initial token ids shaped ``(B, 121)``.
        """
        device = device or torch.device("cpu")
        if num_elements is None:
            prior = torch.tensor(
                self.config.element_count_prior, dtype=torch.float32, device=device
            )
            counts = (
                torch.multinomial(
                    prior, batch_size, replacement=True, generator=generator
                )
                + 1
            )
        else:
            counts = torch.as_tensor(num_elements, dtype=torch.long, device=device)
            if counts.ndim == 0:
                counts = counts.expand(batch_size)
        counts = counts.clamp(1, self.config.max_num_elements)
        input_ids = torch.full(
            (batch_size, self.config.max_token_length),
            self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_id = self.config.mask_token_id
        start = self.config.special_token_ids["START"]
        sep = self.config.special_token_ids["|"]
        end = self.config.special_token_ids["END"]
        for batch_idx in range(batch_size):
            n = int(counts[batch_idx].item())
            tokens = [start, mask_id, mask_id, mask_id, mask_id, mask_id]
            for _ in range(n - 1):
                tokens.extend([sep, mask_id, mask_id, mask_id, mask_id, mask_id])
            tokens.append(end)
            input_ids[batch_idx, : len(tokens)] = torch.tensor(tokens, device=device)
        if condition_type == "label" and labels is not None:
            label_ids = torch.as_tensor(labels, dtype=torch.long, device=device)
            for batch_idx in range(batch_size):
                for elem_idx in range(min(label_ids.shape[1], int(counts[batch_idx]))):
                    pos = 1 + elem_idx * 6
                    label = public_id_to_vendor_label(
                        self.config.dataset_name, int(label_ids[batch_idx, elem_idx])
                    )
                    input_ids[batch_idx, pos] = self._token_to_id[label]
                coord_noise = (
                    torch.randint(
                        self.config.num_coordinate_bins,
                        input_ids.shape,
                        generator=generator,
                        device=device,
                    )
                    + self.config.coordinate_token_offset
                )
                coord_positions = torch.zeros_like(input_ids, dtype=torch.bool)
                for elem_idx in range(self.config.max_num_elements):
                    start_pos = 2 + elem_idx * 6
                    coord_positions[:, start_pos : start_pos + 4] = True
                input_ids = torch.where(
                    coord_positions & input_ids.eq(mask_id), coord_noise, input_ids
                )
        return input_ids

    def token_ids_to_text(self, input_ids: torch.Tensor) -> list[str]:
        """Convert token ids to vendor text lines."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return [
            " ".join(self._convert_id_to_token(int(idx)) for idx in row.tolist())
            for row in input_ids.cpu().long()
        ]

    def text_to_token_ids(self, lines: list[str]) -> torch.Tensor:
        """Convert vendor text lines into padded token ids."""
        rows = []
        for line in lines:
            ids = [self._convert_token_to_id(token) for token in line.strip().split()]
            ids = ids[: self.config.max_token_length]
            ids.extend(
                [self.config.pad_token_id] * (self.config.max_token_length - len(ids))
            )
            rows.append(torch.tensor(ids, dtype=torch.long))
        return torch.stack(rows, dim=0)

    def save_vocabulary(
        self, save_directory: str | Path, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        """Save vocabulary and layout config files.

        Args:
            save_directory: Target directory.
            filename_prefix: Optional Transformers filename prefix.

        Returns:
            Saved file paths.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        prefix = "" if filename_prefix is None else f"{filename_prefix}-"
        vocab_file = save_path / f"{prefix}vocab.json"
        config_file = save_path / f"{prefix}layout_config.json"
        vocab_file.write_text(
            json.dumps(self._token_to_id, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        data = dict(self.config.config)
        data["id2label"] = {str(k): v for k, v in self.config.id2label.items()}
        data["vocab"] = self._token_to_id
        config_file.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        return (str(vocab_file), str(config_file))

    @classmethod
    def from_pretrained(
        cls,
        path: str | PathLike[str],
        *args: object,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: object,
    ) -> "LayoutDiffusionTokenizer":
        """Load tokenizer from a pipeline root or tokenizer directory."""
        path = Path(path)
        if (path / "tokenizer").is_dir():
            path = path / "tokenizer"
        return super().from_pretrained(
            path,
            *args,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )

    @classmethod
    def _load_config(
        cls,
        *,
        layout_config_file: str | Path | None,
        kwargs: dict[str, object],
    ) -> LayoutDiffusionConfig:
        layout_config = kwargs.pop("layout_config", None)
        if layout_config is None:
            if layout_config_file is None:
                raise ValueError("LayoutDiffusionTokenizer requires layout_config_file")
            layout_config = json.loads(
                Path(layout_config_file).read_text(encoding="utf-8")
            )
        if not isinstance(layout_config, Mapping):
            raise TypeError("layout_config must be a mapping")
        return _config_from_mapping(
            {str(key): value for key, value in layout_config.items()}
        )

    def _parse_elements(self, tokens: list[str]) -> list[list[str]]:
        start = tokens.index("START") if "START" in tokens else -1
        end = tokens.index("END") if "END" in tokens else 0
        if end <= start:
            end = max(
                (i for i, token in enumerate(tokens) if token == "|"), default=end
            )
        payload = tokens[start + 1 : end] if end > start else []
        groups: list[list[str]] = []
        current: list[str] = []
        for token in payload:
            if token == "|":
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(token)
        if current:
            groups.append(current)
        elements = []
        for group in groups:
            if len(group) >= 5 and all(token.isdigit() for token in group[-4:]):
                label = group[-5]
                if label in self.config.label2id:
                    elements.append([label, *group[-4:]])
        return elements


def _config_from_mapping(mapping: Mapping[str, object]) -> LayoutDiffusionConfig:
    return LayoutDiffusionConfig(**dict(mapping))
