"""Numeric layout tokenizer for RALF."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Final, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, PreTrainedTokenizer

from .configuration_ralf import RalfConfig, VENDOR_GEO_KEYS

TOKENIZER_CONFIG_FILE: Final[str] = "ralf_tokenizer_config.json"
GEO_KEYS: Final[tuple[str, ...]] = VENDOR_GEO_KEYS


class RalfLayoutTokenizer(PreTrainedTokenizer):
    """PreTrainedTokenizer for RALF's numeric layout token sequences.

    Args:
        config: RALF config that defines label and geometry vocabularies.
        tokenizer_config_file: Optional tokenizer metadata path loaded by
            `from_pretrained`.
        kwargs: Standard `PreTrainedTokenizer` keyword arguments.

    Examples:
        >>> tokenizer = RalfLayoutTokenizer(RalfConfig(max_seq_length=2))
        >>> encoded = tokenizer.encode_layout(
        ...     labels=torch.tensor([[0]]),
        ...     bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        ...     mask=torch.tensor([[True]]),
        ... )
        >>> encoded["input_ids"].shape[1] > 1
        True
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"tokenizer_config_file": TOKENIZER_CONFIG_FILE}

    def __init__(
        self,
        config: RalfConfig | None = None,
        tokenizer_config_file: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize tokenizer metadata and synthetic token strings."""
        if config is None and tokenizer_config_file is not None:
            with Path(tokenizer_config_file).open() as f:
                config = RalfConfig(**json.load(f)["config"])
        self.config = config or RalfConfig()
        self._token2id = self._build_vocab()
        self._id2token = {idx: token for token, idx in self._token2id.items()}
        kwargs.setdefault("pad_token", "[pad]")
        kwargs.setdefault("bos_token", "[bos]")
        kwargs.setdefault("eos_token", "[eos]")
        kwargs.setdefault("unk_token", "[unk]")
        kwargs.setdefault("model_max_length", self.config.max_token_length + 1)
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return self.config.vocab_size

    def get_vocab(self) -> dict[str, int]:
        """Return synthetic token strings mapped to ids."""
        return dict(self._token2id)

    def _build_vocab(self) -> dict[str, int]:
        id2label = cast(dict[int, str], self.config.id2label)
        vocab = {f"label:{label}": int(idx) for idx, label in id2label.items()}
        for key in GEO_KEYS:
            start = self.config.bbox_token_offset(key)
            for idx in range(self.config.num_bin):
                vocab[f"{key}:{idx}"] = start + idx
        for token in self.config.special_tokens:
            vocab[f"[{token}]"] = self.config.special_token_id(token)
        return vocab

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        _ = kwargs
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self.config.pad_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id2token.get(int(index), "[unk]")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join synthetic layout tokens."""
        return " ".join(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        """Save tokenizer metadata for Hub-compatible loading."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (
            TOKENIZER_CONFIG_FILE
            if filename_prefix is None
            else f"{filename_prefix}-{TOKENIZER_CONFIG_FILE}"
        )
        path = out_dir / name
        with path.open("w") as f:
            json.dump({"config": self.config.to_dict()}, f, indent=2, sort_keys=True)
        return (str(path),)

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> tuple[str, ...]:
        """Save tokenizer files and the paired `RalfConfig`.

        Args:
            save_directory: Output directory.
            legacy_format: Standard tokenizer save flag.
            filename_prefix: Optional filename prefix.
            push_to_hub: Whether to push through Hugging Face Hub helpers.
            kwargs: Reserved tokenizer save arguments.

        Returns:
            Written tokenizer file paths.
        """
        paths = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        self.config.save_pretrained(save_directory)
        return paths

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *inputs: object,
        config: RalfConfig | None = None,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
    ) -> "RalfLayoutTokenizer":
        """Load tokenizer metadata from a checkpoint directory."""
        if config is None:
            config = RalfConfig.from_pretrained(pretrained_model_name_or_path)
        loaded = super().from_pretrained(
            pretrained_model_name_or_path,
            *inputs,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            config=config,
        )
        return cast("RalfLayoutTokenizer", loaded)

    def _quantize(self, values: Float[torch.Tensor, "..."]) -> Int[torch.Tensor, "..."]:
        values = values.clamp(0.0, 1.0)
        boundaries = (
            torch.arange(
                1,
                self.config.num_bin + 1,
                device=values.device,
                dtype=values.dtype,
            )
            / self.config.num_bin
        )
        return torch.bucketize(values, boundaries).long()

    def _dequantize(self, ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        ids = ids.clamp(0, self.config.num_bin - 1)
        starts = ids.float() / self.config.num_bin
        return starts + (0.5 / self.config.num_bin)

    def encode_layout(
        self,
        *,
        labels: Int[torch.Tensor, "batch elements"],
        bbox: Float[torch.Tensor, "batch elements 4"],
        mask: Bool[torch.Tensor, "batch elements"] | None = None,
    ) -> BatchEncoding:
        """Encode public normalized center `xywh` layouts to RALF tokens.

        Args:
            labels: Dataset-local integer labels.
            bbox: Normalized center `xywh` boxes.
            mask: Valid-element mask. If omitted, every element is valid.

        Returns:
            BatchEncoding with `input_ids` and `attention_mask`.

        Raises:
            ValueError: If tensor ranks are invalid.
        """
        if labels.ndim != 2 or bbox.ndim != 3 or bbox.shape[-1] != 4:
            raise ValueError("labels must be (B,S) and bbox must be (B,S,4)")
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        batch, elements = labels.shape
        max_elements = min(elements, self.config.max_seq_length)
        seq = labels.new_full(
            (batch, self.config.max_token_length),
            self.config.pad_token_id,
        )
        attention_mask = torch.zeros_like(seq, dtype=torch.bool)
        geometry = {
            "center_x": self._quantize(bbox[..., 0]),
            "center_y": self._quantize(bbox[..., 1]),
            "width": self._quantize(bbox[..., 2]),
            "height": self._quantize(bbox[..., 3]),
        }
        for element_idx in range(max_elements):
            for var_idx, key in enumerate(self.config.var_order):
                token_idx = element_idx * len(self.config.var_order) + var_idx
                valid = mask[:, element_idx]
                if key == "label":
                    values = labels[:, element_idx].clamp(0, self.config.num_labels - 1)
                else:
                    values = geometry[key][
                        :, element_idx
                    ] + self.config.bbox_token_offset(key)
                seq[:, token_idx] = torch.where(valid, values, seq[:, token_idx])
                attention_mask[:, token_idx] = valid
        lengths = mask[:, :max_elements].sum(dim=1) * len(self.config.var_order)
        for batch_idx, length in enumerate(lengths.tolist()):
            if length < seq.size(1):
                seq[batch_idx, length] = self.config.eos_token_id
                attention_mask[batch_idx, length] = True
        bos = labels.new_full((batch, 1), self.config.bos_token_id)
        bos_mask = torch.ones((batch, 1), dtype=torch.bool, device=labels.device)
        return BatchEncoding(
            {
                "input_ids": torch.cat([bos, seq], dim=1),
                "attention_mask": torch.cat([bos_mask, attention_mask], dim=1),
            }
        )

    def decode_layout(
        self, sequences: Int[torch.Tensor, "batch tokens"]
    ) -> dict[str, torch.Tensor]:
        """Decode RALF token ids to normalized layout tensors.

        Args:
            sequences: Generated token ids, with or without a leading BOS.

        Returns:
            Dictionary containing `bbox`, `labels`, and `mask`.
        """
        if sequences.ndim != 2:
            raise ValueError("sequences must have shape (B,T)")
        if sequences.size(1) and torch.all(sequences[:, 0] == self.config.bos_token_id):
            sequences = sequences[:, 1:]
        usable = sequences[:, : self.config.max_token_length]
        batch = usable.size(0)
        padded = usable.new_full(
            (batch, self.config.max_token_length),
            self.config.pad_token_id,
        )
        padded[:, : usable.size(1)] = usable
        tokens = padded.reshape(
            batch, self.config.max_seq_length, len(self.config.var_order)
        )
        labels = torch.zeros(
            (batch, self.config.max_seq_length),
            dtype=torch.long,
            device=sequences.device,
        )
        bbox_parts = {
            key: torch.zeros_like(labels, dtype=torch.float32) for key in GEO_KEYS
        }
        mask = torch.ones_like(labels, dtype=torch.bool)
        for var_idx, key in enumerate(self.config.var_order):
            values = tokens[..., var_idx]
            if key == "label":
                labels = values.clamp(0, self.config.num_labels - 1)
                mask &= values.lt(self.config.num_labels)
                eos_seen = torch.cumsum(values.eq(self.config.eos_token_id), dim=1) > 0
                mask &= ~eos_seen
            else:
                local = values - self.config.bbox_token_offset(key)
                mask &= (local >= 0) & (local < self.config.num_bin)
                bbox_parts[key] = self._dequantize(local)
        bbox = torch.stack(
            (
                bbox_parts["center_x"],
                bbox_parts["center_y"],
                bbox_parts["width"],
                bbox_parts["height"],
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        labels = torch.where(mask, labels, torch.zeros_like(labels))
        bbox = torch.where(mask.unsqueeze(-1), bbox, torch.zeros_like(bbox))
        return {"bbox": bbox, "labels": labels, "mask": mask}

    def token_mask(
        self, device: torch.device | None = None
    ) -> Bool[torch.Tensor, "tokens vocab"]:
        """Return valid-token masks by sequence position."""
        masks: list[torch.Tensor] = []
        for _ in range(self.config.max_seq_length):
            for key in self.config.var_order:
                mask = torch.zeros(
                    self.config.vocab_size, dtype=torch.bool, device=device
                )
                if key == "label":
                    mask[: self.config.num_labels] = True
                    mask[self.config.eos_token_id] = True
                    mask[self.config.pad_token_id] = True
                else:
                    start = self.config.bbox_token_offset(key)
                    mask[start : start + self.config.num_bin] = True
                    mask[self.config.eos_token_id] = True
                    mask[self.config.pad_token_id] = True
                masks.append(mask)
        return torch.stack(masks, dim=0)
