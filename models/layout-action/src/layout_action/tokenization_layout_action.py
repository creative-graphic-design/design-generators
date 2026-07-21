"""Synthetic action-token tokenizer for LayoutAction."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Final, TypedDict, cast

import torch
from transformers import PreTrainedTokenizer

from .configuration_layout_action import LayoutActionConfig

TOKENIZER_CONFIG_FILE: Final[str] = "layout_action_tokenizer_config.json"


class DecodedActions(TypedDict):
    """Decoded action-token details."""

    option: torch.Tensor
    object: torch.Tensor
    value: torch.Tensor


class LayoutActionTokenizer(PreTrainedTokenizer):
    """PreTrainedTokenizer for LayoutAction's 13-token element grammar.

    Args:
        config: LayoutAction config carrying vocabulary metadata.
        tokenizer_config_file: Optional saved tokenizer metadata path.
        kwargs: Standard tokenizer keyword arguments.

    Examples:
        >>> tokenizer = LayoutActionTokenizer(LayoutActionConfig(max_elements=2))
        >>> tokenizer.bos_token_id == tokenizer.config.bos_token_id
        True
    """

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"tokenizer_config_file": TOKENIZER_CONFIG_FILE}

    def __init__(
        self,
        config: LayoutActionConfig | None = None,
        tokenizer_config_file: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize synthetic token strings."""
        if config is None and tokenizer_config_file is not None:
            with Path(tokenizer_config_file).open(encoding="utf-8") as f:
                config = LayoutActionConfig(**json.load(f)["config"])
        self.config = config or LayoutActionConfig()
        self._token2id = self._build_vocab()
        self._id2token = {idx: token for token, idx in self._token2id.items()}
        kwargs.setdefault("bos_token", "[BOS]")
        kwargs.setdefault("eos_token", "[EOS]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("model_max_length", self.config.max_token_length + 1)
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        """Return the LayoutAction vocabulary size."""
        return self.config.vocab_size

    def _build_vocab(self) -> dict[str, int]:
        vocab = {f"value:{idx}": idx for idx in range(self.config.size)}
        vocab["[NO_VALUE]"] = self.config.no_value_token_id
        id2label = cast(dict[int, str], self.config.id2label)
        for label_id, label in id2label.items():
            vocab[f"label:{label_id}:{label}"] = self.config.label_token_id(
                int(label_id)
            )
        vocab["[COPY]"] = self.config.copy_token_id
        vocab["[MARGIN]"] = self.config.margin_token_id
        vocab["[GENERATE]"] = self.config.generate_token_id
        vocab["[NO_OBJ]"] = self.config.no_obj_token_id
        for idx in range(1, self.config.max_elements + 1):
            vocab[f"obj:{idx}"] = self.config.object_token_id(idx)
        vocab["[BOS]"] = self.config.bos_token_id
        vocab["[EOS]"] = self.config.eos_token_id
        vocab["[PAD]"] = self.config.pad_token_id
        vocab["[UNK]"] = self.config.pad_token_id
        return vocab

    def get_vocab(self) -> dict[str, int]:
        """Return synthetic token strings mapped to ids."""
        return dict(self._token2id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        _ = kwargs
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self.config.pad_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id2token.get(int(index), "[UNK]")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join synthetic layout tokens."""
        return " ".join(tokens)

    def save_vocabulary(
        self, save_directory: str | PathLike[str], filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        """Save tokenizer metadata."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (
            TOKENIZER_CONFIG_FILE
            if filename_prefix is None
            else f"{filename_prefix}-{TOKENIZER_CONFIG_FILE}"
        )
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump({"config": self.config.to_dict()}, f, indent=2, sort_keys=True)
        return (str(path),)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *inputs: object,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: object,
    ) -> "LayoutActionTokenizer":
        """Load tokenizer metadata from a local checkpoint directory.

        Args:
            pretrained_model_name_or_path: Local tokenizer directory.
            inputs: Reserved tokenizer inputs.
            cache_dir: Accepted for API compatibility.
            force_download: Accepted for API compatibility.
            local_files_only: Accepted for API compatibility.
            token: Accepted for API compatibility.
            revision: Accepted for API compatibility.
            kwargs: Standard tokenizer keyword arguments.

        Returns:
            Loaded LayoutAction tokenizer.
        """
        _ = (
            inputs,
            cache_dir,
            force_download,
            local_files_only,
            token,
            revision,
            kwargs,
        )
        root = Path(pretrained_model_name_or_path)
        metadata = root / TOKENIZER_CONFIG_FILE
        with metadata.open(encoding="utf-8") as f:
            config = LayoutActionConfig(**json.load(f)["config"])
        return cls(config=config)

    def quantize_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """Quantize normalized center ``xywh`` boxes with vendor binning."""
        return (
            bbox.clamp(0.0, 1.0)
            .mul(self.config.size - 1)
            .round()
            .long()
            .clamp(0, self.config.size - 1)
        )

    def continuize_bbox(self, quantized_bbox: torch.Tensor) -> torch.Tensor:
        """Decode quantized boxes to normalized center ``xywh`` values."""
        return quantized_bbox.float().clamp(0, self.config.size - 1) / (
            self.config.size - 1
        )

    def encode_layout(
        self,
        *,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode public normalized layouts to padded action-token sequences.

        Args:
            bbox: Normalized center ``xywh`` boxes shaped ``(B, E, 4)``.
            labels: Dataset-local labels shaped ``(B, E)``.
            mask: Valid-element mask shaped ``(B, E)``.

        Returns:
            Token ids shaped ``(B, max_token_length + 1)``.
        """
        quantized_bbox = self.quantize_bbox(bbox)
        return self.encode_action_layout(
            quantized_bbox=quantized_bbox,
            labels=labels.long(),
            mask=mask.bool(),
        )

    def encode_action_layout(
        self,
        *,
        quantized_bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode already quantized boxes to action tokens."""
        batch = quantized_bbox.shape[0]
        sequences = torch.full(
            (batch, self.config.max_token_length + 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=quantized_bbox.device,
        )
        sequences[:, 0] = self.config.bos_token_id
        for batch_idx in range(batch):
            cursor = 1
            valid = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
            for elem_idx in valid[: self.config.max_elements]:
                label_id = int(labels[batch_idx, elem_idx].item())
                sequences[batch_idx, cursor] = self.config.label_token_id(label_id)
                cursor += 1
                for geo_idx in range(4):
                    sequences[batch_idx, cursor] = self.config.generate_token_id
                    sequences[batch_idx, cursor + 1] = self.config.no_obj_token_id
                    sequences[batch_idx, cursor + 2] = quantized_bbox[
                        batch_idx, elem_idx, geo_idx
                    ]
                    cursor += 3
            sequences[batch_idx, cursor] = self.config.eos_token_id
        return sequences

    def decode_layout(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode action-token sequences to public layout tensors."""
        return cast(
            dict[str, torch.Tensor],
            self.decode_action_tokens(input_ids, return_actions=False),
        )

    def decode_action_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        return_actions: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Decode action tokens and optionally return raw action details."""
        ids = input_ids.long()
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        batch = ids.shape[0]
        bbox = torch.zeros(batch, self.config.max_elements, 4, dtype=torch.float32)
        labels = torch.zeros(batch, self.config.max_elements, dtype=torch.long)
        mask = torch.zeros(batch, self.config.max_elements, dtype=torch.bool)
        option = torch.full((batch, self.config.max_elements, 4), -1, dtype=torch.long)
        obj = torch.full_like(option, -1)
        value = torch.full_like(option, -1)
        for batch_idx in range(batch):
            tokens = self._trim_special_tokens(ids[batch_idx])
            usable = tokens[
                : (tokens.numel() // self.config.element_token_width)
                * self.config.element_token_width
            ]
            boxes: list[torch.Tensor] = []
            out_idx = 0
            for start in range(0, usable.numel(), self.config.element_token_width):
                if out_idx >= self.config.max_elements:
                    break
                element = usable[start : start + self.config.element_token_width]
                label_id = self.config.label_id_from_token(int(element[0]))
                if label_id is None:
                    continue
                qbox = torch.zeros(4, dtype=torch.long)
                valid = True
                for geo_idx in range(4):
                    triple = element[1 + geo_idx * 3 : 1 + (geo_idx + 1) * 3]
                    opt_id = int(triple[0])
                    obj_id = int(triple[1])
                    val_id = int(triple[2])
                    option[batch_idx, out_idx, geo_idx] = opt_id
                    obj[batch_idx, out_idx, geo_idx] = obj_id
                    value[batch_idx, out_idx, geo_idx] = val_id
                    if (
                        opt_id == self.config.generate_token_id
                        and 0 <= val_id < self.config.size
                    ):
                        qbox[geo_idx] = val_id
                    elif opt_id == self.config.copy_token_id:
                        ref = self.config.back_reference_from_token(obj_id)
                        if ref is None or ref > len(boxes):
                            valid = False
                            break
                        qbox[geo_idx] = boxes[-ref][geo_idx].long()
                    elif opt_id == self.config.margin_token_id and geo_idx < 2:
                        ref = self.config.back_reference_from_token(obj_id)
                        if ref is None or ref > len(boxes):
                            valid = False
                            break
                        base = self.continuize_bbox(boxes[-ref].unsqueeze(0))[0]
                        cur = self.continuize_bbox(qbox.unsqueeze(0))[0]
                        margin = float(val_id) / (self.config.size - 1)
                        coord = (
                            base[geo_idx]
                            + 0.5 * base[geo_idx + 2]
                            + 0.5 * cur[geo_idx + 2]
                            + margin
                        )
                        qbox[geo_idx] = int(
                            round(float(coord.item()) * (self.config.size - 1))
                        )
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                boxes.append(qbox.clone())
                bbox[batch_idx, out_idx] = self.continuize_bbox(qbox)
                labels[batch_idx, out_idx] = label_id
                mask[batch_idx, out_idx] = True
                out_idx += 1
        result: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "bbox": bbox.clamp(0.0, 1.0),
            "labels": labels,
            "mask": mask,
        }
        if return_actions:
            result["actions"] = {"option": option, "object": obj, "value": value}
        return result

    def _trim_special_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.detach().cpu()
        bos = torch.nonzero(tokens == self.config.bos_token_id, as_tuple=False)
        if bos.numel() > 0:
            tokens = tokens[int(bos[0].item()) + 1 :]
        eos = torch.nonzero(tokens == self.config.eos_token_id, as_tuple=False)
        if eos.numel() > 0:
            tokens = tokens[: int(eos[0].item())]
        return tokens[tokens != self.config.pad_token_id]
