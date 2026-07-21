"""Synthetic action-token tokenizer for LayoutAction."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Final, TypedDict, cast

import torch
from transformers import PreTrainedTokenizer
from transformers.utils import cached_file

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
        """Load tokenizer metadata through the standard Transformers resolver.

        Args:
            pretrained_model_name_or_path: Local tokenizer directory or Hub repo id.
            inputs: Reserved tokenizer inputs.
            cache_dir: Cache directory for Hub-backed files.
            force_download: Whether to refresh cached files.
            local_files_only: Whether to disable network resolution.
            token: Hugging Face token.
            revision: Hub revision.
            kwargs: Standard tokenizer keyword arguments.

        Returns:
            Loaded LayoutAction tokenizer.
        """
        _ = inputs
        subfolder = str(kwargs.pop("subfolder", ""))
        metadata = cached_file(
            pretrained_model_name_or_path,
            TOKENIZER_CONFIG_FILE,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
        )
        if metadata is None:
            raise FileNotFoundError(
                f"Could not resolve {TOKENIZER_CONFIG_FILE} from "
                f"{pretrained_model_name_or_path!s}"
            )
        with Path(metadata).open(encoding="utf-8") as f:
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
            encoded_boxes: list[torch.Tensor] = []
            for elem_idx in valid[: self.config.max_elements]:
                label_id = int(labels[batch_idx, elem_idx].item())
                qbox = quantized_bbox[batch_idx, elem_idx].long()
                sequences[batch_idx, cursor] = self.config.label_token_id(label_id)
                cursor += 1
                triples = self._encode_action_triples(qbox, encoded_boxes)
                for geo_idx, (option_id, object_id, value_id) in enumerate(triples):
                    _ = geo_idx
                    sequences[batch_idx, cursor] = option_id
                    sequences[batch_idx, cursor + 1] = object_id
                    sequences[batch_idx, cursor + 2] = value_id
                    cursor += 3
                encoded_boxes.append(qbox.detach().clone())
            sequences[batch_idx, cursor] = self.config.eos_token_id
        return sequences

    def _encode_action_triples(
        self, qbox: torch.Tensor, previous_boxes: list[torch.Tensor]
    ) -> list[tuple[int, int, int]]:
        """Encode one quantized box with vendor copy/margin/generate precedence."""
        if not previous_boxes:
            return [
                (
                    self.config.generate_token_id,
                    self.config.no_obj_token_id,
                    int(qbox[geo_idx].item()),
                )
                for geo_idx in range(4)
            ]
        previous = torch.stack(previous_boxes).to(device=qbox.device, dtype=torch.long)
        current = qbox.to(device=previous.device, dtype=torch.long)
        copy_label = previous.eq(current.unsqueeze(0))
        copy_choice = copy_label.any(dim=0)
        margin_label = torch.zeros(
            (previous.size(0), 2), dtype=torch.bool, device=previous.device
        )
        margin_label[:, 0] = previous[:, 1].eq(current[1])
        margin_label[:, 1] = previous[:, 0].eq(current[0])
        margin_value = (
            current[:2].float().unsqueeze(0)
            - previous[:, :2].float()
            - 0.5 * previous[:, 2:].float()
            - 0.5 * current[2:].float().unsqueeze(0)
        )
        margin_label &= margin_value.ge(0)
        margin_label_x4 = torch.cat(
            [margin_label, torch.zeros_like(margin_label)], dim=1
        )
        margin_choice = margin_label_x4.any(dim=0) & ~copy_choice
        generate_choice = ~(copy_choice | margin_choice)
        triples: list[tuple[int, int, int]] = []
        for geo_idx in range(4):
            if bool(copy_choice[geo_idx].item()):
                ref = self._latest_back_reference(copy_label[:, geo_idx])
                triples.append(
                    (
                        self.config.copy_token_id,
                        self.config.object_token_id(ref),
                        self.config.no_value_token_id,
                    )
                )
            elif bool(margin_choice[geo_idx].item()):
                ref = self._latest_back_reference(margin_label_x4[:, geo_idx])
                value = int(round(float(margin_value[-ref, geo_idx].item())))
                triples.append(
                    (
                        self.config.margin_token_id,
                        self.config.object_token_id(ref),
                        max(0, min(self.config.size - 1, value)),
                    )
                )
            elif bool(generate_choice[geo_idx].item()):
                triples.append(
                    (
                        self.config.generate_token_id,
                        self.config.no_obj_token_id,
                        int(current[geo_idx].item()),
                    )
                )
            else:
                raise ValueError("LayoutAction action selection produced no option")
        return triples

    def _latest_back_reference(self, hits: torch.Tensor) -> int:
        hit_indices = torch.nonzero(hits, as_tuple=False).flatten()
        if hit_indices.numel() == 0:
            raise ValueError("Expected at least one back-reference hit")
        return int(hits.numel() - hit_indices[-1].item())

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
                deferred_margins: list[tuple[int, int, int]] = []
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
                        if (
                            ref is None
                            or ref > len(boxes)
                            or not (0 <= val_id < self.config.size)
                        ):
                            valid = False
                            break
                        deferred_margins.append((geo_idx, ref, val_id))
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                for geo_idx, ref, val_id in deferred_margins:
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
