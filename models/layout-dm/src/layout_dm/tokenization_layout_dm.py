from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .configuration_layout_dm import LayoutDMConfig

KEY_MULT_DICT = {
    "x-y-w-h": {"y": 1, "w": 2, "h": 3},
    "xywh": {},
}


class LayoutDMTokenizer:
    config_name = "tokenizer_config.json"

    def __init__(self, config: LayoutDMConfig) -> None:
        self.config = config
        if self.config.var_order != "c-x-y-w-h":
            raise NotImplementedError(
                "Only c-x-y-w-h LayoutDM token order is supported"
            )
        if (
            "mask" in self.config.special_tokens
            and self.config.special_tokens[-1] != "mask"
        ):
            raise ValueError("LayoutDM requires mask to be the final special token")

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.config.pad_token_id

    @property
    def mask_token_id(self) -> int:
        return self.config.mask_token_id

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.config.var_order.split("-"))

    def _centers(self, key: str, device: torch.device) -> torch.Tensor:
        centers = (
            None
            if self.config.cluster_centers is None
            else self.config.cluster_centers.get(key)
        )
        if centers is None:
            delta = 1.0 / self.config.num_bin_bboxes
            start, stop = (0.0, 1.0 - delta) if key in {"x", "y"} else (delta, 1.0)
            centers = torch.linspace(start, stop, self.config.num_bin_bboxes).tolist()
        return torch.tensor(centers, device=device, dtype=torch.float32).flatten()

    def _encode_bbox(self, bbox: torch.Tensor) -> torch.LongTensor:
        pieces = []
        for i, key in enumerate(("x", "y", "w", "h")):
            values = bbox[..., i].float()
            if self.config.bbox_quantization == "linear":
                delta = 1.0 / self.config.num_bin_bboxes
                if key in {"x", "y"}:
                    ids = (
                        (values.clamp(0.0, 1.0 - delta) * self.config.num_bin_bboxes)
                        .round()
                        .long()
                    )
                else:
                    ids = (
                        (
                            (values.clamp(delta, 1.0) - delta)
                            * self.config.num_bin_bboxes
                        )
                        .round()
                        .long()
                    )
            elif self.config.bbox_quantization in {"kmeans", "percentile"}:
                centers = self._centers(key, values.device)
                ids = (
                    torch.cdist(values.reshape(-1, 1), centers.reshape(-1, 1))
                    .argmin(dim=-1)
                    .reshape(values.shape)
                )
            else:
                raise ValueError(
                    f"Unsupported bbox_quantization: {self.config.bbox_quantization}"
                )
            offset = (
                KEY_MULT_DICT[self.config.shared_bbox_vocab].get(key, 0)
                * self.config.num_bin_bboxes
            )
            pieces.append(ids + offset)
        return torch.stack(pieces, dim=-1)

    def _decode_bbox(self, bbox_ids: torch.LongTensor) -> torch.FloatTensor:
        ids = bbox_ids.clone()
        pieces = []
        for i, key in enumerate(("x", "y", "w", "h")):
            offset = (
                KEY_MULT_DICT[self.config.shared_bbox_vocab].get(key, 0)
                * self.config.num_bin_bboxes
            )
            local_ids = (ids[..., i] - offset).clamp(0, self.config.num_bin_bboxes - 1)
            if self.config.bbox_quantization == "linear":
                delta = 1.0 / self.config.num_bin_bboxes
                values = (
                    local_ids.float() * delta
                    if key in {"x", "y"}
                    else (local_ids.float() + 1.0) * delta
                )
            else:
                centers = self._centers(key, ids.device)
                values = centers[local_ids]
            pieces.append(values)
        return torch.stack(pieces, dim=-1).clamp(0.0, 1.0)

    def encode(
        self,
        *,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bbox = bbox.float()
        labels = labels.long()
        mask = mask.bool()
        batch_size, seq_length = labels.shape
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"seq_length {seq_length} exceeds max_seq_length {self.config.max_seq_length}"
            )
        bbox_ids = self._encode_bbox(bbox) + self.config.num_categories
        labels = labels.unsqueeze(-1)
        seq = torch.cat((labels, bbox_ids), dim=-1)
        pad_len = self.config.max_seq_length - seq_length
        if pad_len:
            pad = torch.full(
                (batch_size, pad_len, 5),
                self.pad_token_id,
                dtype=torch.long,
                device=seq.device,
            )
            seq = torch.cat((seq, pad), dim=1)
            mask = torch.cat(
                (
                    mask,
                    torch.zeros(
                        batch_size, pad_len, dtype=torch.bool, device=mask.device
                    ),
                ),
                dim=1,
            )
        seq = seq.masked_fill(~mask.unsqueeze(-1), self.pad_token_id)
        return {
            "input_ids": seq.reshape(batch_size, -1),
            "mask": mask.repeat_interleave(5, dim=1),
        }

    def decode(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ids = input_ids.long().reshape(
            input_ids.shape[0], self.config.max_seq_length, 5
        )
        labels = ids[..., 0].clone()
        bbox_ids = ids[..., 1:].clone() - self.config.num_categories
        label_valid = (labels >= 0) & (labels < self.config.num_categories)
        bbox_valid = (bbox_ids >= 0) & (bbox_ids < self.config.num_bbox_tokens)
        mask = label_valid & bbox_valid.all(dim=-1)
        bbox = self._decode_bbox(bbox_ids)
        labels = labels.masked_fill(~mask, 0)
        bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0.0)
        return {"bbox": bbox, "labels": labels, "mask": mask}

    def token_mask(self) -> torch.BoolTensor:
        mask = torch.zeros(
            self.config.max_token_length, self.config.vocab_size, dtype=torch.bool
        )
        special_start = self.config.num_categories + self.config.num_bbox_tokens
        for pos, key in enumerate(self.var_names * self.config.max_seq_length):
            if key == "c":
                mask[pos, : self.config.num_categories] = True
                mask[pos, special_start:] = True
            else:
                start, end = self.config.bbox_slices[key]
                mask[pos, start:end] = True
                mask[pos, special_start:] = True
        return mask

    def full_to_partial_ids(self, ids: torch.Tensor, key: str) -> torch.Tensor:
        mapping = self._mapping(key)
        return _bucketize(ids, mapping["full"], mapping["partial"])

    def partial_to_full_ids(self, ids: torch.Tensor, key: str) -> torch.Tensor:
        mapping = self._mapping(key)
        return _bucketize(ids, mapping["partial"], mapping["full"])

    def full_to_partial_log_probs(
        self, log_probs: torch.Tensor, key: str
    ) -> torch.Tensor:
        mapping = self._mapping(key)["full"].to(log_probs.device)
        index = mapping.reshape(1, -1, 1).expand(
            log_probs.shape[0], -1, log_probs.shape[-1]
        )
        return torch.gather(log_probs, dim=1, index=index)

    def partial_to_full_log_probs(
        self, log_probs: torch.Tensor, key: str
    ) -> torch.Tensor:
        mapping = self._mapping(key)["full"].to(log_probs.device)
        out = torch.full(
            (log_probs.shape[0], self.config.vocab_size, log_probs.shape[-1]),
            -70.0,
            device=log_probs.device,
            dtype=log_probs.dtype,
        )
        index = mapping.reshape(1, -1, 1).expand(
            log_probs.shape[0], -1, log_probs.shape[-1]
        )
        return out.scatter(dim=1, index=index, src=log_probs)

    def _mapping(self, key: str) -> dict[str, torch.LongTensor]:
        if key == "c":
            full = list(range(self.config.num_categories)) + [
                self.pad_token_id,
                self.mask_token_id,
            ]
        else:
            start, end = self.config.bbox_slices[key]
            full = list(range(start, end)) + [self.pad_token_id, self.mask_token_id]
        return {
            "partial": torch.arange(len(full), dtype=torch.long),
            "full": torch.tensor(full, dtype=torch.long),
        }

    def full_id_maps(self) -> dict[str, list[int]]:
        return {key: self._mapping(key)["full"].tolist() for key in self.var_names}

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = dict(self.config.config)
        data["id2label"] = {str(k): v for k, v in self.config.id2label.items()}
        (save_path / self.config_name).write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LayoutDMTokenizer":
        data = json.loads((Path(path) / cls.config_name).read_text(encoding="utf-8"))
        return cls(LayoutDMConfig(**data))


def _bucketize(
    inputs: torch.Tensor, from_ids: torch.Tensor, to_ids: torch.Tensor
) -> torch.Tensor:
    from_ids = from_ids.to(inputs.device)
    to_ids = to_ids.to(inputs.device)
    index = torch.bucketize(inputs.reshape(-1), from_ids)
    return to_ids[index].reshape(inputs.shape).to(inputs.device)
