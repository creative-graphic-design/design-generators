from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizer

from .configuration_layout_dm import LayoutDMConfig

KEY_MULT_DICT = {
    "x-y-w-h": {"y": 1, "w": 2, "h": 3},
    "xywh": {},
}


class LayoutDMTokenizer(PreTrainedTokenizer):
    """Structured LayoutDM tokenizer backed by a Transformers vocabulary."""

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "layout_config_file": "layout_config.json",
        "cluster_centers_file": "cluster_centers.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        config: LayoutDMConfig | dict[str, Any] | None = None,
        *,
        vocab_file: str | Path | None = None,
        layout_config_file: str | Path | None = None,
        cluster_centers_file: str | Path | None = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = self._load_config(
                layout_config_file=layout_config_file,
                cluster_centers_file=cluster_centers_file,
                kwargs=kwargs,
            )
        elif isinstance(config, dict):
            config = LayoutDMConfig(**cast(dict[str, Any], config))
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

        vocab = self._build_vocab()
        if vocab_file is not None and Path(vocab_file).exists():
            loaded_vocab = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
            vocab = {str(token): int(idx) for token, idx in loaded_vocab.items()}
        self._token_to_id = vocab
        self._id_to_token = {idx: token for token, idx in vocab.items()}
        pad_token = kwargs.pop("pad_token", "pad")
        mask_token = kwargs.pop("mask_token", "mask")
        model_max_length = kwargs.pop("model_max_length", self.config.max_token_length)

        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def var_names(self) -> tuple[str, ...]:
        return tuple(self.config.var_order.split("-"))

    def __call__(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        if args:
            raise TypeError(
                "LayoutDMTokenizer expects structured layout inputs; use "
                "encode_layout(bbox=..., labels=..., mask=...) instead of text."
            )
        if "bbox" not in kwargs or "labels" not in kwargs:
            raise TypeError(
                "LayoutDMTokenizer.__call__ requires bbox=... and labels=..."
            )
        return self.encode_layout(
            bbox=cast(torch.Tensor, kwargs["bbox"]),
            labels=cast(torch.Tensor, kwargs["labels"]),
            mask=cast(torch.Tensor | None, kwargs.get("mask")),
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        raise TypeError("LayoutDMTokenizer does not tokenize text")

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.pad_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(int(index), self.pad_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def save_vocabulary(
        self, save_directory: str | Path, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        prefix = "" if filename_prefix is None else f"{filename_prefix}-"
        vocab_file = save_path / f"{prefix}vocab.json"
        layout_config_file = save_path / f"{prefix}layout_config.json"
        cluster_centers_file = save_path / f"{prefix}cluster_centers.json"
        vocab_file.write_text(
            json.dumps(self._token_to_id, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        config_data = dict(self.config.config)
        config_data["id2label"] = {str(k): v for k, v in self.config.id2label.items()}
        config_data["cluster_centers"] = None
        layout_config_file.write_text(
            json.dumps(config_data, indent=2, sort_keys=True), encoding="utf-8"
        )
        centers = {
            key: [float(v) for v in self._centers(key, torch.device("cpu")).double()]
            for key in ("x", "y", "w", "h")
        }
        cluster_centers_file.write_text(
            json.dumps(centers, indent=2, sort_keys=True), encoding="utf-8"
        )
        return (str(vocab_file), str(layout_config_file), str(cluster_centers_file))

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *args: object, **kwargs: object
    ) -> "LayoutDMTokenizer":
        path = Path(path)
        if (path / "tokenizer").is_dir():
            path = path / "tokenizer"
        return super().from_pretrained(
            path, *cast(tuple[Any, ...], args), **cast(dict[str, Any], kwargs)
        )

    @classmethod
    def _load_config(
        cls,
        *,
        layout_config_file: str | Path | None,
        cluster_centers_file: str | Path | None,
        kwargs: dict[str, object],
    ) -> LayoutDMConfig:
        layout_config = kwargs.pop("layout_config", None)
        if layout_config is None:
            if layout_config_file is None:
                raise ValueError(
                    "LayoutDMTokenizer requires a LayoutDMConfig or layout_config_file"
                )
            layout_config = json.loads(
                Path(layout_config_file).read_text(encoding="utf-8")
            )
        if cluster_centers_file is not None and Path(cluster_centers_file).exists():
            centers = json.loads(Path(cluster_centers_file).read_text(encoding="utf-8"))
            layout_config["cluster_centers"] = {
                key: [float(v) for v in values] for key, values in centers.items()
            }
        return LayoutDMConfig(**cast(dict[str, Any], layout_config))

    def encode_layout(
        self,
        *,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bbox = torch.as_tensor(bbox, dtype=torch.float64)
        labels = torch.as_tensor(labels, dtype=torch.long)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            bbox = bbox.unsqueeze(0)
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=labels.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        batch_size, seq_length = labels.shape
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"seq_length {seq_length} exceeds max_seq_length {self.config.max_seq_length}"
            )
        bbox_ids = self._encode_bbox(bbox) + self.config.num_categories
        seq = torch.cat((labels.unsqueeze(-1), bbox_ids), dim=-1)
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
            "attention_mask": mask.repeat_interleave(5, dim=1),
            "mask": mask.repeat_interleave(5, dim=1),
        }

    def decode_layout(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        ids = ids.reshape(ids.shape[0], self.config.max_seq_length, 5)
        labels = ids[..., 0].clone()
        bbox_ids = ids[..., 1:].clone() - self.config.num_categories
        label_valid = (labels >= 0) & (labels < self.config.num_categories)
        bbox_valid = (bbox_ids >= 0) & (bbox_ids < self.config.num_bbox_tokens)
        mask = label_valid & bbox_valid.all(dim=-1)
        bbox = self._decode_bbox(bbox_ids)
        labels = labels.masked_fill(~mask, 0)
        bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0.0)
        return {"bbox": bbox.float(), "labels": labels, "mask": mask}

    def token_mask(self) -> torch.Tensor:
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

    def full_id_maps(self) -> dict[str, list[int]]:
        return {key: self._mapping(key)["full"].tolist() for key in self.var_names}

    def _build_vocab(self) -> dict[str, int]:
        vocab: dict[str, int] = {}
        for idx, label in self.config.id2label.items():
            vocab[f"c:{label}"] = int(idx)
        for key in ("x", "y", "w", "h"):
            start, end = self.config.bbox_slices[key]
            for token_id in range(start, end):
                local_id = token_id - start
                vocab[f"{key}:{local_id}"] = token_id
        vocab["pad"] = self.config.pad_token_id
        vocab["mask"] = self.config.mask_token_id
        return vocab

    def _centers(self, key: str, device: torch.device) -> torch.Tensor:
        centers = (
            None
            if self.config.cluster_centers is None
            else self.config.cluster_centers.get(key)
        )
        if centers is None:
            delta = 1.0 / self.config.num_bin_bboxes
            start, stop = (0.0, 1.0 - delta) if key in {"x", "y"} else (delta, 1.0)
            centers = torch.linspace(
                start, stop, self.config.num_bin_bboxes, dtype=torch.float64
            ).tolist()
        return torch.tensor(centers, device=device, dtype=torch.float64).flatten()

    def _encode_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        bbox = bbox.to(dtype=torch.float64)
        pieces = []
        for i, key in enumerate(("x", "y", "w", "h")):
            values = bbox[..., i]
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

    def _decode_bbox(self, bbox_ids: torch.Tensor) -> torch.Tensor:
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
                    local_ids.double() * delta
                    if key in {"x", "y"}
                    else (local_ids.double() + 1.0) * delta
                )
            else:
                centers = self._centers(key, ids.device)
                values = centers[local_ids]
            pieces.append(values)
        return torch.stack(pieces, dim=-1).clamp(0.0, 1.0).float()

    def _mapping(self, key: str) -> dict[str, torch.Tensor]:
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


def _bucketize(
    inputs: torch.Tensor, from_ids: torch.Tensor, to_ids: torch.Tensor
) -> torch.Tensor:
    from_ids = from_ids.to(inputs.device)
    to_ids = to_ids.to(inputs.device)
    index = torch.bucketize(inputs.reshape(-1), from_ids)
    return to_ids[index].reshape(inputs.shape).to(inputs.device)
