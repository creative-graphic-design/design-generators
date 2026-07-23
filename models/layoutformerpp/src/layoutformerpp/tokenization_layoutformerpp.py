"""Tokenizer for LayoutFormer++ layout strings."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Final, cast

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.tokenization import (
    WhitespaceTokenizerMixin,
    build_token_maps,
    save_json_vocabulary,
)
from transformers import BatchEncoding, PreTrainedTokenizer

DEFAULT_MODEL_MAX_LENGTH: Final[int] = 1_000_000_000_000_000_019_884_624_838_656


class LayoutFormerPPTokenizer(WhitespaceTokenizerMixin, PreTrainedTokenizer):
    """Whitespace tokenizer backed by the released `vocab.json` format."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tokens: list[str] | None = None,
        x_grid: int = 128,
        y_grid: int = 128,
        bbox_order: BoxFormat | str = BoxFormat.ltwh,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        sep_token: str = "<sep>",
        unk_token: str = "<unk>",
        model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
        padding_side: str = "right",
        truncation_side: str = "right",
        clean_up_tokenization_spaces: bool = False,
        added_tokens_decoder: dict[int | str, object] | None = None,
        backend: str = "custom",
        tokenizer_file: str | None = None,
        name_or_path: str = "",
        is_local: bool = False,
        local_files_only: bool = False,
        processor_class: str | None = None,
    ) -> None:
        """Initialize a tokenizer from a vocab file or synthetic token list."""
        _ = (backend, tokenizer_file, is_local, local_files_only, processor_class)
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.bbox_order = str(normalize_box_format(bbox_order))
        self._token2id, self._id2token = build_token_maps(
            vocab_file=vocab_file,
            tokens=tokens,
            base_tokens=("<bos>", "<eos>", "<pad>", "<sep>", "<unk>"),
        )
        tokenizer_kwargs: dict[str, object] = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "sep_token": sep_token,
            "unk_token": unk_token,
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "backend": backend,
            "name_or_path": name_or_path,
        }
        if added_tokens_decoder is not None:
            tokenizer_kwargs["added_tokens_decoder"] = added_tokens_decoder
        super().__init__(**tokenizer_kwargs)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Save `vocab.json` in checkpoint-compatible token-to-id format."""
        return save_json_vocabulary(
            save_directory=save_directory,
            filename="vocab.json",
            data=self._token2id,
            filename_prefix=filename_prefix,
        )

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> tuple[str, ...]:
        """Save tokenizer files plus LayoutFormer++ tokenizer metadata."""
        _ = kwargs
        paths = super().save_pretrained(
            str(save_directory),
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
        )
        metadata = {
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "bbox_order": self.bbox_order,
        }
        with (Path(save_directory) / "layoutformerpp_tokenizer_config.json").open(
            "w"
        ) as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return paths

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        x_grid: int | None = None,
        y_grid: int | None = None,
        bbox_order: BoxFormat | str | None = None,
    ) -> "LayoutFormerPPTokenizer":
        """Load tokenizer and LayoutFormer++ metadata."""
        path = Path(pretrained_model_name_or_path)
        metadata_path = path / "layoutformerpp_tokenizer_config.json"
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
        if x_grid is not None:
            metadata["x_grid"] = x_grid
        if y_grid is not None:
            metadata["y_grid"] = y_grid
        if bbox_order is not None:
            metadata["bbox_order"] = bbox_order
        return cast(
            "LayoutFormerPPTokenizer",
            super().from_pretrained(
                str(pretrained_model_name_or_path),
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **metadata,
            ),
        )

    def encode_text(
        self, text: str | list[str], *, add_eos: bool = True, add_bos: bool = False
    ) -> BatchEncoding:
        """Tokenize reference-style text while matching the original EOS/BOS behavior."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        normalized = []
        for item in texts:
            tokens = item.strip().split()
            if add_eos:
                tokens.append(self.eos_token)
            if add_bos:
                tokens.insert(0, self.bos_token)
            normalized.append(" ".join(tokens))
        return self(
            normalized, padding=True, add_special_tokens=False, return_tensors="pt"
        )
