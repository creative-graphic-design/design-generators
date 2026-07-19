"""Tokenizer for LT-Net scene-graph token vocabularies."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Final, cast

from transformers import PreTrainedTokenizer

DEFAULT_MODEL_MAX_LENGTH: Final[int] = 1_000_000_000_000_000_019_884_624_838_656
SPECIAL_TOKENS: Final[tuple[str, ...]] = ("[PAD]", "[CLS]", "[SEP]", "[MASK]")


class LayoutTransformerRelationTokenizer(PreTrainedTokenizer):
    """Discrete scene-graph tokenizer saved as a standard HF tokenizer.

    Args:
        vocab_file: Optional path to ``object_pred_id2name.json``.
        tokens: Optional token list used when no vocab file is supplied.
        object_token_ids: Optional ids that represent object classes.
        relation_token_ids: Optional ids that represent predicates.
        pad_token: Padding token.
        cls_token: Sequence-start token.
        sep_token: Triple separator token.
        mask_token: Mask token.
        unk_token: Unknown token.
        model_max_length: Maximum tokenizer length metadata.
        kwargs: Additional tokenizer compatibility fields.

    Examples:
        >>> tokenizer = LayoutTransformerRelationTokenizer(tokens=["__image__", "person"])
        >>> tokenizer.convert_tokens_to_ids("[CLS]")
        1
    """

    vocab_files_names = {"vocab_file": "object_pred_id2name.json"}
    model_input_names = ["input_token", "input_obj_id", "segment_label", "token_type"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tokens: list[str] | None = None,
        object_token_ids: list[int] | None = None,
        relation_token_ids: list[int] | None = None,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        unk_token: str = "[MASK]",
        model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
        padding_side: str = "right",
        truncation_side: str = "right",
        clean_up_tokenization_spaces: bool = False,
        added_tokens_decoder: dict[int | str, object] | None = None,
        name_or_path: str = "",
        **kwargs: object,
    ) -> None:
        """Initialize vocabulary and object/relation id metadata."""
        _ = kwargs
        if vocab_file is not None:
            with Path(vocab_file).open() as f:
                raw_vocab = json.load(f)
            if all(str(key).isdigit() for key in raw_vocab):
                id2token = {int(key): str(value) for key, value in raw_vocab.items()}
                token2id = {value: key for key, value in id2token.items()}
            else:
                token2id = {str(key): int(value) for key, value in raw_vocab.items()}
                id2token = {value: key for key, value in token2id.items()}
        else:
            token2id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
            for token in tokens or []:
                if token not in token2id:
                    token2id[token] = len(token2id)
            id2token = {idx: token for token, idx in token2id.items()}
        self._token2id = token2id
        self._id2token = id2token
        self.object_token_ids = [int(item) for item in object_token_ids or []]
        self.relation_token_ids = [int(item) for item in relation_token_ids or []]
        tokenizer_kwargs: dict[str, object] = {
            "pad_token": pad_token,
            "cls_token": cls_token,
            "sep_token": sep_token,
            "mask_token": mask_token,
            "unk_token": unk_token,
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            "name_or_path": name_or_path,
        }
        if added_tokens_decoder is not None:
            tokenizer_kwargs["added_tokens_decoder"] = added_tokens_decoder
        super().__init__(**tokenizer_kwargs)

    @property
    def vocab_size(self) -> int:
        """Return the number of vocabulary entries."""
        return len(self._token2id)

    def get_vocab(self) -> dict[str, int]:
        """Return token-to-id mapping."""
        return dict(self._token2id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        _ = kwargs
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id2token.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join scene-graph tokens with spaces."""
        return " ".join(tokens)

    def encode_scene_graph_tokens(self, tokens: list[str]) -> list[int]:
        """Encode already-normalized scene-graph token strings.

        Args:
            tokens: Token strings in LT-Net order.

        Returns:
            Integer token ids.

        Raises:
            ValueError: If an unknown token appears.
        """
        ids: list[int] = []
        for token in tokens:
            token_id = self._convert_token_to_id(token)
            if token_id == self.unk_token_id and token != self.unk_token:
                raise ValueError(f"Unknown scene-graph token: {token}")
            ids.append(token_id)
        return ids

    def decode_scene_graph_tokens(self, input_token: list[int]) -> list[str]:
        """Decode integer scene-graph token ids into token strings."""
        return [self._convert_id_to_token(token_id) for token_id in input_token]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Save ``object_pred_id2name.json`` as id-to-token metadata."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (
            "object_pred_id2name.json"
            if filename_prefix is None
            else f"{filename_prefix}-object_pred_id2name.json"
        )
        out_path = out_dir / name
        with out_path.open("w") as f:
            json.dump(
                {str(key): value for key, value in sorted(self._id2token.items())},
                f,
                indent=2,
                sort_keys=True,
            )
        return (str(out_path),)

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> tuple[str, ...]:
        """Save tokenizer files plus LT-Net tokenizer metadata."""
        _ = kwargs
        paths = super().save_pretrained(
            str(save_directory),
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
        )
        metadata = {
            "object_token_ids": self.object_token_ids,
            "relation_token_ids": self.relation_token_ids,
        }
        with (Path(save_directory) / "layout_transformer_tokenizer_config.json").open(
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
        object_token_ids: list[int] | None = None,
        relation_token_ids: list[int] | None = None,
        **kwargs: object,
    ) -> "LayoutTransformerRelationTokenizer":
        """Load tokenizer and LT-Net metadata."""
        path = Path(pretrained_model_name_or_path)
        metadata_path = path / "layout_transformer_tokenizer_config.json"
        metadata: dict[str, object] = {}
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
        if object_token_ids is not None:
            metadata["object_token_ids"] = object_token_ids
        if relation_token_ids is not None:
            metadata["relation_token_ids"] = relation_token_ids
        metadata.update(kwargs)
        return cast(
            "LayoutTransformerRelationTokenizer",
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
