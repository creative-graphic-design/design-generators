"""Tokenizer for LayoutFormer++ layout strings."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import PreTrainedTokenizer


class LayoutFormerPPTokenizer(PreTrainedTokenizer):
    """Whitespace tokenizer backed by the vendor `vocab.json` format."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        tokens: list[str] | None = None,
        x_grid: int = 128,
        y_grid: int = 128,
        bbox_order: str = "ltwh",
        **kwargs: object,
    ) -> None:
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.bbox_order = bbox_order
        if vocab_file is not None:
            with Path(vocab_file).open() as f:
                token2id = {str(k): int(v) for k, v in json.load(f).items()}
        else:
            base_tokens = ["<bos>", "<eos>", "<pad>", "<sep>", "<unk>"]
            token2id = {token: idx for idx, token in enumerate(base_tokens)}
            for token in tokens or []:
                if token not in token2id:
                    token2id[token] = len(token2id)
        self._token2id = token2id
        self._id2token = {idx: token for token, idx in token2id.items()}
        super().__init__(
            bos_token=kwargs.pop("bos_token", "<bos>"),
            eos_token=kwargs.pop("eos_token", "<eos>"),
            pad_token=kwargs.pop("pad_token", "<pad>"),
            sep_token=kwargs.pop("sep_token", "<sep>"),
            unk_token=kwargs.pop("unk_token", "<unk>"),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        """Return the number of saved vocabulary entries."""
        return len(self._token2id)

    def get_vocab(self) -> dict[str, int]:
        """Return token-to-id vocabulary mapping."""
        return dict(self._token2id)

    def _tokenize(self, text: str, **kwargs: object) -> list[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id2token.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Join synthetic layout tokens using spaces."""
        return " ".join(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        """Save `vocab.json` in vendor-compatible token-to-id format."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = (
            "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        )
        out_path = out_dir / name
        with out_path.open("w") as f:
            json.dump(self._token2id, f, indent=2, sort_keys=True)
        return (str(out_path),)

    def save_pretrained(
        self, save_directory: str | Path, **kwargs: object
    ) -> tuple[str, ...]:
        """Save tokenizer files plus LayoutFormer++ tokenizer metadata."""
        paths = super().save_pretrained(str(save_directory), **kwargs)
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
        cls, pretrained_model_name_or_path: str | Path, *args: object, **kwargs: object
    ):
        """Load tokenizer and LayoutFormer++ metadata."""
        path = Path(pretrained_model_name_or_path)
        metadata_path = path / "layoutformerpp_tokenizer_config.json"
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
            kwargs = {**metadata, **kwargs}
        return super().from_pretrained(
            str(pretrained_model_name_or_path), *args, **kwargs
        )

    def encode_text(
        self, text: str | list[str], *, add_eos: bool = True, add_bos: bool = False
    ):
        """Tokenize vendor-style text while matching the original EOS/BOS behavior."""
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
