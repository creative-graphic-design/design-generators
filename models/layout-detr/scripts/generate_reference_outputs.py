"""Generate LayoutDETR vendor reference tensors outside git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Protocol, cast

import torch
from PIL import Image
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from layout_detr import LayoutDetrProcessor
from layout_detr.vendor_state import (
    extract_generator_state,
    load_vendor_generator,
    temporary_sys_path,
)


class _VendorTokenizerProtocol(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> BatchEncoding: ...


class _VendorGeneratorProtocol(Protocol):
    tokenizer: _VendorTokenizerProtocol
    max_text_length: int
    backbone: torch.nn.Module
    fc_z: torch.nn.Module
    emb_label: torch.nn.Module
    text_encoder: torch.nn.Module
    enc_text_len: torch.nn.Module
    fc_in: torch.nn.Module
    transformer: torch.nn.Module
    input_proj: torch.nn.Module
    bbox_embed: torch.nn.Module

    def eval(self) -> "_VendorGeneratorProtocol": ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _, config, report = extract_generator_state(
        args.checkpoint,
        vendor_root=args.vendor_root,
        device="cpu",
    )
    processor = LayoutDetrProcessor(config=config)
    text_rows = [args.texts.split("|")]
    encoded = processor(
        images=Image.open(args.background).convert("RGB"),
        texts=text_rows[0],
        labels=args.labels.split("|"),
    )
    tokenizer = _build_vendor_tokenizer(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
    )
    encoded["input_ids"], encoded["text_attention_mask"] = _tokenize_vendor_texts(
        tokenizer,
        text_rows=cast(list[list[str]], encoded["texts"]),
        max_seq_length=config.max_seq_length,
        max_text_length=config.max_text_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    latents = torch.randn(
        (1, config.max_seq_length, config.z_dim),
        generator=generator,
    )
    inputs = {**dict(encoded), "latents": latents}
    loaded_generator, custom_op_import_required = load_vendor_generator(
        args.checkpoint,
        vendor_root=args.vendor_root,
        device=device,
    )
    vendor_generator = cast(_VendorGeneratorProtocol, loaded_generator)
    vendor_generator.eval()
    vendor_generator.tokenizer = tokenizer
    _patch_vendor_bert_runtime(cast(torch.nn.Module, vendor_generator))
    with torch.no_grad(), temporary_sys_path(args.vendor_root.resolve()):
        bbox_fake = _run_vendor_forward(
            vendor_generator,
            inputs=inputs,
            device=device,
        )
    torch.save(_cpu_tensors(inputs), args.output_dir / "inputs.pt")
    torch.save(
        {"bbox_fake": bbox_fake.detach().cpu()}, args.output_dir / "bbox_fake.pt"
    )
    meta = {
        "command": "generate_reference_outputs.py",
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "background": str(args.background),
        "texts": args.texts,
        "labels": args.labels,
        "tokenizer_name": args.tokenizer_name,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "custom_op_import_required": (
            custom_op_import_required or report["custom_op_import_required"]
        ),
        "conversion_report": report,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps({"bbox_fake_shape": list(bbox_fake.shape), **meta}, indent=2))


def _build_vendor_tokenizer(
    tokenizer_name: str,
    *,
    local_files_only: bool,
) -> BertTokenizerFast:
    tokenizer = BertTokenizerFast.from_pretrained(
        tokenizer_name,
        local_files_only=local_files_only,
    )
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    return tokenizer


def _tokenize_vendor_texts(
    tokenizer: BertTokenizerFast,
    *,
    text_rows: list[list[str]],
    max_seq_length: int,
    max_text_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_texts = [
        text
        for row in text_rows
        for text in (row[:max_seq_length] + [""] * max(0, max_seq_length - len(row)))
    ]
    encoded = tokenizer(
        flat_texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    batch = len(text_rows)
    width = max_seq_length
    input_ids = torch.zeros(batch * width, max_text_length, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    input_ids[: len(flat_texts)] = encoded.input_ids
    attention_mask[: len(flat_texts)] = encoded.attention_mask
    return (
        input_ids.view(batch, width, max_text_length),
        attention_mask.view(batch, width, max_text_length).bool(),
    )


def _patch_vendor_bert_runtime(module: torch.nn.Module) -> None:
    for child in module.modules():
        config = getattr(child, "config", None)
        if config is not None:
            for name, value in {
                "output_attentions": False,
                "output_hidden_states": False,
                "use_return_dict": True,
                "use_cache": False,
                "_output_attentions": False,
                "_output_hidden_states": False,
                "_use_return_dict": True,
            }.items():
                try:
                    setattr(config, name, value)
                except AttributeError:
                    config.__dict__[name] = value
        if config is not None and not hasattr(child, "get_head_mask"):
            child.get_head_mask = MethodType(_vendor_get_head_mask, child)
        if config is not None and not hasattr(child, "invert_attention_mask"):
            child.invert_attention_mask = MethodType(
                _vendor_invert_attention_mask, child
            )


def _vendor_get_head_mask(
    self: torch.nn.Module,
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
) -> list[torch.Tensor | None]:
    del self
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None].expand(
            num_hidden_layers,
            -1,
            -1,
            -1,
            -1,
        )
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    return [head_mask[index] for index in range(num_hidden_layers)]


def _vendor_invert_attention_mask(
    self: torch.nn.Module,
    encoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    dtype = next(self.parameters()).dtype
    if encoder_attention_mask.dim() == 2:
        encoder_attention_mask = encoder_attention_mask[:, None, None, :]
    elif encoder_attention_mask.dim() == 3:
        encoder_attention_mask = encoder_attention_mask[:, None, :, :]
    return (1.0 - encoder_attention_mask.to(dtype=dtype)) * -10000.0


def _run_vendor_forward(
    vendor_generator: _VendorGeneratorProtocol,
    *,
    inputs: dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    from training.networks_detr import (  # type: ignore[import-not-found]
        merge_lists,
        nested_tensor_from_tensor_list,
        normalize_2nd_moment,
    )

    background = cast(torch.Tensor, inputs["pixel_values"]).to(device)
    bbox_class = cast(torch.Tensor, inputs["bbox_labels"]).to(device)
    padding_mask = ~cast(torch.Tensor, inputs["layout_mask"]).to(device).bool()
    latents = cast(torch.Tensor, inputs["latents"]).to(device)
    bbox_text = cast(list[list[str]], inputs["texts"])

    bg_feat, pos = vendor_generator.backbone(nested_tensor_from_tensor_list(background))
    bg_feat, mask = bg_feat[-1].decompose()
    batch_size, element_count = bbox_class.shape
    z0 = normalize_2nd_moment(latents.reshape(batch_size, -1))
    z = vendor_generator.fc_z(z0).unsqueeze(1).expand(-1, element_count, -1)
    label_features = vendor_generator.emb_label(bbox_class)
    text = vendor_generator.tokenizer(
        merge_lists(bbox_text),
        padding="max_length",
        truncation=True,
        max_length=vendor_generator.max_text_length,
        return_tensors="pt",
    ).to(device)
    text_output = vendor_generator.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    text_features = text_output.last_hidden_state[:, 0, :].reshape(
        batch_size,
        element_count,
        -1,
    )
    text_lengths = torch.tensor(
        [len(text_value) for text_value in merge_lists(bbox_text)],
        dtype=torch.long,
        device=device,
    ).reshape(batch_size, element_count)
    text_len_features = vendor_generator.enc_text_len(text_lengths)
    hidden = torch.cat([z, label_features, text_features, text_len_features], dim=-1)
    hidden = torch.relu(vendor_generator.fc_in(hidden)).permute(1, 0, 2)
    hidden = vendor_generator.transformer(
        src=vendor_generator.input_proj(bg_feat),
        mask=mask,
        pos_embed=pos[-1],
        tgt=hidden,
        tgt_key_padding_mask=padding_mask,
    )[0]
    return vendor_generator.bbox_embed(hidden).sigmoid()


def _cpu_tensors(inputs: dict[str, object]) -> dict[str, object]:
    result = {}
    for key, value in inputs.items():
        result[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return result


if __name__ == "__main__":
    main()
