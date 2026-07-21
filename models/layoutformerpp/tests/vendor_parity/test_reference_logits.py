from __future__ import annotations

import os
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPTokenizer,
)
from layoutformerpp.conversion import load_original_state_dict
from layoutformerpp.serialization import build_default_tokens
from laygen.common.labels import labels_for_dataset
from laygen.common.vendor import vendor_root


PUBLIC_TASKS = ("gen_t", "gen_ts", "gen_r", "refinement", "completion", "ugen")
VENDOR_MARKER = Path("LayoutFormer++/src/model/layout_transformer/model.py")


@dataclass(frozen=True)
class ParityCase:
    """One public LayoutFormer++ checkpoint parity target."""

    dataset: str
    task: str

    @property
    def checkpoint_name(self) -> str:
        return f"{self.dataset}_{self.task}"


PARITY_CASES = tuple(
    ParityCase(dataset=dataset, task=task)
    for dataset in ("rico", "publaynet")
    for task in PUBLIC_TASKS
)
LABEL_CONSTRAINT_CASES = tuple(
    ParityCase(dataset=dataset, task=task)
    for dataset in ("rico", "publaynet")
    for task in ("gen_t", "gen_ts")
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "models").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _original_root(repo_root: Path) -> Path:
    return Path(
        os.environ.get(
            "LAYOUTFORMERPP_ORIGINAL_DIR",
            str(repo_root / ".cache/layoutformerpp/original"),
        )
    )


def _case_paths(repo_root: Path, case: ParityCase) -> tuple[Path, Path, Path | None]:
    original = _original_root(repo_root)
    checkpoint_dir = original / "ckpts" / case.checkpoint_name
    checkpoint = checkpoint_dir / "final_checkpoint.pth.tar"
    vocab = checkpoint_dir / "vocab.json"
    return (
        checkpoint,
        _vendor_src(repo_root),
        (vocab if vocab.exists() else None),
    )


def _vendor_src(repo_root: Path) -> Path:
    try:
        root = vendor_root(
            "ms-layout-generation",
            marker=VENDOR_MARKER,
            repo_root=repo_root,
        )
    except FileNotFoundError:
        return repo_root / "vendor/ms-layout-generation/LayoutFormer++/src"
    return root / "LayoutFormer++/src"


def _vendor_modules(vendor_src: Path):
    _install_vendor_packages(vendor_src)
    model_module = importlib.import_module("model.layout_transformer.model")
    tokenizer_module = importlib.import_module("model.layout_transformer.tokenizer")
    layout_package = sys.modules["model.layout_transformer"]
    setattr(layout_package, "LayoutTransformer", model_module.LayoutTransformer)
    setattr(
        layout_package,
        "LayoutTransformerTokenizer",
        tokenizer_module.LayoutTransformerTokenizer,
    )

    return model_module.LayoutTransformer, tokenizer_module.LayoutTransformerTokenizer


def _install_vendor_packages(vendor_src: Path) -> None:
    model_package = types.ModuleType("model")
    setattr(model_package, "__path__", [str(vendor_src / "model")])
    layout_package = types.ModuleType("model.layout_transformer")
    setattr(layout_package, "__path__", [str(vendor_src / "model/layout_transformer")])
    data_package = types.ModuleType("data")
    setattr(data_package, "__path__", [str(vendor_src / "data")])
    sys.modules.setdefault("model", model_package)
    sys.modules.setdefault("model.layout_transformer", layout_package)
    sys.modules.setdefault("data", data_package)


def _tokenizers(case: ParityCase, vocab: Path | None, vendor_tokenizer_class):
    if vocab is None:
        tokens = build_default_tokens(
            labels_for_dataset(case.dataset), task=case.task, grid=128
        )
        vendor_tokenizer = vendor_tokenizer_class(tokens)
        tokenizer = LayoutFormerPPTokenizer(tokens=tokens)
        return vendor_tokenizer, tokenizer
    vendor_tokenizer = vendor_tokenizer_class([])
    vendor_tokenizer.from_vocab(str(vocab))
    tokenizer = LayoutFormerPPTokenizer(vocab_file=str(vocab))
    return vendor_tokenizer, tokenizer


def _models(case: ParityCase, checkpoint: Path, tokenizer: LayoutFormerPPTokenizer):
    config = LayoutFormerPPConfig(
        vocab_size=tokenizer.vocab_size, dataset=case.dataset, task=case.task
    )
    state_dict = load_original_state_dict(checkpoint)
    model = LayoutFormerPPForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert missing == []
    assert unexpected == []
    model.eval()
    return config, state_dict, model


def _input_text(case: ParityCase) -> str:
    if case.task == "gen_r":
        return (
            "label_1 | label_2 <sep_labels_relations> "
            "label_1 index_1 <sep_ele_rela_ele> relation_3 "
            "<sep_ele_rela_ele> label_2 index_1 <sep_relations>"
        )
    if case.task == "gen_ts":
        return "label_1 10 10 | label_2 11 11"
    if case.task in {"completion", "refinement"}:
        return "label_1 0 0 10 10 | label_2 1 1 11 11 |"
    return "label_1 label_2"


def _output_text() -> str:
    return "label_1 0 0 10 10 | label_2 1 1 11 11 |"


def _index2label(case: ParityCase) -> dict[int, str]:
    return {
        idx + 1: f"label_{idx + 1}"
        for idx, _ in enumerate(labels_for_dataset(case.dataset))
    }


def _label_constraint(case: ParityCase, vendor_src: Path, vendor_tokenizer):
    _install_vendor_packages(vendor_src)
    if "seaborn" not in sys.modules:
        seaborn_stub = types.ModuleType("seaborn")

        def _empty_color_palette(
            palette: object = None,
            n_colors: int | None = None,
            desat: float | None = None,
            as_cmap: bool = False,
        ) -> list[object]:
            _ = (palette, n_colors, desat, as_cmap)
            return []

        setattr(seaborn_stub, "color_palette", _empty_color_palette)
        sys.modules["seaborn"] = seaborn_stub
    if "pycocotools.coco" not in sys.modules:
        pycocotools_stub = types.ModuleType("pycocotools")
        coco_stub = types.ModuleType("pycocotools.coco")
        setattr(coco_stub, "COCO", object)
        sys.modules["pycocotools"] = pycocotools_stub
        sys.modules["pycocotools.coco"] = coco_stub
    constraints = importlib.import_module(
        "model.layout_transformer.constrained_decoding"
    )
    TransformerSortByDictLabelConstraint = (
        constraints.TransformerSortByDictLabelConstraint
    )
    TransformerSortByDictLabelSizeConstraint = (
        constraints.TransformerSortByDictLabelSizeConstraint
    )

    index2label = _index2label(case)
    if case.task == "gen_ts":
        constraint = TransformerSortByDictLabelSizeConstraint(
            vendor_tokenizer,
            128,
            set(index2label.values()),
            index2label,
            add_sep_token=True,
        )
        constraint.prepare([[1, 2]], [[[0, 0, 10, 10], [1, 1, 11, 11]]])
        return constraint
    constraint = TransformerSortByDictLabelConstraint(
        vendor_tokenizer,
        128,
        set(index2label.values()),
        index2label,
        add_sep_token=True,
    )
    constraint.prepare([[1, 2]])
    return constraint


@pytest.mark.vendor_parity
@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.checkpoint_name)
def test_public_checkpoint_logits_match_vendor(case: ParityCase) -> None:
    root = _repo_root()
    checkpoint, vendor_src, vocab = _case_paths(root, case)
    if not checkpoint.exists() or not vendor_src.exists():
        skip_or_fail_vendor_parity(
            f"LayoutFormer++ checkpoint or vendor source is absent for {case.checkpoint_name}",
            missing_paths=[checkpoint, vendor_src],
            regeneration_hint=(
                "initialize the LayoutFormer++ vendor source and run "
                "models/layoutformerpp/scripts/convert_original_checkpoint.py"
            ),
        )

    layout_transformer, vendor_tokenizer_class = _vendor_modules(vendor_src)
    vendor_tokenizer, tokenizer = _tokenizers(case, vocab, vendor_tokenizer_class)
    assert vendor_tokenizer._token2id == tokenizer.get_vocab()
    config, state_dict, model = _models(case, checkpoint, tokenizer)
    vendor_model = layout_transformer(
        vocab_size=len(vendor_tokenizer),
        max_len=config.max_position_embeddings,
        bos_token_id=vendor_tokenizer.bos_token_id,
        pad_token_id=vendor_tokenizer.pad_token_id,
        eos_token_id=vendor_tokenizer.eos_token_id,
        d_model=config.d_model,
        num_layers=config.encoder_layers,
        nhead=config.encoder_attention_heads,
        dropout=config.dropout,
        d_feedforward=config.dim_feedforward,
        share_embedding=config.share_embedding,
    )
    missing, unexpected = vendor_model.load_state_dict(state_dict, strict=False)
    assert missing == []
    assert unexpected == []
    vendor_model.eval()

    encoded = tokenizer.encode_text([_input_text(case)], add_eos=True)
    vendor_encoded = vendor_tokenizer([_input_text(case)], add_eos=True)
    torch.testing.assert_close(encoded["input_ids"], vendor_encoded["input_ids"])
    labels = tokenizer.encode_text([_output_text()], add_eos=True)["input_ids"]
    with torch.no_grad():
        vendor_logits = vendor_model.compute_loss(
            encoded["input_ids"], ~encoded["attention_mask"].bool(), labels
        )["logits"]
        new_logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=labels,
        ).logits
    torch.testing.assert_close(new_logits, vendor_logits, atol=0.0, rtol=0.0)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.checkpoint_name)
def test_public_checkpoint_generation_matches_vendor(case: ParityCase) -> None:
    root = _repo_root()
    checkpoint, vendor_src, vocab = _case_paths(root, case)
    if not checkpoint.exists() or not vendor_src.exists():
        skip_or_fail_vendor_parity(
            f"LayoutFormer++ checkpoint or vendor source is absent for {case.checkpoint_name}",
            missing_paths=[checkpoint, vendor_src],
            regeneration_hint=(
                "initialize the LayoutFormer++ vendor source and run "
                "models/layoutformerpp/scripts/convert_original_checkpoint.py"
            ),
        )

    layout_transformer, vendor_tokenizer_class = _vendor_modules(vendor_src)
    vendor_tokenizer, tokenizer = _tokenizers(case, vocab, vendor_tokenizer_class)
    config, state_dict, model = _models(case, checkpoint, tokenizer)
    vendor_model = layout_transformer(
        vocab_size=len(vendor_tokenizer),
        max_len=config.max_position_embeddings,
        bos_token_id=vendor_tokenizer.bos_token_id,
        pad_token_id=vendor_tokenizer.pad_token_id,
        eos_token_id=vendor_tokenizer.eos_token_id,
        d_model=config.d_model,
        num_layers=config.encoder_layers,
        nhead=config.encoder_attention_heads,
        dropout=config.dropout,
        d_feedforward=config.dim_feedforward,
        share_embedding=config.share_embedding,
    )
    vendor_model.load_state_dict(state_dict, strict=False)
    vendor_model.eval()

    encoded = tokenizer.encode_text([_input_text(case)], add_eos=True)
    do_sample = case.task in {"completion", "ugen", "gen_r"}
    torch.manual_seed(1234)
    with torch.no_grad():
        vendor_sequences = vendor_model(
            encoded["input_ids"],
            ~encoded["attention_mask"].bool(),
            max_length=min(config.decode_max_length, 12),
            do_sample=do_sample,
            top_k=10,
            temperature=0.7,
        )["output"]
    torch.manual_seed(1234)
    new_sequences = model._generate_sequences(
        encoded["input_ids"],
        encoded["attention_mask"],
        max_length=min(config.decode_max_length, 12),
        do_sample=do_sample,
        top_k=10,
        temperature=0.7,
    )
    torch.testing.assert_close(new_sequences, vendor_sequences)


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    "case", LABEL_CONSTRAINT_CASES, ids=lambda case: case.checkpoint_name
)
def test_label_constrained_generation_matches_vendor(case: ParityCase) -> None:
    root = _repo_root()
    checkpoint, vendor_src, vocab = _case_paths(root, case)
    if not checkpoint.exists() or not vendor_src.exists():
        skip_or_fail_vendor_parity(
            f"LayoutFormer++ checkpoint or vendor source is absent for {case.checkpoint_name}",
            missing_paths=[checkpoint, vendor_src],
            regeneration_hint=(
                "initialize the LayoutFormer++ vendor source and run "
                "models/layoutformerpp/scripts/convert_original_checkpoint.py"
            ),
        )

    layout_transformer, vendor_tokenizer_class = _vendor_modules(vendor_src)
    vendor_tokenizer, tokenizer = _tokenizers(case, vocab, vendor_tokenizer_class)
    config, state_dict, model = _models(case, checkpoint, tokenizer)
    vendor_model = layout_transformer(
        vocab_size=len(vendor_tokenizer),
        max_len=config.max_position_embeddings,
        bos_token_id=vendor_tokenizer.bos_token_id,
        pad_token_id=vendor_tokenizer.pad_token_id,
        eos_token_id=vendor_tokenizer.eos_token_id,
        d_model=config.d_model,
        num_layers=config.encoder_layers,
        nhead=config.encoder_attention_heads,
        dropout=config.dropout,
        d_feedforward=config.dim_feedforward,
        share_embedding=config.share_embedding,
    )
    vendor_model.load_state_dict(state_dict, strict=False)
    vendor_model.eval()

    encoded = tokenizer.encode_text([_input_text(case)], add_eos=True)
    with torch.no_grad():
        vendor_sequences = vendor_model(
            encoded["input_ids"],
            ~encoded["attention_mask"].bool(),
            max_length=12,
            generation_constraint_fn=_label_constraint(
                case, vendor_src, vendor_tokenizer
            ),
        )["output"]
    new_sequences = model._generate_sequences(
        encoded["input_ids"],
        encoded["attention_mask"],
        max_length=12,
        generation_constraint_fn=_label_constraint(case, vendor_src, vendor_tokenizer),
    )
    torch.testing.assert_close(new_sequences, vendor_sequences)
