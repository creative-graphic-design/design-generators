"""Regenerate deterministic LayoutGPT vendor parity goldens."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import runpy
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch


class ByteTokenizer:
    """Small tokenizer stand-in for vendor prompt functions."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(text.encode())}


class _TokenizedText:
    def to(self, _device: str) -> _TokenizedText:
        return self


class _ClipStub(types.ModuleType):
    def tokenize(self, _text: str, *, truncate: bool = False) -> _TokenizedText:
        return _TokenizedText()


class _ClipModelStub:
    def encode_text(self, _tokens: _TokenizedText) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0]], dtype=torch.float32)


class _CssUtilsStub(types.ModuleType):
    def parseString(self, _text: str) -> object:  # noqa: N802
        raise ValueError("force vendor fallback parser")


class _StoppingCriteria:
    pass


def find_vendor_root(start: Path | None = None) -> Path:
    """Locate the read-only vendor/layout-gpt checkout."""
    root = start or Path(__file__).resolve().parents[3]
    candidates = [
        root,
        root / "vendor" / "layout-gpt",
        Path(str(root).split("=", 1)[0]) / "vendor" / "layout-gpt",
    ]
    required = Path("run_layoutgpt_2d.py")
    for candidate in candidates:
        if (candidate / required).exists():
            return candidate
    msg = "vendor/layout-gpt is not available"
    raise FileNotFoundError(msg)


def build_golden(vendor_root: Path | None = None) -> dict[str, Any]:
    """Execute vendor deterministic paths and return regenerated goldens."""
    root = find_vendor_root(vendor_root)
    vendor_2d = _load_vendor_module(root, "run_layoutgpt_2d.py")
    vendor_parser = _load_vendor_module(root, "parse_llm_output.py")

    train_records = json.loads(
        (root / "dataset/NSR-1K/counting/counting.train.json").read_text()
    )
    val_records = json.loads(
        (root / "dataset/NSR-1K/counting/counting.val.json").read_text()
    )
    candidates = train_records[:6]

    fixed_records = list(candidates)
    vendor_2d["random"].seed(42)
    vendor_2d["random"].shuffle(fixed_records)
    fixed_records = fixed_records[:3]

    args = vendor_2d["args"]
    args.canvas_size = 64
    args.icl_type = "fixed-random"
    args.setting = "counting"
    args.gpt_input_length_limit = 100_000

    tokenizer = ByteTokenizer()
    query = val_records[0]["prompt"]
    fixed_chat = vendor_2d["form_prompt_for_chatgpt"](
        text_input=query,
        top_k=3,
        tokenizer=tokenizer,
        supporting_examples=fixed_records,
        features=None,
    )
    fixed_completion = vendor_2d["form_prompt_for_gpt3"](
        text_input=query,
        top_k=3,
        tokenizer=tokenizer,
        supporting_examples=fixed_records,
        features=None,
    )

    args.icl_type = "k-similar"
    function_globals = vendor_2d["form_prompt_for_gpt3"].__globals__
    function_globals["device"] = "cpu"
    function_globals["clip_model"] = _ClipModelStub()
    function_globals["features"] = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    function_globals["clip"] = _ClipStub("clip")
    k_candidates = candidates[:4]
    k_completion = vendor_2d["form_prompt_for_gpt3"](
        text_input=query,
        top_k=3,
        tokenizer=tokenizer,
        supporting_examples=k_candidates,
        features=function_globals["features"],
    )
    _, k_indices = (
        (100.0 * torch.tensor([[0.0, 1.0]]) @ function_globals["features"].T)
        .softmax(dim=-1)[0]
        .topk(3)
    )
    k_similar_ids = [k_candidates[index]["id"] for index in k_indices.tolist()]

    parser_lines = [
        "clock {height: 15px; width: 14px; top: 24px; left: 25px; }",
        "clock2 {height: 50px; width: 40px; top: 48px; left: 40px; }",
        "clock {height: 10px; width: 10px; top: 0px; left: 64px; }",
    ]
    parsed_2d = [
        vendor_parser["parse_layout"](line, canvas_size=64) for line in parser_lines
    ]
    parser_3d_lines = [
        "chair {length: 1.5m; width: 0.5m; height: 1m; orientation: 90 degrees; left: 2m; top: 3m; depth: 0m;}"
    ]
    parsed_3d = [
        vendor_parser["parse_3D_layout"](line, unit="m") for line in parser_3d_lines
    ]

    return {
        "metadata": {
            "source": "vendor/layout-gpt",
            "seed": 42,
            "k": 3,
            "canvas_size": 64,
            "query_id": val_records[0]["id"],
            "candidate_ids": [record["id"] for record in candidates],
        },
        "fixed_random_ids": [record["id"] for record in fixed_records],
        "fixed_chat_prompt": fixed_chat,
        "fixed_completion_prompt": fixed_completion,
        "k_similar_ids": k_similar_ids,
        "k_similar_completion_prompt": k_completion,
        "parser_lines": parser_lines,
        "parsed_2d": parsed_2d,
        "parser_3d_lines": parser_3d_lines,
        "parsed_3d": parsed_3d,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    golden = build_golden(args.vendor_root)
    encoded = json.dumps(golden, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


def _load_vendor_module(vendor_root: Path, filename: str) -> dict[str, Any]:
    with _vendor_runtime(vendor_root, filename):
        return runpy.run_path(str(vendor_root / filename))


@contextlib.contextmanager
def _vendor_runtime(vendor_root: Path, filename: str) -> Iterator[None]:
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_modules = {
        key: sys.modules.get(key)
        for key in ["clip", "openai", "transformers", "cssutils"]
    }
    sys.argv = [filename, "--icl_type", "fixed-random"]
    sys.path.insert(0, str(vendor_root))
    sys.modules["clip"] = _ClipStub("clip")
    sys.modules["openai"] = types.ModuleType("openai")
    transformers = types.ModuleType("transformers")
    setattr(transformers, "StoppingCriteria", _StoppingCriteria)
    setattr(transformers, "GPT2TokenizerFast", object)
    setattr(transformers, "LlamaForCausalLM", object)
    setattr(transformers, "LlamaTokenizer", object)
    sys.modules["transformers"] = transformers
    sys.modules["cssutils"] = _CssUtilsStub("cssutils")
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)
        sys.argv = old_argv
        sys.path = old_path
        for key, value in old_modules.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


if __name__ == "__main__":
    main()
