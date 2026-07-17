"""Regenerate deterministic LayoutGPT vendor parity goldens."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import runpy
import sys
import types
from collections.abc import Iterator, MutableMapping, Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Final, Protocol, cast

import torch
from typing_extensions import TypedDict

DEFAULT_SEED: Final[int] = 42
DEFAULT_K: Final[int] = 3
DEFAULT_CANVAS_SIZE: Final[int] = 64
DEFAULT_INPUT_LENGTH_LIMIT: Final[int] = 100_000
COUNTING_TRAIN_JSON: Final[Path] = Path("dataset/NSR-1K/counting/counting.train.json")
COUNTING_VAL_JSON: Final[Path] = Path("dataset/NSR-1K/counting/counting.val.json")
VENDOR_2D_SCRIPT: Final[str] = "run_layoutgpt_2d.py"
VENDOR_PARSER_SCRIPT: Final[str] = "parse_llm_output.py"


class StubbedVendorModule(StrEnum):
    """Vendor imports replaced by local stubs during golden generation."""

    clip = auto()
    openai = auto()
    transformers = auto()
    cssutils = auto()


class GoldenMetadata(TypedDict):
    """Metadata stored with regenerated vendor parity values."""

    source: str
    seed: int
    k: int
    canvas_size: int
    query_id: object
    candidate_ids: list[object]


class VendorGolden(TypedDict):
    """Structured vendor parity golden payload."""

    metadata: GoldenMetadata
    fixed_random_ids: list[object]
    fixed_chat_prompt: object
    fixed_completion_prompt: object
    k_similar_ids: list[object]
    k_similar_completion_prompt: object
    parser_lines: list[str]
    parsed_2d: list[object]
    parser_3d_lines: list[str]
    parsed_3d: list[object]


class VendorRandom(Protocol):
    """Random module surface used by the vendor script."""

    def seed(self, value: int) -> None: ...

    def shuffle(self, value: list[dict[str, object]]) -> None: ...


class VendorArgs(Protocol):
    """Mutable argparse namespace exposed by the vendor script."""

    canvas_size: int
    icl_type: str
    setting: str
    gpt_input_length_limit: int


class VendorPromptFunction(Protocol):
    """Prompt function signature used by the vendor LayoutGPT script."""

    __globals__: MutableMapping[str, object]

    def __call__(
        self,
        *,
        text_input: object,
        top_k: int,
        tokenizer: ByteTokenizer,
        supporting_examples: Sequence[dict[str, object]],
        features: object | None,
    ) -> object:
        del text_input, top_k, tokenizer, supporting_examples, features
        raise NotImplementedError


class VendorParse2D(Protocol):
    """2D vendor parser signature."""

    def __call__(self, line: str, *, canvas_size: int) -> object: ...


class VendorParse3D(Protocol):
    """3D vendor parser signature."""

    def __call__(self, line: str, *, unit: str) -> object: ...


class ByteTokenizer:
    """Small tokenizer stand-in for vendor prompt functions."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(text.encode())}


class _TokenizedText:
    def to(self, _device: str) -> _TokenizedText:
        return self


class _ClipStub(types.ModuleType):
    def tokenize(self, _text: str, *, truncate: bool = False) -> _TokenizedText:
        del truncate
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
    required = Path(VENDOR_2D_SCRIPT)
    for candidate in candidates:
        if (candidate / required).exists():
            return candidate
    msg = "vendor/layout-gpt is not available"
    raise FileNotFoundError(msg)


def build_golden(vendor_root: Path | None = None) -> VendorGolden:
    """Execute vendor deterministic paths and return regenerated goldens."""
    root = find_vendor_root(vendor_root)
    vendor_2d = _load_vendor_module(root, VENDOR_2D_SCRIPT)
    vendor_parser = _load_vendor_module(root, VENDOR_PARSER_SCRIPT)

    train_records = cast(
        list[dict[str, object]],
        json.loads((root / COUNTING_TRAIN_JSON).read_text()),
    )
    val_records = cast(
        list[dict[str, object]],
        json.loads((root / COUNTING_VAL_JSON).read_text()),
    )
    candidates = train_records[:6]

    fixed_records = list(candidates)
    vendor_random = cast(VendorRandom, vendor_2d["random"])
    vendor_random.seed(DEFAULT_SEED)
    vendor_random.shuffle(fixed_records)
    fixed_records = fixed_records[:DEFAULT_K]

    args = cast(VendorArgs, vendor_2d["args"])
    args.canvas_size = DEFAULT_CANVAS_SIZE
    args.icl_type = "fixed-random"
    args.setting = "counting"
    args.gpt_input_length_limit = DEFAULT_INPUT_LENGTH_LIMIT

    tokenizer = ByteTokenizer()
    query = val_records[0]["prompt"]
    form_prompt_for_chatgpt = cast(
        VendorPromptFunction, vendor_2d["form_prompt_for_chatgpt"]
    )
    form_prompt_for_gpt3 = cast(VendorPromptFunction, vendor_2d["form_prompt_for_gpt3"])
    fixed_chat = form_prompt_for_chatgpt(
        text_input=query,
        top_k=DEFAULT_K,
        tokenizer=tokenizer,
        supporting_examples=fixed_records,
        features=None,
    )
    fixed_completion = form_prompt_for_gpt3(
        text_input=query,
        top_k=DEFAULT_K,
        tokenizer=tokenizer,
        supporting_examples=fixed_records,
        features=None,
    )

    args.icl_type = "k-similar"
    function_globals = form_prompt_for_gpt3.__globals__
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
    k_completion = form_prompt_for_gpt3(
        text_input=query,
        top_k=DEFAULT_K,
        tokenizer=tokenizer,
        supporting_examples=k_candidates,
        features=function_globals["features"],
    )
    features = cast(torch.Tensor, function_globals["features"])
    _, k_indices = (
        (100.0 * torch.tensor([[0.0, 1.0]]) @ features.T)
        .softmax(dim=-1)[0]
        .topk(DEFAULT_K)
    )
    k_similar_ids = [k_candidates[index]["id"] for index in k_indices.tolist()]

    parser_lines = [
        "clock {height: 15px; width: 14px; top: 24px; left: 25px; }",
        "clock2 {height: 50px; width: 40px; top: 48px; left: 40px; }",
        "clock {height: 10px; width: 10px; top: 0px; left: 64px; }",
    ]
    parse_layout = cast(VendorParse2D, vendor_parser["parse_layout"])
    parsed_2d = [
        parse_layout(line, canvas_size=DEFAULT_CANVAS_SIZE) for line in parser_lines
    ]
    parser_3d_lines = [
        "chair {length: 1.5m; width: 0.5m; height: 1m; orientation: 90 degrees; left: 2m; top: 3m; depth: 0m;}"
    ]
    parse_3d_layout = cast(VendorParse3D, vendor_parser["parse_3D_layout"])
    parsed_3d = [parse_3d_layout(line, unit="m") for line in parser_3d_lines]

    return {
        "metadata": {
            "source": "vendor/layout-gpt",
            "seed": DEFAULT_SEED,
            "k": DEFAULT_K,
            "canvas_size": DEFAULT_CANVAS_SIZE,
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


def _load_vendor_module(vendor_root: Path, filename: str) -> dict[str, object]:
    with _vendor_runtime(vendor_root, filename):
        return runpy.run_path(str(vendor_root / filename))


@contextlib.contextmanager
def _vendor_runtime(vendor_root: Path, filename: str) -> Iterator[None]:
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_modules = {
        module.value: sys.modules.get(module.value) for module in StubbedVendorModule
    }
    sys.argv = [filename, "--icl_type", "fixed-random"]
    sys.path.insert(0, str(vendor_root))
    sys.modules[StubbedVendorModule.clip.value] = _ClipStub(
        StubbedVendorModule.clip.value
    )
    sys.modules[StubbedVendorModule.openai.value] = types.ModuleType(
        StubbedVendorModule.openai.value
    )
    transformers = types.ModuleType(StubbedVendorModule.transformers.value)
    setattr(transformers, "StoppingCriteria", _StoppingCriteria)
    setattr(transformers, "GPT2TokenizerFast", object)
    setattr(transformers, "LlamaForCausalLM", object)
    setattr(transformers, "LlamaTokenizer", object)
    sys.modules[StubbedVendorModule.transformers.value] = transformers
    sys.modules[StubbedVendorModule.cssutils.value] = _CssUtilsStub(
        StubbedVendorModule.cssutils.value
    )
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
