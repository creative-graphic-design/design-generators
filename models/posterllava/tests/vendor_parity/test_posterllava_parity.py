from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

from PIL import Image
import pytest
import torch
from transformers import CLIPImageProcessor, PreTrainedTokenizerBase

from laygen.common.testing import (
    assert_layout_output_schema,
    skip_or_fail_vendor_parity,
)
from laygen.modeling_outputs import LayoutGenerationOutput
from posterllava.generation_posterllava import IMAGE_TOKEN_INDEX, tokenizer_image_token
from posterllava.image_processing_posterllava import PosterLlavaImageProcessor
from posterllava.processing_posterllava import (
    DEFAULT_DOMAIN_NAME,
    PosterLlavaJsonElement,
    PosterLlavaProcessor,
)


VENDOR_ROOT = Path("vendor/posterllava")
REFERENCE_JSON = Path(".cache/posterllava/reference/posterllava_reference.json")


class TinyEncoding(dict[str, list[int]]):
    @property
    def input_ids(self) -> list[int]:
        return self["input_ids"]


class TinyTokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True) -> TinyEncoding:
        ids = [ord(char) % 37 + 2 for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids]
        return TinyEncoding(input_ids=ids)


def _vendor_source_path() -> Path:
    required = [
        VENDOR_ROOT / "llava" / "conversation.py",
        VENDOR_ROOT / "llava" / "mm_utils.py",
        VENDOR_ROOT / "llava" / "constants.py",
        VENDOR_ROOT / "data" / "prompt_template.txt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "PosterLLaVA original source is not checked out.",
            missing_paths=missing,
            regeneration_hint="git submodule update --init vendor/posterllava",
        )
    return VENDOR_ROOT.resolve()


def _load_vendor_module(module_name: str, relative_path: str) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        _vendor_source_path() / relative_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_vendor(module_name: str) -> ModuleType:
    vendor_root = _vendor_source_path()
    llava_pkg = ModuleType("llava")
    setattr(llava_pkg, "__path__", [(vendor_root / "llava").as_posix()])
    sys.modules["llava"] = llava_pkg
    _load_vendor_module("llava.constants", "llava/constants.py")
    if module_name == "llava.constants":
        return sys.modules[module_name]
    if module_name == "llava.conversation":
        return _load_vendor_module(module_name, "llava/conversation.py")
    if module_name == "llava.mm_utils":
        return _load_vendor_module(module_name, "llava/mm_utils.py")
    return importlib.import_module(module_name)


def _vendor_prompt(body: str, *, conv_mode: str) -> str:
    constants = _import_vendor("llava.constants")
    conversation = _import_vendor("llava.conversation")
    conv = conversation.conv_templates[conv_mode].copy()
    inp = f"{constants.DEFAULT_IMAGE_TOKEN}\n{body}"
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    return cast(str, conv.get_prompt())


def _vendor_cli_parse(outputs: str) -> list[dict[str, object]]:
    lo: int | None = None
    hi: int | None = None
    for idx in range(len(outputs)):
        if idx < len(outputs) - 1 and outputs[idx : idx + 2] == "[{":
            lo = idx
        elif idx > 1 and outputs[idx - 1 : idx + 1] == "}]":
            hi = idx
    if lo is None or hi is None:
        raise ValueError("PosterLLaVA output does not contain a CLI JSON object array")
    return cast(
        list[dict[str, object]], json.loads(outputs[lo : hi + 1].replace("'", '"'))
    )


def _vendor_prompt_body(
    *,
    num_elements: int,
    resolution: list[int],
    elements: Sequence[PosterLlavaJsonElement],
) -> str:
    template = (VENDOR_ROOT / "data" / "prompt_template.txt").read_text()
    human_value = template.format(
        N=num_elements,
        resolution=resolution,
        domain_name=DEFAULT_DOMAIN_NAME,
        json_data=json.dumps(list(elements)),
    )
    return "\n".join(human_value.split("\n")[1:])


@pytest.mark.vendor_parity
def test_cpu_prompt_token_parser_and_padding_contract_match_vendor_source() -> None:
    processor = PosterLlavaProcessor.from_config()
    elements: list[PosterLlavaJsonElement] = [
        {"label": "headline", "box": []},
        {"label": "logo", "box": [0.1, 0.2, 0.3, 0.4]},
    ]
    body = _vendor_prompt_body(
        num_elements=2,
        resolution=[320, 180],
        elements=elements,
    )

    prompt_v0 = processor.build_prompt(
        num_elements=2,
        canvas_size=(320, 180),
        elements=elements,
        conv_mode="llava_v0",
    )
    prompt_v1 = processor.build_prompt(
        num_elements=2,
        canvas_size=(320, 180),
        elements=elements,
        conv_mode="llava_v1",
    )

    assert prompt_v0 == _vendor_prompt(body, conv_mode="llava_v0")
    assert prompt_v1 == _vendor_prompt(body, conv_mode="llava_v1")

    tokenizer = cast(PreTrainedTokenizerBase, TinyTokenizer())
    vendor_mm_utils = _import_vendor("llava.mm_utils")
    vendor_ids = vendor_mm_utils.tokenizer_image_token(
        prompt_v0,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    package_ids = tokenizer_image_token(prompt_v0, tokenizer)
    assert torch.equal(package_ids, vendor_ids)

    image = Image.new("RGB", (4, 2), (255, 255, 255))
    vendor_processor = CLIPImageProcessor(
        image_mean=[0.5, 0.25, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"height": 4, "width": 4},
        crop_size={"height": 4, "width": 4},
    )
    package_processor = PosterLlavaImageProcessor(
        image_mean=[0.5, 0.25, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"height": 4, "width": 4},
        crop_size={"height": 4, "width": 4},
    )
    vendor_tensor = vendor_mm_utils.process_images(
        [image],
        vendor_processor,
        SimpleNamespace(image_aspect_ratio="pad"),
    )
    package_tensor = package_processor.preprocess(image)["pixel_values"]
    assert torch.equal(package_tensor, vendor_tensor)

    outputs = [
        "prefix [{'label': 'text', 'box': [0.1, 0.2, 0.3, 0.4]}] suffix",
        'Assistant: [{"label": "logo", "box": [0.0, 0.0, 1.0, 1.0]}]###',
        (
            "Sure! [{'label': 'headline', 'box': [0.2, 0.3, 0.8, 0.9]}, "
            "{'label': 'cta', 'box': [0.1, 0.1, 0.2, 0.2]}]"
        ),
    ]
    for text in outputs:
        assert processor.parse_output(text) == _vendor_cli_parse(text)


@pytest.mark.vendor_parity
def test_full_generation_reference_json_is_validated_when_present() -> None:
    if not REFERENCE_JSON.exists():
        skip_or_fail_vendor_parity(
            "PosterLLaVA full-generation reference JSON is not present.",
            missing_paths=[REFERENCE_JSON],
            regeneration_hint=(
                "uv run --package posterllava python "
                "models/posterllava/scripts/generate_reference_outputs.py --help"
            ),
        )

    data = json.loads(REFERENCE_JSON.read_text())
    if not isinstance(data, dict) or not data:
        raise AssertionError(
            "reference JSON must be a non-empty object keyed by sample id"
        )

    processor = PosterLlavaProcessor.from_config()
    checked_layouts = 0
    checked_elements = 0
    for sample_id, generations in data.items():
        if not isinstance(sample_id, str):
            raise AssertionError("reference sample ids must be strings")
        if not isinstance(generations, list) or not generations:
            raise AssertionError(f"reference sample {sample_id!r} has no generations")
        for layout in generations:
            if not isinstance(layout, list):
                raise AssertionError(
                    f"reference sample {sample_id!r} generation is not a layout list"
                )
            parsed = processor.parse_output(json.dumps(layout))
            output = cast(
                LayoutGenerationOutput, processor.decode_layout(json.dumps(layout))
            )
            assert_layout_output_schema(output, batch_size=1)
            checked_layouts += 1
            checked_elements += len(parsed)

    if checked_layouts == 0 or checked_elements == 0:
        raise AssertionError("reference JSON did not contain any generated elements")
