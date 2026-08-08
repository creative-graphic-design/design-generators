"""Generate PosterLlama prompt and parser references from the original source."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import TypedDict, cast

import torch
from posterllama import PosterLlamaConfig, PosterLlamaProcessor


CONDITION_TO_SOURCE_KEY = {
    "label": "cond_cate_to_size_pos",
    "label_size": "cond_cate_size_to_pos",
    "completion": "cond_random_mask",
    "refinement": "cond_cate_pos_to_size",
    "unconditional": "unconditional",
}


class ParserIntermediates(TypedDict, total=False):
    """Parser diagnostics used by the reference generator."""

    bbox_ltwh: list[tuple[float, float, float, float]]


ParserDictValue = (
    torch.Tensor
    | dict[int, str]
    | ParserIntermediates
    | list[str]
    | list[tuple[float, float, float, float]]
    | tuple[int, int]
    | str
    | int
    | float
    | bool
    | None
)

PARSER_MARKUP = (
    '<body> <svg width="100" height="200"> '
    '<rect data-category="Text", x="10", y="20", width="30", height="40"/> '
    "</svg> </body>"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        "--vendor-root",
        dest="source_root",
        type=Path,
        required=True,
        help="Original PosterLlama repository root.",
    )
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Reference JSON to write."
    )
    return parser.parse_args()


def main() -> None:
    """Write source-generated prompt/parser references and package comparisons."""
    args = parse_args()
    source_root = args.source_root.resolve()
    global_var = _load_source_module(
        source_root / "helper" / "global_var.py",
        "posterllama_source_global_var",
        source_root=source_root,
    )
    html_to_ui = _load_source_module(
        source_root / "html_to_ui.py",
        "posterllama_source_html_to_ui",
        source_root=source_root,
    )
    source_id2label = {
        int(key): str(value) for key, value in global_var.DATASET_META["cgl"].items()
    }
    processor = PosterLlamaProcessor.from_config(
        PosterLlamaConfig(
            canvas_size=(100, 200),
            dataset_name="cgl",
            id2label=cast(dict[int | str, str], source_id2label),
        )
    )
    prompts = []
    for condition, source_key in CONDITION_TO_SOURCE_KEY.items():
        expected = _source_prompt(global_var, source_key, _source_markup(condition))
        actual = cast(
            str,
            processor.build_prompt(
                condition_type=condition,
                labels=["Text"] if condition != "unconditional" else None,
                bbox=[[[0.25, 0.2, 0.3, 0.2]]]
                if condition != "unconditional"
                else None,
                canvas_size=(100, 200),
            ),
        )
        if actual != expected:
            raise AssertionError(f"Prompt mismatch for {condition}")
        prompts.append(
            {
                "condition_type": condition,
                "source_key": source_key,
                "prompt": expected,
            }
        )
    source_bbox, source_labels = html_to_ui.get_bbox(PARSER_MARKUP)
    parsed = processor.parse_output(
        PARSER_MARKUP,
        output_type="dict",
        return_intermediates=True,
    )
    parsed_dict = cast(dict[str, ParserDictValue], parsed)
    labels = cast(torch.Tensor, parsed_dict["labels"]).tolist()
    intermediates = cast(ParserIntermediates, parsed_dict["intermediates"])
    bbox_ltwh = intermediates["bbox_ltwh"]
    if bbox_ltwh != [tuple(item) for item in source_bbox]:
        raise AssertionError("Parser bbox mismatch")
    if labels != [source_labels]:
        raise AssertionError("Parser label mismatch")
    payload = {
        "source_root": str(source_root),
        "prompt_references": prompts,
        "parser_reference": {
            "markup": PARSER_MARKUP,
            "bbox_ltwh": source_bbox,
            "labels": source_labels,
        },
        "comparison_counts": {
            "prompts": len(prompts),
            "rectangles": len(source_bbox),
            "labels": len(source_labels),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output_json)


def _source_prompt(module: ModuleType, source_key: str, bbox_html: str) -> str:
    task_instruction = module.TASK_INSTRUCTION["cgl"]
    instruction = module.INSTRUCTION[source_key].format(bbox_html=bbox_html)
    return f"{task_instruction}{instruction} <MID>"


def _source_markup(condition: str) -> str:
    rect = {
        "label": '<rect data-category="Text", x="<FILL_1>", y="<FILL_2>", width="<FILL_3>", height="<FILL_4>"/>',
        "label_size": '<rect data-category="Text", x="<FILL_1>", y="<FILL_2>", width="30", height="40"/>',
        "completion": '<rect data-category="Text", x="<FILL_1>", y="<FILL_2>", width="<FILL_3>", height="<FILL_4>"/>',
        "refinement": '<rect data-category="Text", x="10", y="20", width="<FILL_1>", height="<FILL_2>"/>',
        "unconditional": "",
    }[condition]
    return f'<body> <svg width="100" height="200"> {rect} </svg> </body>'


def _load_source_module(
    path: Path,
    module_name: str,
    *,
    source_root: Path,
) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    _install_import_stubs()
    sys.path.insert(0, str(source_root))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(source_root))


def _install_import_stubs() -> None:
    for name in ("matplotlib", "matplotlib.pyplot", "pandas"):
        sys.modules.setdefault(name, ModuleType(name))
    tqdm_module = sys.modules.setdefault("tqdm", ModuleType("tqdm"))
    if not hasattr(tqdm_module, "tqdm"):
        tqdm_module.tqdm = lambda value, *args, **kwargs: value


if __name__ == "__main__":
    main()
