from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import pytest
import torch

from laygen.modeling_outputs import LayoutGenerationOutput
from posterllama import PosterLlamaConfig, PosterLlamaProcessor


REPO_ROOT = Path(__file__).resolve().parents[4]
CONDITION_TO_SOURCE_KEY = {
    "label": "cond_cate_to_size_pos",
    "label_size": "cond_cate_size_to_pos",
    "completion": "cond_random_mask",
    "refinement": "cond_cate_pos_to_size",
    "unconditional": "unconditional",
}
PARSER_MARKUP = (
    '<body> <svg width="100" height="200"> '
    '<rect data-category="Text", x="10", y="20", width="30", height="40"/> '
    "</svg> </body>"
)


@pytest.mark.vendor_parity
def test_prompt_templates_match_original_source() -> None:
    source_root = _source_root()
    global_var = _load_source_module(
        source_root / "helper" / "global_var.py",
        "posterllama_source_global_var_prompt",
        source_root=source_root,
    )
    processor = _processor_from_source_labels(global_var)

    matches = 0
    for condition, source_key in CONDITION_TO_SOURCE_KEY.items():
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
        expected = _source_prompt(global_var, source_key, _source_markup(condition))
        assert actual.encode() == expected.encode()
        matches += 1
    assert matches == 5


@pytest.mark.vendor_parity
def test_parser_matches_original_get_bbox() -> None:
    source_root = _source_root()
    global_var = _load_source_module(
        source_root / "helper" / "global_var.py",
        "posterllama_source_global_var_parser",
        source_root=source_root,
    )
    html_to_ui = _load_source_module(
        source_root / "html_to_ui.py",
        "posterllama_source_html_to_ui",
        source_root=source_root,
    )
    processor = _processor_from_source_labels(global_var)

    source_bbox, source_labels = html_to_ui.get_bbox(PARSER_MARKUP)
    output = cast(
        LayoutGenerationOutput,
        processor.parse_output(
            PARSER_MARKUP,
            return_intermediates=True,
        ),
    )
    intermediates = cast(dict[str, object], output.intermediates)

    assert intermediates["bbox_ltwh"] == [tuple(item) for item in source_bbox]
    assert cast(torch.Tensor, output.labels).tolist() == [source_labels]
    assert len(source_bbox) == 1
    assert len(source_labels) == 1


def _source_root() -> Path:
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 for original-source parity checks")
    root = Path(
        os.environ.get("POSTERLLAMA_VENDOR_ROOT", REPO_ROOT / "vendor" / "posterllama")
    ).resolve()
    required = [
        root / "helper" / "global_var.py",
        root / "html_to_ui.py",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing PosterLlama source files: {missing}")
    return root


def _processor_from_source_labels(global_var: ModuleType) -> PosterLlamaProcessor:
    source_id2label = {
        int(key): str(value) for key, value in global_var.DATASET_META["cgl"].items()
    }
    return PosterLlamaProcessor.from_config(
        PosterLlamaConfig(
            canvas_size=(100, 200),
            dataset_name="cgl",
            id2label=cast(dict[int | str, str], source_id2label),
        )
    )


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
