from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_evaluate_full_run_module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_full_run.py"
    spec = importlib.util.spec_from_file_location("evaluate_full_run", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cgl_s5_protocol_uses_cgl_config_and_seed_fixture_name():
    module = _load_evaluate_full_run_module()

    spec = module.DATASET_EVAL_SPECS["cgl"]

    assert spec.config_name == "cgl.yaml"
    assert spec.package_dataset_name == "cgl"
    assert spec.fixture_dataset_name == "cgl"
    assert f"seed_{7}_{spec.fixture_dataset_name}_unanno_test.pt" == (
        "seed_7_cgl_unanno_test.pt"
    )
