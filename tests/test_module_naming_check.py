from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_check_module_naming() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_module_naming.py"
    )
    spec = importlib.util.spec_from_file_location("check_module_naming", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_module_naming = load_check_module_naming()


def write_module(root: Path, package: str, name: str) -> Path:
    path = root / "models" / package / "src" / package.replace("-", "_") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_direct_model_modules_discovers_only_package_root_modules(
    tmp_path: Path,
) -> None:
    direct = write_module(tmp_path, "layout-dm", "pipeline.py")
    nested = direct.parent / "training" / "cli.py"
    nested.parent.mkdir()
    nested.write_text("", encoding="utf-8")

    modules = check_module_naming.direct_model_modules(tmp_path)

    assert [module.path for module in modules] == [direct]


def test_violation_for_module_accepts_hf_core_package_suffix(
    tmp_path: Path,
) -> None:
    path = write_module(tmp_path, "layout-dm", "pipeline_layout_dm.py")
    module = check_module_naming.ModuleRecord("layout_dm", path, tmp_path)

    assert check_module_naming.violation_for_module(module) is None


def test_violation_for_module_rejects_hf_core_wrong_suffix(tmp_path: Path) -> None:
    path = write_module(tmp_path, "layout-transformer", "modeling_lt_compatible.py")
    module = check_module_naming.ModuleRecord("layout_transformer", path, tmp_path)

    violation = check_module_naming.violation_for_module(module)

    assert violation is not None
    assert "package suffix" in violation.reason


def test_violation_for_module_accepts_allowlist_categories(tmp_path: Path) -> None:
    conversion = write_module(tmp_path, "layout-dm", "conversion.py")
    geometry = write_module(tmp_path, "layout-dm", "geometry.py")

    assert (
        check_module_naming.violation_for_module(
            check_module_naming.ModuleRecord("layout_dm", conversion, tmp_path)
        )
        is None
    )
    assert (
        check_module_naming.violation_for_module(
            check_module_naming.ModuleRecord("layout_dm", geometry, tmp_path)
        )
        is None
    )


def test_violation_for_module_accepts_prompt_agent_modules(tmp_path: Path) -> None:
    path = write_module(tmp_path, "layoutprompter", "agent.py")
    module = check_module_naming.ModuleRecord("layoutprompter", path, tmp_path)

    assert check_module_naming.violation_for_module(module) is None


def test_check_module_naming_fails_on_new_and_stale_entries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_module(tmp_path, "layout-dm", "pipeline.py")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "models/layout-dm/src/layout_dm/old.py\told\n", encoding="utf-8"
    )

    assert check_module_naming.check_module_naming(tmp_path, baseline) == 1

    stderr = capsys.readouterr().err
    assert "New non-conforming model module filenames" in stderr
    assert "Stale module naming baseline entries" in stderr


def test_check_module_naming_passes_when_baseline_matches(tmp_path: Path) -> None:
    write_module(tmp_path, "layout-dm", "pipeline.py")
    baseline = tmp_path / "baseline.txt"
    check_module_naming.write_baseline(
        baseline,
        check_module_naming.current_entries(tmp_path),
    )

    assert check_module_naming.check_module_naming(tmp_path, baseline) == 0
