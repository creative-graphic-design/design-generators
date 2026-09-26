from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType

import pytest


def load_audit_architecture() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_architecture.py"
    )
    spec = importlib.util.spec_from_file_location("audit_architecture", module_path)
    assert spec is not None
    assert isinstance(spec.loader, SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_architecture = load_audit_architecture()


def write_project(path: Path, content: str) -> None:
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text(content, encoding="utf-8")


def test_normalize_requirement_name_handles_extras_and_versions() -> None:
    assert (
        audit_architecture.normalize_requirement_name("laygen[agents]>=0.1") == "laygen"
    )
    assert (
        audit_architecture.normalize_requirement_name("traingen-parity")
        == "traingen-parity"
    )
    assert (
        audit_architecture.normalize_requirement_name("My_Package~=1.2") == "my-package"
    )

    with pytest.raises(ValueError, match="cannot parse requirement name"):
        audit_architecture.normalize_requirement_name("  @invalid")


def test_build_report_discovers_root_members_edges_metadata_and_hotspots(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "root"
dependencies = ["Example_Model", "@invalid"]

[dependency-groups]
tooling = ["trainer"]

[tool.uv.workspace]
members = ["lib/*", "models/*"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    write_project(
        tmp_path / "lib" / "shared",
        """
[project]
name = "shared"
dependencies = []
""".strip()
        + "\n",
    )
    write_project(
        tmp_path / "lib" / "trainer",
        """
[project]
name = "trainer"
dependencies = ["shared"]
""".strip()
        + "\n",
    )
    write_project(
        tmp_path / "models" / "example",
        """
[project]
name = "example-model"
dependencies = ["shared>=0.1", "requests>=2"]

[project.optional-dependencies]
training = ["trainer[lightning]", "shared"]

[tool.design-generators]
framework = "diffusers"
task = "layout-generation"
""".strip()
        + "\n",
    )

    source = tmp_path / "models" / "example" / "src" / "example"
    source.mkdir(parents=True)
    (source / "small.py").write_text("x = 1\n", encoding="utf-8")
    (source / "large.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    tests = tmp_path / "models" / "example" / "tests"
    tests.mkdir()
    (tests / "test_example.py").write_text(
        "def test_example():\n    assert True\n", encoding="utf-8"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "large_script.py").write_text(
        "a = 1\nb = 2\nc = 3\nd = 4\n", encoding="utf-8"
    )

    report = audit_architecture.build_report(tmp_path, hotspot_limit=2)

    assert [member.name for member in report.members] == [
        "root",
        "shared",
        "trainer",
        "example-model",
    ]
    assert report.members[0].kind == "root"
    assert report.members[0].path == "."
    example = report.members[3]
    assert example.kind == "model"
    assert example.framework == "diffusers"
    assert example.task == "layout-generation"
    assert example.source_files == 2
    assert example.source_lines == 4
    assert example.test_files == 1
    assert example.test_lines == 2
    assert example.largest_source_file == "models/example/src/example/large.py"
    assert example.largest_source_lines == 3

    assert report.dependency_edges == (
        audit_architecture.DependencyEdge("example-model", "shared", "core"),
        audit_architecture.DependencyEdge("example-model", "shared", "extra:training"),
        audit_architecture.DependencyEdge("example-model", "trainer", "extra:training"),
        audit_architecture.DependencyEdge("root", "example-model", "core"),
        audit_architecture.DependencyEdge("root", "trainer", "group:tooling"),
        audit_architecture.DependencyEdge("trainer", "shared", "core"),
    )
    assert report.warnings == ("root (core): @invalid",)
    assert "- root (core): @invalid" in audit_architecture.render_text(report)
    assert report.hotspots == (
        audit_architecture.SourceHotspot("scripts/large_script.py", 4),
        audit_architecture.SourceHotspot("models/example/src/example/large.py", 3),
    )


def test_report_payload_and_text_are_stable_for_non_project_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv.workspace]
members = ["lib/*", "models/*"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    write_project(
        tmp_path / "lib" / "shared",
        """
[project]
name = "shared"
dependencies = []
""".strip()
        + "\n",
    )

    report = audit_architecture.build_report(tmp_path, hotspot_limit=0)
    payload = audit_architecture.report_payload(report)

    assert payload["summary"] == {
        "workspace_members": 1,
        "roots": 0,
        "libraries": 1,
        "models": 0,
        "dependency_edges": 0,
    }
    assert payload["warnings"] == []
    encoded = json.dumps(payload, sort_keys=True)
    assert '"name": "shared"' in encoded
    assert audit_architecture.render_text(report).startswith(
        "Workspace: 1 members (0 root, 1 libraries, 0 models)\n"
        "Member stats cover only src/ and tests/; hotspots scan Python files under "
        "lib/, models/, and scripts/ except paths containing tests or vendor.\n"
    )


def test_hotspots_ignore_tests_and_vendor_components(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "root"\n', encoding="utf-8"
    )
    source = tmp_path / "lib" / "shared" / "src" / "shared"
    source.mkdir(parents=True)
    (source / "included.py").write_text("a = 1\n", encoding="utf-8")

    ignored_tests = tmp_path / "lib" / "shared" / "tests"
    ignored_tests.mkdir()
    (ignored_tests / "ignored.py").write_text(
        "\n".join("a = 1" for _ in range(4)), encoding="utf-8"
    )

    ignored_vendor = tmp_path / "models" / "example" / "vendor"
    ignored_vendor.mkdir(parents=True)
    (ignored_vendor / "ignored.py").write_text(
        "\n".join("a = 1" for _ in range(5)), encoding="utf-8"
    )

    report = audit_architecture.build_report(tmp_path)

    assert report.hotspots == (
        audit_architecture.SourceHotspot("lib/shared/src/shared/included.py", 1),
    )


def test_main_rejects_negative_hotspot_limit() -> None:
    with pytest.raises(SystemExit):
        audit_architecture.main(["--top", "-1"])


def test_import_audit_classifies_direct_imports_and_checks_root_declarations(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "root"
dependencies = ["requests"]

[project.optional-dependencies]
evaluation = ["evaluate", "pytorch-fid"]

[dependency-groups]
dev = ["pytest"]

[tool.uv.workspace]
members = ["lib/*"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    write_project(
        tmp_path / "lib" / "shared",
        """
[project]
name = "shared-lib"
""".strip()
        + "\n",
    )
    package = tmp_path / "lib" / "shared" / "src" / "shared_lib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "local_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (scripts / "tool.py").write_text(
        "\n".join(
            (
                "import argparse",
                "from local_helper import VALUE",
                "from shared_lib import VALUE as SHARED_VALUE",
                "import yaml",
                "from PIL import Image",
                "import evaluate",
                "from pytorch_fid import fid_score",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_tool.py").write_text(
        "\n".join(
            (
                "import numpy",
                "import pytest",
                "import torch",
                "from jaxtyping import Float",
                "from pathlib import Path",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (tests / "test_fixture.py").write_text(
        'SOURCE = """\nimport torch\n"""\n',
        encoding="utf-8",
    )

    report = audit_architecture.build_report(tmp_path, hotspot_limit=0)
    records = {
        (record.path, record.module): (
            record.category,
            record.distribution,
            record.declared_scopes,
        )
        for record in report.import_audit.records
    }

    assert records == {
        ("scripts/tool.py", "PIL"): ("third-party-distribution", "Pillow", ()),
        ("scripts/tool.py", "argparse"): ("stdlib", None, ()),
        ("scripts/tool.py", "evaluate"): (
            "third-party-distribution",
            "evaluate",
            ("extra:evaluation",),
        ),
        ("scripts/tool.py", "local_helper"): ("local-root-script", None, ()),
        ("scripts/tool.py", "pytorch_fid"): (
            "third-party-distribution",
            "pytorch-fid",
            ("extra:evaluation",),
        ),
        ("scripts/tool.py", "shared_lib"): ("workspace-package", None, ()),
        ("scripts/tool.py", "yaml"): ("third-party-distribution", "PyYAML", ()),
        ("tests/test_tool.py", "pathlib"): ("stdlib", None, ()),
        ("tests/test_tool.py", "jaxtyping"): (
            "third-party-distribution",
            "jaxtyping",
            (),
        ),
        ("tests/test_tool.py", "numpy"): (
            "third-party-distribution",
            "numpy",
            (),
        ),
        ("tests/test_tool.py", "pytest"): (
            "third-party-distribution",
            "pytest",
            ("group:dev",),
        ),
        ("tests/test_tool.py", "torch"): ("third-party-distribution", "torch", ()),
    }
    assert not any(
        record.path == "tests/test_fixture.py" for record in report.import_audit.records
    )
    assert report.import_audit.undeclared_distributions == (
        "jaxtyping",
        "numpy",
        "Pillow",
        "PyYAML",
        "torch",
    )

    payload = audit_architecture.report_payload(report)
    assert payload["import_audit"] == {
        "records": [
            {
                "category": category,
                "declared_scopes": list(scopes),
                "distribution": distribution,
                "module": module,
                "path": path,
            }
            for (path, module), (category, distribution, scopes) in sorted(
                records.items()
            )
        ],
        "undeclared_distributions": [
            "jaxtyping",
            "numpy",
            "Pillow",
            "PyYAML",
            "torch",
        ],
    }
    assert audit_architecture.render_text(report).endswith(
        "Undeclared root-tooling distributions:\n"
        "- jaxtyping\n"
        "- numpy\n"
        "- Pillow\n"
        "- PyYAML\n"
        "- torch\n"
    )
