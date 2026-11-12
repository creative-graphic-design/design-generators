from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys

import pytest


def load_checker() -> Callable[[Path], int]:
    """Load the Python path-component checker from the repository scripts."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_path_components.py"
    )
    spec = importlib.util.spec_from_file_location("check_path_components", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_path_components.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.check_path_components


check_path_components = load_checker()


def write_source(root: Path, *parts: str, source: str) -> None:
    """Write a Python source file beneath a temporary root."""
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_checker_rejects_path_constructor_and_operand_strings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "paths.py",
        source=(
            "from pathlib import Path as P, PurePath as PP\n"
            'first = P("lib/module.py")\n'
            'second = PP("models\\\\module.py")\n'
            'third = root / "src/model.py"\n'
            'fourth = P("one", "two/three")\n'
        ),
    )

    assert check_path_components(tmp_path) == 1
    stderr = capsys.readouterr().err
    assert all(
        f"models/example/src/paths.py:{line}:" in stderr for line in (2, 3, 4, 5)
    )


def test_checker_rejects_separator_bearing_f_string_operand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "paths.py",
        source=('path = root / f"{dataset_name}-max{max_seq_length}/processed"\n'),
    )

    assert check_path_components(tmp_path) == 1
    assert "models/example/src/paths.py:1:" in capsys.readouterr().err


def test_checker_rejects_pathlib_module_aliases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "aliases.py",
        source=(
            "import pathlib as pl\n"
            'first = pl.PurePosixPath("lib/module.py")\n'
            "second = pl.Path(__file__).parent.parent\n"
        ),
    )

    assert check_path_components(tmp_path) == 1
    stderr = capsys.readouterr().err
    assert "models/example/src/aliases.py:2:" in stderr


def test_checker_rejects_repeated_file_anchor_parent_chains(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "parents.py",
        source=(
            "from pathlib import Path\n"
            "first = Path(__file__).parent.parent\n"
            "second = Path(__file__).parent.parent.parent\n"
            "third = Path(__file__).resolve().parent.parent\n"
        ),
    )

    assert check_path_components(tmp_path) == 1
    stderr = capsys.readouterr().err
    assert all(f"models/example/src/parents.py:{line}:" in stderr for line in (2, 3, 4))


def test_checker_allows_componentwise_paths_and_single_parent(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "allowed.py",
        source=(
            "from pathlib import Path\n"
            'first = Path("single-component")\n'
            'second = root / "one" / "two"\n'
            'third = Path(__file__).parents[1] / "assets"\n'
            'fourth = Path(__file__).parent / "assets"\n'
        ),
    )

    assert check_path_components(tmp_path) == 0


def test_checker_excludes_vendor_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "main.py",
        source='path = root / "assets/image.png"\n',
    )
    write_source(
        tmp_path,
        "models",
        "example",
        "src",
        "vendor",
        "copied.py",
        source='path = root / "assets/image.png"\n',
    )

    assert check_path_components(tmp_path) == 1
    stderr = capsys.readouterr().err
    assert "models/example/src/main.py:1:" in stderr
    assert "models/example/src/vendor/copied.py:" not in stderr


def test_checker_excludes_cache_directories_with_non_utf8_files(tmp_path: Path) -> None:
    cached_source = tmp_path / ".cache" / "runtime" / "joblib" / "test.py"
    cached_source.parent.mkdir(parents=True)
    cached_source.write_bytes(b"\xa4")

    assert check_path_components(tmp_path) == 0
