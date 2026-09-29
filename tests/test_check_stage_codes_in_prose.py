from collections.abc import Callable
import importlib.util
from pathlib import Path


def load_checker() -> Callable[[Path], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_stage_codes_in_prose.py"
    )
    spec = importlib.util.spec_from_file_location("check_stage_codes_in_prose", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_stage_codes_in_prose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_stage_codes_in_prose


check_stage_codes_in_prose = load_checker()


def write_source(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_check_stage_codes_in_prose_rejects_docstring(tmp_path: Path) -> None:
    write_source(tmp_path, "models/example/src/example.py", '"""S1"""\n')

    assert check_stage_codes_in_prose(tmp_path) == 1


def test_check_stage_codes_in_prose_rejects_comment(tmp_path: Path) -> None:
    write_source(tmp_path, "lib/example/source.py", "# S2\n")

    assert check_stage_codes_in_prose(tmp_path) == 1


def test_check_stage_codes_in_prose_ignores_ordinary_string_literals(
    tmp_path: Path,
) -> None:
    write_source(tmp_path, "models/example/src/example.py", 'value = "S3"\n')

    assert check_stage_codes_in_prose(tmp_path) == 0


def test_check_stage_codes_in_prose_ignores_identifiers(tmp_path: Path) -> None:
    write_source(tmp_path, "models/example/src/example.py", "s3_s4_reference = 1\n")

    assert check_stage_codes_in_prose(tmp_path) == 0


def test_check_stage_codes_in_prose_ignores_vendor_paths(tmp_path: Path) -> None:
    write_source(tmp_path, "models/example/vendor/reference.py", "# S4\n")

    assert check_stage_codes_in_prose(tmp_path) == 0
