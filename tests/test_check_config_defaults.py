from pathlib import Path
from collections.abc import Callable
import importlib.util


def load_checker() -> Callable[[Path], int]:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_config_defaults.py"
    )
    spec = importlib.util.spec_from_file_location("check_config_defaults", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_config_defaults.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_config_defaults


check_config_defaults = load_checker()


def write_source(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_check_config_defaults_rejects_synthesized_config(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "config = config or ExampleConfig()\n",
    )

    assert check_config_defaults(tmp_path) == 1


def test_check_config_defaults_allows_artifact_derived_config(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models/example/src/example/pipeline.py",
        "config = config or model.config\n",
    )

    assert check_config_defaults(tmp_path) == 0


def test_check_config_defaults_rejects_nested_config_constructor(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "lib/example/src/example/processor.py",
        "tokenizer = tokenizer or Tokenizer(ExampleConfig())\n",
    )

    assert check_config_defaults(tmp_path) == 1
