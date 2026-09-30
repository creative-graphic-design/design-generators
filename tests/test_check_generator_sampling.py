from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys

import pytest


def load_checker() -> Callable[[Path], int]:
    """Load the generator sampling checker from the repository scripts."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_generator_sampling.py"
    )
    spec = importlib.util.spec_from_file_location("check_generator_sampling", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load check_generator_sampling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.check_generator_sampling


check_generator_sampling = load_checker()


def write_source(root: Path, relative: str, text: str) -> None:
    """Write one temporary package source file."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_checker_rejects_multiline_generator_draws(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models/example/src/example.py",
        """
import torch

values = torch.randn(
    (2, 3),
    generator=generator,
)
indices = torch.multinomial(probabilities, 1, generator=generator)
""",
    )

    assert check_generator_sampling(tmp_path) == 1


def test_checker_rejects_import_alias_and_in_place_draw(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "lib/example/src/example.py",
        """
from torch import randn as draw

draw((2,), generator=generator)
noise.normal_(generator=generator)
""",
    )

    assert check_generator_sampling(tmp_path) == 1


def test_checker_rejects_generator_positional_and_keyword_expansions(
    tmp_path: Path,
) -> None:
    write_source(
        tmp_path,
        "models/example/src/example.py",
        """
import torch

torch.Generator("cuda")
torch.Generator(**generator_kwargs)
""",
    )

    assert check_generator_sampling(tmp_path) == 1


@pytest.mark.parametrize(
    "expression",
    [
        "torch.empty(3).normal_(generator=g)",
        "probs.multinomial(1, generator=g)",
        "probs.bernoulli(generator=g)",
        "torch.ones(3).multinomial(1, generator=g)",
        "x.cauchy_(generator=g)",
    ],
)
def test_checker_rejects_each_chained_or_tensor_method_draw(
    tmp_path: Path, expression: str
) -> None:
    write_source(tmp_path, "models/example/src/example.py", expression)

    assert check_generator_sampling(tmp_path) == 1


def test_checker_rejects_like_draws_and_init_initializers(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models/example/src/example.py",
        """
import torch

torch.randn_like(values, generator=g)
torch.rand_like(values, generator=g)
torch.randint_like(values, 0, 2, generator=g)
torch.nn.init.trunc_normal_(values, generator=g)
torch.nn.init.kaiming_uniform_(values, generator=g)
""",
    )

    assert check_generator_sampling(tmp_path) == 1


@pytest.mark.parametrize(
    "expression", ['"cuda"', "target_device", 'torch.device("cpu")']
)
def test_checker_rejects_non_cpu_or_dynamic_generator_device(
    tmp_path: Path, expression: str
) -> None:
    write_source(
        tmp_path,
        "models/example/src/example.py",
        f"import torch\ngenerator = torch.Generator(device={expression})\n",
    )

    assert check_generator_sampling(tmp_path) == 1


def test_checker_allows_cpu_and_generator_free_sampling(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "models/example/src/example.py",
        """
import torch

generator = torch.Generator()
cpu_generator = torch.Generator(device="cpu")
values = torch.randn((2, 3))
""",
    )

    assert check_generator_sampling(tmp_path) == 0


def test_checker_excludes_shared_helper_path(tmp_path: Path) -> None:
    write_source(
        tmp_path,
        "lib/laygen/src/laygen/common/randomness.py",
        "import torch\ntorch.randn((2,), generator=generator)\n",
    )

    assert check_generator_sampling(tmp_path) == 0


def test_checker_reports_syntax_errors(tmp_path: Path) -> None:
    write_source(tmp_path, "models/example/src/example.py", "def broken(:\n")

    assert check_generator_sampling(tmp_path) == 1
