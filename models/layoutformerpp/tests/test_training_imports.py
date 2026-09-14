import subprocess
import sys
import textwrap


def _assert_training_namespace_contract(*, missing_lightning: bool) -> None:
    script = f"""
import importlib.util
import sys

missing_lightning = {missing_lightning!r}
if missing_lightning:
    _real_find_spec = importlib.util.find_spec

    def _find_spec(name, *args, **kwargs):
        if name == "lightning":
            return None
        return _real_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _find_spec

import layoutformerpp
assert "layoutformerpp.training" not in sys.modules
import layoutformerpp.training as training

eager_exports = (
    "LayoutFormerPPTrainingRecipe",
    "TRAINING_RECIPES",
    "TRAINING_RECIPES_BY_NAME",
    "get_training_recipe",
    "LayoutFormerPPWarmupLR",
)
for name in eager_exports:
    assert name in training.__dict__
assert "__all__" not in training.__dict__
assert "__getattr__" not in training.__dict__
assert "__dir__" not in training.__dict__
if missing_lightning:
    assert "LayoutFormerPPTrainingModule" not in training.__dict__
    assert "layoutformerpp.training.lightning_module" not in sys.modules
    assert "lightning" not in sys.modules
else:
    assert "LayoutFormerPPTrainingModule" in training.__dict__
    assert "layoutformerpp.training.lightning_module" in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_training_namespace_conditionally_exports_training_classes() -> None:
    _assert_training_namespace_contract(missing_lightning=True)
    _assert_training_namespace_contract(missing_lightning=False)
