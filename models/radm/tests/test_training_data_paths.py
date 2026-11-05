from __future__ import annotations

import runpy
from collections.abc import Callable
import os
from pathlib import Path
import tempfile
from typing import cast

import pytest

from radm.training.data_paths import validate_dot_free_data_root


def test_dot_free_data_root_preserves_symlink_alias() -> None:
    with tempfile.TemporaryDirectory(prefix="radm-cgl-", dir="/tmp") as temp_dir:
        root = Path(temp_dir)
        target = root / ".cache" / "cgl"
        target.mkdir(parents=True)
        alias = root / "radm_cgl"
        alias.symlink_to(Path(".cache") / "cgl", target_is_directory=True)

        assert validate_dot_free_data_root(str(alias)) == str(alias)
        assert os.readlink(alias) == os.path.join(".cache", "cgl")


@pytest.mark.parametrize("suffix", (".cache/cgl", "./cgl", "../cgl", "cgl.v1"))
def test_dot_free_data_root_rejects_dotted_components(suffix: str) -> None:
    with pytest.raises(ValueError, match="dot-free path components"):
        validate_dot_free_data_root(f"/data/{suffix}")


def test_s5_launcher_passes_the_same_alias_to_both_arms(tmp_path: Path) -> None:
    root = Path(__file__).parents[3]
    launcher = runpy.run_path(
        str(root / "models/radm/scripts/train_cgl_s5.py"),
        run_name="radm_cgl_s5_test",
    )
    build_command = cast(
        Callable[..., tuple[list[str], Path]], launcher["build_command"]
    )
    package_command, _ = build_command(
        arm="package",
        seed=2,
        data_root="/data/radm_cgl",
        output_dir=tmp_path / "package",
    )
    original_command, _ = build_command(
        arm="original",
        seed=2,
        data_root="/data/radm_cgl",
        output_dir=tmp_path / "original",
    )

    assert "--data.train_image_root=/data/radm_cgl/images/train" in package_command
    assert "--data.train_text_feature_root=/data/radm_cgl/text_features/train" in package_command
    assert original_command[original_command.index("DATASETS.DATASET_PATH") + 1] == "/data/radm_cgl"
    assert original_command[original_command.index("DATASETS.TEXT_FEATURE_PATH") + 1] == "/data/radm_cgl/text_features"
