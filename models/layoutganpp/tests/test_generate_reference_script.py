from pathlib import Path

import pytest

from laygen.common.vendor import vendor_root


def test_vendor_root_prefers_valid_requested_path(tmp_path):
    vendor_dir = tmp_path / "vendor" / "const-layout"
    (vendor_dir / "model").mkdir(parents=True)
    (vendor_dir / "model" / "layoutganpp.py").write_text("")

    assert (
        vendor_root(
            "const-layout",
            path=vendor_dir,
            marker=Path("model") / "layoutganpp.py",
        )
        == vendor_dir
    )


def test_vendor_root_falls_back_to_sibling_worktree(tmp_path):
    repo_root = tmp_path / "design-generators=impl-layoutganpp"
    repo_root.mkdir()
    requested = Path("vendor/const-layout")
    sibling_vendor = tmp_path / "design-generators" / requested
    (sibling_vendor / "model").mkdir(parents=True)
    (sibling_vendor / "model" / "layoutganpp.py").write_text("")

    assert (
        vendor_root(
            "const-layout",
            path=requested,
            marker=Path("model") / "layoutganpp.py",
            repo_root=repo_root,
            cwd=repo_root,
        )
        == sibling_vendor
    )


def test_vendor_root_reports_uninitialized_submodule(tmp_path):
    repo_root = tmp_path / "design-generators"
    vendor_dir = repo_root / "vendor" / "const-layout"
    vendor_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        vendor_root(
            "const-layout",
            marker=Path("model") / "layoutganpp.py",
            repo_root=repo_root,
            cwd=repo_root,
        )
