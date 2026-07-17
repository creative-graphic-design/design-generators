import importlib.util
from pathlib import Path


def _load_generate_reference_module():
    script = (Path(__file__).parents[1] / "scripts" / "generate_reference.py").resolve()
    spec = importlib.util.spec_from_file_location("generate_reference", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_vendor_dir_prefers_valid_requested_path(tmp_path):
    module = _load_generate_reference_module()
    vendor_dir = tmp_path / "vendor" / "const-layout"
    (vendor_dir / "model").mkdir(parents=True)
    (vendor_dir / "model" / "layoutganpp.py").write_text("")

    assert module._resolve_vendor_dir(vendor_dir) == vendor_dir


def test_resolve_vendor_dir_falls_back_to_sibling_worktree(tmp_path):
    module = _load_generate_reference_module()
    repo_root = tmp_path / "design-generators=impl-layoutganpp"
    repo_root.mkdir()
    requested = Path("vendor/const-layout")
    sibling_vendor = tmp_path / "design-generators" / requested
    (sibling_vendor / "model").mkdir(parents=True)
    (sibling_vendor / "model" / "layoutganpp.py").write_text("")

    assert (
        module._resolve_vendor_dir(
            requested,
            repo_root=repo_root,
            cwd=repo_root,
        )
        == sibling_vendor
    )
