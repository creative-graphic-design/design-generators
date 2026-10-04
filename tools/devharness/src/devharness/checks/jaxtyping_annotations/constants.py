"""Policy names and scan paths for annotation checks."""

from __future__ import annotations

from pathlib import Path

SCAN_GLOBS = (
    "models/*/src/**/*.py",
    "lib/*/src/**/*.py",
    "models/*/scripts/**/*.py",
    "lib/*/scripts/**/*.py",
    "tools/*/src/**/*.py",
    "scripts/**/*.py",
)

JAXTYPING_SHAPED_TYPES = {
    "Bool",
    "Complex",
    "Complex64",
    "Complex128",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "Inexact",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Integer",
    "Num",
    "Real",
    "Shaped",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
}

TORCH_TENSOR_TYPES = {
    "Tensor",
    "BoolTensor",
    "ByteTensor",
    "CharTensor",
    "DoubleTensor",
    "FloatTensor",
    "HalfTensor",
    "IntTensor",
    "LongTensor",
    "ShortTensor",
}


def baseline_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """Return the four stable repository baseline paths."""
    scripts = root / "scripts"
    return (
        scripts / "jaxtyping_baseline.txt",
        scripts / "jaxtyping_alias_baseline.txt",
        scripts / "object_annotation_baseline.txt",
        scripts / "weak_cast_type_baseline.txt",
    )
