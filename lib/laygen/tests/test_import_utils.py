import sys
from types import ModuleType

import pytest

from laygen.common import import_utils


def _export(
    *, module: str = "test_import_utils.target", roots: frozenset[str] = frozenset()
):
    return import_utils.LazyClassExport(
        module=module,
        attribute="Target",
        optional_roots=roots,
    )


def _requesting_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("test_import_utils.requesting")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_resolve_lazy_class_rejects_unknown_name_with_canonical_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
):
    requesting_module = _requesting_module(monkeypatch)

    with pytest.raises(
        AttributeError,
        match=r"^module 'test_import_utils\.requesting' has no attribute 'Unknown'$",
    ):
        import_utils.resolve_lazy_class(
            "Unknown",
            module_name=requesting_module.__name__,
            distribution_name="test-distribution",
            exports={},
        )


def test_resolve_lazy_class_translates_a_direct_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
):
    requesting_module = _requesting_module(monkeypatch)
    missing = ModuleNotFoundError("No module named 'lightning'", name="lightning")

    def import_target(module_name: str) -> ModuleType:
        assert module_name == "test_import_utils.target"
        raise missing

    monkeypatch.setattr(import_utils, "import_module", import_target)

    with pytest.raises(ImportError) as caught:
        import_utils.resolve_lazy_class(
            "Target",
            module_name=requesting_module.__name__,
            distribution_name="test-distribution",
            exports={"Target": _export(roots=frozenset({"lightning"}))},
        )

    assert str(caught.value) == (
        "test_import_utils.requesting.Target requires the optional 'lightning' "
        "dependency; install the training extra with `pip install "
        "'test-distribution[training]'`."
    )
    assert caught.value.__cause__ is missing


def test_resolve_lazy_class_propagates_nested_missing_dependency_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    requesting_module = _requesting_module(monkeypatch)
    nested_failure = ModuleNotFoundError(
        "No module named 'torchmetrics'", name="torchmetrics"
    )

    def import_target(module_name: str) -> ModuleType:
        assert module_name == "test_import_utils.target"
        raise nested_failure

    monkeypatch.setattr(import_utils, "import_module", import_target)

    with pytest.raises(ModuleNotFoundError) as caught:
        import_utils.resolve_lazy_class(
            "Target",
            module_name=requesting_module.__name__,
            distribution_name="test-distribution",
            exports={"Target": _export(roots=frozenset({"lightning"}))},
        )

    assert caught.value is nested_failure


def test_resolve_lazy_class_rejects_a_non_class_target(
    monkeypatch: pytest.MonkeyPatch,
):
    requesting_module = _requesting_module(monkeypatch)
    target_module = ModuleType("test_import_utils.target")
    setattr(target_module, "Target", "not a class")
    monkeypatch.setattr(import_utils, "import_module", lambda _: target_module)

    with pytest.raises(TypeError, match="must resolve to a class"):
        import_utils.resolve_lazy_class(
            "Target",
            module_name=requesting_module.__name__,
            distribution_name="test-distribution",
            exports={"Target": _export()},
        )


def test_resolve_lazy_class_imports_once_and_caches_successful_resolution(
    monkeypatch: pytest.MonkeyPatch,
):
    requesting_module = _requesting_module(monkeypatch)
    target_module = ModuleType("test_import_utils.target")

    class Target:
        pass

    setattr(target_module, "Target", Target)
    imported_modules: list[str] = []

    def import_target(module_name: str) -> ModuleType:
        imported_modules.append(module_name)
        return target_module

    monkeypatch.setattr(import_utils, "import_module", import_target)
    exports = {"Target": _export()}

    first = import_utils.resolve_lazy_class(
        "Target",
        module_name=requesting_module.__name__,
        distribution_name="test-distribution",
        exports=exports,
    )
    second = import_utils.resolve_lazy_class(
        "Target",
        module_name=requesting_module.__name__,
        distribution_name="test-distribution",
        exports=exports,
    )

    assert first is Target
    assert second is Target
    assert imported_modules == ["test_import_utils.target"]
    assert requesting_module.Target is Target


def test_build_module_dir_is_stable_and_duplicate_free():
    exports = {
        "LazyTarget": _export(),
        "EagerTarget": _export(),
    }

    first = import_utils.build_module_dir(
        ["__name__", "EagerTarget", "EagerTarget"], exports
    )
    second = import_utils.build_module_dir(
        ["EagerTarget", "__name__", "EagerTarget"], exports
    )

    assert first == ["EagerTarget", "LazyTarget", "__name__"]
    assert second == first
    assert len(first) == len(set(first))
