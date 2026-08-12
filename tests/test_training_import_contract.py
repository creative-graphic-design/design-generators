"""Data-driven contract tests for package-local lazy training imports."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class TrainingImportContract:
    """Model-specific values for the shared lazy training contract."""

    package_name: str
    training_module: str
    lazy_exports: tuple[tuple[str, str], ...]
    top_level_blocked: tuple[str, ...]
    export_blocked: tuple[str, ...]
    optional_errors: tuple[tuple[str, str], ...]
    nested_target: str
    nested_export: str


CONTRACTS = (
    TrainingImportContract(
        package_name="cgb_dm",
        training_module="cgb_dm.training",
        lazy_exports=(
            ("CGBDMDataModule", "cgb_dm.training.datamodule"),
            ("CGBDMTrainingModule", "cgb_dm.training.lightning_module"),
        ),
        top_level_blocked=("lightning", "torchmetrics", "h5py", "h5pickle"),
        export_blocked=("lightning",),
        optional_errors=(
            ("CGBDMDataModule", "lightning"),
            ("CGBDMTrainingModule", "lightning"),
        ),
        nested_target="cgb_dm.training.lightning_module",
        nested_export="CGBDMTrainingModule",
    ),
    TrainingImportContract(
        package_name="dlt",
        training_module="dlt.training",
        lazy_exports=(
            ("DLTDataModule", "dlt.training.datamodule"),
            (
                "DLTReferenceEpochSamplingCallback",
                "dlt.training.callbacks",
            ),
            ("DLTTrainingModule", "dlt.training.lightning_module"),
            (
                "DLTWarmupCosineSchedulerFactory",
                "dlt.training.lightning_module",
            ),
        ),
        top_level_blocked=("lightning", "h5py"),
        export_blocked=("lightning", "h5py"),
        optional_errors=(
            ("DLTDataModule", "lightning"),
            ("DLTDataModule", "h5py"),
            ("DLTReferenceEpochSamplingCallback", "lightning"),
            ("DLTReferenceEpochSamplingCallback", "h5py"),
            ("DLTTrainingModule", "lightning"),
            ("DLTTrainingModule", "h5py"),
            ("DLTWarmupCosineSchedulerFactory", "lightning"),
            ("DLTWarmupCosineSchedulerFactory", "h5py"),
        ),
        nested_target="dlt.training.lightning_module",
        nested_export="DLTTrainingModule",
    ),
    TrainingImportContract(
        package_name="layout_dm",
        training_module="layout_dm.training",
        lazy_exports=(
            ("LayoutDMDataModule", "layout_dm.training.datamodule"),
            ("LayoutDMTrainingModule", "layout_dm.training.lightning_module"),
        ),
        top_level_blocked=("lightning", "datasets"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutDMDataModule", "lightning"),
            ("LayoutDMTrainingModule", "lightning"),
        ),
        nested_target="layout_dm.training.lightning_module",
        nested_export="LayoutDMTrainingModule",
    ),
    TrainingImportContract(
        package_name="layout_flow",
        training_module="layout_flow.training",
        lazy_exports=(
            ("LayoutFlowDataModule", "layout_flow.training.datamodule"),
            ("LayoutFlowTrainingModule", "layout_flow.training.lightning_module"),
        ),
        top_level_blocked=("lightning", "torchmetrics", "h5py", "h5pickle"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutFlowDataModule", "lightning"),
            ("LayoutFlowTrainingModule", "lightning"),
        ),
        nested_target="layout_flow.training.lightning_module",
        nested_export="LayoutFlowTrainingModule",
    ),
    TrainingImportContract(
        package_name="layoutdiffusion",
        training_module="layoutdiffusion.training",
        lazy_exports=(
            ("LayoutDiffusionDataModule", "layoutdiffusion.training.datamodule"),
            (
                "LayoutDiffusionTrainingModule",
                "layoutdiffusion.training.lightning_module",
            ),
        ),
        top_level_blocked=("lightning", "datasets"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutDiffusionDataModule", "lightning"),
            ("LayoutDiffusionTrainingModule", "lightning"),
        ),
        nested_target="layoutdiffusion.training.lightning_module",
        nested_export="LayoutDiffusionTrainingModule",
    ),
)


def _run_script(
    code: str,
    *,
    blocked: tuple[str, ...] = (),
    leaf_exports: tuple[tuple[str, str], ...] = (),
    leaf_errors: tuple[tuple[str, str], ...] = (),
) -> None:
    """Run one isolated contract assertion in a fresh interpreter."""
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.util
        import sys
        import types

        _BLOCKED = {blocked!r}
        _LEAF_EXPORTS = {leaf_exports!r}
        _LEAF_ERRORS = dict({leaf_errors!r})
        _LEAF_MODULES = {{module for _, module in _LEAF_EXPORTS}}
        _REQUESTED = []

        class _ImportBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(name + ".")
                    for name in _BLOCKED
                ):
                    raise ModuleNotFoundError(
                        f"No module named '{{fullname}}'", name=fullname
                    )
                return None

        class _LeafLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return types.ModuleType(spec.name)

            def exec_module(self, module):
                for export_name, module_name in _LEAF_EXPORTS:
                    if module_name == module.__name__:
                        setattr(
                            module,
                            export_name,
                            type(export_name, (), {{"__module__": module_name}}),
                        )

        class _LeafFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname not in _LEAF_MODULES and fullname not in _LEAF_ERRORS:
                    return None
                _REQUESTED.append(fullname)
                if fullname in _LEAF_ERRORS:
                    missing_root = _LEAF_ERRORS[fullname]
                    raise ModuleNotFoundError(
                        f"No module named '{{missing_root}}'", name=missing_root
                    )
                return importlib.util.spec_from_loader(fullname, _LeafLoader())

        if _BLOCKED:
            sys.meta_path.insert(0, _ImportBlocker())
        if _LEAF_MODULES or _LEAF_ERRORS:
            sys.meta_path.insert(0, _LeafFinder())
        """
    )
    subprocess.run(
        [sys.executable, "-c", bootstrap + textwrap.dedent(code)],
        check=True,
    )


def _assert_training_import_contract(contract: TrainingImportContract) -> None:
    """Run the shared lazy training checks for one model contract."""
    lazy_names = tuple(name for name, _ in contract.lazy_exports)
    lazy_modules = tuple(dict.fromkeys(module for _, module in contract.lazy_exports))
    export_modules = dict(contract.lazy_exports)

    _run_script(
        f"""
        import {contract.package_name}

        assert "{contract.training_module}" not in sys.modules
        for root in {contract.top_level_blocked!r}:
            assert root not in sys.modules
        """,
        blocked=contract.top_level_blocked,
    )

    _run_script(
        f"""
        import {contract.training_module} as training

        assert "laygen.common.import_utils" not in sys.modules
        assert "_LAZY_EXPORTS" not in training.__dict__
        for name in {lazy_names!r}:
            assert name in training.__all__
            assert name in dir(training)
            assert name not in training.__dict__
        for name in {lazy_modules!r}:
            assert name not in sys.modules
        """,
        blocked=contract.export_blocked,
    )

    _run_script(
        f"""
        import {contract.training_module} as training

        try:
            getattr(training, "Unknown")
        except AttributeError as error:
            assert str(error) == (
                "module '{contract.training_module}' has no attribute 'Unknown'"
            )
        else:
            raise AssertionError("unknown export unexpectedly resolved")
        """
    )

    for export_name, missing_root in contract.optional_errors:
        expected_message = (
            f"{contract.training_module}.{export_name} requires the optional "
            f"'{missing_root}' dependency; install the training extra with "
            f"`pip install '{contract.package_name.replace('_', '-')}[training]'`."
        )
        _run_script(
            f"""
            import {contract.training_module} as training

            try:
                getattr(training, "{export_name}")
            except ImportError as error:
                assert str(error) == {expected_message!r}
                assert isinstance(error.__cause__, ModuleNotFoundError)
                assert error.__cause__.name == "{missing_root}"
                assert _REQUESTED == ["{export_modules[export_name]}"]
            else:
                raise AssertionError("optional dependency unexpectedly resolved")
            """,
            blocked=(missing_root,),
            leaf_errors=((export_modules[export_name], missing_root),),
        )

    _run_script(
        f"""
        import {contract.training_module} as training
        try:
            getattr(training, "{contract.nested_export}")
        except ModuleNotFoundError as error:
            assert error.name == "torchmetrics"
            assert _REQUESTED == ["{contract.nested_target}"]
        else:
            raise AssertionError("nested dependency failure was hidden")
        """,
        leaf_errors=((contract.nested_target, "torchmetrics"),),
    )

    _run_script(
        f"""
        import {contract.training_module} as training

        resolved = dict(
            (name, getattr(training, name)) for name in {lazy_names!r}
        )
        for name, module_name in {contract.lazy_exports!r}:
            value = resolved[name]
            assert value.__module__ == module_name
            assert getattr(training, name) is value
            assert training.__dict__[name] is value
        assert tuple(_REQUESTED) == {lazy_modules!r}
        """,
        leaf_exports=contract.lazy_exports,
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda value: value.package_name)
def test_training_import_contract(contract: TrainingImportContract) -> None:
    """Check the shared inference-only and lazy training behavior."""
    _assert_training_import_contract(contract)
