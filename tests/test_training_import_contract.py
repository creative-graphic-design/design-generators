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
    lazy_exports: tuple[str, ...]
    lazy_modules: tuple[str, ...]
    top_level_blocked: tuple[str, ...]
    export_blocked: tuple[str, ...]
    optional_errors: tuple[tuple[str, str], ...]
    nested_target: str
    nested_export: str
    resolved_exports: tuple[str, ...]
    class_bases: tuple[tuple[str, str], ...]
    module_assertions: tuple[tuple[str, str], ...] = ()


CONTRACTS = (
    TrainingImportContract(
        package_name="cgb_dm",
        training_module="cgb_dm.training",
        lazy_exports=("CGBDMDataModule", "CGBDMTrainingModule"),
        lazy_modules=(
            "cgb_dm.training.datamodule",
            "cgb_dm.training.lightning_module",
        ),
        top_level_blocked=("lightning", "torchmetrics", "h5py", "h5pickle"),
        export_blocked=("lightning",),
        optional_errors=(
            ("CGBDMDataModule", "lightning"),
            ("CGBDMTrainingModule", "lightning"),
        ),
        nested_target="cgb_dm.training.lightning_module",
        nested_export="CGBDMTrainingModule",
        resolved_exports=("CGBDMDataModule", "CGBDMTrainingModule"),
        class_bases=(
            ("CGBDMDataModule", "LightningDataModule"),
            ("CGBDMTrainingModule", "LightningModule"),
        ),
    ),
    TrainingImportContract(
        package_name="dlt",
        training_module="dlt.training",
        lazy_exports=(
            "DLTDataModule",
            "DLTReferenceEpochSamplingCallback",
            "DLTTrainingModule",
            "DLTWarmupCosineSchedulerFactory",
        ),
        lazy_modules=(
            "dlt.training.callbacks",
            "dlt.training.datamodule",
            "dlt.training.dataset",
            "dlt.training.lightning_module",
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
        resolved_exports=(
            "DLTDataModule",
            "DLTReferenceEpochSamplingCallback",
            "DLTTrainingModule",
            "DLTWarmupCosineSchedulerFactory",
        ),
        class_bases=(
            ("DLTDataModule", "LightningDataModule"),
            ("DLTReferenceEpochSamplingCallback", "Callback"),
            ("DLTTrainingModule", "LightningModule"),
        ),
        module_assertions=(
            ("DLTWarmupCosineSchedulerFactory", "dlt.training.lightning_module"),
        ),
    ),
    TrainingImportContract(
        package_name="layout_dm",
        training_module="layout_dm.training",
        lazy_exports=("LayoutDMDataModule", "LayoutDMTrainingModule"),
        lazy_modules=(
            "layout_dm.training.datamodule",
            "layout_dm.training.lightning_module",
        ),
        top_level_blocked=("lightning", "datasets"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutDMDataModule", "lightning"),
            ("LayoutDMTrainingModule", "lightning"),
        ),
        nested_target="layout_dm.training.lightning_module",
        nested_export="LayoutDMTrainingModule",
        resolved_exports=("LayoutDMDataModule", "LayoutDMTrainingModule"),
        class_bases=(
            ("LayoutDMDataModule", "LightningDataModule"),
            ("LayoutDMTrainingModule", "LightningModule"),
        ),
    ),
    TrainingImportContract(
        package_name="layout_flow",
        training_module="layout_flow.training",
        lazy_exports=("LayoutFlowDataModule", "LayoutFlowTrainingModule"),
        lazy_modules=(
            "layout_flow.training.datamodule",
            "layout_flow.training.lightning_module",
        ),
        top_level_blocked=("lightning", "torchmetrics", "h5py", "h5pickle"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutFlowDataModule", "lightning"),
            ("LayoutFlowTrainingModule", "lightning"),
        ),
        nested_target="layout_flow.training.lightning_module",
        nested_export="LayoutFlowTrainingModule",
        resolved_exports=("LayoutFlowDataModule", "LayoutFlowTrainingModule"),
        class_bases=(
            ("LayoutFlowDataModule", "LightningDataModule"),
            ("LayoutFlowTrainingModule", "LightningModule"),
        ),
    ),
    TrainingImportContract(
        package_name="layoutdiffusion",
        training_module="layoutdiffusion.training",
        lazy_exports=(
            "LayoutDiffusionDataModule",
            "LayoutDiffusionTrainingModule",
        ),
        lazy_modules=(
            "layoutdiffusion.training.datamodule",
            "layoutdiffusion.training.lightning_module",
        ),
        top_level_blocked=("lightning", "datasets"),
        export_blocked=("lightning",),
        optional_errors=(
            ("LayoutDiffusionDataModule", "lightning"),
            ("LayoutDiffusionTrainingModule", "lightning"),
        ),
        nested_target="layoutdiffusion.training.lightning_module",
        nested_export="LayoutDiffusionTrainingModule",
        resolved_exports=(
            "LayoutDiffusionDataModule",
            "LayoutDiffusionTrainingModule",
        ),
        class_bases=(
            ("LayoutDiffusionDataModule", "LightningDataModule"),
            ("LayoutDiffusionTrainingModule", "LightningModule"),
        ),
    ),
)


def _run_script(code: str, *, blocked: tuple[str, ...] = ()) -> None:
    """Run one isolated contract assertion in a fresh interpreter."""
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        _BLOCKED = {blocked!r}

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

        if _BLOCKED:
            sys.meta_path.insert(0, _ImportBlocker())
        """
    )
    subprocess.run(
        [sys.executable, "-c", bootstrap + textwrap.dedent(code)],
        check=True,
    )


def _assert_training_import_contract(contract: TrainingImportContract) -> None:
    """Run the shared lazy training checks for one model contract."""
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

        for name in {contract.lazy_exports!r}:
            assert name in training.__all__
            assert name in dir(training)
            assert name not in training.__dict__
        for name in {contract.lazy_modules!r}:
            assert name not in sys.modules
        """,
        blocked=contract.export_blocked,
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
            else:
                raise AssertionError("optional dependency unexpectedly resolved")
            """,
            blocked=(missing_root,),
        )

    _run_script(
        f"""
        import importlib.abc

        import {contract.training_module} as training

        class _BrokenTarget(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "{contract.nested_target}":
                    raise ModuleNotFoundError(
                        "No module named 'torchmetrics'", name="torchmetrics"
                    )
                return None

        sys.meta_path.insert(0, _BrokenTarget())
        try:
            getattr(training, "{contract.nested_export}")
        except ModuleNotFoundError as error:
            assert error.name == "torchmetrics"
        else:
            raise AssertionError("nested dependency failure was hidden")
        """
    )

    try:
        __import__("lightning.pytorch")
    except ImportError:
        return

    _run_script(
        f"""
        from lightning.pytorch import Callback, LightningDataModule, LightningModule

        import {contract.training_module} as training

        resolved = dict(
            (name, getattr(training, name)) for name in {contract.resolved_exports!r}
        )
        base_types = dict(
            (
                name,
                {{
                    "Callback": Callback,
                    "LightningDataModule": LightningDataModule,
                    "LightningModule": LightningModule,
                }}[base_name],
            )
            for name, base_name in {contract.class_bases!r}
        )
        for name, base_type in base_types.items():
            assert issubclass(resolved[name], base_type)
        for name, module_name in {contract.module_assertions!r}:
            assert resolved[name].__module__ == module_name
        for name, value in resolved.items():
            assert getattr(training, name) is value
            assert training.__dict__[name] is value
        """
    )


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda value: value.package_name)
def test_training_import_contract(contract: TrainingImportContract) -> None:
    """Check the shared inference-only and lazy training behavior."""
    _assert_training_import_contract(contract)
