"""Dependency-free helpers for lazy class exports."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Generic, TypeVar, cast

_T = TypeVar("_T")


@dataclass(frozen=True)
class LazyClassExport(Generic[_T]):
    """Describe a class that can be resolved from a lazy module export.

    Attributes:
        module: Fully qualified module containing the class.
        attribute: Attribute name containing the class.
        optional_roots: Direct optional-dependency roots allowed to be missing.

    Examples:
        >>> from laygen.common.import_utils import LazyClassExport
        >>> export = LazyClassExport(
        ...     module="types",
        ...     attribute="ModuleType",
        ...     optional_roots=frozenset(),
        ... )
        >>> export.attribute
        'ModuleType'
    """

    module: str
    attribute: str
    optional_roots: frozenset[str]


def resolve_lazy_class(
    requested_name: str,
    *,
    module_name: str,
    distribution_name: str,
    exports: Mapping[str, LazyClassExport[_T]],
) -> type[_T]:
    """Resolve and cache one class from a module's lazy export table.

    Args:
        requested_name: Public name requested from the importing module.
        module_name: Fully qualified name of the importing module.
        distribution_name: Distribution name used in the installation hint.
        exports: Mapping of public names to their lazy class specifications.

    Returns:
        The resolved class.

    Raises:
        AttributeError: If ``requested_name`` is not a known export.
        ImportError: If a declared direct optional dependency is missing.
        TypeError: If the resolved target is not a class.
        ModuleNotFoundError: If an undeclared or nested dependency is missing.

    Examples:
        >>> import sys
        >>> from types import ModuleType
        >>> from laygen.common.import_utils import (
        ...     LazyClassExport,
        ...     resolve_lazy_class,
        ... )
        >>> requester = ModuleType("_import_utils_example")
        >>> sys.modules[requester.__name__] = requester
        >>> export = LazyClassExport("types", "ModuleType", frozenset())
        >>> resolve_lazy_class(
        ...     "ModuleType",
        ...     module_name=requester.__name__,
        ...     distribution_name="laygen",
        ...     exports={"ModuleType": export},
        ... ) is ModuleType
        True
        >>> del sys.modules[requester.__name__]
    """
    export = exports.get(requested_name)
    if export is None:
        raise AttributeError(
            f"module '{module_name}' has no attribute '{requested_name}'"
        )

    requesting_module = sys.modules[module_name]
    if requested_name in requesting_module.__dict__:
        cached = requesting_module.__dict__[requested_name]
        if inspect.isclass(cached):
            return cast(type[_T], cached)

    try:
        target_module = import_module(export.module)
    except ModuleNotFoundError as error:
        missing_name = error.name
        missing_root = (
            missing_name.partition(".")[0] if missing_name is not None else None
        )
        if missing_root not in export.optional_roots:
            raise
        raise ImportError(
            f"{module_name}.{requested_name} requires the optional "
            f"'{missing_root}' dependency; install the training extra with "
            f"`pip install '{distribution_name}[training]'`."
        ) from error

    resolved = getattr(target_module, export.attribute)
    if not inspect.isclass(resolved):
        raise TypeError(
            f"{module_name}.{requested_name} must resolve to a class; "
            f"got {type(resolved).__name__}"
        )

    resolved_class = cast(type[_T], resolved)
    setattr(requesting_module, requested_name, resolved_class)
    return resolved_class


def build_module_dir(
    module_names: Iterable[str],
    export_names: Iterable[str],
) -> list[str]:
    """Build stable, duplicate-free names for a lazy module's ``__dir__``.

    Args:
        module_names: Names already present in the module namespace.
        export_names: Lazy public export names to add to the directory listing.

    Returns:
        Sorted unique names from the module namespace and export table.

    Examples:
        >>> from laygen.common.import_utils import build_module_dir
        >>> build_module_dir(["__name__", "Eager"], ["Lazy", "Eager"])
        ['Eager', 'Lazy', '__name__']
    """
    return sorted(set(module_names) | set(export_names))


__all__ = ["LazyClassExport", "build_module_dir", "resolve_lazy_class"]
