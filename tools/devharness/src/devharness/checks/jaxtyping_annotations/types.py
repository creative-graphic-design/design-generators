"""Data records used by annotation policy checks."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class AnnotationViolation:
    """A raw tensor annotation violation."""

    path: str
    annotation: str
    line: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.annotation}\t{self.line}"


@dataclass(frozen=True)
class AliasViolation:
    """A jaxtyping shaped-type alias violation."""

    path: str
    alias: str
    annotation: str
    line: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.alias}\t{self.annotation}\t{self.line}"


@dataclass(frozen=True)
class ObjectAnnotationViolation:
    """An object annotation violation in a function signature."""

    path: str
    annotation: str
    line: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.annotation}\t{self.line}"


@dataclass(frozen=True)
class WeakCastTypeViolation:
    """A weak top-type reference hidden inside a cast target type."""

    path: str
    annotation: str
    line: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.annotation}\t{self.line}"


@dataclass(frozen=True)
class AnnotationRecord:
    """An annotation expression and its source line."""

    node: ast.AST
    line: str


@dataclass(frozen=True)
class FunctionAnnotationRecord:
    """A function signature annotation expression and its source line."""

    node: ast.AST
    line: str
    is_keyword_variadic: bool = False
