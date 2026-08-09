"""Validate package TRAINING.md structure and per-dataset status accounting."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from check_training_stage_evidence import (
    baseline_entries,
    iter_heading_sections,
    iter_unfenced_lines,
    is_table_delimiter,
    print_entries,
    split_markdown_row,
    write_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "training_doc_template_baseline.txt"
TRAINING_GLOB = "models/*/TRAINING.md"
README_NAME = "README.md"
SUPPORTED_CHECKPOINTS_HEADING = "Supported Checkpoints"
REPRODUCTION_RESULTS_HEADING = "Reproduction Results"
REQUIRED_SECTIONS = (
    "Scheduler and Recipe Notes",
    "Seed Policy",
    "Stage Evidence",
    "Reproduction Results",
    "Regeneration Metadata",
    "Training Commands",
)
TERMINAL_STATUSES = {
    "s5-bit-parity",
    "s5-practical-reproduction",
    "recipe-unstable (documented)",
}
NON_TERMINAL_STATUS_RE = re.compile(r"^(not-yet-run|blocked)\s+\((.+)\)$")
PLACEHOLDER_RE = re.compile(r"^<.*>$|^(?:tbd|todo|pending|n/a|na|-)$", re.I)
DATASET_COLUMN_NAMES = ("dataset", "checkpoint")
MISSING_README_TARGET = "README Supported Checkpoints"


@dataclass(frozen=True)
class TrainingDocViolation:
    """A stable TRAINING.md template violation."""

    path: str
    target: str
    reason: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.target}\t{self.reason}"


def normalize_header(value: str) -> str:
    """Normalize a Markdown table header."""
    return re.sub(r"[^a-z0-9]+", "", value.strip().strip("`").lower())


def normalize_cell(value: str) -> str:
    """Normalize Markdown table cell text."""
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<(https?://[^>\s]+)>", r"\1", value)
    value = value.replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_dataset(value: str) -> str:
    """Normalize a dataset/checkpoint name for cross-document matching."""
    return re.sub(r"[^a-z0-9]+", "", normalize_cell(value).lower())


def normalize_status(value: str) -> str:
    """Normalize a status cell for enum validation."""
    return re.sub(r"\s+", " ", normalize_cell(value).lower())


def heading_sections(text: str) -> dict[str, tuple[int, list[str]]]:
    """Return first matching heading sections keyed by heading name."""
    lines = list(iter_unfenced_lines(text))
    sections: dict[str, tuple[int, list[str]]] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match is None:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        content: list[str] = []
        for section_line in lines[index + 1 :]:
            section_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", section_line)
            if section_match is not None and len(section_match.group(1)) <= level:
                break
            content.append(section_line)
        sections.setdefault(heading.lower(), (level, content))
    return sections


def section_lines(text: str, heading_name: str) -> list[str]:
    """Return the lines for the first matching Markdown heading."""
    for heading, lines in iter_heading_sections(text):
        if heading.lower() == heading_name.lower():
            return lines
    return []


def iter_tables(lines: Iterable[str]) -> Iterable[tuple[list[str], list[list[str]]]]:
    """Yield normalized headers and rows for Markdown tables."""
    table_started = False
    headers: list[str] = []
    rows: list[list[str]] = []
    for line in lines:
        if not line.lstrip().startswith("|"):
            if table_started:
                yield headers, rows
                table_started = False
                headers = []
                rows = []
            continue
        cells = split_markdown_row(line)
        if not table_started:
            headers = [normalize_header(cell) for cell in cells]
            table_started = True
            continue
        if is_table_delimiter(line):
            continue
        rows.append(cells)
    if table_started:
        yield headers, rows


def first_table(lines: Iterable[str]) -> tuple[list[str], list[list[str]]]:
    """Return normalized headers and rows for the first Markdown table."""
    return next(iter(iter_tables(lines)), ([], []))


def supported_checkpoint_datasets(readme_path: Path) -> tuple[set[str], str | None]:
    """Return README checkpoint datasets or a reason they could not be read."""
    if not readme_path.is_file():
        return set(), "README.md is missing"
    text = readme_path.read_text(encoding="utf-8")
    lines = section_lines(text, SUPPORTED_CHECKPOINTS_HEADING)
    if not lines:
        return set(), "README.md missing Supported Checkpoints section"
    saw_table = False
    for headers, rows in iter_tables(lines):
        saw_table = True
        for column_name in DATASET_COLUMN_NAMES:
            normalized_column = normalize_header(column_name)
            if normalized_column not in headers:
                continue
            column_index = headers.index(normalized_column)
            datasets = {
                normalize_dataset(row[column_index])
                for row in rows
                if len(row) > column_index and normalize_dataset(row[column_index])
            }
            if datasets:
                return datasets, None
    if not saw_table:
        return set(), "README Supported Checkpoints section has no table"
    return set(), "README Supported Checkpoints tables have no dataset column"


def reproduction_result_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """Return the Reproduction Results table headers and rows."""
    return first_table(section_lines(text, REPRODUCTION_RESULTS_HEADING))


def is_valid_status(value: str) -> bool:
    """Return whether a Reproduction Results status uses the allowed enum."""
    raw_status = value.strip().strip("`").strip()
    status = normalize_status(value)
    if status in TERMINAL_STATUSES:
        return True
    match = NON_TERMINAL_STATUS_RE.fullmatch(status)
    if match is None:
        return False
    raw_match = NON_TERMINAL_STATUS_RE.fullmatch(raw_status.lower())
    detail = match.group(2).strip()
    raw_detail = raw_match.group(2).strip() if raw_match is not None else detail
    if not detail or detail in {"#", "()"}:
        return False
    if is_placeholder_detail(raw_detail):
        return False
    return True


def is_placeholder_detail(value: str) -> bool:
    """Return whether a status detail is a placeholder rather than evidence."""
    detail = value.strip()
    if re.fullmatch(r"<https?://[^>\s]+>", detail) is not None:
        return False
    return PLACEHOLDER_RE.fullmatch(detail) is not None or bool(
        re.search(r"<[^>]+>", detail)
    )


def violations_for_training_doc(path: Path, root: Path) -> list[TrainingDocViolation]:
    """Return template/status violations for one TRAINING.md."""
    text = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(root).as_posix()
    package_dir = path.parent
    violations: list[TrainingDocViolation] = []

    sections = heading_sections(text)
    for section in REQUIRED_SECTIONS:
        section_key = section.lower()
        if section_key not in sections:
            violations.append(
                TrainingDocViolation(
                    relative_path,
                    section,
                    "missing required TRAINING.md section",
                )
            )
            continue
        level, lines = sections[section_key]
        if level != 2:
            violations.append(
                TrainingDocViolation(
                    relative_path,
                    section,
                    "required TRAINING.md section must use level-2 heading",
                )
            )
        if not any(line.strip() for line in lines):
            violations.append(
                TrainingDocViolation(
                    relative_path,
                    section,
                    "required TRAINING.md section must not be empty",
                )
            )

    headers, rows = reproduction_result_rows(text)
    if not headers:
        violations.append(
            TrainingDocViolation(
                relative_path,
                REPRODUCTION_RESULTS_HEADING,
                "missing Reproduction Results table",
            )
        )
        return violations

    required_columns = {"dataset", "status"}
    missing_columns = sorted(required_columns.difference(headers))
    for column in missing_columns:
        violations.append(
            TrainingDocViolation(
                relative_path,
                REPRODUCTION_RESULTS_HEADING,
                f"Reproduction Results table missing {column!r} column",
            )
        )
    if missing_columns:
        return violations

    dataset_index = headers.index("dataset")
    status_index = headers.index("status")
    result_datasets = {
        normalize_dataset(row[dataset_index])
        for row in rows
        if len(row) > dataset_index and normalize_dataset(row[dataset_index])
    }
    for row_number, row in enumerate(rows, start=1):
        if len(row) <= status_index or not is_valid_status(row[status_index]):
            row_dataset = (
                row[dataset_index] if len(row) > dataset_index else str(row_number)
            )
            violations.append(
                TrainingDocViolation(
                    relative_path,
                    normalize_cell(row_dataset) or f"row {row_number}",
                    "Reproduction Results status is not an allowed enum value",
                )
            )

    claimed_datasets, readme_reason = supported_checkpoint_datasets(
        package_dir / README_NAME
    )
    if readme_reason is not None:
        violations.append(
            TrainingDocViolation(
                relative_path,
                MISSING_README_TARGET,
                readme_reason,
            )
        )
        return violations
    missing_datasets = sorted(claimed_datasets.difference(result_datasets))
    for dataset in missing_datasets:
        violations.append(
            TrainingDocViolation(
                relative_path,
                dataset,
                "README Supported Checkpoints dataset missing from Reproduction Results",
            )
        )

    return violations


def training_docs(root: Path) -> list[Path]:
    """Return training reproduction documents covered by this check."""
    return sorted(path for path in root.glob(TRAINING_GLOB) if path.is_file())


def current_entries(root: Path) -> set[str]:
    """Return current violation entries."""
    return {
        violation.as_baseline_entry()
        for path in training_docs(root)
        for violation in violations_for_training_doc(path, root)
    }


def check_training_doc_template(root: Path, baseline_path: Path) -> int:
    """Check current TRAINING.md template violations against the baseline."""
    current_snapshot = current_entries(root)
    baseline_snapshot = baseline_entries(baseline_path)
    unexpected = sorted(current_snapshot.difference(baseline_snapshot))
    stale = sorted(baseline_snapshot.difference(current_snapshot))
    if not unexpected and not stale:
        return 0
    print_entries("New TRAINING.md template violations:", "+", unexpected)
    print_entries("Stale TRAINING.md template baseline entries:", "-", stale)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the TRAINING.md template checker."""
    parser = argparse.ArgumentParser(
        description="Validate TRAINING.md template sections and dataset statuses."
    )
    parser.add_argument("--write-baseline", action="store_true")
    namespace = parser.parse_args(argv)
    if bool(namespace.write_baseline):
        write_baseline(BASELINE_PATH, current_entries(ROOT))
        return 0
    return check_training_doc_template(ROOT, BASELINE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
