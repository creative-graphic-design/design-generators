"""Gate S5 training claims on machine-readable S0-S4 evidence.

The checker validates claim/document shape only. It does not inspect artifact
contents and it does not scan README or model-card S5 claims.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts" / "training_stage_evidence_baseline.txt"
TRAINING_GLOB = "models/*/TRAINING.md"
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
MISSING_SECTION_STAGE = "*"
PENDING_VALUES = {
    "",
    "-",
    "n/a",
    "na",
    "not recorded",
    "not run",
    "pending",
    "tbd",
    "todo",
}
COMMAND_STARTERS = (
    "./",
    "CUDA_VISIBLE_DEVICES=",
    "PARITY_REQUIRE=",
    "bash ",
    "cd ",
    "git ",
    "make ",
    "python ",
    "pytest ",
    "uv ",
)
ARTIFACT_PREFIXES = (
    ".cache/",
    "docs/",
    "lib/",
    "models/",
    "scripts/",
    "tests/",
    "vendor/",
)
GITHUB_ARTIFACT_PREFIX = "https://github.com/creative-graphic-design/design-generators/"
REPRODUCTION_RESULTS_HEADING = "Reproduction Results"
CLAUSE_BOUNDARY_RE = re.compile(r"[.;]")
NEGATED_CLAIM_RE = re.compile(
    r"\b(?:pending|not claimed|not yet claimed|no s-?5|no stage\s*5)\b",
    re.IGNORECASE,
)
POSITIVE_CLAIM_RE = re.compile(
    r"\b(?:accepted|achieved|complete|equivalent|evaluated|passed|reproduced)\b",
    re.IGNORECASE,
)
CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:s-?5|stage\s*5)\b.{0,160}"
        r"\b(?:accepted|achieved|complete|equivalent|evaluated|reproduced|verdict)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btraining-seed\s+n\s*=\s*\d+\b", re.IGNORECASE),
    re.compile(r"\btraining reproduction is achieved\b", re.IGNORECASE),
    re.compile(r"\bstatistically equivalent\b", re.IGNORECASE),
    re.compile(
        r"\bfull[- ]run\b.{0,160}\b(?:comparison|evidence|statistical|verdict)\b"
        r".{0,160}\b(?:accepted|achieved|complete|equivalent|evaluated|passed|reproduced)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class StageEvidence:
    """One machine-readable training stage evidence row."""

    stage: str
    command: str
    artifact: str
    result: str

    @property
    def is_complete(self) -> bool:
        """Return whether the row carries non-placeholder evidence."""
        return (
            is_rerunnable_command(self.command)
            and is_artifact_path(self.artifact)
            and normalize_value(self.result) not in PENDING_VALUES
        )


@dataclass(frozen=True)
class StageEvidenceViolation:
    """A training stage evidence violation."""

    path: str
    stage: str
    reason: str

    def as_baseline_entry(self) -> str:
        """Return a stable baseline entry for this violation."""
        return f"{self.path}\t{self.stage}\t{self.reason}"


def normalize_value(value: str) -> str:
    """Normalize a Markdown table cell value for placeholder checks."""
    return re.sub(r"\s+", " ", value.strip().strip("`")).lower()


def normalize_header(value: str) -> str:
    """Normalize a Markdown table header."""
    return re.sub(r"[^a-z0-9]+", "", value.strip().strip("`").lower())


def unquote_cell(value: str) -> str:
    """Return a Markdown table cell without simple code-span quoting."""
    return value.strip().strip("`").strip()


def is_rerunnable_command(value: str) -> bool:
    """Return whether a stage command cell looks directly rerunnable."""
    command = unquote_cell(value)
    normalized = normalize_value(command)
    if normalized in PENDING_VALUES:
        return False
    return command.startswith(COMMAND_STARTERS)


def is_artifact_path(value: str) -> bool:
    """Return whether an artifact cell is repo-relative or cache-relative."""
    artifact = unquote_cell(value)
    normalized = normalize_value(artifact)
    if normalized in PENDING_VALUES or " " in artifact:
        return False
    if ".." in Path(artifact).parts or artifact.endswith("/TRAINING.md"):
        return False
    return artifact.startswith(ARTIFACT_PREFIXES) or artifact.startswith(
        GITHUB_ARTIFACT_PREFIX
    )


def split_markdown_row(line: str) -> list[str]:
    """Split a simple Markdown table row into stripped cells."""
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line.strip().strip("|"):
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def is_table_delimiter(line: str) -> bool:
    """Return whether a Markdown table row is a delimiter row."""
    cells = split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def iter_heading_sections(text: str) -> Iterable[tuple[str, list[str]]]:
    """Yield Markdown heading text with the lines inside that heading."""
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current_heading is not None:
                yield current_heading, current_lines
                current_lines = []
            current_heading = match.group(2).strip()
            continue
        if current_heading is not None:
            current_lines.append(line)
    if current_heading is not None:
        yield current_heading, current_lines


def section_named(text: str, heading_name: str) -> str:
    """Return the content for the first matching Markdown heading."""
    for heading, lines in iter_heading_sections(text):
        if heading.lower() == heading_name.lower():
            return "\n".join(lines)
    return ""


def has_reproduction_results_heading(text: str) -> bool:
    """Return whether TRAINING.md contains the required results heading."""
    for heading, _ in iter_heading_sections(text):
        if heading.lower().startswith(REPRODUCTION_RESULTS_HEADING.lower()):
            return True
    return False


def claim_text(text: str) -> str:
    """Return whole-document text normalized for claim matching."""
    return re.sub(r"\s+", " ", text).strip()


def clause_around_match(text: str, start: int, end: int) -> str:
    """Return the punctuation-delimited clause containing a regex match."""
    left_boundary = 0
    for match in CLAUSE_BOUNDARY_RE.finditer(text, 0, start):
        left_boundary = match.end()
    right_match = CLAUSE_BOUNDARY_RE.search(text, end)
    right_boundary = len(text) if right_match is None else right_match.start()
    return text[left_boundary:right_boundary].strip()


def has_s5_claim(text: str) -> bool:
    """Return whether the document claims S5/full-run training results."""
    normalized = claim_text(text)
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(normalized):
            clause = clause_around_match(normalized, match.start(), match.end())
            if NEGATED_CLAIM_RE.search(clause):
                if POSITIVE_CLAIM_RE.search(match.group(0)):
                    return True
                continue
            return True
    return False


def parse_stage_evidence(
    text: str,
) -> tuple[dict[str, StageEvidence], set[str]]:
    """Parse the machine-readable Stage Evidence table."""
    for heading, lines in iter_heading_sections(text):
        if heading.lower() != "stage evidence":
            continue
        for index, line in enumerate(lines):
            if not line.lstrip().startswith("|"):
                continue
            headers = [normalize_header(cell) for cell in split_markdown_row(line)]
            if not {"stage", "command", "artifact", "result"}.issubset(headers):
                continue
            row_start = index + 1
            if row_start < len(lines) and is_table_delimiter(lines[row_start]):
                row_start += 1
            positions = {
                name: headers.index(name)
                for name in ("stage", "command", "artifact", "result")
            }
            evidence: dict[str, StageEvidence] = {}
            duplicates: set[str] = set()
            for row in lines[row_start:]:
                if not row.lstrip().startswith("|"):
                    break
                if is_table_delimiter(row):
                    continue
                cells = split_markdown_row(row)
                if len(cells) < len(headers):
                    continue
                stage = cells[positions["stage"]].strip().upper()
                if stage not in STAGES:
                    continue
                if stage in evidence:
                    duplicates.add(stage)
                evidence[stage] = StageEvidence(
                    stage=stage,
                    command=cells[positions["command"]],
                    artifact=cells[positions["artifact"]],
                    result=cells[positions["result"]],
                )
            return evidence, duplicates
    return {}, set()


def training_docs(root: Path) -> list[Path]:
    """Return training reproduction documents covered by this check."""
    return sorted(path for path in root.glob(TRAINING_GLOB) if path.is_file())


def violations_for_training_doc(path: Path, root: Path) -> list[StageEvidenceViolation]:
    """Return evidence violations for one TRAINING.md."""
    text = path.read_text(encoding="utf-8")
    if not has_s5_claim(text):
        return []
    relative_path = path.relative_to(root).as_posix()
    evidence, duplicates = parse_stage_evidence(text)
    violations: list[StageEvidenceViolation] = []
    if not has_reproduction_results_heading(text):
        violations.append(
            StageEvidenceViolation(
                relative_path,
                MISSING_SECTION_STAGE,
                "S5 result claim requires a Reproduction Results heading",
            )
        )
    if not evidence:
        violations.append(
            StageEvidenceViolation(
                relative_path,
                MISSING_SECTION_STAGE,
                "S5 result claim requires a Stage Evidence table",
            )
        )
        return violations
    for stage in sorted(duplicates):
        violations.append(
            StageEvidenceViolation(
                relative_path,
                stage,
                "stage evidence table contains duplicate rows for this stage",
            )
        )
    for stage in STAGES:
        row = evidence.get(stage)
        if row is None:
            violations.append(
                StageEvidenceViolation(
                    relative_path,
                    stage,
                    "S5 result claim requires a complete evidence row for this stage",
                )
            )
        elif not row.is_complete:
            violations.append(
                StageEvidenceViolation(
                    relative_path,
                    stage,
                    "stage evidence row has a placeholder command, artifact, or result",
                )
            )
    return violations


def current_entries(root: Path) -> set[str]:
    """Return current violation entries."""
    return {
        violation.as_baseline_entry()
        for path in training_docs(root)
        for violation in violations_for_training_doc(path, root)
    }


def baseline_entries(path: Path) -> set[str]:
    """Return committed shrink-only baseline entries."""
    if not path.is_file():
        raise FileNotFoundError(path)
    entries: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line and not raw_line.startswith("#"):
            entries.add(raw_line)
    return entries


def write_baseline(path: Path, entries: Iterable[str]) -> None:
    """Write sorted baseline entries."""
    lines = sorted(entries)
    path.write_text("\n".join([*lines, ""]) if lines else "", encoding="utf-8")


def print_entries(header: str, marker: str, entries: list[str]) -> None:
    """Print formatted violation entries to stderr."""
    if not entries:
        return
    print(header, file=sys.stderr)
    for entry in entries:
        print(f"  {marker} {entry}", file=sys.stderr)


def check_training_stage_evidence(root: Path, baseline_path: Path) -> int:
    """Check current training evidence violations against the baseline."""
    current_snapshot = current_entries(root)
    baseline_snapshot = baseline_entries(baseline_path)
    unexpected = sorted(current_snapshot.difference(baseline_snapshot))
    stale = sorted(baseline_snapshot.difference(current_snapshot))
    if not unexpected and not stale:
        return 0
    print_entries("New training stage evidence violations:", "+", unexpected)
    print_entries("Stale training stage evidence baseline entries:", "-", stale)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the training stage evidence checker."""
    parser = argparse.ArgumentParser(
        description="Validate machine-readable S0-S5 training evidence rows."
    )
    parser.add_argument("--write-baseline", action="store_true")
    namespace = parser.parse_args(argv)
    should_update_baseline = bool(namespace.write_baseline)
    if should_update_baseline:
        write_baseline(BASELINE_PATH, current_entries(ROOT))
        return 0
    exit_status = check_training_stage_evidence(ROOT, BASELINE_PATH)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
