"""Validate training claims against recorded stage-evidence rows.

The checker validates claim/document shape only. It does not inspect artifact
contents and it does not scan README or model-card full training run claims.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from devharness.baselines import (
    diff_entry_baseline,
    print_entries,
    read_entry_baseline,
    write_entry_baseline,
)
from devharness.markdown import (
    is_table_delimiter,
    iter_heading_sections,
    split_markdown_row,
)

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
    "<repo/cache-relative path or project issue/pr url>",
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
HISTORICAL_COMMAND_PREFIXES = ("Parameterized `setsid nohup` launchers under ",)
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
EVALUATION_PATH_PARITY_MARKER = "evaluation-path-parity"
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
        artifact_is_complete = is_artifact_path(self.artifact)
        if self.stage == "S5":
            artifact_is_complete = s5_artifact_paths_are_valid(
                *parse_s5_artifact_paths(self.artifact)
            )

        return (
            is_rerunnable_command(self.command)
            and artifact_is_complete
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
    if command.startswith(HISTORICAL_COMMAND_PREFIXES):
        return True

    while True:
        assignment = re.match(
            r"[A-Za-z_][A-Za-z0-9_]*=(?:<[^>]*>|'[^']*'|\"[^\"]*\"|\S+)\s+",
            command,
        )
        if assignment is None:
            break
        command = command[assignment.end() :]

    return command.startswith(COMMAND_STARTERS)


def is_artifact_path(value: str) -> bool:
    """Return whether an artifact cell is repo-relative or cache-relative."""
    quoted_paths = re.findall(r"`([^`]+)`", value)
    if quoted_paths:
        return all(is_artifact_path(path) for path in quoted_paths)

    artifact = unquote_cell(value)
    if ";" in artifact:
        return all(is_artifact_path(part) for part in artifact.split(";"))

    normalized = normalize_value(artifact)
    if normalized in PENDING_VALUES or " " in artifact:
        return False

    if ".." in Path(artifact).parts or artifact.endswith("/TRAINING.md"):
        return False

    return artifact.startswith(ARTIFACT_PREFIXES) or artifact.startswith(
        GITHUB_ARTIFACT_PREFIX
    )


def parse_s5_artifact_paths(value: str) -> tuple[str | None, str | None]:
    """Return the manifest path and evaluation-path parity path from an S5 cell."""
    manifest: str | None = None
    parity_path: str | None = None
    for part in value.strip().split(";"):
        cleaned = part.strip().strip("`").strip()
        marker, _separator, path = cleaned.partition(":")
        if marker.strip().lower() == EVALUATION_PATH_PARITY_MARKER:
            parity_path = path.strip().strip("`").strip() or None
            continue

        if cleaned:
            manifest = cleaned

    return manifest, parity_path


def s5_artifact_paths_are_valid(manifest: str | None, parity_path: str | None) -> bool:
    """Return whether parsed S5 artifact paths satisfy the artifact contract."""
    return (
        manifest is not None
        and parity_path is not None
        and is_artifact_path(manifest)
        and manifest.startswith(ARTIFACT_PREFIXES)
        and Path(manifest).name == "manifest.json"
        and is_artifact_path(parity_path)
        and parity_path != manifest
    )


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
    """Return whether the document claims full training run results."""
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


def _parse_stage_evidence_lines(
    lines: list[str],
) -> tuple[dict[str, StageEvidence], set[str]]:
    """Parse one machine-readable stage-evidence section."""
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


def parse_stage_evidence_sections(
    text: str,
) -> list[tuple[str, dict[str, StageEvidence], set[str]]]:
    """Parse all package or condition-specific stage-evidence sections."""
    sections = []
    for heading, lines in iter_heading_sections(text):
        normalized_heading = heading.lower()
        if normalized_heading != "stage evidence" and not normalized_heading.endswith(
            " stage evidence"
        ):
            continue
        evidence, duplicates = _parse_stage_evidence_lines(lines)
        sections.append((heading, evidence, duplicates))

    return sections


def parse_stage_evidence(
    text: str,
) -> tuple[dict[str, StageEvidence], set[str]]:
    """Parse the first machine-readable stage-evidence table."""
    for _heading, evidence, duplicates in parse_stage_evidence_sections(text):
        return evidence, duplicates
    return {}, set()


def training_docs(root: Path) -> list[Path]:
    """Return training reproduction documents covered by this check."""
    return sorted(path for path in root.glob(TRAINING_GLOB) if path.is_file())


def violations_for_training_doc(path: Path, root: Path) -> list[StageEvidenceViolation]:
    """Return evidence violations for one TRAINING.md.

    S5 artifact diagnostics run before row-completeness diagnostics without
    suppressing a completeness diagnostic for an otherwise valid S5 artifact.
    """
    text = path.read_text(encoding="utf-8")
    if not has_s5_claim(text):
        return []
    relative_path = path.relative_to(root).as_posix()
    evidence_sections = parse_stage_evidence_sections(text)
    violations: list[StageEvidenceViolation] = []
    if not has_reproduction_results_heading(text):
        violations.append(
            StageEvidenceViolation(
                relative_path,
                MISSING_SECTION_STAGE,
                "S5 result claim requires a Reproduction Results heading",
            )
        )
    if not evidence_sections:
        violations.append(
            StageEvidenceViolation(
                relative_path,
                MISSING_SECTION_STAGE,
                "S5 result claim requires a Stage Evidence table",
            )
        )
        return violations
    for _heading, evidence, duplicates in evidence_sections:
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
            elif stage == "S5" and normalize_value(row.artifact) not in PENDING_VALUES:
                manifest, parity_path = parse_s5_artifact_paths(row.artifact)
                if parity_path is None:
                    violations.append(
                        StageEvidenceViolation(
                            relative_path,
                            stage,
                            "S5 artifact must include an evaluation-path parity artifact reference",
                        )
                    )
                elif not s5_artifact_paths_are_valid(manifest, parity_path):
                    violations.append(
                        StageEvidenceViolation(
                            relative_path,
                            stage,
                            "S5 artifact must cite a repository- or cache-relative manifest.json and evaluation-path parity artifact",
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
    entries: set[str] = set()
    for path in training_docs(root):
        violations = violations_for_training_doc(path, root)
        entries.update(violation.as_baseline_entry() for violation in violations)

    return entries


def check_training_stage_evidence(root: Path, baseline_path: Path) -> int:
    """Check current training evidence violations against the baseline."""
    current_snapshot = current_entries(root)
    baseline_snapshot = read_entry_baseline(baseline_path)
    unexpected, stale = diff_entry_baseline(current_snapshot, baseline_snapshot)
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
        write_entry_baseline(BASELINE_PATH, current_entries(ROOT))
        return 0
    exit_status = check_training_stage_evidence(ROOT, BASELINE_PATH)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
