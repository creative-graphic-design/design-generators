"""Validate training evidence formats and new full-run checkpoint smoke claims.

The checker does not execute recorded commands, hash artifact contents, or scan
README and model-card claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
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
CHECKPOINT_GATE_BASELINE_RELATIVE_PATH = Path(
    "scripts/training_s5_checkpoint_gate_baseline.txt"
)
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
    "placeholder",
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
CLAUSE_BOUNDARY_RE = re.compile(r"[.!?](?=\s)")
CLAIM_GAP = r"(?:(?![.!?]\s).){0,160}"
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
        r"\b(?:s-?5|stage\s*5)\b"
        + CLAIM_GAP
        + r"\b(?:accepted|achieved|complete|equivalent|evaluated|reproduced|verdict)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btraining-seed\s+n\s*=\s*\d+\b", re.IGNORECASE),
    re.compile(r"\btraining reproduction is achieved\b", re.IGNORECASE),
    re.compile(r"\bstatistically equivalent\b", re.IGNORECASE),
    re.compile(
        r"\bfull[- ]run\b"
        + CLAIM_GAP
        + r"\b(?:comparison|evidence|statistical|verdict)\b"
        + CLAIM_GAP
        + r"\b(?:accepted|achieved|complete|equivalent|evaluated|passed|reproduced)\b",
        re.IGNORECASE,
    ),
)
CHECKPOINT_SMOKE_VIOLATION = (
    "new or changed S5 claim lacks bounded production checkpoint smoke evidence"
)


@dataclass(frozen=True)
class StageEvidence:
    """One machine-readable training stage evidence row."""

    stage: str
    command: str
    artifact: str
    result: str
    checkpoint_smoke_command: str = ""
    full_run_config: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""

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
            and not is_pending_or_placeholder(self.result)
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


def is_pending_or_placeholder(value: str) -> bool:
    """Return whether a cell carries no recorded evidence."""
    normalized = normalize_value(value)
    return normalized in PENDING_VALUES or (
        normalized.startswith("<") and normalized.endswith(">")
    )


def unquote_cell(value: str) -> str:
    """Return a Markdown table cell without simple code-span quoting."""
    return value.strip().strip("`").strip()


def is_rerunnable_command(value: str) -> bool:
    """Return whether a stage command cell looks directly rerunnable."""
    command = unquote_cell(value)
    normalized = normalize_value(command)
    if normalized in PENDING_VALUES:
        return False

    while True:
        assignment = re.match(
            r"[A-Za-z_][A-Za-z0-9_]*=(?:<[^>]*>|'[^']*'|\"[^\"]*\"|\S*)\s+",
            command,
        )
        if assignment is None:
            break

        command = command[assignment.end() :]

    if command.startswith(COMMAND_STARTERS):
        return True

    return bool(
        re.match(r"""(["'])\$[A-Za-z_][A-Za-z0-9_]*/bin/python\1(?:\s|$)""", command)
    )


def is_artifact_path(value: str) -> bool:
    """Return whether an artifact cell is repo-relative or cache-relative."""
    if ";" in value:
        return all(is_artifact_path(part) for part in value.split(";"))

    quoted_paths = re.findall(r"`([^`]+)`", value)
    if quoted_paths:
        remainder = re.sub(r"`[^`]+`", "", value).strip()
        if remainder:
            return False

        return all(is_artifact_path(path) for path in quoted_paths)

    artifact = unquote_cell(value)
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
    parity_seen = False
    parts = [part.strip().strip("`").strip() for part in value.strip().split(";")]
    if not parts or any(not part for part in parts):
        return None, None

    for cleaned in parts:
        marker, _separator, path = cleaned.partition(":")
        if marker.strip().lower() == EVALUATION_PATH_PARITY_MARKER:
            if parity_seen:
                return None, None

            parity_seen = True
            parity_path = path.strip().strip("`").strip() or None
            continue

        if cleaned:
            if manifest is not None:
                return None, None

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
    return bool(s5_claim_clauses(text))


def s5_claim_clauses(text: str) -> tuple[str, ...]:
    """Return the clauses that contain full-run claims."""
    normalized = claim_text(text)
    clauses = []
    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(normalized):
            clause = clause_around_match(normalized, match.start(), match.end())
            if NEGATED_CLAIM_RE.search(clause):
                if POSITIVE_CLAIM_RE.search(match.group(0)):
                    clauses.append(clause)
                continue
            clauses.append(clause)
    return tuple(sorted(clauses))


def s5_record_signature(text: str) -> str:
    """Return a stable fingerprint for a document's recorded full-run claim."""
    stage_rows = []
    for _heading, evidence, _duplicates in parse_stage_evidence_sections(text):
        row = evidence.get("S5")
        if row is not None:
            stage_rows.append((row.command, row.artifact, row.result))

    payload = json.dumps(
        {"claim_clauses": s5_claim_clauses(text), "stage_rows": stage_rows},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_stage_evidence_lines(
    lines: list[str],
) -> list[tuple[dict[str, StageEvidence], set[str]]]:
    """Parse every machine-readable Stage Evidence table in a section."""
    required_headers = {"stage", "command", "artifact", "result"}
    tables = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().startswith("|"):
            index += 1
            continue
        headers = [normalize_header(cell) for cell in split_markdown_row(line)]
        if not required_headers.issubset(headers):
            index += 1
            continue
        row_start = index + 1
        if row_start < len(lines) and is_table_delimiter(lines[row_start]):
            row_start += 1
        positions = {
            name: headers.index(name)
            for name in ("stage", "command", "artifact", "result")
        }
        optional_positions = {
            name: headers.index(name) if name in headers else None
            for name in (
                "checkpointsmokecommand",
                "fullrunconfig",
                "checkpointpath",
                "checkpointsha256",
            )
        }
        evidence: dict[str, StageEvidence] = {}
        duplicates: set[str] = set()

        row_index = row_start
        while row_index < len(lines):
            row = lines[row_index]
            if not row.lstrip().startswith("|"):
                break
            if required_headers.issubset(
                normalize_header(cell) for cell in split_markdown_row(row)
            ):
                break

            if is_table_delimiter(row):
                row_index += 1
                continue
            cells = split_markdown_row(row)
            if len(cells) < len(headers):
                row_index += 1
                continue

            stage = cells[positions["stage"]].strip().upper()
            if stage not in STAGES:
                row_index += 1
                continue

            if stage in evidence:
                duplicates.add(stage)

            evidence[stage] = StageEvidence(
                stage=stage,
                command=cells[positions["command"]],
                artifact=cells[positions["artifact"]],
                result=cells[positions["result"]],
                checkpoint_smoke_command=(
                    cells[optional_positions["checkpointsmokecommand"]]
                    if optional_positions["checkpointsmokecommand"] is not None
                    else ""
                ),
                full_run_config=(
                    cells[optional_positions["fullrunconfig"]]
                    if optional_positions["fullrunconfig"] is not None
                    else ""
                ),
                checkpoint_path=(
                    cells[optional_positions["checkpointpath"]]
                    if optional_positions["checkpointpath"] is not None
                    else ""
                ),
                checkpoint_sha256=(
                    cells[optional_positions["checkpointsha256"]]
                    if optional_positions["checkpointsha256"] is not None
                    else ""
                ),
            )
            row_index += 1
        tables.append((evidence, duplicates))
        index = row_index

    return tables


def parse_stage_evidence_sections(
    text: str,
) -> list[tuple[str, dict[str, StageEvidence], set[str]]]:
    """Parse every exact Stage Evidence section in a training document."""
    sections = []
    for heading, lines in iter_heading_sections(text):
        if heading.lower() != "stage evidence":
            continue
        for evidence, duplicates in _parse_stage_evidence_lines(lines):
            sections.append((heading, evidence, duplicates))

    return sections


def parse_stage_evidence(
    text: str,
) -> tuple[dict[str, StageEvidence], set[str]]:
    """Parse the first machine-readable Stage Evidence table."""
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
            elif stage == "S5" and not is_pending_or_placeholder(row.artifact):
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


def is_bounded_production_checkpoint_smoke(
    evidence: StageEvidence | None,
    full_run_evidence: StageEvidence | None,
    package: str,
    root: Path,
) -> bool:
    """Return whether bounded production-command evidence identifies its checkpoint."""
    if evidence is None or full_run_evidence is None:
        return False

    checkpoint_path = unquote_cell(evidence.checkpoint_path)
    checkpoint_sha256 = unquote_cell(evidence.checkpoint_sha256)
    if (
        not is_artifact_path(checkpoint_path)
        or not checkpoint_path.endswith(".ckpt")
        or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256) is None
    ):
        return False

    command = unquote_cell(evidence.checkpoint_smoke_command)
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False

    has_production_entry_point = is_production_fit_command(tokens)
    config_values: list[str] = []
    trainer_overrides: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--config" or token.startswith("--config="):
            if "=" in token:
                config_values.append(token.split("=", 1)[1])
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                config_values.append(tokens[index + 1])
                index += 1
            else:
                return False
        elif token.startswith("--trainer."):
            option = token.removeprefix("--trainer.")
            if "=" in option:
                name, value = option.split("=", 1)
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                name, value = option, tokens[index + 1]
                index += 1
            else:
                return False

            trainer_overrides.setdefault(name, []).append(value)

        index += 1

    required = {
        "max_epochs": "1",
        "limit_train_batches": "2",
        "limit_val_batches": "2",
    }
    allowed = {*required, "check_val_every_n_epoch"}
    if (
        not has_production_entry_point
        or len(config_values) != 1
        or set(trainer_overrides) - allowed
        or any(
            trainer_overrides.get(name) != [value] for name, value in required.items()
        )
        or trainer_overrides.get("check_val_every_n_epoch", ["1"]) != ["1"]
    ):
        return False

    config_path = Path(config_values[0])
    full_run_config = unquote_cell(full_run_evidence.full_run_config)
    if config_values[0] != full_run_config:
        return False

    return (
        not config_path.is_absolute()
        and ".." not in config_path.parts
        and config_path.parts[:2] == ("models", package)
        and config_path.suffix in {".yaml", ".yml"}
        and "training" in config_path.parts
        and (root / config_path).is_file()
        and is_tracked_file(root, config_path)
        and manifest_config_matches(full_run_evidence, full_run_config, root)
        and callback_checkpoint_filename_matches(
            root / config_path, Path(checkpoint_path).name
        )
    )


def is_production_fit_command(tokens: list[str]) -> bool:
    """Recognize a direct training entry point or one launched through uv."""
    if any(token in {"&&", "||", ";", "|", ">", "<", "&"} for token in tokens):
        return False

    index = 0
    while index < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]
    ):
        index += 1

    if tokens[index : index + 2] == ["traingen", "fit"]:
        return True

    if tokens[index : index + 4] == ["python", "-m", "traingen.lightning.cli", "fit"]:
        return True

    if tokens[index : index + 2] != ["uv", "run"]:
        return False

    index += 2
    options_with_values = {
        "--config-file",
        "--directory",
        "--env-file",
        "--extra",
        "--package",
        "--project",
        "--python",
        "--with",
    }
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break

        if token in options_with_values:
            index += 2
            continue

        if token.startswith("-"):
            index += 1
            continue

        break

    return tokens[index : index + 2] == ["traingen", "fit"] or tokens[
        index : index + 4
    ] == ["python", "-m", "traingen.lightning.cli", "fit"]


def is_tracked_file(root: Path, path: Path) -> bool:
    """Return whether Git tracks a repository-relative file."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def manifest_config_matches(
    evidence: StageEvidence, full_run_config: str, root: Path
) -> bool:
    """Match the recorded full-run config to a locally available launch manifest."""
    manifest, _parity_path = parse_s5_artifact_paths(evidence.artifact)
    if manifest is None:
        return False

    manifest_path = root / manifest
    if not manifest_path.is_file():
        return True

    try:
        manifest_config = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "config"
        ]
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False

    return isinstance(manifest_config, str) and manifest_config == full_run_config


def callback_checkpoint_filename_matches(
    config_path: Path, checkpoint_filename: str
) -> bool:
    """Reject last-only files and honor explicit monitored callback filename patterns."""
    if re.fullmatch(r"last(?:-v\d+)?\.ckpt", checkpoint_filename, re.IGNORECASE):
        return False

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False

    if not isinstance(config, dict):
        return False

    trainer = config.get("trainer", {})
    callbacks = trainer.get("callbacks", []) if isinstance(trainer, dict) else []
    if not isinstance(callbacks, list):
        return False

    patterns = []
    for callback in callbacks:
        if not isinstance(callback, dict) or not str(
            callback.get("class_path", "")
        ).endswith("ModelCheckpoint"):
            continue

        init_args = callback.get("init_args", {})
        if not isinstance(init_args, dict):
            return False

        try:
            save_top_k = int(init_args.get("save_top_k", 1))
        except (TypeError, ValueError):
            return False

        if save_top_k == 0 or not init_args.get("monitor"):
            continue

        filename = init_args.get("filename")
        auto_insert_metric_name = bool(init_args.get("auto_insert_metric_name", True))
        patterns.append(
            (
                str(filename) if filename is not None else "{epoch}-{step}",
                auto_insert_metric_name,
            )
        )

    if not patterns:
        return False

    return any(
        checkpoint_filename_matches_pattern(pattern, checkpoint_filename, auto_insert)
        for pattern, auto_insert in patterns
    )


def checkpoint_filename_matches_pattern(
    pattern: str, filename: str, auto_insert_metric_name: bool
) -> bool:
    """Match a rendered checkpoint filename against a callback's filename pattern."""
    pattern_parts = re.split(r"(\{[^{}]+\})", pattern)
    rendered = ""
    for part in pattern_parts:
        if part.startswith("{") and part.endswith("}"):
            field = part[1:-1].split(":", 1)[0]
            field_name = rf"(?:{re.escape(field)}=)?" if auto_insert_metric_name else ""
            rendered += field_name + r"[^/]+"
        else:
            rendered += re.escape(part)

    if not pattern.endswith(".ckpt"):
        rendered += r"(?:-v\d+)?\.ckpt"

    return re.fullmatch(rendered, filename) is not None


def checkpoint_gate_entries(root: Path) -> set[str]:
    """Return new S5 claims without production checkpoint smoke evidence."""
    baseline_path = root / CHECKPOINT_GATE_BASELINE_RELATIVE_PATH
    grandfathered = (
        read_entry_baseline(baseline_path) if baseline_path.is_file() else set()
    )
    entries: set[str] = set()

    for path in training_docs(root):
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        claimed = has_s5_claim(text)
        signature_entry = f"{relative_path}\t{s5_record_signature(text)}"
        grandfathered_claim = claimed and signature_entry in grandfathered
        new_claim = claimed and not grandfathered_claim
        sections = parse_stage_evidence_sections(text)

        if not sections:
            if new_claim:
                entries.add(f"{relative_path}\tS3\t{CHECKPOINT_SMOKE_VIOLATION}")

            continue

        for _heading, evidence, _duplicates in sections:
            if not new_claim:
                continue

            if not is_bounded_production_checkpoint_smoke(
                evidence.get("S3"), evidence.get("S5"), path.parent.name, root
            ):
                entries.add(f"{relative_path}\tS3\t{CHECKPOINT_SMOKE_VIOLATION}")

    return entries


def check_training_stage_evidence(root: Path, baseline_path: Path) -> int:
    """Check current training evidence violations against the baseline."""
    current_snapshot = current_entries(root)
    baseline_snapshot = read_entry_baseline(baseline_path)
    unexpected, stale = diff_entry_baseline(current_snapshot, baseline_snapshot)
    checkpoint_gaps = checkpoint_gate_entries(root)
    if not unexpected and not stale and not checkpoint_gaps:
        return 0
    print_entries("New training stage evidence violations:", "+", unexpected)
    print_entries("Stale training stage evidence baseline entries:", "-", stale)
    print_entries("New S5 checkpoint smoke violations:", "+", sorted(checkpoint_gaps))
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
        checkpoint_gaps = checkpoint_gate_entries(ROOT)
        if not checkpoint_gaps:
            return 0

        print_entries(
            "New S5 checkpoint smoke violations:", "+", sorted(checkpoint_gaps)
        )
        return 1
    exit_status = check_training_stage_evidence(ROOT, BASELINE_PATH)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
