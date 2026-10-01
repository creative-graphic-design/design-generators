"""Citation, parity, reproducibility, and banned-content contracts."""

from __future__ import annotations

import re
from pathlib import Path

from .constants import BANNED_PATTERNS, PROMPT_ONLY_SLUGS
from .metadata import _without_frontmatter_and_code
from .repository import _section


ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
ARXIV_URL_RE = re.compile(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?")
BIBTEX_FIELD_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(.+?)(?:,)?\s*$")


def _normalize_arxiv_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.strip())


def _arxiv_ids(text: str) -> set[str]:
    ids = {_normalize_arxiv_id(match.group(1)) for match in ARXIV_URL_RE.finditer(text)}
    ids.update(
        _normalize_arxiv_id(match.group(1))
        for match in re.finditer(
            r"\barXiv(?::|\s+)(\d{4}\.\d{4,5}(?:v\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    return ids


def _bibtex_fences(section: str) -> list[str]:
    return re.findall(r"```bibtex\n(.*?)\n```", section, flags=re.S)


def _clean_bibtex_field_value(value: str) -> str:
    cleaned = value.strip().rstrip(",").strip()
    while len(cleaned) >= 2 and (
        (cleaned.startswith("{") and cleaned.endswith("}"))
        or (cleaned.startswith('"') and cleaned.endswith('"'))
    ):
        cleaned = cleaned[1:-1].strip()

    return cleaned


def _bibtex_fields(entry: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in entry.splitlines():
        match = BIBTEX_FIELD_RE.match(line)
        if match:
            fields[match.group(1).lower()] = _clean_bibtex_field_value(match.group(2))

    return fields


def _assert_arxiv_bibtex_fields(path: Path, fields: dict[str, str]) -> set[str]:
    citation_ids = _arxiv_ids("\n".join(fields.values()))
    has_arxiv_url = "url" in fields and bool(ARXIV_URL_RE.search(fields["url"]))
    has_eprint = "eprint" in fields
    if not has_eprint and has_arxiv_url:
        raise AssertionError(
            f"{path}: Citation arXiv BibTeX is missing required field eprint"
        )

    if not has_eprint:
        return citation_ids

    required = {"archiveprefix", "primaryclass", "url"}
    missing = sorted(field for field in required if not fields.get(field))
    if missing:
        raise AssertionError(
            f"{path}: Citation arXiv BibTeX is missing required fields {missing}"
        )

    if fields["archiveprefix"].lower() != "arxiv":
        raise AssertionError(
            f"{path}: Citation archivePrefix must be arXiv for eprint entries"
        )

    eprint_match = ARXIV_ID_RE.search(fields["eprint"])
    if eprint_match is None:
        raise AssertionError(f"{path}: Citation eprint must contain an arXiv id")

    eprint_id = _normalize_arxiv_id(eprint_match.group(0))
    url_match = ARXIV_URL_RE.search(fields["url"])
    if url_match is None:
        raise AssertionError(f"{path}: Citation url must be an arXiv abs URL")

    url_id = _normalize_arxiv_id(url_match.group(1))
    if eprint_id != url_id:
        raise AssertionError(
            f"{path}: Citation eprint {eprint_id} does not match url {url_id}"
        )

    return citation_ids | {eprint_id, url_id}


def _assert_citation_bibtex(path: Path, text: str) -> None:
    section = _section(text, "## Citation")
    # Coordinator approval is required before adding exceptions to this bibtex
    # requirement; README normalization must preserve citation metadata.
    if "```bibtex" not in section:
        raise AssertionError(f"{path}: Citation must contain a bibtex code fence")

    body = _without_frontmatter_and_code(text.replace(section, "", 1))
    body_arxiv_ids = _arxiv_ids(body)
    citation_arxiv_ids: set[str] = set()
    for entry in _bibtex_fences(section):
        citation_arxiv_ids.update(
            _assert_arxiv_bibtex_fields(path, _bibtex_fields(entry))
        )

    if citation_arxiv_ids and body_arxiv_ids:
        unexpected = sorted(citation_arxiv_ids - body_arxiv_ids)
        if unexpected:
            raise AssertionError(
                f"{path}: Citation arXiv ids {unexpected} do not match README "
                f"arXiv ids {sorted(body_arxiv_ids)}"
            )


def _nonzero_number(text: str) -> bool:
    try:
        return float(text) != 0
    except ValueError:
        return False


def _parity_requires_tolerance(section: str) -> bool:
    for match in re.finditer(r"\b[ra]tol\s*=?\s*`?([0-9.eE+-]+)`?", section):
        if _nonzero_number(match.group(1)):
            return True

    return False


def _assert_vendor_parity_badge(path: Path, text: str) -> None:
    section = _section(text, "### Parity Results")
    badge = re.search(r"!\[vendor-parity\]\([^)]*[?&]message=([^&)]*)", text)
    if badge is None:
        raise AssertionError(f"{path}: missing vendor-parity badge")

    expected = (
        "not-run"
        if "not run" in section
        else "practical-reproduction"
        if "practical training reproduction" in section
        else "cpu-contract"
        if "vendor-parity CPU comparison" in section
        else "tolerance-verified"
        if _parity_requires_tolerance(section)
        else "bit-exact"
    )
    actual = badge.group(1)
    accepted = {
        expected,
        expected.replace("--", "-"),
    }
    if actual not in accepted:
        raise AssertionError(
            f"{path}: vendor-parity badge {actual!r} does not match Parity Results; expected {expected!r}"
        )


def _assert_readme_reproducibility_link(path: Path, text: str) -> None:
    section = _section(text, "## Reproducibility")
    absolute_link = (
        "https://github.com/creative-graphic-design/design-generators/blob/main/"
        f"models/{path.parent.name}/REPRODUCING.md"
    )
    repo_root_link = f"models/{path.parent.name}/REPRODUCING.md"
    if absolute_link not in section and repo_root_link not in section:
        raise AssertionError(
            f"{path}: Reproducibility must link REPRODUCING.md as {repo_root_link} or {absolute_link}"
        )

    if "uv run --package " in section or "```" in section:
        raise AssertionError(
            f"{path}: README Reproducibility must be a short link, not a walkthrough"
        )


def _assert_reproducing_commands(path: Path, text: str) -> None:
    if "uv run --package " not in text:
        raise AssertionError(f"{path}: REPRODUCING.md must contain uv package commands")

    lower = text.lower()
    bad_command_shapes = ["python scripts/", "cd models/", "../.cache", "/tmp/"]
    for bad in bad_command_shapes:
        if bad in text:
            raise AssertionError(
                f"{path}: stale reproducibility command shape contains {bad!r}"
            )

    required_terms: list[str | tuple[str, ...]] = [
        "Workflow order:",
        "download",
        ("reference", "golden"),
        "pytest",
    ]
    if path.parent.name in PROMPT_ONLY_SLUGS:
        required_terms.extend([("prompt configuration", "save_pretrained"), "smoke"])
        if "convert checkpoints" in lower:
            raise AssertionError(
                f"{path}: prompt-only REPRODUCING.md must not mention converting checkpoints"
            )
    else:
        required_terms.extend(["convert", "from_pretrained"])

    for term in required_terms:
        alternatives = (term,) if isinstance(term, str) else term
        position = max(lower.find(alternative.lower()) for alternative in alternatives)
        if position == -1:
            raise AssertionError(f"{path}: missing reproducibility step {term!r}")

    if path.parent.name in {"coarse-to-fine", "layoutganpp"}:
        expected = (
            "Workflow order: download assets, generate references, convert checkpoints, "
            "run parity checks, then smoke-test local loading."
        )
        if expected not in text:
            raise AssertionError(
                f"{path}: reproducibility workflow must state reference -> conversion -> parity order"
            )


def _assert_banned_patterns(path: Path, text: str) -> None:
    for pattern in BANNED_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raise AssertionError(
                f"{path}: banned README content matched {pattern!r}: {match.group(0)!r}"
            )
