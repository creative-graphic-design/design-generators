"""BibTeX and arXiv citation contracts."""

from __future__ import annotations

import re
from pathlib import Path

from .card import section, without_frontmatter_and_code


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


def _bibtex_fences(section_text: str) -> list[str]:
    return re.findall(r"```bibtex\n(.*?)\n```", section_text, flags=re.S)


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


def assert_citation_bibtex(path: Path, text: str) -> None:
    """Require BibTeX citation fields and matching arXiv identifiers."""
    citation_section = section(text, "## Citation")
    # Coordinator approval is required before adding exceptions to this bibtex
    # requirement; README normalization must preserve citation metadata.
    if "```bibtex" not in citation_section:
        raise AssertionError(f"{path}: Citation must contain a bibtex code fence")

    body = without_frontmatter_and_code(text.replace(citation_section, "", 1))
    body_arxiv_ids = _arxiv_ids(body)
    citation_arxiv_ids: set[str] = set()
    for entry in _bibtex_fences(citation_section):
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
