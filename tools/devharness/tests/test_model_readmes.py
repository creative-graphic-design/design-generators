"""Repository README contract tests."""

from __future__ import annotations

import os
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from devharness import cli
from devharness.checks.model_readmes import (
    card,
    check,
    citation,
    install,
    parity,
    reproducing,
    root_readme,
)
from devharness.checks.model_readmes.constants import find_repo_root

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_model_readmes_cli(cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = (
        f"{Path(sys.executable).parent}{os.pathsep}{environment['PATH']}"
    )
    return subprocess.run(
        ["devharness", "check", "model-readmes"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_model_readme_contracts() -> None:
    result = _run_model_readmes_cli(REPO_ROOT)
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout == "Model README checks passed.\n"
    assert result.stderr == ""


def test_model_readme_cli_resolves_root_from_subdirectory() -> None:
    result = _run_model_readmes_cli(REPO_ROOT / "models" / "layout-dm")
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout == "Model README checks passed.\n"
    assert result.stderr == ""


def test_model_readme_cli_reports_missing_workspace_root() -> None:
    result = _run_model_readmes_cli(Path("/proc"))
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Unable to find repository root with [tool.uv.workspace] in pyproject.toml\n"
    )


def test_find_repo_root_requires_workspace_declaration() -> None:
    with pytest.raises(ValueError, match=r"\[tool\.uv\.workspace\]"):
        find_repo_root(Path("/proc"))


def test_model_readme_policy_runs_in_process() -> None:
    check.check()


def test_model_readme_dispatch_rejects_unsupported_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["devharness", "check", "other"])

    assert cli.main() == 2
    assert capsys.readouterr().err == "usage: devharness check model-readmes\n"


def test_model_readme_main_reports_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check.main() == 0
    assert capsys.readouterr().out == "Model README checks passed.\n"


def test_model_readme_main_reports_policy_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise AssertionError("bad README")

    monkeypatch.setattr(check, "check", fail)

    assert check.main() == 1
    assert capsys.readouterr().err == "bad README\n"


def test_model_readme_cli_reports_failing_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "repository"
    archive = subprocess.run(
        ["git", "archive", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    fixture.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        tar.extractall(fixture)
    readme = fixture / "README.md"
    lines = readme.read_text(encoding="utf-8").splitlines(keepends=True)
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("| Model ")
    )
    lines[header_index] = (
        "| Model | Content | Conditioning | Venue | Runtime | Datasets | Ckpt | Train |\n"
    )
    readme.write_text(
        "".join(lines),
        encoding="utf-8",
    )

    result = _run_model_readmes_cli(fixture)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == (
        f"{fixture / 'README.md'}: Models table must use "
        "Model, Content, Conditioning, Venue, Ckpt, Train\n"
    )


def test_citation_contract_accepts_matching_arxiv_metadata() -> None:
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{yang2024posterllava,
  title = {PosterLLaVa: Constructing a Unified Multi-modal Layout Generator},
  author = {Tao Yang and Yingmin Luo},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2406.02884"
}
```
"""

    citation.assert_citation_bibtex(Path("models/example/README.md"), text)


def test_citation_contract_rejects_arxiv_id_mismatch() -> None:
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2303.08137},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2303.08137"
}
```
"""

    with pytest.raises(AssertionError, match="Citation arXiv ids"):
        citation.assert_citation_bibtex(Path("models/example/README.md"), text)


def test_citation_contract_rejects_missing_arxiv_bibtex_fields() -> None:
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  url = "https://arxiv.org/abs/2406.02884"
}
```
"""

    with pytest.raises(AssertionError, match="missing required fields"):
        citation.assert_citation_bibtex(Path("models/example/README.md"), text)


def test_citation_contract_rejects_eprint_url_mismatch() -> None:
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2406.02884)

## Citation

```bibtex
@misc{example2024,
  title = {Example},
  author = {Example Author},
  year = {2024},
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2303.08137"
}
```
"""

    with pytest.raises(AssertionError, match="does not match url"):
        citation.assert_citation_bibtex(Path("models/example/README.md"), text)


def test_citation_contract_allows_conference_bibtex_without_eprint() -> None:
    text = """# Model Card for Example

[Paper](https://arxiv.org/abs/2303.08137)

## Citation

```bibtex
@inproceedings{example2023,
  title = {Example},
  author = {Example Author},
  booktitle = {Proceedings of Example Conference},
  year = {2023}
}
```
"""

    citation.assert_citation_bibtex(Path("models/example/README.md"), text)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"url": "https://arxiv.org/abs/2406.02884"},
            "missing required field eprint",
        ),
        (
            {
                "eprint": "2406.02884",
                "archiveprefix": "bibtex",
                "primaryclass": "cs.CV",
                "url": "https://arxiv.org/abs/2406.02884",
            },
            "archivePrefix must be arXiv",
        ),
        (
            {
                "eprint": "not-an-arxiv-id",
                "archiveprefix": "arXiv",
                "primaryclass": "cs.CV",
                "url": "https://arxiv.org/abs/2406.02884",
            },
            "eprint must contain an arXiv id",
        ),
        (
            {
                "eprint": "2406.02884",
                "archiveprefix": "arXiv",
                "primaryclass": "cs.CV",
                "url": "https://example.com/paper",
            },
            "url must be an arXiv abs URL",
        ),
    ],
)
def test_arxiv_bibtex_fields_reject_invalid_metadata(
    fields: dict[str, str], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        citation._assert_arxiv_bibtex_fields(Path("models/example/README.md"), fields)


def test_citation_contract_requires_bibtex_fence() -> None:
    with pytest.raises(AssertionError, match="must contain a bibtex code fence"):
        citation.assert_citation_bibtex(
            Path("models/example/README.md"), "## Citation\nplain text\n"
        )


def test_citation_contract_accepts_arxiv_bibtex_without_body_id() -> None:
    text = """## Citation

```bibtex
@misc{example2024,
  eprint = {2406.02884},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = "https://arxiv.org/abs/2406.02884"
}
```
"""

    citation.assert_citation_bibtex(Path("models/example/README.md"), text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("### Parity Results\nNo badge", "missing vendor-parity badge"),
        (
            "### Parity Results\n![vendor-parity](https://img.shields.io/static/v1?message=wrong)",
            "does not match Parity Results",
        ),
    ],
)
def test_vendor_parity_badge_contract_rejects_invalid_badges(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        parity.assert_vendor_parity_badge(tmp_path / "README.md", text)


def test_reproducibility_contract_rejects_missing_link_and_walkthrough(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models" / "example" / "README.md"
    with pytest.raises(AssertionError, match="must link REPRODUCING.md"):
        reproducing.assert_readme_reproducibility_link(path, "## Reproducibility\n")

    text = (
        "## Reproducibility\nSee "
        "https://github.com/creative-graphic-design/design-generators/blob/main/"
        "models/example/REPRODUCING.md\n```bash\nrun\n```"
    )
    with pytest.raises(AssertionError, match="must be a short link"):
        reproducing.assert_readme_reproducibility_link(path, text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Workflow order: download reference pytest", "uv package commands"),
        (
            "uv run --package example\nWorkflow order: download reference pytest convert from_pretrained\n/tmp/old",
            "stale reproducibility command shape",
        ),
        (
            "uv run --package example\nWorkflow order: download reference pytest convert checkpoints smoke",
            "prompt-only REPRODUCING.md must not mention converting checkpoints",
        ),
        (
            "uv run --package example\nWorkflow order: download reference pytest convert",
            "missing reproducibility step",
        ),
    ],
)
def test_reproducing_command_contract_rejects_invalid_documents(
    tmp_path: Path, text: str, message: str
) -> None:
    slug = "layout-gpt" if "prompt-only" in message else "example"
    path = tmp_path / "models" / slug / "REPRODUCING.md"
    with pytest.raises(AssertionError, match=message):
        reproducing.assert_reproducing_commands(path, text)


def test_banned_content_contract_rejects_credentials() -> None:
    with pytest.raises(AssertionError, match="GEN_AI_PROXY_PAT"):
        card.assert_banned_patterns(
            Path("models/example/README.md"), "GEN_AI_PROXY_PAT"
        )


def test_card_parsing_helpers_cover_empty_and_inline_cases() -> None:
    assert card._frontmatter("plain") == ""
    assert card._frontmatter("---\nlicense: mit\n") == ""
    assert card._frontmatter_scalar("license: mit", "missing") is None
    assert card._frontmatter_list("license: mit", "datasets") == []
    assert card._dataset_display_name("custom") == "custom"
    assert card.semantic_badge_label("plain", "library") == "library"
    assert card.without_badges("A ![badge](url) B") == "A   B"
    assert card.markdown_link_spans("[link](url)")
    assert not card.in_any_span(0, [])
    assert card.inline_code_span_at("plain", 0) is None
    assert (
        card.without_frontmatter_and_code(
            "---\nlicense: mit\n---\ntext\n```\nhidden\n```\nvisible"
        )
        == "text\nvisible"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("plain", "missing YAML frontmatter"),
        ("---\nlicense: mit\n", "unterminated YAML frontmatter"),
        ("---\nlicense: mit\n---\n", "frontmatter missing language:"),
    ],
)
def test_frontmatter_contract_rejects_invalid_documents(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        card.assert_frontmatter(tmp_path / "README.md", text)


def test_frontmatter_policy_rejects_duplicates_and_invalid_pipeline() -> None:
    path = Path("models/example/README.md")
    with pytest.raises(AssertionError, match="has duplicates"):
        card.assert_frontmatter_list_unique(path, "tags:\n  - one\n  - one\n")

    with pytest.raises(AssertionError, match="pipeline_tag"):
        card.assert_pipeline_tag(path, "pipeline_tag: text")

    with pytest.raises(AssertionError, match="must not be text-to-image"):
        card.assert_pipeline_tag(
            path,
            'pipeline_tag: other\nmodel-index:\n  - name: example\n    results:\n      type: "text-to-image"',
        )

    with pytest.raises(AssertionError, match="task.type must be 'other'"):
        card.assert_pipeline_tag(
            path,
            "pipeline_tag: other\nmodel-index:\n  - task:\n      type: text-classification",
        )


def test_model_index_policy_rejects_wrong_weight_mode() -> None:
    prompt_path = Path("models/layout-gpt/README.md")
    weight_path = Path("models/layout-dm/README.md")
    with pytest.raises(AssertionError, match="prompt-only README"):
        card.assert_model_index_policy(prompt_path, "model-index:\n")

    with pytest.raises(AssertionError, match="must include model-index"):
        card.assert_model_index_policy(weight_path, "pipeline_tag: other\n")


def test_card_and_parity_contracts_reject_missing_heading_and_bad_parity() -> None:
    path = Path("models/example/README.md")
    with pytest.raises(AssertionError, match="missing required heading"):
        card.assert_heading_order(path, "# Model Card for Example\n")

    with pytest.raises(AssertionError, match="Parity Results must contain"):
        parity.assert_parity_table(path, "### Parity Results\nplain\n")

    with pytest.raises(AssertionError, match="has no data rows"):
        parity.assert_parity_table(
            path, "### Parity Results\n| Name | Result |\n| --- | --- |\n"
        )

    with pytest.raises(AssertionError, match="numeric evidence"):
        parity.assert_parity_table(
            path,
            "### Parity Results\n| Name | Result |\n| --- | --- |\n| case | match |\n",
        )


def test_card_contracts_reject_bad_summary_and_code_fences() -> None:
    path = Path("models/layout-dm/README.md")
    with pytest.raises(AssertionError, match="missing model-card H1"):
        card.assert_model_summary_subject(path, "plain")

    with pytest.raises(AssertionError, match="first prose line"):
        card.assert_model_summary_subject(path, "# Model Card for LayoutDM\nWrong")

    with pytest.raises(AssertionError, match="untagged code fence"):
        card.assert_code_fences_tagged(path, "```\ncode\n```\n")

    with pytest.raises(AssertionError, match="unterminated code fence"):
        card.assert_code_fences_tagged(path, "```bash\ncode\n")

    with pytest.raises(AssertionError, match="heredoc examples"):
        card.assert_code_fences_tagged(path, "<<'PY'\ncode\nPY\n")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            "- **Model type:** image-aware; conditioning: none.\n",
            "content value",
        ),
        (
            "- **Model type:** content-agnostic; conditioning: label-size.\n",
            "conditioning value",
        ),
        (
            "- **Model type:** content-agnostic; conditioning: label_size, label.\n",
            "canonical order",
        ),
        (
            "- **Model type:** content-agnostic; conditioning: label, label.\n",
            "must be unique",
        ),
        (
            "- **Model type:** content-agnostic; conditioning: evaluation, label.\n",
            "must appear alone",
        ),
    ],
)
def test_model_type_contract_rejects_invalid_values(line: str, message: str) -> None:
    with pytest.raises(AssertionError, match=message):
        card.parse_model_type(Path("models/example/README.md"), line)


def test_model_type_contract_rejects_missing_and_duplicate_lines() -> None:
    path = Path("models/example/README.md")
    with pytest.raises(AssertionError, match="exactly once"):
        card.parse_model_type(path, "plain")

    duplicate = "- **Model type:** content-agnostic; conditioning: label.\n" * 2
    with pytest.raises(AssertionError, match="exactly once"):
        card.parse_model_type(path, duplicate)


def test_model_type_contract_accepts_structured_line() -> None:
    assert card.parse_model_type(
        Path("models/example/README.md"),
        "- **Model type:** content-aware; conditioning: label, label_size.\n",
    ) == ("content-aware", ("label", "label_size"))


def test_root_models_table_accepts_linked_model_names_and_reproduction_badges(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Content | Conditioning | Venue | Ckpt | Train |
| :--- | :--- | :--- | :---: | --- | --- |
| [`LayoutFormer++`](models/layoutformerpp/README.md) | content-agnostic | unconditional, label, label_size, completion, refinement, relation | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layoutformerpp/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries
""",
        encoding="utf-8",
    )

    assert root_readme.root_model_slugs(readme) == {"layoutformerpp"}


def _write_model_type_fixture(root: Path, slug: str, line: str) -> None:
    model_readme = root / "models" / slug / "README.md"
    model_readme.parent.mkdir(parents=True)
    model_readme.write_text(line, encoding="utf-8")


def _docs_model_row(conditioning: str) -> str:
    return (
        "| [`LayoutDM`](api/models/layout-dm/) | content-agnostic | "
        f"{conditioning} | ![venue: CVPR 2023](https://img.shields.io/static/v1?"
        "label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | "
        "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&"
        "message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | "
        "[![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&"
        "message=train&color=success)](models/layout-dm/TRAINING.md) | "
        "[Paper](https://arxiv.org/abs/2303.03755) | "
        "[README](models/layout-dm/README.md) |"
    )


def test_root_model_table_rejects_content_mismatch(tmp_path: Path) -> None:
    _write_model_type_fixture(
        tmp_path,
        "layout-dm",
        "- **Model type:** content-agnostic; conditioning: unconditional, label, "
        "label_size, completion, refinement.\n",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Models\n\n"
        "| Model | Content | Conditioning | Venue | Ckpt | Train |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| [`LayoutDM`](models/layout-dm/README.md) | content-aware | "
        "unconditional, label, label_size, completion, refinement | "
        "![venue: CVPR 2023](https://img.shields.io/static/v1?label=venue&"
        "message=CVPR%202023&color=0076a8) | "
        "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&"
        "message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | "
        "[![training: train](https://img.shields.io/static/v1?label=training&"
        "message=train&color=success)](models/layout-dm/TRAINING.md) |\n\n"
        "## Libraries\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="package layout-dm Content cell"):
        root_readme.root_model_slugs(readme)


def test_root_model_table_rejects_conditioning_mismatch(tmp_path: Path) -> None:
    _write_model_type_fixture(
        tmp_path,
        "layout-dm",
        "- **Model type:** content-agnostic; conditioning: unconditional, label, "
        "label_size, completion, refinement.\n",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Models\n\n"
        "| Model | Content | Conditioning | Venue | Ckpt | Train |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| [`LayoutDM`](models/layout-dm/README.md) | content-agnostic | label | "
        "![venue: CVPR 2023](https://img.shields.io/static/v1?label=venue&"
        "message=CVPR%202023&color=0076a8) | "
        "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&"
        "message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | "
        "[![training: train](https://img.shields.io/static/v1?label=training&"
        "message=train&color=success)](models/layout-dm/TRAINING.md) |\n\n"
        "## Libraries\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="package layout-dm Conditioning cell"):
        root_readme.root_model_slugs(readme)


def test_docs_index_model_table_accepts_matching_model_type(tmp_path: Path) -> None:
    _write_model_type_fixture(
        tmp_path,
        "layout-dm",
        "- **Model type:** content-agnostic; conditioning: unconditional, label, "
        "label_size, completion, refinement.\n",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text(
        "## Models\n\n"
        "| Model | Content | Conditioning | Venue | Weights | Training | Paper | Docs |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"{_docs_model_row('unconditional, label, label_size, completion, refinement')}\n",
        encoding="utf-8",
    )

    assert root_readme.root_model_slugs(index) == {"layout-dm"}


def test_docs_index_model_table_reports_missing_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_model_type_fixture(
        tmp_path,
        "layout-dm",
        "- **Model type:** content-agnostic; conditioning: unconditional, label, "
        "label_size, completion, refinement.\n",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.md"
    index.write_text(
        "## Models\n\n"
        "| Model | Content | Conditioning | Venue | Weights | Training | Paper | Docs |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        f"{_docs_model_row('unconditional, label, label_size, completion, refinement')}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        root_readme, "model_member_slugs", lambda: {"layout-dm", "layout-flow"}
    )

    slugs = root_readme.root_model_slugs(index)
    with pytest.raises(AssertionError, match="docs/index.md Models table mismatch"):
        root_readme.assert_root_models_table_matches_members(slugs, index)


def test_root_models_table_rejects_metadata_columns(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Venue | Runtime | Datasets | Ckpt | Train |
| --- | --- | --- | --- | --- | --- |
| [`LayoutDM`](models/layout-dm/README.md) | ![venue: CVPR 2023](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=CVPR%202023&color=0076a8) | `🧨diffusers` | ![dataset: PubLayNet](https://img.shields.io/static/v1?label=%F0%9F%97%82%EF%B8%8F&message=PubLayNet&color=2f80ed) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-dm/REPRODUCING.md) | ![training: n/a](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=n%2Fa&color=lightgrey) |

## Libraries
""",
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError, match="Model, Content, Conditioning, Venue, Ckpt, Train"
    ):
        root_readme.root_model_slugs(readme)


def test_model_readme_reproducibility_rejects_repo_root_link(tmp_path: Path) -> None:
    readme = tmp_path / "models" / "layout-dm" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        """# Model Card for LayoutDM

## Reproducibility

See [REPRODUCING.md](models/layout-dm/REPRODUCING.md) for commands.

## Environmental Impact
""",
        encoding="utf-8",
    )

    with pytest.raises(
        AssertionError, match="blob/main/models/layout-dm/REPRODUCING.md"
    ):
        reproducing.assert_readme_reproducibility_link(
            readme, readme.read_text(encoding="utf-8")
        )


def test_root_models_table_requires_training_link_when_file_exists(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        """# Example

## Models

| Model | Content | Conditioning | Venue | Ckpt | Train |
| --- | --- | --- | --- | --- | --- |
| [`LayoutFlow`](models/layout-flow/README.md) | content-agnostic | unconditional, label, label_size, completion, refinement | ![venue: ECCV 2024](https://img.shields.io/static/v1?label=%F0%9F%8E%93&message=ECCV%202024&color=009688) | [![checkpoint: ckpt](https://img.shields.io/static/v1?label=%F0%9F%92%BE&message=ckpt&color=success)](models/layout-flow/REPRODUCING.md) | [![training: train](https://img.shields.io/static/v1?label=%F0%9F%8F%8B%EF%B8%8F&message=train&color=success)](models/layout-flow/TRAINING.md) |

## Libraries
""",
        encoding="utf-8",
    )

    assert root_readme.root_model_slugs(readme) == {"layout-flow"}


def test_root_readme_table_helpers_reject_malformed_cells() -> None:
    with pytest.raises(AssertionError, match="missing Models table"):
        root_readme._root_packages_table_lines("## Models\n\nplain")

    with pytest.raises(AssertionError, match="malformed markdown table row"):
        root_readme._split_markdown_table_row("plain")

    assert not root_readme._is_markdown_table_separator("| --- |", 2)
    assert root_readme._badge_message("plain", "https://example.com", "library") is None
    assert root_readme._linked_static_badge("plain", "library") is None
    assert root_readme._static_badge_messages("plain", "library") == []

    with pytest.raises(AssertionError, match="badge missing color"):
        root_readme._static_badge_colors(
            "![library: laygen](https://img.shields.io/static/v1?label=library&message=laygen)",
            "library",
        )


def test_root_reproduction_cells_require_valid_badges(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    with pytest.raises(AssertionError, match="one linked checkpoint"):
        root_readme._assert_root_reproduction_cells(
            path,
            "layout-dm",
            "![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&message=ckpt&color=success)",
            "![training: n/a](https://img.shields.io/static/v1?label=training&message=n%2Fa&color=lightgrey)",
        )

    with pytest.raises(AssertionError, match="must not link training badge"):
        root_readme._assert_root_reproduction_cells(
            path,
            "layoutformerpp",
            "[![checkpoint: ckpt](https://img.shields.io/static/v1?label=checkpoint&message=ckpt&color=success)](models/layoutformerpp/REPRODUCING.md)",
            "[![training: train](https://img.shields.io/static/v1?label=training&message=train&color=success)](models/layoutformerpp/TRAINING.md)",
        )


@pytest.mark.parametrize(
    "text",
    [
        "Transformers",
        "🤗pydantic-ai",
        "pydantic-ai",
    ],
)
def test_library_name_style_rejects_unannotated_names(
    tmp_path: Path, text: str
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(text, encoding="utf-8")

    with pytest.raises(AssertionError):
        root_readme.assert_library_name_style(readme)


def test_linked_first_reference_policy_requires_dataset_links(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("creative-graphic-design/Rico", encoding="utf-8")

    with pytest.raises(AssertionError, match="dataset id must be linked"):
        root_readme.assert_linked_first_reference_policy(readme)


def test_install_helpers_cover_metadata_and_install_contract_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert card.section("# Title", "## Missing") == ""
    assert root_readme.normalize_root_repo_link("models/example/README.md") == (
        "models/example/README.md"
    )
    assert install._dependency_name("laygen[agents]>=1") == "laygen"
    assert install._dependency_direct_name("laygen[agents]>=1") == "laygen[agents]"
    assert install._pip_install_snippet([("example", "models/example")]).startswith(
        'pip install "example @ '
    )

    monkeypatch.setattr(install, "project_metadata", lambda _: {"project": "bad"})
    with pytest.raises(AssertionError, match=r"missing \[project\]"):
        install._project_name(tmp_path)

    monkeypatch.setattr(install, "project_metadata", lambda _: {"project": {}})
    with pytest.raises(AssertionError, match="project.name missing"):
        install._project_name(tmp_path)

    monkeypatch.setattr(
        install,
        "project_metadata",
        lambda _: {"project": {"dependencies": "bad"}},
    )
    with pytest.raises(AssertionError, match="dependencies must be a list"):
        install._project_dependencies(tmp_path)

    monkeypatch.setattr(install, "project_metadata", lambda _: {"tool": "bad"})
    assert install._workspace_source_names(tmp_path) == set()

    with pytest.raises(AssertionError, match="must include a pip install snippet"):
        install.assert_pip_install_snippet(
            tmp_path / "README.md",
            "## Install\n",
            [("example", "lib/example")],
            "Install",
        )


def test_expected_frontmatter_contract_reports_policy_errors() -> None:
    with pytest.raises(AssertionError, match="no expected frontmatter"):
        card.assert_expected_frontmatter(Path("models/example/README.md"), "---\n---\n")

    frontmatter = """---
license: mit
pipeline_tag: other
datasets:
  - creative-graphic-design/Rico
model-index:
  - task:
      type: other
---
"""
    path = Path("models/layout-dm/README.md")
    with pytest.raises(AssertionError, match="frontmatter license"):
        card.assert_expected_frontmatter(path, frontmatter)

    with pytest.raises(AssertionError, match="datasets missing"):
        card.assert_expected_frontmatter(
            path, frontmatter.replace("license: mit", "license: apache-2.0")
        )

    with pytest.raises(AssertionError, match="dataset badges"):
        card.assert_expected_frontmatter(
            path,
            frontmatter.replace("license: mit", "license: apache-2.0").replace(
                "creative-graphic-design/Rico",
                "creative-graphic-design/Rico\n  - creative-graphic-design/PubLayNet",
            ),
        )


@pytest.mark.parametrize(
    ("project_metadata", "message"),
    [
        ({}, r"missing \[tool\]"),
        ({"tool": {}}, r"missing \[tool.design-generators\]"),
        ({"tool": {"design-generators": {}}}, "framework must be a string"),
        (
            {"tool": {"design-generators": {"framework": "other"}}},
            "expected exactly one base badge",
        ),
    ],
)
def test_card_runtime_contract_rejects_incomplete_metadata(
    monkeypatch: pytest.MonkeyPatch,
    project_metadata: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(card, "project_metadata", lambda _: project_metadata)

    with pytest.raises(AssertionError, match=message):
        card.assert_runtime_contract(
            Path("models/layout-dm/README.md"), "---\nlibrary_name: other\n---\n"
        )


def test_runtime_contract_rejects_surface_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        card,
        "project_metadata",
        lambda _: {"tool": {"design-generators": {"framework": "other"}}},
    )
    text = """---
library_name: transformers
---
![base: other](https://img.shields.io/static/v1?label=base&message=other)
"""

    with pytest.raises(AssertionError, match="runtime mismatch"):
        card.assert_runtime_contract(Path("models/layout-dm/README.md"), text)


def test_card_contracts_cover_prompt_and_unpublished_notes() -> None:
    path = Path("models/layout-gpt/README.md")
    with pytest.raises(AssertionError, match="stale model-package phrase"):
        card.assert_prompt_only_readme(path, "converted checkpoint")

    with pytest.raises(AssertionError, match="must not mention converting"):
        card.assert_prompt_only_readme(path, "convert checkpoints")

    with pytest.raises(AssertionError, match="missing runnable setup parts"):
        card.assert_unpublished_hub_get_started_note(
            path, "## Supported Checkpoints\n## How to Get Started with the Model\n"
        )

    with pytest.raises(AssertionError, match="heredoc examples"):
        card.assert_unpublished_hub_get_started_note(
            Path("models/layout-dm/README.md"),
            "## Supported Checkpoints\ncreative-graphic-design/x not-published\n"
            "## How to Get Started with the Model\n<<'PY'\n",
        )


def test_install_contract_requires_both_install_forms() -> None:
    path = Path("lib/laygen/README.md")
    with pytest.raises(AssertionError, match="direct-reference"):
        install.assert_lib_readme_install_contract(path, "## Install\n")

    direct = (
        'pip install "laygen @ git+https://github.com/creative-graphic-design/'
        'design-generators.git#subdirectory=lib/laygen"'
    )
    with pytest.raises(AssertionError, match="workspace uv sync"):
        install.assert_lib_readme_install_contract(path, f"## Install\n{direct}\n")


def test_card_contracts_require_expected_source_and_summary_subject() -> None:
    with pytest.raises(AssertionError, match="must link expected repository"):
        card.assert_expected_repository_links(
            Path("models/layout-gpt/README.md"), "### Model Sources\n"
        )

    with pytest.raises(AssertionError, match="first prose line"):
        card.assert_model_summary_subject(
            Path("models/layout-dm/README.md"),
            "# Model Card for LayoutDM\nLayoutDM ports layouts.\n",
        )


def test_hugging_face_emoji_contract_allows_multiple_runtime_mentions(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and second `🤗transformers`.",
        encoding="utf-8",
    )

    root_readme.assert_library_name_style(readme)


@pytest.mark.parametrize(
    ("emoji", "library", "expected_message"),
    [
        ("🤗", "transformers", "🤗 must annotate"),
        ("🧨", "diffusers", "🧨 must annotate"),
        ("🤖", "pydantic-ai", "🤖 must annotate"),
    ],
)
def test_runtime_emoji_contract_rejects_space_after_emoji(
    tmp_path: Path,
    emoji: str,
    library: str,
    expected_message: str,
) -> None:
    readme = tmp_path / "README.md"
    runtime_label = f"`{emoji} {library}`"
    readme.write_text(f"This package uses {runtime_label}.", encoding="utf-8")

    with pytest.raises(AssertionError, match=expected_message):
        root_readme.assert_library_name_style(readme)


def test_hugging_face_emoji_contract_rejects_unattached_emoji(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🧨diffusers`](https://huggingface.co/docs/diffusers/index) "
        "and stray 🤗.",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        root_readme.assert_library_name_style(readme)


def test_hugging_face_emoji_contract_rejects_emoji_outside_code_span(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🤗 [`transformers`](https://huggingface.co/docs/transformers/index).",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        root_readme.assert_library_name_style(readme)


def test_diffusers_emoji_contract_rejects_unattached_emoji(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "First [`🤗transformers`](https://huggingface.co/docs/transformers/index) "
        "and stray 🧨.",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        root_readme.assert_library_name_style(readme)


def test_diffusers_emoji_contract_rejects_emoji_outside_code_span(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "First 🧨 [`diffusers`](https://huggingface.co/docs/diffusers/index).",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="must annotate"):
        root_readme.assert_library_name_style(readme)


def test_pip_install_contract_reports_expected_direct_url_example() -> None:
    section = """Clone this repository first.

```bash
uv sync --package layout-dm
```
"""

    with pytest.raises(AssertionError, match="Expected example") as exc_info:
        install.assert_pip_install_snippet(
            Path("models/layout-dm/README.md"),
            section,
            [
                ("laygen", "lib/laygen"),
                ("layout-dm", "models/layout-dm"),
            ],
            "How to Get Started",
        )

    message = str(exc_info.value)
    assert "laygen @ git+https://github.com/creative-graphic-design" in message
    assert "subdirectory=models/layout-dm" in message


def test_library_pip_install_contract_accepts_direct_url() -> None:
    section = """Install directly from this repository.

```bash
pip install "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen"
```
"""

    install.assert_pip_install_snippet(
        Path("lib/laygen/README.md"),
        section,
        [("laygen", "lib/laygen")],
        "Install",
    )


def test_model_install_contract_requires_workspace_model_dependencies() -> None:
    text = """# Model Card for SmartText

## How to Get Started with the Model

```bash
pip install \\
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \\
  "smarttext @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/smarttext"
```

## Training Details
"""

    with pytest.raises(AssertionError, match="models/basnet"):
        install.assert_model_pip_install_snippet(
            REPO_ROOT / "models" / "smarttext" / "README.md",
            text,
        )
