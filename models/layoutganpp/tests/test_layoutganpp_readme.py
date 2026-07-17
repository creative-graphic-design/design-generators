from __future__ import annotations

from pathlib import Path


def test_layoutganpp_readme_code_fences_are_tagged():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "## Reproducibility" in text
    assert (
        "This section reproduces the parity verification against the original "
        "implementation."
    ) in text
    assert "## Reproducing Vendor Parity" not in text
    fence_lines = [line for line in text.splitlines() if line.startswith("```")]

    assert len(fence_lines) % 2 == 0
    opening_fences = fence_lines[::2]
    closing_fences = fence_lines[1::2]

    assert all(
        fence in {"```bash", "```python", "```text", "```bibtex"}
        for fence in opening_fences
    )
    assert all(fence == "```" for fence in closing_fences)


def test_layoutganpp_readme_omits_internal_implementation_wording():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    forbidden = [
        "ModelCard",
        "from_template",
        "validate()",
        "annotated",
        "laygen.common.model_card",
        "GEN_AI",
        "flava",
        "gateway",
        "cloud-dev",
        "PAT",
        "proxy",
    ]
    assert not any(term in text for term in forbidden)
