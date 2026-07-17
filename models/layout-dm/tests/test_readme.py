from __future__ import annotations

from pathlib import Path


def test_reproducibility_code_fences_are_tagged():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    section = readme.read_text(encoding="utf-8").split(
        "## Reproducibility", maxsplit=1
    )[1]
    section = section.split("## Vendor Links", maxsplit=1)[0]
    fence_lines = [line for line in section.splitlines() if line.startswith("```")]

    assert (
        "This section reproduces the parity verification against the original implementation."
        in section
    )
    assert len(fence_lines) % 2 == 0
    opening_fences = fence_lines[::2]
    closing_fences = fence_lines[1::2]

    assert all(fence in {"```bash", "```text"} for fence in opening_fences)
    assert all(fence == "```" for fence in closing_fences)
    assert opening_fences.count("```bash") == 5
    assert opening_fences.count("```text") == 2
