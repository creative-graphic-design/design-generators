from __future__ import annotations

from pathlib import Path


def test_reproducibility_code_fences_are_tagged():
    model_dir = Path(__file__).resolve().parents[1]
    readme = (model_dir / "README.md").read_text(encoding="utf-8")
    reproducing = (model_dir / "REPRODUCING.md").read_text(encoding="utf-8")
    assert "models/layout-dm/REPRODUCING.md" in readme

    fence_lines = [line for line in reproducing.splitlines() if line.startswith("```")]

    assert len(fence_lines) % 2 == 0
    opening_fences = fence_lines[::2]
    closing_fences = fence_lines[1::2]

    assert all(fence in {"```bash", "```text"} for fence in opening_fences)
    assert all(fence == "```" for fence in closing_fences)
    assert opening_fences.count("```bash") == 6
    assert opening_fences.count("```text") == 2
