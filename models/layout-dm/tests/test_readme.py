from __future__ import annotations

from pathlib import Path


def _reproducing_section() -> str:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    section = readme.read_text(encoding="utf-8").split(
        "## Reproducibility", maxsplit=1
    )[1]
    return section.split("## Vendor Links", maxsplit=1)[0]


def test_reproducibility_code_fences_are_tagged():
    section = _reproducing_section()
    fence_lines = [line for line in section.splitlines() if line.startswith("```")]

    assert len(fence_lines) % 2 == 0
    opening_fences = fence_lines[::2]
    closing_fences = fence_lines[1::2]

    assert all(fence in {"```bash", "```text"} for fence in opening_fences)
    assert all(fence == "```" for fence in closing_fences)
    assert opening_fences.count("```bash") == 5
    assert opening_fences.count("```text") == 2


def test_reproducibility_bash_blocks_run_from_repo_root():
    section = _reproducing_section()
    bash_blocks = [
        block.split("```", maxsplit=1)[0] for block in section.split("```bash\n")[1:]
    ]

    joined = "\n".join(bash_blocks)
    assert "cd models/layout-dm" not in joined
    assert "python models/layout-dm/scripts/download_original.py" in joined
    assert "python models/layout-dm/scripts/generate_reference_outputs.py" in joined
    assert "python models/layout-dm/scripts/convert_original_checkpoint.py" in joined
    assert "--starter-dir .cache/layout-dm/original/download" in joined
    assert "uv run --package layout-dm --extra vendor" in joined
