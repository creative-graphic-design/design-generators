"""Check that the real Python bbox implementation is protected by the properties.

This deliberately changes source code only within the ephemeral CI checkout.
No mutation or new regression is committed, and the source is restored on exit.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from laygen.common import bbox

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lib" / "laygen" / "src" / "laygen" / "common" / "bbox.py"
TEST = "verification/bbox/test_bbox_properties.py::test_xywh_ltrb_roundtrip"

BEFORE = "return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)"
AFTER = "return torch.stack((x - w / 3, y - h / 2, x + w / 2, y + h / 2), dim=-1)"


def run_property(*, expect_failure: bool) -> None:
    """Run the real property in a fresh interpreter, detecting assertion failures."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-x"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if expect_failure:
        assert result.returncode == 1, (
            f"Mutant did not cause a pytest assertion failure: {result.returncode}\n{output}"
        )
        assert "Falsifying example:" in output, (
            f"Unexpected failure; mutation was not detected by Hypothesis:\n{output}"
        )
        print("Mutated production function: property FAILED as intended.")
    else:
        assert result.returncode == 0, (
            f"Unmodified production function did not pass: {result.returncode}\n{output}"
        )
        print("Original production function: property PASSED.")


def main() -> None:
    """Compare baseline, actual-source mutant, and restored baseline."""
    loaded = Path(inspect.getsourcefile(bbox.xywh_to_ltrb) or "").resolve()
    assert loaded == SOURCE.resolve(), (
        f"Test did not import the real repository source: {loaded} != {SOURCE}"
    )

    original = SOURCE.read_text(encoding="utf-8")
    assert original.count(BEFORE) == 1, (
        "The expected original expression is absent or ambiguous; review the mutation."
    )

    run_property(expect_failure=False)

    try:
        SOURCE.write_text(original.replace(BEFORE, AFTER, 1), encoding="utf-8")
        run_property(expect_failure=True)
    finally:
        SOURCE.write_text(original, encoding="utf-8")

    assert SOURCE.read_text(encoding="utf-8") == original
    run_property(expect_failure=False)
    print("Mutation test PASSED: actual-code mutation detected and source restored.")


if __name__ == "__main__":
    main()
