from __future__ import annotations

import subprocess
from pathlib import Path


def test_production_segment_accepts_shortened_end_epoch() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "run_crello_training_segment.sh"
    guard = "((start_epoch < end_epoch && end_epoch <= 250))"
    assert guard in runner.read_text(encoding="utf-8")

    for start, end, accepted in (
        (193, 227, True),
        (193, 193, False),
        (193, 251, False),
    ):
        result = subprocess.run(
            ["bash", "-c", f"start_epoch={start}; end_epoch={end}; {guard}"],
            check=False,
        )
        assert (result.returncode == 0) is accepted
