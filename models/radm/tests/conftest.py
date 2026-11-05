"""Load the tracked compatibility needed by the pinned reference runtime."""

from pathlib import Path
import runpy


runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "vendor_compat"
        / "sitecustomize.py"
    )
)
