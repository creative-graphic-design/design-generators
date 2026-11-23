from __future__ import annotations

import sys

from pathlib import Path

scripts = Path(__file__).resolve().parents[3] / "scripts" / "publaynet_s5"
sys.path.insert(0, str(scripts))
import vendor_main_bootstrap  # noqa: E402

from sitecustomize import StopAfterMeasurement, persist  # noqa: E402

try:
    vendor_main_bootstrap.main()
except StopAfterMeasurement:
    persist()
    raise SystemExit(0)
