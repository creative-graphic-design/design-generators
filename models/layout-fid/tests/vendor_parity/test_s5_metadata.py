import json
from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_s5_metadata_records_issue_149_values():
    data = json.loads(
        Path("models/layout-fid/tests/vendor_parity/metadata/s5_rico25.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["single_seed"]["vendor"]["fid"] == pytest.approx(6.5372)
    assert data["single_seed"]["converted"]["fid"] == pytest.approx(5.0896)
