from pathlib import Path

import pytest

from layoutformerpp.conversion import load_original_state_dict


@pytest.mark.vendor_parity
def test_original_checkpoint_loader_requires_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_original_state_dict(Path("missing/final_checkpoint.pth.tar"))
